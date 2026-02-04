import pandas as pd
from PIL import Image
import cv2, os, re, sys
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tools.storage_adapter import *
thread_local = threading.local()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.ML_processing.mask_model import *
from src.DB_processing.tools import get_reader, reader, append_audit
from src.DB_processing.database import DatabaseManager

description_labels_dict = {
    'area':{'axilla':['ax'], # Order matters, axilla takes dominance 
            'breast':['breast'],
            'supraclavicular':['superclavicular','supraclavicular'],
            'subclavicular':['subclavicular','subclavcular']},
    'laterality':{'left':['lt','left', 'eft', 'ft'],
                  'right':['rt','right', 'ight', 'ght', 'ht']},
    'orientation':{'long':['long', 'lon', 'ong'],
                    'trans':['trans', 'tran', 'tra', 'ans'],
                    'anti-radial':['anti-rad','anti-radial'],
                    'radial':['radial'],
                    'oblique':['oblique']}
}

def contains_substring(input_string, substring_list):
    # Convert to lowercase, remove all spaces, and replace $ with s
    input_string = str(input_string).lower().replace(' ', '').replace('$', 's')
    
    for substring in substring_list:
        # Also remove spaces and replace $ with s in the substring before matching
        normalized_substring = substring.lower().replace(' ', '').replace('$', 's')
        if normalized_substring in input_string:
            return True
    return False


def label_parser(x, label_dict={}):
    for k in label_dict.keys():
        
        labels = label_dict[k]
        if contains_substring(x,labels):
            return k

    return 'unknown'

def find_time_substring(text):
    text = str(text)
    if text is None:  # If the input is None, return 'unknown'
        return 'unknown'
    
    # Regular expression to match time substrings of the form HH:MM, H.MM, or H*MM
    # with optional spaces
    pattern = r'\d{1,2}\s*[:.*]\s*\d{1,2}'
    
    # Find all matches in the input text
    matches = re.findall(pattern, text)
    
    if len(matches) > 0:
        # Remove spaces from the first match, convert period and asterisk to colon
        time = re.sub(r'\s', '', matches[0]).replace('.', ':').replace('*', ':')
        return time
    
    # Check for alternative substrings
    alternative_substrings = ["uoq", "uiq", "liq", "loq"]
    for substring in alternative_substrings:
        if substring in text:
            return substring
    
    return 'unknown'

    
def find_cm_substring(input_str):
    """Find first substring of the form #cm or # cm or #-#cm or #-# cm, not case sensitive
    
    Args:
        input_str:  string
        
    Returns:
        processed numeric value or np.nan
    """
    # Regular expression to match s
    pattern = r'\d+(-\d+)?\s*cm'
    
    input_str = str(input_str).lower()
    input_str = input_str.replace("scm","5cm") #easyocr sometimes misreads 5cm as scm
    
    # Find all matches in the input string
    matches = re.finditer(pattern, input_str)
    
    # get list of matches
    list_of_matches = [m.group() for m in matches]
    
    if len(list_of_matches)==0:
        return np.nan
    else:
        # Apply the Fix_CM_Data logic here
        value = list_of_matches[0]
        
        # Remove 'cm' and spaces
        value = value.replace('cm', '').replace(' ', '')
        
        # Handle range values (take average)
        if '-' in value:
            range_values = [int(i) for i in value.split('-')]
            value = round(np.mean(range_values))
        else:
            value = int(value)
        
        # Handle invalid values
        if value > 25 or value == 0:
            return np.nan
        
        return value
    
def extract_descript_features( input_str, labels_dict ):
    
    output_dict = {}
    for feature in labels_dict.keys():
        levels_dict = labels_dict[feature]
        output_dict[feature] = label_parser( input_str, levels_dict)

    output_dict['clock_pos'] = find_time_substring(input_str)
    output_dict['nipple_dist'] = find_cm_substring(input_str)
    
    return output_dict


# ULTRASOUND MASK
######################################################



def process_crop_region(image):
    # Binary threshold
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

    # Erosion
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=8)

    # Find contours
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None

    # Sort contours by area (largest first)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = sorted_contours[0]

    # Check if second largest contour should be merged
    if len(sorted_contours) > 1:
        second_largest = sorted_contours[1]
        largest_area = cv2.contourArea(largest_contour)
        second_area = cv2.contourArea(second_largest)

        # Merge if second contour is at least 2.5% of largest
        if second_area >= 0.025 * largest_area:
            # Merge contours by concatenating points
            largest_contour = np.concatenate([largest_contour, second_largest])

    # Convex hull
    convex_hull = cv2.convexHull(largest_contour)

    # Get bounding box from convex hull
    x_coords = convex_hull[:, 0, 0]
    y_coords = convex_hull[:, 0, 1]
    x_min = int(np.min(x_coords))
    x_max = int(np.max(x_coords))
    y_min = int(np.min(y_coords))
    y_max = int(np.max(y_coords))

    # OCR REGION SETUP: Use convex hull x-range to restrict OCR
    start_row = int(3 * image.shape[0] / 4)
    end_row = image.shape[0]
    ocr_padding = 50  # Inset padding from convex hull edges

    # Calculate restricted x-range for OCR (inset from convex hull edges)
    start_col = max(0, x_min + ocr_padding)
    end_col = min(image.shape[1], x_max - ocr_padding)

    # Ensure valid region (if hull is too narrow, use full hull width)
    if start_col >= end_col:
        start_col = x_min
        end_col = x_max

    # Ensure minimum width for OCR region
    min_ocr_width = 10
    if end_col - start_col < min_ocr_width:
        # Expand region around midpoint
        mid_x = (x_min + x_max) // 2
        start_col = max(0, mid_x - min_ocr_width // 2)
        end_col = min(image.shape[1], mid_x + min_ocr_width // 2)

    # OCR in restricted region
    reader_thread = get_reader()
    output = reader_thread.readtext(image[start_row:end_row, start_col:end_col])

    # Find the minimum y-value of detected text (if any)
    text_y_limit = None
    if output:
        # Get the minimum y-coordinate of all detected text boxes
        min_text_y = min(detection[0][0][1] for detection in output)
        text_y_limit = int(min_text_y + start_row)

    # Calculate bounding box from convex hull
    x = x_min
    y = y_min
    w = x_max - x_min

    # Height: stop at text or end of convex hull, whichever comes first
    if text_y_limit is not None:
        h = min(text_y_limit, y_max) - y
    else:
        h = y_max - y

    return (x, y, w, h)

# DARKNESS
######################################################


def process_darkness(image, row):
    try:
        region_x = row['region_location_min_x0']
        region_y = row['region_location_min_y0']
        region_w = row['region_location_max_x1'] - region_x
        region_h = row['region_location_max_y1'] - region_y
    except KeyError:
        return None

    # Extract region directly from original image (most efficient)
    img_us = image[region_y:region_y+region_h, region_x:region_x+region_w]
    
    # Threshold and calculate darkness
    _, img_us_bw = cv2.threshold(img_us, 20, 255, cv2.THRESH_BINARY)
    darkness = 100 * np.sum(img_us_bw == 0) / (region_w * region_h)
    
    return darkness

######################################################

def process_single_image_combined(row, image_folder_path):
    image_name = row['image_name']
    image_path = os.path.join(image_folder_path, image_name)
    
    # Read image and convert to grayscale
    image = read_image(image_path)
    if image is None:
        return None, None
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Process ultrasound region
    x, y, w, h = process_crop_region(image)
    mask_map = None
    if x is not None:
        mask_map = (image_name, (x, y, w, h))
    
    # Process darkness
    darkness = process_darkness(image, row)
    darkness_map = None
    if darkness is not None:
        darkness_map = (image_name, darkness)

    return mask_map, darkness_map

def process_images_combined(image_folder_path, image_df):
    image_masks = []
    darknesses = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()//2) as executor:
        futures = {executor.submit(process_single_image_combined, row, image_folder_path): row 
                  for _, row in image_df.iterrows()}
        
        with tqdm(total=len(futures), desc='Finding Crop Region / Darkness') as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    mask_result, darkness_result = result
                    if mask_result is not None:
                        image_masks.append(mask_result)
                    if darkness_result is not None:
                        darknesses.append(darkness_result)
                pbar.update()

    return image_masks, darknesses

# OCR
######################################################


def ocr_image(image_file, description_mask, image_folder_path):
    reader_thread = get_reader()

    try:
        image = read_image(os.path.join(image_folder_path, image_file), use_pil=True).convert('L')

        width, height = image.size
        expand_ratio = 0.025  # Change this to control the crop expansion

        if description_mask:
            x0, y0, x1, y1 = description_mask
        else:  # if description_mask is empty, crop upper 2/3 and 1/4 on both sides
            x0 = width // 8
            y0 = (2 * height) // 3
            x1 = width - (width // 8)
            y1 = height

        # Calculate expanded coordinates, while ensuring they are within image bounds
        x0_exp = max(0, x0 - int(width * expand_ratio))
        y0_exp = max(0, y0 - int(height * expand_ratio))
        x1_exp = min(width, x1 + int(width * expand_ratio))
        y1_exp = min(height, y1 + int(height * expand_ratio))

        cropped_image = image.crop((x0_exp, y0_exp, x1_exp, y1_exp))

        # Convert the PIL Image to a numpy array
        cropped_image_np = np.array(cropped_image)

        # Apply blur to help OCR
        img_focused = cv2.GaussianBlur(cropped_image_np, (3, 3), 0)

        result = reader_thread.readtext(img_focused,paragraph=True)

        #Fix OCR miss read
        result = [[r[0], 'logiq' if r[1].lower() == 'loc' or r[1].lower() == 'lo' else r[1].lower()] for r in result]
        result = [ [r[0], r[1].lower()] for r in result]
        
        # now loop over the remaining strings and get the total string and the bounding box
        text = ''
        for r in result:
            text = text + r[1] + ' '

        return [image_file, text]
    except:
        return [image_file, None] # Crop failed

def get_OCR(image_folder_path, description_masks):
    # Create new description_masks with just the basenames
    basename_description_masks = [(os.path.basename(image_file), description_mask) 
                                 for image_file, description_mask in description_masks]
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Pass basename to the future, but also provide a way to access the full path inside ocr_image
        futures = {executor.submit(ocr_image, basename, description_mask, image_folder_path): 
                  basename for basename, description_mask in basename_description_masks}
        progress = tqdm(total=len(futures), desc='Performing OCR')

        # Initialize dictionary to store descriptions
        descriptions = {}

        for future in as_completed(futures):
            result = future.result()  # result is now a list with the filename and the description
            descriptions[result[0]] = result[1]  # Store description at corresponding image file
            progress.update()

        progress.close()
        
    return descriptions





# MAIN
######################################################

def analyze_images(database_path):

    with DatabaseManager() as db:
        image_folder_path = f"{database_path}/images/"

        # Load data from database
        image_df = db.get_images_dataframe()
        breast_df = db.get_study_cases_dataframe()

        append_audit("image_processing.input_images", len(image_df))

        # Finding OCR Masks
        _, description_masks = find_masks(image_folder_path, 'mask_model', image_df, 1920, 1080)
        append_audit("image_processing.extracted_description_masks", len(description_masks))

        # Performing OCR
        descriptions = get_OCR(image_folder_path, description_masks)
        valid_descriptions = sum(1 for desc in descriptions.values() if desc)
        append_audit("image_processing.extracted_ocr_descriptions", valid_descriptions)

        # Finding Darkness and Image Masks
        image_masks, darknesses = process_images_combined(image_folder_path, image_df)
        append_audit("image_processing.extracted_crop_regions", len(image_masks))
        append_audit("image_processing.extracted_darkness_measurements", len(darknesses))

        # Convert lists of tuples to dictionaries
        darknesses_dict = {filename: value for filename, value in darknesses}
        image_masks_dict = {filename: mask for filename, mask in image_masks}

        # Update dataframe using map
        image_df['description'] = image_df['image_name'].map(descriptions)
        image_df['darkness'] = image_df['image_name'].map(darknesses_dict)
        image_df['bounding_box'] = image_df['image_name'].map(image_masks_dict)

        # Fill NaN values with (None, None, None, None) before expanding
        image_df['bounding_box'] = image_df['bounding_box'].apply(
            lambda x: (None, None, None, None) if pd.isna(x) else x
        )
        image_df[['crop_x', 'crop_y', 'crop_w', 'crop_h']] = pd.DataFrame(
            image_df['bounding_box'].tolist(), index=image_df.index
        )

        # Construct a temporary DataFrame with the feature extraction
        temp_df = image_df['description'].apply(lambda x: extract_descript_features(x, labels_dict=description_labels_dict)).apply(pd.Series)
        for column in temp_df.columns:
            image_df[column] = temp_df[column]

        # Overwrite non bilateral cases with known lateralities
        laterality_mapping = breast_df[breast_df['study_laterality'].isin(['LEFT', 'RIGHT'])].set_index('accession_number')['study_laterality'].to_dict()
        image_df['laterality'] = image_df.apply(
            lambda row: laterality_mapping.get(row['accession_number']).lower()
            if row['accession_number'] in laterality_mapping
            else row['laterality'],
            axis=1
        )

        # Count unknown lateralities after correction
        unknown_after = image_df[image_df['laterality'] == 'unknown'].shape[0]
        append_audit("image_processing.bilateral_with_missing_lat", unknown_after)

        # Update database with processed results using upsert
        image_updates = image_df.to_dict('records')
        updated_count = db.insert_images_batch(image_updates, upsert=True)
        print(f"Updated {updated_count} images in database")