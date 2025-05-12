import pandas as pd
from PIL import Image
import cv2, os, re, sys
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from storage_adapter import *
thread_local = threading.local()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.ML_processing.caliper_model import *
from src.ML_processing.mask_model import *
from src.DB_processing.tools import get_reader, reader, append_audit


# configuration
description_kw = ['breast','lt','long','rt','trans','area','palpated','axilla','areolar','radial','marked','supraclavicular','oblique','contrast']

description_labels_dict = {
    'area':{'breast':['breast'],
            'axilla':['axilla', 'axila', 'axlla'],
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

def make_grayscale( img ):
    color = len(img.shape) > 2 # True if C > 1, False if Array is 2D or C = 1  
    if color:
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    return img_gray, color

def contains_substring(input_string, substring_list):
    input_string = str(input_string).lower()  # Convert input string to lowercase
    for substring in substring_list:
        if substring.lower() in input_string:  # Convert substring to lowercase
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
    
    # Regular expression to match time substrings of the form HH:MM or H.MM
    # with optional spaces
    pattern = r'\d{1,2}\s*[:.]\s*\d{1,2}'
    
    # Find all matches in the input text
    matches = re.findall(pattern, text)
    
    if len(matches) > 0:
        # Remove spaces from the first match, convert period to colon
        time = re.sub(r'\s', '', matches[0]).replace('.', ':')
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
        list of matched substrings
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
        return 'unknown'
    else:
        return list_of_matches[0]
    
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



def find_top_edge_points(largest_contour, vertical_range):
    # Find the highest y-coordinate in the contour
    top_y = min(largest_contour[:, 0, 1])

    # Filter points that are within the vertical range from the top y-coordinate
    top_edge_points = [pt[0] for pt in largest_contour if top_y <= pt[0][1] <= top_y + vertical_range]

    # Find the leftmost and rightmost points from the filtered top edge points
    top_left = min(top_edge_points, key=lambda x: x[0])
    top_right = max(top_edge_points, key=lambda x: x[0])

    return top_left, top_right

def process_crop_region(image):
    start_row = int(3 * image.shape[0] / 4)
    end_row = image.shape[0]
    margin = 10
    
    reader_thread = get_reader()
    output = reader_thread.readtext(image[start_row:end_row, :])
    
    for detection in output:
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        top_left = (0, max(0, top_left[1] + start_row - margin))
        bottom_right = (image.shape[1], min(image.shape[0], bottom_right[1] + start_row + margin))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=5)

    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    convex_hull = cv2.convexHull(largest_contour)
    top_left, top_right = find_top_edge_points(convex_hull, vertical_range=20)

    x = top_left[0]
    y = top_left[1]
    w = top_right[0] - x
    h = max(convex_hull[:, 0, 1]) - y
    
    return (x, y, w, h)

# DARKNESS
######################################################


def process_darkness(image, row):
    try:
        region_x = row['RegionLocationMinX0']
        region_y = row['RegionLocationMinY0']
        region_w = row['RegionLocationMaxX1'] - region_x
        region_h = row['RegionLocationMaxY1'] - region_y
    except KeyError:
        return None

    # Convert to PIL Image for consistent resizing
    pil_image = Image.fromarray(image)
    new_size = (800, 600)
    pil_image = pil_image.resize(new_size)
    
    # Calculate scaled coordinates
    scaled_x = int(region_x * new_size[0] / pil_image.size[0])
    scaled_y = int(region_y * new_size[1] / pil_image.size[1])
    scaled_w = int(region_w * new_size[0] / pil_image.size[0])
    scaled_h = int(region_h * new_size[1] / pil_image.size[1])

    # Convert back to numpy for darkness calculation
    image_np = np.array(pil_image)
    img_us = image_np[scaled_y:scaled_y+scaled_h, scaled_x:scaled_x+scaled_w]
    img_us_gray, _ = make_grayscale(img_us)
    _, img_us_bw = cv2.threshold(img_us_gray, 20, 255, cv2.THRESH_BINARY)
    darkness = 100 * np.sum(img_us_bw == 0) / (scaled_w * scaled_h)
    
    return darkness

######################################################

def process_single_image_combined(row, image_folder_path):
    image_name = row['ImageName']
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

def process_images_combined(image_folder_path, db_to_process):
    image_masks = []
    darknesses = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()//2) as executor:
        futures = {executor.submit(process_single_image_combined, row, image_folder_path): row 
                  for _, row in db_to_process.iterrows()}
        
        with tqdm(total=len(futures), miniters=50) as pbar:
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


def ocr_image(image_file, description_mask, image_folder_path, reader, kw_list):
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
        futures = {executor.submit(ocr_image, basename, description_mask, image_folder_path, reader, description_kw): 
                  basename for basename, description_mask in basename_description_masks}
        progress = tqdm(total=len(futures), desc='')

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



# Main method to prefrom operations
def analyze_images(database_path):
        
    image_folder_path = f"{database_path}/images/"
    image_data_file = f'{database_path}/ImageData.csv'
    breast_data_file = f'{database_path}/BreastData.csv'
    image_df = read_csv(image_data_file)
    breast_df = read_csv(breast_data_file)

    # Check if any new features are missing in image_df and add them
    new_features = ['labeled', 'crop_x', 'crop_y', 'crop_w', 'crop_h', 'description', 'has_calipers',  'has_calipers_prediction', 'darkness', 'area', 'laterality', 'orientation', 'clock_pos', 'nipple_dist']
    missing_features = set(new_features) - set(image_df.columns)
    for nf in missing_features:
        image_df[nf] = None
    
    
    
    image_df['labeled'] = False
    
    # Check if 'processed' column exists, if not, create it and set all to False
    if 'processed' not in image_df.columns:
        image_df['processed'] = False

    # Only keep rows where 'processed' is False
    db_to_process = image_df[image_df['processed'] != True]
    db_to_process['processed'] = False
    append_audit("image_processing.input_images", len(db_to_process))

    print("Finding OCR Masks")
    _, description_masks = find_masks(image_folder_path, 'mask_model', db_to_process, 1920, 1080)
    append_audit("image_processing.extracted_description_masks", len(description_masks))
    
    print("Performing OCR")  
    descriptions = get_OCR(image_folder_path, description_masks)
    valid_descriptions = sum(1 for desc in descriptions.values() if desc)
    append_audit("image_processing.extracted_ocr_descriptions", valid_descriptions)
    
    print("Finding Darkness and Image Masks")
    image_masks, darknesses = process_images_combined(image_folder_path, db_to_process)
    append_audit("image_processing.extracted_crop_regions", len(image_masks))
    append_audit("image_processing.extracted_darkness_measurements", len(darknesses))

    print("Finding Calipers")
    caliper_results = find_calipers(image_folder_path, 'caliper_model', db_to_process)
    caliper_count = sum(1 for _, bool_val, _ in caliper_results if bool_val)
    append_audit("image_processing.images_with_calipers", caliper_count)
    
    # Convert lists of tuples to dictionaries
    has_calipers_dict = {filename: bool_val for filename, bool_val, _ in caliper_results}
    has_calipers_confidence_dict = {filename: pred_val for filename, _, pred_val in caliper_results}
    darknesses_dict = {filename: value for filename, value in darknesses}
    image_masks_dict = {filename: mask for filename, mask in image_masks}

    # Update dataframe using map
    db_to_process['description'] = db_to_process['ImageName'].map(pd.Series(descriptions))
    db_to_process['darkness'] = db_to_process['ImageName'].map(pd.Series(darknesses_dict))
    db_to_process['has_calipers'] = db_to_process['ImageName'].map(pd.Series(has_calipers_dict))
    db_to_process['has_calipers_prediction'] = db_to_process['ImageName'].map(pd.Series(has_calipers_confidence_dict))
    db_to_process['bounding_box'] = db_to_process['ImageName'].map(pd.Series(image_masks_dict))
    
    
    db_to_process[['crop_x', 'crop_y', 'crop_w', 'crop_h']] = pd.DataFrame(db_to_process['bounding_box'].tolist(), index=db_to_process.index)

    # Construct a temporary DataFrame with the feature extraction
    temp_df = db_to_process['description'].apply(lambda x: extract_descript_features(x, labels_dict=description_labels_dict)).apply(pd.Series)
    for column in temp_df.columns:
        db_to_process[column] = temp_df[column]
    
    # Overwrite non bilateral cases with known lateralities
    laterality_mapping = breast_df[breast_df['Study_Laterality'].isin(['LEFT', 'RIGHT'])].set_index('Accession_Number')['Study_Laterality'].to_dict()
    db_to_process['laterality'] = db_to_process.apply(
        lambda row: laterality_mapping.get(row['Accession_Number']).lower() 
        if row['Accession_Number'] in laterality_mapping 
        else row['laterality'],
        axis=1
    )
    
    # Count unknown lateralities after correction
    unknown_after = db_to_process[db_to_process['laterality'] == 'unknown'].shape[0]
    append_audit("image_processing.bilateral_with_missing_lat", unknown_after)

    image_df.update(db_to_process, overwrite=True)
    
    save_data(image_df, image_data_file)
    
