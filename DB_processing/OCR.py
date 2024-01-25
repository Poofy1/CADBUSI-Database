import pandas as pd
from PIL import Image
import cv2, os, re, sys
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import easyocr
import threading
thread_local = threading.local()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ML_processing.caliper_model import *
from ML_processing.mask_model import *


def get_reader():
    # Check if this thread already has a reader
    if not hasattr(thread_local, "reader"):
        # If not, create a new reader and store it in the thread-local storage
        thread_local.reader = easyocr.Reader(['en'])
    return thread_local.reader


def is_color( img ):
    """Returns true if numpy array H x W x C has C > 1
    
    Used to determine if images converted to numpy arrays are multi-channel.
    
    Args:
        img:  numpy array that is H x W or H x W x 3
    
    Returns:
        boolean:  True if C > 1, False if Array is 2D or C = 1   
    """
    return len(img.shape) > 2

def make_grayscale( img ):
    color = is_color(img)
    if color:
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    return img_gray, color

# configure easyocr reader
reader = easyocr.Reader(['en'])

# configuration
description_kw = ['breast','lt','long','rt','trans','area','palpated','axilla','areolar','radial','marked','supraclavicular','oblique','contrast']
description_kw_expand= ['cm','fn','breast','lt','long','rt','trans','area',
                        'palpated','axilla','areolar','radial','marked',
                        'supraclavicular','oblique','contrast','retroareolar',
                        'harmonics','axillary','subareolar','nipple','anti', 
                        'periareolar','subclavicular']
description_kw_contract = ['retro areolar', 
                           'sub areoloar', 
                           'peri areolar',
                           'anti -rad']
description_kw_sub = {'scm':'5 cm', 
                      'anti radial':'anti-rad', 
                      'axillary':'axilla', 
                      'axlla':'axilla',
                      'subclavcular':'subclavicular'}

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
    
def extract_descript_features_df(df, labels_dict, col = 'description'):

    # first extract simple text features
    for feature in labels_dict.keys():
        levels_dict = labels_dict[feature]
        df[feature] = df[col].apply( label_parser, label_dict = levels_dict )
    
    # extract clock_position
    df['clock_pos'] = df[col].apply( find_time_substring )
    
    # extract nipple_dist
    df['nipple_dist'] = df[col].apply( find_cm_substring )
    
    return df

def fetch_index_for_patient_id( id, db, only_gray = False, only_calipers = False ):
    # id is a patient id number that should be listed in database
    # only_gray = True → return only monochrome files (not doppler)
    # only_calipers = True → return only files that include calipers
    # returns list of indices
    
    if id in db['Patient_ID'].tolist():
         indices= db.index[db['Patient_ID']==id].tolist()
    else:
        indices = []
    return indices


def find_mixed_lateralities( db ):
    ''' returns a list of all patient ids for which the study contains both left and right lateralities (to be deleted for now)
    
    Args:
        db is a dataframe with one row per image, must have columns for patient_id and laterality
        
    Returns:
        list of patient ids from db for which the lateralities are mixed
    '''
    db['latIsLeft']=(db['laterality']=='left')
    df = db.groupby(['Patient_ID']).agg({'Patient_ID':'count', 'latIsLeft':'sum'})
    df['notPure'] = ~( (df['latIsLeft']==0) |  (df['latIsLeft']==df['Patient_ID']) )
    
    mixedPatientIDs = df[df['notPure']].index.tolist()
    return mixedPatientIDs



# DARKNESS
######################################################



def process_single_darkness(row, image_folder_path):
    image_file = row['ImageName']
    
    try: 
        x = row['RegionLocationMinX0']
        y = row['RegionLocationMinY0']
        w = row['RegionLocationMaxX1'] - x
        h = row['RegionLocationMaxY1'] - y
    except KeyError:
        return (image_file, None)

    with Image.open(os.path.join(image_folder_path, image_file)).convert('L') as image:
        
        new_size = (800, 600)  # or any other size
        image = image.resize(new_size)
        x, y, w, h = int(x * new_size[0] / image.size[0]), int(y * new_size[1] / image.size[1]), int(w * new_size[0] / image.size[0]), int(h * new_size[1] / image.size[1])

        image_np = np.array(image)
        img_us = image_np[y:y+h, x:x+w]
        img_us_gray, _ = make_grayscale(img_us)
        _, img_us_bw = cv2.threshold(img_us_gray, 20, 255, cv2.THRESH_BINARY)
        num_dark = np.sum(img_us_bw == 0)
    
    return (image_file, 100 * num_dark / (w * h))

def get_darkness(image_folder_path, df):
    darknesses = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()//2) as executor:  # Reduced number of threads
        futures = {executor.submit(process_single_darkness, row, image_folder_path): row for _, row in df.iterrows()}
        with tqdm(total=len(futures), miniters=50) as pbar:  # Update progress bar less frequently
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    darknesses.append(result)
                pbar.update()

    return darknesses





# OCR
######################################################


def process_image(image_file, description_mask, image_folder_path, reader, kw_list):
    reader_thread = get_reader()
    
    try:
        image = Image.open(os.path.join(image_folder_path, image_file)).convert('L')
    except:
        return [image_file, None]
    
    

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



def get_OCR(image_folder_path, description_masks):
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_image, image_file, description_mask, image_folder_path, reader, description_kw): image_file for image_file, description_mask in description_masks}
        progress = tqdm(total=len(futures), desc='')

        # Initialize dictionary to store descriptions
        descriptions = {}

        for future in as_completed(futures):
            result = future.result()  # result is now a list with the filename and the description
            descriptions[result[0]] = result[1]  # Store description at corresponding image file
            progress.update()

        progress.close()
        
    return descriptions




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

def process_single_image(image_path):
    image_name = os.path.basename(image_path)  # Get image name from image path
    
    image = cv2.imread(image_path, 0)  # Load grayscale image directly

    # Calculate the start and end row indices for the bottom quarter of the image
    start_row = int(3 * image.shape[0] / 4)
    end_row = image.shape[0]
    margin = 10
    
    # Remove caliper box
    reader_thread = get_reader()
        
    output = reader_thread.readtext(image[start_row:end_row, :])
    for detection in output:
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        top_left = (0, max(0, top_left[1] + start_row - margin))
        bottom_right = (image.shape[1], min(image.shape[0], bottom_right[1] + start_row + margin))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
        
        
    # Create binary mask of nonzero pixels
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=5)

    # Find contours and get the largest one
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:  # Check if contours is empty
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    convex_hull = cv2.convexHull(largest_contour)
    
    # Use the function to find the top edge points
    top_left, top_right = find_top_edge_points(convex_hull, vertical_range=20)

    # Now you have the top left and top right points, use them to find x, y, w, h
    x = top_left[0]
    y = top_left[1]
    w = top_right[0] - x
    h = max(convex_hull[:, 0, 1]) - y  # Bottom y-coordinate - top y-coordinate

    return (image_name, (x, y, w, h))


def get_ultrasound_region(image_folder_path, db_to_process):
    # Construct image paths for only the new data
    image_paths = [os.path.join(image_folder_path, filename) for filename in db_to_process['ImageName']]

    # Collect image data in list
    image_data = []
    
    # Thread pool and TQDM
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_image, image_path): image_path for image_path in image_paths}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    image_data.append(result)
                pbar.update()

    return image_data



# MAIN
######################################################



# Main method to prefrom operations
def Perform_OCR(database_path):
        
    image_folder_path = f"{database_path}/images/"
    input_file = f'{database_path}/ImageData.csv'
    db_out = pd.read_csv(input_file)

    # Check if any new features are missing in db_out and add them
    new_features = ['labeled', 'crop_x', 'crop_y', 'crop_w', 'crop_h', 'description', 'has_calipers', 'darkness', 'area', 'laterality', 'orientation', 'clock_pos', 'nipple_dist']
    missing_features = set(new_features) - set(db_out.columns)
    for nf in missing_features:
        db_out[nf] = None
    
    
    
    db_out['labeled'] = False
    
    # Check if 'processed' column exists, if not, create it and set all to False
    if 'processed' not in db_out.columns:
        db_out['processed'] = False

    # Only keep rows where 'processed' is False
    db_to_process = db_out[db_out['processed'] != True]
    db_to_process['processed'] = False
    
    print("Finding Darkness")
    darknesses = get_darkness(image_folder_path, db_to_process)
    
    print("Finding OCR Masks")
    _, description_masks = find_masks(image_folder_path, 'mask_model', db_to_process, 1920, 1080)

    print("Performing OCR")  
    descriptions = get_OCR(image_folder_path, description_masks)

    print("Finding Image Masks")
    image_masks = get_ultrasound_region(image_folder_path, db_to_process)
    
    print("Finding Calipers")
    has_calipers = find_calipers(image_folder_path, 'caliper_model', db_to_process)
    
    # Convert lists of tuples to dictionaries
    has_calipers_dict = {filename: value for filename, value in has_calipers}
    darknesses_dict = {filename: value for filename, value in darknesses}
    image_masks_dict = {filename: mask for filename, mask in image_masks}
    
    # Convert dictionaries to Series for easy mapping
    descriptions_series = pd.Series(descriptions)
    darknesses_series = pd.Series(darknesses_dict)
    has_calipers_series = pd.Series(has_calipers_dict)
    image_masks_series = pd.Series(image_masks_dict)

    # Update dataframe using map
    db_to_process['description'] = db_to_process['ImageName'].map(descriptions_series)
    db_to_process['darkness'] = db_to_process['ImageName'].map(darknesses_series)
    db_to_process['has_calipers'] = db_to_process['ImageName'].map(has_calipers_series)
    db_to_process['bounding_box'] = db_to_process['ImageName'].map(image_masks_series)
    
    
    db_to_process[['crop_x', 'crop_y', 'crop_w', 'crop_h']] = pd.DataFrame(db_to_process['bounding_box'].tolist(), index=db_to_process.index)

    # Construct a temporary DataFrame with the feature extraction
    temp_df = db_to_process.loc[db_to_process['description'].str.len() > 0, 'description'].apply(lambda x: extract_descript_features(x, labels_dict=description_labels_dict)).apply(pd.Series)

    # Update the columns in db_out directly from temp_df
    for column in temp_df.columns:
        db_to_process[column] = temp_df[column]
        
    db_out.update(db_to_process, overwrite=True)
    db_out.to_csv(input_file,index=False)


