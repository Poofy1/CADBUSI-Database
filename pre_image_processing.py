import pandas as pd
from PIL import Image
import cv2, os, re
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import easyocr
from ML_processing.caliper_model import *
from ML_processing.mask_model import *
from collections.abc import Iterable
env = os.path.dirname(os.path.abspath(__file__))


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
    """Returns grayscale version of image stored in numpy array, no change if image is already monochrome
    
    Args:
        img: numpy array that is monchrome (HxW) or color (HxWx3)
        
    Returns:
        img_gray: numpy array, HxW, with UINT8 values from 0 to 255
        color: boolean that is True if input image was color 
    """
    
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
    
    if id in db['Accession_Number'].tolist():
         indices= db.index[db['Accession_Number']==id].tolist()
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
    df = db.groupby(['Accession_Number']).agg({'Accession_Number':'count', 'latIsLeft':'sum'})
    df['notPure'] = ~( (df['latIsLeft']==0) |  (df['latIsLeft']==df['Accession_Number']) )
    
    mixedPatientIDs = df[df['notPure']].index.tolist()
    return mixedPatientIDs

def choose_images_to_label(db, case_data):
    db['label'] = True

    #Remove images that are too dark
    db.loc[db['darkness'] > 65, 'label'] = False
    
    # find all of the rows with calipers
    caliper_rows = db[db['has_calipers']]

    # loop over caliper rows and tag twin images (not efficient)
    for idx, row in caliper_rows.iterrows():
        distance = row['distance']
        if distance <= 5:
            db.at[idx,'label'] = False

    # set label = False for all non-breast images
    db.loc[(db['area'] == 'unknown') | (db['area'].isna()), 'area'] = 'breast'
    db.loc[(db['area'] != 'breast'), 'label'] = False

    # Remove Males from studies
    male_patient_ids = case_data[case_data['PatientSex'] == 'M']['Patient_ID'].values
    db.loc[db['Patient_ID'].isin(male_patient_ids),'label'] = False
    
    #Add back majority focused images
    # Parse 'StudyDescription' for 'FOCUSED LEFT' or 'FOCUSED RIGHT'
    focus_left_studies = case_data[case_data['StudyDescription'].str.contains('FOCUSED LEFT')]['Accession_Number'].values
    db.loc[(db['Accession_Number'].isin(focus_left_studies)) & (db['laterality'] == 'right'), 'label'] = False

    focus_right_studies = case_data[case_data['StudyDescription'].str.contains('FOCUSED RIGHT')]['Accession_Number'].values
    db.loc[(db['Accession_Number'].isin(focus_right_studies)) & (db['laterality'] == 'left'), 'label'] = False
    
    # If 'BILATERAL' is present in 'StudyDescription', set 'label' to False for all images in that study
    bilateral_studies = case_data[case_data['StudyDescription'].str.contains('BILATERAL')]['Accession_Number'].values
    db.loc[db['Accession_Number'].isin(bilateral_studies), 'label'] = False
    
    # Check for mixed lateralities only in these non-focus studies
    non_focus_studies = case_data[~case_data['StudyDescription'].str.contains('FOCUSED|BILATERAL')]['Accession_Number'].values
    mixedIDs = find_mixed_lateralities( db[db['Accession_Number'].isin(non_focus_studies)] )
    db.loc[db['Accession_Number'].isin(mixedIDs),'label'] = False
    
    # Set label = False for all images with 'unknown' laterality
    db.loc[db['laterality'] == 'unknown', 'label'] = False
    

    # If 'chest' or 'mastectomy' is present in 'StudyDescription', set 'label' to False for all images in that study
    chest_or_mastectomy_studies = case_data[case_data['StudyDescription'].str.contains('chest|mastectomy', case=False)]['Accession_Number'].values
    db.loc[db['Accession_Number'].isin(chest_or_mastectomy_studies), 'label'] = False
    
     # Set label = False for all images with 'RegionCount' > 1
    db.loc[db['RegionCount'] > 1, 'label'] = False
    
    return db


def add_labeling_categories(db):
    db['label_cat'] = ''

    for idx, row in db.iterrows():
        if row['label']:
            orient = row['orientation']
            image_type = row['PhotometricInterpretation']
            if image_type == 'RGB':
                label_cat = 'doppler'
            elif orient in ['trans', 'long']:
                label_cat = orient
            else:
                label_cat = 'other'
            
            db.at[idx, 'label_cat'] = label_cat

    return db




def get_darkness(image_folder_path, image_masks):
    
    darknesses = []
    
    for mask in tqdm(image_masks):
        
        image_file, coordinates = mask
        
        image = Image.open(os.path.join(image_folder_path, image_file)).convert('L')
        image_np = np.array(image)  # Convert image to numpy array

        try: 
            x, y, w, h = coordinates
        except:
            darknesses.append((image_file, None))
            continue
        
        img_us = image_np[y:y+h, x:x+w]
        img_us_gray, isColor = make_grayscale(img_us)
        _,img_us_bw = cv2.threshold(img_us_gray, 20, 255, cv2.THRESH_BINARY)
        num_dark = np.sum( img_us_bw == 0)
        darknesses.append((image_file, 100*num_dark/(w*h)))  # Append the filename along with its darkness
        
    return darknesses





def process_image(image_file, description_mask, image_folder_path, reader, kw_list):
    image = Image.open(os.path.join(image_folder_path, image_file)).convert('L')

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
    
    cv2.imwrite(os.path.join(f'{env}/database/test/' + image_file), img_focused)

    result = reader.readtext(img_focused,paragraph=True)

    #Fix OCR miss read
    result = [[r[0], 'logiq' if r[1].lower() == 'loc' or r[1].lower() == 'lo' else r[1].lower()] for r in result]
    result = [ [r[0], r[1].lower()] for r in result]
    
    # now loop over the remaining strings and get the total string and the bounding box
    text = ''
    for r in result:
        text = text + r[1] + ' '

    return [image_file, text]




def get_image_sizes(image_folder_path, filenames):
    image_sizes = {}
    for filename in tqdm(filenames):
        with Image.open(os.path.join(image_folder_path, filename)) as img:
            image_sizes[filename] = list(img.size)  # Store image size as a list [width, height]
    return image_sizes




# Main method to prefrom operations
def Perform_OCR():
        
    image_folder_path = f"{env}/database/images/"
    input_file = f'{env}/database/ImageData.csv'
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
    

    
    print("Finding Calipers")
    has_calipers = find_calipers(image_folder_path, 'caliper_model', db_to_process)
    
    
    print("Finding Image Masks")
    image_masks, description_masks = find_masks(image_folder_path, 'mask_model', db_to_process, 1920, 1080)
    
    
    # Convert mask data
    outer_crop = []
    for (filename, image_mask) in image_masks:  # Unpack filename and image_mask here
        if image_mask:
            x0, y0, x1, y1 = image_mask
            x = x0
            y = y0
            w = x1 - x0
            h = y1 - y0
            outer_crop.append([filename, x, y, w, h])  # Include filename in the result
        else:
            outer_crop.append([filename, []])  # Include filename even if no mask
    
    print("Performing OCR")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_image, image_file, description_mask, image_folder_path, reader, description_kw): image_file for image_file, description_mask in description_masks}  # description_masks now include the filenames
        progress = tqdm(total=len(futures), desc='')

        # Initialize dictionary to store descriptions
        descriptions = {}

        for future in as_completed(futures):
            result = future.result()  # result is now a list with the filename and the description
            descriptions[result[0]] = result[1]  # Store description at corresponding image file
            progress.update()

        progress.close()

    print("Finding Darkness")
    darknesses = get_darkness(image_folder_path, image_masks)
    
    # Convert lists of tuples to dictionaries
    has_calipers_dict = {filename: value for filename, value in has_calipers}
    darknesses_dict = {filename: value for filename, value in darknesses}
    
    # Convert dictionaries to Series for easy mapping
    descriptions_series = pd.Series(descriptions)
    darknesses_series = pd.Series(darknesses_dict)
    has_calipers_series = pd.Series(has_calipers_dict)

    # Update dataframe using map
    db_to_process['description'] = db_to_process['ImageName'].map(descriptions_series)
    db_to_process['darkness'] = db_to_process['ImageName'].map(darknesses_series)
    db_to_process['has_calipers'] = db_to_process['ImageName'].map(has_calipers_series)

    # Process the 'outer_crop' and 'inner_crop' lists
    outer_crop_dict = {filename: values for filename, *values in outer_crop}

    outer_crop_series = pd.Series(outer_crop_dict)

    db_to_process[['crop_x', 'crop_y', 'crop_w', 'crop_h']] = db_to_process['ImageName'].map(outer_crop_series).apply(pd.Series)

    # Construct a temporary DataFrame with the feature extraction
    temp_df = db_to_process.loc[db_to_process['description'].str.len() > 0, 'description'].apply(lambda x: extract_descript_features(x, labels_dict=description_labels_dict)).apply(pd.Series)

    # Update the columns in db_out directly from temp_df
    for column in temp_df.columns:
        db_to_process[column] = temp_df[column]
        
    db_out.update(db_to_process, overwrite=True)

    db_out.to_csv(input_file,index=False)













def find_nearest_images(db, patient_id, image_folder_path):
    # Filter to only RGB images
    subset = db[db['PhotometricInterpretation'] != 'RGB'] 
    # Get index array 
    idx = np.array(fetch_index_for_patient_id(patient_id, subset))
    result = {}
    image_pairs_checked = set()

    for j,c in enumerate(idx):
        if c in image_pairs_checked:
            continue

        x = int(db.loc[c]['RegionLocationMinX0'])
        y = int(db.loc[c]['RegionLocationMinY0'])
        w = int(db.loc[c]['RegionLocationMaxX1']) - x
        h = int(db.loc[c]['RegionLocationMaxY1']) - y

        
        img_list = []
        for i,image_id in enumerate(idx):
            
            file_name = db.loc[image_id]['ImageName']
            full_filename = os.path.join(image_folder_path, file_name)
            img = Image.open(full_filename)
            img = np.array(img).astype(np.uint8)
            (rows, cols) = img.shape[0:2]
            if rows >= y + h and cols >= x + w:
                img,_ = make_grayscale(img)
                img = img[y:y+h,x:x+w]
            else: # fill in all ones for an image that will be distant
                img = np.full((h,w),255,dtype=np.uint8)
            img = img.flatten()
            img_list.append(img)
        
        img_stack = np.array(img_list, dtype=np.uint8)
        
        
        img_stack = np.abs(img_stack - img_stack[j, :])
        img_stack = np.mean(img_stack, axis=1)
        img_stack[j] = 1000
        sister_image = np.argmin(img_stack)
        distance = img_stack[sister_image]


        # Save result for the current image
        result[c] = {
            'image_filename': db.loc[c]['ImageName'],
            'sister_filename': db.loc[idx[sister_image]]['ImageName'],
            'distance': distance
        }

        # Save result for the sister image, if not already done
        if idx[sister_image] not in result:
            result[idx[sister_image]] = {
                'image_filename': db.loc[idx[sister_image]]['ImageName'],
                'sister_filename': db.loc[c]['ImageName'],
                'distance': distance
            }

        # Add the images to the set of checked pairs
        image_pairs_checked.add(c)
        image_pairs_checked.add(idx[sister_image])

    return result


def process_patient_id(pid, db_out, image_folder_path):
    subset = db_out[db_out['Accession_Number'] == pid]
    
    result = find_nearest_images(subset, pid, image_folder_path)
    idxs = result.keys()
    for i in idxs:
        subset.loc[i, 'closest_fn'] = result[i]['sister_filename']
        subset.loc[i, 'distance'] = result[i]['distance']
    return subset


def Pre_Process():
    input_file = f'{env}/database/ImageData.csv'
    case_file = f'{env}/database/CaseStudyData.csv'
    image_folder_path = f"{env}/database/images/"
    db_out = pd.read_csv(input_file)
    case_data = pd.read_csv(case_file)

    # Check if 'processed' column exists, if not create it
    if 'processed' not in db_out.columns:
        db_out['processed'] = False

    # Remove rows with missing data in crop_x, crop_y, crop_w, crop_h
    db_out = db_out.dropna(subset=['crop_x', 'crop_y', 'crop_w', 'crop_h'])

    # Filter the rows where 'processed' is False
    db_to_process = db_out[db_out['processed'] == False]

    print("Finding Similar Images")
    patient_ids = db_to_process['Accession_Number'].unique()

    db_to_process['closest_fn']=''
    db_to_process['distance'] = -1

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, tqdm(total=len(patient_ids), desc='') as progress:
        futures = {executor.submit(process_patient_id, pid, db_to_process, image_folder_path): pid for pid in patient_ids}

        for future in as_completed(futures):
            result = future.result()
            if result is not None and not result.empty:
                db_to_process.update(result)
            progress.update()

    db_to_process = choose_images_to_label(db_to_process, case_data)
    db_to_process = add_labeling_categories(db_to_process)
    
    # Update 'processed' status to True for processed rows and merge back to the original dataframe
    db_to_process['processed'] = True
    # List all columns that are common to both dataframes
    common_columns = db_out.columns.intersection(db_to_process.columns)

    # Set 'ImageName' as the key for merging
    key_column = 'ImageName'

    # Take in all rows from db_to_process and only rows from db_out that aren't already present.
    db_out = pd.merge(db_to_process, db_out[~db_out[key_column].isin(db_to_process[key_column])], on=common_columns.tolist(), how='outer')

    if 'latIsLeft' in db_out.columns:
        db_out = db_out.drop(columns=['latIsLeft'])
    db_out.to_csv(input_file, index=False)

