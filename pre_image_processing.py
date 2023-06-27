import pandas as pd
from PIL import Image
import cv2, os, re
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import easyocr
from ML_processing.caliper_model import *
from ML_processing.mask_model import *
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
            'axilla':['axilla'],
            'supraclavicular':['superclavicular','supraclavicular'],
            'subclavicular':['subclavicular','subclavcular']},
    'laterality':{'left':['lt','left'],
                  'right':['rt','right']},
    'orientation':{'long':['long'],
                    'trans':['trans'],
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
    
    # Regular expression to match time substrings of the form HH:MM or HH.MM
    # does not need to have blank spaces
    pattern = r'\d{1,2}[:.]\d{2}'
    
    # Find all matches in the input text
    matches = re.findall(pattern, str(text))
    
    if len(matches) > 0:
        # Return the first match
        time = matches[0].replace('.', ':')
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
    
    if id in db['anonymized_accession_num'].tolist():
         indices= db.index[db['anonymized_accession_num']==id].tolist()
    else:
        indices = []
    return indices



def find_nearest_images(db, patient_id, image_folder_path):
    idx = np.array(fetch_index_for_patient_id(patient_id, db))
    num_images = len(idx)
    result = {}

    for j,c in enumerate(idx):
        x = int(db.loc[c]['crop_x'])
        y = int(db.loc[c]['crop_y'])
        w = int(db.loc[c]['crop_w'])
        h = int(db.loc[c]['crop_h'])
        img_stack = np.zeros((num_images,w*h)).astype(np.uint8)
        for i,image_id in enumerate(idx):
            file_name = db.loc[image_id]['image_filename']
            full_filename = os.path.join(image_folder_path, file_name)
            img = Image.open(full_filename)
            img = np.array(img).astype(np.uint8)
            (rows, cols) = img.shape[0:2]
            if rows >= y + h and cols >= x + w:
                img,_ = make_grayscale(img)
                img = img[y:y+h,x:x+w] # this can break if the root image is too big
            else: # fill in all ones for an image that will be distant
                img = np.full((h,w),255,dtype=np.uint8)
            img = img.flatten()
            img_stack[i,:] = img
        img_stack = np.abs(img_stack - img_stack[j, :])
        img_stack = np.mean(img_stack, axis=1)
        img_stack[j] = 1000
        sister_image = np.argmin(img_stack)
        distance = img_stack[sister_image]
        result[c] = {
            'image_filename': db.loc[c]['image_filename'],
            'sister_filename': db.loc[idx[sister_image]]['image_filename'],
            'distance': distance
        }
    return result



def get_description(image_folder_path, description_masks, reader, kw_list):
    
    image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

    descriptions = []
    
    for image_file, description_mask in tqdm(zip(image_files, description_masks), total=min(len(image_files), len(description_masks))):

        image = Image.open(os.path.join(image_folder_path, image_file)).convert('L')

        for mask in description_mask:
            x0, y0, x1, y1 = mask
            cropped_image = image.crop((x0, y0, x1, y1))
        
            # Convert the PIL Image to a numpy array
            cropped_image_np = np.array(cropped_image)
        
            # Apply blur to help OCR
            img_focused = cv2.GaussianBlur(cropped_image_np, (3, 3), 0)

        
        
        result = reader.readtext(img_focused,paragraph=True)
        if False: #Debug
            plt.figure()
            plt.imshow(img_focused)
            plt.title(result)
            plt.show()
        
        #Fix OCR miss read
        result = [[r[0], 'logiq' if r[1].lower() == 'loc' or r[1].lower() == 'lo' else r[1].lower()] for r in result]
        
        result = [ [r[0], r[1].lower()] for r in result if contains_substring(r[1].lower(), kw_list) ]
        
        #print('Easyocr results v2: ',result)
        lengths = [ len(r[1]) for r in result ]
        
        if len(lengths)==0: # no valid text detected in region
            descriptions.append('') # return empty and try again
        
        # now loop over the remaining strings and get the total string and the bounding box
        x0 = 10000
        y0 = 10000
        x1 = 0
        y1 = 0
        text = ''
        for r in result:
            for c in r[0]:
                x0 = min(x0,c[0])
                y0 = min(y0,c[1])
                x1 = max(x1,c[0])
                y1 = max(y1,c[1])
            text = text + r[1] + ' '
        
        if len(text)==0:
            text = ''
        
        descriptions.append(text)
        #print(text)

    return descriptions



def find_mixed_lateralities( db ):
    ''' returns a list of all patient ids for which the study contains both left and right lateralities (to be deleted for now)
    
    Args:
        db is a dataframe with one row per image, must have columns for patient_id and laterality
        
    Returns:
        list of patient ids from db for which the lateralities are mixed
    '''
    db['latIsLeft']=(db['laterality']=='left')
    df = db.groupby(['anonymized_accession_num']).agg({'anonymized_accession_num':'count', 'latIsLeft':'sum'})
    df['notPure'] = ~( (df['latIsLeft']==0) |  (df['latIsLeft']==df['anonymized_accession_num']) )
    
    mixedPatientIDs = df[df['notPure']].index.tolist()
    return mixedPatientIDs

def choose_images_to_label(db):
    
    db['label']=True

    # find all of the rows with calipers
    caliper_indices = np.where( db['has_calipers'])[0]

    # loop over caliper rows and tag twin images (not efficient)
    for idx in caliper_indices:
        distance = db.loc[idx,'distance']
        if distance <= 5:
            twin_filename = db.loc[idx,'closest_fn']
            twin_idx = np.where( db['image_filename'] == twin_filename )[0][0]
            db.loc[twin_idx,'label'] = True # redundant
            db.loc[idx,'label'] = False
            
    # set label = False for all non-breast images
    db.loc[(db['area'] != 'breast') & (db['area'] != 'unknown'), 'label'] = False
    
    mixedIDs = find_mixed_lateralities( db )
    db.loc[np.isin(db['anonymized_accession_num'],mixedIDs),'label']=False
    
    return db

def add_labeling_categories(db):
    db['label_cat'] = ''
    num_rows = db.shape[0]
    for i in range(num_rows):
        if db.loc[i,'label']:
            orient = db.loc[i,'orientation']
            image_type = db.loc[i,'PhotometricInterpretation']
            if image_type=='RGB':
                label_cat = 'doppler'
            elif orient in ['trans','long']:
                label_cat = orient
            else:
                label_cat = 'other'
            db.loc[i,'label_cat'] = label_cat
    return db



def get_darkness(image_folder_path, image_masks):
    
    image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

    darknesses = []
    
    for i, image in enumerate(tqdm(image_files)):
        
        image = Image.open(os.path.join(image_folder_path, image)).convert('L')
        image_np = np.array(image)  # Convert image to numpy array
            
        x, y, w, h = image_masks[i][0]
        
        img_us = image_np[y:y+h, x:x+w]
        img_us_gray, isColor = make_grayscale(img_us)
        _,img_us_bw = cv2.threshold(img_us_gray, 20, 255, cv2.THRESH_BINARY)
        num_dark = np.sum( img_us_bw == 0)
        darknesses.append(100*num_dark/(w*h))
        
    return darknesses


# Main method to prefrom operations
def Perform_OCR():
        
    image_folder_path = f"{env}/database/images/"
    input_file = f'{env}/database/unlabeled_data.csv'
    db_out = pd.read_csv(input_file)

    files = db_out['image_filename']
    image_numbers = np.arange(len(files))

    # Check if any new features are missing in db_out and add them
    new_features = ['processed', 'crop_x', 'crop_y', 'crop_w', 'crop_h', 'description', 'has_calipers', 'sector_detected', 'darkness', 'area', 'laterality', 'orientation', 'clock_pos', 'nipple_dist']
    missing_features = set(new_features) - set(db_out.columns)
    for nf in missing_features:
        db_out[nf] = None
    
    
    print("Finding Calipers")
    has_calipers = find_calipers(image_folder_path, 'caliper_model')
    
    
    print("Finding Image Masks")
    image_masks, description_masks = find_masks(image_folder_path, 'mask_model', 1292, 970)
    
    print("Performing OCR")
    descriptions = get_description(image_folder_path, description_masks, reader, kw_list = description_kw)
    
    
    # Convert mask data
    converted_masks = []
    for mask_list in image_masks:
        for mask in mask_list:
            x0, y0, x1, y1 = mask
            x = x0
            y = y0
            w = x1 - x0
            h = y1 - y0
            converted_masks.append([x, y, w, h])
    
    
    

    print("Finding Darkness")
    darknesses = get_darkness(image_folder_path, image_masks)
    
    
    for i in image_numbers:
        # insert into total database
        new_features = ['crop_x', 'crop_y', 'crop_w', 'crop_h', 'has_calipers', 'description', 'darkness']
        crop_x, crop_y, crop_w, crop_h = converted_masks[i]
        db_out.loc[i,'crop_x'] = crop_x
        db_out.loc[i,'crop_y'] = crop_y
        db_out.loc[i,'crop_w'] = crop_w
        db_out.loc[i,'crop_h'] = crop_h
        db_out.loc[i,'description'] = descriptions[i]
        db_out.loc[i,'darkness'] = darknesses[i]
        db_out.loc[i,'has_calipers'] = has_calipers[i]
        if len(descriptions[i])>0:
            feature_dict = extract_descript_features( descriptions[i], description_labels_dict )
            display_str = ''
            for feature in feature_dict.keys():
                db_out.loc[i,feature] = feature_dict[feature]
                display_str = display_str + feature_dict[feature] + ' '
        else:
            display_str = ''


    #db_out = extract_descript_features_df( db_out, description_labels_dict )
    db_out.to_csv(input_file,index=False)




def Pre_Process():
    
    input_file = f'{env}/database/unlabeled_data.csv'
    image_folder_path = f"{env}/database/images/"
    db_out = pd.read_csv(input_file)
    
    # Finding closest images
    patient_ids = db_out['anonymized_accession_num'].unique()
            
    db_out['closest_fn']=''
    db_out['distance'] = -1

    for pid in tqdm(patient_ids):
        result = find_nearest_images(db_out, pid, image_folder_path)
        idxs = result.keys()
        for i in idxs:
            db_out.loc[i,'closest_fn'] = result[i]['sister_filename']
            db_out.loc[i,'distance'] = result[i]['distance']

    
    db_out = choose_images_to_label(db_out)
    db_out = add_labeling_categories(db_out)
    
    
    db_out = db_out.drop(columns=['latIsLeft'])
    db_out.to_csv(input_file,index=False)