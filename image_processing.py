import usImageProc as uip

import pandas as pd
from PIL import Image
import cv2
import os
import re
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import largestinteriorrectangle as lir
from tqdm import tqdm
from time import sleep
env = os.path.dirname(os.path.abspath(__file__))

import re

import easyocr
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

# image id followed by dictionary of corrections to apply
corrections = { 4094:{'cleaned_text':'long lt breast 10.00 scm fn area palpated', 'area':'breast'} }







# note these numbers refer to the filenames, e.g.
# 2 → 000002_cropped.png
# subtract 1 from each to get the index in the alphabetized list

sectors = [2,11,12,94,100,287,291,522,523,525,526,
          527,528,530,531,533,536,544,559,635,637,
          638,639,640,641,645,646,892,916,917,918,
          919,920,972,973,978,983,984,1140,1146,1147,
          1150,1498,1553,1555,1556,1557,1710,1711,
           1712,1713,1714,1715,1716,1717,1718,1856,
          1857,1861,1862,1863,1864,1973,1978,1979,
          1982,1984,1985,1987,1988,1992,1995,1998]
traps = [ ]
len(sectors)

# this cell is for text utilities
import re

# Helper functions for text processing - mostly used to extract description from image

def contains_substring(input_string, substring_list):
    input_string = str(input_string).lower()  # Convert input string to lowercase
    for substring in substring_list:
        if substring.lower() in input_string:  # Convert substring to lowercase
            return True
    return False

def has_digit(input_string):
    pattern = re.compile(r'\d') # Compile a regular expression pattern to match digits
    return bool(pattern.search(input_string)) # Return True if a match is found, False otherwise

def text_freq_df_column( df, col = 'cleaned_text'):
    """Compute frequencies of words in column of strings.  Not case sensitive.
    
    Helps to identify keywords and also mispellings
    
    Args:
        df: Pandas dataframe 
        col: column of strings for frequency analysis (defaults to 'description')
    
    Returns:
        counts:  pd series of counts indexed by words in descending order of frequency
        
    Example:
        db = pd.read_csv('database_total.csv')
        db = db.fillna('')
        counts = text_freq_df_column(db)
        counts[0:60]
    """

    soup = []
    for d in df[col]:
        if d is not None:
            split_soup = d.split(' ')
            for s in split_soup:
                s = s.lower()
                s = s.replace('/','')
                if s !='' and not has_digit(s):
                    soup.append(s)
    print(len(soup))
    counts = pd.Series(soup).value_counts()
    return counts

def pad_substrings_with_spaces(substrings, input_str):
    # Iterate over each substring in the list
    for substring in substrings:
        # Replace each occurrence of the substring with the same substring padded with spaces
        input_str = input_str.replace(substring, f" {substring} ")

    # remove duplicate spaces    
    words = input_str.split()
    input_str = ' '.join(words)
    
    # Return the string
    return input_str

def clean_text(input_str, sub_dict, kw_expand, kw_contract):
    """Process input string and add spaces around keywords, substitute for common OCR mistakes
    
    Args:
        input_str: string to be processed
        kw_list:  any word in this list will be searched and padded with spaces
        repair_dict: keys are substrings that will be replaced by their corresponding values
    
    Returns:
        output_str: repaired string
    """
    
    # first make substitutions
    for k in sub_dict.keys():
        input_str = input_str.replace( k, sub_dict[k] )
        
    # now add spaces around all substrings in kw_expand
    for substring in kw_expand:
        input_str = input_str.replace( substring, f" {substring} " )
        
    # remove duplicate spaces
    words = input_str.split()
    input_str = ' '.join(words)
    
    # remove spaces from words in kw_contract
    for substring in kw_contract:
        input_str = input_str.replace( substring, substring.replace(' ','') )
        
    return input_str

def clean_text_df( df, sub_dict, kw_expand, kw_contract, col = 'cleaned_text'):
    df[col] = df[col].apply(clean_text, args = (sub_dict, kw_expand, kw_contract) )

def label_parser(x, label_dict={}):
    for k in label_dict.keys():
        
        labels = label_dict[k]
        if contains_substring(x,labels):
            return k
    return 'unknown'

def find_time_substring(text):
    # Regular expression to match time substrings of the form HH:MM or HH.MM
    # does not need to be have blank spaces
    pattern = r'\d{1,2}[:.]\d{2}'
    
    # Find all matches in the input text
    matches = re.findall(pattern, str(text))
    
    if len(matches)==0:
        return 'unknown'
    else:
        # Return only the first match
        time = matches[0].replace('.',':')
        return time
    
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
    
def extract_descript_features_df(df, labels_dict, col = 'cleaned_text'):

    # first extract simple text features
    for feature in labels_dict.keys():
        levels_dict = labels_dict[feature]
        df[feature] = df[col].apply( label_parser, label_dict = levels_dict )
    
    # extract clock_position
    df['clock_pos'] = df[col].apply( find_time_substring )
    
    # extract nipple_dist
    df['nipple_dist'] = df[col].apply( find_cm_substring )
    
    return df

image_folder_path = f"{env}/database/images/"
proc_images_folder = f"{env}/database/test/"

input_file = f'{env}/database/unlabeled_data.csv'
output_file = f'{env}/database/temp.csv'

# processing configuration
debug = False
write_images = False
display_images = False

# open database and get filenames to be processed
db_in = pd.read_csv(input_file)

#Get first part of data
db_in = extract_descript_features_df( db_in, description_labels_dict )

files = db_in['image_filename']
image_numbers = np.arange(len(files))

# open or create output database
import os.path
check_db_out = os.path.isfile(output_file)
if check_db_out:
    db_out = pd.read_csv(output_file)
else:
    db_out = db_in.copy()
    new_features = ['processed','crop_x', 'crop_y', 'crop_w', 'crop_h', 'description', 'size', 'sector_detected', 'darkness','area','laterality','orientation','clock_pos','nipple_dist']
    for nf in new_features:
        db_out[nf] = None
    db_out['processed'] = False
                

for i in tqdm(image_numbers):
    #sleep(0.01)
    if not db_out['processed'][i]:
        file_name = db_in['image_filename'][i]
        us_x = min(db_in['RegionLocationMinX0'][i],0)
        us_y = db_in['RegionLocationMinY0'][i]
        us_w = db_in['RegionLocationMaxX1'][i]-us_x
        us_h = db_in['RegionLocationMaxY1'][i]-us_y
        rect_us = (us_x, us_y, us_w, us_h)
        #print('rect_us: ', rect_us)
        # Check if the file is an image
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Construct the full path to the image file

            if debug:
                print('Processing: ', file_name )

            full_filename = os.path.join(image_folder_path, file_name)
            image_out_path = os.path.join(proc_images_folder, file_name)

            # Open the image file and store it in an image object
            img = Image.open(full_filename)

            # recast image as numpy array
            img = np.array(img)
            img_orig = img.copy()

            img_dict = uip.img_processor(img, reader, 
                                         rect_US = rect_us,
                                         kw_list = description_kw)
            if debug: 
                print(img_dict)
                print('Processing Complete: ', file_name)

            # insert into total database
            new_features = ['crop_x', 'crop_y', 'crop_w', 'crop_h', 'description', 'size', 'is_sector', 'darkness']
            crop_x, crop_y, crop_w, crop_h = img_dict['rect_crop']
            description = img_dict['text_description']
            db_out.loc[i,'crop_x'] = crop_x
            db_out.loc[i,'crop_y'] = crop_y
            db_out.loc[i,'crop_w'] = crop_w
            db_out.loc[i,'crop_h'] = crop_h
            db_out.loc[i,'description'] = description
            db_out.loc[i,'size'] = img_dict['text_size']
            db_out.loc[i,'sector_detected'] = img_dict['sector_detected']
            db_out.loc[i,'processed'] = True
            db_out.loc[i,'darkness'] = img_dict['darkness']
            if len(description)>0:
                feature_dict = extract_descript_features( description, description_labels_dict )
                display_str = ''
                for feature in feature_dict.keys():
                    db_out.loc[i,feature] = feature_dict[feature]
                    display_str = display_str + feature_dict[feature] + ' '
            else:
                display_str = ''

            if write_images or display_images: # add description and crop region to image
                img_orig = uip.add_rect(img_orig, img_dict['rect_crop'])
                img_orig = uip.add_text(img_orig, display_str)

            if write_images:
                cv2.imwrite(image_out_path,img_orig)
            if display_images:
                img2 = img_orig.copy()
                img2 = uip.add_rect(img2, img_dict['rect_machine'])
                img2 = uip.add_rect(img2, img_dict['rect_description'])
                img2 = uip.add_rect(img2, img_dict['rect_colorbar'])
                if len(img_dict['rect_sizebox'])>0:
                    img2 = uip.add_rect(img2, img_dict['rect_sizebox'])
                    
                fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20, 15)) 

                ax1.imshow(img_orig,cmap='gray')   
                ax2.imshow(img2,cmap='gray')
                fig.show()                
                
db_out.to_csv(output_file,index=False)