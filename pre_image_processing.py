import pandas as pd
from PIL import Image
import cv2, os, re
import numpy as np
from matplotlib import pyplot as plt
import largestinteriorrectangle as lir
from tqdm import tqdm
import easyocr
env = os.path.dirname(os.path.abspath(__file__))


# helper functions for image processing

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

def make_mask( img_gray, thresh = 127):
    """Convert grayscale uint8 numpy array to binary numpy array using only 0 and 255
    
    Args:
        img_gray:  uint8 numpy array
        thresh: (default 127) values above thresh get assigned 255, else 0
        
    Returns:
        img_bw:  binary uint8 numpy array with only 0 and 255
    """
    
    _,img_bw = cv2.threshold(img_gray,thresh,255,cv2.THRESH_BINARY)
    return img_bw
    
def blackout_rectangle( img, rect, val = 0 ):
    """fill a rectangular region in the last two dimensions of the input numpy array, default fill value 0
    
    Args:
        img: numpy array that is H x W or C x H x w
        rect: (x,y,w,h) for region to be filled, note that y corresponds to the H dimension
        val: fill-value, defaults to 0
        
    Returns:
        img: returns the image with filled rectangle
    """

    dim = len(img.shape)
    x,y,w,h = rect
    if dim > 2:
        img[:,y:y+h,x:x+w] = val
    else:
        img[y:y+h,x:x+w] = val
    return img

def blackout_rectangle_exterior( img, rect, val = 0):
    """fill outside a rectangular region in the last two dimensions of the input numpy array, default fill value 0
    
    Args:
        img: numpy array that is H x W or C x H x w
        rect: (x,y,w,h) for region to be filled, note that y corresponds to the H dimension
        val: fill-value, defaults to 0
        
    Returns:
        img2: returns the image with region outside rectangel filled with value
    """

    # consider refactoring to avoid forming img2 by just setting array slices to val
    
    dim = len(img.shape)
    img2 = np.full_like(img,val)
    x,y,w,h = rect
    if dim > 2:
        img2[y:y+h,x:x+w,:] = img[y:y+h,x:x+w,:]
    else:
        img2[y:y+h,x:x+w] = img[y:y+h,x:x+w]
    return img2

def first_nonzero(arr, axis, invalid_val=-1):
    """Return index of first nonzero entry in array along specified axis
    
    Args:
        arr: at least 2D numpy array (might work with 1D)
        axis:  dimension to search along for first nonzero (no safety check)
        invalid_val: ?
        
    Returns:
        numpy array with one fewer dimensions containing index of first nonzero
    """
    
    # figure out role of invalid_val
    
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    """Return index of lst nonzero entry in array along specified axis
    
    Args:
        arr: at least 2D numpy array (might work with 1D)
        axis:  dimension to search along for last nonzero (no safety check)
        invalid_val: ?
        
    Returns:
        numpy array with one fewer dimensions containing index of last nonzero
    """
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def mask_fill_hv(img_bw):
    """Fills interior zeros in rows and columns with 255.  Good for rectangular masks.
    
    Args:
        img_bw: HxW uint8 numpy array (binary image)
        
    Returns:
        img_fill_gaps: HxW uint8 numpy array (binary image) with horiz and vert interior gaps filled
    """

    # refactor to do without loops?  
    
    img_fill_gaps = img_bw.copy()
    img_fill_gaps = np.array(img_fill_gaps,dtype='uint8')
    numrows, numcols = img_fill_gaps.shape
    
    # first fill columns
    first_nz_cols = first_nonzero(img_fill_gaps,axis=0)
    last_nz_cols = last_nonzero(img_fill_gaps,axis=0)
    for c in range(numcols):
        if first_nz_cols[c] > -1:
            img_fill_gaps[first_nz_cols[c]:last_nz_cols[c]+1,c] = 255
    
    # second fill_rows
    first_nz_rows = first_nonzero(img_fill_gaps,axis=1)
    last_nz_rows = last_nonzero(img_fill_gaps,axis=1)
    width_row = np.zeros(numrows)
    for r in range(numrows):
        left = first_nz_rows[r]
        if left > -1:
            right = last_nz_rows[r]
            width_row[r] = right - left
            img_fill_gaps[r,left:right+1] = 255
    
    return img_fill_gaps
            
def row_widths(img_bw):
    """Compute width of positive mask in each row.  Used to classify shape of connected region.
    
    Args:
        img_bw: HxW uint8 numpy array (binary image)

    Returns:
        widths: 1D numpy array with H integers for width of mask in each row
    """

    # refactor to do without loop

    img_fill_gaps = img_bw.copy()
    img_fill_gaps = np.array(img_fill_gaps,dtype='uint8')
    numrows, numcols = img_fill_gaps.shape
    
    first_nz_rows = first_nonzero(img_fill_gaps,axis=1)
    last_nz_rows = last_nonzero(img_fill_gaps,axis=1)
    
    widths = np.zeros(numrows)
    for r in range(numrows):
        left = first_nz_rows[r]
        right = last_nz_rows[r]
        if left > -1:
            widths[r] = right - left
            
    return widths

def is_sector(img_bw, num_rows = 150):
    """Detect sector shaped masks
    
    Args:
        img_bw:  HxW numpy uint8 binary image
        num_rows: max number of rows to test for increasing width (default 150)
        
    Returns:
        boolean True if sector mask is detected  
    """
    
    # rework this to make it more robust, short wide sectors may not be detected
    
    widths = np.trim_zeros( row_widths(img_bw) )
    first = min( len(widths), num_rows )
    widths = widths[0:first]
    max_width = np.max(widths)
    mean_width = np.mean(widths)
    return (max_width-mean_width)/max_width > 0.05

def extract_largest_connected(img_bw):
    """Extra largest connected component of mask in img_bw
    
    Erode a little to disconnect components.  Find largest.  Dilate to restore mask to full size.
    
    Args:
        img_bw: HxW numpy uint8 binary image
        
    Returns:
        img_dilated: HxW numpy uint8 with only largest connected
        
    Requires:
        opencv imported as cv2
    """
    
    # Could add some default input parameters for controlling erosion, dilation, and connectivity
    
    kernel = np.ones((3,3), np.uint8)
    img_eroded = cv2.erode(img_bw,kernel,iterations=3)

    # retrieve largest connected component
    nbComp,output,stats,_ = cv2.connectedComponentsWithStats(img_eroded, connectivity = 4)
    sizes = stats[1:nbComp,-1]
    maxLabel = np.argmax(sizes)+1
    img_connect = np.zeros(output.shape)
    img_connect[ output == maxLabel] = 255

    # dilate to restore original size for mask
    kernel = np.ones((3,3), np.uint8)
    img_dilated = cv2.dilate(img_connect,kernel,iterations=3)

    return img_dilated
    
def find_colorbar(img, color_image = False):
    """Compute rectangle containing "colorbar" on left side of Logiq E9 images
    
    Args:
        img: one or three channel numpy uint8 image
        color_image:  boolean that indicates if img was orginally color
        
    Returns:
        rect: contains rectangle region that includes colorbar and is anchored to left edgue of image
        
    Requires:
        opencv imported as cv2
    
    """
    
    # make this configurable so that we can setup templates depending on the machine

    img_bw = make_mask(img, thresh = 140)
    img_bw = blackout_rectangle_exterior(img_bw,(10,300,50,60))

    contours, _ = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    c = max(contours,key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

    # make adjustments to snip out entire colorbar
    if color_image:
        rect = (0, y-20, 60, 220)
    else:
        rect = (0, y, x + w, 165)

    return rect

def extract_largest_rectangle(mask, tol = 0.005):
    """Find rectangle of largest area inside mask
    
    Args:
        mask:  HxW numpy uint8 binary image
        tol: fit contour with relative error tol
        
    Returns:
        rect: (x,y,w,h) 
        
    Requires:
        lir package
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    c = max(contours,key=cv2.contourArea)
    epsilon = tol*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,epsilon,True)
    poly = np.array( [[ [a[0][0], a[0][1] ] for a in approx ]] )
    rect = lir.lir(poly)
    return(rect)

def add_rect( img, rect, color = (255,255,255), thickness = 2):
    """Draw rectangle on numpy image
    
    Args:
        img: CxHxW or HxW numpy uint8 image
        rect: (x,y,w,h) 
        color: (R,G,B) defaults to white
        thickness: pixel width of rectangle boundary (default 2)
        
    Returns:
        CxHxW or HxW numpy uint8 image - copy of img with rectangle
        
    Requires:
        opencv imported as cv2
    """
    
    if len(rect)==0: # in case no rectangle passed
        rect = (0,0,1,1)
    x,y,w,h = rect
    return cv2.rectangle( img, (x,y),(x+w,y+h),color,thickness)

def add_text( img, text, org = (100,50), color = (255,255,255), thickness = 2 ):
    """Draw text on numpy image
    
    Args:
        img: CxHxW or HxW numpy uint8 image
        text: string
        org: position of lower corner of text (y,x)
        color: (R,G,B) defaults to white
        thickness: pixel width of rectangle boundary (default 2)
        
    Returns:
        CxHxW or HxW numpy uint8 image - copy of img with text
        
    Requires:
        opencv imported as cv2
    """
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    if len(text)==0:
        text='No Label Found'
    return cv2.putText(img,text,org,font,fontScale,color,thickness,cv2.LINE_AA)

def bbox(img_bw):
    """Returns top, bottom, left, right of mask in binary numpy image
    
    Args:
        img_bw: HxW uinty numpy array with binary image
        
    Returns:
        rmin: topmost index of mask
        rmax: bottommost index of mask
        cmin: leftmost index of mask
        cmax: rightmost index of mask
    """

    rows = np.any(img_bw, axis=1)
    cols = np.any(img_bw, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def fill_bbox(img_bw, val = 255):
    """Find bounding coordinates of mask and fill rectangle to generate smallest rectangular mask
    
    Args:
        img_bw:  HxW uint8 binary numpy image,
        val:  fill value 0 to 255 (default 255)
        
    Returns:
        HxW uint8 binary numpy image with rectangular mask
    """
    
    top,bottom,left,right = bbox(img_bw)
    img_bw = np.zeros( img_bw.shape, dtype = np.uint8)
    img_bw[top:bottom,left:right]=val
    return img_bw

def easyocr_coord_to_rect( coord ):
    """Converts easyocr bounding rectangle to (x,y,w,h) form
    
    Args:
        coord: nested list containing corners of bounding rectangle
        
    Returns:
        tuple (x,y,w,h)
        
    Example: easyocr_coord_to_rect( [[12, 134], [138, 134], [138, 194], [12, 194]] )
             returns (12,134,126,60)
    """
    
    x = coord[0][0]
    y = coord[0][1]
    w = coord[1][0]-x
    h = coord[2][1]-y
    return (x,y,w,h)

def contains_substring(input_string, substring_list):
    for substring in substring_list:
        if substring in input_string:
            return True
    return False

def get_text_box(img, reader, rect = (22,500,811,220), kw_list = None):
    """Extract text from rectangular region and ensure each detected string contains one keyword
    
    Args:
        img: HxW or CxHxW uint8 numpy array
        reader: reader object from easyocr
        rect: (x,y,w,h) for seeking text (defaults to (22,500,811,220) )
        kw_list: list of keywords, at least one of which must be a substring for accepting text
    
    Returns:
        rect_txt: bounding rectangle that contains detected text
        text: text extracted by easyocr
        
    Requires:
        opencv imported as cv2
    
    """

    # FUTURE: add the ability to handle empty kw_list
    
    img_focused = blackout_rectangle_exterior( img, rect)
    result = reader.readtext(img_focused,paragraph=True)
    
    result = [ [r[0], r[1].lower()] for r in result if contains_substring(r[1].lower(), kw_list) ]
    
    #print('Easyocr results v2: ',result)
    lengths = [ len(r[1]) for r in result ]
    
    if len(lengths)==0: # no valid text detected in region
        return [],'' # return empty and try again
    
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
    rect_txt = (x0, y0, x1-x0, y1-y0)
    
    if len(text)==0:
        text = ''

    return rect_txt, text

def size_box_extracter(img, reader):
    """Get bounding box and text from "calipers size box" in lower right corner of image if present
    
    Look at bottom row for a segment of at least 120 nonzero pixels to indicate the presence of a
    size box.  Look above that position for complete box.
    
    Args:
        img: HxW or HxWxC numpy uint8 array
        reader: reader object from easyocr
    
    Returns:
        rect_txt:  bounding rectangle for size box (x,y,w,h)
        size_string:  string containing all of the extracted sizes  
    """
    
    # should pass-in the easyocr reader instead of grabbing from global scope
    # this code is really specific to images from LOGIQ E9, it would be nice to have something which simply finds rectangles

    img_gray,_ = make_grayscale(img)
    last_row = img_gray[-1,:]
    last_row[ last_row <= 110 ] = 0
    nz = np.nonzero(last_row)[0]
    width = len(nz)
    if width < 120:
        return [],''
    
    left = nz[0]
    right = nz[-1]
    stripe = img_gray[:,left:right+1]
    mean_horiz = np.mean(stripe,axis=1)
    std_horiz = np.std(stripe,axis=1)
    constant_rows = (mean_horiz > 100) & (std_horiz < 3)
    constant_rows[0:550] = False
    idx = np.where(constant_rows)[0]
    top = idx[0]
    x = left
    y = top
    w = right-left+1
    h = img_gray.shape[1]-top
    rect_box = (x,y,w,h)
    
    img_cropped = img[y:y+h,x:x+w]

    result = reader.readtext(img_cropped,paragraph=True)
    
    size_string = result[0][1]
    
    return rect_box, size_string

def img_processor(img, reader, rect_US = (0,101,818,554) , debug = False, kw_list = None):
    """Process Ultrasound Image from LOGIQ E9
    
    Args:
        img: HxW or HxWxC uint8 numpy array
        reader: reader object from easyocr
        rect_US: (x,y,w,h) for part of image that constains US data (get from dicom header)
        debug: True to print extra info (defaults to False)
        
    Returns:
        dictionary with keys and values for 
            rect_machine
            rect_colorbar
            rect_sizebox
            rect_description 
            rect_crop
            text_description
            text_size
            darkness: 0-100 percent dark pixels

        
    Requires:
    
    """
    
    us_x, us_y, us_w, us_h = rect_US
    us_w = us_w - 28 #subtract offset for ruler
    rect_US = (us_x, us_y, us_w, us_h)
    
    # convert to grayscale and crop out borders
    img_gray, isColor = make_grayscale(img)
    img_gray = blackout_rectangle_exterior(img_gray, rect_US)
    
    # get mask for largest connected component in US region
    img_bw = make_mask(img_gray, thresh = 1)
    img_bw = extract_largest_connected(img_bw)
    
    # fill horizontal and vertical gaps in mask
    img_bw = mask_fill_hv(img_bw)
    
    # look for size box in lower right
    rect_sizebox, size_strings = size_box_extracter(img, reader)
    has_calipers = len(rect_sizebox)>0 
    
    # blackout 
    if has_calipers:
        img_bw = blackout_rectangle(img_bw, rect_sizebox)
        img = blackout_rectangle(img, rect_sizebox)
    
    # clean up rectangular mask
    sector_shape = is_sector(img_bw)
    if not sector_shape:
        img_bw = fill_bbox(img_bw) # completely fill rectangle and remove size_box again
        if has_calipers:
            img_bw = blackout_rectangle(img_bw, rect_sizebox)

    # delete color bar from left side of image
    rect_colorbar = find_colorbar(img_gray, isColor )
    img_gray = blackout_rectangle(img_gray, rect_colorbar )
    img_bw = blackout_rectangle(img_bw, rect_colorbar )
    
    # detect machine label in upper left corner and get bounding box
    rect_machine, machine = get_text_box(img, reader, rect = (0,100,400,400) ,kw_list = ['logiq','log','giq','e9'] )
    (x,y,w,h) = rect_machine
    rect_machine = (0,0,x+w,y+h)
    y_txt_bottom = y + h
    img_bw = blackout_rectangle(img_bw, rect_machine)
    
    # extract description
    w_guess = us_w-35
    h_guess = img_bw.shape[1]-101
    rect_guess = (0,y_txt_bottom,w_guess,h_guess)
    if has_calipers:
        rect_guess = (rect_guess[0],rect_guess[1],rect_sizebox[0]-rect_guess[0],rect_guess[3])
    rect_description, description = get_text_box(img, reader, 
                                                 rect = rect_guess,
                                                 kw_list = kw_list)
    
    # blackout description and below
    if len(rect_description)>0:
        ytxt = rect_description[1]
        img_gray[ytxt:,:] = 0
        img_bw[ytxt:,:] = 0
    
    # determine largest rectangle that fits inside mask
    rect_crop = extract_largest_rectangle(img_bw)
    
    # compute fraction of dark pixels inside final rectangle
    (x,y,w,h) = rect_crop
    # print('Rect Final: ',rect_final)
    img_us = img[y:y+h,x:x+w]
    img_us_gray, isColor = make_grayscale(img_us)
    img_us_bw = make_mask(img_us_gray, thresh = 20)
    num_dark = np.sum( img_us_bw == 0)
    pct_dark = 100*num_dark/(h*w)

    # build return dictionary
    img_dict = {}
    img_dict['rect_machine'] = rect_machine
    img_dict['rect_colorbar'] = rect_colorbar
    img_dict['rect_sizebox'] = rect_sizebox
    img_dict['rect_crop'] = rect_crop
    img_dict['rect_description'] = rect_description
    img_dict['text_description'] = description
    img_dict['text_size'] = size_strings
    img_dict['darkness'] = pct_dark
    img_dict['sector_detected'] = sector_shape
    
    return img_dict





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


def db_filters( db_in, db_out = None, only_breast = True, only_gray = False, only_calipers = False, max_darkness = 50):
    # db_in is the name of the csv file containing our database
    # returns a dataframe with filterrs applied and optionally writes to db_out
    pass

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
    db['has_calipers'] = ~np.isnan(db['size'].str.len())
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
    db.loc[db['area']!='breast','label'] = False
    
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





# Main method to prefrom operations
def Perform_OCR():
    image_folder_path = f"{env}/database/images/"
    proc_images_folder = f"{env}/database/test/"

    input_file = f'{env}/database/unlabeled_data.csv'

    # processing configuration
    write_images = False


    # open database and get filenames to be processed
    db_out = pd.read_csv(input_file)

    files = db_out['image_filename']
    image_numbers = np.arange(len(files))

    new_features = ['processed', 'crop_x', 'crop_y', 'crop_w', 'crop_h', 'description', 'size', 'sector_detected', 'darkness', 'area', 'laterality', 'orientation', 'clock_pos', 'nipple_dist']
    
    # Check if any new features are missing in db_out and add them
    missing_features = set(new_features) - set(db_out.columns)
    for nf in missing_features:
        db_out[nf] = None
    
    db_out['processed'] = False
                    

    for i in tqdm(image_numbers):
        #sleep(0.01)
        if not db_out['processed'][i]:
            file_name = db_out['image_filename'][i]
            us_x = min(db_out['RegionLocationMinX0'][i],0)
            us_y = db_out['RegionLocationMinY0'][i]
            us_w = db_out['RegionLocationMaxX1'][i]-us_x
            us_h = db_out['RegionLocationMaxY1'][i]-us_y
            rect_us = (us_x, us_y, us_w, us_h)
            #print('rect_us: ', rect_us)
            # Check if the file is an 
            if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Construct the full path to the image file

                
                full_filename = os.path.join(image_folder_path, file_name)
                image_out_path = os.path.join(proc_images_folder, file_name)

                # Open the image file and store it in an image object
                img = Image.open(full_filename)

                # recast image as numpy array
                img = np.array(img)
                img_orig = img.copy()

                img_dict = img_processor(img, reader, 
                                            rect_US = rect_us,
                                            kw_list = description_kw)

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

                if write_images: # add description and crop region to image
                    img_orig = add_rect(img_orig, img_dict['rect_crop'])
                    img_orig = add_text(img_orig, display_str)
                    cv2.imwrite(image_out_path,img_orig)



    db_out = extract_descript_features_df( db_out, description_labels_dict )
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