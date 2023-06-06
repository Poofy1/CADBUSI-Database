import pandas as pd
import cv2
import numpy as np
import largestinteriorrectangle as lir
import easyocr

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