import pandas as pd
import numpy as np
import re

# this cell is for text utilities
import re

# Helper functions for text processing - mostly used to extract description from image

def contains_substring(input_string, substring_list):
    """detect any of the strings in substring_list
    
    Args:
        input_string:  target string
        substring_list:  list of strings to detect
        
    Returns:
        Boolean that is True if any of the listed strings is found in input_string
    """
    
    for substring in substring_list:
        if substring in input_string:
            return True
    return False

def has_digit(input_string):
    """ detect any digit in string
    
    Args:
        input_string:  string to inspect
        
    Returns:
        Boolean: true if any digit detected in string
    """
    
    pattern = re.compile(r'\d') # Compile a regular expression pattern to match digits
    return bool(pattern.search(input_string)) # Return True if a match is found, False otherwise

def text_freq_df_column( df, col = 'description'):
    """Compute frequencies of words in column of strings.  Not case sensitive.
    
    Helps to identify keywords and also mispellings.  Use to explore word frequency distribution.
    
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
    """Add spaces around each detected substring so that 5cm → 5 cm, etc.
    
    Args:
        substrings:  list of strings that need added padding
        input_str:  target string
        
    Returns:
        string:  each substring will now include a space on both ends
    """
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
        sub_dict:  dictionary - for each key:value pair any instance of key in input_str is replace by value
        kw_expand:  list of keywords that will be padded with spaces
        kw_contract:  list of strings containing spaces for which spaces will be removed, e.g. sub clavicular → subclavicular
    
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

def clean_text_df( df, sub_dict, kw_expand, kw_contract, col = 'description'):
    """Process column and apply clean_text() to each string in the column
    
        Args:
        df:  pandas dataframe with column to be cleaned
        sub_dict:  dictionary - for each key:value pair any instance of key in input_str is replace by value
        kw_expand:  list of keywords that will be padded with spaces
        kw_contract:  list of strings containing spaces for which spaces will be removed, e.g. sub clavicular → subclavicular
        col: string designating column of df to be cleaned
    
    Returns:
        output_str: repaired string
    
    """
    df[col] = df[col].apply(clean_text, args = (sub_dict, kw_expand, kw_contract) )

def label_parser(x, label_dict={}):
    """used to extract feature values from input string x
    
    Args:
        label_dict: dictionary where each entry is of the form 'label':[list of strings that match to label]
    
    Returns:
        string:  detected value of label (assumes each string contains only one label)
        
    Example:
       x = '5 cm superclavicluar'
       label_dict = {'breast':['breast'],
                     'axilla':['axilla'],
                    'supraclavicular':['superclavicular','supraclavicular'],
                     'subclavicular':['subclavicular','subclavcular'] }
                     
      output: 'supraclavicluar'
    
    """
    for k in label_dict.keys():
        labels = label_dict[k]
        if contains_substring(x,labels):
            return k
    return 'unknown'

def find_time_substring(text):
    """uses regex to find and extract first time substring
    
    Args:
        text: string that may contain a time substring of the form HH:MM or HH.MM
    
    Returns: string 'HH:MM' or 'unknown' if time not found
    
    """
    # Regular expression to match time substrings of the form HH:MM or HH.MM
    # does not need to be have blank spaces
    pattern = r'\d{1,2}[:.]\d{2}'
    
    # Find all matches in the input text
    matches = re.findall(pattern, text)
    
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
    
    input_str = input_str.lower()
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
    """extract values for each feature designated in the dictionary and for 
    'clock_pos' and 'nipple_dist'
    
    Args:
        input_str: string from which we want to extract feature values
        labels_dict:  nested dictionary where each feature is a key to a dictionary of matching levels for each label value
    
    Returns:
        output_dict: dict of feature_name:label_value pairs
        
    Example:
        input_str = 'long rt breast 8.00 5cm fn'
        labels_dict = {
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
                                        'oblique':['oblique']} }
                                        
        output_dict = { 'area':'breast',
                        'laterality':'right',
                        'orientation':'long',
                        'clock_pos':'8:00',
                        'nipple_dist':'5 cm' }
    
    """
    
    output_dict = {}
    for feature in labels_dict.keys():
        levels_dict = labels_dict[feature]
        output_dict[feature] = label_parser( input_str, levels_dict)

    output_dict['clock_pos'] = find_time_substring(input_str)
    output_dict['nipple_dist'] = find_cm_substring(input_str)
    
    return output_dict
    
def extract_descript_features_df(df, labels_dict, col = 'description'):
    
    """Apply extract_descript_features to a column in a dataframe and add new columns for each feature to dataframe
    """

    # first extract simple text features
    for feature in labels_dict.keys():
        levels_dict = labels_dict[feature]
        df[feature] = df[col].apply( label_parser, label_dict = levels_dict )
    
    # extract clock_position
    df['clock_pos'] = df[col].apply( find_time_substring )
    
    # extract nipple_dist
    df['nipple_dist'] = df[col].apply( find_cm_substring )
    
    return df
