import pandas as pd
import pytesseract
from PIL import Image
import tqdm, cv2, json, shutil, os, io, re
import numpy as np
import warnings

# Initialization
warnings.simplefilter(action='ignore', category=FutureWarning)
env = os.path.dirname(os.path.abspath(__file__))

# Static
database_CSV = f"{env}/database/data.csv"


def set_tesseract_path(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


#text extraction function
def ExtractText(folder_path):
    # Initialize an empty list to store the extracted text and filenames
    text_list = []
    files = os.listdir(folder_path)
    
    # Initialize a progress bar with the total number of files
    pbar = tqdm.tqdm(total=len(files))
    
    # Loop through all the files in the folder
    for file_name in files:
        # Check if the file is an image
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Extract the ID from the filename
            id = file_name.split("_")[0].lstrip('0')
            
            # Construct the full path to the image file
            image_path = os.path.join(folder_path, file_name)
            
            # Open the image file and store it in an image object
            img = Image.open(image_path)
            
            # recast image as numpy array
            img = np.array(img)
            
            # if image has multiple channels convert it to grayscale (could probably improve this)
            if len(img.shape)>2:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            # crop to include only bottom of image
            # note in most images the ultrasound part is between 10 and 665 for the first coord (y), and 2 and 853 for the second coord (x)
            img = img[ 500:, 22:833 ]
            
            # convert to black and white to increase contrast
            threshold = 160
            th,img = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
            
            
            # show a dilated image
            #A larger kernel size will result in a more dilated image, which can make it easier to find contours.
            kernel = np.ones((7,7),np.uint8)
            img_dilated = cv2.dilate(img,kernel,iterations=5)
            contours,hierarchy = cv2.findContours(img_dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            c = max(contours,key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)

            # Extract the text from the image using pytesseract
            text = pytesseract.image_to_string(img[y:y+h+1,x:x+w+1])
            text = text.replace("\n"," ")
            
            # If the extracted text is empty, try extracting text from the entire image
            if not text or '¢' in text or len(text.split()) == 1:
                text = pytesseract.image_to_string(img)
                text = text.replace("\n"," ")

            # Append the extracted text and filename to the list
            text_list.append((id, file_name, text[:-1])) # exclude the last character which is usually a newline character

        
        # Update the progress bar after processing each file
        pbar.update(1)
    
    # Close the progress bar
    pbar.close()
    
    # Create a pandas DataFrame from the text list
    df = pd.DataFrame(text_list, columns=['id', 'image_filename', 'text'])
    
    return df



def CleanData(data):
    """
    This function takes a pandas DataFrame column as input, removes noisy data and returns the cleaned text.
    """
    cleaned_data = []
    for text in data:
        text = text.upper()
        if pd.isnull(text):
            # Skip null values
            continue
        
        # Remove newline characters
        text = text.replace('\n', ' ')
        
       # Add space before and after certain words
        text = re.sub(r'(?<![ \t])(TRANS|RADIAL|LONG|RT|LT|ANTI RADIAL|FN)(?![ \t])', r' \1 ', text)
        
        # Remove text after [ or { bracket
        text = re.sub(r'([\[}]).+', r'\1', text)
        text = re.sub(r'(ol|es|on|eer)\b', '', text)
        #text = re.sub(r'1L\d*\.?\d*\b', '', text)
        text = re.sub(r'\d+L.*', '', text)

        
        # Add space after pattern '8:00' or anything in this pattern if not there
        text = re.sub(r'(8:\d\d)(?!\s)', r'\1 ', text)
        
        # Remove noisy data at the end of the text
        text = re.sub(r'\b\d+¢|\b\d+L\b|\b\d+L\s+\d+\s*\b', '', text)
        
        # Extract relevant information
        match = re.search(r'(TRANS|RADIAL|LONG|RT|LT|ANTI RADIAL|OBLIQUE)', text)
        if match:
            text = text[match.start():]
        
        # Remove noisy characters
        cleaned_text = re.sub(r'[^\w\s:-]', '', text)
        
        # Remove extra spaces
        cleaned_text = ' '.join(cleaned_text.split())
       
        # Remove '.' character at the end of the text
        cleaned_text = cleaned_text.rstrip('.')
        
        cleaned_data.append(cleaned_text)
    
    return pd.Series(cleaned_data)


def PopulateFromOCR(df):
    # create new columns in the existing dataframe
    df[['Scanning_Area', 'Location', 'Time', 'Distance_FN']] = pd.DataFrame([['']*4]*len(df))

    # iterate over each row in the dataframe
    for index, row in df.iterrows():
        # get the input string from the 'String' column
        input_string = row['cleaned_text']
        
        # split string by space
        split_string = input_string.split()

        # iterate over each word in the split string
        for i, word in enumerate(split_string):
            if 'TRANS' in word or 'LONG' in word or'RADIAL' in word:
                # check if word contains 'TRANS' or 'LONG' or 'RADIAL'
                if i > 0 and split_string[i-1] == 'ANTI':
                    df.loc[index, 'Scanning_Area'] = 'ANTI RADIAL'
                else:
                    df.loc[index, 'Scanning_Area'] = word
            elif word == 'LT' or word == 'RT' or word == 'RIGHT' or word == 'LEFT':
                # check if word is 'LT' or 'RT'
                df.loc[index, 'Location'] = word
                #resetting distance to empty string to avoid carrying over any potentially incorrect or irrelevant distance values
                df.loc[index, 'Distance_FN'] = ''
            elif re.match(r'\d{1,2}\s*:\s*\d{2}$', word) and df.loc[index, 'Time'] == '':
                # check if word is a time in the format 'HH:MM' and the 'Time' column is still empty
                df.loc[index, 'Time'] = word
            elif re.match(r'\d+\.?\d*\s?CM', word):
                # check if word contains a distance in centimeters
                df.loc[index, 'Distance_FN'] = word

        # use a regular expression to extract the time range
        match = re.search(r'\d+:\d+\s*-\s*\d+:\d+', input_string)
        if match:
            # if a match is found, extract the substring and assign it to the 'Time' column
            df.loc[index, 'Time'] = match.group()
    
    return df
            

def MergeSess(combined_df, merged_df):
    # Merge the current file's DataFrame to the combined DataFrame using 'id' column
    if combined_df.empty:
        return merged_df
    elif 'id' in merged_df.columns:
        return pd.merge(combined_df, merged_df, on='id', how='outer')
    else:
        # Find the new columns in merged_df that are not present in combined_df
        new_columns = [col for col in merged_df.columns if col not in combined_df.columns]
        
        # Add the new columns to combined_df with NaN values
        for col in new_columns:
            combined_df[col] = ''

        updated_rows = []

        # Iterate through unique 'anonymized_accession_num' values in merged_df
        for unique_num in merged_df['anonymized_accession_num'].unique():
            # Filter rows from both dataframes with matching 'anonymized_accession_num'
            combined_rows = combined_df[combined_df['anonymized_accession_num'] == unique_num].copy()
            merged_rows = merged_df[merged_df['anonymized_accession_num'] == unique_num]

            # Merge the filtered rows on 'anonymized_accession_num' and update the new columns
            combined_rows = combined_rows.merge(merged_rows, on='anonymized_accession_num', suffixes=('', '_y'))

            # Copy the new column values to the original columns in combined_rows
            for col in new_columns:
                combined_rows[col] = combined_rows[col + '_y']
                combined_rows.drop(col + '_y', axis=1, inplace=True)

            updated_rows.append(combined_rows)

        # Concatenate updated rows and drop the original rows from combined_df
        updated_rows_df = pd.concat(updated_rows, ignore_index=True)
        combined_df = pd.concat([combined_df, updated_rows_df]).drop_duplicates(subset=['id'], keep='last').reset_index(drop=True)
        return combined_df     

def TransformJSON(data_labels, folder):
    combined_df = pd.DataFrame()
    
    # Iterate through the files and their labels
    for file_name, labels in data_labels.items():

        if not os.path.exists(os.path.join(env, folder, file_name)):
            print(f"File {file_name} not found. Skipping this file.")
            continue
        
        if os.path.splitext(file_name)[1] == ".json":
            
            with open(os.path.join(env, folder, file_name), 'r') as f:
                data = json.load(f)
            
            # Process the JSON data, similar to what you did earlier
            dicom_items = []
            for obj in data:
                for item in obj['dicoms']:
                    new_item = item.copy()

                    for key, value in obj.items():
                        if key != 'dicoms':
                            new_item[key] = value

                    dicom_items.append(new_item)

            dicoms_df = pd.json_normalize(dicom_items)
            regions_df = pd.json_normalize(dicom_items, record_path=["metadata", "SequenceOfUltrasoundRegions"])
            merged_df = pd.merge(dicoms_df, regions_df, left_index=True, right_index=True)
            merged_df = merged_df.rename(columns=lambda x: x.replace("metadata.", ""))
            
        elif os.path.splitext(file_name)[1] == ".csv":
            # Read the CSV file using pandas
            merged_df = pd.read_csv(os.path.join(env, folder, file_name))
        


        # Convert the 'id' columns in both DataFrames to strings
        merged_df = merged_df.astype(str)

        # Filter the DataFrame to keep only the desired columns
        merged_df = merged_df[labels]
        
        combined_df = MergeSess(combined_df, merged_df)
        
        #debug
        #combined_df['id'] = combined_df['id'].astype(int)
        #combined_df = combined_df[combined_df['id'] >= 10]
        

    # Convert the combined DataFrame to CSV and read it back
    csv_string = io.StringIO()
    combined_df.to_csv(csv_string, index=False)
    
    return pd.read_csv(io.StringIO(csv_string.getvalue())).astype(str).replace('nan', '')



def PerformEntry(folder, data_labels, reparse_data, enable_overwritting):
    
    # Check Dirs
    if not os.path.exists(f"{env}/database/"):
        os.makedirs(f"{env}/database/", exist_ok=True)
    if not os.path.exists(f"{env}/database/images/"):
        os.makedirs(f"{env}/database/images/", exist_ok=True)
    if not os.path.exists(f"{env}/raw_data/"):
        os.makedirs(f"{env}/raw_data/", exist_ok=True)
    
    
    print("Transforming JSON/CSV...")
    df = TransformJSON(data_labels, folder)
    
    # Read the existing data.csv file into a DataFrame, if it exists
    if os.path.exists(database_CSV):
        existing_df = pd.read_csv(database_CSV, dtype=str)
    else:
        existing_df = pd.DataFrame(columns=df.columns)
    
    
    # if Stage 1
    new_images = f"{env}/{folder}/images/"
    if os.path.exists(new_images):
        print("Performing OCR...")
        image_df = ExtractText(new_images)
        image_df['cleaned_text'] = CleanData(image_df['text'])
        image_df = PopulateFromOCR(image_df)
        df = pd.merge(image_df, df, on='id', how='inner') # Merge both df with ids and remove incomplete rows
    
        print("Compiling Database...")
        #Copy all images into database
        for filename in os.listdir(new_images):
            source_file = os.path.join(new_images, filename)
            
                
            destination_file = os.path.join(f"{env}/database/images/", filename)
            
            # Only copy the file if it doesn't already exist in the destination folder
            if not os.path.exists(destination_file):
                shutil.copyfile(source_file, destination_file)
        
        
        #cols_to_remove = set(existing_df.columns) - set(df.columns)
        #existing_df = existing_df.drop(columns=cols_to_remove)
        cols_to_add = set(df.columns) - set(existing_df.columns)
        
        # Add the columns to df
        for col in cols_to_add:
            existing_df[col] = ''


        # Append the new dataframe (df) to the existing dataframe (existing_df)
        combined_df = existing_df.append(df, ignore_index=True)

        # Remove duplicate rows
        combined_df = combined_df.drop_duplicates(subset=['id'], keep='last')
        
    else: #Stage 2
        if enable_overwritting:
            combined_df = MergeSess(existing_df, df)
        else:
            combined_df = MergeSess(existing_df, df)
            #combined_df = existing_df.apply(lambda col: existing_df.fillna(df[col.name]), axis=0)
        
        
    # Save the updated DataFrame back to the CSV file
    combined_df = combined_df.astype(str)
    combined_df['id'] = combined_df['id'].astype(int)
    combined_df = combined_df.sort_values(by='id')
        

    combined_df.to_csv(database_CSV, index=False, na_rep='')



    if not reparse_data:
        print("Storing Raw Data...")  
        #Finding index
        entry_index = 0
        while os.path.exists(f"{env}/raw_data/entry_{entry_index}"):
            entry_index += 1
        
        #Move data
        raw_folder = f"{env}/raw_data/entry_{entry_index}"

        shutil.copytree(os.path.join(env, "downloads"), raw_folder)

        # Remove the "downloads" folder and its contents
        shutil.rmtree(os.path.join(env, "downloads"))

        # Recreate the "downloads" folder as an empty folder
        os.mkdir(os.path.join(env, "downloads"))


    print("Entry Complete")