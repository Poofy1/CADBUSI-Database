import pandas as pd
import numpy as np
import ast

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

def is_nan_list(val):
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], float) and np.isnan(val[0]):
        return True
    return False

def Find_Trust(database_path):
    # Read the CSV file
    csv_file = f'{database_path}/CaseStudyData.csv'
    
    df = pd.read_csv(csv_file)

    # Create a new 'trustworthiness' column
    df['trustworthiness'] = 3  # default to 3

    # Convert string representation of lists to actual lists
    df['Biopsy'] = df['Biopsy'].apply(safe_literal_eval)
    df['Biopsy_Laterality'] = df['Biopsy_Laterality'].apply(safe_literal_eval)

    # Conditions for trustworthiness = 1
    condition1 = ((df['Study_Laterality'].str.lower().isin(["left", "right"])) &
                (df.apply(lambda row: len(row['Biopsy_Laterality']) == 1 and all(i.lower() == row['Study_Laterality'].lower() for i in row['Biopsy_Laterality']), axis=1)) &
                (df['Biopsy'].apply(lambda x: len(x) == 1 if isinstance(x, list) else 0)))

    # Conditions for trustworthiness = 2
    is_duplicate = df.duplicated(subset=['Patient_ID', 'Study_Laterality'], keep=False)

    # Conditions for trustworthiness = 2, including duplicates
    condition2 = ((df['Study_Laterality'].str.lower().isin(["left", "right"])) &
                (df.apply(lambda row: len(row['Biopsy_Laterality']) >= 2 and all(i.lower() == row['Study_Laterality'].lower() for i in row['Biopsy_Laterality']), axis=1)) &
                (df['Biopsy'].apply(lambda x: len(x) >= 2 if isinstance(x, list) else 0))) | is_duplicate
    
    df['is_duplicate'] = df.duplicated(subset=['Patient_ID', 'Study_Laterality'], keep=False)
    # Set trustworthiness to 2 for duplicates
    df.loc[df['is_duplicate'], 'trustworthiness'] = 2
    # Drop the helper column
    df.drop(columns=['is_duplicate'], inplace=True)

    # Conditions for trustworthiness = 3
    condition3 = ((df['BI-RADS'].astype(str).isin(['1', '2'])) & 
                  (df['Biopsy'].apply(lambda x: any('Malignant' in i for i in x) if isinstance(x, list) else False)))

    # Condition for trustworthiness = 3 when biopsy is [nan]
    condition_nan = df['Biopsy'].apply(is_nan_list)

    df.loc[condition1, 'trustworthiness'] = 1
    df.loc[condition2, 'trustworthiness'] = 2
    df.loc[condition3 | condition_nan, 'trustworthiness'] = 3

    # Save the DataFrame back to a CSV file
    df.to_csv(csv_file, index=False)
