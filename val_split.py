import pandas as pd
import os
import numpy as np
env = os.path.dirname(os.path.abspath(__file__))

def PerformVal(val_split):
    database_CSV = f"{env}/database/CaseStudyData.csv"
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(database_CSV)

    # Check if the 'valid' column exists, if not, create it
    if 'valid' not in df.columns:
        df['valid'] = None  # You can replace None with any default value

    # Get unique anonymized_accession_num values
    unique_accession_nums = df['Accession_Number'].unique()

    # Shuffle the unique accession numbers
    np.random.shuffle(unique_accession_nums)

    # Calculate the split index
    twenty_percent_index = int(len(unique_accession_nums) * val_split)

    # Assign '1' to 20% of the unique accession numbers and '0' to the remaining 80%
    valid_accession_nums = set(unique_accession_nums[:twenty_percent_index])

    # Assign the 'valid' column values based on the anonymized_accession_num
    df['valid'] = df['Accession_Number'].apply(lambda x: 1 if x in valid_accession_nums else 0)

    # Save the modified DataFrame to a new CSV file
    df.to_csv(database_CSV, index=False)
    
    print("Updated val split")