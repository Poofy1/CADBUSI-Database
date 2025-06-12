import pandas as pd

def compare_csvs(csv1_path, csv2_path):
    """
    Compare PATIENT_ID from csv1 with OriginalPatientID from csv2
    """
    
    # Read both CSV files
    try:
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Check if required columns exist
    if 'PATIENT_ID' not in df1.columns:
        print("Error: 'PATIENT_ID' column not found in CSV1")
        return
    
    if ' OriginalPatientID' not in df2.columns:
        print("Error: 'OriginalPatientID' column not found in CSV2")
        return
    
    # Get the relevant columns and remove any NaN values
    accession_numbers = set(df1['PATIENT_ID'].dropna().astype(str))
    original_patient_ids = df2[' OriginalPatientID'].dropna().astype(str)
    
    # Find matches
    matches = []
    for patient_id in original_patient_ids:
        if patient_id in accession_numbers:
            matches.append(patient_id)
    
    # Print matching ones
    print("MATCHING IDs:")
    print("-" * 40)
    if matches:
        for match in matches:
            print(match)
    else:
        print("No matches found")
    
    print("\n" + "=" * 50)
    
    # Calculate percentage
    total_patient_ids = len(original_patient_ids)
    num_matches = len(matches)
    
    if total_patient_ids > 0:
        percentage = (num_matches / total_patient_ids) * 100
        print(f"RESULTS:")
        print(f"Total OriginalPatientIDs in CSV2: {total_patient_ids}")
        print(f"Number of matches found: {num_matches}")
        print(f"Percentage of matches: {percentage:.2f}%")
    else:
        print("No OriginalPatientIDs found in CSV2")

# Usage
if __name__ == "__main__":
    # Replace with your actual file paths
    csv1_file = "C:/Users/Tristan/Desktop/endpoint_data.csv"  # File with ACCESSION_NUMBER column
    csv2_file = "D:/DATA/CASBUSI/backups/maps/master_anon_map.csv"  # File with OriginalPatientID column
    
    compare_csvs(csv1_file, csv2_file)