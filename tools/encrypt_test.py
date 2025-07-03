import pandas as pd
import numpy as np

# Load the CSV files
csv1 = pd.read_csv("D:/DATA/CASBUSI/hash_translation5.csv")
csv2 = pd.read_csv("C:/Users/Tristan/Desktop/raw_radiology (1).csv")

# Convert both accession number columns to string to ensure they match
csv1['OriginalAccessionNumber'] = csv1['OriginalAccessionNumber'].astype(str)
csv2['ACCESSION_NUMBER'] = csv2['ACCESSION_NUMBER'].astype(str)

print(f"CSV1 total records: {len(csv1)}")
print(f"CSV2 total records: {len(csv2)}")

# Filter csv1 to only Exists_In_New = False records
false_records = csv1[csv1['Exists_In_New'] == False].copy()
print(f"Records with Exists_In_New = False: {len(false_records)}")

# Use LEFT JOIN to see which false records have matches in csv2
left_merged = pd.merge(
    false_records, 
    csv2, 
    left_on='OriginalAccessionNumber', 
    right_on='ACCESSION_NUMBER', 
    how='left',
    indicator=True  # This adds a column showing merge results
)

# Count records that don't have matches
no_match = left_merged[left_merged['_merge'] == 'left_only']
has_match = left_merged[left_merged['_merge'] == 'both']

print(f"\n=== ANALYSIS FOR Exists_In_New = False ===")
print(f"Total Exists_In_New = False records: {len(false_records)}")
print(f"Found in radiology CSV: {len(has_match)}")
print(f"NOT found in radiology CSV: {len(no_match)}")

# Calculate percentages
percentage_not_found = (len(no_match) / len(false_records) * 100) if len(false_records) > 0 else 0
percentage_found = (len(has_match) / len(false_records) * 100) if len(false_records) > 0 else 0

print(f"\nPercentage NOT found in radiology CSV: {percentage_not_found:.2f}%")
print(f"Percentage found in radiology CSV: {percentage_found:.2f}%")

# Show some examples of missing accession numbers
print(f"\n=== SAMPLE MISSING ACCESSION NUMBERS ===")
if len(no_match) > 0:
    print("Sample accession numbers that are missing from radiology CSV:")
    missing_accessions = no_match[['OriginalAccessionNumber', 'AnonymizedAccessionNumber']].head(10)
    print(missing_accessions)
else:
    print("All Exists_In_New = False records were found in the radiology CSV!")

# For comparison, let's also check Exists_In_New = True records
true_records = csv1[csv1['Exists_In_New'] == True].copy()
if len(true_records) > 0:
    left_merged_true = pd.merge(
        true_records, 
        csv2, 
        left_on='OriginalAccessionNumber', 
        right_on='ACCESSION_NUMBER', 
        how='left',
        indicator=True
    )
    
    no_match_true = left_merged_true[left_merged_true['_merge'] == 'left_only']
    has_match_true = left_merged_true[left_merged_true['_merge'] == 'both']
    
    percentage_not_found_true = (len(no_match_true) / len(true_records) * 100) if len(true_records) > 0 else 0
    
    print(f"\n=== COMPARISON WITH Exists_In_New = True ===")
    print(f"Total Exists_In_New = True records: {len(true_records)}")
    print(f"NOT found in radiology CSV: {len(no_match_true)} ({percentage_not_found_true:.2f}%)")
    print(f"Found in radiology CSV: {len(has_match_true)} ({100-percentage_not_found_true:.2f}%)")

# Summary table
summary = pd.DataFrame({
    'Exists_In_New': ['False', 'True'],
    'Total_Records': [len(false_records), len(true_records) if len(true_records) > 0 else 0],
    'Not_Found_in_Radiology': [len(no_match), len(no_match_true) if len(true_records) > 0 else 0],
    'Percentage_Not_Found': [percentage_not_found, percentage_not_found_true if len(true_records) > 0 else 0]
})

print(f"\n=== SUMMARY TABLE ===")
print(summary)

# Save the missing accession numbers for further investigation
if len(no_match) > 0:
    missing_accessions_full = no_match[['OriginalAccessionNumber', 'AnonymizedAccessionNumber', 'OriginalPatientID']]
    missing_accessions_full.to_csv("missing_accessions_false.csv", index=False)
    print(f"\nFull list of missing accessions saved to 'missing_accessions_false.csv'")