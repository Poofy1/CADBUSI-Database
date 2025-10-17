import pandas as pd
import json
import os
import re
env = os.path.dirname(os.path.abspath(__file__))

def anonymize_dates_times_and_names(text):
    """
    Remove dates, times, and names from text.
    """
    if pd.isna(text):
        return text
    
    # Convert to string if not already
    text = str(text)
    
    # Remove everything after capitalized name + initial pattern (e.g., "John S.")
    text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z]\..*', '[NAME AND SUBSEQUENT TEXT REMOVED]', text, flags=re.DOTALL)
    
    # Remove everything after "Dictated on" (including the phrase itself)
    text = re.sub(r'Dictated (?:on|by).*', '[DICTATION SECTION REMOVED]', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove everything after "Interpreted by" (including the phrase itself)
    text = re.sub(r'Interpreted by.*', '[INTERPRETATION SECTION REMOVED]', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove everything after "Transcribed by" (including the phrase itself)
    text = re.sub(r'Transcribed by.*', '[TRANSCRIPTION SECTION REMOVED]', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove everything after "Signed by" (including the phrase itself)
    text = re.sub(r'Signed by.*', '[SIGNATURE SECTION REMOVED]', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove everything after "Assisted by" (including the phrase itself)
    text = re.sub(r'Assisted by.*', '[ASSISTANT INFO REMOVED]', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove everything after "Electronically signed by:" (original pattern)
    text = re.sub(r'Electronically signed by.*', '[SIGNATURE SECTION REMOVED]', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove everything after "electronically signed" (without requiring "by")
    text = re.sub(r'electronically signed.*', '[SIGNATURE SECTION REMOVED]', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove everything after "Imaging Technologist:" (including the phrase itself)
    text = re.sub(r'Imaging Technologist:.*', '[TECHNOLOGIST INFO REMOVED]', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove M.D. (and variations) along with 10 characters before it
    text = re.sub(r'.{10}M\.D\.', '[NAME REMOVED]', text)
    text = re.sub(r'.{10}M\.\s?D\.', '[NAME REMOVED]', text)  # Handles "M. D." with optional space
    
    # Remove full names with middle initial and credentials
    # Pattern: FirstName MiddleInitial. LastName M.D. or MD
    text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\s+(?:M\.D\.|MD|M\. D\.|D\.O\.|DO|Ph\.D\.|PhD)\b', '[NAME REMOVED]', text)
    
    # Pattern: FirstName LastName M.D. or MD
    text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:M\.D\.|MD|M\. D\.|D\.O\.|DO|Ph\.D\.|PhD)\b', '[NAME REMOVED]', text)
    
    # Pattern: LastName, FirstName MD
    text = re.sub(r'\b[A-Z][a-z]+,\s+[A-Z][a-z]+\s+(?:M\.D\.|MD|M\. D\.|D\.O\.|DO|Ph\.D\.|PhD)\b', '[NAME REMOVED]', text)
    
    # Pattern: Initials + Last Name + Credentials
    text = re.sub(r'\b[A-Z]\.\s?[A-Z]\.\s+[A-Z][a-z]+,?\s+(?:MD|DO|PA|NP|RN|DDS|DMD|PhD|MBBS|M\.D\.|D\.O\.)\b', '[NAME REMOVED]', text)
    
    # Pattern: Full name followed by credentials (e.g., John Smith, MD)
    text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+,?\s+(?:MD|DO|PA|NP|RN|DDS|DMD|PhD|MBBS|M\.D\.|D\.O\.)\b', '[NAME REMOVED]', text)
    
    # Pattern: Dr. [Name]
    text = re.sub(r'\bDr\.\s+[A-Z][A-Za-z.\s]+\b', '[NAME REMOVED]', text)
    
    # Pattern: Single initial + Last Name + Credentials (e.g., J. Smith, MD)
    text = re.sub(r'\b[A-Z]\.\s+[A-Z][a-z]+,?\s+(?:MD|DO|PA|NP|RN|DDS|DMD|PhD|MBBS|M\.D\.|D\.O\.)\b', '[NAME REMOVED]', text)
    
    # Remove times (before dates to avoid conflicts)
    # Pattern: HH:MM:SS (e.g., 01:01:01)
    text = re.sub(r'\b\d{1,2}:\d{2}:\d{2}\b', '[TIME REMOVED]', text)
    
    # Pattern: HH:MM (e.g., 01:01 or 1:32PM)
    text = re.sub(r'\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b', '[TIME REMOVED]', text, flags=re.IGNORECASE)
    
    # Remove dates
    # Pattern: Month DD (e.g., "Apr 20", "January 15")
    text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b', '[DATE REMOVED]', text, flags=re.IGNORECASE)
    
    # Pattern: MM/DD/YY or MM/DD/YYYY (e.g., 01/01/01 or 01/01/0001)
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[DATE REMOVED]', text)
    
    # Pattern: Standalone year (4 digits)
    text = re.sub(r'\b\d{4}\b', '[YEAR REMOVED]', text)
    
    return text

# Configuration
SAMPLE_SIZE = 10000  # Default sample size, set to None to use all data

# Read the CSV file
print("Reading CSV file...")
df = pd.read_csv(f'{env}/parsed_radiology.csv')

# Filter for rows that have data in FINDINGS column AND exclude BILATERAL laterality
print("Applying filters...")
df_filtered = df[
    (df['FINDINGS'].notna()) & 
    (df['FINDINGS'].str.strip() != '') &
    (df['Study_Laterality'] != 'BILATERAL') &
    (df['MODALITY'] == 'US') &
    (df['is_biopsy'] != 'T') & 
    (df['BI-RADS'].notna()) &
    (df['BI-RADS'].astype(str).str.strip() != '')
]

print(f"Total rows: {len(df)}")
print(f"Rows after filtering: {len(df_filtered)}")

# Random sampling if needed
if SAMPLE_SIZE is not None and len(df_filtered) > SAMPLE_SIZE:
    print(f"Randomly sampling {SAMPLE_SIZE} rows...")
    df_filtered = df_filtered.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Sampled rows: {len(df_filtered)}")
elif SAMPLE_SIZE is not None:
    print(f"Dataset has {len(df_filtered)} rows, which is less than sample size {SAMPLE_SIZE}. Using all rows.")

# Anonymize the FINDINGS column
print("\nAnonymizing dates, times, and names...")
df_filtered['FINDINGS_ANON'] = df_filtered['FINDINGS'].apply(anonymize_dates_times_and_names)

# Debug: Save anonymized CSV
debug_csv_filename = 'anonymized_findings_debug.csv'
df_filtered.to_csv(debug_csv_filename, index=False)
print(f"Debug CSV saved: {debug_csv_filename}")

# Create Labelbox format from anonymized data
print("\n=== CREATING LABELBOX JSON ===")
labelbox_data = []

for index, row in df_filtered.iterrows():
    global_key = f"radiology_{index}"
    
    labelbox_data.append({
        "row_data": row['FINDINGS_ANON'],
        "global_key": global_key,
        "media_type": "TEXT"
    })

# Save to JSON file
output_filename = 'labelbox_import.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(labelbox_data, f, indent=2, ensure_ascii=False)

print(f"\nJSON file created: {output_filename}")
print(f"Total entries: {len(labelbox_data)}")

# Preview first item
if labelbox_data:
    print("\n=== PREVIEW OF FIRST LABELBOX ENTRY ===")
    print(json.dumps(labelbox_data[0], indent=2))

print("\n✓ Complete! Files created:")
print(f"  1. {debug_csv_filename} (for review)")
print(f"  2. {output_filename} (for Labelbox upload)")