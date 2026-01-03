"""
Data preparation script for T5-based structured extraction from ultrasound findings.

This script:
1. Loads data from CSV
2. Filters rows with valid ultrasound findings
3. Creates structured output format: [clock_position, distance, size]
4. Splits data into train/val sets
5. Saves processed data for T5 training
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_DATA_PATH = r"F:\CODE\CADBUSI\CADBUSI-Database\training\dataset\project-2-at-2026-01-01-19-12-e4721624.csv"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'dataset', 't5_data')

# Columns
TEXT_INPUT_COL = 'ultrasound_findings'  # Input text
TARGET_COL = 'tumor_size'  # Output: structured measurements like "[9:00, 70mm, 24mm]"


def load_and_prepare_data():
    """Load and prepare the data."""
    print("="*80)
    print("T5 DATA PREPARATION - Structured Extraction")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {INPUT_DATA_PATH}")
    df = pd.read_csv(INPUT_DATA_PATH)
    print(f"Total rows loaded: {len(df)}")

    # Check if required columns exist
    required_cols = [TEXT_INPUT_COL, TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"\n[WARNING] Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        print("\nPlease ensure your data has these columns:")
        print(f"  - {TEXT_INPUT_COL}: Input ultrasound findings text")
        print(f"  - {TARGET_COL}: Structured output (e.g., '[9:00, 70mm, 24mm]')")
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter to rows with valid ultrasound findings
    df_valid = df[df[TEXT_INPUT_COL].notna()].copy()
    print(f"Rows with valid ultrasound findings: {len(df_valid)}")

    # Clean up text column (strip whitespace)
    df_valid[TEXT_INPUT_COL] = df_valid[TEXT_INPUT_COL].astype(str).str.strip()

    # Remove empty strings
    df_valid = df_valid[df_valid[TEXT_INPUT_COL] != ''].copy()
    print(f"Rows after removing empty findings: {len(df_valid)}")

    # Rename tumor_size to structured_output for consistency with training script
    df_valid = df_valid.rename(columns={TARGET_COL: 'structured_output'})

    # Convert NaN/empty targets to empty string (keep them for training)
    df_valid['structured_output'] = df_valid['structured_output'].fillna('')

    # Print statistics
    populated = (df_valid['structured_output'] != '').sum()
    empty = (df_valid['structured_output'] == '').sum()
    print(f"Rows with populated targets: {populated}")
    print(f"Rows with empty targets: {empty}")
    print(f"Total rows: {len(df_valid)}")

    # Print distribution statistics
    print("\n" + "="*80)
    print("STRUCTURED OUTPUT STATISTICS")
    print("="*80)

    # Show sample outputs
    print("\nSample structured outputs:")
    for i, output in enumerate(df_valid['structured_output'].head(10), 1):
        print(f"  {i}. {output}")

    return df_valid


def split_data(df, val_size=0.2, random_state=42):
    """
    Split data into train/val sets.
    """
    print("\n" + "="*80)
    print("DATA SPLITTING")
    print("="*80)

    # Split: train vs val
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        shuffle=True
    )

    print(f"Train set: {len(train_df)} examples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val set:   {len(val_df)} examples ({len(val_df)/len(df)*100:.1f}%)")

    return train_df, val_df


def save_data(train_df, val_df):
    """Save processed data."""
    print("\n" + "="*80)
    print("SAVING DATA")
    print("="*80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save CSV files (only keep the two columns we need)
    columns_to_save = [TEXT_INPUT_COL, 'structured_output']

    train_path = os.path.join(OUTPUT_DIR, 'train.csv')
    val_path = os.path.join(OUTPUT_DIR, 'val.csv')

    train_df[columns_to_save].to_csv(train_path, index=False)
    val_df[columns_to_save].to_csv(val_path, index=False)

    print(f"[OK] Saved train data to: {train_path}")
    print(f"[OK] Saved val data to:   {val_path}")

    # Print some example data
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)
    print("\nFirst 3 training examples:")
    for _, row in train_df.head(3).iterrows():
        findings_preview = row[TEXT_INPUT_COL][:100] + "..." if len(row[TEXT_INPUT_COL]) > 100 else row[TEXT_INPUT_COL]
        print(f"\n  Input:  {findings_preview}")
        print(f"  Output: {row['structured_output']}")


def main():
    """Main execution function."""
    # Load and prepare data
    df = load_and_prepare_data()

    # Split data
    train_df, val_df = split_data(df)

    # Save everything
    save_data(train_df, val_df)
if __name__ == "__main__":
    main()
