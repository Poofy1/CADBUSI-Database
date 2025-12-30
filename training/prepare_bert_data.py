"""
Data preparation script for BERT-based ultrasound findings classification.

This script:
1. Loads labeled data from Labelbox
2. Normalizes inconsistent labels
3. Creates label encodings for each feature
4. Splits data into train/val/test sets
5. Saves processed data for model training
"""

import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LABELBOX_DATA_PATH = os.path.join(PROJECT_ROOT, 'labeling', 'labelbox_annotations.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'dataset', 'bert_data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'src', 'ML_processing', 'models')

# Features to classify
FEATURES = ['margin', 'shape', 'orientation', 'echo', 'posterior', 'boundary']

# Label normalization rules
LABEL_NORMALIZATION = {
    'margin': {
        'circumscribed or not circumscribed': 'circumscribed',
    },
    'echo': {
        'complex or combined echogenicity': 'complex',
    },
}


def normalize_label(feature, label):
    """Normalize a label value according to predefined rules."""
    if pd.isna(label):
        return None

    label = str(label).strip()

    # Apply feature-specific normalization
    if feature in LABEL_NORMALIZATION:
        label = LABEL_NORMALIZATION[feature].get(label, label)

    return label if label else None


def create_label_encodings(df, features):
    """
    Create label-to-index mappings for each feature.

    Index 0 is always reserved for None/NaN (not mentioned).
    """
    label_encodings = {}

    for feature in features:
        # Get all unique non-null values
        unique_values = df[feature].dropna().unique().tolist()
        unique_values = sorted(unique_values)  # Sort for consistency

        # Create encoding: 0 = None, 1+ = actual values
        encoding = {None: 0}
        for idx, value in enumerate(unique_values, start=1):
            encoding[value] = idx

        # Create reverse mapping (index to label)
        reverse_encoding = {idx: label for label, idx in encoding.items()}

        label_encodings[feature] = {
            'label_to_idx': encoding,
            'idx_to_label': reverse_encoding,
            'num_classes': len(encoding)
        }

    return label_encodings


def load_and_prepare_data():
    """Load and prepare the labeled data."""
    print("="*80)
    print("BERT DATA PREPARATION")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {LABELBOX_DATA_PATH}")
    df = pd.read_csv(LABELBOX_DATA_PATH)
    print(f"Total rows loaded: {len(df)}")

    # Filter out disqualifiers
    df_valid = df[df['has_disqualifier'] == False].copy()
    print(f"Rows after removing disqualifiers: {len(df_valid)}")

    # Normalize labels
    print("\nNormalizing labels...")
    for feature in FEATURES:
        if feature in df_valid.columns:
            df_valid[feature] = df_valid[feature].apply(lambda x: normalize_label(feature, x))

    # Print label distribution
    print("\n" + "="*80)
    print("LABEL DISTRIBUTION (after normalization)")
    print("="*80)
    for feature in FEATURES:
        print(f"\n{feature.upper()}:")
        counts = df_valid[feature].value_counts(dropna=False)
        print(counts)
        print(f"  Total labeled: {df_valid[feature].notna().sum()}/{len(df_valid)}")

    return df_valid


def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train/val/test sets.

    Note: No stratification due to extremely imbalanced classes.
    """
    print("\n" + "="*80)
    print("DATA SPLITTING")
    print("="*80)

    # First split: train+val vs test (no stratification due to class imbalance)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val size relative to train+val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True
    )

    print(f"Train set: {len(train_df)} examples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val set:   {len(val_df)} examples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set:  {len(test_df)} examples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def save_data(train_df, val_df, test_df, label_encodings):
    """Save processed data and label encodings."""
    print("\n" + "="*80)
    print("SAVING DATA")
    print("="*80)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save CSV files
    train_path = os.path.join(OUTPUT_DIR, 'train.csv')
    val_path = os.path.join(OUTPUT_DIR, 'val.csv')
    test_path = os.path.join(OUTPUT_DIR, 'test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] Saved train data to: {train_path}")
    print(f"[OK] Saved val data to:   {val_path}")
    print(f"[OK] Saved test data to:  {test_path}")

    # Save label encodings
    encodings_path = os.path.join(MODELS_DIR, 'bert_encodings.json')
    with open(encodings_path, 'w') as f:
        # Convert None keys to string "null" for JSON serialization
        encodings_json = {}
        for feature, encoding_dict in label_encodings.items():
            encodings_json[feature] = {
                'label_to_idx': {str(k) if k is None else k: v for k, v in encoding_dict['label_to_idx'].items()},
                'idx_to_label': {str(k): (None if v is None else v) for k, v in encoding_dict['idx_to_label'].items()},
                'num_classes': encoding_dict['num_classes']
            }
        json.dump(encodings_json, f, indent=2)

    print(f"[OK] Saved label encodings to: {encodings_path}")

    # Print encoding summary
    print("\n" + "="*80)
    print("LABEL ENCODINGS SUMMARY")
    print("="*80)
    for feature in FEATURES:
        print(f"\n{feature.upper()}: {label_encodings[feature]['num_classes']} classes")
        for label, idx in sorted(label_encodings[feature]['label_to_idx'].items(), key=lambda x: x[1]):
            label_str = "None (not mentioned)" if label is None else label
            print(f"  {idx}: {label_str}")


def main():
    """Main execution function."""
    # Load and prepare data
    df = load_and_prepare_data()

    # Create label encodings
    print("\n" + "="*80)
    print("CREATING LABEL ENCODINGS")
    print("="*80)
    label_encodings = create_label_encodings(df, FEATURES)

    # Split data
    train_df, val_df, test_df = split_data(df)

    # Save everything
    save_data(train_df, val_df, test_df, label_encodings)

    print("\n" + "="*80)
    print("[OK] DATA PREPARATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review the label encodings in models/label_encodings.json")
    print("  2. Run train_findings_BERT.py to train the model")


if __name__ == "__main__":
    main()
