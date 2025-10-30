import labelbox
import pandas as pd
import os
import sys

env = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from src.data_ingest.findings_parser import add_ultrasound_classifications


# Mapping from Labelbox classification names to parser feature names
LABELBOX_TO_PARSER_MAPPING = {
    'Echo pattern': 'echo',
    'Margin': 'margin',
    'Shape': 'shape',
    'Orientation': 'orientation',
    'Posterior features': 'posterior',
    'Posterior': 'posterior',
    'Boundary zone': 'boundary',
    'Boundary': 'boundary',
}


def Read_Labelbox_Data(LB_API_KEY, PROJECT_ID):
    print("(Newly created data in labelbox will take time to update!)")
    
    client = labelbox.Client(api_key=LB_API_KEY)
    project = client.get_project(PROJECT_ID)
    
    print("Contacting Labelbox")
    
    export_params = {
        "data_row_details": True,
        "label_details": True,
        "project_details": True
    }
    
    export_task = project.export(params=export_params)
    export_task.wait_till_done()
    
    print("Getting export data...")
    export_data = []
    
    for data_row in export_task.get_buffered_stream():
        export_data.append(data_row.json)
    
    print(f"Retrieved {len(export_data)} data rows")
    
    csv_data = parse_labelbox_annotations(export_data)
    csv_data.to_csv(f'{env}/labelbox_annotations.csv', index=False)
    print(f"Annotations saved to {env}/labelbox_annotations.csv")
    
    return csv_data


def parse_labelbox_annotations(export_data):
    print("Parsing annotations...")
    
    parsed_rows = []
    
    for item in export_data:
        data_row = item.get('data_row', {})
        row_id = data_row.get('id', '')
        global_key = data_row.get('global_key', '')
        text = data_row.get('row_data', '')
        
        projects = item.get('projects', {})
        
        row = {
            'id': row_id,
            'global_key': global_key,
            'FINDINGS': text,
            # Initialize parser columns
            'margin': None,
            'shape': None,
            'orientation': None,
            'echo': None,
            'posterior': None,
            'boundary': None,
            'has_disqualifier': False,  # Track if this row should be skipped
        }
        
        all_classifications = []
        
        for project_id, project_data in projects.items():
            labels = project_data.get('labels', [])
            
            for label in labels:
                annotations = label.get('annotations', {})
                classifications = annotations.get('classifications', [])
                
                for classification in classifications:
                    feature_name = classification.get('name', '')
                    checklist_answers = classification.get('checklist_answers', [])
                    
                    # Check for disqualifier
                    if 'Disqualifiers' in feature_name or 'Skip Labeling' in feature_name:
                        row['has_disqualifier'] = True
                    
                    for answer in checklist_answers:
                        answer_name = answer.get('name', '')
                        answer_value = answer.get('value', '')
                        
                        all_classifications.append({
                            'feature': feature_name,
                            'value': answer_name or answer_value
                        })
        
        if all_classifications:
            # Map Labelbox classification names to parser column names
            for classification in all_classifications:
                labelbox_feature = classification['feature']
                value = classification['value']
                
                # Map to parser column name
                parser_column = LABELBOX_TO_PARSER_MAPPING.get(labelbox_feature)
                
                if parser_column:
                    row[parser_column] = value
                else:
                    # If no mapping found, use original name (might be disqualifier)
                    if 'Disqualifiers' not in labelbox_feature and 'Skip Labeling' not in labelbox_feature:
                        print(f"Warning: No mapping found for Labelbox feature '{labelbox_feature}'")
            
            parsed_rows.append(row)
    
    df = pd.DataFrame(parsed_rows)
    print(f"Parsed {len(df)} rows with classifications")
    
    # Filter out disqualifiers
    disqualified_count = df['has_disqualifier'].sum()
    print(f"Found {disqualified_count} rows with disqualifiers - will exclude from accuracy")
    
    # Show which columns we have
    parser_columns = ['margin', 'shape', 'orientation', 'echo', 'posterior', 'boundary']
    print("\nColumn check:")
    for col in parser_columns:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"  ✓ {col}: {non_null} non-null values")
        else:
            print(f"  ✗ {col}: MISSING")
    
    return df


def prepare_for_parser(labelbox_df):
    """Add required columns for the parser"""
    df = labelbox_df.copy()
    
    # Add required columns (since all data is ultrasound)
    df['MODALITY'] = 'US'
    df['Study_Laterality'] = 'RIGHT'  # Default, adjust if you have laterality info
    df['Study_ID'] = df['global_key']  # Use global_key as Study_ID
    
    return df


def combine_true_and_parsed_labels(labelbox_df, parsed_df):
    """Combine true labels from Labelbox with parsed labels"""
    
    parser_features = ['margin', 'shape', 'orientation', 'echo', 'posterior', 'boundary']
    
    # Create a copy of parsed_df
    combined_df = parsed_df.copy()
    
    # Add true labels with prefix
    for feature in parser_features:
        if feature in labelbox_df.columns:
            # Create a mapping from global_key to true label value
            true_label_map = labelbox_df.set_index('global_key')[feature].to_dict()
            
            # Add true label column
            combined_df[f'true_{feature}'] = combined_df['Study_ID'].map(true_label_map)
    
    # Also add the has_disqualifier flag
    if 'has_disqualifier' in labelbox_df.columns:
        disqualifier_map = labelbox_df.set_index('global_key')['has_disqualifier'].to_dict()
        combined_df['has_disqualifier'] = combined_df['Study_ID'].map(disqualifier_map)
    
    # Reorder columns to put true_ columns next to their parsed counterparts
    ordered_columns = []
    
    # First add metadata columns
    metadata_cols = ['id', 'global_key', 'Study_ID', 'FINDINGS', 'MODALITY', 'Study_Laterality']
    for col in metadata_cols:
        if col in combined_df.columns:
            ordered_columns.append(col)
    
    # Then add feature pairs (true_feature, feature)
    for feature in parser_features:
        if f'true_{feature}' in combined_df.columns:
            ordered_columns.append(f'true_{feature}')
        if feature in combined_df.columns:
            ordered_columns.append(feature)
    
    # Add has_disqualifier if it exists
    if 'has_disqualifier' in combined_df.columns:
        ordered_columns.append('has_disqualifier')
    
    # Add any remaining columns
    for col in combined_df.columns:
        if col not in ordered_columns:
            ordered_columns.append(col)
    
    combined_df = combined_df[ordered_columns]
    
    return combined_df


def calculate_accuracy(labelbox_df, parsed_df):
    """Calculate accuracy metrics between Labelbox labels and parser output"""
    
    # DON'T filter out disqualifiers - use all rows
    valid_rows = labelbox_df.copy()
    print(f"\nCalculating accuracy on {len(valid_rows)} rows")
    
    # Get the corresponding parsed rows
    valid_parsed = parsed_df[parsed_df['Study_ID'].isin(valid_rows['global_key'])].copy()
    
    parser_features = ['margin', 'shape', 'orientation', 'echo', 'posterior', 'boundary']
    
    # Per-feature accuracy results
    feature_results = {}
    
    # For per-row accuracy calculation
    row_match_data = []
    
    for idx in valid_rows.index:
        row_id = valid_rows.loc[idx, 'id']
        global_key = valid_rows.loc[idx, 'global_key']
        
        # Find corresponding parsed row
        parsed_row = valid_parsed[valid_parsed['Study_ID'] == global_key]
        
        if len(parsed_row) == 0:
            continue
        
        parsed_row = parsed_row.iloc[0]
        
        # Check if ALL features match for this row
        all_features_match = True
        feature_matches = {}
        
        # Check if ground truth has any labels (at least one non-null feature)
        ground_truth_values = [valid_rows.loc[idx, f] for f in parser_features]
        has_any_label = any(pd.notna(v) for v in ground_truth_values)
        all_blank = all(pd.isna(v) for v in ground_truth_values)
        
        for feature in parser_features:
            lb_value = valid_rows.loc[idx, feature]
            parser_value = parsed_row[feature]
            
            # Compare values (both None counts as a match)
            matches = (pd.isna(lb_value) and pd.isna(parser_value)) or (lb_value == parser_value)
            feature_matches[feature] = matches
            
            if not matches:
                all_features_match = False
        
        row_match_data.append({
            'id': row_id,
            'global_key': global_key,
            'all_match': all_features_match,
            'has_any_label': has_any_label,
            'all_blank': all_blank,
            **feature_matches
        })
    
    # Calculate per-feature accuracy
    for feature in parser_features:
        if feature not in valid_rows.columns:
            print(f"Warning: Feature '{feature}' not found in Labelbox data")
            continue
        
        if feature not in valid_parsed.columns:
            print(f"Warning: Feature '{feature}' not found in parser output")
            continue
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'labelbox': valid_rows[feature].values,
            'parser': valid_parsed[feature].values,
            'id': valid_rows['id'].values,
            'findings': valid_rows['FINDINGS'].values
        })
        
        # Calculate metrics
        total = len(comparison_df)
        
        # Both None OR both have same value = match
        matches = ((comparison_df['labelbox'].isna() & comparison_df['parser'].isna()) | 
                   (comparison_df['labelbox'] == comparison_df['parser'])).sum()
        
        true_positives = ((comparison_df['labelbox'].notna()) & 
                         (comparison_df['parser'].notna()) & 
                         (comparison_df['labelbox'] == comparison_df['parser'])).sum()
        
        false_positives = ((comparison_df['labelbox'].isna()) & 
                          (comparison_df['parser'].notna())).sum()
        
        false_negatives = ((comparison_df['labelbox'].notna()) & 
                          (comparison_df['parser'].isna())).sum()
        
        true_negatives = ((comparison_df['labelbox'].isna()) & 
                         (comparison_df['parser'].isna())).sum()
        
        accuracy = matches / total if total > 0 else 0
        
        # NEW METRIC: Accuracy only when true label exists (non-null)
        labeled_only = comparison_df[comparison_df['labelbox'].notna()]
        labeled_total = len(labeled_only)
        labeled_matches = (labeled_only['labelbox'] == labeled_only['parser']).sum()
        labeled_accuracy = labeled_matches / labeled_total if labeled_total > 0 else None
        
        mismatch_mask = ~((comparison_df['labelbox'].isna() & comparison_df['parser'].isna()) | 
                         (comparison_df['labelbox'] == comparison_df['parser']))
        mismatches = comparison_df[mismatch_mask]
        
        feature_results[feature] = {
            'total_comparisons': total,
            'matches': matches,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'labeled_only_total': labeled_total,
            'labeled_only_matches': labeled_matches,
            'labeled_only_accuracy': labeled_accuracy,
            'mismatches': mismatches
        }
    
    # Calculate per-row accuracy with breakdown
    row_df = pd.DataFrame(row_match_data)
    
    # Overall row accuracy
    total_rows = len(row_df)
    exact_matches = row_df['all_match'].sum()
    per_row_accuracy = exact_matches / total_rows if total_rows > 0 else 0
    
    # Rows where ALL columns were blank (ground truth)
    blank_rows = row_df[row_df['all_blank']]
    blank_total = len(blank_rows)
    blank_matches = blank_rows['all_match'].sum()
    blank_accuracy = blank_matches / blank_total if blank_total > 0 else 0
    
    # Rows where AT LEAST ONE column had a label (ground truth)
    labeled_rows = row_df[row_df['has_any_label']]
    labeled_total = len(labeled_rows)
    labeled_matches = labeled_rows['all_match'].sum()
    labeled_accuracy = labeled_matches / labeled_total if labeled_total > 0 else 0
    
    return {
        'per_feature': feature_results,
        'per_row': {
            'total_rows': total_rows,
            'exact_matches': exact_matches,
            'accuracy': per_row_accuracy,
            'blank_rows_total': blank_total,
            'blank_rows_matches': blank_matches,
            'blank_rows_accuracy': blank_accuracy,
            'labeled_rows_total': labeled_total,
            'labeled_rows_matches': labeled_matches,
            'labeled_rows_accuracy': labeled_accuracy,
            'row_details': row_df
        }
    }


def print_accuracy_report(results):
    """Print a simplified accuracy report"""
    print("\n" + "="*80)
    print("PARSER ACCURACY REPORT")
    print("="*80)
    
    # Per-feature accuracy (traditional - includes null/null as matches)
    print("\nPER-FEATURE ACCURACY (includes null/null as matches):")
    print("-" * 40)
    for feature, metrics in results['per_feature'].items():
        print(f"{feature:15s}: {metrics['accuracy']:.2%} ({metrics['matches']}/{metrics['total_comparisons']})")
    
    # Overall per-feature accuracy
    total_comparisons = sum(m['total_comparisons'] for m in results['per_feature'].values())
    total_matches = sum(m['matches'] for m in results['per_feature'].values())
    overall_feature_accuracy = total_matches / total_comparisons if total_comparisons > 0 else 0
    
    print("-" * 40)
    print(f"{'OVERALL':15s}: {overall_feature_accuracy:.2%} ({total_matches}/{total_comparisons})")
    
    # NEW METRIC: Accuracy only when true label exists
    print("\n" + "="*80)
    print("PER-FEATURE ACCURACY (only when true label exists):")
    print("-" * 40)
    
    total_labeled = 0
    total_labeled_matches = 0
    
    for feature, metrics in results['per_feature'].items():
        if metrics['labeled_only_accuracy'] is not None:
            print(f"{feature:15s}: {metrics['labeled_only_accuracy']:.2%} ({metrics['labeled_only_matches']}/{metrics['labeled_only_total']})")
            total_labeled += metrics['labeled_only_total']
            total_labeled_matches += metrics['labeled_only_matches']
        else:
            print(f"{feature:15s}: N/A (no labels)")
    
    # Overall accuracy for labeled data
    overall_labeled_accuracy = total_labeled_matches / total_labeled if total_labeled > 0 else 0
    print("-" * 40)
    print(f"{'OVERALL':15s}: {overall_labeled_accuracy:.2%} ({total_labeled_matches}/{total_labeled})")
    
    # Per-row accuracy
    print("\n" + "="*80)
    print("PER-ROW ACCURACY:")
    print("-" * 40)
    per_row = results['per_row']
    
    print(f"Overall (all features match): {per_row['accuracy']:.2%} ({per_row['exact_matches']}/{per_row['total_rows']})")
    print()
    print(f"Blank rows (no labels):       {per_row['blank_rows_accuracy']:.2%} ({per_row['blank_rows_matches']}/{per_row['blank_rows_total']})")
    print(f"Labeled rows (≥1 label):      {per_row['labeled_rows_accuracy']:.2%} ({per_row['labeled_rows_matches']}/{per_row['labeled_rows_total']})")
    print("="*80)


if __name__ == "__main__":
    # Load Labelbox data
    print("Loading Labelbox data...")
    labelbox_df = Read_Labelbox_Data(CONFIG['LABELBOX_API_KEY'], 'cmgb7l0mv0e6q07vu8z7w4pro')
    
    print("\nLabelbox columns:", labelbox_df.columns.tolist())
    
    # Prepare data for parser
    print("\nPreparing data for parser...")
    prepared_df = prepare_for_parser(labelbox_df)
    
    # Run parser
    print("\nRunning parser...")
    parsed_df = add_ultrasound_classifications(prepared_df, f'{env}/parser_output')
    
    print("\nParser output columns:", parsed_df.columns.tolist())
    
    # Combine true and parsed labels
    print("\nCombining true and parsed labels...")
    combined_df = combine_true_and_parsed_labels(labelbox_df, parsed_df)
    
    # Calculate accuracy
    print("\nCalculating accuracy...")
    accuracy_results = calculate_accuracy(labelbox_df, parsed_df)
    
    # Print report
    print_accuracy_report(accuracy_results)
    
    # Save detailed results with both true and parsed labels
    combined_df.to_csv(f'{env}/parser_output.csv', index=False)
    print(f"\nParsed output with true labels saved to {env}/parser_output.csv")