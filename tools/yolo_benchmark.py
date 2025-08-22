import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
env = os.path.dirname(os.path.abspath(__file__))

# Load the CSV files
breast_data = pd.read_csv(r"C:\Users\Tristan\Desktop\Databases_database_2025_8_11_main_BreastData.csv")
image_data = pd.read_csv(r"C:\Users\Tristan\Desktop\Databases_database_2025_8_11_main_ImageData.csv")

# Convert Accession_Number to string in both datasets
breast_data['Accession_Number'] = breast_data['Accession_Number'].astype(str)
image_data['Accession_Number'] = image_data['Accession_Number'].astype(str)

# Function to count lesions from caliper_boxes column
def count_lesions_from_boxes(box_string):
    if pd.isna(box_string) or box_string == '[]' or box_string.strip() == '':
        return 0
    
    box_string = str(box_string).strip()
    
    if box_string == '[]':
        return 0
    
    if ';' in box_string:
        return len(box_string.split(';'))
    elif box_string.startswith('[') and box_string.endswith(']') and box_string != '[]':
        return 1
    else:
        return 0

# Count lesions per row in image_data
image_data['lesion_count'] = image_data['caliper_boxes'].apply(count_lesions_from_boxes)

# Count total lesions per accession number
lesion_counts = image_data.groupby('Accession_Number')['lesion_count'].sum().reset_index()
lesion_counts.columns = ['Accession_Number', 'total_lesion_count']

# Create capped version (limit to 20, group >20)
lesion_counts['lesion_count_capped'] = lesion_counts['total_lesion_count'].apply(
    lambda x: '>20' if x > 20 else str(x)
)

# Use LEFT JOIN to keep all breast data records
merged_data = pd.merge(breast_data, lesion_counts, on='Accession_Number', how='left')

# Fill missing lesion counts with 0 (cases that don't have image data)
merged_data['total_lesion_count'] = merged_data['total_lesion_count'].fillna(0)
merged_data['lesion_count_capped'] = merged_data['lesion_count_capped'].fillna('0')

# Prepare data for analysis
malignant_zero_lesions = merged_data[
    (merged_data['final_interpretation'] == 'MALIGNANT') & 
    (merged_data['total_lesion_count'] == 0)
]

all_malignant = merged_data[merged_data['final_interpretation'] == 'MALIGNANT']

print("="*80)
print("MALIGNANT CASES WITH 0 LESIONS DETECTED BY YOLO")
print("="*80)
print(f"Total malignant cases with 0 lesions: {len(malignant_zero_lesions)}")
print(f"Total malignant cases: {len(all_malignant)}")
print(f"Percentage: {len(malignant_zero_lesions)/len(all_malignant)*100:.1f}%")
print("\n" + "="*80)
print("SAMPLE OF 10 MALIGNANT CASES WITH 0 LESIONS:")
print("="*80)

# Select relevant columns to display
display_columns = ['Accession_Number', 'BI-RADS', 'final_interpretation', 'total_lesion_count']

# Add any other interesting columns if they exist
optional_columns = ['patient_age', 'laterality', 'study_date', 'modality', 'breast_density']
for col in optional_columns:
    if col in malignant_zero_lesions.columns:
        display_columns.append(col)

# Get first 10 cases
sample_cases = malignant_zero_lesions.head(10)[display_columns]

# Display with nice formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)

for idx, (_, case) in enumerate(sample_cases.iterrows(), 1):
    print(f"\nCase {idx}:")
    print("-" * 40)
    for col, value in case.items():
        if pd.notna(value):
            print(f"  {col:20}: {value}")
        else:
            print(f"  {col:20}: N/A")

print("\n" + "="*80)

# Reset display options
pd.reset_option('display.max_columns')
pd.reset_option('display.width')
pd.reset_option('display.max_colwidth')

# Calculate percentage distributions for each interpretation (for first graph)
def calculate_percentages(interpretation):
    subset = merged_data[merged_data['final_interpretation'] == interpretation]
    if len(subset) == 0:
        return pd.Series(dtype=float)
    
    counts = subset['lesion_count_capped'].value_counts()
    
    result = {}
    for i in range(21):  # 0 to 20
        result[str(i)] = counts.get(str(i), 0)
    result['>20'] = counts.get('>20', 0)
    
    total = len(subset)
    percentages = {k: (v / total * 100) for k, v in result.items() if v > 0 or k in ['0', '>20']}
    
    return percentages

# Prepare data for BI-RADS analysis (for second graph)
birads_analysis = []
for birads in sorted(all_malignant['BI-RADS'].dropna().unique()):
    total_malignant_birads = len(all_malignant[all_malignant['BI-RADS'] == birads])
    zero_lesions_birads = len(malignant_zero_lesions[malignant_zero_lesions['BI-RADS'] == birads])
    
    if total_malignant_birads > 0:
        percentage = (zero_lesions_birads / total_malignant_birads) * 100
    else:
        percentage = 0
    
    birads_analysis.append({
        'BI-RADS': birads,
        'Total_Malignant': total_malignant_birads,
        'Zero_Lesions': zero_lesions_birads,
        'Percentage': percentage
    })

birads_df = pd.DataFrame(birads_analysis)

# Create first visualization: BENIGN and MALIGNANT lesion counts
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

interpretations = ['BENIGN', 'MALIGNANT']
colors = ['lightblue', 'lightcoral']

for i, interpretation in enumerate(interpretations):
    if interpretation in merged_data['final_interpretation'].values:
        percentages = calculate_percentages(interpretation)
        
        if len(percentages) > 0:
            x_labels = []
            y_values = []
            
            # Add 0-20
            for j in range(21):
                key = str(j)
                if key in percentages and percentages[key] > 0:
                    x_labels.append(j)
                    y_values.append(percentages[key])
            
            # Add >20
            if '>20' in percentages and percentages['>20'] > 0:
                x_labels.append(21)
                y_values.append(percentages['>20'])
            
            # Create the bar plot
            bars = axes1[i].bar(range(len(x_labels)), y_values, color=colors[i], alpha=0.7)
            
            # Set labels
            tick_labels = [str(x) if x <= 20 else '>20' for x in x_labels]
            axes1[i].set_xticks(range(len(x_labels)))
            axes1[i].set_xticklabels(tick_labels, rotation=45)
            
            axes1[i].set_title(f'{interpretation} Cases')
            axes1[i].set_xlabel('Number of detected lesions (YOLO)')
            axes1[i].set_ylabel('Percentage of Cases')
            axes1[i].grid(True, alpha=0.3)
            
            # Add percentage labels on bars (only for bars > 1% to avoid clutter)
            for j, (bar, pct) in enumerate(zip(bars, y_values)):
                if pct > 1.0:
                    axes1[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                               f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# Save the first graph
plt.savefig(f'{env}\lesion_count_distribution.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Create second visualization: BI-RADS analysis for MALIGNANT cases with 0 lesions
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

bars = ax2.bar(birads_df['BI-RADS'].astype(str), birads_df['Percentage'], 
               color='crimson', alpha=0.7)
ax2.set_title('MALIGNANT Cases with 0 Lesions\nby BI-RADS Category')
ax2.set_xlabel('BI-RADS Category')
ax2.set_ylabel('Percentage (%)')
ax2.grid(True, alpha=0.3)

# Add both percentage and count labels on bars
for bar, pct, count in zip(bars, birads_df['Percentage'], birads_df['Zero_Lesions']):
    if pct > 0:
        # Percentage label above bar
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        # Count label inside the bar (centered vertically)
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                f'({int(count)})', ha='center', va='center', fontsize=8, color='black')

plt.tight_layout()

# Save the second graph
plt.savefig(f'{env}\malignant_zero_lesions_by_birads.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()