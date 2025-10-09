import pandas as pd
from src.DB_processing.tools import append_audit


def calculate_patient_stats(breast_df):
    """Calculate and log patient-related statistics."""
    unique_patients = breast_df['Patient_ID'].nunique()
    append_audit("export.num_patients", unique_patients)
    
    # Year range
    breast_df['study_date'] = pd.to_datetime(breast_df['study_date'], errors='coerce')
    append_audit("export.year_range_start", int(breast_df['study_date'].dt.year.min()))
    append_audit("export.year_range_end", int(breast_df['study_date'].dt.year.max()))
    
    # Age statistics
    valid_ages = pd.to_numeric(breast_df['AGE_AT_EVENT'], errors='coerce').dropna()
    append_audit("export.min_patient_age", float(valid_ages.min()))
    append_audit("export.max_patient_age", float(valid_ages.max()))
    append_audit("export.avg_patient_age", float(valid_ages.mean()))


def calculate_image_stats(image_df):
    """Calculate and log image-related statistics."""
    # Images per exam
    exam_counts = image_df.groupby('Accession_Number').size()
    append_audit("export.min_images_per_exam", int(exam_counts.min()))
    append_audit("export.max_images_per_exam", int(exam_counts.max()))
    append_audit("export.avg_images_per_exam", float(exam_counts.mean()))
    
    # Image dimensions
    append_audit("export.avg_image_width", float(image_df['crop_w'].mean()))
    append_audit("export.avg_image_height", float(image_df['crop_h'].mean()))
    
    # Per-case counts
    case_counts = image_df.groupby('Accession_Number').size().tolist()
    append_audit("export.img_per_case", case_counts)


def calculate_video_stats(video_df, video_images_df=None):
    """Calculate and log video-related statistics."""
    if video_df.empty:
        append_audit("export.min_videos_per_exam", 0)
        append_audit("export.max_videos_per_exam", 0)
        append_audit("export.avg_videos_per_exam", 0)
        append_audit("export.vid_per_case", [])
        return
    
    # Videos per exam
    exam_counts = video_df.groupby('Accession_Number').size()
    append_audit("export.min_videos_per_exam", int(exam_counts.min()))
    append_audit("export.max_videos_per_exam", int(exam_counts.max()))
    append_audit("export.avg_videos_per_exam", float(exam_counts.mean()))
    
    # Video dimensions
    append_audit("export.avg_video_width", float(video_df['crop_w'].mean()))
    append_audit("export.avg_video_height", float(video_df['crop_h'].mean()))
    
    # Frame counts
    if video_images_df is not None and not video_images_df.empty:
        frame_counts = video_images_df['images'].apply(
            lambda x: len(x) if isinstance(x, list) else len(eval(x))
        )
        append_audit("export.avg_video_frames", float(frame_counts.mean()))
        append_audit("export.min_video_frames", int(frame_counts.min()))
        append_audit("export.max_video_frames", int(frame_counts.max()))
    
    # Per-case counts
    case_counts = video_df.groupby('Accession_Number').size().tolist()
    append_audit("export.vid_per_case", case_counts)


def calculate_laterality_stats(breast_df):
    """Calculate and log laterality distribution."""
    laterality_counts = breast_df['Study_Laterality'].value_counts()
    append_audit("export.num_left_breasts", int(laterality_counts.get('LEFT', 0)))
    append_audit("export.num_right_breasts", int(laterality_counts.get('RIGHT', 0)))
    append_audit("export.num_bilateral_breasts", int(laterality_counts.get('BILATERAL', 0)))


def calculate_split_diagnosis_stats(breast_df):
    """Calculate and log diagnosis counts by split and laterality."""
    breast_counts = {
        f"{lat.lower()}_{diag.lower()}": [0, 0, 0]
        for lat in ['RIGHT', 'LEFT']
        for diag in ['MALIGNANT', 'BENIGN']
    }
    
    for split_num in [0, 1, 2]:
        split_data = breast_df[breast_df['Valid'] == split_num]
        
        for laterality in ['RIGHT', 'LEFT']:
            for diagnosis in ['MALIGNANT', 'BENIGN']:
                condition = split_data['final_interpretation'].isin([diagnosis])
                key = f"{laterality.lower()}_{diagnosis.lower()}"
                breast_counts[key][split_num] = len(
                    split_data[(split_data['Study_Laterality'] == laterality) & condition]
                )
    
    for key, counts in breast_counts.items():
        append_audit(f'export.{key}_breasts', counts)


def calculate_machine_model_stats(image_df, breast_df):
    """Calculate and log machine model distribution by split."""
    if 'ManufacturerModelName' not in image_df.columns:
        append_audit("export.machine_models", "Column not found")
        return
    
    model_df = image_df.merge(
        breast_df[['Patient_ID', 'Accession_Number', 'Valid']], 
        on=['Patient_ID', 'Accession_Number'],
        how='left'
    )
    
    train_models = {}
    val_models = {}
    test_models = {}
    
    for model in model_df['ManufacturerModelName'].unique():
        safe_model = str(model).replace(' ', '_').replace('-', '_').replace('.', '_').replace("'", "")
        
        for split_num, models_dict in [(0, train_models), (1, val_models), (2, test_models)]:
            count = len(model_df[(model_df['Valid'] == split_num) & 
                                (model_df['ManufacturerModelName'] == model)])
            if count > 0:
                models_dict[safe_model] = count
    
    append_audit("export.train_machine_models", train_models)
    append_audit("export.val_machine_models", val_models)
    append_audit("export.test_machine_models", test_models)


def calculate_density_stats(breast_df):
    """Calculate and log breast density distribution by split."""
    if 'Density_Desc' not in breast_df.columns:
        append_audit("export.breast_densities", "Column not found")
        return
    
    density_keywords = {
        'entirely_fatty': ['entirely fatty'],
        'fibroglandular': ['fibroglandular'],
        'heterogeneously': ['heterogeneously'],
        'extremely_dense': ['extremely dense'],
        'unknown': []
    }
    
    def classify_density(desc):
        if pd.isna(desc):
            return 'unknown'
        desc = str(desc).lower()
        for category, keywords in density_keywords.items():
            if any(kw in desc for kw in keywords):
                return category
        return 'unknown'
    
    breast_df['density_category'] = breast_df['Density_Desc'].apply(classify_density)
    
    for split_num, split_name in [(0, 'train'), (1, 'val'), (2, 'test')]:
        split_data = breast_df[breast_df['Valid'] == split_num]
        density_counts = split_data['density_category'].value_counts().to_dict()
        densities = {cat: density_counts.get(cat, 0) for cat in density_keywords.keys()}
        append_audit(f"export.{split_name}_breast_densities", densities)


def calculate_birads_stats(breast_df):
    """Calculate and log BI-RADS distribution by split."""
    birad_values = ['0', '1', '2', '3', '4', '4A', '4B', '4C', '5', '6']
    
    for birad in birad_values:
        counts = [
            len(breast_df[(breast_df['Valid'] == split) & (breast_df['BI-RADS'] == birad)])
            for split in [0, 1, 2]
        ]
        safe_birad = birad.replace('-', '_').replace('/', '_')
        append_audit(f'export.birad_{safe_birad}', counts)


def calculate_split_image_stats(image_df, breast_df):
    """Calculate and log image counts per split."""
    merged_df = image_df.merge(
        breast_df[['Patient_ID', 'Accession_Number', 'Study_Laterality', 'Valid']], 
        on=['Patient_ID', 'Accession_Number'],
        how='left'
    )
    
    valid_counts = merged_df.groupby('Valid').size()
    
    for split_code, split_name in [(0, 'train'), (1, 'val'), (2, 'test')]:
        count = int(valid_counts.get(split_code, 0))
        append_audit(f'export.images_in_{split_name}', count)


def generate_audit_report(image_df, breast_df, video_df, video_images_df=None):
    """
    Generate comprehensive audit report for the export.
    
    Orchestrates all statistics calculations and logging.
    """
    calculate_patient_stats(breast_df)
    calculate_image_stats(image_df)
    calculate_video_stats(video_df, video_images_df)
    calculate_laterality_stats(breast_df)
    calculate_split_diagnosis_stats(breast_df)
    calculate_machine_model_stats(image_df, breast_df)
    calculate_density_stats(breast_df)
    calculate_birads_stats(breast_df)
    calculate_split_image_stats(image_df, breast_df)