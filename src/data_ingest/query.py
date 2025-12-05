from google.cloud import bigquery
import time
import os
import pandas as pd
from tqdm import tqdm
from src.DB_processing.tools import append_audit
# Get the current script directory and go back one directory
env = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(env)  # Go back one directory
env = os.path.dirname(env)  # Go back one directory

def get_radiology_data(limit=None):
    """
    Get radiology data for breast imaging studies, ensuring all relevant accessions
    for each patient are included, but still filtering for breast imaging only
    
    Args:
        limit (int, optional): Limit the number of patients returned
    
    Returns:
        pandas.DataFrame: Query results as a dataframe
    """
    
    print("Initializing BigQuery client for radiology data...")
    client = bigquery.Client()
    
    # Build the limit clause for the CTE
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    
    query = f"""
    -- First identify ALL patients with US modality at series level (excluding males)
    WITH us_imaging_patients AS (
      SELECT DISTINCT PAT_PATIENT.CLINIC_NUMBER AS PATIENT_ID
      FROM `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.ImagingStudy` imaging
      INNER JOIN `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.ImagingStudySeries` imaging_series
        ON (imaging.id = imaging_series.imaging_study_id)
      INNER JOIN `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.Patient` PAT_PATIENT 
        ON (imaging.clinic_number = PAT_PATIENT.clinic_number)
      WHERE imaging_series.SERIES_MODALITY_CODE = 'US'
        AND PAT_PATIENT.US_CORE_BIRTHSEX != 'M'
      {limit_clause}
    )
    -- Then get radiology data for these patients with BREAST test descriptions
    SELECT DISTINCT 
      PAT_PATIENT.CLINIC_NUMBER AS PATIENT_ID,
      RAD_FACT_RADIOLOGY.ACCESSION_NBR AS ACCESSION_NUMBER,
      imaging_studies.DESCRIPTION,
      imaging_studies.PROCEDURE_CODE_TEXT,
      ENDPOINT.ADDRESS AS ENDPOINT_ADDRESS,
      PAT_PATIENT.US_CORE_BIRTHSEX,
      IMAGINGSTUDYSERIES.SERIES_MODALITY_CODE AS MODALITY,
      RAD_FACT_RADIOLOGY.RADIOLOGY_NARRATIVE,
      RAD_FACT_RADIOLOGY.RADIOLOGY_REPORT,
      RAD_FACT_RADIOLOGY.SERVICE_RESULT_STATUS,
      RAD_FACT_RADIOLOGY.RADIOLOGY_DTM,
      RAD_FACT_RADIOLOGY.RADIOLOGY_REVIEW_DTM,
      RADTEST_DIM_RADIOLOGY_TEST_NAME.RADIOLOGY_TEST_DESCRIPTION AS TEST_DESCRIPTION,
      -- Added demographic fields
      PAT_DIM_PATIENT.PATIENT_ETHNICITY_NAME AS ETHNICITY,
      PAT_DIM_PATIENT.PATIENT_DEATH_DATE AS DEATH_DATE,
      PAT_DIM_PATIENT.PATIENT_PRIMARY_ZIPCODE AS ZIPCODE,
      PAT_DIM_PATIENT.PATIENT_RACE_NAME AS RACE,
      DATE_DIFF(EXTRACT(DATE FROM RAD_FACT_RADIOLOGY.RADIOLOGY_DTM), PAT_DIM_PATIENT.PATIENT_BIRTH_DATE, YEAR) - 
        IF(EXTRACT(MONTH FROM PAT_DIM_PATIENT.PATIENT_BIRTH_DATE)*100 + EXTRACT(DAY FROM PAT_DIM_PATIENT.PATIENT_BIRTH_DATE) > 
          EXTRACT(MONTH FROM RAD_FACT_RADIOLOGY.RADIOLOGY_DTM)*100 + EXTRACT(DAY FROM RAD_FACT_RADIOLOGY.RADIOLOGY_DTM),1,0) AS AGE_AT_EVENT,
      PAT_DIM_PATIENT.PATIENT_BIRTH_DATE AS BIRTH_DATE,
      -- Added breast-specific pathology fields
      breast_results.A1_PATHOLOGY_TXT,
      breast_results.A1_PATHOLOGY_CATEGORY_DESC,
      breast_results.A2_PATHOLGY_TXT,
      breast_results.A2_PATHOLOGY_CATEGORY_DESC
    FROM us_imaging_patients
    INNER JOIN 
      `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.Patient` PAT_PATIENT 
      ON us_imaging_patients.PATIENT_ID = PAT_PATIENT.CLINIC_NUMBER
    INNER JOIN 
      `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_PATIENT` PAT_DIM_PATIENT
      ON PAT_PATIENT.CLINIC_NUMBER = PAT_DIM_PATIENT.PATIENT_CLINIC_NUMBER
    INNER JOIN 
      `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.FACT_RADIOLOGY` RAD_FACT_RADIOLOGY 
      ON PAT_DIM_PATIENT.PATIENT_DK = RAD_FACT_RADIOLOGY.PATIENT_DK
    LEFT JOIN 
      `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.ImagingStudy` imaging_studies
      ON (RAD_FACT_RADIOLOGY.ACCESSION_NBR = imaging_studies.ACCESSION_IDENTIFIER_VALUE)
    LEFT JOIN
      `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.ImagingStudySeries` IMAGINGSTUDYSERIES
      ON (imaging_studies.id = IMAGINGSTUDYSERIES.imaging_study_id)
    LEFT JOIN 
      `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.Endpoint` ENDPOINT 
      ON (imaging_studies.gcp_endpoint_id = ENDPOINT.id)
    INNER JOIN 
      `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_RADIOLOGY_TEST_NAME` RADTEST_DIM_RADIOLOGY_TEST_NAME 
      ON (RAD_FACT_RADIOLOGY.RADIOLOGY_TEST_NAME_DK = RADTEST_DIM_RADIOLOGY_TEST_NAME.RADIOLOGY_TEST_NAME_DK)
    LEFT JOIN 
      `ml-mps-adl-intudp-phi-p-d5cb.phi_rad_udpwh_us_p.DIM_RADIOLOGY_EXAM` rad_exam
      ON (RAD_FACT_RADIOLOGY.ACCESSION_NBR = rad_exam.ACCESSION_NBR_ID)
    LEFT JOIN 
      `ml-mps-adl-intudp-phi-p-d5cb.phi_rad_udpwh_us_p.DIM_RADIOLOGY_EXAM_RESULTS_BREAST` breast_results
      ON (rad_exam.RADIOLOGY_EXAM_DK = breast_results.RADIOLOGY_EXAM_DK)
    WHERE RADTEST_DIM_RADIOLOGY_TEST_NAME.RADIOLOGY_TEST_DESCRIPTION LIKE '%BREAST%'
      AND imaging_studies.DESCRIPTION LIKE '%BREAST%'
    """

    query_start_time = time.time()
    print("Executing radiology query...")
    df = client.query(query).to_dataframe()
    query_end_time = time.time()
    query_duration = query_end_time - query_start_time

    print(f"Radiology query complete. Retrieved {len(df)} rows for {len(df['PATIENT_ID'].unique())} patients in {query_duration:.2f} seconds.")

    return df

def get_pathology_data(patient_ids, batch_size=1000):
    """
    Get pathology data for specific patient IDs, processing in batches
    
    Args:
        patient_ids (list): List of patient IDs to query
        batch_size (int): Number of patients to process in each batch
    
    Returns:
        pandas.DataFrame: Query results as a dataframe
    """
    start_time = time.time()
    print("Initializing BigQuery client for pathology data...")
    client = bigquery.Client()
    
    # Process in batches
    all_results = []
    total_patients = len(patient_ids)
    total_batches = (total_patients + batch_size - 1) // batch_size
    
    # Create a tqdm progress bar
    for i in tqdm(range(0, total_patients, batch_size), total=total_batches):
        batch = patient_ids[i:i+batch_size]

        # Format IDs appropriately
        if batch and all(str(id).isdigit() for id in batch):
            ids_str = ', '.join([str(id) for id in batch])
        else:
            ids_str = ', '.join([f"'{id}'" for id in batch])
        
        query = f"""
        SELECT 
          PAT_DIM_PATIENT.PATIENT_CLINIC_NUMBER AS PATIENT_ID,
          PATH_FACT_PATHOLOGY.SPECIMEN_NOTE,
          PATH_FACT_PATHOLOGY.SPECIMEN_UPDATE_DTM,
          PATH_FACT_PATHOLOGY.SPECIMEN_RESULT_DTM,
          PATH_FACT_PATHOLOGY.SPECIMEN_RECEIVED_DTM,
          PATH_FACT_PATHOLOGY.SPECIMEN_SERVICE_DESCRIPTION,
          PATH_FACT_PATHOLOGY.ENCOUNTER_ID,
          DIAGCODE_DIM_DIAGNOSIS_CODE.DIAGNOSIS_NAME,
          PATH_FACT_PATHOLOGY.PATHOLOGY_COUNT,
          PATH_FACT_PATHOLOGY.SPECIMEN_COMMENT,
          PATH_FACT_PATHOLOGY.SPECIMEN_ACCESSION_NUMBER,
          SPECDET.PART_DESCRIPTION,
          SPECPARTYP.SPECIMEN_PART_TYPE_CODE,
          SPECPARTYP.SPECIMEN_PART_TYPE_NAME
        FROM `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.FACT_PATHOLOGY` PATH_FACT_PATHOLOGY
        INNER JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_PATIENT` PAT_DIM_PATIENT
          ON (PATH_FACT_PATHOLOGY.PATIENT_DK = PAT_DIM_PATIENT.PATIENT_DK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_PATHOLOGY_DIAGNOSIS_CODE_BRIDGE` PATHDIAG
          ON (PATH_FACT_PATHOLOGY.PATHOLOGY_FPK = PATHDIAG.PATHOLOGY_FPK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_DIAGNOSIS_CODE` DIAGCODE_DIM_DIAGNOSIS_CODE
          ON (PATHDIAG.DIAGNOSIS_CODE_DK = DIAGCODE_DIM_DIAGNOSIS_CODE.DIAGNOSIS_CODE_DK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.FACT_PATHOLOGY_SPECIMEN_DETAIL` SPECDET
          ON (PATH_FACT_PATHOLOGY.PATHOLOGY_FPK = SPECDET.PATHOLOGY_FPK)
        LEFT JOIN
          `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_SPECIMEN_PART_TYPE` SPECPARTYP
          ON (SPECDET.SPECIMEN_PART_TYPE_DK = SPECPARTYP.SPECIMEN_PART_TYPE_DK)
        WHERE PAT_DIM_PATIENT.PATIENT_CLINIC_NUMBER IN ({ids_str})
        AND (
          LOWER(SPECPARTYP.SPECIMEN_PART_TYPE_CODE) IN ('breast','breast1','breast2','breast3','breast4','breast5','breast6','breast7','breast8','breast9','breast10','breast11')
        )
        """
        
        batch_df = client.query(query).to_dataframe()
        all_results.append(batch_df)
    
    # Combine all batch results
    if all_results:
        df = pd.concat(all_results, ignore_index=True)
    else:
        df = pd.DataFrame()
    
    total_duration = time.time() - start_time
    print(f"Pathology query complete. Retrieved {len(df)} total rows in {total_duration:.2f} seconds.")
    
    return df

  
def run_breast_imaging_query(limit=None):
    """
    Run queries with complete radiology, pathology, and lab data per patient
    
    Args:
        limit (int, optional): Limit the number of patients to process
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(env, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Set up our audit destination
    append_audit("query.patient_limit", limit, new_file=True)
    
    total_start_time = time.time()
    print("Starting breast imaging query process...")
    
    # Step 1: Get all radiology data for breast imaging
    print("\n=== RADIOLOGY DATA QUERY ===")
    rad_df = get_radiology_data(limit=limit)

    # Audit radiology results
    rad_path = os.path.join(env, "data", "raw_radiology.csv")
    rad_df.to_csv(rad_path, index=False)
    append_audit("query.raw_rad_record_count", len(rad_df))
    
    # Extract unique patient IDs from the radiology data
    patient_ids = rad_df['PATIENT_ID'].unique().tolist()
    append_audit("query.raw_rad_unique_patients", len(patient_ids))
    print(f"Extracted {len(patient_ids)} unique patient IDs for pathology query")
    
    # Step 2: Get pathology data for these patients
    print("\n=== PATHOLOGY DATA QUERY ===")
    path_df = get_pathology_data(patient_ids)
    
    # Audit pathology results
    path_path = os.path.join(env, "data", "raw_pathology.csv")
    path_df.to_csv(path_path, index=False)
    append_audit("query.raw_path_record_count", len(path_df))
    
    
    
    # Calculate patient coverage metrics
    patients_with_path = path_df['PATIENT_ID'].nunique()
    path_coverage_percentage = (patients_with_path / len(patient_ids)) * 100 if patient_ids else 0

    print(f"{patients_with_path} of {len(patient_ids)} radiology patients ({path_coverage_percentage:.1f}%) have pathology data")

    append_audit("query.rad_patients_with_path", patients_with_path)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nAll queries complete! Total execution time: {total_duration:.2f} seconds")
    
    return rad_df, path_df