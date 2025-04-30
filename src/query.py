from google.cloud import bigquery
import time
import os
import pandas as pd
from tqdm import tqdm
from tools.audit import append_audit
# Get the current script directory and go back one directory
env = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(env)  # Go back one directory


BREAST_FILTER = """'IMG3425','IMG3426','IMG10897','IMG3571','IMG3557','IMG3558','IMG4636','IMG4637','IMG1100','IMG3506','IMG3545',
                  'IMG3508','IMG616','IMG3566','IMG1974','IMG3543','IMG3245','IMG3507','IMG3546','IMG3509','IMG1950','IMG3567',
                  'IMG1975','IMG3544','IMG3246','IMG10829','IMG10900','IMG3333','IMG3373','IMG3376','IMG3374','IMG3375','IMG3518',
                  'IMG3519','IMG3504','IMG3505','IMG3371','IMG3251','IMG3252','IMG10738','IMG600','IMG610','IMG3287','IMG3288',
                  'IMG588','IMG612','IMG3283','IMG3285','IMG589','IMG611','IMG3284','IMG3286','IMG1983','IMG579','IMG1982',
                  'IMG3524','IMG3523','IMG617','IMG1952','IMG3551','IMG3512','IMG3520','IMG3513','IMG3517','IMG3521','IMG10826',
                  'IMG3428','IMG3430','IMG3432','IMG3429','IMG3431','IMG3433','IMG3540','IMG3539','IMG3538','IMG3584','IMG3572',
                  'IMG3574','IMG3576','IMG3573','IMG3575','IMG3577','IMG3552','IMG3302','IMG3215','IMG605','IMG3221','IMG3570',
                  'IMG608','IMG3427','IMG3569','IMG609','IMG3222','IMG3568','IMG10828','IMG10902','IMG10808','IMG3382','IMG621',
                  'IMG3388','IMG3381','IMG3387','IMG10832','IMG1972','IMG1976','IMG3255','IMG1973','IMG1977','IMG3256','IMG10833',
                  'IMG3207','IMG2360','IMG3322','IMG3240','IMG3265','IMG3315','IMG3233','IMG3341','IMG3342','IMG3326','IMG3329',
                  'IMG3330','IMG3241','IMG3248','IMG3249','IMG3308','IMG3211','IMG4225','IMG1073','IMG4009','IMG4115','IMG4159',
                  'IMG3503','IMG3502','IMG4503','IMG3247','IMG4585','IMG5967','IMG10000','IMG3788','IMG1045','IMG1143','IMG1144',
                  'IMG10769','IMG1069','IMG3510','IMG1971','IMG3511','IMG1970','IMG4852','IMG1044','IMG1142','IMG1639','IMG3979',
                  'IMG4957','IMG4959','IMG1043','IMG1141','IMG1638','IMG3607','IMG2442','IMG13019','IMG13020','CMP021','CMP026',
                  'IMG10823','IMG4307','IMG4838','IMG4839','IMG4765','IMG4897','IMG4898','IMG4899'"""

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
    
    query = f"""
    -- First identify patients with breast imaging studies and at least one US modality
    -- excluding patients with US_CORE_BIRTHSEX = 'M'
    WITH breast_imaging_patients AS (
      SELECT DISTINCT PAT_PATIENT.CLINIC_NUMBER AS PATIENT_ID
      FROM `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.ImagingStudy` imaging
      INNER JOIN `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.Patient` PAT_PATIENT 
        ON (imaging.clinic_number = PAT_PATIENT.clinic_number)
      INNER JOIN `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.FACT_RADIOLOGY` RAD_FACT_RADIOLOGY 
        ON (imaging.ACCESSION_IDENTIFIER_VALUE = RAD_FACT_RADIOLOGY.ACCESSION_NBR)
      WHERE procedure_code_coding_code IN ({BREAST_FILTER})
        AND RAD_FACT_RADIOLOGY.SERVICE_MODALITY_CODE = 'US'
        AND PAT_PATIENT.US_CORE_BIRTHSEX != 'M'
    """
    
    if limit is not None:
        query += f"\nLIMIT {limit}\n"
        
    query += f"""
    )
    -- Then get all BREAST radiology data for these patients
    SELECT DISTINCT 
      PAT_PATIENT.CLINIC_NUMBER AS PATIENT_ID,
      RAD_FACT_RADIOLOGY.ACCESSION_NBR AS ACCESSION_NUMBER,
      imaging_studies.DESCRIPTION,
      imaging_studies.PROCEDURE_CODE_TEXT,
      ENDPOINT.ADDRESS AS ENDPOINT_ADDRESS,
      PAT_PATIENT.US_CORE_BIRTHSEX,
      RAD_FACT_RADIOLOGY.SERVICE_MODALITY_CODE AS MODALITY,
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
      PAT_DIM_PATIENT.PATIENT_BIRTH_DATE AS BIRTH_DATE
    FROM breast_imaging_patients
    INNER JOIN 
      `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.Patient` PAT_PATIENT 
      ON breast_imaging_patients.PATIENT_ID = PAT_PATIENT.CLINIC_NUMBER
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
      `ml-mps-adl-intfhr-phi-p-3b6e.phi_secondary_use_fhir_clinicnumber_us_p.Endpoint` ENDPOINT 
      ON (imaging_studies.gcp_endpoint_id = ENDPOINT.id)
    INNER JOIN 
      `ml-mps-adl-intudp-phi-p-d5cb.phi_udpwh_etl_us_p.DIM_RADIOLOGY_TEST_NAME` RADTEST_DIM_RADIOLOGY_TEST_NAME 
      ON (RAD_FACT_RADIOLOGY.RADIOLOGY_TEST_NAME_DK = RADTEST_DIM_RADIOLOGY_TEST_NAME.RADIOLOGY_TEST_NAME_DK)
    WHERE RADTEST_DIM_RADIOLOGY_TEST_NAME.RADIOLOGY_TEST_DESCRIPTION LIKE '%BREAST%'
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
          PATH_FACT_PATHOLOGY.SPECIMEN_ACCESSION_DTM,
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
    Run queries with complete radiology and pathology data per patient
    
    Args:
        limit (int, optional): Limit the number of patients to process
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(env, "raw_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Set up our audit destination
    append_audit(data_dir, f"Starting breast imaging query process with patient limit={limit}", new_file=True)
    
    total_start_time = time.time()
    print("Starting breast imaging query process...")
    
    # Step 1: Get all radiology data for breast imaging
    print("\n=== RADIOLOGY DATA QUERY ===")
    rad_df = get_radiology_data(limit=limit)

    # Audit radiology results
    rad_path = os.path.join(env, "raw_data", "raw_radiology.csv")
    rad_df.to_csv(rad_path, index=False)
    append_audit(data_dir, f"Found {len(rad_df)} radiology records")
    print(f"Radiology data saved")
    
    # Extract unique patient IDs from the radiology data
    patient_ids = rad_df['PATIENT_ID'].unique().tolist()
    append_audit(data_dir, f"Found {len(patient_ids)} radiology patients")
    print(f"Extracted {len(patient_ids)} unique patient IDs for pathology query")
    
    # Step 2: Get pathology data for these patients
    print("\n=== PATHOLOGY DATA QUERY ===")
    path_df = get_pathology_data(patient_ids)
    
    # Audit pathology results
    path_path = os.path.join(env, "raw_data", "raw_pathology.csv")
    path_df.to_csv(path_path, index=False)
    append_audit(data_dir, f"Found {len(path_df)} pathology records")
    print(f"Pathology data saved")
    
    # Calculate patient coverage metrics
    patients_with_path = path_df['PATIENT_ID'].nunique()
    coverage_percentage = (patients_with_path / len(patient_ids)) * 100 if patient_ids else 0
    append_audit(data_dir, f"{patients_with_path} of {len(patient_ids)} radiology patients ({coverage_percentage:.1f}%) have pathology data")
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nAll queries complete! Total execution time: {total_duration:.2f} seconds")
    
    return rad_df, path_df