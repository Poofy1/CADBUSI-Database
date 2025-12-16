"""
SQLite database handler for CADBUSI Database processing.
"""
import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any
import os


class DatabaseManager:
    """Manages SQLite database connections and operations for DICOM processing."""

    def __init__(self):
        self.database_path = "data"
        self.db_file = os.path.join(self.database_path, 'cadbusi.db')
        self.conn = None

    def connect(self):
        """Create database connection and enable foreign keys."""
        os.makedirs(self.database_path, exist_ok=True)  # ensure dir exists
        self.db_file = os.path.abspath(self.db_file)
        self.conn = sqlite3.connect(self.db_file)
        self.conn.execute("PRAGMA foreign_keys = ON")
        return self.conn

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.close()

    def create_schema(self):
        """Create database schema with all tables."""
        cursor = self.conn.cursor()

        # StudyCases table (Breast/Accession level data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS StudyCases (
                accession_number TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                study_laterality TEXT,
                birth_date TEXT,
                test_description TEXT,
                has_malignant INTEGER DEFAULT 0,
                has_benign INTEGER DEFAULT 0,
                is_biopsy INTEGER DEFAULT 0,
                is_us_biopsy INTEGER DEFAULT 0,
                left_diagnosis TEXT,
                right_diagnosis TEXT,
                findings TEXT,
                synoptic_report TEXT,
                description TEXT,
                us_core_birthsex TEXT,
                radiology_review_dtm TEXT,
                death_date TEXT,
                density_desc TEXT,
                rad_pathology_txt TEXT,
                rad_impression TEXT,
                date TEXT,
                bi_rads TEXT,
                biopsy TEXT,
                modality TEXT,
                modality_guidance TEXT,
                days_to_biopsy INTEGER,
                age_at_event INTEGER,
                ethnicity TEXT,
                race TEXT,
                zipcode TEXT,
                margin TEXT,
                shape TEXT,
                orientation TEXT,
                echo TEXT,
                posterior TEXT,
                boundary TEXT,
                left_prior_breast_biopsies INTEGER,
                right_prior_breast_biopsies INTEGER,
                left_prior_breast_cancer INTEGER,
                right_prior_breast_cancer INTEGER,
                left_diagnosis_source TEXT,
                right_diagnosis_source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Images table (Individual image/frame data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                accession_number TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                image_name TEXT UNIQUE NOT NULL,
                dicom_hash TEXT UNIQUE NOT NULL,
                laterality TEXT,
                area TEXT,
                orientation TEXT,
                clock_pos TEXT,
                nipple_dist INTEGER,
                description TEXT,
                region_spatial_format INTEGER,
                region_data_type INTEGER,
                region_location_min_x0 INTEGER,
                region_location_min_y0 INTEGER,
                region_location_max_x1 INTEGER,
                region_location_max_y1 INTEGER,
                crop_x INTEGER,
                crop_y INTEGER,
                crop_w INTEGER,
                crop_h INTEGER,
                crop_aspect_ratio REAL,
                photometric_interpretation TEXT,
                rows INTEGER,
                columns INTEGER,
                physical_delta_x REAL,
                has_calipers INTEGER DEFAULT 0,
                has_calipers_prediction REAL,
                caliper_boxes TEXT,
                caliper_coordinates TEXT,
                yolo_confidence TEXT,
                has_caliper_mask INTEGER DEFAULT 0,
                samus_confidence TEXT,
                darkness REAL,
                label INTEGER DEFAULT 1,
                region_count INTEGER DEFAULT 1,
                closest_fn TEXT,
                distance REAL DEFAULT 99999,
                file_name TEXT,
                inpainted_from TEXT,
                software_versions TEXT,
                manufacturer_model_name TEXT,
                exclusion_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (accession_number) REFERENCES StudyCases(accession_number) ON DELETE CASCADE
            )
        """)


        # Videos table (Video sequence data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Videos (
                video_id INTEGER PRIMARY KEY AUTOINCREMENT,
                accession_number TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                images_path TEXT UNIQUE NOT NULL,
                dicom_hash TEXT UNIQUE NOT NULL,
                laterality TEXT,
                saved_frames INTEGER,
                region_spatial_format INTEGER,
                region_data_type INTEGER,
                region_location_min_x0 INTEGER,
                region_location_min_y0 INTEGER,
                region_location_max_x1 INTEGER,
                region_location_max_y1 INTEGER,
                crop_x INTEGER,
                crop_y INTEGER,
                crop_w INTEGER,
                crop_h INTEGER,
                photometric_interpretation TEXT,
                rows INTEGER,
                columns INTEGER,
                physical_delta_x REAL,
                file_name TEXT,
                software_versions TEXT,
                manufacturer_model_name TEXT,
                nipple_dist REAL,
                orientation TEXT,
                clock_pos INTEGER,
                area REAL,
                description TEXT,
                crop_aspect_ratio REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (accession_number) REFERENCES StudyCases(accession_number) ON DELETE CASCADE
            )
        """)

        # Pathology table (Separate pathology data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Pathology (
                path_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                accession_number TEXT,
                date TEXT,
                lesion_diag TEXT,
                cancer_type TEXT,
                synoptic_report TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_accession ON Images(accession_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_patient ON Images(patient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_dicom_hash ON Images(dicom_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_laterality ON Images(laterality)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_accession ON Videos(accession_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_patient ON Videos(patient_id)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_studies_patient ON StudyCases(patient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_studies_laterality ON StudyCases(study_laterality)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_studies_malignant ON StudyCases(has_malignant)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_path_patient ON Pathology(patient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_path_accession ON Pathology(accession_number)")

        self.conn.commit()

    def _batch_upsert_helper(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        all_columns: List[str],
        unique_key: str,
        string_columns: List[str] = None,
        boolean_columns: List[str] = None,
        upsert: bool = False,
        update_only: bool = False
    ) -> int:
        """
        Generic helper for batch insert/update operations.
        
        Args:
            table_name: Name of the database table
            data: List of dictionaries with row data
            all_columns: List of all possible column names for the table
            unique_key: Column name used as unique identifier (for conflicts/updates)
            string_columns: Columns that should always be converted to strings
            boolean_columns: Columns that should be converted to 0/1 integers
            upsert: If True, updates existing records on conflict
            update_only: If True, only updates existing records
        
        Returns:
            Number of rows affected
        """
        if not data:
            return 0
        
        string_columns = string_columns or []
        boolean_columns = boolean_columns or []
        
        cursor = self.conn.cursor()
        
        # Find which columns are present in the data
        first_row = data[0]
        present_columns = [col for col in all_columns if col in first_row]
        
        # Handle UPDATE-only mode
        if update_only:
            if unique_key not in present_columns:
                raise ValueError(f"update_only mode requires '{unique_key}' in data")
            
            # Build UPDATE query
            update_cols = [col for col in present_columns if col != unique_key]
            if not update_cols:
                return 0
                
            set_clause = ', '.join([f"{col} = ?" for col in update_cols])
            update_query = f"UPDATE {table_name} SET {set_clause} WHERE {unique_key} = ?"
            
            # Extract values for UPDATE
            rows_to_update = []
            for row in data:
                values = []
                for col in update_cols:
                    val = row.get(col)
                    if col in boolean_columns:
                        values.append(1 if val else 0)
                    elif col in string_columns:
                        values.append(str(val) if val is not None else '')
                    else:
                        values.append(val)
                # Add unique key for WHERE clause
                unique_val = row.get(unique_key)
                values.append(str(unique_val) if unique_val is not None else '')
                rows_to_update.append(tuple(values))
            
            cursor.executemany(update_query, rows_to_update)
            self.conn.commit()
            return cursor.rowcount
        
        # Build INSERT query
        placeholders = ', '.join(['?' for _ in present_columns])
        columns_str = ', '.join(present_columns)
        
        if upsert:
            # Update all columns except unique keys on conflict
            update_cols = [col for col in present_columns if col != unique_key]
            update_str = ', '.join([f"{col} = excluded.{col}" for col in update_cols])
            insert_query = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT({unique_key}) DO UPDATE SET {update_str}
            """
        else:
            insert_query = f"""
                INSERT OR IGNORE INTO {table_name} ({columns_str})
                VALUES ({placeholders})
            """
        
        # Extract values with type conversions
        rows_to_insert = []
        for row in data:
            values = []
            for col in present_columns:
                val = row.get(col)
                if col in boolean_columns:
                    values.append(1 if val else 0)
                elif col in string_columns:
                    values.append(str(val) if val is not None else '')
                else:
                    values.append(val)
            rows_to_insert.append(tuple(values))
        
        cursor.executemany(insert_query, rows_to_insert)
        self.conn.commit()
        return cursor.rowcount


    def insert_images_batch(self, image_data: List[Dict[str, Any]], upsert: bool = False, update_only: bool = False) -> int:
        """Insert multiple images in a single transaction."""
        all_columns = [
            'accession_number', 'patient_id', 'image_name', 'dicom_hash',
            'laterality', 'area', 'orientation', 'clock_pos', 'nipple_dist', 'description',
            'region_spatial_format', 'region_data_type', 'inpainted_from',
            'region_location_min_x0', 'region_location_min_y0',
            'region_location_max_x1', 'region_location_max_y1',
            'crop_x', 'crop_y', 'crop_w', 'crop_h', 'crop_aspect_ratio',
            'photometric_interpretation', 'rows', 'columns', 'physical_delta_x',
            'has_calipers', 'has_calipers_prediction', 'caliper_boxes', 'caliper_coordinates', 'yolo_confidence', 'has_caliper_mask', 'samus_confidence',
            'darkness', 'label', 'region_count', 'closest_fn', 'distance',
            'file_name', 'software_versions', 'manufacturer_model_name', 'exclusion_reason'
        ]

        string_columns = ['accession_number', 'patient_id', 'image_name', 'dicom_hash', 'caliper_boxes', 'caliper_coordinates', 'yolo_confidence', 'samus_confidence', 'inpainted_from', 'exclusion_reason']
        boolean_columns = ['has_calipers', 'has_caliper_mask', 'label']
        
        return self._batch_upsert_helper(
            table_name='Images',
            data=image_data,
            all_columns=all_columns,
            unique_key='image_name',
            string_columns=string_columns,
            boolean_columns=boolean_columns,
            upsert=upsert,
            update_only=update_only
        )


    def insert_videos_batch(self, video_data: List[Dict[str, Any]], upsert: bool = False, update_only: bool = False) -> int:
        """Insert multiple videos in a single transaction."""
        all_columns = [
            'accession_number', 'patient_id', 'images_path', 'dicom_hash', 'laterality',
            'saved_frames', 'region_spatial_format', 'region_data_type',
            'region_location_min_x0', 'region_location_min_y0',
            'region_location_max_x1', 'region_location_max_y1',
            'crop_x', 'crop_y', 'crop_w', 'crop_h',
            'photometric_interpretation', 'rows', 'columns', 'physical_delta_x',
            'file_name', 'software_versions', 'manufacturer_model_name',
            'nipple_dist', 'orientation', 'clock_pos', 'area', 'description', 'crop_aspect_ratio'
        ]
        
        string_columns = ['accession_number', 'patient_id', 'images_path', 'dicom_hash']
        boolean_columns = []  # No boolean columns in Videos table
        
        return self._batch_upsert_helper(
            table_name='Videos',
            data=video_data,
            all_columns=all_columns,
            unique_key='images_path',
            string_columns=string_columns,
            boolean_columns=boolean_columns,
            upsert=upsert,
            update_only=update_only
        )


    def insert_study_cases_batch(self, study_data: List[Dict[str, Any]], upsert: bool = True, update_only: bool = False) -> int:
        """Insert multiple study cases in a single transaction."""
        all_columns = [
            'accession_number', 'patient_id', 'study_laterality',
            'birth_date', 'test_description', 'has_malignant', 'has_benign',
            'is_biopsy', 'is_us_biopsy', 'left_diagnosis', 'right_diagnosis', 'findings',
            'synoptic_report', 'description', 'us_core_birthsex',
            'radiology_review_dtm', 'death_date', 'density_desc',
            'rad_pathology_txt', 'rad_impression', 'date',
            'bi_rads', 'biopsy', 'modality', 'modality_guidance', 'days_to_biopsy',
            'age_at_event', 'ethnicity', 'race', 'zipcode', 'margin', 'shape',
            'orientation', 'echo', 'posterior', 'boundary',
            'left_prior_breast_biopsies', 'right_prior_breast_biopsies',
            'left_prior_breast_cancer', 'right_prior_breast_cancer',
            'left_diagnosis_source', 'right_diagnosis_source'
        ]
        
        string_columns = ['accession_number', 'patient_id']
        boolean_columns = ['has_malignant', 'has_benign', 'is_biopsy', 'is_us_biopsy']
        
        return self._batch_upsert_helper(
            table_name='StudyCases',
            data=study_data,
            all_columns=all_columns,
            unique_key='accession_number',
            string_columns=string_columns,
            boolean_columns=boolean_columns,
            upsert=upsert,
            update_only=update_only
        )

    def insert_pathology_batch(self, pathology_data: List[Dict[str, Any]]) -> int:
        """Insert multiple pathology/lesion records in a single transaction."""
        cursor = self.conn.cursor()

        insert_query = """
            INSERT OR REPLACE INTO Pathology (
                patient_id, accession_number, date,
                lesion_diag, synoptic_report, cancer_type
            ) VALUES (?, ?, ?, ?, ?, ?)
        """

        rows_to_insert = [
            (
                str(row.get('patient_id', '')), 
                str(row.get('accession_number', '')), 
                str(row.get('date', '')),
                str(row.get('lesion_diag', '')),
                str(row.get('synoptic_report', '')),
                str(row.get('cancer_type', ''))
            )
            for row in pathology_data
        ]

        cursor.executemany(insert_query, rows_to_insert)
        self.conn.commit()
        return cursor.rowcount
    
    def get_images_dataframe(self, where_clause: str = "", params: tuple = ()) -> pd.DataFrame:
        """Get images as a pandas DataFrame with optional filtering."""
        query = "SELECT * FROM Images"
        if where_clause:
            query += f" WHERE {where_clause}"

        return pd.read_sql_query(query, self.conn, params=params)

    def get_videos_dataframe(self, where_clause: str = "", params: tuple = ()) -> pd.DataFrame:
        """Get videos as a pandas DataFrame with optional filtering."""
        query = "SELECT * FROM Videos"
        if where_clause:
            query += f" WHERE {where_clause}"

        return pd.read_sql_query(query, self.conn, params=params)

    def get_study_cases_dataframe(self, where_clause: str = "", params: tuple = ()) -> pd.DataFrame:
        """Get study cases as a pandas DataFrame with optional filtering."""
        query = "SELECT * FROM StudyCases"
        if where_clause:
            query += f" WHERE {where_clause}"

        return pd.read_sql_query(query, self.conn, params=params)

    def get_pathology_dataframe(self, where_clause: str = "", params: tuple = ()) -> pd.DataFrame:
        """Get pathology data as a pandas DataFrame with optional filtering."""
        query = "SELECT * FROM Pathology"
        if where_clause:
            query += f" WHERE {where_clause}"
        return pd.read_sql_query(query, self.conn, params=params)

    def check_existing_patient_ids(self) -> set:
        """Get set of all existing patient IDs in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT patient_id FROM Images")
        return {row[0] for row in cursor.fetchall()}

    def get_existing_accession_numbers(self) -> set:
        """Get set of all existing accession numbers from StudyCases table."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT accession_number FROM StudyCases")
        return {row[0] for row in cursor.fetchall()}

    def update_image_metadata_from_studies(self):
        """Update image laterality and area from StudyCases where missing."""
        cursor = self.conn.cursor()

        # Update laterality from StudyCases
        cursor.execute("""
            UPDATE Images
            SET laterality = (
                SELECT study_laterality
                FROM StudyCases
                WHERE StudyCases.accession_number = Images.accession_number
            )
            WHERE laterality IS NULL
            AND EXISTS (
                SELECT 1 FROM StudyCases
                WHERE StudyCases.accession_number = Images.accession_number
            )
        """)
        self.conn.commit()

    def extract_metadata_from_filenames(self):
        """Extract laterality and area from image/video filenames where missing."""
        cursor = self.conn.cursor()

        # Common patterns: RT (right), LT (left), followed by area codes
        # This is a placeholder - adjust based on actual filename patterns
        cursor.execute("""
            UPDATE Images
            SET laterality = CASE
                WHEN image_name LIKE '%_RT_%' OR image_name LIKE '%RT%' THEN 'R'
                WHEN image_name LIKE '%_LT_%' OR image_name LIKE '%LT%' THEN 'L'
                ELSE laterality
            END,
            area = CASE
                WHEN image_name LIKE '%_A_%' THEN 'A'
                WHEN image_name LIKE '%_P_%' THEN 'P'
                WHEN image_name LIKE '%_M_%' THEN 'M'
                WHEN image_name LIKE '%_L_%' THEN 'L'
                ELSE area
            END
            WHERE laterality IS NULL OR area IS NULL
        """)
        self.conn.commit()

    def add_column_if_not_exists(self, table_name: str, column_name: str, column_type: str, default_value: str = None):
        """Add a column to a table if it doesn't already exist."""
        cursor = self.conn.cursor()

        # Check if column exists
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]

        if column_name not in columns:
            # Build ALTER TABLE statement
            alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            if default_value is not None:
                alter_sql += f" DEFAULT {default_value}"

            cursor.execute(alter_sql)
            self.conn.commit()
            print(f"Added column '{column_name}' to table '{table_name}'")
            return True
        return False