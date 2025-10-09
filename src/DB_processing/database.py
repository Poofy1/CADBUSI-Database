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
                study_date TEXT,
                has_malignant INTEGER DEFAULT 0,
                has_benign INTEGER DEFAULT 0,
                final_interpretation TEXT,
                findings TEXT,
                synoptic_report TEXT,
                bi_rads TEXT,
                modality TEXT,
                age_at_event INTEGER,
                ethnicity TEXT,
                race TEXT,
                zipcode TEXT,
                split INTEGER,
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
                has_caliper_mask INTEGER DEFAULT 0,
                darkness REAL,
                is_labeled INTEGER DEFAULT 1,
                label INTEGER DEFAULT 1,
                region_count INTEGER DEFAULT 1,
                closest_fn TEXT,
                distance REAL DEFAULT 99999,
                file_name TEXT,
                software_versions TEXT,
                manufacturer_model_name TEXT,
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
                label INTEGER DEFAULT 1,
                software_versions TEXT,
                manufacturer_model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (accession_number) REFERENCES StudyCases(accession_number) ON DELETE CASCADE
            )
        """)

        # Lesions table (Cropped lesion data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Lesions (
                lesion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_image_name TEXT NOT NULL,
                image_name TEXT UNIQUE NOT NULL,
                accession_number TEXT,
                patient_id TEXT,
                crop_x INTEGER,
                crop_y INTEGER,
                crop_w INTEGER,
                crop_h INTEGER,
                has_mask INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_image_name) REFERENCES Images(image_name) ON DELETE CASCADE
            )
        """)

        # Pathology table (Separate pathology data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Pathology (
                path_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                accession_number TEXT,
                specimen_date TEXT,
                laterality TEXT,
                interpretation TEXT,
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_labeled ON Images(is_labeled)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_accession ON Videos(accession_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_patient ON Videos(patient_id)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_studies_patient ON StudyCases(patient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_studies_laterality ON StudyCases(study_laterality)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_studies_malignant ON StudyCases(has_malignant)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_studies_split ON StudyCases(split)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lesions_source ON Lesions(source_image_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lesions_accession ON Lesions(accession_number)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_path_patient ON Pathology(patient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_path_accession ON Pathology(accession_number)")

        self.conn.commit()
        print("Database schema created successfully")

    def insert_images_batch(self, image_data: List[Dict[str, Any]], upsert: bool = False, update_only: bool = False) -> int:
        """
        Insert multiple images in a single transaction.
        Only inserts columns that are present in the data.
        
        Args:
            image_data: List of dictionaries with image data
            upsert: If True, updates existing records. If False, ignores duplicates.
            update_only: If True, only updates existing records (no insert attempt). Requires 'image_name' in data.
        """
        if not image_data:
            return 0
        
        cursor = self.conn.cursor()
        
        # Get all possible columns (excluding auto-increment and timestamp)
        all_columns = [
            'accession_number', 'patient_id', 'image_name', 'dicom_hash',
            'laterality', 'area', 'orientation', 'clock_pos', 'nipple_dist', 'description',
            'region_spatial_format', 'region_data_type',
            'region_location_min_x0', 'region_location_min_y0',
            'region_location_max_x1', 'region_location_max_y1',
            'crop_x', 'crop_y', 'crop_w', 'crop_h', 'crop_aspect_ratio',
            'photometric_interpretation', 'rows', 'columns', 'physical_delta_x',
            'has_calipers', 'has_calipers_prediction', 'caliper_boxes', 'has_caliper_mask',
            'darkness', 'is_labeled', 'label', 'region_count', 'closest_fn', 'distance',
            'file_name', 'software_versions', 'manufacturer_model_name'
        ]
        
        # Find which columns are actually present in the data
        first_row = image_data[0]
        present_columns = [col for col in all_columns if col in first_row]
        
        # Handle UPDATE-only mode
        if update_only:
            if 'image_name' not in present_columns:
                raise ValueError("update_only mode requires 'image_name' in data")
            
            # Build UPDATE query
            update_cols = [col for col in present_columns if col != 'image_name']
            set_clause = ', '.join([f"{col} = ?" for col in update_cols])
            update_query = f"UPDATE Images SET {set_clause} WHERE image_name = ?"
            
            # Extract values for UPDATE (excluding image_name, which goes at the end)
            rows_to_update = [
                tuple(
                    # Handle boolean conversions
                    [(1 if row.get(col) else 0) if col in ['has_calipers', 'has_caliper_mask', 'is_labeled', 'label']
                    else str(row.get(col, '')) if col in ['caliper_boxes']
                    else row.get(col)
                    for col in update_cols] +
                    [str(row.get('image_name', ''))]  # image_name for WHERE clause
                )
                for row in image_data
            ]
            
            cursor.executemany(update_query, rows_to_update)
            self.conn.commit()
            return cursor.rowcount
        
        # Original INSERT logic for non-update_only mode
        placeholders = ', '.join(['?' for _ in present_columns])
        columns_str = ', '.join(present_columns)
        
        if upsert:
            # Update all columns except the primary key on conflict
            update_cols = [col for col in present_columns if col not in ['image_name', 'dicom_hash']]
            update_str = ', '.join([f"{col} = excluded.{col}" for col in update_cols])
            insert_query = f"""
                INSERT INTO Images ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT(image_name) DO UPDATE SET {update_str}
            """
        else:
            insert_query = f"""
                INSERT OR IGNORE INTO Images ({columns_str})
                VALUES ({placeholders})
            """
        
        # Extract values in the same order as present_columns
        rows_to_insert = [
            tuple(
                # Handle boolean conversions
                (1 if row.get(col) else 0) if col in ['has_calipers', 'has_caliper_mask', 'is_labeled', 'label']
                else str(row.get(col, '')) if col in ['accession_number', 'patient_id', 'image_name', 'dicom_hash', 'caliper_boxes']
                else row.get(col)
                for col in present_columns
            )
            for row in image_data
        ]
                
        cursor.executemany(insert_query, rows_to_insert)
        self.conn.commit()
        return cursor.rowcount

    def insert_videos_batch(self, video_data: List[Dict[str, Any]]) -> int:
        """Insert multiple videos in a single transaction."""
        cursor = self.conn.cursor()
        
        insert_query = """
            INSERT OR IGNORE INTO Videos (
                accession_number, patient_id, images_path, dicom_hash, saved_frames,
                region_spatial_format, region_data_type,
                region_location_min_x0, region_location_min_y0,
                region_location_max_x1, region_location_max_y1,
                photometric_interpretation, rows, columns, physical_delta_x,
                file_name, software_versions, manufacturer_model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        rows_to_insert = [
            (
                str(row.get('accession_number', '')),
                str(row.get('patient_id', '')),
                str(row.get('images_path', '')),
                str(row.get('dicom_hash', '')),
                row.get('saved_frames'),
                row.get('region_spatial_format'),
                row.get('region_data_type'),
                row.get('region_location_min_x0'),
                row.get('region_location_min_y0'),
                row.get('region_location_max_x1'),
                row.get('region_location_max_y1'),
                row.get('photometric_interpretation'),
                row.get('rows'),
                row.get('columns'),
                row.get('physical_delta_x'),
                row.get('file_name'),
                row.get('software_versions'),
                row.get('manufacturer_model_name')
            )
            for row in video_data
        ]
        
        cursor.executemany(insert_query, rows_to_insert)
        self.conn.commit()
        return cursor.rowcount

    def insert_study_cases_batch(self, study_data: List[Dict[str, Any]]) -> int:
        """Insert multiple study cases in a single transaction."""
        cursor = self.conn.cursor()
        
        insert_query = """
            INSERT OR REPLACE INTO StudyCases (
                accession_number, patient_id, study_laterality, study_date,
                has_malignant, has_benign, final_interpretation,
                findings, synoptic_report, bi_rads, modality, age_at_event,
                ethnicity, race, zipcode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        rows_to_insert = [
            (
                str(row.get('accession_number', '')),
                str(row.get('patient_id', '')),
                row.get('study_laterality'),
                row.get('study_date'),
                1 if row.get('has_malignant') else 0,
                1 if row.get('has_benign') else 0,
                row.get('final_interpretation'),
                row.get('findings'),
                row.get('synoptic_report'),
                row.get('bi_rads'),
                row.get('modality'),
                row.get('age_at_event'),
                row.get('ethnicity'),
                row.get('race'),
                row.get('zipcode')
            )
            for row in study_data
        ]
        
        cursor.executemany(insert_query, rows_to_insert)
        self.conn.commit()
        return cursor.rowcount

    def get_images_dataframe(self, where_clause: str = "", params: tuple = ()) -> pd.DataFrame:
        """
        Get images as a pandas DataFrame with optional filtering.

        Args:
            where_clause: Optional SQL WHERE clause (without 'WHERE' keyword)
            params: Parameters for the WHERE clause

        Returns:
            DataFrame with image data
        """
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

    def insert_pathology_batch(self, pathology_data: List[Dict[str, Any]]) -> int:
        """Insert multiple pathology/lesion records in a single transaction."""
        cursor = self.conn.cursor()

        insert_query = """
            INSERT OR REPLACE INTO Pathology (
                patient_id, accession_number, specimen_date,
                lesion_diag, synoptic_report, cancer_type
            ) VALUES (?, ?, ?, ?, ?, ?)
        """

        # Change this part - use snake_case
        rows_to_insert = [
            (
                str(row.get('patient_id', '')), 
                str(row.get('accession_number', '')), 
                str(row.get('specimen_date', '')),
                str(row.get('lesion_diag', '')),
                str(row.get('synoptic_report', '')),
                str(row.get('cancer_type', ''))
            )
            for row in pathology_data
        ]

        cursor.executemany(insert_query, rows_to_insert)
        self.conn.commit()
        return cursor.rowcount

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
