"""
Labels database manager for CADBUSI.

Manages the cadbusi_labels.db SQLite database containing:
  - ImageLabels   — per-image classification labels
  - LesionLabels  — bounding box / mask annotations
  - CaliperLabels — caliper annotation labels

Provides two convenience functions for the bucket sync workflow:

    # Download latest from bucket and start reading / editing
    db = open_labels_db_from_bucket()
    df = db.get_image_labels_dataframe()
    db.close()

    # After making local edits, push back to the bucket
    # Raises StaleDBError if another developer pushed since your pull
    update_labels_db_in_bucket()
"""

import sqlite3
import os
import sys
import pandas as pd
from typing import Optional, List, Dict, Any

# Resolve paths so this module can be run from anywhere
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))          # repo root
_TOOLS_DIR = os.path.join(_ROOT_DIR, "tools")

sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _TOOLS_DIR)

from config import CONFIG
import storage_adapter as storage_mod
from storage_adapter import StorageClient, read_binary
from google.api_core import exceptions as gcp_exceptions


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StaleDBError(RuntimeError):
    """Raised when push_to_bucket() is rejected because the bucket was updated
    since the last pull_from_bucket() call."""


# ---------------------------------------------------------------------------
# LabelsDatabase
# ---------------------------------------------------------------------------

class LabelsDatabase:
    """Manages the cadbusi_labels.db SQLite database."""

    DEFAULT_DB_FILE = os.path.join(CONFIG["DATABASE_DIR"], "cadbusi_labels.db")
    DEFAULT_BUCKET_PATH = "databases/cadbusi_labels.db"

    def __init__(self, db_file: str = None, bucket_path: str = None):
        self.db_file = db_file or self.DEFAULT_DB_FILE
        self.bucket_path = bucket_path or self.DEFAULT_BUCKET_PATH
        self.conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self):
        """Open the SQLite connection and enable foreign keys."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_file)), exist_ok=True)
        self.db_file = os.path.abspath(self.db_file)
        self.conn = sqlite3.connect(self.db_file)
        self.conn.execute("PRAGMA foreign_keys = ON")
        return self.conn

    def close(self):
        """Close the SQLite connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_schema(self):
        """Create the labels database schema."""
        cursor = self.conn.cursor()

        # ImageLabels — per-image classification labels
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ImageLabels (
                dicom_hash   TEXT PRIMARY KEY,
                reject       INTEGER,
                only_normal  INTEGER,
                cyst         INTEGER,
                benign       INTEGER,
                malignant    INTEGER,
                quality      TEXT,
                version      TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # LesionLabels — bounding box + mask annotations per lesion
        # Internal FK: dicom_hash → ImageLabels.dicom_hash (enforced within this DB)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS LesionLabels (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                dicom_hash  TEXT,
                x1          INTEGER,
                y1          INTEGER,
                x2          INTEGER,
                y2          INTEGER,
                mask_image  TEXT,
                quality     TEXT,
                version     TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dicom_hash) REFERENCES ImageLabels(dicom_hash)
            )
        """)

        # CaliperLabels — per-image caliper annotation labels
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS CaliperLabels (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                dicom_hash       TEXT NOT NULL,
                has_calipers     INTEGER,
                caliper_points   TEXT,
                n_points         INTEGER,
                split            TEXT,
                bi_rads          TEXT,
                quality          TEXT,
                accession_number TEXT,
                version          TEXT,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_imagelabels_dicom   ON ImageLabels(dicom_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lesionlabels_dicom  ON LesionLabels(dicom_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_caliperlabels_dicom ON CaliperLabels(dicom_hash)")

        self.conn.commit()

    # ------------------------------------------------------------------
    # Private batch helper (mirrors DatabaseManager._batch_upsert_helper)
    # ------------------------------------------------------------------

    def _batch_upsert_helper(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        all_columns: List[str],
        unique_key: str,
        string_columns: List[str] = None,
        boolean_columns: List[str] = None,
        upsert: bool = False,
        update_only: bool = False,
    ) -> int:
        if not data:
            return 0

        string_columns = string_columns or []
        boolean_columns = boolean_columns or []

        cursor = self.conn.cursor()

        present_columns = [col for col in all_columns if col in data[0]]

        # Auto-migrate: add missing columns
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for col in present_columns:
            if col not in existing_columns:
                if col in boolean_columns:
                    col_type = "INTEGER DEFAULT 0"
                elif col in string_columns:
                    col_type = "TEXT"
                else:
                    col_type = "REAL"
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")
                print(f"Auto-added column '{col}' ({col_type}) to {table_name}")
        self.conn.commit()

        def _coerce(col, val):
            if col in boolean_columns:
                return 1 if val in ("T", True, 1, "1") else 0
            if col in string_columns:
                return str(val) if val is not None else ""
            return val

        if update_only:
            if unique_key not in present_columns:
                raise ValueError(f"update_only mode requires '{unique_key}' in data")
            update_cols = [col for col in present_columns if col != unique_key]
            if not update_cols:
                return 0
            set_clause = ", ".join([f"{col} = ?" for col in update_cols])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {unique_key} = ?"
            rows = [
                tuple(_coerce(c, row.get(c)) for c in update_cols)
                + (str(row.get(unique_key)) if row.get(unique_key) is not None else "",)
                for row in data
            ]
            cursor.executemany(query, rows)
            self.conn.commit()
            return cursor.rowcount

        placeholders = ", ".join(["?" for _ in present_columns])
        columns_str = ", ".join(present_columns)

        if upsert:
            update_cols = [col for col in present_columns if col != unique_key]
            update_str = ", ".join([f"{col} = excluded.{col}" for col in update_cols])
            query = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT({unique_key}) DO UPDATE SET {update_str}
            """
        else:
            query = f"""
                INSERT OR IGNORE INTO {table_name} ({columns_str})
                VALUES ({placeholders})
            """

        rows = [tuple(_coerce(c, row.get(c)) for c in present_columns) for row in data]
        cursor.executemany(query, rows)
        self.conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Insert methods
    # ------------------------------------------------------------------

    def insert_image_labels_batch(
        self, label_data: List[Dict[str, Any]], upsert: bool = True
    ) -> int:
        """Insert image classification labels."""
        return self._batch_upsert_helper(
            table_name="ImageLabels",
            data=label_data,
            all_columns=["dicom_hash", "reject", "only_normal", "cyst", "benign", "malignant", "quality", "version"],
            unique_key="dicom_hash",
            string_columns=["dicom_hash", "quality", "version"],
            boolean_columns=["reject", "only_normal", "cyst", "benign", "malignant"],
            upsert=upsert,
        )

    def insert_lesion_labels_batch(self, lesion_data: List[Dict[str, Any]]) -> int:
        """Insert lesion bounding box and mask annotations."""
        if not lesion_data:
            return 0
        cursor = self.conn.cursor()
        query = """
            INSERT INTO LesionLabels (dicom_hash, x1, y1, x2, y2, mask_image, quality, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                str(row.get("dicom_hash", "")),
                row.get("x1"),
                row.get("y1"),
                row.get("x2"),
                row.get("y2"),
                row.get("mask_image"),
                row.get("quality"),
                row.get("version"),
            )
            for row in lesion_data
        ]
        cursor.executemany(query, rows)
        self.conn.commit()
        return cursor.rowcount

    def insert_caliper_labels_batch(
        self, caliper_data: List[Dict[str, Any]], upsert: bool = True
    ) -> int:
        """Insert caliper annotation labels."""
        return self._batch_upsert_helper(
            table_name="CaliperLabels",
            data=caliper_data,
            all_columns=[
                "dicom_hash", "has_calipers", "caliper_points", "n_points",
                "split", "bi_rads", "quality", "accession_number", "version",
            ],
            unique_key="dicom_hash",
            string_columns=["dicom_hash", "caliper_points", "split", "bi_rads", "quality", "accession_number", "version"],
            boolean_columns=["has_calipers"],
            upsert=upsert,
        )

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def get_image_labels_dataframe(
        self, where_clause: str = "", params: tuple = ()
    ) -> pd.DataFrame:
        """Return ImageLabels as a pandas DataFrame."""
        query = "SELECT * FROM ImageLabels"
        if where_clause:
            query += f" WHERE {where_clause}"
        return pd.read_sql_query(query, self.conn, params=params)

    def get_lesion_labels_dataframe(
        self, where_clause: str = "", params: tuple = ()
    ) -> pd.DataFrame:
        """Return LesionLabels as a pandas DataFrame."""
        query = "SELECT * FROM LesionLabels"
        if where_clause:
            query += f" WHERE {where_clause}"
        return pd.read_sql_query(query, self.conn, params=params)

    def get_caliper_labels_dataframe(
        self, where_clause: str = "", params: tuple = ()
    ) -> pd.DataFrame:
        """Return CaliperLabels as a pandas DataFrame."""
        query = "SELECT * FROM CaliperLabels"
        if where_clause:
            query += f" WHERE {where_clause}"
        return pd.read_sql_query(query, self.conn, params=params)

    # ------------------------------------------------------------------
    # Bucket sync
    # ------------------------------------------------------------------

    @staticmethod
    def _gen_file(db_path: str) -> str:
        """Path of the sidecar file that stores the pulled GCS generation number."""
        return db_path + ".gen"

    def pull_from_bucket(self, local_path: str = None) -> str:
        """Download the latest labels DB from GCS bucket to local disk.

        Records the GCS object generation number in a sidecar file so that
        push_to_bucket() can use it for a safe conditional upload.

        Args:
            local_path: Destination file path. Defaults to self.db_file.

        Returns:
            The absolute path where the file was saved.
        """
        local_path = os.path.abspath(local_path or self.db_file)
        client = StorageClient.get_instance()
        if not client.is_gcp:
            raise RuntimeError(
                "Storage is not configured for GCP. "
                "Initialise StorageClient with a bucket_name before calling pull_from_bucket()."
            )

        blob = client._bucket.blob(self.bucket_path)
        blob.reload()  # fetch current metadata (generation, size, etc.)
        generation = blob.generation

        print(f"Downloading {self.bucket_path} (generation {generation}) -> {local_path} ...")
        data = blob.download_as_bytes()
        if data is None:
            raise FileNotFoundError(
                f"Labels DB not found in bucket at path: {self.bucket_path}"
            )

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)

        # Persist the generation number so push_to_bucket() can verify it later
        with open(self._gen_file(local_path), "w") as f:
            f.write(str(generation))

        print(f"Downloaded labels DB ({len(data):,} bytes) to {local_path}")
        return local_path

    def push_to_bucket(self, local_path: str = None):
        """Upload the local labels DB to GCS bucket, replacing the existing file.

        Uses a GCS conditional write (if_generation_match) to atomically verify
        that the bucket file has not changed since pull_from_bucket() was called.
        If another developer pushed in the meantime, this raises StaleDBError.

        Args:
            local_path: Source file path. Defaults to self.db_file.

        Raises:
            StaleDBError: The bucket was updated after your pull. Pull again first.
            FileNotFoundError: Local DB or sidecar .gen file not found.
        """
        local_path = os.path.abspath(local_path or self.db_file)
        gen_file = self._gen_file(local_path)

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local labels DB not found: {local_path}")

        client = StorageClient.get_instance()
        if not client.is_gcp:
            raise RuntimeError(
                "Storage is not configured for GCP. "
                "Initialise StorageClient with a bucket_name before calling push_to_bucket()."
            )

        # Determine the generation to match
        if os.path.exists(gen_file):
            with open(gen_file) as f:
                expected_generation = int(f.read().strip())
        else:
            # No sidecar — check if the object exists in the bucket
            blob_check = client._bucket.blob(self.bucket_path)
            blob_check.reload() if blob_check.exists() else None
            if blob_check.exists():
                raise FileNotFoundError(
                    f"No .gen sidecar found at {gen_file}. "
                    "You must call pull_from_bucket() before push_to_bucket() "
                    "to avoid overwriting another developer's changes."
                )
            # First-ever upload — require the object not to exist yet
            expected_generation = 0

        blob = client._bucket.blob(self.bucket_path)
        print(f"Uploading {local_path} -> {self.bucket_path} (if_generation_match={expected_generation}) ...")

        try:
            blob.upload_from_filename(local_path, if_generation_match=expected_generation)
        except gcp_exceptions.PreconditionFailed:
            blob.reload()
            raise StaleDBError(
                f"Push rejected: the labels DB in the bucket was updated (now generation "
                f"{blob.generation}) since you pulled (generation {expected_generation}).\n"
                f"Pull the latest version, re-apply your changes, then push again."
            )

        # Clean up sidecar now that the push succeeded
        if os.path.exists(gen_file):
            os.remove(gen_file)

        print(f"Uploaded labels DB to gs://{CONFIG['storage']['bucket_name']}/{self.bucket_path}")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def open_labels_db_from_bucket(
    local_path: str = None, bucket_path: str = None
) -> LabelsDatabase:
    """Download the latest labels DB from the bucket and return a connected instance.

    Usage::

        db = open_labels_db_from_bucket()
        df = db.get_image_labels_dataframe()
        db.close()

    Args:
        local_path:  Where to save the downloaded DB locally.
                     Defaults to LabelsDatabase.DEFAULT_DB_FILE.
        bucket_path: Path within the GCS bucket.
                     Defaults to LabelsDatabase.DEFAULT_BUCKET_PATH.

    Returns:
        A connected :class:`LabelsDatabase` instance ready to query.
    """
    db = LabelsDatabase(db_file=local_path, bucket_path=bucket_path)
    db.pull_from_bucket()
    db.connect()
    return db


def update_labels_db_in_bucket(
    local_path: str = None, bucket_path: str = None
):
    """Upload the local labels DB to the bucket, replacing the existing version.

    Typical workflow::

        # 1. Download and edit
        db = open_labels_db_from_bucket()
        db.insert_image_labels_batch([...])
        db.close()

        # 2. Push changes back
        update_labels_db_in_bucket()

    Args:
        local_path:  Path to the local DB file to upload.
                     Defaults to LabelsDatabase.DEFAULT_DB_FILE.
        bucket_path: Destination path within the GCS bucket.
                     Defaults to LabelsDatabase.DEFAULT_BUCKET_PATH.
    """
    db = LabelsDatabase(db_file=local_path, bucket_path=bucket_path)
    db.push_to_bucket()
