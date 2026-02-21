"""Shared utilities for P0 and P2 preprocessing pipelines.

Callers must set up sys.path before importing this module:
    sys.path.insert(0, str(_root))          # repo root (for tools/, config)
    sys.path.insert(0, str(_export))        # src/export/ (for ui_mask, etc.)
    sys.path.insert(0, str(_export / "config_processing"))
"""

import sqlite3
from pathlib import Path

import pandas as pd

from tools.storage_adapter import StorageClient
from export_configurable import ExportConfig


# ---------------------------------------------------------------------------
# Config -> SQL
# ---------------------------------------------------------------------------

def build_query(config: ExportConfig) -> str:
    """Build the Images+StudyCases SQL query from a structured ExportConfig."""
    conditions = [
        "i.crop_x IS NOT NULL",
        "i.crop_w IS NOT NULL",
        "i.crop_h IS NOT NULL",
        "i.image_name NOT IN (SELECT image_name FROM BadImages)",
    ]

    sf = config.scanner_filters
    if sf.allowed_scanners:
        names = ", ".join(f"'{s}'" for s in sf.allowed_scanners)
        conditions.append(f"i.manufacturer_model_name IN ({names})")
    elif sf.exclude_scanners:
        names = ", ".join(f"'{s}'" for s in sf.exclude_scanners)
        conditions.append(f"i.manufacturer_model_name NOT IN ({names})")

    stf = config.study_filters
    if stf.min_year is not None:
        conditions.append(f"s.date >= '{stf.min_year}-01-01'")
    if stf.max_year is not None:
        conditions.append(f"s.date <= '{stf.max_year}-12-31'")
    if stf.is_biopsy is not None:
        conditions.append(f"s.is_biopsy = {stf.is_biopsy}")
    if stf.exclude_unknown_label:
        conditions.append("(s.has_malignant != -1 OR s.has_malignant IS NULL)")

    imf = config.image_filters
    if imf.darkness_max is not None:
        conditions.append(f"(i.darkness IS NULL OR i.darkness <= {imf.darkness_max})")

    where = "\n    AND ".join(conditions)
    return f"""
    SELECT
        i.image_id,
        i.image_name,
        i.accession_number,
        i.patient_id,
        i.laterality,
        i.crop_x, i.crop_y, i.crop_w, i.crop_h,
        i.crop_aspect_ratio,
        i.us_polygon,
        i.debris_polygons,
        i.rows,
        i.columns,
        i.manufacturer_model_name,
        i.has_calipers,
        i.label,
        i.darkness,
        i.area,
        i.region_count,
        s.has_malignant,
        s.has_benign,
        s.valid,
        s.bi_rads,
        s.date
    FROM Images i
    LEFT JOIN StudyCases s ON i.accession_number = s.accession_number
    WHERE {where}
    """


# ---------------------------------------------------------------------------
# DB loading
# ---------------------------------------------------------------------------

def load_from_db(db_path: str, query: str) -> pd.DataFrame:
    """Execute a SQL query against the database and return results as a DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Loaded {len(df):,} images from DB")
    return df


# ---------------------------------------------------------------------------
# Image filters
# ---------------------------------------------------------------------------

def apply_image_filters(df: pd.DataFrame, config: ExportConfig) -> pd.DataFrame:
    """Apply image-level filters in Python after SQL load."""
    imf = config.image_filters

    if imf.allowed_areas:
        before = len(df)
        df = df[df["area"].isin(imf.allowed_areas) | df["area"].isna()].copy()
        print(f"  After area filter        : {len(df):,}  (-{before - len(df):,})")

    if imf.region_count_max:
        before = len(df)
        df = df[df["region_count"].fillna(1) <= imf.region_count_max].copy()
        print(f"  After region_count filter: {len(df):,}  (-{before - len(df):,})")

    if imf.aspect_ratio_min is not None and "crop_aspect_ratio" in df.columns:
        before = len(df)
        ar = df["crop_aspect_ratio"].fillna(1.0)
        df = df[ar.between(imf.aspect_ratio_min, imf.aspect_ratio_max)].copy()
        print(f"  After aspect_ratio filter: {len(df):,}  (-{before - len(df):,})")

    if imf.min_dimension is not None:
        before = len(df)
        ok = (df["crop_w"].fillna(0) >= imf.min_dimension) & (df["crop_h"].fillna(0) >= imf.min_dimension)
        df = df[ok].copy()
        print(f"  After min_dimension filter: {len(df):,}  (-{before - len(df):,})")

    return df


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def download_bytes(image_name: str, image_dir: str, storage: StorageClient) -> bytes:
    """Download image bytes from GCP or local filesystem."""
    if storage.is_gcp:
        blob_path = f"{image_dir}/{image_name}".replace("//", "/").lstrip("/")
        blob = storage._bucket.blob(blob_path)
        return blob.download_as_bytes()
    else:
        file_path = Path(image_dir) / image_name
        return file_path.read_bytes()
