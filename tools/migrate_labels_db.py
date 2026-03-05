"""
Temporary migration script: splits cadbusi_full.db into:
  - cadbusi.db        — main database (all tables except ImageLabels/LesionLabels/CaliperLabels)
  - labeled_cadbusi.db — labels-only database (ImageLabels, LesionLabels, CaliperLabels)

Usage:
    python tools/migrate_labels_db.py
"""

import shutil
import sqlite3
import os
import sys

SRC_DB  = r"C:\Users\Tristan\Desktop\cadbusi_full.db"
MAIN_DB = r"C:\Users\Tristan\Desktop\cadbusi.db"
LBLS_DB = r"C:\Users\Tristan\Desktop\labeled_cadbusi.db"

LABEL_TABLES = ["ImageLabels", "LesionLabels", "CaliperLabels"]


def row_count(conn, table):
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except Exception:
        return 0


# ------------------------------------------------------------------
# 1. Build cadbusi.db — copy full db, then drop the three label tables
# ------------------------------------------------------------------
print(f"Copying {SRC_DB} -> {MAIN_DB} ...")
shutil.copy2(SRC_DB, MAIN_DB)

conn = sqlite3.connect(MAIN_DB)
conn.execute("PRAGMA foreign_keys = OFF")

for table in LABEL_TABLES:
    n = row_count(conn, table)
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    print(f"  Dropped {table} ({n:,} rows)")

conn.commit()
conn.execute("VACUUM")
conn.close()
print(f"  cadbusi.db ready.\n")


# ------------------------------------------------------------------
# 2. Build labeled_cadbusi.db — create schema, copy label tables from full db
# ------------------------------------------------------------------
if os.path.exists(LBLS_DB):
    os.remove(LBLS_DB)

conn = sqlite3.connect(LBLS_DB)
conn.execute("PRAGMA foreign_keys = ON")

# Create schema (mirrors LabelsDatabase.create_schema)
conn.executescript("""
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
    );

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
    );

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
    );

    CREATE INDEX IF NOT EXISTS idx_imagelabels_dicom   ON ImageLabels(dicom_hash);
    CREATE INDEX IF NOT EXISTS idx_lesionlabels_dicom  ON LesionLabels(dicom_hash);
    CREATE INDEX IF NOT EXISTS idx_caliperlabels_dicom ON CaliperLabels(dicom_hash);
""")

# Attach source and copy each label table
conn.execute("PRAGMA foreign_keys = OFF")  # disable during bulk copy
conn.execute("ATTACH DATABASE ? AS src", (SRC_DB,))

for table in LABEL_TABLES:
    src_n = conn.execute(f"SELECT COUNT(*) FROM src.{table}").fetchone()[0]

    # Get columns present in source (handles any extra columns gracefully)
    cols_info = conn.execute(f"PRAGMA src.table_info({table})").fetchall()
    src_cols  = [row[1] for row in cols_info]

    dst_cols_info = conn.execute(f"PRAGMA main.table_info({table})").fetchall()
    dst_cols = {row[1] for row in dst_cols_info}

    # Copy only columns that exist in both source and destination (skip id autoincrement)
    shared = [c for c in src_cols if c in dst_cols and c != "id"]
    cols_str = ", ".join(shared)

    conn.execute(f"INSERT INTO main.{table} ({cols_str}) SELECT {cols_str} FROM src.{table}")
    dst_n = row_count(conn, table)
    print(f"  Copied {table}: {src_n:,} -> {dst_n:,} rows")

conn.commit()
conn.execute("DETACH DATABASE src")
conn.close()
print(f"\n  labeled_cadbusi.db ready.")

print("\nDone.")
print(f"  Main DB:   {MAIN_DB}")
print(f"  Labels DB: {LBLS_DB}")
