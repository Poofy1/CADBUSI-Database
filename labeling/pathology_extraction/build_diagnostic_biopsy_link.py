"""Build the full diagnostic US ↔ pathology link table.

Tristan's existing pipeline links each Pathology row to the most-recent-prior
US exam for the patient. This misses cases where a patient had multiple prior
US exams — only one gets the pathology pointer.

This script reconstructs the link with the CADBUSI paper's rules WITHOUT the
"most recent only" restriction: pathology propagates to ALL US exams for the
patient within the 8-month window matching on laterality.

Inputs:
  - src.StudyCases (patient_id, accession_number, date, study_laterality, is_biopsy)
  - src.Pathology (path_id, patient_id, accession_number, date, cancer_type, lesion_diag, synoptic_report)
  - main.pathology_extracted (laterality from rad_pathology_txt LLM extraction)
  - main.pathology_synoptic_extracted (laterality from synoptic_report LLM extraction)

Output: main.diagnostic_biopsy_link
  Grain: one row per (diag_accession, path_id, laterality)
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3
import pandas as pd
from bus_data import ManifestDB

MANIFEST = '/home/jbaggett/BUS_framework/data/registry/bus_manifest_v3.db'

WINDOW_DAYS = 240  # 8 months per CADBUSI paper

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS diagnostic_biopsy_link (
    diag_accession      TEXT NOT NULL,
    patient_id          TEXT NOT NULL,
    us_date             TEXT,
    study_laterality    TEXT,
    path_id             INTEGER NOT NULL,
    path_accession      TEXT,
    path_date           TEXT,
    path_laterality     TEXT,
    cancer_type         TEXT,
    primary_diagnosis   TEXT,
    days_gap            INTEGER,
    laterality_match    TEXT,
    link_source         TEXT,
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (diag_accession, path_id)
);
CREATE INDEX IF NOT EXISTS idx_dbl_patient  ON diagnostic_biopsy_link(patient_id);
CREATE INDEX IF NOT EXISTS idx_dbl_path     ON diagnostic_biopsy_link(path_id);
CREATE INDEX IF NOT EXISTS idx_dbl_cancer   ON diagnostic_biopsy_link(cancer_type);
"""


def _normalize_laterality(val):
    if val is None: return None
    v = str(val).upper().strip()
    if v in ('LEFT', 'L'): return 'LEFT'
    if v in ('RIGHT', 'R'): return 'RIGHT'
    if v in ('BILATERAL', 'BOTH'): return 'BILATERAL'
    if v in ('NOT_SPECIFIED', 'UNKNOWN', 'NAN', ''): return None
    return v


def _laterality_match(us_lat, path_lat):
    """Return match category. Accept BILATERAL US for either side; unknown = permissive."""
    u = _normalize_laterality(us_lat)
    p = _normalize_laterality(path_lat)
    if p is None and u is None:        return 'UNKNOWN_BOTH'
    if p is None:                       return 'UNKNOWN_PATH'
    if u == 'BILATERAL':                return 'BILATERAL_US'
    if u == p:                          return 'EXACT'
    if u is None:                       return 'UNKNOWN_US'
    return 'MISMATCH'


def main():
    db = ManifestDB()

    # 1. Pull all Pathology rows with their patient_id + date + (Tristan's) accession
    print("Loading Pathology rows...")
    path = db.query_df("""
        SELECT path_id, patient_id, accession_number AS path_accession,
               date AS path_date, cancer_type, lesion_diag
        FROM src.Pathology
    """)
    print(f"  {len(path):,} pathology rows, {path['patient_id'].nunique():,} unique patients")

    # 2. Pull LLM-extracted laterality from phase1 + phase2 tables. Join by accession
    conn = sqlite3.connect(MANIFEST)
    p1_lat = pd.read_sql("""
        SELECT accession_number, laterality FROM pathology_extracted
        WHERE extraction_error IS NULL AND laterality IS NOT NULL
          AND laterality NOT IN ('NOT_SPECIFIED','')
    """, conn)
    p2_lat = pd.read_sql("""
        SELECT path_id, laterality, primary_diagnosis FROM pathology_synoptic_extracted
        WHERE extraction_error IS NULL AND laterality IS NOT NULL
          AND laterality NOT IN ('NOT_SPECIFIED','')
    """, conn)
    conn.close()
    print(f"  Phase1 laterality (by accession): {len(p1_lat):,} rows")
    print(f"  Phase2 laterality (by path_id):    {len(p2_lat):,} rows")

    # Merge lateralities. Phase 2 is per-path_id (preferred — one row per specimen).
    # Phase 1 is per-accession (one laterality per whole accession's addendum).
    path = path.merge(p2_lat, on='path_id', how='left')  # brings p2 laterality + primary_diagnosis
    # Fall-back: if p2 missing, use p1 laterality by accession
    path = path.merge(p1_lat.rename(columns={'laterality': 'p1_laterality'}),
                       left_on='path_accession', right_on='accession_number', how='left')
    path.drop(columns=['accession_number'], inplace=True, errors='ignore')
    path['path_laterality'] = path['laterality'].fillna(path['p1_laterality'])
    path.drop(columns=['laterality', 'p1_laterality'], inplace=True)

    # 3. Pull StudyCases: diagnostic US exams (is_biopsy=0) with their date + laterality
    print("\nLoading StudyCases US exams...")
    sc = db.query_df("""
        SELECT accession_number AS diag_accession, patient_id, date AS us_date,
               study_laterality
        FROM src.StudyCases
        WHERE is_biopsy = 0 AND date IS NOT NULL
    """)
    print(f"  {len(sc):,} diagnostic exams, {sc['patient_id'].nunique():,} unique patients")

    # 4. Cross-join by patient. For each (patient, path_date), find all prior US within window.
    print(f"\nBuilding links (window = {WINDOW_DAYS} days)...")
    sc['us_date'] = pd.to_datetime(sc['us_date'], errors='coerce')
    path['path_date'] = pd.to_datetime(path['path_date'], errors='coerce')
    sc = sc[sc['us_date'].notna()]
    path = path[path['path_date'].notna()]

    # Merge by patient_id — can be large, keep it lean
    merged = path.merge(sc, on='patient_id', how='inner')
    print(f"  Patient merge: {len(merged):,} candidate pairs")

    merged['days_gap'] = (merged['path_date'] - merged['us_date']).dt.days
    # Window filter: US must precede path by at most WINDOW_DAYS and after path is NOT allowed
    in_window = merged[(merged['days_gap'] >= 0) & (merged['days_gap'] <= WINDOW_DAYS)].copy()
    print(f"  After ±window: {len(in_window):,} pairs ({in_window['diag_accession'].nunique():,} unique diag accessions)")

    # Laterality match
    in_window['laterality_match'] = [
        _laterality_match(u, p) for u, p in zip(in_window['study_laterality'], in_window['path_laterality'])
    ]
    print("  Laterality match distribution (pre-filter):")
    print(in_window['laterality_match'].value_counts().to_string())

    # Keep only compatible matches
    compatible = in_window[in_window['laterality_match'].isin(
        ['EXACT', 'BILATERAL_US', 'UNKNOWN_PATH', 'UNKNOWN_US', 'UNKNOWN_BOTH']
    )].copy()
    print(f"\n  After laterality filter: {len(compatible):,} compatible links")

    # Source flag: TRISTAN if this diag_accession matches the Pathology.accession_number
    compatible['link_source'] = compatible.apply(
        lambda r: 'TRISTAN' if r['diag_accession'] == r['path_accession'] else 'REBUILT',
        axis=1
    )

    # Write to DB
    conn = sqlite3.connect(MANIFEST)
    conn.executescript(SCHEMA_SQL)
    conn.execute("DELETE FROM diagnostic_biopsy_link")
    compatible_out = compatible.rename(columns={'study_laterality': 'study_laterality'})[[
        'diag_accession', 'patient_id', 'us_date', 'study_laterality',
        'path_id', 'path_accession', 'path_date', 'path_laterality',
        'cancer_type', 'primary_diagnosis', 'days_gap', 'laterality_match', 'link_source',
    ]].copy()
    compatible_out['us_date']   = compatible_out['us_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    compatible_out['path_date'] = compatible_out['path_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    compatible_out.to_sql('diagnostic_biopsy_link', conn, if_exists='append', index=False)
    conn.close()
    print(f"\n✓ Wrote {len(compatible_out):,} links to diagnostic_biopsy_link")

    # Summary stats
    print("\n=== Final link summary ===")
    print(f"  Unique diagnostic accessions with pathology: {compatible_out['diag_accession'].nunique():,}")
    print(f"  Unique pathology rows used:                  {compatible_out['path_id'].nunique():,}")
    print(f"  Avg links per diag accession:                {len(compatible_out)/compatible_out['diag_accession'].nunique():.2f}")
    print()
    print("Link source breakdown:")
    print(compatible_out['link_source'].value_counts().to_string())
    print()
    print("Laterality match breakdown:")
    print(compatible_out['laterality_match'].value_counts().to_string())


if __name__ == '__main__':
    main()
