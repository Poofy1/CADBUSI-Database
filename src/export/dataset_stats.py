#!/usr/bin/env python3
"""Dataset statistics: shows how many images each filter removes and final split/label counts.

Run from src/export/ (or anywhere — uses absolute paths).

Usage:
    python dataset_stats.py
    python dataset_stats.py --db ../../data/cadbusi.db --dataset configs/P2.yaml
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

_export = Path(__file__).resolve().parent
_root = _export.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_export))
sys.path.insert(0, str(_export / "config_processing"))

from export_configurable import ExportConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Base query: everything except BadImages — no other filters
# ---------------------------------------------------------------------------

BASE_QUERY = """
SELECT
    i.image_id,
    i.image_name,
    i.accession_number,
    i.patient_id,
    i.manufacturer_model_name,
    i.darkness,
    i.area,
    i.region_count,
    i.crop_x, i.crop_w, i.crop_h,
    i.crop_aspect_ratio,
    i.has_calipers,
    s.has_malignant,
    s.is_biopsy,
    s.valid,
    CAST(strftime('%Y', s.date) AS INTEGER) AS year
FROM Images i
LEFT JOIN StudyCases s ON i.accession_number = s.accession_number
WHERE i.image_name NOT IN (SELECT image_name FROM BadImages)
"""


def load_all(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(BASE_QUERY, conn)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Filter funnel
# ---------------------------------------------------------------------------

# Human-readable labels for BadImages exclusion_reason values.
_REASON_LABELS: dict[str, str] = {
    "non-breast area":                      "Non-breast area",
    "multiple regions":                     "Multiple ultrasound regions",
    "too dark (>75)":                       "Darkness > 75",
    "unknown laterality (bilateral study)": "Unknown laterality",
    "image does not exist":                 "Image not on disk",
    "bad aspect ratio (0.5-4.0)":           "Aspect ratio outside [0.5, 4.0]",
    "too small (<200px)":                   "Crop dimension < 200 px",
}


def load_funnel_prefix(db_path: str) -> tuple[int, list[tuple[str, int, int]]]:
    """Return (raw_total, rows) for each BadImages exclusion reason."""
    conn = sqlite3.connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM Images").fetchone()[0]
    bad_rows = conn.execute(
        "SELECT exclusion_reason, COUNT(*) FROM BadImages "
        "GROUP BY exclusion_reason ORDER BY COUNT(*) DESC"
    ).fetchall()
    conn.close()

    rows: list[tuple[str, int, int]] = []
    remaining = total
    for reason, cnt in bad_rows:
        remaining -= cnt
        label = _REASON_LABELS.get(reason, reason)
        rows.append((label, cnt, remaining))
    return total, rows


def run_funnel(df: pd.DataFrame, config: ExportConfig) -> list[tuple[str, int, int]]:
    """Apply each filter step by step. Returns list of (label, n_removed, n_remaining)."""
    rows: list[tuple[str, int, int]] = []

    def step(label: str, mask_keep):
        nonlocal df
        n_before = len(df)
        df = df[mask_keep].copy()
        removed = n_before - len(df)
        rows.append((label, removed, len(df)))

    # --- scanner allowlist ---
    sf = config.scanner_filters
    if sf.allowed_scanners:
        step(
            f"Scanner allowlist ({len(sf.allowed_scanners)} models)",
            df["manufacturer_model_name"].isin(sf.allowed_scanners),
        )
    elif sf.exclude_scanners:
        step(
            "Scanner denylist",
            ~df["manufacturer_model_name"].isin(sf.exclude_scanners),
        )

    # --- study filters ---
    stf = config.study_filters
    if stf.is_biopsy is not None:
        step(
            "Non-biopsy exams only",
            df["is_biopsy"] == stf.is_biopsy,
        )
    if stf.min_year is not None:
        step(
            f"Acquisition year >= {stf.min_year}",
            df["year"].fillna(0) >= stf.min_year,
        )
    if stf.exclude_unknown_label:
        step(
            "Known label only",
            df["has_malignant"].isin([0, 1]),
        )

    # --- image filters ---
    # Darkness, area, region count, min dimension, and aspect ratio are already
    # represented by their BadImages exclusion_reason rows in the funnel prefix,
    # so only calipers (which has no corresponding BadImages reason) is included here.
    imf = config.image_filters
    if imf.exclude_calipers:
        step(
            "No measurement calipers",
            df["has_calipers"].isna() | (df["has_calipers"] == 0),
        )

    return rows, df


# ---------------------------------------------------------------------------
# Gold masks
# ---------------------------------------------------------------------------

def load_gold_masks(db_path: str, config: ExportConfig) -> pd.DataFrame:
    """Query LesionLabels with the same study/scanner filters as the main dataset."""
    conditions = [
        "ll.mask_image IS NOT NULL",
        "ll.mask_image != ''",
        "s.has_malignant IN (0, 1)",
        "i.image_name NOT IN (SELECT image_name FROM BadImages)",
    ]

    stf = config.study_filters
    if stf.is_biopsy is not None:
        conditions.append(f"s.is_biopsy = {stf.is_biopsy}")
    if stf.min_year is not None:
        conditions.append(f"s.date >= '{stf.min_year}-01-01'")

    sf = config.scanner_filters
    if sf.allowed_scanners:
        names = ", ".join(f"'{s}'" for s in sf.allowed_scanners)
        conditions.append(f"i.manufacturer_model_name IN ({names})")

    where = "\n    AND ".join(conditions)
    query = f"""
    SELECT
        ll.id AS lesion_id,
        ll.quality,
        i.image_name,
        i.patient_id,
        i.accession_number,
        s.has_malignant,
        s.valid
    FROM LesionLabels ll
    JOIN Images i ON ll.dicom_hash = i.dicom_hash
    JOIN StudyCases s ON i.accession_number = s.accession_number
    WHERE {where}
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def print_gold_masks(db_path: str, config: ExportConfig):
    df = load_gold_masks(db_path, config)

    split_map = {0: "train", 1: "valid", 2: "test"}
    df["split"] = df["valid"].apply(lambda v: split_map.get(int(v) if pd.notna(v) else 0, "train"))

    print()
    print("GOLD MASK ANNOTATIONS")
    print("-" * 50)
    print(f"  {'Split':<8} {'Malignant':>10} {'Benign':>8} {'Total':>8}")
    print("-" * 50)
    for split in ["train", "valid", "test"]:
        sub = df[df["split"] == split]
        mal = int((sub["has_malignant"] == 1).sum())
        ben = int((sub["has_malignant"] == 0).sum())
        print(f"  {split:<8} {mal:>10,} {ben:>8,} {len(sub):>8,}")
    mal = int((df["has_malignant"] == 1).sum())
    ben = int((df["has_malignant"] == 0).sum())
    print("-" * 50)
    print(f"  {'Total':<8} {mal:>10,} {ben:>8,} {len(df):>8,}")
    print()
    print(f"  Unique images   : {df['image_name'].nunique():,}")
    print(f"  Unique studies  : {df['accession_number'].nunique():,}")
    print(f"  Unique patients : {df['patient_id'].nunique():,}")
    if df["quality"].notna().any():
        print()
        print("  Quality breakdown:")
        for q, cnt in df["quality"].value_counts().sort_index().items():
            print(f"    {q:<20} {cnt:>8,}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_funnel(rows: list[tuple[str, int, int]], total: int):
    print()
    print("=" * 68)
    print("FILTER FUNNEL  (sorted by images removed, largest first)")
    print("=" * 68)
    print(f"  {'Step':<46} {'Removed':>9} {'Remaining':>9}")
    print("-" * 68)
    print(f"  {'All images':<46} {'':>9} {total:>9,}")
    for label, removed, remaining in rows:
        print(f"  {label:<46} {removed:>9,} {remaining:>9,}")
    print("=" * 68)
    final = rows[-1][2]
    print(f"  Total removed: {total - final:,} of {total:,} ({100*(total-final)/total:.1f}%)")


def print_splits(df: pd.DataFrame):
    split_map = {0: "train", 1: "valid", 2: "test"}
    df = df.copy()
    df["split"] = df["valid"].apply(lambda v: split_map.get(int(v) if pd.notna(v) else 0, "train"))

    print()
    print("DATASET COMPOSITION")
    print("-" * 50)
    print(f"  {'Split':<8} {'Malignant':>10} {'Benign':>8} {'Total':>8}")
    print("-" * 50)
    for split in ["train", "valid", "test"]:
        sub = df[df["split"] == split]
        mal = int((sub["has_malignant"] == 1).sum())
        ben = int((sub["has_malignant"] == 0).sum())
        print(f"  {split:<8} {mal:>10,} {ben:>8,} {len(sub):>8,}")
    mal = int((df["has_malignant"] == 1).sum())
    ben = int((df["has_malignant"] == 0).sum())
    print("-" * 50)
    print(f"  {'Total':<8} {mal:>10,} {ben:>8,} {len(df):>8,}")
    print()
    n_studies = df["accession_number"].nunique()
    n_patients = df["patient_id"].nunique()
    print(f"  Unique studies  : {n_studies:,}")
    print(f"  Unique patients : {n_patients:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dataset filter funnel statistics")
    parser.add_argument(
        "--db",
        default=str(_root / "data" / "cadbusi.db"),
        help="Path to cadbusi.db",
    )
    parser.add_argument(
        "--dataset",
        default=str(_export / "configs" / "P2.yaml"),
        help="Path to dataset YAML config (default: configs/P2.yaml)",
    )
    args = parser.parse_args()

    config = ExportConfig.from_yaml(Path(args.dataset))
    print(f"Config  : {config.name}")
    print(f"DB      : {args.db}")

    print("\nLoading all images ...")
    df = load_all(args.db)
    print(f"  {len(df):,} images (BadImages excluded)")

    raw_total, prefix_rows = load_funnel_prefix(args.db)
    funnel_rows, df_filtered = run_funnel(df, config)

    # Combine, sort by removed DESC, recalculate running remaining
    combined = prefix_rows + funnel_rows
    combined.sort(key=lambda x: x[1], reverse=True)

    remaining = raw_total
    sorted_funnel: list[tuple[str, int, int]] = []
    for label, removed, _ in combined:
        remaining -= removed
        sorted_funnel.append((label, removed, remaining))

    print_funnel(sorted_funnel, total=raw_total)
    print_splits(df_filtered)
    print_gold_masks(args.db, config)


if __name__ == "__main__":
    main()
