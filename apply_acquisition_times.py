"""
One-shot temp script: apply `acquisition_times.csv` (produced by
gather_acquisition_times.py) into a copy of the database.

  IN:  data/cadbusi_2026_4_20.db   (untouched)
       acquisition_times.csv       (dicom_hash, acquisition_time)
  OUT: data/cadbusi_2026_4_28.db   (Images + Videos populated)

Adds the `acquisition_time TEXT` column to Images and Videos if missing,
then UPDATEs rows by dicom_hash. Empty/blank times in the CSV are skipped
so NULL keeps meaning "no data".
"""
import os
import shutil
import sqlite3
import sys

import pandas as pd

SRC_DB = "data/cadbusi_2026_4_20.db"
DST_DB = "data/cadbusi_2026_4_28.db"
CSV    = "acquisition_times.csv"


def ensure_column(conn, table):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if "acquisition_time" not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN acquisition_time TEXT")
        print(f"  added acquisition_time column to {table}")


def apply_to_table(conn, table, rows):
    """rows: list of (acquisition_time, dicom_hash) tuples."""
    cur = conn.cursor()
    cur.executemany(
        f"UPDATE {table} SET acquisition_time = ? WHERE dicom_hash = ?",
        rows,
    )
    conn.commit()
    return cur.rowcount


def main():
    if not os.path.exists(SRC_DB):
        sys.exit(f"Source DB not found: {SRC_DB}")
    if not os.path.exists(CSV):
        sys.exit(f"CSV not found: {CSV}")

    os.makedirs(os.path.dirname(DST_DB), exist_ok=True)
    print(f"Copying {SRC_DB} -> {DST_DB} ...")
    shutil.copyfile(SRC_DB, DST_DB)

    df = pd.read_csv(CSV, dtype={"dicom_hash": str, "acquisition_time": str})
    df["acquisition_time"] = df["acquisition_time"].fillna("").str.strip()
    before = len(df)
    df = df[df["acquisition_time"] != ""]
    df = df.drop_duplicates(subset="dicom_hash")
    print(f"Loaded {before:,} CSV rows; {len(df):,} have non-empty acquisition_time.")

    rows = list(zip(df["acquisition_time"], df["dicom_hash"]))

    conn = sqlite3.connect(DST_DB)
    try:
        ensure_column(conn, "Images")
        ensure_column(conn, "Videos")

        n_img = apply_to_table(conn, "Images", rows)
        n_vid = apply_to_table(conn, "Videos", rows)
        print(f"Updated {n_img:,} Images rows, {n_vid:,} Videos rows.")

        for table in ("Images", "Videos"):
            total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            filled = conn.execute(
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE acquisition_time IS NOT NULL AND acquisition_time != ''"
            ).fetchone()[0]
            pct = (filled / total * 100) if total else 0.0
            print(f"  {table}: {filled:,}/{total:,} have acquisition_time ({pct:.2f}%)")
    finally:
        conn.close()

    print(f"Done. Output: {DST_DB}")


if __name__ == "__main__":
    main()
