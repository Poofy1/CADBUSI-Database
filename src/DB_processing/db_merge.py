"""Merge two cadbusi.db files via INSERT OR IGNORE per table."""
import os
import sqlite3


TABLES = [
    'StudyCases',
    'Images',
    'Videos',
    'Pathology',
    'Lesions',
    'BadImages',
    'CaliperPairs',
    'RegionLabels',
]


def merge_databases(src_path, dest_path):
    """Copy every row from each table in SRC into DEST using INSERT OR IGNORE,
    preserving existing DEST rows on conflict. Reports rows added per table."""
    src_path = os.path.abspath(src_path)
    dest_path = os.path.abspath(dest_path)
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"SRC database not found: {src_path}")
    if not os.path.exists(dest_path):
        raise FileNotFoundError(f"DEST database not found: {dest_path}")

    print(f"Merging {src_path}")
    print(f"   into {dest_path}")
    print()

    conn = sqlite3.connect(dest_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"ATTACH DATABASE '{src_path}' AS src")

    try:
        for tbl in TABLES:
            try:
                before = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                conn.execute(f"INSERT OR IGNORE INTO {tbl} SELECT * FROM src.{tbl}")
                after = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                print(f"  {tbl:<14} +{after - before:>7,} rows (now {after:>10,})")
            except sqlite3.Error as e:
                print(f"  {tbl:<14} SKIPPED ({e})")
        conn.commit()
    finally:
        conn.execute("DETACH DATABASE src")
        conn.close()

    print("\nMerge complete.")
    print("Reminder: copy `images/` and `videos/` from the source DB's directory")
    print("into the destination's directory so the merged DB rows still point at")
    print("files that exist on disk.")
