"""Seed frame_lesion_views from the already-seeded lesion_annotations.

For every source `lesion_id` collapsed into a `lesion_annotations` row, find
its `image_name` in `cadbusi.Lesions`, look up the matching `dicom_hash` in
`cadbusi.Images`, and insert one row in `frame_lesion_views` linking the
frame to the lesion.

Idempotent: relies on `UNIQUE(dicom_hash, lesion_annot_id)`.

Usage:
    python label_migration/seed_frame_lesion_views.py \
        --source label_migration/cadbusi_2026_4_20.db \
        --labels label_migration/labeled_cadbusi.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.DB_processing.labels_database import LabelsDatabase


def _connect_ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def build_lesion_id_to_annot_id(labels_db: str) -> dict:
    """Read lesion_annotations and invert source_lesion_ids -> lesion_annot_id."""
    conn = sqlite3.connect(labels_db)
    rows = conn.execute(
        "SELECT lesion_annot_id, source_lesion_ids FROM lesion_annotations"
    ).fetchall()
    conn.close()
    out = {}
    for annot_id, srcs in rows:
        try:
            ids = json.loads(srcs or "[]")
        except json.JSONDecodeError:
            continue
        for lid in ids:
            out[int(lid)] = annot_id
    return out


def collect_links(source_db: str, lid_to_annot: dict):
    """For every Lesions row whose lesion_id maps to an annot_id, return (dicom_hash, annot_id)."""
    conn = _connect_ro(source_db)
    try:
        rows = conn.execute("""
            SELECT L.lesion_id, I.dicom_hash
            FROM Lesions L
            JOIN Images I ON I.image_name = L.image_name
                         AND I.accession_number = L.accession_number
            WHERE I.dicom_hash IS NOT NULL
        """).fetchall()
    finally:
        conn.close()
    pairs = []
    for lid, dh in rows:
        annot = lid_to_annot.get(int(lid))
        if annot is not None:
            pairs.append((dh, annot))
    return pairs


INSERT_SQL = """
INSERT OR IGNORE INTO frame_lesion_views (dicom_hash, lesion_annot_id)
VALUES (?, ?)
"""


def seed(source_db: str, labels_db: str, dry_run: bool = False):
    print(f"Source DB: {source_db}")
    print(f"Labels DB: {labels_db}")

    if not Path(source_db).exists():
        raise SystemExit(f"Source DB not found: {source_db}")
    if not Path(labels_db).exists():
        raise SystemExit(f"Labels DB not found: {labels_db}")

    db = LabelsDatabase(db_file=labels_db)
    db.connect()
    db.create_schema()
    db.close()

    print("Building lesion_id -> lesion_annot_id map...")
    lid_to_annot = build_lesion_id_to_annot_id(labels_db)
    print(f"  mapped lesion_ids: {len(lid_to_annot):,}")
    if not lid_to_annot:
        raise SystemExit("lesion_annotations is empty — run seed_lesion_annotations.py first.")

    print("Joining Lesions -> Images to collect (dicom_hash, lesion_annot_id) pairs...")
    pairs = collect_links(source_db, lid_to_annot)
    distinct_pairs = set(pairs)
    print(f"  raw pairs: {len(pairs):,}   distinct (dicom_hash, lesion_annot_id): {len(distinct_pairs):,}")

    by_lesion = defaultdict(int)
    for _, a in distinct_pairs:
        by_lesion[a] += 1
    if by_lesion:
        sample = sorted(by_lesion.values(), reverse=True)
        print(f"  frames per lesion: max={sample[0]}, p50={sample[len(sample)//2]}")

    if dry_run:
        print("Dry run — not writing.")
        return

    conn = sqlite3.connect(labels_db)
    cur = conn.cursor()
    cur.executemany(INSERT_SQL, list(distinct_pairs))
    inserted = cur.rowcount
    conn.commit()
    conn.close()
    print(f"Inserted (or ignored if existing): {inserted:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seed(
        source_db=str(Path(args.source).expanduser().resolve()),
        labels_db=str(Path(args.labels).expanduser().resolve()),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
