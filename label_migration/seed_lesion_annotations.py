"""Seed lesion_annotations from cadbusi.Lesions.

Dedupe rule (per Tristan_Tasks_04_22_26.md / MASK_SAVE_SPEC.md §11):
  - Rows with both `clock` and `distance_cm` non-null:
        group by (accession_number, clock, distance_cm) → ONE skeleton row.
  - Rows with NULL clock OR NULL distance_cm:
        kept as their own row (one per cadbusi.Lesions.lesion_id) and flagged
        with status='needs_adjudication' so they can be manually merged later.

Idempotent: re-running skips accessions already seeded (`INSERT OR IGNORE`
on the unique index, plus `source_lesion_ids` JSON list rebuilt each pass).

Usage:
    python tools/seed_lesion_annotations.py \
        --source data/cadbusi.db \
        --labels data/labeled_cadbusi.db
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.DB_processing.labels_database import LabelsDatabase


def _connect_ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_lesions(source_db: str):
    """Yield Lesions rows + per-image laterality joined from Images.

    Laterality comes from Images.laterality keyed on image_name. Lesions with no
    matching Images row get laterality=NULL (later filled in by descriptor LLM
    or by the annotator).
    """
    conn = _connect_ro(source_db)
    try:
        rows = conn.execute("""
            SELECT
                L.lesion_id,
                L.accession_number,
                L.clock,
                L.distance_cm,
                L.parsed_lesion_measurement_cm,
                L.lesion_type,
                L.description,
                I.laterality
            FROM Lesions L
            LEFT JOIN Images I ON I.image_name = L.image_name
        """).fetchall()
    finally:
        conn.close()
    return rows


def parse_clock(raw) -> Optional[float]:
    """Convert a `cadbusi.Lesions.clock` value to a decimal-hour float.

    Accepted formats: '1:00', '1:30', '9 o\\'clock', '12', '12:00'. Returns None if
    the value is missing or unparseable.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower().replace("o'clock", "").strip()
    if not s:
        return None
    if ":" in s:
        try:
            h, m = s.split(":", 1)
            return float(int(h)) + float(int(m)) / 60.0
        except (ValueError, TypeError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _coalesce_laterality(values):
    """Pick a single laterality from a group, prefer non-null and majority."""
    counts = defaultdict(int)
    for v in values:
        if v in ("LEFT", "RIGHT"):
            counts[v] += 1
    if not counts:
        return None
    if len(counts) == 2 and counts["LEFT"] == counts["RIGHT"]:
        return None  # tied → leave for adjudication
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _coalesce_size(values):
    """Take max parsed measurement from a group (cm → mm)."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return max(nums) * 10.0  # cm → mm


def build_skeletons(rows):
    """Group rows into skeleton lesion_annotations entries.

    Returns:
      grouped_rows: list of dicts ready to INSERT
    """
    groups = defaultdict(list)        # (acc, clock, dist) → list of source rows
    nullish: list = []                 # rows with NULL clock or distance

    for r in rows:
        (lesion_id, acc, clock, dist, parsed_cm, lesion_type, desc, lat) = r
        clock_f = parse_clock(clock)
        try:
            dist_f = float(dist) if dist is not None else None
        except (TypeError, ValueError):
            dist_f = None
        if clock_f is None or dist_f is None:
            nullish.append((lesion_id, acc, clock_f, dist_f, parsed_cm, lesion_type, desc, lat))
        else:
            groups[(acc, clock_f, dist_f)].append(
                (lesion_id, acc, clock_f, dist_f, parsed_cm, lesion_type, desc, lat)
            )

    out = []
    # Grouped (clock + distance present) — one skeleton row per group
    for (acc, clock, dist), group in groups.items():
        ids = [g[0] for g in group]
        out.append({
            "source_lesion_ids": json.dumps(ids),
            "accession_number":  acc,
            "laterality":        _coalesce_laterality(g[7] for g in group),
            "clock_hr":          clock,
            "distance_cm":       dist,
            "size_mm":           _coalesce_size(g[4] for g in group),
            "status":            "seeded",
        })

    # Nullish — preserve each lesion_id as its own skeleton, flagged
    for r in nullish:
        (lesion_id, acc, clock_f, dist_f, parsed_cm, lesion_type, desc, lat) = r
        out.append({
            "source_lesion_ids": json.dumps([lesion_id]),
            "accession_number":  acc,
            "laterality":        lat if lat in ("LEFT", "RIGHT") else None,
            "clock_hr":          clock_f,
            "distance_cm":       dist_f,
            "size_mm":           (parsed_cm * 10.0) if parsed_cm is not None else None,
            "status":            "needs_adjudication",
            "notes":             "missing clock or distance_cm",
        })
    return out


INSERT_SQL = """
INSERT OR IGNORE INTO lesion_annotations
  (source_lesion_ids, accession_number, laterality, clock_hr, distance_cm,
   size_mm, status, notes)
VALUES
  (:source_lesion_ids, :accession_number, :laterality, :clock_hr, :distance_cm,
   :size_mm, :status, :notes)
"""


def seed(source_db: str, labels_db: str, dry_run: bool = False) -> dict:
    print(f"Source DB (read-only): {source_db}")
    print(f"Labels DB:             {labels_db}")

    if not Path(source_db).exists():
        raise SystemExit(f"Source DB not found: {source_db}")

    # Make sure labels DB exists with schema
    db = LabelsDatabase(db_file=labels_db)
    db.connect()
    db.create_schema()

    print("Reading Lesions...")
    rows = load_lesions(source_db)
    print(f"  raw rows: {len(rows):,}")

    skeletons = build_skeletons(rows)
    grouped_n = sum(1 for s in skeletons if s["status"] == "seeded")
    flagged_n = sum(1 for s in skeletons if s["status"] == "needs_adjudication")
    print(f"  skeleton rows: {len(skeletons):,}  ({grouped_n:,} dedupe-grouped, {flagged_n:,} need adjudication)")

    if dry_run:
        print("Dry run — not writing.")
        db.close()
        return {"raw": len(rows), "skeletons": len(skeletons), "inserted": 0}

    # Make sure each row has every key for executemany
    for s in skeletons:
        s.setdefault("notes", None)

    cur = db.conn.cursor()
    cur.executemany(INSERT_SQL, skeletons)
    inserted = cur.rowcount
    db.conn.commit()
    db.close()

    print(f"Inserted (or ignored if existing): {inserted:,}")
    return {"raw": len(rows), "skeletons": len(skeletons), "inserted": inserted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to cadbusi source DB")
    ap.add_argument("--labels", required=True, help="Path to labels DB (will be created if missing)")
    ap.add_argument("--dry-run", action="store_true", help="Print counts, don't write")
    args = ap.parse_args()

    seed(
        source_db=str(Path(args.source).expanduser().resolve()),
        labels_db=str(Path(args.labels).expanduser().resolve()),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
