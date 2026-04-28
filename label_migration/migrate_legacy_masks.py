"""Migrate legacy drawn-mask PNGs into mask_artifacts.

Per Tristan_Tasks_04_22_26.md step 5 and MASK_SAVE_SPEC.md §8:

For each `drawn_masks/{clean_hash}.png`:
  1. Look up `cadbusi.Images` where `dicom_hash = clean_hash` to find the
     image_name + accession_number.
  2. Find lesions on that frame in `cadbusi.Lesions` by joining on image_name
     + accession_number.
  3. Resolve the lesion_annot_id via the JSON list of source_lesion_ids
     stored in lesion_annotations.
  4. Upsert a frame_lesion_views row (dicom_hash, lesion_annot_id).
  5. Insert a mask_artifacts row pointing at the spec'd GCS layout
     (`gs://shared-aif-bucket-87d1/annotation_hub/masks/{frame_lesion_id}.preparer.png`).

Edge cases (per spec §8):
  - No Images row for clean_hash       → log + skip
  - No Lesions row for that frame      → create lesion_annotations row with
                                          NULL clock/distance and notes='legacy_migration',
                                          then proceed.
  - Multiple Lesions rows for that frame → log + skip (manual disambiguation).

Idempotent: re-running skips masks already present (UNIQUE on
mask_artifacts(frame_lesion_id, stage)).

NOTE: The actual GCS upload of PNGs is NOT done by this script — it only
writes the metadata pointing at the intended GCS paths. The upload step
runs separately at Cloud Run deployment time.

Usage:
    python label_migration/migrate_legacy_masks.py \
        --source label_migration/cadbusi_2026_4_20.db \
        --labels label_migration/labeled_cadbusi.db \
        --masks  label_migration/annotation_hub/drawn_masks \
        --bucket shared-aif-bucket-87d1
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # already in requirements.txt

from src.DB_processing.labels_database import LabelsDatabase


def _connect_ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def get_or_create_user(conn: sqlite3.Connection, name: str, role: str) -> int:
    cur = conn.cursor()
    row = cur.execute("SELECT user_id FROM users WHERE name = ?", (name,)).fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO users(name, role) VALUES (?, ?)", (name, role))
    conn.commit()
    return cur.lastrowid


def build_lesion_id_to_annot_id(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT lesion_annot_id, source_lesion_ids FROM lesion_annotations"
    ).fetchall()
    out = {}
    for annot_id, srcs in rows:
        try:
            ids = json.loads(srcs or "[]")
        except json.JSONDecodeError:
            continue
        for lid in ids:
            out[int(lid)] = annot_id
    return out


def lookup_image(source_conn: sqlite3.Connection, clean_hash: str):
    """Return (image_name, accession_number) for a dicom_hash, or None."""
    return source_conn.execute(
        "SELECT image_name, accession_number FROM Images WHERE dicom_hash = ? LIMIT 1",
        (clean_hash,),
    ).fetchone()


def lookup_lesions_for_frame(source_conn: sqlite3.Connection, image_name: str, acc: str):
    """Return all lesion_id rows on a given frame."""
    return source_conn.execute(
        "SELECT lesion_id FROM Lesions WHERE image_name = ? AND accession_number = ?",
        (image_name, acc),
    ).fetchall()


def upsert_frame_lesion_view(conn: sqlite3.Connection, dicom_hash: str, lesion_annot_id: int) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO frame_lesion_views (dicom_hash, lesion_annot_id) VALUES (?, ?)",
        (dicom_hash, lesion_annot_id),
    )
    conn.commit()
    row = cur.execute(
        "SELECT frame_lesion_id FROM frame_lesion_views WHERE dicom_hash = ? AND lesion_annot_id = ?",
        (dicom_hash, lesion_annot_id),
    ).fetchone()
    return row[0]


def create_legacy_lesion_annot(conn: sqlite3.Connection, accession: str, clean_hash: str) -> int:
    """For frames with no Lesions row at all — spec §8 fallback.

    Uses the legacy clean_hash as a synthetic source key so the partial
    unique index on source_lesion_ids stays satisfied and re-runs are idempotent.
    """
    cur = conn.cursor()
    src_key = f"legacy:{clean_hash}"
    cur.execute("""
        INSERT OR IGNORE INTO lesion_annotations
          (source_lesion_ids, accession_number, notes, status)
        VALUES (?, ?, ?, ?)
    """, (src_key, accession, "legacy_migration: no source Lesions row", "needs_adjudication"))
    conn.commit()
    row = cur.execute(
        "SELECT lesion_annot_id FROM lesion_annotations WHERE source_lesion_ids = ?",
        (src_key,),
    ).fetchone()
    return row[0]


def insert_mask_artifact(
    conn: sqlite3.Connection,
    *,
    frame_lesion_id: int,
    bucket: str,
    width: int,
    height: int,
    annotator_user_id: int,
    file_mtime: float,
) -> bool:
    """Returns True if a row was inserted, False if it already existed."""
    json_path = f"gs://{bucket}/annotation_hub/masks/{frame_lesion_id}.preparer.json"
    png_path  = f"gs://{bucket}/annotation_hub/masks/{frame_lesion_id}.preparer.png"
    created   = datetime.fromtimestamp(file_mtime, tz=timezone.utc).isoformat()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO mask_artifacts
          (frame_lesion_id, stage, format, gcs_path_json, gcs_path_png,
           image_width, image_height, annotator_user_id, created_at)
        VALUES (?, 'preparer_pass', 'png', ?, ?, ?, ?, ?, ?)
    """, (frame_lesion_id, json_path, png_path, width, height, annotator_user_id, created))
    conn.commit()
    return cur.rowcount > 0


def migrate(
    source_db: str,
    labels_db: str,
    masks_dir: str,
    bucket: str,
    annotator: str = "jbaggett",
    dry_run: bool = False,
):
    print(f"Source DB: {source_db}")
    print(f"Labels DB: {labels_db}")
    print(f"Masks dir: {masks_dir}")
    print(f"Bucket:    {bucket}")

    masks_path = Path(masks_dir)
    pngs = sorted(masks_path.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No PNGs found in {masks_dir}")
    print(f"Found {len(pngs)} legacy PNG masks.")

    db = LabelsDatabase(db_file=labels_db)
    db.connect()
    db.create_schema()
    src_conn = _connect_ro(source_db)

    if dry_run:
        print("Dry run — only counting matches, not writing.")
        annotator_id = -1
    else:
        annotator_id = get_or_create_user(db.conn, annotator, "preparer")
        print(f"Annotator user_id: {annotator_id} ({annotator})")

    lid_to_annot = build_lesion_id_to_annot_id(db.conn)
    print(f"Mapped lesion_ids: {len(lid_to_annot):,}")

    counts = defaultdict(int)
    for png in pngs:
        clean_hash = png.stem
        img_row = lookup_image(src_conn, clean_hash)
        if img_row is None:
            counts["no_image"] += 1
            continue
        image_name, acc = img_row

        lesion_rows = lookup_lesions_for_frame(src_conn, image_name, acc)
        if len(lesion_rows) == 0:
            counts["no_lesion_fallback"] += 1
            if dry_run:
                continue
            annot_id = create_legacy_lesion_annot(db.conn, acc, clean_hash)
        elif len(lesion_rows) > 1:
            # Multi-lesion frames: skip per spec; report for manual review
            counts["multi_lesion_skipped"] += 1
            continue
        else:
            lesion_id = lesion_rows[0][0]
            annot_id = lid_to_annot.get(int(lesion_id))
            if annot_id is None:
                counts["unmapped_lesion_id"] += 1
                continue

        if dry_run:
            counts["would_migrate"] += 1
            continue

        flv_id = upsert_frame_lesion_view(db.conn, clean_hash, annot_id)
        try:
            with Image.open(png) as im:
                w, h = im.size
        except Exception as e:
            counts[f"image_read_error"] += 1
            print(f"  ! cannot open {png.name}: {e}")
            continue
        inserted = insert_mask_artifact(
            db.conn,
            frame_lesion_id=flv_id,
            bucket=bucket,
            width=w,
            height=h,
            annotator_user_id=annotator_id,
            file_mtime=png.stat().st_mtime,
        )
        counts["inserted" if inserted else "already_present"] += 1

    src_conn.close()
    db.close()

    print("\nSummary:")
    for k, v in sorted(counts.items()):
        print(f"  {k:<22} {v}")

    if not dry_run:
        print(f"\nNote: actual GCS upload of PNG bytes is a separate step "
              f"(not run by this script).  PNGs to upload live in {masks_dir}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--masks",  required=True, help="Directory of {clean_hash}.png files")
    ap.add_argument("--bucket", default="shared-aif-bucket-87d1",
                    help="Target GCS bucket name (used to construct gcs_path_*)")
    ap.add_argument("--annotator", default="jbaggett",
                    help="Annotator name to attribute these masks to (creates a users row)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    migrate(
        source_db=str(Path(args.source).expanduser().resolve()),
        labels_db=str(Path(args.labels).expanduser().resolve()),
        masks_dir=str(Path(args.masks).expanduser().resolve()),
        bucket=args.bucket,
        annotator=args.annotator,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
