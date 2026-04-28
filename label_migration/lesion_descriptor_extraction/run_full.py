"""Run BI-RADS descriptor extraction over all unique lesion descriptions and
backfill the extracted columns on `lesion_annotations`.

Strategy:
  1. Read the source DB (read-only) to map every lesion_id → description.
  2. Read lesion_annotations from the labels DB; resolve each row's
     `source_lesion_ids` JSON list to its set of source descriptions, pick
     the longest one as the representative description for that lesion.
  3. Deduplicate by representative description text — typically ~30K unique
     descriptions across hundreds of thousands of lesions, so Gemini is
     called at most once per unique phrase.
  4. Update each lesion_annotations row whose `shape IS NULL` (i.e. not yet
     extracted) with the cached extraction.

Resume-safe: a row whose `shape` is already non-NULL is skipped on the
update pass. A small `descriptor_extraction_cache` table inside the labels
DB persists per-description results so repeated runs don't re-call Gemini.

Usage:
    python labeling/lesion_descriptor_extraction/run_full.py \
        --source data/cadbusi.db \
        --labels data/labeled_cadbusi.db \
        --vertex --project $VERTEX_PROJECT --location $VERTEX_LOCATION \
        --workers 32
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from label_migration.lesion_descriptor_extraction import LesionDescriptorExtractor


CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS descriptor_extraction_cache (
    description        TEXT PRIMARY KEY,
    shape              TEXT,
    orientation        TEXT,
    margin_clarity     TEXT,
    margin_detail      TEXT,
    echo_pattern       TEXT,
    posterior_features TEXT,
    calcifications     TEXT,
    vascularity        TEXT,
    lesion_kind        TEXT,
    confidence         TEXT,
    notes              TEXT,
    model_name         TEXT,
    extraction_error   TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

CACHE_INSERT = """
INSERT OR REPLACE INTO descriptor_extraction_cache
  (description, shape, orientation, margin_clarity, margin_detail,
   echo_pattern, posterior_features, calcifications, vascularity,
   lesion_kind, confidence, notes, model_name, extraction_error)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _connect_ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_lesion_descriptions(source_db: str) -> dict:
    """Map cadbusi.Lesions.lesion_id → description (text). NULLs/empties dropped."""
    conn = _connect_ro(source_db)
    try:
        rows = conn.execute(
            "SELECT lesion_id, description FROM Lesions WHERE description IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return {lid: (desc or "").strip() for lid, desc in rows if (desc or "").strip()}


def representative_description(source_lesion_ids_json: str, descs: dict) -> str:
    """Pick longest non-empty description for the lesion_annotations row."""
    try:
        ids = json.loads(source_lesion_ids_json or "[]")
    except json.JSONDecodeError:
        return ""
    candidates = [descs.get(int(i), "") for i in ids if int(i) in descs]
    candidates = [c for c in candidates if c]
    if not candidates:
        return ""
    return max(candidates, key=len)


def gather_pending(labels_db: str, descs: dict):
    """Return (lesion_annot_id, description) pairs for rows that still need extraction."""
    conn = sqlite3.connect(labels_db)
    try:
        rows = conn.execute("""
            SELECT lesion_annot_id, source_lesion_ids
            FROM lesion_annotations
            WHERE shape IS NULL
        """).fetchall()
    finally:
        conn.close()
    pending = []
    for lid, srcs in rows:
        d = representative_description(srcs, descs)
        if d:
            pending.append((lid, d))
    return pending


def load_cache(labels_db: str) -> dict:
    conn = sqlite3.connect(labels_db)
    conn.executescript(CACHE_SCHEMA)
    rows = conn.execute("""
        SELECT description, shape, orientation, margin_clarity, margin_detail,
               echo_pattern, posterior_features, calcifications, vascularity,
               lesion_kind, confidence, notes, extraction_error
        FROM descriptor_extraction_cache
    """).fetchall()
    conn.close()
    keys = ["shape", "orientation", "margin_clarity", "margin_detail",
            "echo_pattern", "posterior_features", "calcifications", "vascularity",
            "lesion_kind", "confidence", "notes", "extraction_error"]
    return {r[0]: dict(zip(keys, r[1:])) for r in rows}


def save_cache_batch(labels_db: str, results, model_name: str):
    conn = sqlite3.connect(labels_db)
    cur = conn.cursor()
    rows = []
    for r in results:
        e = r.extraction
        if e is None:
            rows.append((r.text, None, None, None, None, None, None, None, None, None,
                         None, None, model_name, r.error or "extraction_failed"))
        else:
            rows.append((
                r.text,
                e.shape.value, e.orientation.value,
                e.margin_clarity.value, e.margin_detail.value,
                e.echo_pattern.value, e.posterior_features.value,
                e.calcifications.value, e.vascularity.value,
                e.lesion_kind.value, e.confidence.value, e.notes,
                model_name, None,
            ))
    cur.executemany(CACHE_INSERT, rows)
    conn.commit()
    conn.close()


UPDATE_SQL = """
UPDATE lesion_annotations
SET shape              = :shape,
    orientation        = :orientation,
    margin_clarity     = :margin_clarity,
    margin_detail      = :margin_detail,
    echo_pattern       = :echo_pattern,
    posterior_features = :posterior_features,
    calcifications     = :calcifications,
    vascularity        = :vascularity,
    lesion_kind        = :lesion_kind
WHERE lesion_annot_id = :lesion_annot_id
"""


def backfill_lesion_rows(labels_db: str, pending, cache):
    """Apply cache to lesion_annotations rows. Skip rows whose description had an extraction error."""
    conn = sqlite3.connect(labels_db)
    cur = conn.cursor()
    updates = []
    skipped = 0
    for lid, desc in pending:
        c = cache.get(desc)
        if c is None or c["extraction_error"]:
            skipped += 1
            continue
        updates.append({
            "lesion_annot_id":    lid,
            "shape":              c["shape"],
            "orientation":        c["orientation"],
            "margin_clarity":     c["margin_clarity"],
            "margin_detail":      c["margin_detail"],
            "echo_pattern":       c["echo_pattern"],
            "posterior_features": c["posterior_features"],
            "calcifications":     c["calcifications"],
            "vascularity":        c["vascularity"],
            "lesion_kind":        c["lesion_kind"],
        })
    cur.executemany(UPDATE_SQL, updates)
    conn.commit()
    conn.close()
    return len(updates), skipped


def main(source_db: str, labels_db: str, batch_size: int = 200, max_workers: int = 16,
         use_vertex: bool = False, project: str = None, location: str = None,
         model: str = None, limit: int = 0):
    if not Path(source_db).exists():
        raise SystemExit(f"Source DB not found: {source_db}")
    if not Path(labels_db).exists():
        raise SystemExit(f"Labels DB not found: {labels_db}")

    print(f"Source DB: {source_db}")
    print(f"Labels DB: {labels_db}")

    print("Loading description map from source DB...")
    descs = load_lesion_descriptions(source_db)
    print(f"  lesion_id → description rows: {len(descs):,}")

    print("Loading cache from labels DB...")
    cache = load_cache(labels_db)
    print(f"  cached descriptions: {len(cache):,}")

    print("Finding lesion_annotations rows needing extraction...")
    pending = gather_pending(labels_db, descs)
    print(f"  pending rows (shape IS NULL with a description): {len(pending):,}")

    unique_texts = sorted({d for _, d in pending}) if pending else []
    to_extract = [t for t in unique_texts if t not in cache]
    print(f"  unique descriptions: {len(unique_texts):,};  not yet extracted: {len(to_extract):,}")

    if limit and len(to_extract) > limit:
        print(f"  (limiting this run to {limit:,} unique descriptions)")
        to_extract = to_extract[:limit]

    if to_extract:
        extractor_kwargs = {"use_vertex": use_vertex}
        if project: extractor_kwargs["project"] = project
        if location: extractor_kwargs["location"] = location
        if model: extractor_kwargs["model"] = model
        extractor = LesionDescriptorExtractor(**extractor_kwargs)
        print(f"Mode: {extractor.mode}  Model: {extractor.model}  workers: {max_workers}  batch: {batch_size}")

        start = time.time()
        for i in range(0, len(to_extract), batch_size):
            batch_texts = to_extract[i:i + batch_size]
            items = [(t, t) for t in batch_texts]   # key == text since text is unique here
            t0 = time.time()
            results = extractor.extract_batch_parallel(items, max_workers=max_workers, show_progress=False)
            save_cache_batch(labels_db, results, extractor.model)
            done = i + len(batch_texts)
            ok = sum(1 for r in results if r.extraction is not None)
            dt = time.time() - t0
            total_dt = time.time() - start
            rate = done / total_dt
            eta_min = (len(to_extract) - done) / rate / 60 if rate else float('inf')
            print(f"[{done:>6}/{len(to_extract):>6}] ok={ok}/{len(batch_texts)} in {dt:.1f}s  "
                  f"| rate={rate:.2f}/s  ETA={eta_min:.1f}min", flush=True)

        # Reload cache to include the just-written entries
        cache = load_cache(labels_db)

    print("Backfilling lesion_annotations rows...")
    updated, skipped = backfill_lesion_rows(labels_db, pending, cache)
    print(f"  updated: {updated:,}  skipped (no cache or error): {skipped:,}")
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Path to cadbusi source DB (read-only)")
    p.add_argument("--labels", required=True, help="Path to labels DB (must already have lesion_annotations seeded)")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap unique descriptions extracted this run (0 = no cap). Useful for smoke tests.")
    p.add_argument("--vertex", action="store_true")
    p.add_argument("--project", default=None)
    p.add_argument("--location", default=None)
    p.add_argument("--model", default=None)
    args = p.parse_args()
    main(
        source_db=str(Path(args.source).expanduser().resolve()),
        labels_db=str(Path(args.labels).expanduser().resolve()),
        batch_size=args.batch_size, max_workers=args.workers,
        use_vertex=args.vertex, project=args.project, location=args.location,
        model=args.model, limit=args.limit,
    )
