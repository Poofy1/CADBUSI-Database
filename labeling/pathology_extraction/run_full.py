"""Full extraction over all rad_pathology_txt records (~34K accessions).

Reads source pathology text from the CADBUSI DB (read-only) and writes
results to a separate output DB (`pathology_extracted.db` by default) so
the source is never modified. Supports resume — skips accessions already
in the output table.
"""
from __future__ import annotations
import sys
import time
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from labeling.pathology_extraction import PathologyExtractor

DEFAULT_SOURCE_DB = str(PROJECT_ROOT / 'data' / 'cadbusi.db')
DEFAULT_OUTPUT_DB = str(PROJECT_ROOT / 'data' / 'pathology_extracted.db')

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pathology_extracted (
    accession_number     TEXT PRIMARY KEY,
    source_text          TEXT NOT NULL,
    primary_diagnosis    TEXT NOT NULL,
    cancer_subtypes      TEXT NOT NULL,       -- pipe-separated enum values
    benign_subtypes      TEXT NOT NULL,       -- pipe-separated enum values
    laterality           TEXT NOT NULL,
    size_mm              REAL,
    grade                TEXT,
    lymph_node_status    TEXT,
    is_lymph_node_biopsy INTEGER NOT NULL,    -- 0/1
    confidence           TEXT NOT NULL,
    notes                TEXT,
    model_name           TEXT NOT NULL,
    extraction_error     TEXT,                -- populated if extraction failed
    created_at           TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pe_primary ON pathology_extracted(primary_diagnosis);
"""

INSERT_SQL = """
INSERT OR REPLACE INTO pathology_extracted
  (accession_number, source_text, primary_diagnosis, cancer_subtypes, benign_subtypes,
   laterality, size_mm, grade, lymph_node_status, is_lymph_node_biopsy,
   confidence, notes, model_name, extraction_error)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def ensure_table(output_db: str):
    Path(output_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_db)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()


def already_extracted(output_db: str) -> set:
    conn = sqlite3.connect(output_db)
    try:
        rows = conn.execute("SELECT accession_number FROM pathology_extracted").fetchall()
    finally:
        conn.close()
    return set(r[0] for r in rows)


def save_results(output_db: str, results, model_name: str):
    conn = sqlite3.connect(output_db)
    cur = conn.cursor()
    for r in results:
        e = r.extraction
        if e is None:
            cur.execute(INSERT_SQL, (
                r.accession, r.text, 'UNKNOWN', '', '', 'NOT_SPECIFIED',
                None, None, None, 0, 'LOW', None, model_name, r.error or 'extraction_failed'
            ))
        else:
            cur.execute(INSERT_SQL, (
                r.accession,
                r.text,
                e.primary_diagnosis.value,
                '|'.join(c.value for c in e.cancer_subtypes),
                '|'.join(b.value for b in e.benign_subtypes),
                e.laterality.value,
                e.size_mm,
                e.grade.value,
                e.lymph_node_status.value,
                1 if e.is_lymph_node_biopsy else 0,
                e.confidence.value,
                e.notes,
                model_name,
                None,
            ))
    conn.commit()
    conn.close()


def load_source_rows(source_db: str) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(
            """
            SELECT DISTINCT accession_number, rad_pathology_txt
            FROM StudyCases
            WHERE rad_pathology_txt IS NOT NULL
              AND LENGTH(TRIM(rad_pathology_txt)) > 20
            ORDER BY accession_number
            """,
            conn,
        )
    finally:
        conn.close()


def main(source_db: str = DEFAULT_SOURCE_DB, output_db: str = DEFAULT_OUTPUT_DB,
         batch_size: int = 200, max_workers: int = 16,
         use_vertex: bool = False, project: str = None, location: str = None,
         model: str = None):
    if not Path(source_db).exists():
        raise SystemExit(f"Source DB not found: {source_db}")
    print(f"Source DB (read-only): {source_db}")
    print(f"Output DB:             {output_db}")

    ensure_table(output_db)
    done = already_extracted(output_db)
    print(f"Already extracted: {len(done):,}")

    rows = load_source_rows(source_db)
    todo = rows[~rows['accession_number'].isin(done)]
    print(f"Total records: {len(rows):,}, remaining: {len(todo):,}")
    if len(todo) == 0:
        print("All done!")
        return

    extractor_kwargs = {'use_vertex': use_vertex}
    if project: extractor_kwargs['project'] = project
    if location: extractor_kwargs['location'] = location
    if model: extractor_kwargs['model'] = model
    extractor = PathologyExtractor(**extractor_kwargs)
    print(f"Mode: {extractor.mode}, Model: {extractor.model}, workers: {max_workers}, batch size: {batch_size}")
    if extractor.mode == 'vertex':
        print(f"  Vertex project: {extractor.project}, location: {extractor.location}")

    start = time.time()
    processed = 0
    items = list(zip(todo['accession_number'], todo['rad_pathology_txt']))
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        t0 = time.time()
        results = extractor.extract_batch_parallel(
            batch, max_workers=max_workers, show_progress=False
        )
        save_results(output_db, results, extractor.model)
        processed += len(batch)
        dt = time.time() - t0
        total_dt = time.time() - start
        rate = processed / total_dt
        remaining = len(items) - processed
        eta_min = (remaining / rate) / 60 if rate else float('inf')
        ok = sum(1 for r in results if r.extraction is not None)
        print(f"[{processed:>6}/{len(items):>6}] batch ok={ok}/{len(batch)} in {dt:.1f}s  "
              f"| rate={rate:.2f}/s  ETA={eta_min:.1f}min", flush=True)

    print(f"\n✓ Full extraction complete in {(time.time()-start)/60:.1f} minutes")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--db', default=DEFAULT_SOURCE_DB,
                   help='Source CADBUSI DB (read-only) — where rad_pathology_txt is read from')
    p.add_argument('--out', default=DEFAULT_OUTPUT_DB,
                   help='Output DB for pathology_extracted table')
    p.add_argument('--batch-size', type=int, default=200)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--vertex', action='store_true',
                   help='Use GCP Vertex AI auth instead of GEMINI_API_KEY (higher quotas, pay-as-you-go)')
    p.add_argument('--project', default=None, help='GCP project ID (for --vertex)')
    p.add_argument('--location', default=None, help='Vertex region, default us-central1')
    p.add_argument('--model', default=None,
                   help='Override default model (e.g. gemini-2.5-pro, gemini-3-flash-preview)')
    args = p.parse_args()
    main(
        source_db=args.db, output_db=args.out,
        batch_size=args.batch_size, max_workers=args.workers,
        use_vertex=args.vertex, project=args.project, location=args.location,
        model=args.model,
    )
