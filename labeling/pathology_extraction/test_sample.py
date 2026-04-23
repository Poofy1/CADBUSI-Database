"""Quick sanity run: extract on 50 records, print a readable summary."""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from labeling.pathology_extraction import PathologyExtractor

def load_sample(db_path: Path, n: int) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(
            """
            SELECT accession_number, rad_pathology_txt
            FROM StudyCases
            WHERE rad_pathology_txt IS NOT NULL
              AND LENGTH(TRIM(rad_pathology_txt)) > 20
            ORDER BY RANDOM()
            LIMIT ?
            """,
            conn,
            params=(n,),
        )
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True,
                    help="Path to CADBUSI sqlite DB (e.g. data/cadbusi.db)")
    ap.add_argument("--n", type=int, default=50, help="Number of sample rows")
    ap.add_argument("--vertex", action="store_true",
                    help="Use Vertex AI auth instead of GEMINI_API_KEY")
    ap.add_argument("--project", default=None, help="GCP project (for --vertex)")
    ap.add_argument("--location", default=None, help="Vertex region (for --vertex)")
    ap.add_argument("--model", default=None, help="Override default model")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}  (cwd={Path.cwd()})")

    rows = load_sample(db_path, args.n)
    print(f"Loaded {len(rows)} sample texts from {db_path}")
    if len(rows) == 0:
        raise SystemExit("No rad_pathology_txt rows found — is this the populated DB?")
    print()

    extractor_kwargs = {"use_vertex": args.vertex}
    if args.project: extractor_kwargs["project"] = args.project
    if args.location: extractor_kwargs["location"] = args.location
    if args.model: extractor_kwargs["model"] = args.model
    extractor = PathologyExtractor(**extractor_kwargs)
    print(f"Mode: {extractor.mode}")
    items = list(zip(rows["accession_number"], rows["rad_pathology_txt"]))
    print(f"Extracting with model={extractor.model}...")
    t0 = time.time()
    results = extractor.extract_batch(items, show_progress=True)
    elapsed = time.time() - t0

    ok = [r for r in results if r.extraction is not None]
    err = [r for r in results if r.extraction is None]
    print()
    print(f"=== Results: {len(ok)}/{len(results)} success in {elapsed:.1f}s "
          f"({elapsed/len(results):.2f}s/record) ===")
    if err:
        print(f"Errors ({len(err)}):")
        for e in err[:5]:
            print(f"  {e.accession}: {e.error}")

    print("\n=== Sample extractions ===")
    for r in ok[:12]:
        e = r.extraction
        print(f"\n--- {r.accession} ---")
        print(f"  TEXT: {r.text[:180]!r}")
        print(f"  primary={e.primary_diagnosis.value}  confidence={e.confidence.value}")
        if e.cancer_subtypes:
            print(f"  cancer={[c.value for c in e.cancer_subtypes]}")
        if e.benign_subtypes:
            print(f"  benign={[b.value for b in e.benign_subtypes]}")
        extras = []
        if e.laterality.value != "NOT_SPECIFIED": extras.append(f"side={e.laterality.value}")
        if e.size_mm is not None: extras.append(f"size={e.size_mm}mm")
        if e.grade.value != "NOT_SPECIFIED": extras.append(f"grade={e.grade.value}")
        if e.lymph_node_status.value != "NOT_REPORTED": extras.append(f"LN={e.lymph_node_status.value}")
        if e.is_lymph_node_biopsy: extras.append("LN_bx")
        if extras: print(f"  {' '.join(extras)}")
        if e.notes: print(f"  notes: {e.notes}")


if __name__ == "__main__":
    main()
