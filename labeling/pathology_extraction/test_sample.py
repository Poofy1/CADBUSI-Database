"""Quick sanity run: extract on 50 records, print a readable summary."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import random
from bus_data import ManifestDB
from data.pathology_extraction import PathologyExtractor


def main():
    db = ManifestDB()
    # Pull a diverse 50-row sample: mix of short and long texts
    rows = db.query_df("""
        SELECT accession_number, rad_pathology_txt
        FROM src.StudyCases
        WHERE rad_pathology_txt IS NOT NULL
          AND LENGTH(TRIM(rad_pathology_txt)) > 20
        ORDER BY RANDOM()
        LIMIT 50
    """)
    print(f"Loaded {len(rows)} sample texts")
    print()

    extractor = PathologyExtractor()
    items = list(zip(rows["accession_number"], rows["rad_pathology_txt"]))
    print(f"Extracting with model={extractor.model}...")
    import time
    t0 = time.time()
    results = extractor.extract_batch(items, show_progress=True)
    elapsed = time.time() - t0

    # Stats
    ok = [r for r in results if r.extraction is not None]
    err = [r for r in results if r.extraction is None]
    print()
    print(f"=== Results: {len(ok)}/{len(results)} success in {elapsed:.1f}s "
          f"({elapsed/len(results):.2f}s/record) ===")
    if err:
        print(f"Errors ({len(err)}):")
        for e in err[:5]:
            print(f"  {e.accession}: {e.error}")

    # Print a few extractions for eyeballing
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
