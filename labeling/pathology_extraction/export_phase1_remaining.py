"""Export remaining Phase 1 rad_pathology_txt inputs for VM extraction.

Excludes accessions already extracted cleanly in `pathology_extracted`.
Produces JSONL suitable for scp to the Vertex VM.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3
from bus_data import ManifestDB

MANIFEST = '/home/jbaggett/BUS_framework/data/registry/bus_manifest_v3.db'
OUT = PROJECT_ROOT / 'data/pathology_extraction/phase1_remaining_inputs.jsonl'


def main():
    # Already extracted (clean) accessions
    conn = sqlite3.connect(MANIFEST)
    cur = conn.execute("SELECT accession_number FROM pathology_extracted WHERE extraction_error IS NULL")
    done = set(r[0] for r in cur.fetchall())
    conn.close()
    print(f"Already extracted (clean): {len(done):,}")

    db = ManifestDB()
    rows = db.query_df("""
        SELECT DISTINCT accession_number, rad_pathology_txt
        FROM src.StudyCases
        WHERE rad_pathology_txt IS NOT NULL
          AND LENGTH(TRIM(rad_pathology_txt)) > 20
        ORDER BY accession_number
    """)
    print(f"All rad_pathology_txt records: {len(rows):,}")

    todo = rows[~rows['accession_number'].isin(done)]
    print(f"Remaining: {len(todo):,}")

    with open(OUT, 'w') as f:
        for _, row in todo.iterrows():
            obj = {
                'accession_number': str(row['accession_number']),
                'rad_pathology_txt': row['rad_pathology_txt'],
            }
            f.write(json.dumps(obj) + '\n')
    print(f"\n✓ Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
