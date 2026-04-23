"""Export Pathology.synoptic_report rows to JSONL for Phase 2 extraction on GCP VM.

Run locally. Produces data/pathology_extraction/synoptic_inputs.jsonl which gets
scp'd to the Vertex VM for processing.

Each line is one JSON object: {path_id, accession_number, synoptic_report}.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from bus_data import ManifestDB


OUT_PATH = PROJECT_ROOT / 'data/pathology_extraction/synoptic_inputs.jsonl'


def main(min_length: int = 50):
    db = ManifestDB()
    rows = db.query_df(f"""
        SELECT path_id, accession_number, synoptic_report
        FROM src.Pathology
        WHERE synoptic_report IS NOT NULL
          AND LENGTH(TRIM(synoptic_report)) >= {min_length}
        ORDER BY path_id
    """)
    print(f"Pathology rows to export: {len(rows):,}")
    print(f"Avg synoptic_report length: {rows['synoptic_report'].str.len().mean():.0f} chars")
    print(f"Max length: {rows['synoptic_report'].str.len().max():,} chars")

    # Size estimate
    total_chars = rows['synoptic_report'].str.len().sum()
    print(f"Total chars: {total_chars:,}")
    print(f"Estimated input tokens: {int(total_chars * 0.75 / 1000)}K")
    # cost @ gemini-2.5-flash: $0.075/M input, $0.30/M output
    # assume ~500 output tokens per record
    est_in_toks = total_chars * 0.75 / 1e6
    est_out_toks = len(rows) * 500 / 1e6
    print(f"Estimated cost (Gemini 2.5 Flash): ${est_in_toks*0.075 + est_out_toks*0.30:.2f}")

    with open(OUT_PATH, 'w') as f:
        for _, row in rows.iterrows():
            obj = {
                'path_id': int(row['path_id']),
                'accession_number': str(row['accession_number']),
                'synoptic_report': row['synoptic_report'],
            }
            f.write(json.dumps(obj) + '\n')
    print(f"\n✓ Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
