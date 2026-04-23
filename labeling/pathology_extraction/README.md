# Pathology Extraction

Structured extraction of Mayo pathology text into a queryable table.
Uses `google.genai` with Pydantic response schemas for reliable structured output.

Source: `StudyCases.rad_pathology_txt` in a CADBUSI sqlite DB.
Output: `pathology_extracted` table in a **separate** sqlite DB — the source
DB is opened read-only and never modified.

## Layout

```
labeling/pathology_extraction/
├── schema.py           # Pydantic models (enums + PathologyExtraction)
├── prompts.py          # Extraction prompt
├── extractor.py        # PathologyExtractor class (API-key OR Vertex AI mode)
├── test_sample.py      # 50-record eyeball test
├── run_full.py         # Full extraction, resume-safe, writes to output DB
└── validate.py         # (not yet updated for this repo)
```

## Setup

```bash
pip install -r requirements.txt   # includes google-genai
```

Auth — pick one:

- **API key** (simple): `export GEMINI_API_KEY=...` (or `$env:GEMINI_API_KEY=...` in PowerShell)
- **Vertex AI** (higher quotas, pay-as-you-go): `gcloud auth application-default login`, then
  `export VERTEX_PROJECT=your-project` and `export VERTEX_LOCATION=us-central1`

## Quick test (50 records)

`--db` is required — point it at whichever CADBUSI sqlite you want to read from.

```bash
python labeling/pathology_extraction/test_sample.py --db data/cadbusi.db --n 50
```

## Full run

```bash
# API key mode
python labeling/pathology_extraction/run_full.py \
    --db data/cadbusi.db \
    --out data/pathology_extracted.db \
    --workers 16 --batch-size 200

# Vertex AI mode
python labeling/pathology_extraction/run_full.py \
    --db data/cadbusi.db \
    --out data/pathology_extracted.db \
    --vertex --project $VERTEX_PROJECT --location $VERTEX_LOCATION \
    --workers 32 --batch-size 200
```

`--out` defaults to `data/pathology_extracted.db` if omitted.
Re-running `run_full.py` skips accessions already in the output table (resume-safe).

## Output table

`pathology_extracted` columns:
`accession_number`, `source_text`, `primary_diagnosis`, `cancer_subtypes`,
`benign_subtypes`, `laterality`, `size_mm`, `grade`, `lymph_node_status`,
`is_lymph_node_biopsy`, `confidence`, `notes`, `model_name`,
`extraction_error`, `created_at`.

## Gotchas

- **Rate limits** bite quickly on preview models (`gemini-3-flash-preview`). Use GA models (`gemini-2.5-flash`) for production volume — ~5× the RPM budget.
- **Errors are written as rows** with `extraction_error` populated — good for post-hoc triage. To retry them:
  `DELETE FROM pathology_extracted WHERE extraction_error IS NOT NULL;` then rerun.
- **`Warning: thought_signature` in stdout** is Gemini 3's thinking-mode artifacts. Harmless; filter out of logs with `grep -v thought_signature`.
- **`GCP_VM_RUNBOOK.md`** in this folder is stale — it references the BUS_framework repo layout and the Phase-2 jsonl-export scripts, which haven't been ported to CADBUSI-Database.
