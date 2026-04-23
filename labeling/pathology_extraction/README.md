# Pathology Extraction

Structured extraction of Mayo pathology text into a queryable table.
Uses `google.genai` with Pydantic response schemas for reliable structured output.

## Architecture

```
data/pathology_extraction/
├── schema.py           # Pydantic models (enums + PathologyExtraction)
├── prompts.py          # Extraction prompt
├── extractor.py        # PathologyExtractor class (API-key OR Vertex AI mode)
├── test_sample.py      # 50-record eyeball test
├── validate.py         # 924-overlap validation vs structured Pathology
└── run_full.py         # Full extraction, resume-safe, writes to manifest DB
```

## Two run modes

### Mode A: Direct Gemini API (default, free tier)
```bash
source ~/.bashrc  # loads GEMINI_API_KEY
python data/pathology_extraction/run_full.py --workers 16 --batch-size 200
```

- Uses `GEMINI_API_KEY` from `~/.bashrc`.
- Consumer-tier quotas. `gemini-2.5-flash` (GA) has reasonable quotas; preview models like `gemini-3-flash-preview` are very restricted and hit 429 after ~15K requests.
- Best for prototyping, small runs, the current Phase 1 extraction.

### Mode B: Vertex AI on GCP (recommended for Phase 2 synoptic extraction)
```bash
# One-time setup (local dev or GCP VM):
gcloud auth application-default login    # sets Application Default Credentials
export VERTEX_PROJECT=your-gcp-project-id
export VERTEX_LOCATION=us-central1        # or your nearest region

# Run:
python data/pathology_extraction/run_full.py \
    --vertex \
    --project $VERTEX_PROJECT \
    --location $VERTEX_LOCATION \
    --workers 32 --batch-size 200
```

- Auth via IAM (service account on GCP VMs, or ADC locally). **No API key in .bashrc.**
- Higher quotas + pay-as-you-go pricing — no preview-model rate limits.
- Same `google.genai` SDK; just constructs the client with `vertexai=True`.
- On a GCP VM with a service account that has `aiplatform.user` role, everything just works.
- **Recommended for Phase 2** (50K synoptic reports, larger token budget per record).

## Extracted table

Schema: `data/registry/SCHEMA_GUIDE.md#pathology_extracted`. Key columns:
`primary_diagnosis`, `cancer_subtypes`, `benign_subtypes`, `laterality`, `size_mm`,
`grade`, `lymph_node_status`, `is_lymph_node_biopsy`, `confidence`, `notes`,
`model_name`, `extraction_error`.

## Phase 1 vs Phase 2

| | Phase 1 (running) | Phase 2 (planned) |
|---|---|---|
| Source | `StudyCases.rad_pathology_txt` | `Pathology.synoptic_report` |
| Rows | ~34K | ~50K (per-Pathology-row, not per-accession) |
| Avg text length | ~200 chars | ~3,158 chars |
| Output table | `pathology_extracted` | `pathology_synoptic_extracted` |
| Schema richness | Coarse (primary + subtype) | Full (grade, size, margin, pTNM, ER/PR, LN counts) |
| Auth | API key OK | Vertex AI strongly recommended |
| Cost | <$1 | ~$16 on Gemini 2.5 Flash |

## Gotchas

- **Rate limits** bite quickly on preview models (`gemini-3-flash-preview`). Use GA models (`gemini-2.5-flash`) for production volume — ~5× the RPM budget.
- **Resume works**: re-running `run_full.py` skips accessions already in the output table. Useful when quota resets or you want to retry errors.
- **Errors are written as rows** with `extraction_error` populated — good for post-hoc triage. If you want to retry them, `DELETE FROM pathology_extracted WHERE extraction_error IS NOT NULL;` before rerunning.
- **`Warning: thought_signature` in stdout** is Gemini 3's thinking-mode artifacts. Harmless; filter out of logs with `grep -v thought_signature`.
