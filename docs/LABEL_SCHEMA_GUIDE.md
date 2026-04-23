# Labels Database Schema Guide

**Purpose**: authoritative reference for the tables, columns, and semantics in `labeled_cadbusi.db` (aka `cadbusi_labels.db`) — the downstream extraction / annotation layer in the three-DB CADBUSI architecture. Update this file whenever a column's meaning becomes clearer or a new table is added. If the code disagrees with this doc, fix the doc.

**Last updated**: 2026-04-23

---

## Database

`labeled_cadbusi.db` holds anything derived from `cadbusi_source.db` by LLM extraction, image processing, or model inference, plus hand-labeled annotations. Expected to iterate independently of ingestion.

| DB | Purpose | Mutability |
|---|---|---|
| `cadbusi_source.db` | Immutable clinical record per Mayo export | Read-only after ingestion |
| **`labeled_cadbusi.db`** | **Extraction / inference / annotation outputs** | **Append-only per `version`; new runs add rows, don't overwrite** |
| `bus_manifest_v3.db` | Research enrichment (cohorts, splits, custom joins) | Mutable; Jeff's |

Architecture spec: [../../BUS_framework/docs/tristan_handoff_2026_04_22/docs/THREE_DB_ARCHITECTURE.md](../../BUS_framework/docs/tristan_handoff_2026_04_22/docs/THREE_DB_ARCHITECTURE.md).

Joins to source: `dicom_hash` (image level) or `accession_number` (exam level).

### Versioning convention

Every table has a `version` TEXT column. **New runs append rows under a new version string, they don't overwrite.** The version string is a human-readable tag, not a strict `vN` — it identifies the run / model / task. Current contents:

| Table | Version strings in data |
|---|---|
| ImageLabels | `'pre-2024 labels'` |
| LesionLabels | `'2026_1_15_cropping'` (bbox rows), `'2026_1_15_seg'` (mask rows) |
| CaliperLabels | `'task_5'`, `'task_9'` (CVAT task IDs) |
| CaseLabels | `'birads4a_bert_v1'`, `'birads_4abc_bert_v1'` (two coexisting models) |
| PathologyExtracted | `'v1'` |
| PathologySynoptic | `'v1'` |

For the pathology tables, future runs should use `'v2'`, `'v3'`, etc. For the others, continue the existing tag convention (task ID, date-scoped tag, or model string).

### Indexes (as created by `migrate_labels_db.py`)

```
idx_imagelabels_dicom    ON ImageLabels(dicom_hash)
idx_lesionlabels_dicom   ON LesionLabels(dicom_hash)
idx_caliperlabels_dicom  ON CaliperLabels(dicom_hash)
```

No indexes on CaseLabels, PathologyExtracted, PathologySynoptic yet.

---

## Quick navigation

- [ImageLabels](#imagelabels) — per-image instance labels (9,923 rows)
- [LesionLabels](#lesionlabels) — lesion bboxes (cropping) + mask refs (segmentation), 16,473 rows
- [CaliperLabels](#caliperlabels) — per-image caliper annotations (18,059 rows)
- [CaseLabels](#caselabels) — BERT BI-RADS 4a/4b/4c subcategory predictions (96,622 rows)
- [PathologyExtracted](#pathologyextracted) — Phase 1 LLM extraction from `rad_pathology_txt` (34,171 rows)
- [PathologySynoptic](#pathologysynoptic) — Phase 2 LLM extraction from `synoptic_report` (50,616 rows)

---

## ImageLabels

**Grain**: one row per DICOM image. **9,923 rows.**
**Join key**: `dicom_hash` → `cadbusi_source.Images.dicom_hash`.
**Source**: hand-labeled instance labels. All current rows are `quality='gold'`, `version='pre-2024 labels'`.
**Defined in**: [src/DB_processing/labels_database.py:105-116](../src/DB_processing/labels_database.py#L105-L116).

| Column | Type | Meaning |
|---|---|---|
| `dicom_hash` | TEXT PRIMARY KEY | → `Images.dicom_hash` |
| `reject` | INTEGER (0/1) | Frame is non-diagnostic / junk (21.6% set) |
| `only_normal` | INTEGER (0/1) | Frame shows only normal tissue (3.6% set) |
| `cyst` | INTEGER (0/1) | Cyst present (0.9% set) |
| `benign` | INTEGER (0/1) | Benign lesion present (32.8% set) |
| `malignant` | INTEGER (0/1) | Malignant lesion present (41.5% set) |
| `quality` | TEXT | Annotation tier. Values: `gold` |
| `version` | TEXT | Label batch tag. Values: `pre-2024 labels` |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |

Note: the classes are not mutually exclusive (a frame can be both `benign` and `cyst`, etc.).

---

## LesionLabels

**Grain**: one row per lesion annotation. **16,473 rows.**
**Join key**: `dicom_hash` → `cadbusi_source.Images.dicom_hash`.
**Defined in**: [src/DB_processing/labels_database.py:120-134](../src/DB_processing/labels_database.py#L120-L134).

A foreign key on `dicom_hash → ImageLabels` is declared in the schema but **only ~190 rows** (1.2%) actually reference an `ImageLabels` row — the vast majority of lesion annotations are on images not in `ImageLabels`. Treat the FK as advisory, not as a lookup guarantee.

This table is a hybrid: each row is either a bounding box **or** a mask path, partitioned by `version`:

| `version` | Row count | `x1..y2` | `mask_image` | Meaning |
|---|---:|---|---|---|
| `2026_1_15_cropping` | 15,168 | populated | NULL | Lesion bounding boxes (pixel coords in original DICOM space) |
| `2026_1_15_seg` | 1,305 | NULL | populated (e.g. `mask_1351_1441_right_2.png`) | Gold segmentation mask filename |

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | |
| `dicom_hash` | TEXT | → `Images.dicom_hash` (advisory FK to ImageLabels — rarely matches) |
| `x1`, `y1`, `x2`, `y2` | INTEGER | Bounding box corners in original DICOM pixel space. Populated only for `version='2026_1_15_cropping'`. |
| `mask_image` | TEXT | Mask PNG filename. Populated only for `version='2026_1_15_seg'` (~8% of rows). |
| `quality` | TEXT | Tier. Values: `gold` (14.0%), `silver` (51.2%), `bronze` (34.8%) |
| `version` | TEXT | See table above |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |

---

## CaliperLabels

**Grain**: one row per image with caliper annotation. **18,059 rows.**
**Join keys**: `dicom_hash` → `Images`; `accession_number` → `StudyCases`.
**Source**: CVAT annotation tasks. All rows are `quality='gold'`.
**Defined in**: [src/DB_processing/labels_database.py:138-151](../src/DB_processing/labels_database.py#L138-L151).

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | |
| `dicom_hash` | TEXT NOT NULL | → `Images.dicom_hash` |
| `has_calipers` | INTEGER (0/1) | 1 if calipers are present (55.3%), 0 if not |
| `caliper_points` | TEXT | JSON list of caliper-tip (x, y) coordinates |
| `n_points` | INTEGER | Count of caliper points in `caliper_points` |
| `split` | TEXT | Dataset split: `train` (70.1%) / `val` (14.5%) / `test` (15.3%) |
| `bi_rads` | TEXT | BI-RADS at annotation time. Mostly `1`–`6` as strings; subcategories `4A` / `4B` / `4C` also present. 0.7% null. |
| `quality` | TEXT | Tier. Values: `gold` |
| `accession_number` | TEXT | → `StudyCases.accession_number` |
| `version` | TEXT | CVAT task tag. Values: `task_5` (59.9%), `task_9` (40.1%) |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |

---

## CaseLabels

**Grain**: one row per `(accession_number, version)`. **96,622 rows = 48,311 accessions × 2 model versions.** Each accession is scored by **both** models.
**Join key**: `accession_number` → `cadbusi_source.StudyCases`.
**Source**: BERT text classifier(s) predicting BI-RADS 4 subcategory from exam text.
**Pipeline code**: BI-RADS text classification lives in [src/data_ingest/](../src/data_ingest/) (see `gemini_parsing.py`, `clean_radiology.py`, `findings_parser.py`, `classification.py`).

Two coexisting model versions, each populating a different subset of the prediction columns:

| `version` | Rows | Populates | 4b/4c columns |
|---|---:|---|---|
| `birads4a_bert_v1` | 48,311 | `pred_4a`, `pred_prob_4a` only | NULL |
| `birads_4abc_bert_v1` | 48,311 | `pred_4a`, `pred_4b`, `pred_4c` + probs | all populated |

The 4a-only version is the older model; the 4abc version is the newer one that also predicts 4b and 4c. Both are kept so experiments pinned to either remain reproducible.

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | |
| `accession_number` | TEXT NOT NULL | → `StudyCases` |
| `patient_id` | TEXT NOT NULL | → `StudyCases.patient_id` |
| `pred_4a` | INTEGER (0/1) | Thresholded class for BI-RADS 4a |
| `pred_prob_4a` | REAL | Model probability for 4a |
| `pred_4b` | INTEGER (0/1) | Thresholded class for 4b (NULL for `birads4a_bert_v1`) |
| `pred_prob_4b` | REAL | Probability for 4b (NULL for `birads4a_bert_v1`) |
| `pred_4c` | INTEGER (0/1) | Thresholded class for 4c (NULL for `birads4a_bert_v1`) |
| `pred_prob_4c` | REAL | Probability for 4c (NULL for `birads4a_bert_v1`) |
| `version` | TEXT | Model identifier. See table above. |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP |

---

## PathologyExtracted

**Grain**: one row per `accession_number` per extraction run. **34,171 rows (v1).** In v1 each accession appears exactly once; re-extractions will add rows under a new `version` string.
**Join key**: `accession_number` → `cadbusi_source.StudyCases` / `cadbusi_source.Pathology`. Combined with `version`, the pair is unique; `id` is the PK.
**Source**: Phase 1 LLM extraction from `cadbusi_source.StudyCases.rad_pathology_txt` — short pathology addenda appended to the imaging report (~34K non-null, avg ~200 chars). Coarse primary diagnosis + subtype + brief details.
**v1 model mix**: `gemini-2.5-flash` (25,091 rows) and `gemini-3-flash-preview` (9,080 rows). Per-row model is recorded in `model_name`.
**Pipeline code**: [src/data_ingest/](../src/data_ingest/) (pathology extraction runners live alongside the BI-RADS Gemini code). Imported from `gs://shared-aif-bucket-87d1/tristan_handoff_2026_04_22/pathology_dump/pathology_extracted.parquet`.

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | Synthetic row ID, assigned at import |
| `accession_number` | TEXT | → `StudyCases`. Unique in v1; not unique across future versions. |
| `source_text` | TEXT | Raw `rad_pathology_txt` fed to the LLM |
| `primary_diagnosis` | TEXT | `BENIGN` (62.0%) / `MALIGNANT` (32.0%) / `ATYPICAL` (5.1%) / `UNKNOWN` (0.3%) / `INSUFFICIENT` (0.3%) / `SUSPICIOUS` (0.3%) |
| `cancer_subtypes` | TEXT | Pipe-separated enum (e.g. `INVASIVE_DUCTAL_CARCINOMA`, `DUCTAL_CARCINOMA_IN_SITU`, `INVASIVE_LOBULAR_CARCINOMA`, `LOBULAR_CARCINOMA_IN_SITU`, `METASTATIC_CARCINOMA`, combinations with `\|`). Empty string if no carcinoma. |
| `benign_subtypes` | TEXT | Pipe-separated (e.g. `FIBROADENOMA`, `BENIGN_BREAST_TISSUE`, `LYMPH_NODE_BENIGN`, `FIBROSIS_OR_SCAR`, `FAT_NECROSIS`, `PAPILLOMA`, `OTHER_BENIGN`). Empty if malignant. |
| `laterality` | TEXT | `NOT_SPECIFIED` (66.6%) / `LEFT` (16.9%) / `RIGHT` (16.4%) / `BILATERAL` (0.2%) |
| `size_mm` | REAL | Largest dimension if stated. **88.2% null.** |
| `grade` | TEXT | Nottingham or nuclear grade if reported |
| `lymph_node_status` | TEXT | `NOT_REPORTED` (93.8%) / `NEGATIVE` (5.0%) / `POSITIVE` (1.1%) / `MICROMETASTASIS` / `ISOLATED_TUMOR_CELLS`. ~0% null. |
| `is_lymph_node_biopsy` | INTEGER (0/1) | 1 if the text describes a lymph-node (not primary) biopsy (6.1%) |
| `confidence` | TEXT | LLM self-assessed: `HIGH` (92.5%) / `LOW` (7.2%) / `MEDIUM` (0.3%) |
| `notes` | TEXT | Brief free-text note if not capturable in enums |
| `model_name` | TEXT | `gemini-2.5-flash` or `gemini-3-flash-preview` |
| `extraction_error` | TEXT | Populated if extraction failed |
| `created_at` | TEXT | Extraction timestamp |
| `version` | TEXT | Extraction version. Values: `v1` |

---

## PathologySynoptic

**Grain**: one row per synoptic specimen. **50,616 rows across 7,595 unique accessions.**
**Join key**: `accession_number` → `cadbusi_source.StudyCases` / `cadbusi_source.Pathology`. **Many rows per accession** (1–102; median ≈ 4) — multi-part specimens (breast core bx, axillary sentinel, later excision, etc.) each produce a row. Mirrors the grain of `cadbusi_source.Pathology`. `id` is the PK.
**Source**: Phase 2 LLM extraction (`gemini-2.5-flash`, all rows) from `cadbusi_source.Pathology.synoptic_report` — full formal synoptic pathology (avg ~3K chars). Structured breast-pathology schema with enums and sub-models (invasive, in situ, margins, lymph nodes, staging, receptors, treatment context, benign).
**Pipeline code**: [src/data_ingest/](../src/data_ingest/) (alongside Phase 1; see `synoptic_schema.py`, `synoptic_prompt.py`, `run_synoptic.py`). Imported from `gs://shared-aif-bucket-87d1/tristan_handoff_2026_04_22/pathology_dump/pathology_synoptic_extracted.parquet`. On import, the source column `path_id` was renamed to `id`.

**65 rows (0.1%)** are extraction failures — most structured fields NULL; check `extraction_error` before relying on values.

### Identifiers

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | Synthetic row ID (was `path_id` in source parquet) |
| `accession_number` | TEXT | → `StudyCases`. Not unique; 1–102 rows per accession. |

### Specimen

| Column | Type | Meaning |
|---|---|---|
| `specimen_type` | TEXT | `TOTAL_MASTECTOMY` / `LUMPECTOMY` / `EXCISIONAL_BIOPSY` / `SIMPLE_MASTECTOMY` / `SEGMENTAL_MASTECTOMY` / `CORE_BIOPSY` / `NOT_SPECIFIED` / `OTHER` / `RADICAL_MASTECTOMY` / `SENTINEL_LYMPH_NODE_BIOPSY` |
| `specimen_site` | TEXT | Anatomic site (free text, site-specific) |
| `laterality` | TEXT | `LEFT` (49.0%) / `RIGHT` (48.2%) / `NOT_SPECIFIED` / `BILATERAL` |
| `multi_part_specimen` | REAL (0/1) | 1 if specimen has multiple labeled parts (19.4%) |

### Primary diagnosis / invasive tumor

| Column | Type | Meaning |
|---|---|---|
| `primary_diagnosis` | TEXT | `INVASIVE_CARCINOMA` (81.4%) / `IN_SITU_ONLY` (14.8%) / `OTHER` / `UNKNOWN` / `BENIGN` / `METASTATIC_LYMPH_NODE` / `ATYPICAL` |
| `histologic_type` | TEXT | `INVASIVE_CARCINOMA_NO_SPECIAL_TYPE` / `INVASIVE_DUCTAL_CARCINOMA` / `INVASIVE_LOBULAR_CARCINOMA` / `MIXED_DUCTAL_AND_LOBULAR` / `INVASIVE_DUCTAL_WITH_LOBULAR_FEATURES` / `MUCINOUS_CARCINOMA` / `TUBULAR_CARCINOMA` / `METAPLASTIC_CARCINOMA` / `MICROPAPILLARY_CARCINOMA` / `PAPILLARY_CARCINOMA` / `APOCRINE_CARCINOMA` / `PHYLLODES` / `OTHER` / `NOT_APPLICABLE`. 17.2% null. |
| `grade_overall` | TEXT | `GRADE_1` / `GRADE_2` / `GRADE_3` / `NOT_APPLICABLE` / `NOT_REPORTED`. 17.2% null. |
| `nottingham_glandular` | REAL | Nottingham sub-score — tubule formation (1–3) |
| `nottingham_nuclear` | REAL | Nottingham sub-score — nuclear pleomorphism (1–3) |
| `nottingham_mitotic` | REAL | Nottingham sub-score — mitotic count (1–3) |
| `tumor_size_mm` | REAL | Largest invasive focus (mm) |
| `tumor_size_additional_mm` | TEXT | Additional foci sizes (free text when multiple) |
| `focality` | TEXT | `SINGLE` / `MULTIFOCAL` / `MULTICENTRIC` / `NOT_REPORTED` / `NOT_APPLICABLE`. 17.2% null. |
| `lymphovascular_invasion` | REAL (0/1) | LVI present |
| `dermal_lvi` | REAL (0/1) | Dermal LVI |
| `perineural_invasion` | REAL (0/1) | PNI present |

### In-situ disease

| Column | Type | Meaning |
|---|---|---|
| `dcis_present` | REAL (0/1) | DCIS present |
| `dcis_nuclear_grade` | TEXT | `LOW` / `INTERMEDIATE` / `HIGH` / `NOT_APPLICABLE` / `NOT_REPORTED`. 3.5% null. |
| `dcis_pattern` | TEXT | **JSON-encoded list** of DCIS patterns (e.g. `["Solid"]`, `["Cribriform", "solid"]`, `["comedo"]`, `[]`). Case-inconsistent — normalize before filtering. |
| `dcis_necrosis` | REAL (0/1) | Necrosis present within DCIS |
| `dcis_size_mm` | REAL | DCIS extent in mm |
| `lcis_present` | REAL (0/1) | LCIS present |

### Margins

| Column | Type | Meaning |
|---|---|---|
| `invasive_margin_status` | TEXT | `NEGATIVE` (73.9%) / `NOT_APPLICABLE` (16.2%) / `POSITIVE` (2.5%) / `CLOSE` (1.3%) / `NOT_REPORTED` (0.3%). 5.8% null. |
| `closest_invasive_margin_name` | TEXT | Named margin closest to tumor (free text, e.g. "posterior", "superior") |
| `closest_invasive_margin_mm` | REAL | Distance to closest invasive margin (mm) |
| `dcis_margin_status` | TEXT | Same enum as `invasive_margin_status`, scoped to DCIS |
| `closest_dcis_margin_mm` | REAL | Distance to closest DCIS margin (mm) |

### Lymph nodes

| Column | Type | Meaning |
|---|---|---|
| `ln_total_examined` | REAL | Total nodes examined |
| `ln_sentinel_examined` | REAL | Sentinel nodes examined |
| `ln_with_macromets` | REAL | Nodes with macrometastases (> 2 mm) |
| `ln_with_micromets` | REAL | Nodes with micrometastases (0.2–2 mm) |
| `ln_with_itc` | REAL | Nodes with isolated tumor cells (< 0.2 mm) |
| `ln_largest_deposit_mm` | REAL | Size of largest nodal deposit (mm) |
| `ln_extranodal_extension` | REAL (0/1) | ENE present |

### Staging (pTNM)

| Column | Type | Meaning |
|---|---|---|
| `pT` | TEXT | `pTis` / `pT1mi` / `pT1a` / `pT1b` / `pT1c` / `pT1` / `pT2` / `pT3` / `pT4` / `pT4a` / `pT4b` / `pT4d` / `pTx` / `NOT_REPORTED`. 5.6% null. |
| `pN` | TEXT | `pN0` / `pN0_i_plus` / `pN1mi` / `pN1` / `pN1a` / `pN1b` / `pN2` / `pN2a` / `pN2b` / `pN3a` / `pN3b` / `pN3c` / `pNx` / `NOT_REPORTED`. 5.6% null. |
| `pM` | TEXT | `NOT_APPLICABLE` (71.0%) / `NOT_REPORTED` (19.8%) / `pM0` / `cM0` / `pM1`. 5.6% null. |
| `sentinel_modifier` | REAL (0/1) | `(sn)` modifier applied to pN (60.1% set) |
| `overall_stage` | TEXT | AJCC overall stage. **~100% null in v1** (only 7 non-null rows). Treat as effectively unused. |

### Receptors / biomarkers

Receptor fields are **78.2% null** in v1 — only reported when the synoptic explicitly included biomarkers (mostly excision specimens, not core biopsies).

| Column | Type | Meaning |
|---|---|---|
| `er_status` | TEXT | `POSITIVE` / `NEGATIVE` / `LOW_POSITIVE` / `PENDING` / `NOT_REPORTED` / `NOT_PERFORMED` |
| `er_percent` | REAL | ER-positive nuclei (%) |
| `er_intensity` | TEXT | `WEAK` / `MODERATE` / `STRONG` (free text in source) |
| `pr_status` | TEXT | Same enum as `er_status` |
| `pr_percent` | REAL | PR-positive nuclei (%) |
| `pr_intensity` | TEXT | Same as `er_intensity` |
| `her2_status` | TEXT | `POSITIVE` / `NEGATIVE` / `EQUIVOCAL` / `PENDING` / `NOT_REPORTED` / `NOT_PERFORMED` |
| `her2_method` | TEXT | `IHC` / `FISH` / `DUAL` / `NOT_REPORTED` |
| `her2_ihc_score` | REAL | IHC score (0 / 1 / 2 / 3) |
| `her2_fish_ratio` | REAL | HER2/CEP17 ratio |
| `ki67_percent` | REAL | Ki-67 proliferation index (%) |

### Treatment context

| Column | Type | Meaning |
|---|---|---|
| `post_neoadjuvant` | REAL (0/1) | Specimen is post-neoadjuvant therapy |
| `treatment_effect` | TEXT | `NOT_APPLICABLE` (58.0%) / `PARTIAL_RESPONSE` / `COMPLETE_RESPONSE` / `NOT_REPORTED` / `NO_RESPONSE` / `MINIMAL_RESPONSE`. 18.5% null. |

### Benign / atypia

These columns are only populated for non-malignant specimens; **both are ~99.5% null** overall. Contents are **free text**, not a fixed enum — normalize before filtering.

| Column | Type | Meaning |
|---|---|---|
| `primary_benign_dx` | TEXT | Free text (e.g. "Fibrocystic changes", "fibroadenoma", "Phyllodes tumor, benign") |
| `atypia_present` | REAL (0/1) | Atypia present |
| `atypia_type` | TEXT | Free text (e.g. "ADH", "ALH", "atypical lobular hyperplasia", "Flat epithelial atypia") |

### Bookkeeping

| Column | Type | Meaning |
|---|---|---|
| `confidence` | TEXT | LLM self-assessed: `HIGH` (98.7%) / `MEDIUM` / `LOW`. 0.1% null (extraction failures). |
| `extraction_notes` | TEXT | Free-text notes from the extraction |
| `model_name` | TEXT | LLM model used. Values in v1: `gemini-2.5-flash` (all rows) |
| `extraction_error` | TEXT | Populated if extraction failed (~65 rows) |
| `created_at` | TEXT | Extraction timestamp |
| `version` | TEXT | Extraction version. Values: `v1` |

---

## Common join patterns

**Case-level labels + source metadata** (pin to a specific model/version):
```sql
ATTACH DATABASE 'cadbusi_source.db' AS src;

SELECT sc.accession_number, sc.bi_rads, sc.has_malignant,
       cl.pred_4a, cl.pred_prob_4a, cl.pred_4b, cl.pred_4c,
       pe.primary_diagnosis, pe.grade, pe.size_mm
FROM src.StudyCases sc
LEFT JOIN CaseLabels cl
  ON sc.accession_number = cl.accession_number
 AND cl.version = 'birads_4abc_bert_v1'
LEFT JOIN PathologyExtracted pe
  ON sc.accession_number = pe.accession_number
 AND pe.version = 'v1'
WHERE sc.bi_rads LIKE '4%';
```

**Synoptic pathology is 1:many** — either aggregate per accession or select specimen rows explicitly:
```sql
-- All specimens for an accession
SELECT id, specimen_type, primary_diagnosis, tumor_size_mm, pT, pN, er_status, her2_status
FROM PathologySynoptic
WHERE accession_number = ? AND version = 'v1'
ORDER BY id;
```

**Per-image labels** join on `dicom_hash`:
```sql
-- Instance labels + bounding boxes (cropping version)
SELECT i.image_name, il.malignant, ll.x1, ll.y1, ll.x2, ll.y2
FROM src.Images i
LEFT JOIN ImageLabels il ON i.dicom_hash = il.dicom_hash
LEFT JOIN LesionLabels ll
  ON i.dicom_hash = ll.dicom_hash
 AND ll.version = '2026_1_15_cropping';
```
