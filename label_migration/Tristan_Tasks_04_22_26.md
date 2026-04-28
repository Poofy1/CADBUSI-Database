# Tristan Tasks — 2026-04-22 (self-directed week)

**Jeff is away from 2026-04-22 through the weekend; back Monday 2026-04-27.**
Monitoring email and Teams async through the week — use either for
blockers. For non-urgent discussion items, queue them in the
"Questions for Jeff" section at the bottom of this doc; we'll work
through them Monday.

This task doc is designed to stand on its own — there's no meeting to
substitute for reading it. Everything you need to do the work is here,
in the specs it points at, or in the bucket.

## Handoff bucket

Everything you need is at
**`gs://shared-aif-bucket-87d1/tristan_handoff_2026_04_22/`**.

Contents:

| Path in bucket | What |
|---|---|
| `README.md` | Index of the bucket contents |
| `bus_manifest_v3.db` | Jeff's manifest DB (2.3 GB). ATTACH this from your scripts to reach the research layer. |
| `pathology_dump/*.parquet` | Phase 1 + Phase 2 + diagnostic_biopsy_link as parquets |
| `pathology_extraction_code.tar.gz` | The pipeline code to drop into CADBUSI-Database repo |
| `annotation_hub_handoff_2026_04_22.tar.gz` | Sample data for testing the refactored app locally (401 masks, CVAT polygons, 20-exam batch) |
| `docs/THREE_DB_ARCHITECTURE.md` | **Read first.** The cadbusi/labels/manifest split. |
| `docs/MASK_SAVE_SPEC.md` | **Read second.** Mask save workflow spec; decisions are locked in. |
| `docs/annotation_hub_README.md` | Refactored app architecture, endpoints, what's stubbed |

Pull the whole directory with:
```bash
gsutil -m cp -r gs://shared-aif-bucket-87d1/tristan_handoff_2026_04_22/ ~/handoff/
```

---

## The architecture, in one paragraph

Three databases, one rule. **`cadbusi_source.db`** (yours): the stabilized
clinical record of a Mayo export; immutable after ingestion; drop-in replace
when a new export arrives. **`cadbusi_labels.db`** (yours): downstream
extraction on top of source — LLM extractions, image processing, OCR,
model inference, and (new) per-lesion annotation output from the labeling
app; expected to iterate, versioned independently. **`bus_manifest_v3.db`**
(Jeff's): research-layer tables that live on top — experiment cohorts, splits,
versioning infrastructure, custom research joins. The manifest ATTACHes the
other two, and `build_log` records which versions each table was built against.

Test for ambiguity: *"Would I ever want multiple versions of this coexisting?"*
Yes → labels. No, stable within an export → cadbusi. Experiment-specific
derivation → manifest. Full rule is in `docs/THREE_DB_ARCHITECTURE.md`.

Going forward, **you'll work primarily in labels**. Pathology extractions,
BI-RADS Gemini outputs, new annotation writes from the labeling app — all
go to labels. Manifest is read-mostly from your side; Jeff writes to it.

---

## This week's tracks

Two tracks, in priority order. Do Track 1 end-to-end before starting
Track 2 if you have to pick; there's more handshake work in Track 2
that can use Jeff back in the loop.

### Track 1 — Move pathology extraction to labels + CADBUSI-Database repo

**Goal**: Phase 1 and Phase 2 pathology extraction tables live in labels
and the pipeline code lives alongside your BI-RADS Gemini work so you
own extraction end-to-end going forward.

**Steps**:

1. **Pull the pathology dump** from
   `gs://shared-aif-bucket-87d1/tristan_handoff_2026_04_22/pathology_dump/`.
   Three parquet files:
   - `pathology_extracted.parquet` (34,171 rows, Phase 1 — rad_pathology_txt → structured)
   - `pathology_synoptic_extracted.parquet` (50,616 rows, Phase 2 — Pathology.synoptic_report → 33-enum schema)
   - `diagnostic_biopsy_link.parquet` (132,974 rows, **provided for reference only — stays in manifest, not your table to own**)

2. **Import Phase 1 and Phase 2 into labels DB**. Use whatever table
   names fit your existing naming convention (e.g.,
   `pathology_extracted_v1`, `path_synoptic_v1`). Document the naming
   in your schema notes. The schema is already clean — each parquet
   file is one-to-one with the SQLite tables it came from.

3. **Copy the extraction code into CADBUSI-Database**. Pull
   `pathology_extraction_code.tar.gz` and extract into your repo
   alongside the BI-RADS Gemini code. What's in it:
   - `schema.py` — Pydantic model for Phase 1 (rad_pathology_txt extraction)
   - `synoptic_schema.py` — Pydantic models for Phase 2 (33 enums + 7 sub-models: invasive, in_situ, margins, lymph_nodes, staging, receptors, treatment_context, benign). This is the heavy one — full structured breast pathology.
   - `prompts.py`, `synoptic_prompt.py` — Gemini prompts
   - `extractor.py`, `run_phase1_jsonl.py`, `run_synoptic.py` — runners
   - `import_results.py` — importer; point it at your labels DB
   - `export_phase1_remaining.py`, `export_synoptic_inputs.py` — input prep
   - `GCP_VM_RUNBOOK.md` — Vertex AI batch runbook
   - `README.md`, `__init__.py`

4. **Own the pipeline going forward**. When a new Mayo export arrives,
   run extraction on the new rad reports into labels. When you want to
   tweak prompts or schema, cut a new labels version file per the
   three-DB spec (don't edit in place once researchers start pinning
   to a version).

5. **Signal Jeff when done** (brief email / Teams) so he can delete the
   manifest-side copies and switch `bus_data` reads to point at labels.

**Acceptance criteria**:
- Phase 1 + Phase 2 parquets imported into labels with tables you can
  query from your analysis scripts.
- Extraction pipeline running from CADBUSI-Database repo, writing to labels.
- You've decided what to call the tables in labels and documented it.

**Est. effort**: half day for the import, half day for the code transplant + smoke test.

### Track 2 — Take ownership of the labeling app; plan Cloud Run deployment

**Goal**: you pick up Jeff's refactored `annotation_hub` app and extend it
with (a) per-lesion structured annotation, (b) a labels-DB backend, and
(c) Cloud Run deployment behind Mayo VPN. Dr. Ellis's VPN access is
being set up in parallel (see `docs/admin/ellis_labeling_scope.md`).

**Read in this order**, before writing any code:

1. **`docs/THREE_DB_ARCHITECTURE.md`** — where data lives.
2. **`docs/MASK_SAVE_SPEC.md`** — the authoritative mask-save workflow.
   Decisions locked in on 2026-04-22; read the "Decision log" at the
   bottom to see what we already argued out.
3. **`docs/annotation_hub_README.md`** — app architecture after the
   2026-04-22 refactor. Explains the Storage / AnnotationStore /
   LesionStore interfaces you'll implement against.

**Biggest scope flag from the mask-save decisions**:

Multi-mask-per-frame is **v1 scope, not deferred**. The mask page
needs a lesion-switcher affordance — the annotator picks which lesion
they're drawing a mask for before they save. This means `LesionStore`
can't ship as a v2 feature: it has to work before the lesion-grained
mask save endpoint can work.

Sequencing suggestion (matches §10 of `MASK_SAVE_SPEC.md`):

1. **Labels-DB schema** — create `lesion_annotations`,
   `frame_lesion_views`, `mask_artifacts`, `users`, `tasks` tables.
   Schema details are in `docs/tasks/Tristan_Tasks_04_22_26.md` Task 2b
   and `MASK_SAVE_SPEC.md` §2.
2. **Seed lesion_annotations** from `cadbusi.Lesions` — dedupe on
   `(accession_number, clock, distance_cm)` → one skeleton row per
   physical lesion. Expected ~300K lesions after dedupe (779K raw rows,
   most exams have multiple measurements per lesion).
3. **LLM-extract descriptors** from `cadbusi.Lesions.description` →
   pre-fill `shape / orientation / margin / echo / posterior /
   vascularity` columns. You already have the Gemini infrastructure;
   this is a smaller extraction than Phase 2 pathology (descriptions
   are single phrases, not full reports).
4. **Implement `LabelsDBLesionStore`** against those tables. Stub
   interface is in `annotation_hub/storage.py`.
5. **Migrate legacy masks** — 401 PNGs from `annotation_hub_handoff`
   bundle; assign to lesions via the dedupe key; write
   `mask_artifacts` rows with `stage=preparer_pass, format=png`.
   Raw PNGs go to
   `gs://shared-aif-bucket-87d1/annotation_hub/masks/{frame_lesion_id}.preparer.png`.
6. **Add `routes/lesion_api.py`** and
   **`routes/frame_lesion_mask_api.py`** (or merge into the former)
   with the endpoints in §9 of `MASK_SAVE_SPEC.md`.
7. **UI work** — lesion switcher on the mask page, lesion list panel
   on the exam page, per-lesion characteristics form. This is where
   the Q6 scope decision is felt most.
8. **GCS storage** — fill in `GCSStorage` and `GCSAnnotationStore`
   stubs when deploying to Cloud Run.
9. **Auth / users** — IAP headers feed the `users` table; preparer
   vs radiologist role gating.

**Acceptance criteria for this week**:

Track 2 is too big to finish in one week. Target by end of this week:

- Labels-DB schema created (steps 1 above).
- Lesion_annotations seeded via dedupe of `cadbusi.Lesions` (step 2).
- LLM descriptor extraction running on `Lesions.description` (step 3,
  can be in progress; full corpus run is fine to complete next week).
- `LabelsDBLesionStore` implemented against the seeded tables (step 4).
- Legacy mask migration plan drafted (step 5, execution can be next week).

The API + UI + GCS + auth work (steps 6–9) is Monday-onward, after
Jeff is back to triage UI decisions.

**Est. effort this week**: ~3 days, mostly DB schema + seeding + the
LesionStore implementation.

### Tristan's schema (proposed — refine as you implement)

From Task 2b in the original version of this doc; reproduced for
reference:

```sql
lesion_annotations(
  lesion_annot_id PK,
  source_lesion_ids TEXT,      -- JSON list of cadbusi.Lesions.lesion_id rows collapsed
  accession_number, laterality, clock_hr, distance_cm, size_mm,
  shape, orientation, margin_clarity, margin_detail,
  echo_pattern, posterior_features, calcifications, vascularity,
  pathology, notes,
  preparer_user_id, preparer_ts,
  radiologist_user_id, radiologist_ts,
  stage,                        -- preparer_pass / radiologist_review / finalized
  status
)

frame_lesion_views(
  frame_lesion_id PK,
  dicom_hash,
  lesion_annot_id FK,
  view_type,                    -- LONG / TRANS / DOPPLER / OBLIQUE / OTHER
  doppler_on BOOL,
  is_best_view BOOL
)

mask_artifacts(
  mask_id PK,
  frame_lesion_id FK,
  stage,                        -- preparer_pass | radiologist_review
  format,                       -- png | polygon
  gcs_path_json,                -- always present
  gcs_path_png,                 -- present only when format = png
  image_width, image_height,
  annotator_user_id FK,
  created_at,
  UNIQUE(frame_lesion_id, stage)
)

users(user_id, name, role)
tasks(task_id, target_type, target_id, assignee_user_id, stage, status)
```

---

## Carryover items from last week (status checks)

Still on your plate from `Tristan_Tasks_04_15_26.md`. No new pressure from
Jeff this week — if you can get to them, great, if not, they queue for Monday:

- [ ] **Caliper twin matching patch** — applied / rebuild scheduled?
- [ ] **Dual-image split schema** (`derived_from` + `derived_kind` + `split_pair_id`) — decision + implementation status
- [ ] **Clock position + nipple distance coverage** — count on `cadbusi.Images.clock_pos IS NOT NULL`
- [ ] **Cine export sampling rule** — bucket export downsampled or full clip?

---

## Questions for Jeff (queue here, he'll answer Monday)

Put any non-urgent design questions here. I'll paste answers inline when I'm back.

- _(Tristan: add questions here as you hit them)_

---

## Communication

- **Blockers** → email (baggett.jeff@gmail.com) or Teams. Async monitored
  through the week; same-day for short questions, next-business-day for
  larger ones.
- **Progress updates** → whenever convenient. A short email or Teams
  message Friday summarizing the week's work would be ideal so Jeff
  walks into Monday with context.
- **Non-urgent design questions** → add to the section above.

## Pointer summary

| Where | What |
|---|---|
| `gs://shared-aif-bucket-87d1/tristan_handoff_2026_04_22/` | Everything you need |
| `docs/THREE_DB_ARCHITECTURE.md` | Three-DB split, rule for ambiguity, attach pattern |
| `segmentation/annotation/scripts/active_learning/MASK_SAVE_SPEC.md` | Mask save workflow (decisions locked in) |
| `segmentation/annotation/scripts/active_learning/README.md` | Refactored app architecture |
| `docs/admin/ellis_labeling_scope.md` | Ellis VPN request (for context on app deployment timing) |
| `data/pathology_extraction/` (in BUS_framework repo) | Source of the extraction code you're moving |
