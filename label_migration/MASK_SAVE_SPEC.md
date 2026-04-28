# Mask Save Workflow — Spec

**Status**: decisions landed, ready for Tristan to implement
**Last updated**: 2026-04-22
**Scope**: how the annotation hub persists masks, how they connect to the
lesion/frame data model, and what the storage layout looks like on GCS.

## 1. Principles

- **App is a capture tool, not a processor.** Whatever the annotator produced is
  what gets saved, byte-for-byte. No server-side smoothing, rasterization,
  or format rewriting on save.
- **Save format follows the input tool.** Brush drawing → binary PNG. Polygon
  editing (vertex-based, when that tool exists) → polygon JSON. Both are
  first-class.
- **Smoothing is cosmetic only.** The "Smooth" button is a client-side
  preview, like opacity or zoom. It does not change what the save endpoint
  persists. If downstream work needs smoothed polygons, downstream tools
  extract and smooth from the raw saved artifact.
- **Masks belong to a (frame, lesion) pair.** Per the lesion-grained
  annotation model (see `docs/tasks/Tristan_Tasks_04_22_26.md` Task 2b and
  `data/registry/THREE_DB_ARCHITECTURE.md`), a mask is never "a mask on a
  frame" without a lesion attached. If a frame shows two lesions, each has
  its own mask.
- **Two versions per mask, max.** Preparer version and radiologist version.
  Within-stage re-saves overwrite.

## 2. How masks connect to the data model

Recap from the lesion-grained design:

```
lesion_annotations        — one row per physical lesion per exam
  lesion_annot_id PK, accession, laterality, clock, distance, characteristics, ...

frame_lesion_views        — association: which frames show which lesion, in which view
  frame_lesion_id PK, dicom_hash, lesion_annot_id FK,
  view_type, doppler_on, is_best_view
```

**Masks attach at the `frame_lesion_views` grain.** One mask per `(frame, lesion)`
pair per stage. A lesion seen in three views and masked in all three has three
preparer masks + up to three radiologist masks. The "canonical" mask for
descriptor extraction and downstream training is whichever view has
`is_best_view=True`.

New labels-DB table:

```sql
mask_artifacts(
  mask_id            INTEGER PK,
  frame_lesion_id    INTEGER FK,
  stage              TEXT,      -- 'preparer_pass' | 'radiologist_review'
  format             TEXT,      -- 'png' | 'polygon'
  gcs_path_json      TEXT,      -- always present (metadata + polygon if applicable)
  gcs_path_png       TEXT,      -- present only if format = 'png'
  image_width        INTEGER,   -- native image dimensions at save time
  image_height       INTEGER,
  annotator_user_id  INTEGER FK,
  created_at         TIMESTAMP,
  UNIQUE(frame_lesion_id, stage)
)
```

The UNIQUE constraint enforces the two-max-per-mask rule (decision Q3d).

## 3. Save format

### JSON envelope (always written)

`gs://bucket/annotation_hub/masks/{frame_lesion_id}.{stage}.json`:

```json
{
  "frame_lesion_id": 123,
  "dicom_hash": "abc...",
  "lesion_annot_id": 456,
  "stage": "preparer_pass",
  "format": "png",
  "image_size": [960, 720],
  "polygon": null,
  "annotator": "jbaggett",
  "version": 1,
  "timestamp": "2026-04-22T14:30:00Z"
}
```

If `format == "polygon"`, the `polygon` field holds the vertex list
`[[x,y], ...]` in native image pixel coordinates.

### Binary PNG (present only for `format == "png"`)

`gs://bucket/annotation_hub/masks/{frame_lesion_id}.{stage}.png`:

1-channel (L-mode) binary PNG at native image resolution. The app saves
exactly what the annotator produced — no binarization beyond what the
overlay already does, no resolution change.

### Why not dual-save (PNG + polygon) by default?

Per decisions Q1/Q2/Q4: the app doesn't produce a polygon from a PNG
drawing, and it doesn't produce a PNG from polygon vertices. Format-conversion
is a downstream pipeline concern. A batch job can convert PNGs to polygons
(or vice versa) when some downstream tool needs the other format; those
derived artifacts live in a separate GCS prefix (e.g.,
`derived_polygons/` or `derived_masks/`) and don't mingle with the
authoritative capture records.

## 4. Input format — app accepts PNG or polygon

The save endpoint accepts whichever format the UI produced:

```
POST /api/frame_lesion/{frame_lesion_id}/mask

  Body (brush-drawing path):
    {"format": "png", "mask_base64": "...", "stage": "preparer_pass"}

  Body (polygon-edit path, future once vertex-edit UI exists):
    {"format": "polygon", "polygon": [[x,y], ...], "stage": "preparer_pass"}
```

Stage is determined from the user's role and current task state, not
passed by the browser — this is a server-side concern so the client can't
falsify stage.

Smoothing sliders on the mask page are **preview only** — they toggle a
CSS/canvas rendering of the smoothed contour but don't change what the
POST body contains.

## 5. GCS layout

```
gs://shared-aif-bucket-87d1/annotation_hub/
  masks/
    {frame_lesion_id}.preparer.json       # always present once preparer saves
    {frame_lesion_id}.preparer.png        # present only if format=png
    {frame_lesion_id}.radiologist.json    # present once radiologist reviews
    {frame_lesion_id}.radiologist.png     # present only if format=png
  preannotations/
    cvat/{dicom_hash}.json                # CVAT polygon exports, preview-only source
    sam/{dicom_hash}.json                 # (future) SAM-generated polygons
  annotations/
    {clean_hash}.json                     # legacy frame-grain pathology+descriptors
                                          # superseded by lesion_annotations table
                                          # once lesion-grain is wired
  batches/
    {batch_id}/
      batch_manifest.json
      panels/*.png
  frame_manifest.json
  _tags/{user_id}.json                    # per-user review status tags
  _position/{user_id}.json                # per-user last nav position
```

Pre-annotations and user-saved masks are strictly separated, per decision
Q5. The app loads pre-annotations as read-only overlays; when an annotator
saves, they produce a new `masks/...` artifact, leaving the preannotation
intact for provenance.

Tristan is encouraged to adjust the exact layout during Cloud Run
deployment if GCS-specific concerns (cache settings, lifecycle rules,
IAM prefixes) argue for a different shape.

## 6. Versioning and stage transitions

Per decision Q3d, exactly two stages persist per (frame, lesion) pair:

- **`preparer_pass`**: first-pass annotation by Jeff or other preparers.
  Within-stage re-saves overwrite the same GCS object and bump version in
  the JSON.
- **`radiologist_review`**: adjudication by Ellis. Creates the
  `radiologist_review` objects; does not delete `preparer_pass`. Within-stage
  re-saves overwrite the same radiologist object.

Both versions are kept so downstream analysis can learn from preparer →
radiologist deltas (which cases get adjusted, by how much, where
disagreement concentrates).

A `mask_edits` log table is not needed at this stage — the two snapshots
capture the only diff that matters (preparer output vs radiologist final).

## 7. Pre-annotation workflow

CVAT polygons (the ~2600 already loaded by the app) get handled as follows:

1. On startup, the app reads them into `PRELOADED_MASKS` as today. No change.
2. Separately, an offline script exports the CVAT polygons to
   `gs://...annotation_hub/preannotations/cvat/{dicom_hash}.json` so they
   exist as persistent preview data in GCS.
3. When an annotator opens a frame with a CVAT preannotation and no
   existing preparer_pass mask, the UI shows the CVAT polygon as a
   starting point.
4. If the annotator saves (as preparer), a new `masks/...preparer.*`
   artifact is created. The CVAT polygon stays in `preannotations/` for
   the audit trail.
5. Jeff's earlier note: CVAT polygons will be uploaded separately "for me
   to adjust and radiologist to adjudicate." That workflow runs the same
   path as any other preparer → radiologist task.

## 8. Migration of existing data

### 401 legacy drawn masks

These are 4-channel binary PNGs at `/mnt/wsl_data/cadbusi/active_drawn_masks/`
keyed by `clean_hash`. Under the new data model they need a `frame_lesion_id`
and need to land in `mask_artifacts`.

Migration script (Tristan territory, since it touches labels DB):

1. For each legacy `{clean_hash}.png`:
   a. Look up frame via `Images.dicom_hash = clean_hash`.
   b. Find the lesion for that frame via `cadbusi.Lesions` dedupe on
      `(accession, clock, distance_cm)` — this also populates
      `lesion_annotations`.
   c. Insert a `frame_lesion_views` row linking the frame to the lesion.
   d. Upload the PNG as-is to `masks/{frame_lesion_id}.preparer.png`.
      No smoothing or format conversion — per Q2, we keep the raw artifact.
   e. Write the companion JSON with `format: "png"`, `stage: "preparer_pass"`,
      `annotator: "jbaggett"`, `version: 1`, timestamp from file mtime.
   f. Insert a `mask_artifacts` row.

2. Edge cases to handle:
   - Legacy mask for a frame with no `Lesions` row (rare but happens for
     caliper-bearing frames): create a `lesion_annotations` row with
     `clock=NULL, distance=NULL, source="legacy_migration"`. Flag for
     later cleanup by a radiologist.
   - Legacy mask for a frame with multiple `Lesions` rows: require
     manual disambiguation. Small N; flag and surface in a report.

### CVAT polygons (~2600)

These already have `dicom_hash` but not `frame_lesion_id`. Same
disambiguation step as above — each CVAT polygon maps to a lesion via
the `cadbusi.Lesions` dedupe. After that, upload to
`preannotations/cvat/{dicom_hash}.json`; when an annotator opens the
associated (frame, lesion) the preannotation is offered.

## 9. API endpoints this implies

In addition to the existing mask endpoints, new ones for lesion-grained work:

```
GET    /api/exam/{bag_id}/lesions               → list lesions for an exam
GET    /api/lesion/{lesion_annot_id}             → full lesion record (characteristics, views, masks)
POST   /api/lesion                               → create new lesion (preparer only)
PATCH  /api/lesion/{lesion_annot_id}             → update characteristics / notes
POST   /api/lesion/{lesion_annot_id}/link_frame  → link a dicom_hash to this lesion (→ frame_lesion_views)

GET    /api/frame_lesion/{frame_lesion_id}/mask  → current mask (JSON + PNG if applicable)
POST   /api/frame_lesion/{frame_lesion_id}/mask  → save mask in current stage (body: {format, mask_base64|polygon})
DELETE /api/frame_lesion/{frame_lesion_id}/mask  → remove (current stage only)
```

Existing endpoints that get deprecated once the lesion layer lands:
`/api/mask/{clean_hash}` (GET/POST), `/api/annotations/{clean_hash}`
(GET/POST). Keep them for backward-compat through the migration window;
remove once all legacy masks are re-keyed to `frame_lesion_id`.

## 10. Implementation ownership

Per decision Q10, **Tristan owns the implementation.** Jeff's refactored
app provides the seams (`LesionStore`, `Storage` interfaces, route
modules); Tristan fills in the GCS-backed and labels-DB-backed
implementations plus the UI work for lesion switching and polygon editing.

### Where the seams already exist

- `storage.py::Storage` — `write_mask_png`, `read_mask_png`, `image_bytes`.
  Needs extension to `write_mask_artifact(frame_lesion_id, stage, format, bytes_or_polygon, metadata)`.
- `storage.py::LesionStore` (Protocol) — stub for lesion CRUD and
  frame-link operations. Tristan implements against labels DB.
- `storage.py::GCSStorage` / `GCSAnnotationStore` — stubs for Cloud Run.
- `routes/` — each router takes injected storage/store objects, so new
  lesion routes plug in without touching existing modules.

### Where new work lives

- New router `routes/lesion_api.py` for the `/api/lesion/*` and
  `/api/frame_lesion/*` endpoints.
- New router `routes/frame_lesion_mask_api.py` (or merged into
  `lesion_api.py`) for the mask endpoints. Existing
  `routes/masks_api.py` stays for backward-compat during migration.
- UI: lesion switcher on the mask page (dropdown or tab strip); lesion
  list panel on the exam page replacing/augmenting the current frame
  grid. This is where **decision Q6 bites** — multi-mask-per-frame in v1
  means the UI work is larger than "refactor + GCS," not a small follow-up.

### Expected sequencing

1. Tristan lands `lesion_annotations` + `frame_lesion_views` + `mask_artifacts` + `users` + `tasks` tables in labels DB.
2. Dedupe `cadbusi.Lesions` to seed `lesion_annotations`; LLM-extract descriptors from `Lesions.description`.
3. Implement `LabelsDBLesionStore` against those tables.
4. Migrate the 401 legacy masks + CVAT polygons into the new layout.
5. Build the new API routes (lesion + frame_lesion_mask).
6. Rebuild the mask page UI around the (lesion, frame) grain — lesion switcher + per-(frame,lesion) mask load/save.
7. Rebuild the exam page around lesion-level workflow (list lesions with characteristics panel, launch mask drawing per lesion per frame).
8. Implement `GCSStorage` for images + mask artifacts when deploying to Cloud Run.
9. Implement user layer + auth (IAP or similar).

Steps 1–3 block everything after. Step 4 can run in parallel with 5–6.

## 11. Risks to name

- **Lesion seeding ambiguity.** `cadbusi.Lesions` dedupe on
  `(accession, clock, distance)` will be imperfect. Cases with NULL clock
  or duplicate reports will need human adjudication. Budget for a
  manual-review pass during migration.
- **Legacy mask disambiguation.** 401 masks × some fraction will map to
  frames with unclear or missing lesion rows. Expect maybe 5–10% to need
  manual assignment.
- **UI scope.** Q6(a) is a bigger UI task than it looks. The current
  mask page assumes one drawing context per frame; multi-lesion support
  means lesion state is first-class and must survive frame navigation.
  Tristan should budget accordingly.
- **Smoothing-is-cosmetic needs to be obvious in the UI.** If annotators
  can click Smooth, see a changed display, and save — they will expect
  the save to match what they see. Needs a clear visual affordance (e.g.,
  the smoothed overlay is distinctly styled — dashed outline, different
  color — to signal "preview only").

## 12. Not covered by this spec

- **Prototype embedding / descriptor prediction**. Separate workstream
  (see `docs/papers/paper7_descriptor_prediction/`).
- **SAM interactive mode**. Future feature; slots in as a new entry under
  `preannotations/sam/` with the same read-only semantics as CVAT.
- **Multi-annotator collaboration** beyond preparer/radiologist (e.g.,
  two preparers on the same case). Not in v1; would require extending
  `mask_artifacts` with an annotator-specific stage variant or a separate
  `drafts` layer.

## Decision log

For the record — these were resolved in the Q&A reply on 2026-04-22:

| Question | Decision |
|---|---|
| Server-side smoothing on save? | **No** — app is a capture tool; smoothing is cosmetic client-side preview only. Downstream pipelines smooth as needed. |
| Raw vs smoothed as authoritative? | **Raw** — what the annotator produced is what gets saved. |
| Versioning on re-save? | **Two snapshots per mask max** (preparer + radiologist); within-stage re-saves overwrite. |
| Save format (PNG vs polygon)? | **Both** — follows the input tool. JSON metadata always written; PNG sibling file only when `format=png`. |
| Existing 401 masks migration? | **One-shot migration to new schema**, preserving raw PNG (no smoothing). |
| CVAT polygons? | **Separate `preannotations/cvat/` prefix** — preview-only, persisted for provenance, never overwritten by user saves. |
| Multi-mask-per-frame in v1? | **Yes** — lesion-switcher UI, one save per (frame, lesion) pair, `mask_artifacts` keyed by `frame_lesion_id`. |
| GCS storage layout? | **Approved as proposed**; Tristan to adjust if GCS-specific concerns arise. |
| Tags / nav position per-user? | **Per-user JSON files** — `_tags/{user_id}.json`, `_position/{user_id}.json`. |
| Smoothing defaults? | **Looser than current**, user-adjustable cosmetic slider, tune from Ellis feedback. |
| Implementation owner? | **Tristan**, using the refactored app seams. |
