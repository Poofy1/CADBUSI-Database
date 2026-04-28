# Annotation Hub — Sample Data (2026-04-22 handoff)

This bundle contains the real masks, tags, position state, and a sample
exam batch from Jeff's local annotation work. Use it to stand up the
refactored annotation hub locally and verify it behaves correctly before
writing the Cloud Run / GCS / labels DB implementations.

## Contents

| Path | Size | Purpose |
|---|---|---|
| `drawn_masks/*.png` | 401 files, ~23 MB | User-drawn mask PNGs, keyed by clean_hash. Point `--masks-output` here to see them load as green overlays. |
| `tags.json` | 41 entries | Per-frame review tags (Accept/Edit/Needs Review/Reject). |
| `position.json` | 1 entry | Last nav position (which frame the tool should open on). |
| `frame_manifest.json` | ~730 KB | The frame manifest used by the mask tool. Point `--manifest` here. |
| `sample_batch/batch_manifest.json` | 20 exams | First 20 exams from the gold_review batch, for exam-mode testing. |
| `sample_batch/*.png` | 20 panel PNGs | Per-exam frame panels referenced by the manifest. |

## Running the hub against this sample

```bash
cd BUS_framework

# Assuming you've extracted this bundle to /tmp/annotation_hub_handoff/

python segmentation/annotation/scripts/active_learning/annotation_hub.py \
    --port 8765 \
    --images-dir segmentation/cvat/share \
    --masks-output /tmp/annotation_hub_handoff/drawn_masks \
    --annotations-dir /tmp/annotation_hub_handoff/annotations \
    --manifest /tmp/annotation_hub_handoff/frame_manifest.json \
    --batch-dir /tmp/annotation_hub_handoff/sample_batch \
    --exam-image-dir /tmp/annotation_hub_handoff/sample_batch
```

Note: `tags.json` and `position.json` paths are still hardcoded to
`/mnt/wsl_data/cadbusi/active_drawn_masks/` in config.py defaults;
either copy those two files to that path or edit the Config defaults
for your environment.

## What's here for testing

- **Drawn masks**: enough real masks to verify the read-back + green-overlay
  flow without making you draw new ones.
- **Tags**: verify the tag pill state restores correctly on frame load.
- **Position**: verify the app opens on the last-viewed frame.
- **Sample batch**: enough exams to exercise the exam-mode grid, zoom
  overlay, comment box, and label pill UI. 20 exams is small enough to
  keep the panel PNGs light (~5 MB) but big enough to click through a
  realistic workflow.

## Full batch (not in this bundle)

The complete gold_review batch has 988 exams with ~300 MB of panel PNGs.
If you need the full batch, it's at
`/mnt/wsl_data/cadbusi/labeling_batches/gold_review/` on Jeff's machine,
or you can pull it from GCS (ask Jeff for the path).

## Not included

- The cadbusi_source.db file — Jeff will upload the manifest DB separately
  to `gs://shared-aif-bucket-87d1/registry/bus_manifest_v3.db`.
- The Mayo source DB — that's yours, you have the current version.
- Pathology extraction tables — Jeff will dump those to
  `gs://shared-aif-bucket-87d1/registry/pathology_dump_2026_04_22/`
  for you to import into labels DB.
