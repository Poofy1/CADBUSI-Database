# GCP VM Runbook — Pathology Extraction

Run both Phase 1 (rad_pathology_txt) and Phase 2 (synoptic_report) LLM extractions on a GCP VM using Vertex AI, then pull results back locally.

## VM

- **Instance**: `aif-vertex-cpu-us-central1-b`
- **Type**: n2-standard-8 (8 vCPU, 32 GB RAM — CPU-only, Vertex is IO-bound)
- **Region**: us-central1
- **GCP project**: (set via `$VERTEX_PROJECT`)

---

## Step 1 — Local: export inputs (≤ 5 min)

Already done once; re-run if the manifest DB has been updated.

```bash
cd ~/BUS_framework

# Phase 1: remaining rad_pathology_txt rows not yet extracted (resumes)
python data/pathology_extraction/export_phase1_remaining.py
# → data/pathology_extraction/phase1_remaining_inputs.jsonl (~3 MB, ~16K records)

# Phase 2: all Pathology.synoptic_report rows
python data/pathology_extraction/export_synoptic_inputs.py
# → data/pathology_extraction/synoptic_inputs.jsonl (~168 MB, ~50K records)
```

---

## Step 2 — Local: ship code + inputs to VM

Two options. Pick one.

### Option A: scp (simplest, no bucket needed)

```bash
# From local machine
VM_NAME=aif-vertex-cpu-us-central1-b
ZONE=us-central1-b

# Tar the extraction module + inputs
tar -czf /tmp/pathology_extraction.tar.gz -C ~/BUS_framework/data pathology_extraction

gcloud compute scp /tmp/pathology_extraction.tar.gz \
    $VM_NAME:~/ --zone=$ZONE

# On VM:
gcloud compute ssh $VM_NAME --zone=$ZONE
tar -xzf ~/pathology_extraction.tar.gz
cd ~/pathology_extraction

# Make the notebook visible from JupyterLab's root. Vertex Workbench opens
# JupyterLab rooted at the user's home directory, so the easiest path is to
# either untar directly into ~ (above already does) or symlink the notebook
# to the home dir for easy discovery:
ln -sf ~/pathology_extraction/pathology_extraction_monitor.ipynb ~/pathology_extraction_vm.ipynb
# Then in JupyterLab, open `pathology_extraction_vm.ipynb` at the root.
```

### Option B: via GCS bucket (better for larger DB transfers)

```bash
# Local
BUCKET=gs://your-mayo-bucket/pathology_extraction

gsutil cp data/pathology_extraction/*.jsonl $BUCKET/
gsutil cp data/pathology_extraction/*.py $BUCKET/
gsutil cp data/pathology_extraction/*.md $BUCKET/

# On VM
gsutil cp -r $BUCKET/ ~/pathology_extraction/
```

If you ever need the full manifest DB on the VM (e.g. for Tristan to run local-mode scripts), use the bucket — sqlite files are ~1 GB.

---

## Step 3 — VM: one-time setup

```bash
# SSH in
gcloud compute ssh aif-vertex-cpu-us-central1-b --zone=us-central1-b

# Python + deps (n2-standard-8 usually has Python 3.10+ preinstalled)
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv

python3 -m venv ~/pathvenv
source ~/pathvenv/bin/activate
pip install google-genai pydantic

# Auth for Vertex AI
# If the VM was created with a service account that has aiplatform.user role,
# ADC works automatically — no explicit login needed. To verify:
gcloud auth list
# If no credentials, do:
#    gcloud auth application-default login

# Set env vars (add to ~/.bashrc to persist)
export VERTEX_PROJECT=$(gcloud config get-value project)
export VERTEX_LOCATION=us-central1
echo "VERTEX_PROJECT=$VERTEX_PROJECT"
```

**Verify Vertex auth with a quick test:**

```bash
cd ~/pathology_extraction
python - <<'EOF'
from google import genai
import os
client = genai.Client(vertexai=True,
                      project=os.environ['VERTEX_PROJECT'],
                      location=os.environ['VERTEX_LOCATION'])
r = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Reply with the single word OK.'
)
print("Vertex OK:", r.text)
EOF
```

---

## Step 4 — VM: run extractions

Run Phase 1 and Phase 2 in separate tmux panes (they share Vertex quota but run independently).

```bash
cd ~/pathology_extraction

# Phase 1 — should take ~15-30 min at 32 workers
tmux new -s phase1
python run_phase1_jsonl.py \
    --input phase1_remaining_inputs.jsonl \
    --output phase1_outputs.jsonl \
    --workers 32

# Detach: Ctrl+b d

# Phase 2 — ~2-3 hours for 50K records at 32 workers (texts are ~10x longer)
tmux new -s phase2
python run_synoptic.py \
    --input synoptic_inputs.jsonl \
    --output synoptic_outputs.jsonl \
    --workers 32

# Detach: Ctrl+b d

# Monitor:
# tmux attach -t phase1
# tmux attach -t phase2
```

Both scripts are resume-safe: if a run crashes or you Ctrl+C, re-launching skips records already in the output JSONL.

**If you see 429 quota errors**: lower `--workers` (try 16 → 8). Vertex quotas differ across projects; you may need to request higher pay-as-you-go limits in GCP console (IAM & Admin → Quotas → search `aiplatform.googleapis.com/generate_content_requests_per_model_per_minute`).

---

## Step 5 — VM → local: pull results back

```bash
# Option A: scp
gcloud compute scp $VM_NAME:~/pathology_extraction/phase1_outputs.jsonl \
    ~/BUS_framework/data/pathology_extraction/ --zone=$ZONE
gcloud compute scp $VM_NAME:~/pathology_extraction/synoptic_outputs.jsonl \
    ~/BUS_framework/data/pathology_extraction/ --zone=$ZONE

# Option B: bucket
gsutil cp ~/pathology_extraction/*.jsonl $BUCKET/
# Locally:
gsutil cp $BUCKET/phase1_outputs.jsonl data/pathology_extraction/
gsutil cp $BUCKET/synoptic_outputs.jsonl data/pathology_extraction/
```

---

## Step 6 — Local: import into manifest DB

```bash
cd ~/BUS_framework

# Phase 1 → existing pathology_extracted table
python data/pathology_extraction/import_results.py \
    --phase 1 \
    --input data/pathology_extraction/phase1_outputs.jsonl

# Phase 2 → creates pathology_synoptic_extracted if missing, populates
python data/pathology_extraction/import_results.py \
    --phase 2 \
    --input data/pathology_extraction/synoptic_outputs.jsonl
```

Verify with:
```bash
sqlite3 data/registry/bus_manifest_v3.db <<'SQL'
SELECT COUNT(*) AS n, COUNT(extraction_error) AS errs FROM pathology_extracted;
SELECT COUNT(*) AS n, COUNT(extraction_error) AS errs FROM pathology_synoptic_extracted;
SELECT primary_diagnosis, COUNT(*) FROM pathology_synoptic_extracted
WHERE extraction_error IS NULL GROUP BY primary_diagnosis ORDER BY 2 DESC;
SQL
```

---

## Step 7 — Shut down the VM

Don't forget. n2-standard-8 is ~$0.50/hr.

```bash
gcloud compute instances stop aif-vertex-cpu-us-central1-b --zone=us-central1-b
```

---

## Cost + time summary

| Phase | Records | Model | Est. wall time (32w) | Est. cost |
|---|---|---|---|---|
| 1 | ~16K rad_pathology_txt | gemini-2.5-flash | ~30 min | ~$1 |
| 2 | ~50K synoptic_report | gemini-2.5-flash | ~2-3 hrs | ~$16 |

Plus VM time @ $0.50/hr × ~4 hrs = ~$2.

---

## What Tristan handles (separate work)

The extraction pipeline above works on the **pathology text we already have**. For Mayo data Jeff doesn't have access to yet — more accessions' synoptic reports, older archives, additional EHR fields — Tristan's job is to pull from Mayo source systems via BigQuery / Cloud Run. That workflow is different:

- Pull new records from Mayo clinical data warehouse → BQ staging tables → export to GCS → drop into this pipeline.
- Not something to run reactively on the VM; belongs in a scheduled Cloud Run job or Dataflow.

Not in scope here. Hand off to Tristan after Jeff's first pass on this extraction.
