"""Phase 2 extraction: run structured extraction on Pathology.synoptic_report via Vertex AI.

Designed to run on a GCP VM (n2-standard-8 is fine; Vertex is IO-bound).

Inputs: JSONL with {path_id, accession_number, synoptic_report} per line.
Outputs: JSONL with {path_id, accession_number, extraction (dict), error} per line.

Resume-safe: if the output file exists, skips path_ids already present in it.

Usage on GCP VM:
  gcloud auth application-default login                 # once, for ADC
  export VERTEX_PROJECT=your-project-id
  export VERTEX_LOCATION=us-central1

  python run_synoptic.py \\
      --input synoptic_inputs.jsonl \\
      --output synoptic_outputs.jsonl \\
      --workers 32

Usage locally with API key (for debugging only — don't try 50K on consumer quota):
  python run_synoptic.py --input synoptic_inputs.jsonl --output synoptic_outputs.jsonl \\
      --workers 8 --api-key
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

# Local imports work whether run from repo root or from VM after scp of this dir
try:
    from data.pathology_extraction.synoptic_schema import SynopticExtraction
    from data.pathology_extraction.synoptic_prompt import (
        SYNOPTIC_SYSTEM_INSTRUCTION, build_synoptic_prompt
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from synoptic_schema import SynopticExtraction
    from synoptic_prompt import SYNOPTIC_SYSTEM_INSTRUCTION, build_synoptic_prompt


DEFAULT_MODEL = "gemini-2.5-flash"


def build_client(use_vertex: bool, project: Optional[str], location: Optional[str],
                 api_key: Optional[str]):
    if use_vertex:
        project = project or os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = location or os.environ.get("VERTEX_LOCATION") or "us-central1"
        if not project:
            raise RuntimeError("Vertex mode requires --project or VERTEX_PROJECT env var.")
        client = genai.Client(vertexai=True, project=project, location=location)
        print(f"[Vertex AI] project={project}, location={location}")
        return client
    else:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Either --vertex (with GCP ADC) or GEMINI_API_KEY required.")
        print("[API key mode]")
        return genai.Client(api_key=key)


def extract_one(client, model: str, text: str, max_retries: int = 2) -> dict:
    """Call Gemini; return a dict with either 'extraction' or 'error'."""
    if not text or not text.strip():
        return {'error': 'empty_text'}

    config = types.GenerateContentConfig(
        system_instruction=SYNOPTIC_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=SynopticExtraction,
        temperature=0.0,
    )
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=build_synoptic_prompt(text),
                config=config,
            )
            # Prefer resp.parsed; fall back to re-parsing text
            parsed = resp.parsed
            if parsed is None:
                raw = resp.text or "{}"
                parsed = SynopticExtraction.model_validate_json(raw)
            return {'extraction': parsed.model_dump(mode='json')}
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:300]}"
            if attempt < max_retries:
                time.sleep(2.0 * (attempt + 1))
    return {'error': last_err}


def load_done(output_path: Path) -> set:
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(int(obj['path_id']))
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='JSONL of {path_id, accession_number, synoptic_report}')
    ap.add_argument('--output', required=True, help='JSONL of extraction results (resumable)')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--vertex', action='store_true', default=True, help='Use Vertex AI (default on GCP VM)')
    ap.add_argument('--api-key', action='store_true', help='Force API-key mode (for local debug)')
    ap.add_argument('--project', default=None)
    ap.add_argument('--location', default=None)
    ap.add_argument('--limit', type=int, default=0, help='Stop after N records (0 = all)')
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input {input_path} not found.")

    use_vertex = args.vertex and not args.api_key
    client = build_client(use_vertex, args.project, args.location, None)

    # Load inputs
    with open(input_path) as f:
        items = [json.loads(line) for line in f]
    print(f"Input rows: {len(items):,}")

    # Resume
    done = load_done(output_path)
    print(f"Already done (in output file): {len(done):,}")
    todo = [it for it in items if it['path_id'] not in done]
    print(f"Remaining: {len(todo):,}")
    if args.limit > 0:
        todo = todo[:args.limit]
        print(f"Limited to first {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    start = time.time()
    processed = 0
    errors = 0

    def _one(it):
        r = extract_one(client, args.model, it['synoptic_report'])
        return {
            'path_id': it['path_id'],
            'accession_number': it['accession_number'],
            **r,
        }

    # Open output in append mode, line-buffered
    with open(output_path, 'a', buffering=1) as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_one, it) for it in todo]
            for fut in as_completed(futures):
                r = fut.result()
                out.write(json.dumps(r) + '\n')
                processed += 1
                if 'error' in r: errors += 1
                if processed % 100 == 0:
                    elapsed = time.time() - start
                    rate = processed / max(elapsed, 1e-6)
                    eta_min = (len(todo) - processed) / max(rate, 1e-6) / 60
                    print(f"[{processed}/{len(todo)}] ok={processed-errors} err={errors} "
                          f"rate={rate:.1f}/s ETA={eta_min:.1f}min", flush=True)

    total_min = (time.time() - start) / 60
    print(f"\n✓ Done in {total_min:.1f} min. Total: {processed}, errors: {errors}")


if __name__ == '__main__':
    main()
