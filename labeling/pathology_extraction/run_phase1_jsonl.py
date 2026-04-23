"""Phase 1 extraction via JSONL (VM-friendly variant).

Reads phase1_remaining_inputs.jsonl, extracts via Vertex or API key,
writes phase1_outputs.jsonl. Resume-safe.

Usage on GCP VM:
  gcloud auth application-default login    # one-time
  export VERTEX_PROJECT=your-project
  export VERTEX_LOCATION=us-central1
  python run_phase1_jsonl.py \\
      --input phase1_remaining_inputs.jsonl \\
      --output phase1_outputs.jsonl \\
      --workers 32
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

try:
    from data.pathology_extraction.schema import PathologyExtraction
    from data.pathology_extraction.prompts import SYSTEM_INSTRUCTION, build_prompt
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from schema import PathologyExtraction
    from prompts import SYSTEM_INSTRUCTION, build_prompt


DEFAULT_MODEL = "gemini-2.5-flash"


def build_client(use_vertex: bool, project: Optional[str], location: Optional[str]):
    if use_vertex:
        project = project or os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = location or os.environ.get("VERTEX_LOCATION") or "us-central1"
        if not project:
            raise RuntimeError("Vertex mode requires --project or VERTEX_PROJECT env var.")
        print(f"[Vertex AI] project={project}, location={location}", flush=True)
        return genai.Client(vertexai=True, project=project, location=location)
    else:
        key = os.environ.get("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY required for API-key mode.")
        print("[API key mode]", flush=True)
        return genai.Client(api_key=key)


def extract_one(client, model: str, text: str, max_retries: int = 2) -> dict:
    if not text or not text.strip():
        return {'error': 'empty_text'}
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=PathologyExtraction,
        temperature=0.0,
    )
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model, contents=build_prompt(text), config=config,
            )
            parsed = resp.parsed
            if parsed is None:
                parsed = PathologyExtraction.model_validate_json(resp.text or "{}")
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
                    if 'error' not in obj or obj.get('error') is None:
                        done.add(obj['accession_number'])
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--vertex', action='store_true', default=True)
    ap.add_argument('--api-key', action='store_true', help='Force API-key mode')
    ap.add_argument('--project', default=None)
    ap.add_argument('--location', default=None)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    use_vertex = args.vertex and not args.api_key
    client = build_client(use_vertex, args.project, args.location)

    with open(args.input) as f:
        items = [json.loads(line) for line in f]
    print(f"Input rows: {len(items):,}", flush=True)

    done = load_done(Path(args.output))
    print(f"Already done (in output file): {len(done):,}", flush=True)
    todo = [it for it in items if it['accession_number'] not in done]
    if args.limit > 0: todo = todo[:args.limit]
    print(f"Remaining: {len(todo):,}", flush=True)
    if not todo:
        return

    start = time.time()
    processed = 0
    errors = 0

    def _one(it):
        r = extract_one(client, args.model, it['rad_pathology_txt'])
        return {'accession_number': it['accession_number'], **r}

    with open(args.output, 'a', buffering=1) as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_one, it) for it in todo]
            for fut in as_completed(futures):
                r = fut.result()
                out.write(json.dumps(r) + '\n')
                processed += 1
                if 'error' in r and r['error']: errors += 1
                if processed % 100 == 0:
                    el = time.time() - start
                    rate = processed / max(el, 1e-6)
                    eta_min = (len(todo) - processed) / max(rate, 1e-6) / 60
                    print(f"[{processed}/{len(todo)}] ok={processed-errors} err={errors} "
                          f"rate={rate:.1f}/s ETA={eta_min:.1f}min", flush=True)

    print(f"\n✓ Done in {(time.time()-start)/60:.1f} min. Total: {processed}, errors: {errors}", flush=True)


if __name__ == '__main__':
    main()
