"""Gemini-based structured extraction of Mayo pathology addenda.

Uses google.genai's structured-output feature with our Pydantic schema.
Requires GEMINI_API_KEY in the environment (set in ~/.bashrc).
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types
from pydantic import ValidationError

from .schema import PathologyExtraction
from .prompts import SYSTEM_INSTRUCTION, build_prompt


DEFAULT_MODEL = "gemini-2.5-flash"   # GA model; preview has tight quotas


@dataclass
class ExtractionResult:
    """Wraps one extraction call outcome."""
    accession: str
    text: str
    extraction: Optional[PathologyExtraction]
    error: Optional[str] = None
    retry_count: int = 0


class PathologyExtractor:
    """Thin wrapper around google.genai with retries and structured output.

    Two auth modes:
      1. Direct Gemini API (consumer-tier): pass api_key= or set GEMINI_API_KEY.
         Subject to per-project RPM/TPM quotas; preview models have tight limits.
      2. Vertex AI (GCP): set use_vertex=True and either pass project/location
         or set VERTEX_PROJECT / VERTEX_LOCATION env vars. Uses Application
         Default Credentials (ADC) — run `gcloud auth application-default login`
         locally, or rely on the service account on a GCP VM. No API key needed.
         Vertex typically has higher quotas and pay-as-you-go pricing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        use_vertex: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        if use_vertex or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
            self.mode = "vertex"
            self.project = project or os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            self.location = location or os.environ.get("VERTEX_LOCATION") or "us-central1"
            if not self.project:
                raise RuntimeError(
                    "Vertex AI mode requires a GCP project. Set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT, "
                    "or pass project= to the constructor. Also ensure ADC is configured "
                    "(`gcloud auth application-default login`)."
                )
            self.client = genai.Client(vertexai=True, project=self.project, location=self.location)
            self.api_key = None
        else:
            self.mode = "api_key"
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. Source ~/.bashrc or pass use_vertex=True "
                    "(with VERTEX_PROJECT set) for GCP Vertex AI auth."
                )
            self.client = genai.Client(api_key=self.api_key)
            self.project = None
            self.location = None
        self.model = model

    def extract_one(
        self,
        text: str,
        accession: str = "",
        max_retries: int = 2,
        retry_delay_sec: float = 2.0,
    ) -> ExtractionResult:
        """Extract one pathology addendum. Returns ExtractionResult with optional error."""
        if not text or not text.strip():
            return ExtractionResult(
                accession=accession, text=text, extraction=None, error="empty_text"
            )

        prompt = build_prompt(text)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=PathologyExtraction,
            temperature=0.0,   # deterministic for extraction
        )

        last_err = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                parsed = response.parsed
                if parsed is None:
                    # Fallback: parse the text ourselves
                    import json
                    raw = response.text or "{}"
                    data = json.loads(raw)
                    parsed = PathologyExtraction.model_validate(data)
                return ExtractionResult(
                    accession=accession, text=text, extraction=parsed, retry_count=attempt
                )
            except ValidationError as e:
                last_err = f"schema_validation_failed: {str(e)[:200]}"
                # Don't retry on validation errors — usually a prompt issue, not transient
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt < max_retries:
                    time.sleep(retry_delay_sec * (attempt + 1))
                    continue

        return ExtractionResult(
            accession=accession, text=text, extraction=None, error=last_err, retry_count=max_retries
        )

    def extract_batch(
        self, items: list, show_progress: bool = True
    ) -> list:
        """Extract a batch of (accession, text) tuples. Serial; Gemini has generous rate limits.

        For the 34K full run, use async / threading. For prototype / validation,
        serial is fine and easier to debug.
        """
        results = []
        for i, (acc, txt) in enumerate(items):
            r = self.extract_one(txt, accession=acc)
            results.append(r)
            if show_progress and (i + 1) % 25 == 0:
                ok = sum(1 for x in results if x.extraction is not None)
                print(f"  [{i+1}/{len(items)}]  ok={ok}  err={i+1-ok}")
        return results

    def extract_batch_parallel(
        self, items: list, max_workers: int = 16, show_progress: bool = True,
        progress_every: int = 100,
    ) -> list:
        """Parallel extraction via threadpool. Gemini 2.5 Flash free tier limits
        are ~1000 RPM; paid is higher. 16 workers × ~2.5s/call = ~6.4 QPS → safe
        under any tier. For 34K records → ~90 minutes wall time.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(items)
        done = 0
        errors = 0

        def _one(idx_item):
            idx, (acc, txt) = idx_item
            return idx, self.extract_one(txt, accession=acc)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_one, (i, it)) for i, it in enumerate(items)]
            for fut in as_completed(futures):
                idx, r = fut.result()
                results[idx] = r
                done += 1
                if r.extraction is None: errors += 1
                if show_progress and done % progress_every == 0:
                    print(f"  [{done}/{len(items)}]  ok={done-errors}  err={errors}", flush=True)
        return results
