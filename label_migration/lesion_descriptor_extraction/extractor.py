"""Gemini structured extraction of BI-RADS descriptors from lesion descriptions.

Pattern mirrors labeling/pathology_extraction/extractor.py — same auth modes
(API key or Vertex AI), same retry/backoff, same threadpool batch helper.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types
from pydantic import ValidationError

from .schema import LesionDescriptors
from .prompts import SYSTEM_INSTRUCTION, build_prompt


DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass
class ExtractionResult:
    key: str                                  # whatever caller wants to track (e.g. description text hash)
    text: str
    extraction: Optional[LesionDescriptors]
    error: Optional[str] = None
    retry_count: int = 0


class LesionDescriptorExtractor:
    """Gemini wrapper with structured-output parsing and retries.

    Two auth modes (same as the pathology extractor):
      1. Direct Gemini API: pass api_key or set GEMINI_API_KEY.
      2. Vertex AI:  use_vertex=True; project/location from args or env vars
         VERTEX_PROJECT / VERTEX_LOCATION.
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
                    "Vertex AI mode requires a GCP project. Set VERTEX_PROJECT or pass project="
                )
            self.client = genai.Client(vertexai=True, project=self.project, location=self.location)
            self.api_key = None
        else:
            self.mode = "api_key"
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. Source ~/.bashrc or pass use_vertex=True."
                )
            self.client = genai.Client(api_key=self.api_key)
            self.project = None
            self.location = None
        self.model = model

    def extract_one(
        self,
        text: str,
        key: str = "",
        max_retries: int = 2,
        retry_delay_sec: float = 2.0,
    ) -> ExtractionResult:
        if not text or not text.strip():
            return ExtractionResult(key=key, text=text, extraction=None, error="empty_text")

        prompt = build_prompt(text)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=LesionDescriptors,
            temperature=0.0,
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
                    import json
                    raw = response.text or "{}"
                    parsed = LesionDescriptors.model_validate(json.loads(raw))
                return ExtractionResult(key=key, text=text, extraction=parsed, retry_count=attempt)
            except ValidationError as e:
                last_err = f"schema_validation_failed: {str(e)[:200]}"
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt < max_retries:
                    time.sleep(retry_delay_sec * (attempt + 1))
                    continue

        return ExtractionResult(
            key=key, text=text, extraction=None, error=last_err, retry_count=max_retries
        )

    def extract_batch_parallel(
        self,
        items: list,
        max_workers: int = 16,
        show_progress: bool = True,
        progress_every: int = 200,
    ) -> list:
        """items: list of (key, text) tuples. Returns list of ExtractionResult in original order."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(items)
        done = 0
        errors = 0

        def _one(idx_item):
            idx, (key, txt) = idx_item
            return idx, self.extract_one(txt, key=key)

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
