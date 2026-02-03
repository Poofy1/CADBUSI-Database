#!/usr/bin/env python3
"""
Caliper detection pipeline.
"""

import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.ML_processing.caliper_pipeline.caliper_pipeline import decide, load_models, run_classifier_on_arrays, run_locator_with_points
from config import CONFIG
from src.DB_processing.tools import append_audit
from src.DB_processing.database import DatabaseManager
from tools.storage_adapter import *


def crop_image(img: np.ndarray, crop_x: int, crop_y: int, crop_w: int, crop_h: int) -> np.ndarray:
    """Crop ultrasound region from uncropped DICOM frame."""
    return img[int(crop_y) : int(crop_y + crop_h), int(crop_x) : int(crop_x + crop_w)]


def points_to_uncropped(points: list[dict], crop_x: float, crop_y: float) -> list[dict]:
    """Translate locator points from crop-relative to uncropped image coordinates."""
    return [
        {"x": round(p["x"] + crop_x, 1), "y": round(p["y"] + crop_y, 1), "score": p["score"]}
        for p in points
    ]


def run_caliper_pipeline():

    DEVICE = "cuda"  # "auto", "cpu", "cuda", "cuda:0", etc.
    BATCH_SIZE = 32

    device = torch.device(
        DEVICE if DEVICE != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    with DatabaseManager() as db:
        image_folder_path = f"{CONFIG['DATABASE_DIR']}/images/"

        # Load data from database (exclude RGB images, only studies from 2018+)
        image_df = db.get_images_dataframe(
            where_clause="photometric_interpretation != 'RGB' "
                         "AND accession_number IN "
                         "(SELECT accession_number FROM StudyCases WHERE date >= '2018-01-01')"
        )
        n = len(image_df)
        print(f"Processing {n} images")
        append_audit("caliper_pipeline.input_images", n)

        # Determine which rows have crop info
        crop_cols = {"crop_x", "crop_y", "crop_w", "crop_h"}
        has_crop_info = crop_cols.issubset(image_df.columns) and image_df[list(crop_cols)].notna().all(axis=1)
        if isinstance(has_crop_info, bool):
            has_crop_info = pd.Series([has_crop_info] * n)

        n_with_crop = int(has_crop_info.sum())
        print(f"  With crop info: {n_with_crop}")
        print(f"  Uncropped only: {n - n_with_crop}")

        # Load models
        print("Loading models...")
        clf_uncropped, clf_cropped, locator = load_models(device)

        # Load all images
        print("Loading images...")
        images = []
        for _, row in tqdm(image_df.iterrows(), total=n, desc="Loading"):
            img = read_image(os.path.join(image_folder_path, row.image_name))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)

        # Step 1: Uncropped classifier on all images
        print("Running uncropped classifier...")
        uncropped_arrays = [img if img is not None else np.zeros((100, 100), dtype=np.uint8) for img in images]
        probs_uncropped = run_classifier_on_arrays(
            clf_uncropped, uncropped_arrays, device, BATCH_SIZE
        )

        # Step 2: For images with crop info, crop and run cropped classifier + locator
        probs_cropped = [0.0] * n
        locator_n_peaks = [0] * n
        locator_peak_max = [0.0] * n
        all_points = [[] for _ in range(n)]

        if n_with_crop > 0:
            crop_indices = [i for i in range(n) if has_crop_info.iloc[i]]
            crop_arrays = []
            crop_params = []

            print("Cropping images...")
            for i in tqdm(crop_indices, desc="Crop"):
                row = image_df.iloc[i]
                img = images[i]
                if img is None:
                    crop_arrays.append(np.zeros((100, 100), dtype=np.uint8))
                    crop_params.append((0, 0))
                    continue
                cropped = crop_image(img, row.crop_x, row.crop_y, row.crop_w, row.crop_h)
                crop_arrays.append(cropped)
                crop_params.append((float(row.crop_x), float(row.crop_y)))

            # Cropped classifier (batched)
            print("Running cropped classifier...")
            crop_probs = run_classifier_on_arrays(
                clf_cropped, crop_arrays, device, BATCH_SIZE
            )
            for j, i in enumerate(crop_indices):
                probs_cropped[i] = crop_probs[j]

            # Locator (per-image)
            print("Running locator...")
            for j, i in enumerate(tqdm(crop_indices, desc="Locator")):
                n_pk, pk_max, pts = run_locator_with_points(
                    locator, crop_arrays[j], device
                )
                locator_n_peaks[i] = n_pk
                locator_peak_max[i] = pk_max
                cx, cy = crop_params[j]
                all_points[i] = points_to_uncropped(pts, cx, cy)

        # Step 3: Decide and build results
        print("Assembling results...")
        results = []
        for i in range(n):
            row = image_df.iloc[i]
            has_caliper, confidence, source = decide(
                probs_uncropped[i], probs_cropped[i],
                locator_n_peaks[i], locator_peak_max[i],
            )
            pts_json = json.dumps(all_points[i]) if all_points[i] else ""
            results.append({
                "image_name": row.image_name,
                "has_calipers": has_caliper,
                "has_calipers_prediction": round(confidence, 4),
                "has_caliper_source": source,
                "has_caliper_prob_uncropped": round(probs_uncropped[i], 4),
                "has_caliper_prob_cropped": round(probs_cropped[i], 4),
                "caliper_n_peaks": locator_n_peaks[i],
                "caliper_peak_max_score": round(locator_peak_max[i], 4),
                "caliper_coordinates": pts_json,
            })

        # Update database
        updated_count = db.insert_images_batch(results, upsert=True)
        print(f"Updated {updated_count} images in database")

        # Summary
        n_caliper = sum(r["has_calipers"] for r in results)
        append_audit("caliper_pipeline.images_with_calipers", n_caliper)
        print(f"  Total images:  {n}")
        print(f"  Has caliper:   {n_caliper} ({100 * n_caliper / max(n, 1):.1f}%)")
        print(f"  No caliper:    {n - n_caliper}")


if __name__ == "__main__":
    run_caliper_pipeline()
