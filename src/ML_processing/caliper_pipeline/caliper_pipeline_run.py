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
from PIL import Image
from scipy.ndimage import maximum_filter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.ML_processing.caliper_pipeline.caliper_pipeline import decide, load_models
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


def _extract_peaks(heatmap: np.ndarray, scale: float, threshold=0.3, min_distance=10) -> tuple[int, float, list[dict]]:
    """Peak finding on a locator heatmap. Returns (n_peaks, peak_max, points) in crop coords."""
    local_max = maximum_filter(heatmap, size=min_distance)
    peaks_mask = (heatmap == local_max) & (heatmap > threshold)
    ys, xs = np.where(peaks_mask)

    points = [
        {
            "x": round(float(x_px) / scale, 1),
            "y": round(float(y_px) / scale, 1),
            "score": round(float(heatmap[y_px, x_px]), 4),
        }
        for y_px, x_px in zip(ys, xs)
    ]
    points.sort(key=lambda p: p["score"], reverse=True)
    peak_max = max((p["score"] for p in points), default=0.0)
    return len(points), peak_max, points


class _CaliperDataset(Dataset):
    """Reads each image once, produces tensors for all three models."""

    def __init__(self, image_df, image_folder_path, has_crop_info):
        self.image_names = image_df["image_name"].tolist()
        self.crop_x = image_df["crop_x"].tolist() if "crop_x" in image_df.columns else [0.0] * len(image_df)
        self.crop_y = image_df["crop_y"].tolist() if "crop_y" in image_df.columns else [0.0] * len(image_df)
        self.crop_w = image_df["crop_w"].tolist() if "crop_w" in image_df.columns else [0.0] * len(image_df)
        self.crop_h = image_df["crop_h"].tolist() if "crop_h" in image_df.columns else [0.0] * len(image_df)
        self.has_crop = has_crop_info.tolist()
        self.image_folder_path = image_folder_path
        self.clf_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = read_image(os.path.join(self.image_folder_path, self.image_names[idx]))

        # Uncropped tensor (224x224 for classifier)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            uncropped_tensor = self.clf_transform(Image.fromarray(gray))
        else:
            uncropped_tensor = torch.zeros(1, 224, 224)
            gray = None

        # Cropped tensor (224x224 for classifier) + locator tensor (512x512)
        scale = 0.0
        if self.has_crop[idx] and gray is not None:
            cropped = crop_image(gray, self.crop_x[idx], self.crop_y[idx], self.crop_w[idx], self.crop_h[idx])
            cropped_tensor = self.clf_transform(Image.fromarray(cropped))

            # Locator: resize maintaining aspect ratio, pad to 512x512
            h, w = cropped.shape[:2]
            scale = 512.0 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(cropped, (new_w, new_h))
            padded = np.zeros((512, 512), dtype=np.uint8)
            padded[:new_h, :new_w] = resized
            locator_tensor = torch.from_numpy(padded).float().div_(255.0).unsqueeze(0)
        else:
            cropped_tensor = torch.zeros(1, 224, 224)
            locator_tensor = torch.zeros(1, 512, 512)

        return uncropped_tensor, cropped_tensor, locator_tensor, scale, self.has_crop[idx], idx


def run_caliper_pipeline():

    DEVICE = "cuda"  # "auto", "cpu", "cuda", "cuda:0", etc.
    BATCH_SIZE = 32
    LIMIT = None  # Set to an int (e.g. 100) to process only a subset for testing

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
        if LIMIT is not None:
            image_df = image_df.head(LIMIT)
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

        # Run all three models in a single DataLoader pass (one image read per sample)
        print("Running models...")
        dataset = _CaliperDataset(image_df, image_folder_path, has_crop_info)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

        probs_uncropped = [0.0] * n
        probs_cropped = [0.0] * n
        locator_n_peaks = [0] * n
        locator_peak_max = [0.0] * n
        all_points = [[] for _ in range(n)]

        with torch.no_grad():
            for unc_batch, crop_batch, loc_batch, scales, has_crop_batch, indices in tqdm(loader, desc="Models"):
                unc_batch = unc_batch.to(device)
                crop_batch = crop_batch.to(device)
                loc_batch = loc_batch.to(device)

                # All three forward passes
                p_unc = torch.sigmoid(clf_uncropped(unc_batch).squeeze(-1)).cpu().numpy()
                p_crop = torch.sigmoid(clf_cropped(crop_batch).squeeze(-1)).cpu().numpy()
                heatmaps = torch.sigmoid(locator(loc_batch)).squeeze(1).cpu().numpy()

                # Per-image post-processing
                for j, idx in enumerate(indices.numpy()):
                    probs_uncropped[idx] = float(p_unc[j])

                    if has_crop_batch[j]:
                        probs_cropped[idx] = float(p_crop[j])

                        # Peak finding on locator heatmap (CPU)
                        n_pk, pk_max, pts = _extract_peaks(heatmaps[j], float(scales[j]))
                        locator_n_peaks[idx] = n_pk
                        locator_peak_max[idx] = pk_max
                        cx, cy = dataset.crop_x[idx], dataset.crop_y[idx]
                        all_points[idx] = points_to_uncropped(pts, float(cx), float(cy))

        # Decide and build results
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
        updated_count = db.insert_images_batch(results, update_only=True)
        print(f"Updated {updated_count} images in database")

        # Summary
        n_caliper = sum(r["has_calipers"] for r in results)
        append_audit("caliper_pipeline.images_with_calipers", n_caliper)
        print(f"  Total images:  {n}")
        print(f"  Has caliper:   {n_caliper} ({100 * n_caliper / max(n, 1):.1f}%)")
        print(f"  No caliper:    {n - n_caliper}")


if __name__ == "__main__":
    run_caliper_pipeline()
