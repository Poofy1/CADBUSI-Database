"""Core pipeline logic: model loading, inference, and decision ensemble."""

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import maximum_filter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from src.ML_processing.caliper_pipeline.caliper_models import UNetLite, create_classifier

PIPELINE_DIR = Path(__file__).parent
CKPT_UNCROPPED = PIPELINE_DIR / "checkpoints" / "uncropped_clf.pt"
CKPT_CROPPED = PIPELINE_DIR / "checkpoints" / "cropped_clf.pt"
CKPT_LOCATOR = PIPELINE_DIR / "checkpoints" / "locator.pt"


def _load_state_dict(path, device):
    """Load checkpoint — handles both raw state_dict and training checkpoint dicts."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def load_models(device):
    """Load all three models onto device. Returns (clf_uncropped, clf_cropped, locator)."""
    clf_uncropped = create_classifier()
    clf_uncropped.load_state_dict(_load_state_dict(CKPT_UNCROPPED, device))
    clf_uncropped.to(device).eval()

    clf_cropped = create_classifier()
    clf_cropped.load_state_dict(_load_state_dict(CKPT_CROPPED, device))
    clf_cropped.to(device).eval()

    locator = UNetLite()
    locator.load_state_dict(_load_state_dict(CKPT_LOCATOR, device))
    locator.to(device).eval()

    return clf_uncropped, clf_cropped, locator


# ---------------------------------------------------------------------------
# Datasets for batched classifier inference
# ---------------------------------------------------------------------------

class _ImageListDataset(Dataset):
    """Loads grayscale images at 224x224 for classifier inference."""

    def __init__(self, image_paths: list[Path]):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("L")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(1, 224, 224)
        return img, idx


class _NumpyListDataset(Dataset):
    """Wraps numpy arrays (cropped images) at 224x224 for classifier inference."""

    def __init__(self, arrays: list[np.ndarray]):
        self.arrays = arrays
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        try:
            img = Image.fromarray(self.arrays[idx]).convert("L")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(1, 224, 224)
        return img, idx


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_classifier_on_paths(model, image_paths: list[Path], device, batch_size=32) -> list[float]:
    """Run classifier on image files. Returns list of probabilities aligned with input."""
    ds = _ImageListDataset(image_paths)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    probs = [0.0] * len(image_paths)
    with torch.no_grad():
        for batch_imgs, batch_idxs in loader:
            batch_imgs = batch_imgs.to(device)
            logits = model(batch_imgs).squeeze(-1)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            for prob, idx in zip(batch_probs, batch_idxs.numpy()):
                probs[idx] = float(prob)
    return probs


def run_classifier_on_arrays(model, arrays: list[np.ndarray], device, batch_size=32) -> list[float]:
    """Run classifier on numpy arrays (crops). Returns list of probabilities."""
    ds = _NumpyListDataset(arrays)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    probs = [0.0] * len(arrays)
    with torch.no_grad():
        for batch_imgs, batch_idxs in loader:
            batch_imgs = batch_imgs.to(device)
            logits = model(batch_imgs).squeeze(-1)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            for prob, idx in zip(batch_probs, batch_idxs.numpy()):
                probs[idx] = float(prob)
    return probs


def run_locator_with_points(
    model, crop_array: np.ndarray, device, threshold=0.3, min_distance=10
) -> tuple[int, float, list[dict]]:
    """Run UNetLite on a cropped grayscale array.

    Returns (n_peaks, peak_max_score, points) where points are in crop coordinates:
        [{"x": float, "y": float, "score": float}, ...]
    """
    h, w = crop_array.shape[:2]
    input_size = 512
    scale = input_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(crop_array, (new_w, new_h))

    padded = np.zeros((input_size, input_size), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    tensor = torch.from_numpy(padded).float().div_(255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        heatmap = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

    local_max = maximum_filter(heatmap, size=min_distance)
    peaks_mask = (heatmap == local_max) & (heatmap > threshold)

    ys, xs = np.where(peaks_mask)
    points = []
    for y_px, x_px in zip(ys, xs):
        # Map back from 512x512 padded space to original crop coordinates
        points.append({
            "x": round(float(x_px) / scale, 1),
            "y": round(float(y_px) / scale, 1),
            "score": round(float(heatmap[y_px, x_px]), 4),
        })

    n_peaks = len(points)
    peak_max = max((p["score"] for p in points), default=0.0)
    # Sort by score descending
    points.sort(key=lambda p: p["score"], reverse=True)
    return n_peaks, peak_max, points


def decide(prob_uncropped, prob_cropped, n_peaks, peak_max_score):
    """Corroborated ensemble. Returns (has_caliper, confidence, decision_source).

    A single classifier firing requires corroboration from either the locator
    (>=1 peak) or weak agreement from the other classifier (>0.2). Both
    classifiers agreeing or locator alone (>=2 peaks) are sufficient.

    On the corrected 13k gold set: 2 FP, 0 FN (P=0.9998, R=1.0000).
    """
    sources = []
    if prob_uncropped > 0.5:
        sources.append("uncropped")
    if prob_cropped > 0.5:
        sources.append("cropped")
    loc1 = n_peaks >= 1 and peak_max_score > 0.3
    loc2 = n_peaks >= 2 and peak_max_score > 0.3
    if loc1:
        sources.append("locator")

    unc = prob_uncropped > 0.5
    crop = prob_cropped > 0.5

    has_caliper = int(
        (unc and crop)  # both classifiers agree
        or (unc and (loc1 or prob_cropped > 0.2))  # uncropped + corroboration
        or (crop and (loc1 or prob_uncropped > 0.2))  # cropped + corroboration
        or loc2  # locator alone (>=2 peaks)
    )

    locator_conf = peak_max_score if loc1 else 0.0
    confidence = max(prob_uncropped, prob_cropped, locator_conf)
    decision_source = "+".join(sources) if sources else "none"
    return has_caliper, confidence, decision_source
