#!/usr/bin/env python3
"""Build the Paper 2 saliency-evaluation dataset from the local mask_set/.

Reads mask_set/manifest.csv (authoritative, per-exam) and uses `dicom_hash`
to join against data/cadbusi.db for the fields P2 preprocessing needs:
    us_polygon, debris_polygons, crop_x/y/w/h,
    patient_id, accession_number, manufacturer_model_name, has_malignant.

Per row:
  * resolve the on-disk filename from the DB:
      - if manifest.image_filename contains "inpainted": use Images.inpainted_version
      - else: use Images.image_name
  * load mask_set/images/<resolved> + mask_set/masks/<mask_filename>
  * apply P2 preprocessing (crop from shrunk us_polygon + FOV fill + 256px canvas)
  * apply the same crop/resize to the lesion mask so the two stay pixel-aligned

Outputs:
  {output_dir}/images/<resolved>.png         -- 256x256 RGB, gray(128) fill
  {output_dir}/lesion_masks/<resolved>.png   -- 256x256 binary lesion mask
  {output_dir}/tissue_masks/<resolved>.png   -- 256x256 binary FOV mask
  {output_dir}/patch_tissue/<resolved>.png   -- 16x16 tissue-count map
  {output_dir}/manifest.csv                  -- one row per input manifest row

Usage:
  python src/export/generate_p2_saliency_set.py
  python src/export/generate_p2_saliency_set.py --workers 16 --resume
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

_export = Path(__file__).resolve().parent
_root = _export.parent.parent
sys.path.insert(0, str(_export))
sys.path.insert(0, str(_export / "p2"))

from ui_mask import compute_ui_mask, parse_polygon

TARGET_SIZE = 256
FILL = 128


# ---------------------------------------------------------------------------
# P2 spatial transforms (mirrors src/export/p2/main.py + transform_gold_masks.py)
# ---------------------------------------------------------------------------

def shrink_polygon(poly_str: str, factor: float = 0.97) -> str:
    pts = parse_polygon(poly_str)
    if len(pts) == 0:
        return poly_str
    centroid = pts.mean(axis=0)
    shrunk = centroid + factor * (pts - centroid)
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in shrunk)


def _crop_box(row: dict, img_h: int, img_w: int) -> tuple[int, int, int, int, Optional[str]]:
    us_poly = row.get("us_polygon") or None
    if isinstance(us_poly, float):
        us_poly = None

    if us_poly:
        shrunk = shrink_polygon(us_poly)
        pts = parse_polygon(shrunk)
        if len(pts) > 0:
            cx = max(0, int(np.floor(pts[:, 0].min())))
            cy = max(0, int(np.floor(pts[:, 1].min())))
            cw = min(img_w - cx, int(np.ceil(pts[:, 0].max())) - cx)
            ch = min(img_h - cy, int(np.ceil(pts[:, 1].max())) - cy)
            return cx, cy, cw, ch, shrunk

    cx = max(0, int(row.get("crop_x") or 0))
    cy = max(0, int(row.get("crop_y") or 0))
    cw = min(int(row.get("crop_w") or img_w), img_w - cx)
    ch = min(int(row.get("crop_h") or img_h), img_h - cy)
    return cx, cy, cw, ch, None


def preprocess_pair(
    img: np.ndarray,
    lesion_mask_full: Optional[np.ndarray],
    row: dict,
    target_size: int = TARGET_SIZE,
    fill: int = FILL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run the P2 pipeline on the image and the same crop+resize on the lesion mask."""
    img_h, img_w = img.shape[:2]
    cx, cy, cw, ch, shrunk_us = _crop_box(row, img_h, img_w)
    if cw <= 0 or ch <= 0:
        raise ValueError(f"empty crop box: {cx},{cy},{cw},{ch}")

    debris_poly = row.get("debris_polygons") or None
    if isinstance(debris_poly, float):
        debris_poly = None
    tissue_full = compute_ui_mask(shrunk_us, debris_poly, img_h, img_w)

    img_crop = img[cy : cy + ch, cx : cx + cw].copy()
    tissue_crop = tissue_full[cy : cy + ch, cx : cx + cw]
    img_crop[tissue_crop == 0] = fill

    scale = min(target_size / cw, target_size / ch)
    new_w = min(round(cw * scale), target_size)
    new_h = min(round(ch * scale), target_size)
    img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    tissue_resized = cv2.resize(tissue_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas_img = np.full((target_size, target_size, 3), fill, dtype=np.uint8)
    canvas_tissue = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas_img[:new_h, :new_w] = img_resized
    canvas_tissue[:new_h, :new_w] = tissue_resized

    ps = target_size // 16
    patch_counts = np.zeros((16, 16), dtype=np.uint8)
    for py in range(16):
        for px in range(16):
            patch = canvas_tissue[py * ps : (py + 1) * ps, px * ps : (px + 1) * ps]
            patch_counts[py, px] = min(int(np.count_nonzero(patch)), 255)

    canvas_lesion: Optional[np.ndarray] = None
    if lesion_mask_full is not None:
        if lesion_mask_full.shape[:2] != (img_h, img_w):
            raise ValueError(
                f"lesion mask shape {lesion_mask_full.shape[:2]} != image shape {(img_h, img_w)}"
            )
        lesion_crop = lesion_mask_full[cy : cy + ch, cx : cx + cw]
        lesion_resized = cv2.resize(lesion_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        canvas_lesion = np.zeros((target_size, target_size), dtype=np.uint8)
        canvas_lesion[:new_h, :new_w] = lesion_resized

    return canvas_img, canvas_tissue, patch_counts, canvas_lesion


# ---------------------------------------------------------------------------
# DB enrichment (dicom_hash-keyed)
# ---------------------------------------------------------------------------

def enrich_from_db(db_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Join Images + StudyCases onto the manifest using `dicom_hash`."""
    df = df.copy()
    df["dicom_hash"] = df["dicom_hash"].astype(str).str.strip()
    hashes = sorted({h for h in df["dicom_hash"] if h})
    if not hashes:
        return df

    conn = sqlite3.connect(str(db_path))
    image_cols = {r[1] for r in conn.execute("PRAGMA table_info(Images)").fetchall()}
    inpainted_sql = (
        "i.inpainted_version AS db_inpainted_version"
        if "inpainted_version" in image_cols
        else "NULL AS db_inpainted_version"
    )

    placeholders = ",".join("?" * len(hashes))
    q = f"""
        SELECT i.dicom_hash AS db_dicom_hash,
               i.image_name AS db_image_name,
               {inpainted_sql},
               i.us_polygon, i.debris_polygons,
               i.crop_x, i.crop_y, i.crop_w, i.crop_h,
               i.patient_id, i.accession_number, i.manufacturer_model_name,
               s.has_malignant, s.valid
        FROM Images i
        LEFT JOIN StudyCases s ON i.accession_number = s.accession_number
        WHERE i.dicom_hash IN ({placeholders})
    """
    hit = pd.read_sql_query(q, conn, params=hashes)
    conn.close()

    return df.merge(hit, left_on="dicom_hash", right_on="db_dicom_hash", how="left")


def resolved_image_filename(row: dict) -> Optional[str]:
    """Pick the on-disk filename for this row: inpainted_version if inpainted, else image_name."""
    image_filename = row.get("image_filename") or ""
    want_inpainted = "inpainted" in image_filename
    if want_inpainted:
        iv = row.get("db_inpainted_version") or ""
        iv = "" if isinstance(iv, float) and pd.isna(iv) else str(iv).strip()
        if iv:
            return iv
        return None
    name = row.get("db_image_name") or ""
    name = "" if isinstance(name, float) and pd.isna(name) else str(name).strip()
    return name or None


# ---------------------------------------------------------------------------
# Per-row worker
# ---------------------------------------------------------------------------

def process_one(
    row: dict,
    images_in: Path,
    masks_in: Path,
    out_dir: Path,
) -> tuple[str, bool, str]:
    resolved = row.get("resolved_image") or ""
    mask_fn = row.get("mask_filename") or ""
    if not resolved:
        return ("<unresolved>", False, "no resolved image filename")
    try:
        img_path = images_in / resolved
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return (resolved, False, f"cannot read image: {img_path}")

        lesion_mask = None
        if mask_fn:
            mask_path = masks_in / mask_fn
            if mask_path.exists():
                lesion_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if lesion_mask is None:
                    return (resolved, False, f"cannot read mask: {mask_path}")

        canvas_img, canvas_tissue, patch_counts, canvas_lesion = preprocess_pair(
            img, lesion_mask, row
        )

        cv2.imwrite(str(out_dir / "images" / resolved), canvas_img)
        cv2.imwrite(str(out_dir / "tissue_masks" / resolved), canvas_tissue)
        cv2.imwrite(str(out_dir / "patch_tissue" / resolved), patch_counts)
        if canvas_lesion is not None:
            cv2.imwrite(str(out_dir / "lesion_masks" / resolved), canvas_lesion)

        return (resolved, True, "")
    except Exception as exc:
        return (resolved, False, str(exc))


# ---------------------------------------------------------------------------
# Output dir helper
# ---------------------------------------------------------------------------

def resolve_output_dir(base: Path, resume: bool) -> Path:
    if resume or not base.exists():
        return base
    i = 1
    while True:
        candidate = base.parent / f"{base.name}_{i:04d}"
        if not candidate.exists():
            return candidate
        i += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Paper 2 saliency dataset from local mask_set/"
    )
    parser.add_argument(
        "--mask-set", type=Path, default=_root / "mask_set",
        help="Directory containing manifest.csv, images/, masks/",
    )
    parser.add_argument(
        "--db", type=Path, default=_root / "data" / "cadbusi.db",
        help="cadbusi.db used for us_polygon / crop / patient lookups",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/p2_saliency"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest_in = args.mask_set / "manifest.csv"
    images_in = args.mask_set / "images"
    masks_in = args.mask_set / "masks"

    for p in (manifest_in, images_in, args.db):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    df = pd.read_csv(manifest_in, dtype=str).fillna("")
    print(f"manifest rows: {len(df):,}")

    print(f"Enriching from {args.db}")
    df = enrich_from_db(args.db, df)

    unresolved_hash = df["db_dicom_hash"].isna() | (df["db_dicom_hash"].astype(str).str.strip() == "")
    df["resolved_image"] = df.apply(resolved_image_filename, axis=1)
    missing_resolve = df["resolved_image"].isna() | (df["resolved_image"].astype(str) == "")
    missing_poly = df["us_polygon"].isna() | (df["us_polygon"].astype(str).str.strip() == "")
    missing_crop = df["crop_w"].isna() | df["crop_h"].isna()
    unusable = missing_poly & missing_crop

    print(f"  dicom_hash found in DB:        {int((~unresolved_hash).sum()):,} / {len(df):,}")
    if unresolved_hash.any():
        print(f"  dicom_hash NOT in DB (skipped): {int(unresolved_hash.sum()):,}")
    if missing_resolve.any():
        print(f"  no on-disk name resolvable (skipped): {int(missing_resolve.sum()):,}")
        for fn in df.loc[missing_resolve, "image_filename"].head(5):
            print(f"    {fn}")
    print(f"  rows with us_polygon:          {int((~missing_poly).sum()):,}")
    print(f"  rows crop-only (no polygon):   {int((missing_poly & ~missing_crop).sum()):,}")
    if unusable.any():
        print(f"  rows with neither (skipped):   {int(unusable.sum()):,}")

    processable = df[~unresolved_hash & ~missing_resolve & ~unusable].copy()

    if args.limit > 0:
        processable = processable.head(args.limit).copy()
        print(f"  After --limit: {len(processable):,}")

    if processable.empty:
        print("Nothing to process.")
        return 0

    if args.dry_run:
        print(f"[DRY RUN] would process {len(processable):,} rows to {args.output_dir}")
        cols = [c for c in ("dicom_hash", "resolved_image", "mask_filename",
                            "patient_id", "manufacturer_model_name",
                            "split", "malignant", "has_malignant")
                if c in processable.columns]
        print(processable[cols].head().to_string(index=False))
        return 0

    out_dir = resolve_output_dir(args.output_dir, args.resume)
    if out_dir != args.output_dir:
        print(f"Output dir auto-incremented: {out_dir}")
    for sub in ("images", "lesion_masks", "tissue_masks", "patch_tissue"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    rows_for_work = processable.to_dict("records")
    if args.resume:
        existing = {p.name for p in (out_dir / "images").iterdir()}
        before = len(rows_for_work)
        rows_for_work = [r for r in rows_for_work if r["resolved_image"] not in existing]
        print(f"  resume: skipping {before - len(rows_for_work):,} already-done")

    print(f"Processing {len(rows_for_work):,} rows ({args.workers} threads)")
    ok_names: set[str] = set()
    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, r, images_in, masks_in, out_dir): r["resolved_image"]
            for r in rows_for_work
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            name, ok, err = fut.result()
            if ok:
                ok_names.add(name)
            else:
                errors.append((name, err))

    done_names = {p.name for p in (out_dir / "images").iterdir()}
    print(f"\nDone: {len(ok_names):,} processed, {len(errors):,} errors, {len(done_names):,} total on disk")
    if errors:
        print("First 10 errors:")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    manifest_df = processable[processable["resolved_image"].isin(done_names)].copy()
    manifest_df["image_path"] = "images/" + manifest_df["resolved_image"]
    manifest_df["lesion_mask_path"] = "lesion_masks/" + manifest_df["resolved_image"]
    manifest_df["tissue_mask_path"] = "tissue_masks/" + manifest_df["resolved_image"]
    manifest_df["patch_tissue_path"] = "patch_tissue/" + manifest_df["resolved_image"]

    keep = [
        "dicom_hash", "resolved_image", "image_filename", "mask_filename",
        "split", "malignant", "has_malignant",
        "patient_id", "accession_number", "manufacturer_model_name",
        "image_path", "lesion_mask_path", "tissue_mask_path", "patch_tissue_path",
    ]
    keep = [c for c in keep if c in manifest_df.columns]
    manifest_out = out_dir / "manifest.csv"
    manifest_df[keep].to_csv(manifest_out, index=False)
    print(f"Manifest: {manifest_out}  ({len(manifest_df):,} rows)")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
