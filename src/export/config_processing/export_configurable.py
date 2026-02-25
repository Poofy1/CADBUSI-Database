#!/usr/bin/env python3
"""Configurable dataset export using YAML configuration files.

This script reads filter settings from a YAML config file, allowing
different dataset versions to be created without code changes.

Usage:
    python export_configurable.py config.yaml /path/to/cadbusi.db /path/to/output

Examples:
    # Standard baseline export
    python export_configurable.py configs/baseline_v1.yaml /data/cadbusi.db ./output/baseline

    # Strict breast-only export
    python export_configurable.py configs/strict_breast_only.yaml /data/cadbusi.db ./output/strict
"""

import argparse
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import polars as pl
import yaml


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class ImageFilters:
    darkness_max: int = 75
    allowed_areas: List[str] = field(default_factory=lambda: ["breast", "unknown"])
    region_count_max: int = 1
    require_crop_coords: bool = True
    aspect_ratio_min: float = 0.5
    aspect_ratio_max: float = 4.0
    min_dimension: int = 200
    require_on_disk: bool = False  # Only images that exist on disk
    on_disk_image_dir: Optional[str] = None  # Compute on_disk from this dir if column missing
    exclude_calipers: bool = False  # Exclude images with measurement calipers (has_calipers = 1)


@dataclass
class BilateralHandling:
    split_bilateral: bool = True
    remove_if_no_laterality: bool = True


@dataclass
class NullLateralityHandling:
    split_if_both_sides: bool = True
    keep_unsplit: bool = True


@dataclass
class StudyProcessing:
    bilateral_handling: BilateralHandling = field(default_factory=BilateralHandling)
    null_laterality_handling: NullLateralityHandling = field(default_factory=NullLateralityHandling)


@dataclass
class SplitConfig:
    method: str = "hash"
    split_by: str = "patient_id"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class ScannerFilters:
    exclude_scanners: List[str] = field(default_factory=list)
    allowed_scanners: List[str] = field(default_factory=list)  # If set, ONLY these scanners


@dataclass
class StudyFilters:
    is_biopsy: Optional[int] = None        # Filter by is_biopsy value (0 or 1)
    min_year: Optional[int] = None         # Minimum year (inclusive)
    max_year: Optional[int] = None         # Maximum year (inclusive)
    exclude_unknown_label: bool = False    # Exclude has_malignant == -1


@dataclass
class PreprocessingConfig:
    pipeline: str = "structural_tissue_aware"  # structural_tissue_aware or simple_crop_letterbox
    target_size: int = 256
    fill: int = 128


@dataclass
class V6Config:
    enabled: bool = False
    mini_enabled: bool = False
    mini_fraction: float = 0.20
    mini_seed: str = "v6mini"


@dataclass
class OutputConfig:
    apply_image_filters: bool = False
    subfolder: str = ""  # Output CSVs to subfolder (e.g., "baseline_dataset")
    files: List[str] = field(default_factory=lambda: [
        "BreastData.csv", "ImageData.csv", "LesionData.csv",
        "PathologyData.csv", "VideoData.csv"
    ])


@dataclass
class ExportConfig:
    name: str = "default"
    description: str = ""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    image_filters: ImageFilters = field(default_factory=ImageFilters)
    scanner_filters: ScannerFilters = field(default_factory=ScannerFilters)
    study_filters: StudyFilters = field(default_factory=StudyFilters)
    study_processing: StudyProcessing = field(default_factory=StudyProcessing)
    split: SplitConfig = field(default_factory=SplitConfig)
    v6: V6Config = field(default_factory=V6Config)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExportConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()
        config.name = data.get("name", "default")
        config.description = data.get("description", "")

        if "image_filters" in data:
            f = data["image_filters"]
            config.image_filters = ImageFilters(
                darkness_max=f.get("darkness_max", 75),
                allowed_areas=f.get("allowed_areas", ["breast", "unknown"]),
                region_count_max=f.get("region_count_max", 1),
                require_crop_coords=f.get("require_crop_coords", True),
                aspect_ratio_min=f.get("aspect_ratio_min", 0.5),
                aspect_ratio_max=f.get("aspect_ratio_max", 4.0),
                min_dimension=f.get("min_dimension", 200),
                require_on_disk=f.get("require_on_disk", False),
                on_disk_image_dir=f.get("on_disk_image_dir"),
                exclude_calipers=f.get("exclude_calipers", False),
            )

        if "study_processing" in data:
            sp = data["study_processing"]
            bh = sp.get("bilateral_handling", {})
            nh = sp.get("null_laterality_handling", {})
            config.study_processing = StudyProcessing(
                bilateral_handling=BilateralHandling(
                    split_bilateral=bh.get("split_bilateral", True),
                    remove_if_no_laterality=bh.get("remove_if_no_laterality", True),
                ),
                null_laterality_handling=NullLateralityHandling(
                    split_if_both_sides=nh.get("split_if_both_sides", True),
                    keep_unsplit=nh.get("keep_unsplit", True),
                ),
            )

        if "split" in data:
            s = data["split"]
            config.split = SplitConfig(
                method=s.get("method", "hash"),
                split_by=s.get("split_by", "patient_id"),
                train_ratio=s.get("train_ratio", 0.70),
                val_ratio=s.get("val_ratio", 0.15),
                test_ratio=s.get("test_ratio", 0.15),
            )

        if "scanner_filters" in data:
            sf = data["scanner_filters"]
            config.scanner_filters = ScannerFilters(
                exclude_scanners=sf.get("exclude_scanners", []),
                allowed_scanners=sf.get("allowed_scanners", []),
            )

        if "preprocessing" in data:
            p = data["preprocessing"]
            config.preprocessing = PreprocessingConfig(
                pipeline=p.get("pipeline", "structural_tissue_aware"),
                target_size=p.get("target_size", 256),
                fill=p.get("fill", 128),
            )

        if "study_filters" in data:
            stf = data["study_filters"]
            config.study_filters = StudyFilters(
                is_biopsy=stf.get("is_biopsy"),
                min_year=stf.get("min_year"),
                max_year=stf.get("max_year"),
                exclude_unknown_label=stf.get("exclude_unknown_label", False),
            )

        if "v6" in data:
            v = data["v6"]
            config.v6 = V6Config(
                enabled=v.get("enabled", False),
                mini_enabled=v.get("mini_enabled", False),
                mini_fraction=v.get("mini_fraction", 0.20),
                mini_seed=v.get("mini_seed", "v6mini"),
            )

        if "output" in data:
            o = data["output"]
            config.output = OutputConfig(
                apply_image_filters=o.get("apply_image_filters", False),
                subfolder=o.get("subfolder", ""),
                files=o.get("files", ["BreastData.csv", "ImageData.csv"]),
            )

        return config


# =============================================================================
# PARQUET CACHE (imported from cache_parquet.py)
# =============================================================================

from cache_parquet import ensure_cache as ensure_parquet_cache, get_cache_dir


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def compute_exclusion_reason(df: pl.LazyFrame, config: ImageFilters) -> pl.LazyFrame:
    """Compute exclusion_reason based on config filters."""
    f = config

    return df.with_columns(
        pl.concat_str([
            pl.when(pl.col("darkness") > f.darkness_max)
            .then(pl.lit(f"darkness>{f.darkness_max};"))
            .otherwise(pl.lit("")),

            pl.when(~pl.col("area").is_in(f.allowed_areas) & pl.col("area").is_not_null())
            .then(pl.lit("non-breast;"))
            .otherwise(pl.lit("")),

            pl.when(pl.col("region_count") > f.region_count_max)
            .then(pl.lit(f"region_count>{f.region_count_max};"))
            .otherwise(pl.lit("")),

            pl.when(pl.col("crop_x").is_null())
            .then(pl.lit("missing_crop;"))
            .otherwise(pl.lit("")) if f.require_crop_coords else pl.lit(""),

            pl.when(
                (pl.col("crop_w") / pl.col("crop_h") < f.aspect_ratio_min) |
                (pl.col("crop_w") / pl.col("crop_h") > f.aspect_ratio_max)
            )
            .then(pl.lit("aspect_ratio;"))
            .otherwise(pl.lit("")),

            pl.when(
                (pl.col("crop_w") < f.min_dimension) |
                (pl.col("crop_h") < f.min_dimension)
            )
            .then(pl.lit("small_dimension;"))
            .otherwise(pl.lit("")),
        ]).str.strip_chars(";").alias("exclusion_reason")
    ).with_columns(
        pl.when(pl.col("exclusion_reason") == "")
        .then(None)
        .otherwise(pl.col("exclusion_reason"))
        .alias("exclusion_reason")
    )


def compute_split(patient_id: str, config: SplitConfig) -> int:
    """Compute train/val/test split."""
    if patient_id is None:
        return 0
    hash_bytes = hashlib.md5(str(patient_id).encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
    hash_float = hash_int / (2**64)

    if hash_float < config.test_ratio:
        return 2
    elif hash_float < config.test_ratio + config.val_ratio:
        return 1
    return 0


def compute_study_has_malignant(study_meta: pl.DataFrame) -> pl.DataFrame:
    """Compute has_malignant for each study based on diagnosis columns."""
    # Add has_malignant based on laterality-specific diagnosis
    df = study_meta.with_columns(
        pl.col("study_laterality").fill_null("").str.to_uppercase().alias("_lat_upper")
    )

    both_null = pl.col("left_diagnosis").is_null() & pl.col("right_diagnosis").is_null()
    diag_expr = (
        pl.when(pl.col("_lat_upper") == "LEFT").then(pl.col("left_diagnosis"))
        .when(pl.col("_lat_upper") == "RIGHT").then(pl.col("right_diagnosis"))
        .otherwise(
            pl.when(pl.col("left_diagnosis").is_not_null())
            .then(pl.col("left_diagnosis"))
            .otherwise(pl.col("right_diagnosis"))
        )
    )
    is_malignant = diag_expr.cast(pl.Utf8).str.to_uppercase().str.contains("MALIGNANT")

    return df.with_columns(
        pl.when(both_null).then(pl.lit(-1))
        .when(is_malignant).then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int32).alias("has_malignant")
    ).drop("_lat_upper")


def export_image_data(
    parquet_paths: dict,
    config: ExportConfig,
    output_dir: Path,
    lesion_images: set,
    study_meta: pl.DataFrame,
) -> pl.DataFrame:
    """Export ImageData.csv with config-based filtering."""
    print("\nExporting ImageData.csv...")
    start = time.time()

    # Compute has_malignant for studies
    study_with_label = compute_study_has_malignant(study_meta)

    # Load and filter
    df = pl.scan_parquet(parquet_paths["Images"])

    # Remove unprocessed images
    df = df.filter(pl.col("darkness").is_not_null())

    # Normalize laterality
    df = df.with_columns(
        pl.col("laterality").fill_null("").str.to_uppercase().alias("laterality")
    )

    # Build lookup maps
    study_lat_map = dict(zip(
        study_with_label["accession_number"].to_list(),
        study_with_label["study_laterality"].to_list()
    ))
    age_map = dict(zip(
        study_with_label["accession_number"].to_list(),
        study_with_label["age_at_event"].to_list()
    ))
    # has_malignant map: accession_number -> has_malignant
    has_malignant_map = dict(zip(
        study_with_label["accession_number"].to_list(),
        study_with_label["has_malignant"].to_list()
    ))

    df = df.with_columns([
        pl.col("accession_number").replace_strict(study_lat_map, default=None).alias("study_laterality"),
        pl.col("accession_number").replace_strict(age_map, default=None).alias("Age"),
    ])

    # Compute exclusion_reason based on config
    df = compute_exclusion_reason(df, config.image_filters)

    # Add dimensions
    df = df.with_columns([
        pl.col("crop_w").fill_null(0).cast(pl.Int32).alias("image_w"),
        pl.col("crop_h").fill_null(0).cast(pl.Int32).alias("image_h"),
    ])

    # Collect
    df = df.collect()

    # Add is_lesion
    df = df.with_columns(
        pl.col("image_name").is_in(lesion_images).cast(pl.Int8).alias("is_lesion")
    )

    # Add has_malignant from study-level
    has_malignant_col = [has_malignant_map.get(acc, -1) for acc in df["accession_number"].to_list()]
    df = df.with_columns(pl.Series("has_malignant", has_malignant_col).cast(pl.Int32))

    # Apply image quality filters if requested
    if config.output.apply_image_filters:
        before = len(df)
        df = df.filter(pl.col("exclusion_reason").is_null())
        print(f"  Applied quality filters: {before:,} → {len(df):,}")

    # Compute on_disk dynamically if column is missing and image_dir is provided
    if config.image_filters.require_on_disk and "on_disk" not in df.columns:
        image_dir = config.image_filters.on_disk_image_dir
        if image_dir:
            import os
            print(f"  Computing on_disk from: {image_dir}")
            image_dir_path = Path(image_dir)
            # Build set of all files on disk (fast: one os.walk pass)
            print("  Scanning image directory...", end=" ", flush=True)
            scan_start = time.time()
            files_on_disk = set()
            for dirpath, _, filenames in os.walk(image_dir_path):
                for fn in filenames:
                    files_on_disk.add(fn)
            print(f"{len(files_on_disk):,} files in {time.time() - scan_start:.1f}s")
            # Match by image_name
            df = df.with_columns(
                pl.col("image_name").is_in(files_on_disk).cast(pl.Int8).alias("on_disk")
            )
            n_found = df.filter(pl.col("on_disk") == 1).height
            n_missing = len(df) - n_found
            print(f"  Computed on_disk: {n_found:,} found, {n_missing:,} missing")
        else:
            print("  WARNING: require_on_disk=true but no on_disk column and no on_disk_image_dir set")

    # Apply on_disk filter if requested
    if config.image_filters.require_on_disk and "on_disk" in df.columns:
        before = len(df)
        df = df.filter(pl.col("on_disk") == 1)
        print(f"  Applied on_disk filter: {before:,} → {len(df):,}")

    # Apply scanner filtering if specified
    if config.scanner_filters.allowed_scanners:
        # Use allowlist (only keep these scanners)
        before = len(df)
        df = df.filter(
            pl.col("manufacturer_model_name").is_in(config.scanner_filters.allowed_scanners)
        )
        print(f"  Applied scanner allowlist: {before:,} → {len(df):,}")
    elif config.scanner_filters.exclude_scanners:
        # Use blocklist (exclude these scanners)
        before = len(df)
        df = df.filter(
            ~pl.col("manufacturer_model_name").is_in(config.scanner_filters.exclude_scanners)
        )
        print(f"  Applied scanner blocklist: {before:,} → {len(df):,}")

    # Save
    df.write_csv(output_dir / "ImageData.csv")
    print(f"  Saved: {len(df):,} rows in {time.time() - start:.2f}s")

    excluded = df.filter(pl.col("exclusion_reason").is_not_null()).height
    usable = len(df) - excluded
    print(f"  Usable (no exclusion): {usable:,}")

    # Report polygon coverage if columns present
    if "us_polygon" in df.columns:
        has_poly = df.filter(pl.col("us_polygon").is_not_null() & (pl.col("us_polygon") != "")).height
        pct = has_poly / len(df) * 100 if len(df) > 0 else 0
        print(f"  Polygon coverage: {has_poly:,}/{len(df):,} ({pct:.2f}%)")

    return df


def export_breast_data(
    parquet_paths: dict,
    images_df: pl.DataFrame,
    lesions_df: pl.DataFrame,
    config: ExportConfig,
    output_dir: Path,
) -> pl.DataFrame:
    """Export BreastData.csv with config-based processing."""
    print("\nExporting BreastData.csv...")
    start = time.time()

    study_df = pl.read_parquet(parquet_paths["StudyCases"])
    print(f"  StudyCases: {len(study_df):,} rows")

    # Add exam_site from location_city (granular site for per-site analysis)
    if "location_city" in study_df.columns:
        study_df = study_df.with_columns(
            pl.col("location_city").fill_null("UNKNOWN").str.to_uppercase().alias("exam_site")
        )
        n_with_site = study_df.filter(pl.col("exam_site") != "UNKNOWN").height
        print(f"  exam_site from location_city: {n_with_site:,}/{len(study_df):,} mapped")

    # Prepare images
    images_df = images_df.with_columns(
        pl.col("laterality").fill_null("").str.to_uppercase().alias("laterality_upper")
    )

    # Build laterality info per accession (vectorized)
    lat_info = (
        images_df
        .group_by("accession_number")
        .agg([
            pl.col("laterality_upper").is_in(["LEFT"]).any().alias("has_left"),
            pl.col("laterality_upper").is_in(["RIGHT"]).any().alias("has_right"),
            pl.col("laterality_upper").is_in(["", "UNKNOWN"]).any().alias("has_unknown"),
        ])
    )

    # Track axilla
    axilla_accs = set(
        images_df.filter(pl.col("area") == "axilla")
        .select("accession_number").unique().to_series().to_list()
    )
    study_df = study_df.with_columns(
        pl.col("accession_number").is_in(axilla_accs).cast(pl.Int8).alias("contained_axilla")
    )

    # Join laterality info
    study_df = study_df.with_columns(
        pl.col("study_laterality").fill_null("").str.to_uppercase().alias("lat_upper")
    )
    study_df = study_df.join(lat_info, on="accession_number", how="left")
    study_df = study_df.with_columns([
        pl.col("has_left").fill_null(False),
        pl.col("has_right").fill_null(False),
        pl.col("has_unknown").fill_null(False),
    ])

    # Separate by category
    sp = config.study_processing
    bilateral_df = study_df.filter(pl.col("lat_upper") == "BILATERAL")
    null_df = study_df.filter(pl.col("lat_upper") == "")
    normal_df = study_df.filter((pl.col("lat_upper") != "BILATERAL") & (pl.col("lat_upper") != ""))

    normal_df = normal_df.with_columns([
        pl.lit(0).cast(pl.Int8).alias("was_bilateral"),
        pl.lit(0).cast(pl.Int8).alias("has_unknown_laterality"),
    ])

    result_parts = [normal_df]
    stats = {"split_lr": 0, "converted_single": 0, "removed": 0, "kept_null": 0}

    # Process bilateral
    if len(bilateral_df) > 0 and sp.bilateral_handling.split_bilateral:
        bi_both = bilateral_df.filter(pl.col("has_left") & pl.col("has_right"))
        bi_left = bilateral_df.filter(pl.col("has_left") & ~pl.col("has_right"))
        bi_right = bilateral_df.filter(~pl.col("has_left") & pl.col("has_right"))
        bi_neither = bilateral_df.filter(~pl.col("has_left") & ~pl.col("has_right"))

        if len(bi_both) > 0:
            for side in ["LEFT", "RIGHT"]:
                part = bi_both.with_columns([
                    pl.lit(side).alias("study_laterality"),
                    pl.lit(1).cast(pl.Int8).alias("was_bilateral"),
                    pl.col("has_unknown").cast(pl.Int8).alias("has_unknown_laterality"),
                ])
                result_parts.append(part)
            stats["split_lr"] = len(bi_both)

        for df_part, side in [(bi_left, "LEFT"), (bi_right, "RIGHT")]:
            if len(df_part) > 0:
                part = df_part.with_columns([
                    pl.lit(side).alias("study_laterality"),
                    pl.lit(1).cast(pl.Int8).alias("was_bilateral"),
                    pl.col("has_unknown").cast(pl.Int8).alias("has_unknown_laterality"),
                ])
                result_parts.append(part)
                stats["converted_single"] += len(df_part)

        if sp.bilateral_handling.remove_if_no_laterality:
            stats["removed"] = len(bi_neither)
        else:
            # Keep them with NULL laterality
            if len(bi_neither) > 0:
                part = bi_neither.with_columns([
                    pl.lit(None).cast(pl.Utf8).alias("study_laterality"),
                    pl.lit(1).cast(pl.Int8).alias("was_bilateral"),
                    pl.lit(0).cast(pl.Int8).alias("has_unknown_laterality"),
                ])
                result_parts.append(part)

    # Process null laterality
    if len(null_df) > 0:
        nh = sp.null_laterality_handling
        if nh.split_if_both_sides:
            null_both = null_df.filter(pl.col("has_left") & pl.col("has_right"))
            null_other = null_df.filter(~(pl.col("has_left") & pl.col("has_right")))

            if len(null_both) > 0:
                for side in ["LEFT", "RIGHT"]:
                    part = null_both.with_columns([
                        pl.lit(side).alias("study_laterality"),
                        pl.lit(1).cast(pl.Int8).alias("was_bilateral"),
                        pl.col("has_unknown").cast(pl.Int8).alias("has_unknown_laterality"),
                    ])
                    result_parts.append(part)
                stats["split_lr"] += len(null_both)

            if nh.keep_unsplit and len(null_other) > 0:
                part = null_other.with_columns([
                    pl.lit(None).cast(pl.Utf8).alias("study_laterality"),
                    pl.lit(0).cast(pl.Int8).alias("was_bilateral"),
                    pl.lit(0).cast(pl.Int8).alias("has_unknown_laterality"),
                ])
                result_parts.append(part)
                stats["kept_null"] = len(null_other)
        elif nh.keep_unsplit:
            part = null_df.with_columns([
                pl.lit(None).cast(pl.Utf8).alias("study_laterality"),
                pl.lit(0).cast(pl.Int8).alias("was_bilateral"),
                pl.lit(0).cast(pl.Int8).alias("has_unknown_laterality"),
            ])
            result_parts.append(part)
            stats["kept_null"] = len(null_df)

    print(f"    Split L+R: {stats['split_lr']}, converted: {stats['converted_single']}, "
          f"removed: {stats['removed']}, kept_null: {stats['kept_null']}")

    result_df = pl.concat(result_parts, how="diagonal")
    result_df = result_df.drop(["lat_upper", "has_left", "has_right", "has_unknown"], strict=False)

    # Vectorized has_malignant
    result_df = result_df.with_columns(
        pl.col("study_laterality").fill_null("").str.to_uppercase().alias("_lat_upper")
    )

    both_null = pl.col("left_diagnosis").is_null() & pl.col("right_diagnosis").is_null()
    diag_expr = (
        pl.when(pl.col("_lat_upper") == "LEFT").then(pl.col("left_diagnosis"))
        .when(pl.col("_lat_upper") == "RIGHT").then(pl.col("right_diagnosis"))
        .otherwise(
            pl.when(pl.col("left_diagnosis").is_not_null())
            .then(pl.col("left_diagnosis"))
            .otherwise(pl.col("right_diagnosis"))
        )
    )
    is_malignant = diag_expr.cast(pl.Utf8).str.to_uppercase().str.contains("MALIGNANT")

    result_df = result_df.with_columns(
        pl.when(both_null).then(pl.lit(-1))
        .when(is_malignant).then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int32).alias("has_malignant")
    )
    result_df = result_df.drop(["_lat_upper"], strict=False)

    # Compute image lists and scanner_model
    images_df = images_df.with_columns(
        (pl.col("accession_number").cast(pl.Utf8) + "_" + pl.col("laterality_upper")).alias("study_key")
    )
    img_lists = images_df.group_by("study_key").agg(pl.col("image_name").alias("images"))
    img_dict = dict(zip(img_lists["study_key"].to_list(), img_lists["images"].to_list()))

    # Aggregate scanner_model (most common per study)
    if "manufacturer_model_name" in images_df.columns:
        scanner_agg = (
            images_df
            .filter(pl.col("manufacturer_model_name").is_not_null() & (pl.col("manufacturer_model_name") != ""))
            .group_by("study_key")
            .agg(pl.col("manufacturer_model_name").mode().first().alias("scanner_model"))
        )
        scanner_dict = dict(zip(scanner_agg["study_key"].to_list(), scanner_agg["scanner_model"].to_list()))
    else:
        scanner_dict = {}

    result_df = result_df.with_columns(
        (pl.col("accession_number").cast(pl.Utf8) + "_" +
         pl.col("study_laterality").fill_null("").str.to_uppercase()).alias("study_key")
    )
    images_col = [img_dict.get(k, []) for k in result_df["study_key"].to_list()]
    result_df = result_df.with_columns(pl.Series("images", images_col))

    # Add scanner_model column
    scanner_col = [scanner_dict.get(k, None) for k in result_df["study_key"].to_list()]
    result_df = result_df.with_columns(pl.Series("scanner_model", scanner_col))

    # Lesion images
    if len(lesions_df) > 0:
        lesion_lat_map = dict(zip(
            images_df["image_name"].to_list(),
            images_df["laterality_upper"].to_list()
        ))
        lesions_with_lat = lesions_df.with_columns(
            pl.col("image_name").replace_strict(lesion_lat_map, default="").alias("laterality_upper")
        )
        lesions_with_lat = lesions_with_lat.with_columns(
            (pl.col("accession_number").cast(pl.Utf8) + "_" + pl.col("laterality_upper")).alias("study_key")
        )
        lesion_lists = lesions_with_lat.group_by("study_key").agg(
            pl.col("image_name").unique().alias("lesion_images")
        )
        lesion_dict = dict(zip(lesion_lists["study_key"].to_list(), lesion_lists["lesion_images"].to_list()))
        lesion_col = [lesion_dict.get(k, []) for k in result_df["study_key"].to_list()]
    else:
        lesion_col = [[] for _ in range(len(result_df))]
    result_df = result_df.with_columns(pl.Series("lesion_images", lesion_col))

    # Compute split
    split_config = config.split
    valid_col = [compute_split(pid, split_config) for pid in result_df["patient_id"].to_list()]
    result_df = result_df.with_columns(pl.Series("valid", valid_col))

    # Add row ID
    result_df = result_df.with_row_index("id", offset=1)

    # Clean up and convert lists to strings for CSV
    result_df = result_df.drop(["study_key"], strict=False)
    if "images" in result_df.columns:
        result_df = result_df.with_columns(
            pl.col("images").cast(pl.List(pl.Utf8)).list.join(", ").alias("images")
        )
    if "lesion_images" in result_df.columns:
        result_df = result_df.with_columns(
            pl.col("lesion_images").cast(pl.List(pl.Utf8)).list.join(", ").alias("lesion_images")
        )

    # Save
    result_df.write_csv(output_dir / "BreastData.csv")
    print(f"  Saved: {len(result_df):,} rows in {time.time() - start:.2f}s")

    return result_df


def export_simple_table(parquet_path: Path, output_dir: Path, name: str):
    """Export a table directly."""
    if parquet_path is None or not parquet_path.exists():
        print(f"\nSkipping {name}: Not available")
        return
    print(f"\nExporting {name}...")
    df = pl.read_parquet(parquet_path)
    df.write_csv(output_dir / f"{name}.csv")
    print(f"  Saved: {len(df):,} rows")


def in_mini_split(exam_id: str, frac: float, seed: str) -> bool:
    """Return True if exam belongs to the mini split (deterministic hash-based)."""
    h = hashlib.md5(f"{seed}:{exam_id}".encode()).hexdigest()
    return int(h, 16) % 10000 < frac * 10000


def export_v6_files(
    images_df: pl.DataFrame,
    breast_df: pl.DataFrame,
    config: ExportConfig,
    output_dir: Path,
):
    """Export v6-specific files: ImageData_v6, BreastData_labeled_v6, BreastData_unlabeled_v6."""
    print("\n=== Exporting v6 Files ===")

    # Get accessions that still have images after filtering
    valid_accessions = set(images_df["accession_number"].unique().to_list())

    # Filter breast_df to only exams with remaining images
    breast_v6 = breast_df.filter(pl.col("accession_number").is_in(valid_accessions))
    print(f"  BreastData after image filtering: {len(breast_df):,} → {len(breast_v6):,}")

    # Apply study-level filters
    sf = config.study_filters
    if sf.is_biopsy is not None:
        before = len(breast_v6)
        breast_v6 = breast_v6.filter(pl.col("is_biopsy") == sf.is_biopsy)
        print(f"  Applied is_biopsy={sf.is_biopsy}: {before:,} → {len(breast_v6):,}")

    if sf.min_year is not None:
        before = len(breast_v6)
        breast_v6 = breast_v6.with_columns(
            pl.col("date").str.slice(0, 4).cast(pl.Int32).alias("_year")
        )
        breast_v6 = breast_v6.filter(pl.col("_year") >= sf.min_year)
        breast_v6 = breast_v6.drop("_year")
        print(f"  Applied min_year={sf.min_year}: {before:,} → {len(breast_v6):,}")

    if sf.max_year is not None:
        before = len(breast_v6)
        breast_v6 = breast_v6.with_columns(
            pl.col("date").str.slice(0, 4).cast(pl.Int32).alias("_year")
        )
        breast_v6 = breast_v6.filter(pl.col("_year") <= sf.max_year)
        breast_v6 = breast_v6.drop("_year")
        print(f"  Applied max_year={sf.max_year}: {before:,} → {len(breast_v6):,}")

    # Filter images to only those with remaining exams
    remaining_accessions = set(breast_v6["accession_number"].unique().to_list())
    images_v6 = images_df.filter(pl.col("accession_number").is_in(remaining_accessions))
    print(f"  Images after study filters: {len(images_df):,} → {len(images_v6):,}")

    # Split into labeled and unlabeled
    labeled_df = breast_v6.filter(pl.col("has_malignant") >= 0)
    unlabeled_df = breast_v6.filter(pl.col("has_malignant") == -1)

    print(f"  Labeled (has_malignant >= 0): {len(labeled_df):,}")
    print(f"  Unlabeled (has_malignant == -1): {len(unlabeled_df):,}")

    # Save v6 files
    images_v6.write_csv(output_dir / "ImageData_v6.csv")
    print(f"  Saved: ImageData_v6.csv ({len(images_v6):,} rows)")

    labeled_df.write_csv(output_dir / "BreastData_labeled_v6.csv")
    print(f"  Saved: BreastData_labeled_v6.csv ({len(labeled_df):,} rows)")

    unlabeled_df.write_csv(output_dir / "BreastData_unlabeled_v6.csv")
    print(f"  Saved: BreastData_unlabeled_v6.csv ({len(unlabeled_df):,} rows)")

    # Generate mini versions if enabled
    if config.v6.mini_enabled:
        print("\n=== Exporting v6 Mini Files ===")
        frac = config.v6.mini_fraction
        seed = config.v6.mini_seed

        # Sample labeled exams
        labeled_mini_mask = [
            in_mini_split(str(acc), frac, seed)
            for acc in labeled_df["accession_number"].to_list()
        ]
        labeled_mini = labeled_df.filter(pl.Series(labeled_mini_mask))

        # Sample unlabeled exams
        unlabeled_mini_mask = [
            in_mini_split(str(acc), frac, seed)
            for acc in unlabeled_df["accession_number"].to_list()
        ]
        unlabeled_mini = unlabeled_df.filter(pl.Series(unlabeled_mini_mask))

        # Get mini accessions and filter images
        mini_accessions = set(labeled_mini["accession_number"].to_list()) | set(unlabeled_mini["accession_number"].to_list())
        images_mini = images_v6.filter(pl.col("accession_number").is_in(mini_accessions))

        print(f"  Mini sampling at {frac*100:.0f}%:")
        print(f"    Labeled: {len(labeled_df):,} → {len(labeled_mini):,}")
        print(f"    Unlabeled: {len(unlabeled_df):,} → {len(unlabeled_mini):,}")
        print(f"    Images: {len(images_v6):,} → {len(images_mini):,}")

        # Save mini files
        images_mini.write_csv(output_dir / "ImageData_v6_mini.csv")
        print(f"  Saved: ImageData_v6_mini.csv ({len(images_mini):,} rows)")

        labeled_mini.write_csv(output_dir / "BreastData_labeled_v6_mini.csv")
        print(f"  Saved: BreastData_labeled_v6_mini.csv ({len(labeled_mini):,} rows)")

        unlabeled_mini.write_csv(output_dir / "BreastData_unlabeled_v6_mini.csv")
        print(f"  Saved: BreastData_unlabeled_v6_mini.csv ({len(unlabeled_mini):,} rows)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Configurable dataset export")
    parser.add_argument("config", type=Path, help="YAML config file")
    parser.add_argument("db_path", type=Path, help="Path to cadbusi.db")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild Parquet cache")
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Image directory for on_disk check (overrides config)")
    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.config}")
    config = ExportConfig.from_yaml(args.config)

    # CLI override for image_dir
    if args.image_dir:
        config.image_filters.on_disk_image_dir = str(args.image_dir)
    print(f"  Name: {config.name}")
    print(f"  Description: {config.description}")

    if not args.db_path.exists():
        print(f"Database not found: {args.db_path}")
        return 1

    # Determine output directory (with optional subfolder for CSVs)
    csv_output_dir = args.output_dir
    if config.output.subfolder:
        csv_output_dir = args.output_dir / config.output.subfolder
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output for reproducibility
    import shutil
    shutil.copy(args.config, csv_output_dir / "export_config.yaml")

    total_start = time.time()

    # Ensure Parquet cache (stays at db level, not in subfolder)
    print("\n=== Parquet Cache ===")
    parquet_paths = ensure_parquet_cache(args.db_path, rebuild=args.rebuild_cache)

    # Load shared data
    print("\n=== Loading Data ===")
    if parquet_paths.get("Lesions"):
        lesions_df = pl.read_parquet(parquet_paths["Lesions"])
        lesion_images = set(lesions_df["image_name"].to_list())
        print(f"  Lesions: {len(lesions_df):,} rows")
    else:
        lesions_df = pl.DataFrame()
        lesion_images = set()
        print("  Lesions: Not available")

    # Load study metadata with diagnosis columns for has_malignant
    study_meta = pl.read_parquet(parquet_paths["StudyCases"]).select([
        "accession_number", "study_laterality", "age_at_event",
        "left_diagnosis", "right_diagnosis"
    ])

    # Export based on config
    output_files = config.output.files

    images_df = None
    breast_df = None
    if "ImageData.csv" in output_files or "BreastData.csv" in output_files:
        images_df = export_image_data(parquet_paths, config, csv_output_dir, lesion_images, study_meta)

    if "BreastData.csv" in output_files:
        breast_df = export_breast_data(parquet_paths, images_df, lesions_df, config, csv_output_dir)

    if "LesionData.csv" in output_files:
        export_simple_table(parquet_paths["Lesions"], csv_output_dir, "LesionData")

    if "PathologyData.csv" in output_files:
        export_simple_table(parquet_paths["Pathology"], csv_output_dir, "PathologyData")

    if "VideoData.csv" in output_files:
        export_simple_table(parquet_paths["Videos"], csv_output_dir, "VideoData")

    # Generate v6 files if enabled
    if config.v6.enabled and breast_df is not None and images_df is not None:
        export_v6_files(images_df, breast_df, config, csv_output_dir)

    # Summary
    print("\n" + "=" * 60)
    print(f"EXPORT COMPLETE: {config.name}")
    print("=" * 60)
    print(f"  Config: {args.config.name}")
    print(f"  Output: {args.output_dir}")
    print(f"  Total time: {time.time() - total_start:.2f}s")

    return 0


if __name__ == "__main__":
    exit(main())
