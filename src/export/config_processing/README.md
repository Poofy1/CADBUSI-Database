# Data Processing Scripts

Scripts for exporting and processing data from CADBUSI databases.

## Scripts

### export_configurable.py (Main Export Script)

Creates dataset exports using YAML configuration files. Supports different filtering profiles without code changes.

**Performance:** ~10s with cached Parquet (first run ~120s to build cache)

```bash
# v6 dataset (5 scanners, year >= 2018, is_biopsy = 0)
python export_configurable.py configs/v6.yaml /path/to/cadbusi.db ./output

# Standard baseline export (keeps all images with exclusion_reason)
python export_configurable.py configs/baseline_v1.yaml /path/to/cadbusi.db ./output

# Strict breast-only export (removes excluded images, tighter filters)
python export_configurable.py configs/strict_breast_only.yaml /path/to/cadbusi.db ./output
```

### cache_parquet.py (Parquet Cache Manager)

Converts SQLite tables to Parquet format for ~100x faster loading. Used automatically by `export_configurable.py` but can be run standalone.

```bash
# Build cache (first run takes ~15-20s)
python cache_parquet.py /path/to/cadbusi.db

# Check cache status
python cache_parquet.py /path/to/cadbusi.db --status

# Force rebuild all cache files
python cache_parquet.py /path/to/cadbusi.db --rebuild
```

## Configuration Files

All configs are in `configs/`:

| Config | Description |
|--------|-------------|
| `v6.yaml` | Production v6 dataset: 5 active scanners, year >= 2018, non-biopsy only, with mini (20%) versions |
| `baseline_v1.yaml` | Standard filters, keeps all images with exclusion_reason column |
| `strict_breast_only.yaml` | Strict filtering: breast area only, tighter thresholds, removes excluded |

### Creating Custom Configs

```yaml
name: "my_config"
description: "Custom dataset configuration"

image_filters:
  darkness_max: 70        # Lower = stricter
  allowed_areas: [breast] # No 'unknown'
  min_dimension: 256      # Larger minimum

scanner_filters:
  allowed_scanners:       # Only these scanners
    - "LOGIQE10"
    - "EPIQ 7G"

study_filters:
  is_biopsy: 0            # Non-biopsy only
  min_year: 2020          # Recent data only

output:
  apply_image_filters: true  # Actually remove excluded
  subfolder: "my_dataset"    # Output to subfolder
```

See config files for full options.

## Documentation

| File | Description |
|------|-------------|
| `Baseline_Filtering.md` | Detailed filter documentation |
| `v6_filtering.md` | V6 dataset filters and summary statistics |

## Output Locations

Exports typically go to:
- `data/splits/v6/` - Current v6 splits
- `data/exports/` - Raw exports from cadbusi.db

## Requirements

```
polars
connectorx
pyarrow
pyyaml
```
