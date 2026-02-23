# P2 Developer Guide: Reusable Code & Build Requirements

This document maps what we already have in the BUS_framework evaluation pipeline
to what Paper 2 needs, specifically for the **P2 preprocessing condition**
("Structural Tissue-Aware Test") and the saliency reliability analysis.

---

## P2 Outcomes (from Experiment Matrix)

| Outcome | Description | Existing Code? |
|---------|-------------|----------------|
| Classification AUC | Per-backbone, per-preprocessing | Partial (need adapter) |
| Pointing accuracy | Saliency peak inside lesion | **Yes** |
| Lesion energy | Fraction of saliency mass inside lesion | **Yes** |
| Saliency vs boundary distance | Boundary enrichment ratio at patch resolution | **Yes** |

---

## What We Already Have

### 1. Patch-Level Tissue Decomposition (model-independent)

**File:** `mil_algorithms/evaluation/boundary_damping.py`

This module is **pure numpy** -- no model dependencies. It takes a raw
`[16, 16]` saliency array and a `[256, 256]` FOV mask and returns all the
structural metrics P2 needs.

| Function | What It Does | P2 Use |
|----------|-------------|--------|
| `compute_tissue_fraction(fov_mask_256, patch_size=16)` | Downsamples FOV mask to 16x16 patch grid, returns tissue fraction per patch | Core of P2 condition: identifies which ViT tokens are tissue vs background |
| `compute_tissue_mask(tissue_frac, threshold=0.10)` | Binary tissue/non-tissue at patch resolution | Structural mask for saliency analysis |
| `compute_boundary_band(tissue_mask, width=1)` | 1-patch-wide inner ring where tissue meets background | The zone P2 tests for boundary attraction |
| `compute_distance_to_boundary(tissue_mask)` | EDT at patch resolution (distance in patches from each tissue token to nearest non-tissue) | Stratify saliency by proximity to FOV edge |
| `compute_boundary_diagnostics(saliency, tissue_mask, band_1, band_2)` | Returns `r_band1`, `r_band2` (enrichment ratios), `r_out_of_tissue`, `mass_in_tissue` | **Exact metric** for "saliency vs boundary distance" outcome |
| `get_all_variants(saliency_16x16, fov_mask_256)` | Applies all 10 damping variants, returns dict | Compare raw vs tissue-corrected saliency across backbones |
| `classify_lesion_proximity(lesion_mask_256, tissue_mask_16)` | Labels lesion as peripheral / mid / central relative to FOV boundary | Stratified analysis: do peripheral lesions suffer more from boundary effects? |
| `postprocess_saliency(saliency_16x16, fov_mask_256)` | Recommended post-processing (edge mask + soft-tissue weighting) | Baseline correction to apply across all models |

**Usage pattern for P2:**
```python
from mil_algorithms.evaluation.boundary_damping import (
    compute_tissue_fraction,
    compute_tissue_mask,
    compute_boundary_band,
    compute_boundary_diagnostics,
    get_all_variants,
)

# These work with ANY model's saliency output
tissue_frac = compute_tissue_fraction(fov_mask_256)
tissue_mask = compute_tissue_mask(tissue_frac)
band_1 = compute_boundary_band(tissue_mask, width=1)
band_2 = compute_boundary_band(tissue_mask, width=2)

diagnostics = compute_boundary_diagnostics(saliency_16x16, tissue_mask, band_1, band_2)
# diagnostics['r_band1']  -> boundary enrichment ratio (>1.0 = attracted to boundary)
```

### 2. End-to-End Boundary Sensitivity Pipeline

**File:** `mil_algorithms/evaluation/boundary_sensitivity.py`

These functions are model-dependent (currently wired to URFM+DSMIL) but
contain the full experimental logic:

| Function | What It Does | P2 Reuse |
|----------|-------------|----------|
| `compute_boundary_saliency_metrics()` | Loads images, generates attention rollout, computes boundary energy fraction | Template for P2 -- swap saliency generation per backbone |
| `compute_bg_adjacency_saliency()` | Operates at 16x16 patch grid: downsamples FOV mask, computes ring vs interior energy, returns enrichment ratio | **Core P2 metric** -- needs backbone adapter |
| `compute_delta_max_metrics()` | Swaps background fill values (60, 80, 160, 200) and measures logit shift | Tests whether model uses padding intensity -- directly relevant to P2 structural hypothesis |
| `run_boundary_sensitivity()` | Runs all three analyses in one call | Entry point to refactor |

### 3. Standard Saliency Localization Metrics

**File:** `mil_algorithms/evaluation/saliency_metrics.py`

| Function / Class | What It Does | P2 Reuse |
|-----------------|-------------|----------|
| `compute_pointing_accuracy(saliency, mask)` | 1.0 if saliency peak inside lesion, 0.0 otherwise | Direct reuse |
| `compute_distance_to_lesion(saliency, mask)` | Euclidean distance (px) from peak to nearest lesion pixel | Direct reuse |
| `compute_lesion_energy(saliency, mask)` | Fraction of saliency energy inside lesion mask | Direct reuse |
| `compute_saliency_auc(saliency, mask)` | ROC-AUC treating saliency as pixel-level lesion predictor | Direct reuse |
| `SaliencyEvaluator` class | Full pipeline: loads gold masks, checks leakage, computes all metrics with stratification by pathology/size/split | Refactor to accept arbitrary backbone |
| `smooth_saliency(mode='patch')` | Resolution-aware Gaussian smoothing (sigma=patch_size) | Direct reuse |

### 4. Saliency Visualization

**File:** `mil_algorithms/evaluation/saliency_metrics.py` (bottom half)

| Function | P2 Reuse |
|----------|----------|
| `generate_paper_saliency_grid()` | Reuse for P2 figure generation |
| `generate_clinical_saliency_panel()` | Reuse for cross-backbone comparison panels |

### 5. Evaluation Reporter Pipeline

**File:** `mil_algorithms/evaluation/v5_reporters/boundary_sensitivity.py`

The v5 reporter wraps `boundary_sensitivity.py` and generates markdown
reports with summary statistics. This is the integration layer that
`evaluate_v5.py` calls. P2 would extend or parallel this.

---

## What Needs to Be Built

### 1. Backbone Adapter Interface (CRITICAL)

All existing code assumes URFM+DSMIL for two operations:
- **Feature extraction** (forward pass through backbone)
- **Saliency generation** (attention rollout through ViT layers)

P2 needs a unified interface so the same downstream analysis works with
any backbone.

```python
# Proposed interface
class BackboneAdapter:
    """Unified interface for P2 backbone comparison."""

    def extract_features(self, image_tensor) -> torch.Tensor:
        """[B, 3, 256, 256] -> [B, D] or [B, N, D] features."""
        ...

    def get_saliency_map(self, image_tensor) -> np.ndarray:
        """[1, 3, 256, 256] -> [16, 16] saliency map.

        Mechanism varies by architecture:
        - ViT (URFM, USFM, OpenUS): attention rollout
        - CNN (ResNet, ConvNeXt): GradCAM
        - UltraSam: encoder attention or GradCAM
        """
        ...

    def get_logit(self, image_tensor) -> float:
        """Single-image inference, return raw logit."""
        ...
```

**Backbones to implement:**

| Backbone | Architecture | Saliency Mechanism | Patch Grid | Notes |
|----------|-------------|-------------------|------------|-------|
| URFM | ViT-B/16 | Attention rollout | 16x16 | Already implemented |
| USFM | ViT-B/16 (likely) | Attention rollout | 16x16 | Check architecture |
| OpenUS | ViT variant | Attention rollout | TBD | Check patch size |
| UltraSam | SAM encoder | Encoder attention or GradCAM | TBD | Classification head needed |
| ResNet-50 | CNN | GradCAM | Variable | Spatial map != 16x16 -- need resampling |
| ConvNeXt-T | CNN | GradCAM | Variable | Same resampling issue |

**Key issue for CNNs:** GradCAM produces spatial maps at the final conv
layer resolution (e.g., 8x8 for ResNet-50), not 16x16. Need a resampling
step to normalize all saliency maps to a common grid before feeding into
`boundary_damping.py`.

### 2. MIL Aggregator Wiring for Non-URFM Backbones

The DSMIL aggregator expects `[N, 768]` features (URFM output dim).
Other backbones have different feature dimensions:

| Backbone | Feature Dim | Action Needed |
|----------|-------------|---------------|
| URFM | 768 | None (current) |
| USFM | 768 (if ViT-B) | Likely none |
| OpenUS | TBD | Check, may need projection |
| UltraSam | 256 (SAM encoder) | Projection layer |
| ResNet-50 | 2048 | Projection layer |
| ConvNeXt-T | 768 | Likely none |

Options: (a) add a `nn.Linear(backbone_dim, 768)` projection, or
(b) configure DSMIL's input dim per backbone.

### 3. Preprocessing Pipeline Variants (P0, P1, P2)

Three data directories need to exist:

| ID | Description | Data Dir | Status |
|----|-------------|----------|--------|
| P0 | Baseline (AJR pipeline) | `mayo_256_v7_old_crops/` | **Exists** |
| P1 | Debris removed | `mayo_256_v7_tight_crops/` or `mayo_v7_256_masked/` | **Exists** (verify which maps to P1) |
| P2 | Structural tissue-aware (mask + boundary mitigation) | TBD | **Needs definition** |

P2 preprocessing likely involves:
- FOV masking with `preprocessing_pipeline/ui_mask.py`
- Fill-jitter or matched-noise in background regions
- Possibly masked [CLS] token aggregation (from `soft_topk_only_v7_no_calipers_k5_fill_jitter_masked_cls`)

Check existing experiment configs for the debris/masked-CLS variants --
some of this may already exist as training configs.

### 4. Cross-Backbone Evaluation Script

A new script (or extension of `evaluate_v5.py`) that:
1. Loads each backbone via the adapter interface
2. Runs inference on the same test set under each preprocessing condition
3. Computes classification AUC, saliency metrics, and boundary diagnostics
4. Generates comparison tables and figures

Skeleton:
```python
for backbone in [URFM, USFM, OpenUS, UltraSam, ResNet50, ConvNeXt]:
    adapter = BackboneAdapter(backbone)
    for preprocess in [P0, P1, P2]:
        # Classification
        auc = evaluate_classification(adapter, test_loader[preprocess])

        # Saliency (using existing gold masks)
        for image, mask in gold_saliency_set:
            sal_16x16 = adapter.get_saliency_map(image)
            pointing = compute_pointing_accuracy(sal_upsampled, mask)
            energy = compute_lesion_energy(sal_upsampled, mask)
            diag = compute_boundary_diagnostics(sal_16x16, tissue_mask, ...)

        # Delta Max (structural sensitivity)
        delta_max = compute_delta_max_metrics(adapter, ...)
```

### 5. Gold Saliency Masks for P2 Preprocessing

The existing gold masks are registered under crop-specific directories:

| Crop Type | Gold Dir |
|-----------|----------|
| loose_crops (P0) | `segmentation/annotation/gold_saliency/loose_crops/gold_only` |
| tight_crops | `segmentation/annotation/gold_saliency/tight_crops/gold_only` |
| debris_crops | `segmentation/annotation/gold_saliency/debris_crops/gold_only` |

P2's structural preprocessing may need a new gold mask remap if the image
geometry changes. If P2 uses the same 256x256 letterbox geometry as P0,
the existing loose_crops gold masks can be reused directly.

---

## Reuse Summary

```
                        EXISTING (reuse)              NEEDS BUILDING
                        ================              ==============

Saliency metrics        saliency_metrics.py           --
  pointing accuracy     compute_pointing_accuracy()
  lesion energy         compute_lesion_energy()
  distance (mm)         compute_distance_to_lesion()
  Hit@2mm, Hit@4mm      SaliencyMethodResults

Boundary analysis       boundary_damping.py           --
  tissue decomposition  compute_tissue_fraction()
  boundary band         compute_boundary_band()
  enrichment ratio      compute_boundary_diagnostics()
  damping variants      get_all_variants()
  lesion proximity      classify_lesion_proximity()

Boundary sensitivity    boundary_sensitivity.py       Backbone adapter swap
  boundary energy       compute_boundary_saliency_metrics()
  BG adjacency          compute_bg_adjacency_saliency()
  delta max             compute_delta_max_metrics()

Saliency generation     AttentionRollout (URFM only)  Adapters for 5 models
                                                      GradCAM for CNNs
                                                      Saliency grid normalization

Classification eval     evaluate_v5.py                Backbone adapter
                                                      Cross-backbone comparison script

Preprocessing data      P0 exists, P1 likely exists   P2 data dir definition
Gold masks              loose/tight/debris exist      P2 remap if geometry changes
Visualization           saliency_metrics.py panels    Cross-backbone comparison figs
```

---

## Suggested Build Order

1. **Backbone adapter interface** -- define the contract, implement URFM first
   (wrapping existing code) to validate the interface works end-to-end.
2. **Second ViT backbone** (USFM or OpenUS) -- should be straightforward if
   also ViT-B/16 with attention rollout.
3. **CNN adapter** (ResNet-50) -- requires GradCAM and saliency grid resampling.
   This is the hardest adapter since the saliency mechanism is fundamentally
   different.
4. **P2 preprocessing pipeline** -- define and generate the data directory.
5. **Cross-backbone evaluation script** -- wire the adapters into the existing
   metric functions.
6. **Remaining backbones** (UltraSam, ConvNeXt-T) -- fill out the matrix.
