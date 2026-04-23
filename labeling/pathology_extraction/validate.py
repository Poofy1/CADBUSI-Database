"""Validate LLM extraction against the 924 accessions that exist in BOTH
`Pathology` table (structured) and `rad_pathology_txt` (free text).

For each accession:
  - Our extraction's cancer_subtypes vs all carcinoma-containing cancer_type rows in Pathology
  - primary_diagnosis agreement (MALIGNANT/BENIGN/ATYPICAL)
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from bus_data import ManifestDB
from data.pathology_extraction import (
    PathologyExtractor, PrimaryDiagnosis, CancerSubtype, BenignSubtype
)

OUT_DIR = PROJECT_ROOT / 'data/pathology_extraction/validation_output'
OUT_DIR.mkdir(exist_ok=True)

# Map structured Pathology.cancer_type strings to our CancerSubtype enum
# for direct comparison on the 924-overlap set.
PATH_TABLE_TO_SUBTYPE = {
    'INVASIVE DUCTAL CARCINOMA': CancerSubtype.IDC,
    'DUCTAL CARCINOMA IN SITU': CancerSubtype.DCIS,
    'INVASIVE LOBULAR CARCINOMA': CancerSubtype.ILC,
    'LOBULAR CARCINOMA IN SITU': CancerSubtype.LCIS,
    'INVASIVE CARCINOMA': CancerSubtype.INVASIVE_CARCINOMA_NOS,
    'INVASIVE MAMMARY CARCINOMA': CancerSubtype.INVASIVE_MAMMARY_CARCINOMA,
    'MUCINOUS CARCINOMA': CancerSubtype.MUCINOUS,
    'TUBULAR CARCINOMA': CancerSubtype.TUBULAR,
    'PAPILLARY CARCINOMA': CancerSubtype.PAPILLARY,
    'MEDULLARY CARCINOMA': CancerSubtype.MEDULLARY,
    'METASTATIC CARCINOMA': CancerSubtype.METASTATIC,
    'ADENOID CYSTIC CARCINOMA': CancerSubtype.ADENOID_CYSTIC,
    'INFLAMMATORY CARCINOMA': CancerSubtype.INFLAMMATORY,
    'ADENOCARCINOMA': CancerSubtype.CARCINOMA_NOS,
    'CARCINOMA': CancerSubtype.CARCINOMA_NOS,
}
MALIGNANT_STRINGS = set(PATH_TABLE_TO_SUBTYPE.keys())


def classify_path_row_primary(cancer_type: str) -> str:
    """Map a Pathology.cancer_type string to MALIGNANT / BENIGN / ATYPICAL / OTHER."""
    if cancer_type in MALIGNANT_STRINGS:
        return 'MALIGNANT'
    ct = (cancer_type or '').upper()
    if ct == 'BENIGN':
        return 'BENIGN'
    if 'ATYPICAL' in ct or 'HYPERPLASIA' in ct:
        return 'ATYPICAL'
    if ct in ('UNKNOWN', 'OTHER', 'ISOLATED TUMOR CELLS', 'MICROMETASTASIS'):
        return 'OTHER'  # ambiguous; skip in primary agreement
    return 'OTHER'


def main():
    db = ManifestDB()

    # Pull the 924 overlap
    overlap = db.query_df("""
        SELECT sc.accession_number, sc.rad_pathology_txt
        FROM src.StudyCases sc
        WHERE sc.rad_pathology_txt IS NOT NULL
          AND LENGTH(TRIM(sc.rad_pathology_txt)) > 20
          AND sc.accession_number IN (
              SELECT DISTINCT accession_number FROM src.Pathology
          )
    """)
    print(f"Overlap set: {len(overlap)} accessions with BOTH Pathology rows and rad_pathology_txt")
    print()

    # Get ALL Pathology rows per accession (not drop_duplicates)
    path_rows = db.query_df("""
        SELECT accession_number, cancer_type
        FROM src.Pathology
        WHERE cancer_type IS NOT NULL AND cancer_type != ''
    """)

    # Aggregate per accession: set of mapped subtypes + primary categories observed
    by_acc_subtypes = {}
    by_acc_primary = {}
    for acc, grp in path_rows.groupby('accession_number'):
        subs = set()
        prims = set()
        for ct in grp['cancer_type']:
            if ct in PATH_TABLE_TO_SUBTYPE:
                subs.add(PATH_TABLE_TO_SUBTYPE[ct])
            prims.add(classify_path_row_primary(ct))
        by_acc_subtypes[acc] = subs
        by_acc_primary[acc] = prims

    # Extract with Gemini
    items = list(zip(overlap['accession_number'], overlap['rad_pathology_txt']))
    extractor = PathologyExtractor()
    print(f"Extracting {len(items)} records with {extractor.model} (parallel)...")
    import time
    t0 = time.time()
    results = extractor.extract_batch_parallel(items, max_workers=16, progress_every=100)
    elapsed = time.time() - t0
    ok = [r for r in results if r.extraction is not None]
    print(f"\n✓ Done in {elapsed:.1f}s ({elapsed/len(items):.2f}s/record avg)")
    print(f"  Success: {len(ok)}/{len(items)}  Errors: {len(items)-len(ok)}")

    # Compare
    primary_agree = 0
    primary_disagree = 0
    primary_skipped = 0   # neither side had a clean category
    per_subtype_stats = Counter()    # (subtype, outcome) → count
    exact_subtype_match = 0
    any_subtype_overlap = 0
    no_overlap = 0
    none_expected = 0    # pathology side had no cancer subtype mapped (e.g., all ADH/UNKNOWN)

    disagreements = []
    for r in ok:
        acc = r.accession
        ext = r.extraction
        llm_primary = ext.primary_diagnosis.value
        ref_prims = by_acc_primary.get(acc, set())
        # Primary agreement — if any Pathology row is MALIGNANT, the reference primary is MALIGNANT
        if 'MALIGNANT' in ref_prims:
            ref_primary = 'MALIGNANT'
        elif 'BENIGN' in ref_prims and 'ATYPICAL' not in ref_prims:
            ref_primary = 'BENIGN'
        elif 'ATYPICAL' in ref_prims:
            ref_primary = 'ATYPICAL'
        else:
            ref_primary = None

        if ref_primary is None:
            primary_skipped += 1
        elif ref_primary == llm_primary:
            primary_agree += 1
        else:
            primary_disagree += 1
            if len(disagreements) < 15:
                disagreements.append({
                    'acc': acc,
                    'text': r.text[:180],
                    'llm': llm_primary,
                    'ref': ref_primary,
                    'llm_cancer': [c.value for c in ext.cancer_subtypes],
                    'llm_benign': [b.value for b in ext.benign_subtypes],
                    'ref_primary_set': list(ref_prims),
                })

        # Subtype-level comparison (only when there are carcinoma rows on reference side)
        ref_subs = by_acc_subtypes.get(acc, set())
        llm_subs = set(ext.cancer_subtypes)
        if not ref_subs:
            none_expected += 1
            continue
        if llm_subs == ref_subs:
            exact_subtype_match += 1
            any_subtype_overlap += 1
        elif llm_subs & ref_subs:
            any_subtype_overlap += 1
        else:
            no_overlap += 1

        # Per-subtype hit rate
        for s in ref_subs:
            if s in llm_subs:
                per_subtype_stats[(s.value, 'hit')] += 1
            else:
                per_subtype_stats[(s.value, 'miss')] += 1
        for s in llm_subs - ref_subs:
            per_subtype_stats[(s.value, 'extra')] += 1

    # Report
    print()
    print("=" * 70)
    print("Primary diagnosis agreement (MALIGNANT / BENIGN / ATYPICAL)")
    print("=" * 70)
    total_primary = primary_agree + primary_disagree
    if total_primary > 0:
        print(f"  Agree:    {primary_agree}/{total_primary} ({primary_agree/total_primary:.1%})")
        print(f"  Disagree: {primary_disagree}/{total_primary} ({primary_disagree/total_primary:.1%})")
    print(f"  Skipped (ref ambiguous): {primary_skipped}")

    print()
    print("=" * 70)
    print("Cancer subtype agreement (when Pathology table has carcinoma rows)")
    print("=" * 70)
    total_with_cancer_ref = exact_subtype_match + any_subtype_overlap - exact_subtype_match + no_overlap
    # Actually: any_subtype_overlap already includes exact. Report cleanly:
    total_with_cancer_ref = exact_subtype_match + (any_subtype_overlap - exact_subtype_match) + no_overlap
    print(f"  Total with reference cancer subtype: {total_with_cancer_ref}")
    print(f"  Exact match (sets equal):            {exact_subtype_match} ({exact_subtype_match/max(total_with_cancer_ref,1):.1%})")
    print(f"  Any overlap (at least one match):    {any_subtype_overlap} ({any_subtype_overlap/max(total_with_cancer_ref,1):.1%})")
    print(f"  No overlap (complete miss):          {no_overlap} ({no_overlap/max(total_with_cancer_ref,1):.1%})")
    print(f"  No cancer in reference (skipped):    {none_expected}")

    print()
    print("=" * 70)
    print("Per-subtype recall (hit / hit+miss from reference side)")
    print("=" * 70)
    all_subs = set(s for s,_ in per_subtype_stats.keys())
    for s in sorted(all_subs):
        hit = per_subtype_stats[(s, 'hit')]
        miss = per_subtype_stats[(s, 'miss')]
        extra = per_subtype_stats[(s, 'extra')]
        tot = hit + miss
        recall = hit / tot if tot else 0.0
        print(f"  {s:<35} hit={hit:>3} miss={miss:>3} extra={extra:>3}  recall={recall:.1%}")

    # Save disagreements for eyeball review
    with open(OUT_DIR / 'validation_disagreements.json', 'w') as f:
        json.dump(disagreements, f, indent=2)
    # Save full results
    pd.DataFrame([{
        'accession': r.accession,
        'text': r.text[:500],
        'llm_primary': r.extraction.primary_diagnosis.value if r.extraction else None,
        'llm_cancer': '|'.join(c.value for c in r.extraction.cancer_subtypes) if r.extraction else None,
        'llm_benign': '|'.join(b.value for b in r.extraction.benign_subtypes) if r.extraction else None,
        'llm_laterality': r.extraction.laterality.value if r.extraction else None,
        'llm_confidence': r.extraction.confidence.value if r.extraction else None,
        'ref_primary_set': '|'.join(sorted(by_acc_primary.get(r.accession, set()))),
        'ref_subtypes': '|'.join(sorted(s.value for s in by_acc_subtypes.get(r.accession, set()))),
        'error': r.error,
    } for r in results]).to_csv(OUT_DIR / 'validation_full.csv', index=False)
    print()
    print(f"✓ Saved disagreements: {OUT_DIR/'validation_disagreements.json'}")
    print(f"✓ Saved full results:  {OUT_DIR/'validation_full.csv'}")


if __name__ == "__main__":
    main()
