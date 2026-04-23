"""Prompt for structured extraction of full synoptic pathology reports."""

SYNOPTIC_SYSTEM_INSTRUCTION = """\
You are extracting structured data from formal breast pathology reports. \
These reports often use standard synoptic formats (Nottingham grade breakdown, \
AJCC 8th edition pTNM staging, margin distances in mm, lymph node counts, \
ER/PR/HER2 immunohistochemistry results). Extract only what is EXPLICITLY \
stated. Do not infer or default — use NOT_REPORTED / NOT_APPLICABLE when \
information is absent.
"""

SYNOPTIC_USER_PROMPT = """\
Extract structured data from this synoptic pathology report.

REPORT:
\"\"\"
{text}
\"\"\"

Key extraction rules:

TUMOR SIZING:
- Report all sizes in MILLIMETERS. If the report says "5.0 cm" → 50.0. "7 mm" → 7.0.
- tumor_size_mm = largest single dimension stated.
- tumor_size_additional_mm = other dimensions if given (e.g., "50 x 30 x 22 mm" → largest 50, additional [30, 22]).

NOTTINGHAM GRADE:
- Extract each sub-score (glandular, nuclear, mitotic) as int 1-3.
- grade_overall is the final Nottingham grade (Grade 1/2/3) — sometimes stated as "Score 6 of 9; Grade 2" → GRADE_2.

MARGINS:
- "Closest margin is posterior at least 10 mm" → invasive_margin_status=NEGATIVE, closest_invasive_margin_name="posterior", closest_invasive_margin_distance_mm=10.0.
- "Positive margin" or "tumor at inked margin" → POSITIVE.
- Not applicable for core biopsies (not excised).

LYMPH NODES:
- Extract integer counts for total examined, macromets, micromets, ITCs.
- "Number of Lymph Nodes with Micrometastases: 0" → with_micrometastases=0.
- If not discussed, leave the `lymph_nodes` object as null.

STAGING (AJCC 8th):
- Extract pT, pN, pM as stated. Handle variants like "pT2 (tumor > 20 mm...)" → pT="pT2".
- "pN0(i+)" with isolated tumor cells → pN="pN0_i_plus".
- "Regional Lymph Nodes Modifier: sn" → sentinel_modifier=true.

RECEPTOR STATUS:
- "ESTROGEN RECEPTOR: Positive. Percent cells staining: 95%" → er_status=POSITIVE, er_percent=95.
- "PROGESTERONE RECEPTOR: Negative" → pr_status=NEGATIVE.
- "HER2: Negative (1+ by IHC)" → her2_status=NEGATIVE, her2_method=IHC, her2_ihc_score=1.
- "HER2: Equivocal (2+) — FISH pending" → her2_status=EQUIVOCAL, her2_method=IHC, her2_ihc_score=2.
- "ER 1-10%" → er_status=LOW_POSITIVE (not just POSITIVE).
- If biomarkers were tested on a PRIOR specimen, say so in notes but do NOT populate the receptors block (those results belong to the prior specimen, not this one).

MULTI-PART SPECIMENS:
- If the report has parts "A. Breast, right, mastectomy" + "B. Lymph node, sentinel biopsy", set multi_part_specimen=true.
- Extract the PRIMARY breast finding as the main diagnosis.
- Still capture lymph_nodes findings (they refer to the same case).

DIAGNOSIS ROUTING:
- Invasive carcinoma present → primary_diagnosis=INVASIVE_CARCINOMA, populate `invasive`.
- DCIS/LCIS only (no invasion mentioned) → primary_diagnosis=IN_SITU_ONLY, populate `in_situ`.
- ADH/ALH only → primary_diagnosis=ATYPICAL, populate `benign` with atypia.
- Benign findings (fibroadenoma, fibrocystic, papilloma) → primary_diagnosis=BENIGN, populate `benign`.
- Lymph node specimen only:
  - "positive for metastatic carcinoma" → primary_diagnosis=METASTATIC_LYMPH_NODE.
  - "negative for metastatic carcinoma" → primary_diagnosis=NEGATIVE_LYMPH_NODE.
- "Nondiagnostic", "insufficient tissue" → primary_diagnosis=INSUFFICIENT.

CONFIDENCE:
- HIGH if the report is a clear synoptic with most fields stated.
- MEDIUM for partial reports or brief descriptive diagnoses.
- LOW if the text is truncated, ambiguous, or mostly prose without structured data.

Return a single JSON object matching the SynopticExtraction schema.
"""


def build_synoptic_prompt(text: str) -> str:
    return SYNOPTIC_USER_PROMPT.format(text=text.strip())
