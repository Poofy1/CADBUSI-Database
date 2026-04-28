"""Prompt for Gemini structured extraction of BI-RADS US descriptors."""

SYSTEM_INSTRUCTION = """\
You convert short free-text breast-ultrasound lesion descriptions into the \
ACR BI-RADS US lexicon. Inputs are very short (a handful of words to one \
sentence) — extract ONLY what is explicitly stated. Use NOT_SPECIFIED when \
a descriptor is not mentioned.\
"""

USER_PROMPT_TEMPLATE = """\
Convert this lesion description to BI-RADS-style descriptors.

DESCRIPTION:
\"\"\"
{text}
\"\"\"

Vocabulary cheatsheet (apply liberally — many synonyms map to the same enum):

- shape: oval, round, irregular, lobulated. Default NOT_SPECIFIED.
- orientation: "parallel" or "wider than tall" → PARALLEL.
  "taller than wide", "non parallel" → NOT_PARALLEL.
- margin_clarity:
   "circumscribed", "smooth", "well-defined" → CIRCUMSCRIBED
   "indistinct", "ill-defined" → INDISTINCT
   "obscured" → OBSCURED
   "spiculated", "angular", "microlobulated", "irregular border", "not circumscribed"
       → NOT_CIRCUMSCRIBED, then set margin_detail accordingly.
- margin_detail: only set if margin_clarity = NOT_CIRCUMSCRIBED.
- echo_pattern:
   "anechoic" → ANECHOIC; "hypoechoic" → HYPOECHOIC; "isoechoic" → ISOECHOIC;
   "hyperechoic" → HYPERECHOIC; "complex cystic and solid" or "mixed" → COMPLEX_CYSTIC_AND_SOLID;
   "heterogeneous" → HETEROGENEOUS.
- posterior_features:
   "posterior enhancement" → ENHANCEMENT
   "posterior acoustic shadowing", "shadowing" → SHADOWING
   "no shadowing", "non shadowing", "no posterior features" → NONE
   "mixed pattern" → COMBINED
- calcifications: only set if explicitly mentioned. ABSENT only if text says no calcs.
- vascularity:
   "nonvascular", "no flow", "avascular" → ABSENT
   "internal vascularity", "internal flow" → INTERNAL
   "peripheral vascularity", "vessels in rim" → RIM
- lesion_kind:
   "mass", "lesion" → MASS
   "simple cyst" → SIMPLE_CYST
   "complicated cyst", "complex cyst" → COMPLICATED_CYST
   "intramammary lymph node", "lymph node" → LYMPH_NODE
   "ductal" findings → DUCTAL
   anything else clearly described → OTHER

Confidence:
- HIGH if text yields ≥3 descriptors clearly.
- LOW for very short descriptions (≤3 words) or contradictions.
- MEDIUM otherwise.

Return a single JSON object matching the schema.
"""


def build_prompt(text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(text=text.strip())
