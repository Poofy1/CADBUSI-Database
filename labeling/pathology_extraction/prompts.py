"""Prompt for Gemini structured extraction of pathology addenda."""

SYSTEM_INSTRUCTION = """\
You extract structured pathology information from brief addenda attached to \
Mayo Clinic breast imaging reports. The addenda state biopsy results in 1-5 \
sentences. Extract only what is EXPLICITLY stated.
"""

USER_PROMPT_TEMPLATE = """\
Extract the pathology information from this report addendum.

TEXT:
\"\"\"
{text}
\"\"\"

Rules:
- If the text contains both carcinoma and benign findings (e.g., "invasive ductal \
carcinoma with associated fibrocystic change"), list BOTH.
- Treat "DCIS", "ductal carcinoma in situ" → DCIS (even without the word "invasive").
- Treat "IDC" or "invasive ductal carcinoma" → INVASIVE_DUCTAL_CARCINOMA.
- Treat "infiltrating ductal carcinoma" as synonymous with IDC.
- "Suspicious for carcinoma" without definitive diagnosis → primary = SUSPICIOUS, \
cancer_subtypes empty.
- "Atypical ductal/lobular hyperplasia" without carcinoma → primary = ATYPICAL, \
appropriate benign_subtype (ADH / ALH), cancer_subtypes empty.
- If the text describes a LYMPH NODE biopsy (e.g., "lymph node, sentinel biopsy"), \
set is_lymph_node_biopsy=true and populate lymph_node_status accordingly.
  - "negative for metastatic carcinoma" → NEGATIVE
  - "positive for metastatic carcinoma" → POSITIVE
  - "isolated tumor cells" → ISOLATED_TUMOR_CELLS
  - "micrometastasis" → MICROMETASTASIS
- If the text is only about lymph nodes AND negative, primary_diagnosis = BENIGN.
- Extract size only if a specific measurement is given (e.g., "measuring 8 mm" → 8.0).
  Ignore ranges or "at least" phrasing unless a single number is clear.
- Use NOT_SPECIFIED / NOT_REPORTED when information is absent. Do not guess.
- Confidence LOW if the text is very short (<10 words), truncated, or ambiguous.

Return a single JSON object matching the schema.
"""


def build_prompt(text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(text=text.strip())
