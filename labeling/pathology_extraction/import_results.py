"""Import Phase 1 / Phase 2 result JSONLs (from VM) back into the local manifest DB.

For Phase 1: writes to existing `pathology_extracted` table.
For Phase 2: creates/populates `pathology_synoptic_extracted` table.
"""
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path

MANIFEST = '/home/jbaggett/BUS_framework/data/registry/bus_manifest_v3.db'

PHASE1_INSERT = """
INSERT OR REPLACE INTO pathology_extracted
  (accession_number, source_text, primary_diagnosis, cancer_subtypes, benign_subtypes,
   laterality, size_mm, grade, lymph_node_status, is_lymph_node_biopsy,
   confidence, notes, model_name, extraction_error)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

PHASE2_SCHEMA = """
CREATE TABLE IF NOT EXISTS pathology_synoptic_extracted (
    path_id                  INTEGER PRIMARY KEY,
    accession_number         TEXT NOT NULL,
    specimen_type            TEXT,
    specimen_site            TEXT,
    laterality               TEXT,
    multi_part_specimen      INTEGER,
    primary_diagnosis        TEXT,
    -- Invasive tumor
    histologic_type          TEXT,
    grade_overall            TEXT,
    nottingham_glandular     INTEGER,
    nottingham_nuclear       INTEGER,
    nottingham_mitotic       INTEGER,
    tumor_size_mm            REAL,
    tumor_size_additional_mm TEXT,      -- JSON list
    focality                 TEXT,
    lymphovascular_invasion  INTEGER,
    dermal_lvi               INTEGER,
    perineural_invasion      INTEGER,
    -- In-situ
    dcis_present             INTEGER,
    dcis_nuclear_grade       TEXT,
    dcis_pattern             TEXT,       -- JSON list
    dcis_necrosis            INTEGER,
    dcis_size_mm             REAL,
    lcis_present             INTEGER,
    -- Margins
    invasive_margin_status       TEXT,
    closest_invasive_margin_name TEXT,
    closest_invasive_margin_mm   REAL,
    dcis_margin_status           TEXT,
    closest_dcis_margin_mm       REAL,
    -- LN
    ln_total_examined        INTEGER,
    ln_sentinel_examined     INTEGER,
    ln_with_macromets        INTEGER,
    ln_with_micromets        INTEGER,
    ln_with_itc              INTEGER,
    ln_largest_deposit_mm    REAL,
    ln_extranodal_extension  INTEGER,
    -- Staging
    pT                       TEXT,
    pN                       TEXT,
    pM                       TEXT,
    sentinel_modifier        INTEGER,
    overall_stage            TEXT,
    -- Receptors
    er_status                TEXT,
    er_percent               REAL,
    er_intensity             TEXT,
    pr_status                TEXT,
    pr_percent               REAL,
    pr_intensity             TEXT,
    her2_status              TEXT,
    her2_method              TEXT,
    her2_ihc_score           INTEGER,
    her2_fish_ratio          REAL,
    ki67_percent             REAL,
    -- Treatment
    post_neoadjuvant         INTEGER,
    treatment_effect         TEXT,
    -- Benign
    primary_benign_dx        TEXT,
    atypia_present           INTEGER,
    atypia_type              TEXT,
    -- Meta
    confidence               TEXT,
    extraction_notes         TEXT,
    model_name               TEXT,
    extraction_error         TEXT,
    created_at               TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pse_accession ON pathology_synoptic_extracted(accession_number);
CREATE INDEX IF NOT EXISTS idx_pse_primary ON pathology_synoptic_extracted(primary_diagnosis);
"""

PHASE2_INSERT = """
INSERT OR REPLACE INTO pathology_synoptic_extracted (
    path_id, accession_number, specimen_type, specimen_site, laterality, multi_part_specimen,
    primary_diagnosis, histologic_type, grade_overall, nottingham_glandular, nottingham_nuclear, nottingham_mitotic,
    tumor_size_mm, tumor_size_additional_mm, focality, lymphovascular_invasion, dermal_lvi, perineural_invasion,
    dcis_present, dcis_nuclear_grade, dcis_pattern, dcis_necrosis, dcis_size_mm, lcis_present,
    invasive_margin_status, closest_invasive_margin_name, closest_invasive_margin_mm,
    dcis_margin_status, closest_dcis_margin_mm,
    ln_total_examined, ln_sentinel_examined, ln_with_macromets, ln_with_micromets, ln_with_itc,
    ln_largest_deposit_mm, ln_extranodal_extension,
    pT, pN, pM, sentinel_modifier, overall_stage,
    er_status, er_percent, er_intensity, pr_status, pr_percent, pr_intensity,
    her2_status, her2_method, her2_ihc_score, her2_fish_ratio, ki67_percent,
    post_neoadjuvant, treatment_effect,
    primary_benign_dx, atypia_present, atypia_type,
    confidence, extraction_notes, model_name, extraction_error
) VALUES (?, ?, ?, ?, ?, ?,  ?, ?, ?, ?, ?, ?,  ?, ?, ?, ?, ?, ?,  ?, ?, ?, ?, ?, ?,
         ?, ?, ?,  ?, ?,  ?, ?, ?, ?, ?,  ?, ?,  ?, ?, ?, ?, ?,
         ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,  ?, ?,  ?, ?, ?,  ?, ?, ?, ?)
"""


def _b(v):
    """Python bool → 0/1 (or None)."""
    if v is None: return None
    return 1 if v else 0


def _j(v):
    return None if v is None else json.dumps(v)


def import_phase1(jsonl_path: Path, model_name: str = 'gemini-2.5-flash'):
    conn = sqlite3.connect(MANIFEST)
    cur = conn.cursor()
    n_ok = 0
    n_err = 0
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            acc = r['accession_number']
            if r.get('error'):
                cur.execute(PHASE1_INSERT, (acc, r.get('text', ''), 'UNKNOWN', '', '', 'NOT_SPECIFIED',
                                            None, None, None, 0, 'LOW', None, model_name, r['error']))
                n_err += 1
            else:
                e = r['extraction']
                cur.execute(PHASE1_INSERT, (
                    acc, r.get('text', ''),
                    e['primary_diagnosis'],
                    '|'.join(e.get('cancer_subtypes') or []),
                    '|'.join(e.get('benign_subtypes') or []),
                    e.get('laterality', 'NOT_SPECIFIED'),
                    e.get('size_mm'),
                    e.get('grade', 'NOT_SPECIFIED'),
                    e.get('lymph_node_status', 'NOT_REPORTED'),
                    _b(e.get('is_lymph_node_biopsy', False)),
                    e.get('confidence', 'MEDIUM'),
                    e.get('notes'),
                    model_name,
                    None,
                ))
                n_ok += 1
    conn.commit()
    conn.close()
    print(f"Phase 1 import: {n_ok} ok, {n_err} errors")


def import_phase2(jsonl_path: Path, model_name: str = 'gemini-2.5-flash'):
    conn = sqlite3.connect(MANIFEST)
    conn.executescript(PHASE2_SCHEMA)
    cur = conn.cursor()
    n_ok = 0
    n_err = 0
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            path_id = r['path_id']
            acc = r['accession_number']

            if r.get('error'):
                # Insert a stub row for audit trail
                params = [path_id, acc] + [None] * 57 + [model_name, r['error']]
                cur.execute(PHASE2_INSERT, params)
                n_err += 1
                continue

            e = r['extraction']
            inv = e.get('invasive') or {}
            isi = e.get('in_situ') or {}
            mar = e.get('margins') or {}
            ln  = e.get('lymph_nodes') or {}
            stg = e.get('staging') or {}
            rc  = e.get('receptors') or {}
            tx  = e.get('treatment_context') or {}
            bn  = e.get('benign') or {}

            params = (
                path_id, acc,
                e.get('specimen_type'), e.get('specimen_site'),
                e.get('laterality'), _b(e.get('multi_part_specimen')),
                e.get('primary_diagnosis'),
                inv.get('histologic_type'), inv.get('grade_overall'),
                inv.get('nottingham_glandular'), inv.get('nottingham_nuclear'), inv.get('nottingham_mitotic'),
                inv.get('tumor_size_mm'), _j(inv.get('tumor_size_additional_mm')),
                inv.get('focality'),
                _b(inv.get('lymphovascular_invasion')),
                _b(inv.get('dermal_lymphovascular_invasion')),
                _b(inv.get('perineural_invasion')),
                _b(isi.get('dcis_present')), isi.get('dcis_nuclear_grade'),
                _j(isi.get('dcis_pattern')), _b(isi.get('dcis_necrosis')), isi.get('dcis_size_mm'),
                _b(isi.get('lcis_present')),
                mar.get('invasive_margin_status'), mar.get('closest_invasive_margin_name'),
                mar.get('closest_invasive_margin_distance_mm'),
                mar.get('dcis_margin_status'), mar.get('closest_dcis_margin_distance_mm'),
                ln.get('total_examined'), ln.get('sentinel_examined'),
                ln.get('with_macrometastases'), ln.get('with_micrometastases'), ln.get('with_isolated_tumor_cells'),
                ln.get('largest_deposit_mm'), _b(ln.get('extranodal_extension')),
                stg.get('pT'), stg.get('pN'), stg.get('pM'),
                _b(stg.get('sentinel_modifier')), stg.get('overall_stage'),
                rc.get('er_status'), rc.get('er_percent'), rc.get('er_intensity'),
                rc.get('pr_status'), rc.get('pr_percent'), rc.get('pr_intensity'),
                rc.get('her2_status'), rc.get('her2_method'),
                rc.get('her2_ihc_score'), rc.get('her2_fish_ratio'), rc.get('ki67_percent'),
                _b(tx.get('post_neoadjuvant')), tx.get('treatment_effect'),
                bn.get('primary_benign_dx'), _b(bn.get('atypia_present')), bn.get('atypia_type'),
                e.get('confidence'), e.get('extraction_notes'),
                model_name, None,
            )
            cur.execute(PHASE2_INSERT, params)
            n_ok += 1

    conn.commit()
    conn.close()
    print(f"Phase 2 import: {n_ok} ok, {n_err} errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['1', '2'], required=True)
    ap.add_argument('--input', required=True, help='JSONL from VM')
    ap.add_argument('--model', default='gemini-2.5-flash')
    args = ap.parse_args()
    if args.phase == '1':
        import_phase1(Path(args.input), args.model)
    else:
        import_phase2(Path(args.input), args.model)


if __name__ == '__main__':
    main()
