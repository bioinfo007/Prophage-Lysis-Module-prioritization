"""
m03_gate1.py
============
Module 03: Gate 1 — Track-aware expressibility filtering.

Each track (endolysin/holin/spanin) has its own biologically appropriate
filter criteria. Holins are NOT penalized for TM helices — they require them.
SAR endolysins are NOT penalized for their single N-terminal TM helix.

CAI is computed from the actual nucleotide CDS sequence (fixed from original).

Input:  data/intermediate/02_lysis_modules/candidates.json
Output: data/intermediate/03_gate1/gate1_results.tsv
        data/intermediate/03_gate1/candidates_passing.json
        data/intermediate/02_lysis_modules/candidates.json (updated)
"""

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from Bio.SeqUtils.ProtParam import ProteinAnalysis

from pipeline.utils.data_model import (
    _BaseRecord, EndolysínRecord, HolinRecord, SpanínRecord,
    load_candidates, save_candidates, split_by_track,
)
from pipeline.utils.hmmer import predict_tm_helices_simple
from pipeline.utils.cai import compute_cai_if_available

log = logging.getLogger("m03_gate1")


def run(cfg: dict) -> None:
    paths  = cfg["paths"]
    g1_cfg = cfg["gate1"]

    in_dir  = Path(paths["intermediate_dir"]) / "02_lysis_modules"
    out_dir = Path(paths["intermediate_dir"]) / "03_gate1"
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path  = in_dir / "candidates.json"
    candidates = load_candidates(str(cand_path))

    log.info(f"Gate 1: evaluating {len(candidates)} candidates")

    tracks = split_by_track(candidates)
    log.info(
        f"  Endolysins: {len(tracks['endolysin'])} | "
        f"Holins: {len(tracks['holin'])} | "
        f"Spanins: {len(tracks['spanin'])}"
    )

    n_pass = n_warn = n_fail = 0

    # ── Endolysin track ────────────────────────────────────────────────────────
    for c in tracks["endolysin"]:
        _set_physicochemical(c)
        flags = _evaluate_endolysin(c, g1_cfg)
        _apply_gate1_result(c, flags, g1_cfg.get("flags_to_fail", 2))
        if c.gate1_status == "pass":   n_pass += 1
        elif c.gate1_status == "warn": n_warn += 1
        else:                          n_fail += 1

    # ── Holin track ────────────────────────────────────────────────────────────
    for c in tracks["holin"]:
        _set_physicochemical(c)
        flags = _evaluate_holin(c, g1_cfg)
        _apply_gate1_result(c, flags, g1_cfg.get("flags_to_fail_holin", 2))
        if c.gate1_status == "pass":   n_pass += 1
        elif c.gate1_status == "warn": n_warn += 1
        else:                          n_fail += 1

    # ── Spanin track ───────────────────────────────────────────────────────────
    for c in tracks["spanin"]:
        _set_physicochemical(c)
        flags = _evaluate_spanin(c, g1_cfg)
        _apply_gate1_result(c, flags, g1_cfg.get("flags_to_fail_spanin", 1))
        if c.gate1_status == "pass":   n_pass += 1
        elif c.gate1_status == "warn": n_warn += 1
        else:                          n_fail += 1

    log.info(
        f"Gate 1 results: {n_pass} pass | {n_warn} warn | {n_fail} fail"
    )

    # Flag breakdown
    all_flags = [f for c in candidates for f in c.gate1_flags]
    for flag, count in Counter(all_flags).most_common():
        log.info(f"  {flag}: {count}")

    # Save all candidates (updated with gate1_status)
    save_candidates(candidates, str(cand_path))

    # Write passing candidates to separate file for M04
    passing = [c for c in candidates if c.gate1_status in ("pass", "warn")]
    save_candidates(passing, str(out_dir / "candidates_passing.json"))

    # Write TSV report
    _write_gate1_results(candidates, out_dir / "gate1_results.tsv")

    log.info(
        f"M03 complete — {n_fail} eliminated | "
        f"{len(passing)} passed to M04"
    )


# ── Evaluators per track ──────────────────────────────────────────────────────

def _evaluate_endolysin(
    c:      EndolysínRecord,
    g1_cfg: dict,
) -> List[Tuple[str, float]]:
    """
    Endolysin-specific Gate 1 criteria.
    SAR endolysins exempt from TM helix penalty.
    """
    flags = []

    if c.mw_kda is not None and c.mw_kda > g1_cfg.get("max_mw_kda", 70.0):
        flags.append((f"MW_too_large_{c.mw_kda:.1f}kDa", c.mw_kda))

    if c.gravy is not None and c.gravy > g1_cfg.get("max_gravy_endolysin", 0.1):
        if not c.is_sar_endolysin:   # SAR endolysins are membrane-associated — exempt
            flags.append((f"hydrophobic_GRAVY_{c.gravy:.2f}", c.gravy))

    if (c.instability_index is not None and
            c.instability_index > g1_cfg.get("max_instability_index", 60.0)):
        flags.append((f"unstable_II_{c.instability_index:.1f}", c.instability_index))

    # TM helix filter — exempt SAR endolysins (they have exactly 1 N-terminal TM)
    n_tm, _ = predict_tm_helices_simple(c.sequence)
    c.n_tm_helices = n_tm
    if not c.is_sar_endolysin and n_tm >= 2:
        flags.append((f"TM_helices_{n_tm}_likely_holin", float(n_tm)))

    # Real CAI from nucleotide sequence
    cai = compute_cai_if_available(c.nucleotide_seq, fallback=0.5)
    c.cai_score = cai
    if cai < g1_cfg.get("min_cai_score", 0.55):
        flags.append((f"low_CAI_{cai:.3f}", cai))

    return flags


def _evaluate_holin(
    c:      HolinRecord,
    g1_cfg: dict,
) -> List[Tuple[str, float]]:
    """
    Holin-specific Gate 1 criteria.
    TM helices are REQUIRED — missing them is the red flag.
    No CAI or instability filter — holins are naturally divergent.
    """
    flags = []

    if c.mw_kda is not None and c.mw_kda > g1_cfg.get("max_mw_kda_holin", 25.0):
        flags.append((f"MW_too_large_{c.mw_kda:.1f}kDa", c.mw_kda))

    # Holins need TM helices — fewer than 1 is a red flag
    n_tm, tm_pos = predict_tm_helices_simple(c.sequence)
    c.n_tm_helices = n_tm
    if n_tm < 1:
        flags.append(("no_TM_helices_unlikely_holin", 0.0))
    if n_tm > g1_cfg.get("max_tm_helices_holin", 4):
        flags.append((f"too_many_TM_{n_tm}", float(n_tm)))

    # Holins must be hydrophobic
    if c.gravy is not None and c.gravy < g1_cfg.get("min_gravy_holin", -0.1):
        flags.append((f"too_hydrophilic_holin_GRAVY_{c.gravy:.2f}", c.gravy))

    return flags


def _evaluate_spanin(
    c:      SpanínRecord,
    g1_cfg: dict,
) -> List[Tuple[str, float]]:
    """
    Spanin-specific Gate 1 criteria.
    Mostly structural topology checks — spanins are kept unless clearly wrong.
    """
    flags = []

    if c.mw_kda is not None and c.mw_kda > g1_cfg.get("max_mw_kda_spanin", 35.0):
        flags.append((f"MW_too_large_{c.mw_kda:.1f}kDa", c.mw_kda))

    # i-spanin should have exactly 1 TM helix
    n_tm, _ = predict_tm_helices_simple(c.sequence)
    c.n_tm_helices = n_tm
    spanin_type = c.spanin_type or ""
    if "i_spanin" in spanin_type and n_tm != 1:
        flags.append((f"i_spanin_wrong_TM_count_{n_tm}", float(n_tm)))

    # o-spanin should have 0 TM helices
    if "o_spanin" in spanin_type and n_tm > 1:
        flags.append((f"o_spanin_too_many_TM_{n_tm}", float(n_tm)))

    return flags


# ── Shared helpers ────────────────────────────────────────────────────────────

def _set_physicochemical(c: _BaseRecord) -> None:
    """Compute and store BioPython physicochemical features."""
    try:
        seq = c.sequence.replace("*", "").replace("X", "A")  # X → A for ProtParam
        if not seq or len(seq) < 5:
            return
        pa = ProteinAnalysis(seq)
        c.mw_kda            = round(pa.molecular_weight() / 1000, 3)
        c.isoelectric_point = round(pa.isoelectric_point(), 2)
        c.gravy             = round(pa.gravy(), 4)
        c.instability_index = round(pa.instability_index(), 2)
        c.aromaticity       = round(pa.aromaticity(), 4)
        c.length_aa         = len(c.sequence)
    except Exception as e:
        log.debug(f"BioPython analysis failed for {c.candidate_id}: {e}")


def _apply_gate1_result(
    c:              _BaseRecord,
    flags:          List[Tuple[str, float]],
    flags_to_fail:  int,
) -> None:
    c.gate1_flags = [f[0] for f in flags]
    n             = len(flags)
    if n >= flags_to_fail:
        c.gate1_status      = "fail"
        c.final_status      = "eliminated"
        c.elimination_gate  = "gate1"
        c.elimination_reason= "; ".join(c.gate1_flags)
    elif n == 1:
        c.gate1_status = "warn"
    else:
        c.gate1_status = "pass"


def _write_gate1_results(
    candidates: List[_BaseRecord],
    path:       Path,
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "candidate_id", "track", "gate1_status",
            "mw_kda", "gravy", "instability_index",
            "cai_score", "length_aa", "flags"
        ])
        for c in candidates:
            writer.writerow([
                c.candidate_id,
                c.track,
                c.gate1_status or "",
                c.mw_kda or "",
                c.gravy or "",
                c.instability_index or "",
                c.cai_score or "",
                c.length_aa or "",
                "|".join(c.gate1_flags),
            ])


# ── Snakemake / standalone entry point ───────────────────────────────────────
# snakemake object check must come FIRST — when Snakemake calls this via
# script: directive, __name__ == '__main__', so snakemake takes priority.
if 'snakemake' in dir():
    from pipeline.utils.logging_config import setup_logging
    setup_logging(snakemake.config['paths'].get('log_dir', 'logs'))
    run(snakemake.config)
elif __name__ == '__main__':
    import sys, yaml
    from pipeline.utils.logging_config import setup_logging
    _cfg = yaml.safe_load(open(sys.argv[1]))
    setup_logging(_cfg['paths'].get('log_dir', 'logs'))
    run(_cfg)
