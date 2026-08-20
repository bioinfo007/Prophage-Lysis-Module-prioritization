import json
"""
m08_selection.py
================
Module 08: Module-aware MaxMin diversity selection.

Selection runs on endolysin ESM-2 embeddings (richest signal).
For each selected endolysin, its cognate holin and spanin from the same
module are co-selected automatically — you are selecting modules, not proteins.

Saturation detection: stops when marginal distance drops below
saturation_threshold × initial distance. Respects min/max bounds.

Optional pathogen coverage constraint (if pg_matching.enabled):
  Selection continues until each target pathogen has at least
  min_pathogen_coverage candidates with PG score ≥ 2.

BLAST novelty check tags selected endolysins — novel candidates highlighted.

Input:  data/intermediate/03_gate1/candidates_passing.json
        data/intermediate/02_lysis_modules/modules.json
        data/intermediate/04_embeddings/embedding_matrix.npy
Output: data/intermediate/08_selection/selection_summary.json
        data/intermediate/08_selection/diversity_curve.tsv
        data/intermediate/03_gate1/candidates_passing.json (updated)
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from pipeline.utils.data_model import (
    _BaseRecord, EndolysínRecord,
    LysisModule, load_candidates, load_modules, save_candidates, split_by_track,
)
from pipeline.utils.numba_kernels import maxmin_select, compute_mean_pairwise_distance
from pipeline.utils.pg_database import PathogenDatabase

log = logging.getLogger("m08_selection")


def run(cfg: dict) -> None:
    paths   = cfg["paths"]
    sel_cfg = cfg["selection"]
    pg_cfg  = cfg.get("pg_matching", {})

    in_dir  = Path(paths["intermediate_dir"]) / "03_gate1"
    emb_dir = Path(paths["intermediate_dir"]) / "04_embeddings"
    mod_dir = Path(paths["intermediate_dir"]) / "02_lysis_modules"
    out_dir = Path(paths["intermediate_dir"]) / "08_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path  = in_dir / "candidates_passing.json"
    candidates = load_candidates(str(cand_path))
    modules    = load_modules(str(mod_dir / "modules.json"))

    tracks = split_by_track(candidates)
    cand_by_id = {c.candidate_id: c for c in candidates}

    # Only representatives go to selection
    reps_endo = [
        c for c in tracks["endolysin"]
        if c.gate3_status == "representative"
    ]

    strategy  = sel_cfg.get("selection_strategy", "saturation")
    min_n     = sel_cfg.get("min_expression_n", 5)
    max_n     = sel_cfg.get("max_expression_n", 200)
    threshold = sel_cfg.get("saturation_threshold", 0.30)

    log.info(
        f"Selection: {len(reps_endo)} endolysin representatives | "
        f"strategy: {strategy}"
    )

    if not reps_endo:
        log.warning(
            "No endolysin representatives found in this dataset. "
            "Possible reasons: (1) these phages use non-canonical lysis mechanisms, "
            "(2) endolysins are annotated as hypothetical proteins with no Pfam hits, "
            "(3) endolysin domains are absent from the Pfam version used. "
            "Pipeline will continue with holin-only output."
        )
        # Write empty selection outputs so downstream rules can proceed
        out_dir = Path(paths["intermediate_dir"]) / "08_selection"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "strategy": strategy,
            "n_selected_endolysins": 0,
            "n_total_candidates": len(candidates),
            "n_priority": 0,
            "n_reserve": 0,
            "note": "No endolysins detected — check hmmer_hits.tsv and phage biology",
            "selected_ids": [],
        }
        (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2))
        # Write empty diversity curve
        import csv
        with open(out_dir / "diversity_curve.tsv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step","candidate_id","marginal_distance","pct_of_initial","selected"],
                delimiter="\t"
            )
            writer.writeheader()
        # Mark all holins as reserve since no endolysin module exists
        for c in candidates:
            if c.final_status is None:
                c.final_status = "reserve"
        save_candidates(candidates, str(cand_path))
        log.info("M08 complete — 0 endolysins | holins available in reserve list")
        return

    # Load embedding matrix
    full_matrix = np.load(emb_dir / "embedding_matrix.npy")
    full_index  = json.loads((emb_dir / "embedding_index.json").read_text())
    id_to_row   = {cid: i for i, cid in enumerate(full_index)}

    # Sub-matrix for endolysin representatives
    endo_ids   = [c.candidate_id for c in reps_endo if c.candidate_id in id_to_row]
    endo_rows  = [full_matrix[id_to_row[cid]] for cid in endo_ids]

    if not endo_rows:
        raise ValueError("No embedding vectors found for endolysin representatives.")

    endo_matrix = np.vstack(endo_rows).astype(np.float32)

    # ── Run MaxMin selection ──────────────────────────────────────────────────
    if len(reps_endo) <= min_n:
        selected_endo = reps_endo
        order_log     = _trivial_order_log(reps_endo)
        strategy_used = "all_below_minimum"

    elif strategy == "saturation":
        selected_endo, order_log = _maxmin_saturation(
            reps_endo, endo_matrix, endo_ids,
            threshold=threshold, min_n=min_n, max_n=max_n,
        )
        strategy_used = "maxmin_saturation"
    else:
        selected_endo, order_log = _maxmin_fixed(
            reps_endo, endo_matrix, endo_ids, n_select=min(max_n, len(reps_endo))
        )
        strategy_used = "maxmin_fixed"

    # ── Optional pathogen coverage extension ──────────────────────────────────
    if pg_cfg.get("enabled", False) and pg_cfg.get("enforce_coverage", False):
        db_path = cfg.get("targets", {}).get("pathogen_db", "targets/pathogen_db.yaml")
        db      = PathogenDatabase(db_path)
        min_cov = sel_cfg.get("min_pathogen_coverage", 2)

        selected_endo = _extend_for_pathogen_coverage(
            selected_endo, reps_endo, endo_matrix, endo_ids,
            db=db, min_coverage=min_cov, max_n=max_n,
            order_log=order_log,
        )

    # ── Co-select module partners ─────────────────────────────────────────────
    module_by_endo = _build_module_index(modules)
    selected_endo_ids = {c.candidate_id for c in selected_endo}

    # Find all co-selected holins and spanins
    coselected_partners: List[str] = []
    for endo in selected_endo:
        mod = module_by_endo.get(endo.candidate_id)
        if mod is None:
            continue
        for partner_id in [mod.holin_id, mod.ispanin_id, mod.ospanin_id, mod.uspanin_id]:
            if partner_id and partner_id in cand_by_id:
                coselected_partners.append(partner_id)

    # ── Assign final status ───────────────────────────────────────────────────
    reserve_endo_ids = {
        c.candidate_id for c in reps_endo
        if c.candidate_id not in selected_endo_ids
    }

    for c in candidates:
        if c.final_status == "eliminated":
            continue   # gate1/3 eliminations already set

        if c.candidate_id in selected_endo_ids:
            c.final_status       = "priority"
            c.selection_strategy = strategy_used
        elif c.candidate_id in coselected_partners:
            c.final_status       = "priority"   # module partner — co-selected
            c.selection_strategy = "module_coselection"
        elif c.candidate_id in reserve_endo_ids:
            c.final_status       = "reserve"
            c.selection_strategy = strategy_used
        elif c.gate3_status == "collapsed":
            c.final_status       = "eliminated"
            c.elimination_gate   = "gate3"
            c.elimination_reason = f"redundant — represented by {c.similar_to}"
        else:
            # Holins/spanins not linked to a selected module → reserve
            if c.final_status is None:
                c.final_status = "reserve"

    for rank, c in enumerate(selected_endo, start=1):
        c.final_rank     = rank
        c.diversity_rank = rank

    # ── BLAST novelty check ───────────────────────────────────────────────────
    if sel_cfg.get("run_blast_novelty", True):
        _run_blast_novelty(selected_endo, cfg["api"], sel_cfg)
    else:
        for c in selected_endo:
            c.novelty_flag  = "not_checked"
            c.closest_known = ""

    # ── Diversity metrics ─────────────────────────────────────────────────────
    if len(selected_endo) > 1:
        sel_rows = [full_matrix[id_to_row[c.candidate_id]]
                    for c in selected_endo if c.candidate_id in id_to_row]
        if sel_rows:
            mean_dist = compute_mean_pairwise_distance(
                np.vstack(sel_rows), list(range(len(sel_rows)))
            )
            log.info(f"Selected set mean pairwise distance: {mean_dist:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_candidates(candidates, str(cand_path))
    _write_outputs(out_dir, selected_endo, candidates, order_log, strategy_used, sel_cfg)

    priority_count = sum(1 for c in candidates if c.final_status == "priority")
    reserve_count  = sum(1 for c in candidates if c.final_status == "reserve")
    novel_count    = sum(1 for c in selected_endo if c.novelty_flag == "novel")

    log.info(
        f"M08 complete — {len(selected_endo)} endolysins selected | "
        f"{len(coselected_partners)} module partners co-selected | "
        f"{priority_count} total priority | {reserve_count} reserve"
    )
    if sel_cfg.get("run_blast_novelty", True):
        log.info(f"  Novel candidates: {novel_count}/{len(selected_endo)}")


# ── MaxMin implementations ────────────────────────────────────────────────────

def _maxmin_saturation(
    candidates: List[_BaseRecord],
    matrix:     np.ndarray,
    index:      List[str],
    threshold:  float,
    min_n:      int,
    max_n:      int,
) -> Tuple[List[_BaseRecord], List[dict]]:
    """
    Saturation-based MaxMin using Numba JIT kernel.
    Stops when marginal distance < threshold × initial distance.
    """
    id_to_idx = {cid: i for i, cid in enumerate(index)}
    cand_map  = {c.candidate_id: c for c in candidates}

    # Run MaxMin kernel
    sel_indices, marginal_dists = maxmin_select(matrix, max_n)

    initial_dist    = marginal_dists[1] if len(marginal_dists) > 1 else 1.0
    if initial_dist <= 0:
        initial_dist = 1e-6

    selected: List[_BaseRecord] = []
    order_log: List[dict]        = []
    saturation_reached           = False

    for step, (idx, dist) in enumerate(zip(sel_indices, marginal_dists)):
        cid = index[idx]
        c   = cand_map.get(cid)
        if c is None:
            continue

        pct = dist / initial_dist * 100 if step > 0 else 100.0

        # Check saturation after first selection
        is_saturated = (
            step > 0 and
            pct < threshold * 100 and
            len(selected) >= min_n
        )

        if is_saturated and not saturation_reached:
            saturation_reached = True
            log.info(
                f"Saturation at step {step + 1}: "
                f"marginal dist {dist:.4f} = {pct:.1f}% of initial {initial_dist:.4f}"
            )

        # Add to selected before checking (ensures we don't miss the seed)
        selected.append(c)
        c.min_dist_at_selection = round(dist, 4)

        order_log.append({
            "step":              step + 1,
            "candidate_id":      cid,
            "marginal_distance": round(dist, 4),
            "pct_of_initial":    round(pct, 1),
            "selected":          True,
        })

        if is_saturated:
            break

    if not saturation_reached:
        log.info(
            f"No saturation detected — selected all {len(selected)} representatives "
            f"(flat diversity curve, likely few closely related phages)"
        )

    return selected, order_log


def _maxmin_fixed(
    candidates: List[_BaseRecord],
    matrix:     np.ndarray,
    index:      List[str],
    n_select:   int,
) -> Tuple[List[_BaseRecord], List[dict]]:
    """Fixed-N MaxMin — legacy mode."""
    cand_map = {c.candidate_id: c for c in candidates}

    sel_indices, marginal_dists = maxmin_select(matrix, n_select)

    initial_dist = marginal_dists[1] if len(marginal_dists) > 1 else 1.0

    selected  = []
    order_log = []
    for step, (idx, dist) in enumerate(zip(sel_indices, marginal_dists)):
        cid = index[idx]
        c   = cand_map.get(cid)
        if c is None:
            continue
        c.min_dist_at_selection = round(dist, 4)
        selected.append(c)
        order_log.append({
            "step": step + 1,
            "candidate_id": cid,
            "marginal_distance": round(dist, 4),
            "pct_of_initial": round(dist / (initial_dist or 1e-6) * 100, 1),
            "selected": True,
        })

    return selected, order_log


# ── Pathogen coverage extension ───────────────────────────────────────────────

def _extend_for_pathogen_coverage(
    selected:       List[_BaseRecord],
    all_reps:       List[_BaseRecord],
    matrix:         np.ndarray,
    index:          List[str],
    db:             PathogenDatabase,
    min_coverage:   int,
    max_n:          int,
    order_log:      List[dict],
) -> List[_BaseRecord]:
    """
    After saturation, add candidates from reserve to fill pathogen coverage gaps.
    Adds one candidate at a time targeting the least-covered pathogen.
    """
    from pipeline.utils.data_model import EndolysínRecord

    selected_ids = {c.candidate_id for c in selected}
    reserve = [c for c in all_reps if c.candidate_id not in selected_ids]

    while len(selected) < max_n and reserve:
        # Compute current coverage
        sel_pg = [
            c.get_pg_compatibility()
            for c in selected
            if isinstance(c, EndolysínRecord) and c.pg_compatibility
        ]
        uncovered = db.uncovered_pathogens(sel_pg, min_coverage)

        if not uncovered:
            break   # all pathogens covered

        # Pick next candidate that best covers the most under-served pathogen
        target_pathogen = uncovered[0]
        best_cand: Optional[_BaseRecord] = None
        best_score = -1

        for c in reserve:
            if isinstance(c, EndolysínRecord) and c.pg_compatibility:
                scores = c.get_pg_compatibility()
                s      = scores.get(target_pathogen, 0)
                if s > best_score:
                    best_score = s
                    best_cand  = c

        if best_cand is None or best_score == 0:
            break   # no candidate can help

        selected.append(best_cand)
        reserve.remove(best_cand)

        order_log.append({
            "step":              len(selected),
            "candidate_id":      best_cand.candidate_id,
            "marginal_distance": 0.0,
            "pct_of_initial":    0.0,
            "selected":          True,
            "reason":            f"pathogen_coverage_{target_pathogen}",
        })

        log.info(
            f"  Coverage extension: added {best_cand.candidate_id} "
            f"for {target_pathogen} (score={best_score})"
        )

    return selected


# ── Module index ──────────────────────────────────────────────────────────────

def _build_module_index(modules: List[LysisModule]) -> Dict[str, LysisModule]:
    """Map endolysin_id → LysisModule for fast lookup."""
    index: Dict[str, LysisModule] = {}
    for mod in modules:
        if mod.endolysin_id:
            index[mod.endolysin_id] = mod
    return index


# ── BLAST novelty check ───────────────────────────────────────────────────────

def _run_blast_novelty(
    candidates: List[_BaseRecord],
    api_cfg:    dict,
    sel_cfg:    dict,
) -> None:
    """
    BLAST selected endolysins against SwissProt.
    Tags each candidate as "novel" or "known_homolog".
    Non-blocking: failures are logged but don't stop selection.
    """
    import requests

    blast_url    = api_cfg.get(
        "ncbi_blast_url", "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
    )
    evalue       = sel_cfg.get("blast_evalue", 1e-10)
    id_threshold = sel_cfg.get("blast_identity_threshold", 90.0)

    log.info(f"BLAST novelty check: {len(candidates)} candidates")

    for c in candidates:
        if c.novelty_flag:
            continue
        try:
            result = _blast_swissprot(c.sequence, blast_url, evalue, id_threshold)
            c.novelty_flag  = result["novelty_flag"]
            c.closest_known = result["closest_known"]
        except Exception as e:
            log.warning(f"  {c.candidate_id}: BLAST failed — {e}")
            c.novelty_flag  = "blast_failed"
            c.closest_known = str(e)


def _blast_swissprot(
    sequence:     str,
    blast_url:    str,
    evalue:       float,
    id_threshold: float,
) -> dict:
    """
    Submit sequence to NCBI BLAST against SwissProt.
    Returns novelty_flag and closest_known hit.
    """
    import requests

    # Submit
    params = {
        "CMD":      "Put",
        "PROGRAM":  "blastp",
        "DATABASE": "swissprot",
        "QUERY":    sequence,
        "FORMAT_TYPE": "JSON2",
        "EXPECT":   str(evalue),
        "HITLIST_SIZE": "5",
    }
    r = requests.post(blast_url, data=params, timeout=60)
    r.raise_for_status()
    rid = r.text.split("RID = ")[1].split()[0] if "RID = " in r.text else None

    if not rid:
        raise ValueError("Could not parse BLAST RID")

    # Poll for results.
    # NCBI BLAST: HTTP 200 is returned for BOTH waiting and ready states.
    # Must check Status= in response body, NOT status_code.
    for attempt in range(30):
        time.sleep(10)

        # Status check — returns HTML with "Status=WAITING" or "Status=READY"
        status_resp = requests.get(
            blast_url,
            params={"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"},
            timeout=60,
        )
        body = status_resp.text

        if "Status=FAILED" in body:
            raise RuntimeError(f"BLAST job {rid} failed on NCBI server")
        if "Status=UNKNOWN" in body:
            raise RuntimeError(f"BLAST RID {rid} expired or unknown")
        if "Status=READY" not in body:
            log.debug(f"  BLAST {rid}: waiting (attempt {attempt + 1}/30)")
            continue

        # Ready — fetch JSON results
        result_resp = requests.get(
            blast_url,
            params={"CMD": "Get", "RID": rid, "FORMAT_TYPE": "JSON2"},
            timeout=120,
        )
        result_resp.raise_for_status()

        data = result_resp.json()
        hits = (
            data.get("BlastOutput2", [{}])[0]
                .get("report", {})
                .get("results", {})
                .get("search", {})
                .get("hits", [])
        )
        if not hits:
            return {"novelty_flag": "novel", "closest_known": "no_hit"}

        top    = hits[0]
        hsp    = top["hsps"][0]
        pct_id = hsp.get("identity", 0) / max(hsp.get("align_len", 1), 1) * 100
        title  = top.get("description", [{}])[0].get("title", "unknown")
        flag   = "known_homolog" if pct_id >= id_threshold else "novel"
        return {"novelty_flag": flag, "closest_known": f"{title} [{pct_id:.1f}%]"}

    return {"novelty_flag": "blast_timeout", "closest_known": ""}


# ── Writers ───────────────────────────────────────────────────────────────────

def _trivial_order_log(candidates: List[_BaseRecord]) -> List[dict]:
    return [
        {
            "step": i + 1,
            "candidate_id": c.candidate_id,
            "marginal_distance": 0.0,
            "pct_of_initial": 100.0,
            "selected": True,
        }
        for i, c in enumerate(candidates)
    ]


def _write_outputs(
    out_dir:    Path,
    selected:   List[_BaseRecord],
    candidates: List[_BaseRecord],
    order_log:  List[dict],
    strategy:   str,
    sel_cfg:    dict,
) -> None:
    summary = {
        "strategy":           strategy,
        "n_selected_endolysins": len(selected),
        "n_total_candidates": len(candidates),
        "n_priority":         sum(1 for c in candidates if c.final_status == "priority"),
        "n_reserve":          sum(1 for c in candidates if c.final_status == "reserve"),
        "n_eliminated_gate1": sum(1 for c in candidates if c.elimination_gate == "gate1"),
        "n_eliminated_gate3": sum(1 for c in candidates if c.elimination_gate == "gate3"),
        "novel_candidates":   sum(1 for c in selected if c.novelty_flag == "novel"),
        "selected_ids":       [c.candidate_id for c in selected],
    }
    (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2))

    if order_log:
        with open(out_dir / "diversity_curve.tsv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=order_log[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(order_log)


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
