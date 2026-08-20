"""
m02_lysis_modules.py
====================
Module 02: Identify lysis module genes and group them into modules.

Three parallel identification tracks:
  - Endolysin track: catalytic PG-degrading enzymes (including SAR endolysins)
  - Holin track:     inner membrane permeabilizers
  - Spanin track:    outer membrane disruptors

Module linkage:
  After identifying all proteins, genomically proximal holin + endolysin pairs
  (within 10 ORFs in the same prophage) are linked into LysisModule objects.
  Spanins found between them upgrade the module to "complete".

Input:  data/intermediate/01_pharokka/all_proteins.faa
        data/intermediate/01_pharokka/all_nucleotides.ffn
        data/intermediate/01_pharokka/annotation_table.tsv
        data/intermediate/01_pharokka/nucleotide_lookup.json
Output: data/intermediate/02_lysis_modules/candidates.json   (all three tracks)
        data/intermediate/02_lysis_modules/modules.json       (linked modules)
        data/intermediate/02_lysis_modules/candidates.faa     (FASTA for embedding)
        data/intermediate/02_lysis_modules/hmmer_hits.tsv
"""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

from Bio import SeqIO

from pipeline.utils.data_model import (
    EndolysínRecord, HolinRecord, SpanínRecord,
    LysisModule, save_candidates, save_modules, _BaseRecord,
)
from pipeline.utils.hmmer import (
    run_hmmscan, parse_domtblout,
    has_endolysin_catalytic, has_holin_domain, has_spanin_domain,
    catalytic_domain_type, classify_protein,
    is_sar_endolysin, has_cbd, has_domain,
    predict_tm_helices_simple, predict_signal_peptide_heuristic,
    ENDOLYSIN_CATALYTIC_DOMAINS, HOLIN_DOMAINS, SPANIN_DOMAINS, CBD_DOMAINS,
    get_accessions,
)

log = logging.getLogger("m02_lysis_modules")

# Categories in Pharokka output that indicate lysis-related function
_LYSIS_KEYWORDS = {
    "endolysin", "lysin", "lysozyme", "muramidase",
    "amidase", "peptidoglycan", "murein", "autolysin",
    "transglycosylase", "glucosaminidase", "bacteriolytic",
    "n-acetylmuramidase", "lytic enzyme", "cell wall hydrolase",
}

# Keywords that EXCLUDE a protein from lysis track even if lysis keyword matches
_LYSIS_EXCLUSION_KEYWORDS = {
    "tail", "capsid", "terminase", "portal", "baseplate",
    "fiber", "spike", "tape", "neck", "head", "structural",
    "coat", "assembly", "chaperone", "scaffold", "decoration",
    "integrase", "repressor", "transposase", "replication",
    "helicase", "polymerase", "recombinase", "resolvase",
}

_HOLIN_KEYWORDS = {
    "holin", "hole-forming", "phage holin", "membrane hole",
    "class ii holin", "class i holin",
}

_SPANIN_KEYWORDS = {
    "spanin", "rz", "rz1", "i-spanin", "o-spanin", "outer membrane",
    "lipoprotein spanin",
}

# Max ORF distance to link holin and endolysin into a module
_MODULE_LINK_ORF_WINDOW = 10


def run(cfg: dict) -> None:
    paths   = cfg["paths"]
    ec_cfg  = cfg["candidate_extraction"]

    in_dir  = Path(paths["intermediate_dir"]) / "01_pharokka"
    out_dir = Path(paths["intermediate_dir"]) / "02_lysis_modules"
    out_dir.mkdir(parents=True, exist_ok=True)

    faa_path   = in_dir / "all_proteins.faa"
    annot_path = in_dir / "annotation_table.tsv"
    nuc_path   = in_dir / "nucleotide_lookup.json"

    if not faa_path.exists():
        raise FileNotFoundError(f"Protein FASTA not found: {faa_path}")

    # Load inputs
    all_proteins  = _load_proteins(faa_path)
    annotations   = _load_annotations(annot_path)
    nuc_lookup    = json.loads(nuc_path.read_text()) if nuc_path.exists() else {}

    log.info(f"Loaded {len(all_proteins)} proteins")

    # Run HMMER on all proteins (single pass — cheaper than three passes)
    domtblout   = run_hmmscan(
        fasta_path = str(faa_path),
        pfam_hmm   = paths["pfam_hmm"],
        output_dir = str(out_dir),
        threads    = ec_cfg.get("hmmer_threads", 4),
        evalue     = ec_cfg.get("hmmer_evalue_threshold", 1e-5),
        prefix     = "lysis_modules",
    )
    domain_hits = parse_domtblout(
        domtblout,
        evalue_threshold=ec_cfg.get("hmmer_evalue_threshold", 1e-5),
    )
    _write_hmmer_summary(domain_hits, out_dir / "hmmer_hits.tsv")

    # Length filter bounds
    min_len_endo = ec_cfg.get("min_length_aa_endolysin", 80)
    max_len_endo = ec_cfg.get("max_length_aa_endolysin", 700)
    min_len_holi = ec_cfg.get("min_length_aa_holin",     40)
    max_len_holi = ec_cfg.get("max_length_aa_holin",     300)
    min_len_span = ec_cfg.get("min_length_aa_spanin",    40)
    max_len_span = ec_cfg.get("max_length_aa_spanin",    400)

    # ── Three-track identification ─────────────────────────────────────────────
    endolysins: List[EndolysínRecord] = []
    holins:     List[HolinRecord]     = []
    spanins:    List[SpanínRecord]    = []

    skipped = defaultdict(int)

    for pid, record in all_proteins.items():
        seq     = str(record.seq).replace("*", "")
        seq_len = len(seq)
        annot   = annotations.get(pid, {})
        hits    = domain_hits.get(pid, [])
        func    = annot.get("function", "hypothetical protein").lower()
        nuc_seq = nuc_lookup.get(pid, "")

        parts      = pid.split("__", 1)
        genome_id  = parts[0] if len(parts) == 2 else "unknown"
        locus_tag  = parts[1] if len(parts) == 2 else pid

        n_tm, tm_pos = predict_tm_helices_simple(seq)
        has_sig_pep  = predict_signal_peptide_heuristic(seq)
        classified   = classify_protein(hits, n_tm, tm_pos, seq)

        # ── Endolysin track ────────────────────────────────────────────
        is_endo_by_domain    = has_endolysin_catalytic(hits)
        is_endo_by_keyword   = (
            _keyword_match(func, _LYSIS_KEYWORDS) and
            not _keyword_match(func, _LYSIS_EXCLUSION_KEYWORDS)
        )
        is_sar               = classified == "sar_endolysin"

        if is_endo_by_domain or is_endo_by_keyword or is_sar:
            if not (min_len_endo <= seq_len <= max_len_endo):
                skipped["endolysin_length"] += 1
            else:
                acc_set = get_accessions(hits)
                reason  = []
                if is_endo_by_domain:  reason.append("hmmer_domain")
                if is_endo_by_keyword: reason.append(f"pharokka_keyword:{func[:40]}")
                if is_sar:             reason.append("sar_endolysin_topology")

                c = EndolysínRecord(
                    candidate_id     = pid,
                    genome_id        = genome_id,
                    protein_id       = locus_tag,
                    sequence         = seq,
                    nucleotide_seq   = nuc_seq,
                    track            = "endolysin",
                    source_organism  = annot.get("organism", genome_id),
                    cds_start        = _safe_int(annot.get("start")),
                    cds_end          = _safe_int(annot.get("end")),
                    cds_strand       = annot.get("strand"),
                    pharokka_function= annot.get("function", "hypothetical protein"),
                    pharokka_category= annot.get("category", ""),
                    pfam_domains     = [h["accession"] for h in hits],
                    pfam_descriptions= [h["description"] for h in hits],
                    pfam_evalues     = [h["evalue"] for h in hits],
                    has_chap         = has_domain(hits, {"PF04851"}),
                    has_amidase      = has_domain(hits, {"PF01520", "PF13743"}),
                    has_lysozyme     = has_domain(hits, {"PF00959"}),
                    has_glucosaminidase = has_domain(hits, {"PF13529"}),
                    has_transglycosylase= has_domain(hits, {"PF03237"}),
                    has_cbd          = has_cbd(hits),
                    is_sar_endolysin = is_sar,
                    initial_class    = "sar_endolysin" if is_sar else "endolysin",
                    inclusion_reason = "; ".join(reason),
                    length_aa        = seq_len,
                )
                endolysins.append(c)
                continue   # don't double-classify

        # ── Holin track ───────────────────────────────────────────────
        is_holin_by_domain  = has_holin_domain(hits)
        is_holin_by_keyword = _keyword_match(func, _HOLIN_KEYWORDS)

        # Topology-based holin detection — strict criteria to avoid false positives:
        # Must be small (≤ 200 aa), hydrophobic (GRAVY > 0.1), 1-4 TM helices,
        # no other known functional annotation, and no non-holin domain hits
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        try:
            _pa    = ProteinAnalysis(seq.replace("*","").replace("X","A"))
            _gravy = _pa.gravy()
        except Exception:
            _gravy = 0.0

        _has_other_function = (
            hits and all(h["domain_type"] == "other" for h in hits) and
            any(kw in func for kw in _LYSIS_EXCLUSION_KEYWORDS)
        )
        is_holin_by_topology = (
            not is_sar and
            1 <= n_tm <= 4 and
            seq_len <= 200 and          # strict size limit
            _gravy > 0.1 and            # must be hydrophobic
            not _has_other_function and  # exclude annotated non-holins
            not hits                     # no domain hits at all (unknown protein)
        )

        if is_holin_by_domain or is_holin_by_keyword or is_holin_by_topology:
            if not (min_len_holi <= seq_len <= max_len_holi):
                skipped["holin_length"] += 1
            else:
                reason = []
                if is_holin_by_domain:   reason.append("hmmer_domain")
                if is_holin_by_keyword:  reason.append(f"pharokka_keyword:{func[:40]}")
                if is_holin_by_topology: reason.append(f"tm_topology:{n_tm}_TM")

                h = HolinRecord(
                    candidate_id     = pid,
                    genome_id        = genome_id,
                    protein_id       = locus_tag,
                    sequence         = seq,
                    nucleotide_seq   = nuc_seq,
                    track            = "holin",
                    source_organism  = annot.get("organism", genome_id),
                    cds_start        = _safe_int(annot.get("start")),
                    cds_end          = _safe_int(annot.get("end")),
                    cds_strand       = annot.get("strand"),
                    pharokka_function= annot.get("function", "hypothetical protein"),
                    pharokka_category= annot.get("category", ""),
                    pfam_domains     = [h2["accession"] for h2 in hits],
                    pfam_descriptions= [h2["description"] for h2 in hits],
                    pfam_evalues     = [h2["evalue"] for h2 in hits],
                    inclusion_reason = "; ".join(reason),
                    n_tm_helices     = n_tm,
                    has_signal_peptide= has_sig_pep,
                    length_aa        = seq_len,
                )
                holins.append(h)
                continue

        # ── Spanin track ──────────────────────────────────────────────
        is_spanin_by_domain   = has_spanin_domain(hits)
        is_spanin_by_keyword  = _keyword_match(func, _SPANIN_KEYWORDS)

        # i-spanin topology: 1 TM helix, no lipoprotein signal
        is_i_spanin_topology = (
            n_tm == 1 and
            not has_sig_pep and
            min_len_span <= seq_len <= max_len_span
        )
        # o-spanin topology: lipoprotein signal (Cys after cleavage), no TM
        is_o_spanin_topology = (
            has_sig_pep and
            n_tm == 0 and
            seq_len >= 1 and seq[0] == "M" and
            min_len_span <= seq_len <= max_len_span
        )

        if is_spanin_by_domain or is_spanin_by_keyword:
            if not (min_len_span <= seq_len <= max_len_span):
                skipped["spanin_length"] += 1
            else:
                spanin_type = _infer_spanin_type(hits, func, n_tm, has_sig_pep)
                reason = []
                if is_spanin_by_domain:  reason.append("hmmer_domain")
                if is_spanin_by_keyword: reason.append(f"pharokka_keyword:{func[:40]}")

                s = SpanínRecord(
                    candidate_id     = pid,
                    genome_id        = genome_id,
                    protein_id       = locus_tag,
                    sequence         = seq,
                    nucleotide_seq   = nuc_seq,
                    track            = "spanin",
                    source_organism  = annot.get("organism", genome_id),
                    cds_start        = _safe_int(annot.get("start")),
                    cds_end          = _safe_int(annot.get("end")),
                    cds_strand       = annot.get("strand"),
                    pharokka_function= annot.get("function", "hypothetical protein"),
                    pharokka_category= annot.get("category", ""),
                    pfam_domains     = [h2["accession"] for h2 in hits],
                    pfam_descriptions= [h2["description"] for h2 in hits],
                    pfam_evalues     = [h2["evalue"] for h2 in hits],
                    inclusion_reason = "; ".join(reason),
                    spanin_type      = spanin_type,
                    has_lipoprotein_signal = has_sig_pep,
                    n_tm_helices     = n_tm,
                    length_aa        = seq_len,
                )
                spanins.append(s)

    log.info(
        f"Identified: {len(endolysins)} endolysins | "
        f"{len(holins)} holins | {len(spanins)} spanins"
    )
    log.info(f"Skipped by length: {dict(skipped)}")

    # ── Module linkage ─────────────────────────────────────────────────────────
    all_candidates: List[_BaseRecord] = endolysins + holins + spanins
    modules = _link_modules(endolysins, holins, spanins, annotations)

    # Annotate module_id and module_complete on each candidate
    _annotate_module_fields(all_candidates, modules)

    log.info(
        f"Module linking: {len(modules)} modules | "
        f"{sum(1 for m in modules if m.completeness == 'complete')} complete"
    )

    # ── Save outputs ───────────────────────────────────────────────────────────
    cand_path = out_dir / "candidates.json"
    save_candidates(all_candidates, str(cand_path))

    mod_path = out_dir / "modules.json"
    save_modules(modules, str(mod_path))

    # Combined FASTA for ESM-2 embedding (all tracks together)
    faa_out = out_dir / "candidates.faa"
    with open(faa_out, "w") as f:
        for c in all_candidates:
            f.write(f">{c.candidate_id} track={c.track}\n{c.sequence}\n")

    log.info(f"M02 complete — {len(all_candidates)} candidates saved")
    log.info(f"  Outputs: {cand_path}, {mod_path}, {faa_out}")


# ── Module linking ────────────────────────────────────────────────────────────

def _link_modules(
    endolysins: List[EndolysínRecord],
    holins:     List[HolinRecord],
    spanins:    List[SpanínRecord],
    annotations: Dict[str, Dict],
) -> List[LysisModule]:
    """
    Link holins and endolysins from the same genome that are within
    _MODULE_LINK_ORF_WINDOW ORFs of each other.

    Algorithm:
      - Build sorted ORF position lists per genome
      - For each endolysin, find holins within window
      - Nearest holin becomes the module partner
      - Check for spanins between holin and endolysin positions
    """
    # Group by genome
    endo_by_genome: Dict[str, List[EndolysínRecord]] = defaultdict(list)
    holi_by_genome: Dict[str, List[HolinRecord]]     = defaultdict(list)
    span_by_genome: Dict[str, List[SpanínRecord]]     = defaultdict(list)

    for e in endolysins:
        endo_by_genome[e.genome_id].append(e)
    for h in holins:
        holi_by_genome[h.genome_id].append(h)
    for s in spanins:
        span_by_genome[s.genome_id].append(s)

    # Build ORF order index per genome: candidate_id → orf_rank
    orf_order: Dict[str, Dict[str, int]] = {}
    for genome_id in set(
        list(endo_by_genome) + list(holi_by_genome) + list(span_by_genome)
    ):
        # Sort all candidates in genome by genomic start position
        all_in_genome = (
            endo_by_genome.get(genome_id, []) +
            holi_by_genome.get(genome_id, []) +
            span_by_genome.get(genome_id, [])
        )
        sorted_genome = sorted(
            all_in_genome,
            key=lambda c: c.cds_start or 0
        )
        orf_order[genome_id] = {
            c.candidate_id: i for i, c in enumerate(sorted_genome)
        }

    modules: List[LysisModule] = []
    used_holins:   Set[str] = set()
    used_ispanins: Set[str] = set()
    used_ospanins: Set[str] = set()
    mod_counter = 0

    for genome_id, endos in endo_by_genome.items():
        order = orf_order.get(genome_id, {})
        holins_here  = holi_by_genome.get(genome_id, [])
        spanins_here = span_by_genome.get(genome_id, [])

        for endo in endos:
            endo_rank = order.get(endo.candidate_id, -1)
            if endo_rank < 0:
                continue

            # Find nearest holin within window
            best_holin:  Optional[HolinRecord]  = None
            best_holin_dist = _MODULE_LINK_ORF_WINDOW + 1

            for hol in holins_here:
                hol_rank = order.get(hol.candidate_id, -1)
                if hol_rank < 0:
                    continue
                dist = abs(endo_rank - hol_rank)
                if dist < best_holin_dist:
                    best_holin_dist = dist
                    best_holin      = hol

            if best_holin is None:
                # No holin found — still create a partial module for the endolysin
                mod_id = f"{genome_id}__mod{mod_counter:04d}"
                mod_counter += 1
                mod = LysisModule(
                    module_id    = mod_id,
                    genome_id    = genome_id,
                    prophage_id  = genome_id,
                    endolysin_id = endo.candidate_id,
                    completeness = "partial",
                    genomic_span_bp = 0,
                )
                modules.append(mod)
                continue

            used_holins.add(best_holin.candidate_id)

            # Determine ORF range between holin and endolysin
            lo_rank = min(endo_rank, order.get(best_holin.candidate_id, endo_rank))
            hi_rank = max(endo_rank, order.get(best_holin.candidate_id, endo_rank))

            # Find spanins within that range
            i_spanin: Optional[SpanínRecord] = None
            o_spanin: Optional[SpanínRecord] = None
            u_spanin: Optional[SpanínRecord] = None

            for span in spanins_here:
                span_rank = order.get(span.candidate_id, -1)
                if not (lo_rank <= span_rank <= hi_rank):
                    continue
                st = span.spanin_type or ""
                if "i_spanin" in st and i_spanin is None:
                    i_spanin = span
                    used_ispanins.add(span.candidate_id)
                elif "o_spanin" in st and o_spanin is None:
                    o_spanin = span
                    used_ospanins.add(span.candidate_id)
                elif "u_spanin" in st and u_spanin is None:
                    u_spanin = span

            has_spanin = (i_spanin or o_spanin or u_spanin) is not None
            completeness = "complete" if has_spanin else "partial"

            # Genomic span
            positions = [
                endo.cds_start or 0, endo.cds_end or 0,
                best_holin.cds_start or 0, best_holin.cds_end or 0,
            ]
            for sp in [i_spanin, o_spanin, u_spanin]:
                if sp:
                    positions += [sp.cds_start or 0, sp.cds_end or 0]
            span_bp = max(positions) - min(positions) if positions else 0

            mod_id = f"{genome_id}__mod{mod_counter:04d}"
            mod_counter += 1

            mod = LysisModule(
                module_id    = mod_id,
                genome_id    = genome_id,
                prophage_id  = genome_id,
                endolysin_id = endo.candidate_id,
                holin_id     = best_holin.candidate_id,
                ispanin_id   = i_spanin.candidate_id if i_spanin else None,
                ospanin_id   = o_spanin.candidate_id if o_spanin else None,
                uspanin_id   = u_spanin.candidate_id if u_spanin else None,
                completeness = completeness,
                genomic_span_bp = span_bp,
            )
            modules.append(mod)

    return modules


def _annotate_module_fields(
    candidates: List[_BaseRecord],
    modules:    List[LysisModule],
) -> None:
    """Back-annotate module_id and module_complete onto each candidate."""
    for mod in modules:
        is_complete = mod.completeness == "complete"
        for cid in mod.component_ids():
            # Find candidate
            for c in candidates:
                if c.candidate_id == cid:
                    c.module_id       = mod.module_id
                    c.module_complete = is_complete
                    break


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_proteins(faa_path: Path) -> Dict[str, SeqIO.SeqRecord]:
    out = {}
    for record in SeqIO.parse(faa_path, "fasta"):
        pid = record.id.split()[0]
        out[pid] = record
    return out


def _load_annotations(annot_path: Path) -> Dict[str, Dict]:
    if not annot_path.exists():
        log.warning(f"Annotation table missing: {annot_path}")
        return {}
    out = {}
    with open(annot_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cid = row.get("candidate_id", row.get("locus_tag", ""))
            if cid:
                out[cid] = row
    return out


def _keyword_match(func_str: str, keywords: Set[str]) -> bool:
    func_lower = func_str.lower()
    return any(kw in func_lower for kw in keywords)


def _infer_spanin_type(
    hits:        List[Dict],
    func:        str,
    n_tm:        int,
    has_sig_pep: bool,
) -> str:
    func_lower = func.lower()
    if "i-spanin" in func_lower or "rz" in func_lower:
        return "i_spanin"
    if "o-spanin" in func_lower or "rz1" in func_lower:
        return "o_spanin"
    # Infer from topology
    if n_tm == 1 and not has_sig_pep:
        return "i_spanin"
    if n_tm == 0 and has_sig_pep:
        return "o_spanin"
    if n_tm == 1 and has_sig_pep:
        return "u_spanin"
    return "spanin_unknown"


def _write_hmmer_summary(domain_hits: Dict, output_path: Path) -> None:
    rows = []
    for pid, hits in domain_hits.items():
        for h in hits:
            rows.append({
                "protein_id":  pid,
                "pfam_acc":    h["accession"],
                "domain_type": h["domain_type"],
                "description": h["description"],
                "evalue":      h["evalue"],
                "score":       h["score"],
                "start":       h["start"],
                "end":         h["end"],
            })
    if not rows:
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
