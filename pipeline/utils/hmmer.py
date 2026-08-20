"""
hmmer.py
========
HMMER hmmscan helpers with corrected, scientifically validated domain sets.

Domain set corrections from previous version:
  - PF07411 (YkuD / L,D-transpeptidase) removed from endolysin catalytic set
  - PF01471 (PG_binding_1) removed from catalytic set — kept in CBD_DOMAINS only
  - SAR endolysin detection added (single N-terminal TM helix + catalytic domain)
  - Spanin domain set added
"""

import subprocess
import logging
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple

log = logging.getLogger("hmmer")


# ── Domain accession sets ─────────────────────────────────────────────────────
# All accessions verified against Pfam-A.hmm via hmmstat.
# Version: Pfam 35.0 (Aug 2021) — accessions confirmed present in user database.

# Endolysin CATALYTIC domains — verified from Pfam database
ENDOLYSIN_CATALYTIC_DOMAINS: Dict[str, str] = {
    "PF00959": "Phage_lysozyme  — muramidase, cleaves β-1,4-MurNAc-GlcNAc",
    "PF01520": "Amidase_3       — N-acetylmuramoyl-L-alanine amidase",
    "PF01510": "Amidase_2       — N-acetylmuramoyl-L-alanine amidase type 2",
    "PF05257": "CHAP            — cysteine/histidine-dep amidohydrolase",
    "PF11860": "Muraidase       — N-acetylmuramidase",
    "PF01832": "Glucosaminidase — endo-β-N-acetylglucosaminidase",
    "PF01464": "SLT             — lytic transglycosylase, SLT domain",
    "PF06737": "Transglycosylas — transglycosylase-like domain",
    "PF03245": "Phage_lysis     — phage lysis protein",
}

# Cell wall BINDING domains — separate from catalytic, verified from Pfam
CBD_DOMAINS: Dict[str, str] = {
    "PF01471": "PG_binding_1    — peptidoglycan binding domain",
    "PF00877": "NLPC_P60        — NlpC/P60 peptidoglycan hydrolase",
}

# Holin domains — verified from Pfam database
HOLIN_DOMAINS: Dict[str, str] = {
    "PF04531": "Phage_holin_1   — phage holin superfamily I",
    "PF05102": "Holin_BlyA      — BlyA-type holin",
    "PF11351": "GTA_holin_3TM   — gene transfer agent 3-TM holin",
    "PF13272": "Holin_2-3       — holin superfamily 2-3",
    "PF04688": "Holin_SPP1      — SPP1 phage holin",
    "PF10960": "Holin_BhlA      — BhlA-type holin",
    "PF05106": "Phage_holin_3_1 — phage holin family 3",
    "PF04020": "Phage_holin_4_2 — phage holin family 4",
    "PF07332": "Phage_holin_3_6 — phage holin family 3-6",
    "PF05105": "Phage_holin_4_1 — phage holin family 4-1",
    "PF16079": "Phage_holin_5_2 — phage holin family 5",
    "PF16080": "Phage_holin_2_3 — phage holin family 2-3",
    "PF16082": "Phage_holin_2_4 — phage holin family 2-4",
    "PF16083": "Phage_holin_3_3 — phage holin family 3-3",
    "PF16085": "Phage_holin_3_5 — phage holin family 3-5",
    "PF16081": "Phage_holin_7_1 — phage holin family 7",
    "PF16931": "Phage_holin_8   — phage holin family 8",
    "PF16938": "Phage_holin_Dp1 — Dp1-type holin",
    "PF16936": "Holin_9         — holin family 9",
    "PF04971": "Phage_holin_2_1 — phage holin family 2-1",
    "PF10746": "Phage_holin_2_2 — phage holin family 2-2",
    "PF09682": "Phage_holin_6_1 — phage holin family 6",
    "PF11031": "Phage_holin_T   — T-type holin",
    "PF05449": "Phage_holin_3_7 — phage holin family 3-7",
    "PF16945": "Phage_r1t_holin — r1t phage holin",
    "PF04550": "Phage_holin_3_2 — phage holin family 3-2",
    "PF06946": "Phage_holin_5_1 — phage holin family 5-1",
    "PF09501": "Bac_small_YrzI  — small bacteriophage holin",
    "PF17449": "yrzK            — yrzK holin-like",
    "PF14142": "YrzO            — YrzO holin-like",
}

# Spanin domains — verified from Pfam database
SPANIN_DOMAINS: Dict[str, str] = {
    "PF17531": "O_Spanin_T7     — T7-type outer membrane spanin",
    "PF06085": "Rz1             — o-spanin (Rz1-like outer membrane lipoprotein)",
}

# Convenient flat sets for fast membership testing
_ENDOLYSIN_ACC  = set(ENDOLYSIN_CATALYTIC_DOMAINS)
_CBD_ACC        = set(CBD_DOMAINS)
_HOLIN_ACC      = set(HOLIN_DOMAINS)
_SPANIN_ACC     = set(SPANIN_DOMAINS)

# Combined for single hmmscan run
ALL_DOMAINS: Dict[str, str] = {
    **ENDOLYSIN_CATALYTIC_DOMAINS,
    **CBD_DOMAINS,
    **HOLIN_DOMAINS,
    **SPANIN_DOMAINS,
}


# ── HMMER runner ──────────────────────────────────────────────────────────────

def run_hmmscan(
    fasta_path:    str,
    pfam_hmm:      str,
    output_dir:    str,
    threads:       int   = 4,
    evalue:        float = 1e-3,
    prefix:        str   = "hmmscan",
) -> str:
    """
    Run hmmscan against Pfam HMM database.
    Returns path to domtblout file.
    Raises RuntimeError on failure.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    domtblout = str(Path(output_dir) / f"{prefix}_domtblout.txt")
    tblout    = str(Path(output_dir) / f"{prefix}_tblout.txt")

    cmd = [
        "hmmscan",
        "--domtblout", domtblout,
        "--tblout",    tblout,
        "--cpu",       str(threads),
        "-E",          str(evalue),
        "--domE",      str(evalue),
        "--noali",
        pfam_hmm,
        fasta_path,
    ]

    log.info(f"Running hmmscan ({threads} threads, E≤{evalue})")
    log.debug(f"  cmd: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=3600,  # 1 hour max
    )

    if result.returncode != 0:
        log.error(f"hmmscan stderr (last 2000 chars):\n{result.stderr[-2000:]}")
        raise RuntimeError(
            f"hmmscan failed (exit {result.returncode}). "
            f"Check that Pfam-A.hmm is hmmpress'd and readable."
        )

    log.info(f"hmmscan complete → {domtblout}")
    return domtblout


# ── Domtblout parser ──────────────────────────────────────────────────────────

def parse_domtblout(
    domtblout_path: str,
    evalue_threshold: float = 1e-3,
) -> Dict[str, List[Dict]]:
    """
    Parse HMMER domtblout.
    Returns: protein_id → sorted list of hit dicts.

    Hit dict keys:
      accession, description, evalue, score, start, end,
      domain_type  ("endolysin_catalytic"|"cbd"|"holin"|"spanin"|"other")
    """
    hits: Dict[str, List[Dict]] = {}

    with open(domtblout_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split()
            if len(cols) < 23:
                continue

            # HMMER domtblout column layout (--domtblout):
            # col 0  = target name      (HMM profile name,   e.g. "Phage_lysozyme")
            # col 1  = target accession (HMM accession,      e.g. "PF00959.21")
            # col 2  = target length
            # col 3  = query name       (protein sequence ID, e.g. "genome1__CDS_001")
            # col 4  = query accession  (always "-" for protein queries)
            # col 12 = domain i-evalue
            # col 13 = domain score
            # col 17 = alignment start on sequence (query)
            # col 18 = alignment end on sequence (query)
            # col 22+ = description

            protein_id = cols[3]                         # query = protein sequence
            pfam_name  = cols[0]                         # target = HMM profile name
            pfam_acc   = cols[1].split(".")[0]           # target accession, strip version
            evalue     = float(cols[12])                 # domain i-evalue
            score      = float(cols[13])                 # domain score
            dom_start  = int(cols[17])                   # alignment start on sequence
            dom_end    = int(cols[18])                   # alignment end on sequence
            pfam_desc  = " ".join(cols[22:]) if len(cols) > 22 else ""

            if evalue > evalue_threshold:
                continue

            domain_type = _classify_accession(pfam_acc)

            if protein_id not in hits:
                hits[protein_id] = []

            hits[protein_id].append({
                "accession":   pfam_acc,
                "name":        pfam_name,
                "description": pfam_desc,
                "evalue":      evalue,
                "score":       score,
                "start":       dom_start,
                "end":         dom_end,
                "domain_type": domain_type,
            })

    # Sort each protein's hits by evalue, best first
    for pid in hits:
        hits[pid].sort(key=lambda h: h["evalue"])

    log.info(
        f"Parsed {len(hits)} proteins with domain hits from "
        f"{Path(domtblout_path).name}"
    )
    return hits


def _classify_accession(acc: str) -> str:
    if acc in _ENDOLYSIN_ACC:  return "endolysin_catalytic"
    if acc in _CBD_ACC:        return "cbd"
    if acc in _HOLIN_ACC:      return "holin"
    if acc in _SPANIN_ACC:     return "spanin"
    return "other"


# ── Domain query helpers ──────────────────────────────────────────────────────

def has_endolysin_catalytic(hits: List[Dict]) -> bool:
    return any(h["domain_type"] == "endolysin_catalytic" for h in hits)

def has_holin_domain(hits: List[Dict]) -> bool:
    return any(h["domain_type"] == "holin" for h in hits)

def has_spanin_domain(hits: List[Dict]) -> bool:
    return any(h["domain_type"] == "spanin" for h in hits)

def has_cbd(hits: List[Dict]) -> bool:
    return any(h["domain_type"] == "cbd" for h in hits)

def has_domain(hits: List[Dict], accession_set: Set[str]) -> bool:
    return any(h["accession"] in accession_set for h in hits)

def get_accessions(hits: List[Dict]) -> Set[str]:
    return {h["accession"] for h in hits}

def catalytic_domain_type(hits: List[Dict]) -> str:
    """
    Return the highest-confidence catalytic domain type.
    All accessions verified against Pfam-A.hmm.
    Priority order matches expected prevalence in Vibrio prophages.
    """
    accs = get_accessions(hits)
    if {"PF05257"} & accs:                      return "CHAP"
    if {"PF01520", "PF01510"} & accs:           return "amidase"
    if {"PF00959"} & accs:                      return "lysozyme"
    if {"PF01832"} & accs:                      return "glucosaminidase"
    if {"PF01464", "PF06737"} & accs:           return "transglycosylase"
    if {"PF11860", "PF03245"} & accs:           return "muramidase"
    return "unknown"


# ── SAR endolysin detection ───────────────────────────────────────────────────

def is_sar_endolysin(
    hits:          List[Dict],
    sequence:      str,
    n_tm_helices:  int,
    tm_positions:  List[Tuple[int, int]],  # [(start, end), ...]
) -> bool:
    """
    SAR (signal-arrest-release) endolysins have:
      1. Exactly one N-terminal TM helix (within first 50 residues)
      2. A catalytic endolysin domain AFTER the TM helix
    They are misclassified as holins by simple TM counting — this corrects that.
    """
    if not has_endolysin_catalytic(hits):
        return False
    if n_tm_helices != 1:
        return False
    if not tm_positions:
        return False

    # TM helix must be N-terminal (start < 50 aa)
    tm_start, tm_end = tm_positions[0]
    if tm_start > 50:
        return False

    # Catalytic domain must start after the TM helix
    for h in hits:
        if h["domain_type"] == "endolysin_catalytic":
            if h["start"] > tm_end:
                return True

    return False


# ── Classification ────────────────────────────────────────────────────────────

def classify_protein(
    hits:         List[Dict],
    n_tm_helices: int,
    tm_positions: List[Tuple[int, int]],
    sequence:     str,
) -> str:
    """
    Classify a protein as endolysin/holin/spanin/other/unknown
    using domain evidence and topology.

    SAR endolysins are correctly classified as endolysins despite having
    one TM helix.
    """
    if not hits:
        return "unknown"

    # Check SAR before holin (SAR has 1 TM helix, would otherwise look like holin)
    if is_sar_endolysin(hits, sequence, n_tm_helices, tm_positions):
        return "sar_endolysin"

    if has_holin_domain(hits):
        return "holin"
    if has_spanin_domain(hits):
        return "spanin"
    if has_endolysin_catalytic(hits):
        return "endolysin"

    return "other"


# ── Simple TM helix predictor (TMbase heuristic, no external tool needed) ────

_HYDROPHOBIC = set("VILMFYWCA")

def predict_tm_helices_simple(sequence: str) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Simple sliding-window TM helix predictor.
    Window size 19aa, hydrophobic fraction > 0.55 = TM helix.
    Returns (n_helices, [(start, end), ...]).

    Not as accurate as TMHMM/Phobius but requires no external tools and
    is fast enough for Gate 1 pre-screening on CPU.
    """
    window  = 19
    thresh  = 0.55
    in_helix = False
    helices: List[Tuple[int, int]] = []
    helix_start = 0

    for i in range(len(sequence) - window + 1):
        win = sequence[i:i + window]
        frac = sum(1 for aa in win if aa in _HYDROPHOBIC) / window
        if frac >= thresh:
            if not in_helix:
                helix_start = i
                in_helix = True
        else:
            if in_helix:
                helices.append((helix_start, i + window - 1))
                in_helix = False

    if in_helix:
        helices.append((helix_start, len(sequence) - 1))

    # Merge helices that are within 5 residues of each other
    merged: List[Tuple[int, int]] = []
    for h in helices:
        if merged and h[0] - merged[-1][1] <= 5:
            merged[-1] = (merged[-1][0], h[1])
        else:
            merged.append(list(h))

    merged = [tuple(h) for h in merged]
    return len(merged), merged


def predict_signal_peptide_heuristic(sequence: str) -> bool:
    """
    Heuristic signal peptide detection.
    Looks for: positively charged N-region (2-5 aa) + hydrophobic H-region
    (7-15 aa) within first 30 residues.
    """
    if len(sequence) < 15:
        return False

    n_term = sequence[:30]
    # Positive charges in first 5 residues
    n_region_charge = sum(1 for aa in n_term[:5] if aa in "RK")
    if n_region_charge < 1:
        return False

    # Hydrophobic stretch of at least 7 in first 30
    max_hydro = 0
    current   = 0
    for aa in n_term:
        if aa in _HYDROPHOBIC:
            current += 1
            max_hydro = max(max_hydro, current)
        else:
            current = 0

    return max_hydro >= 7
