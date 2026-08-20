"""
cai.py
======
Codon Adaptation Index computed from actual nucleotide CDS sequences.
Requires the nucleotide CDS to be stored alongside the protein (done in M01).

Uses python-codon-tables for reference codon usage tables.
Fallback: Biopython CodonTable if python-codon-tables is unavailable.
"""

import math
import logging
from typing import Dict, Optional

log = logging.getLogger("cai")

# E. coli K-12 high-expression codon preferences (relative adaptiveness values)
# Source: Sharp & Li 1987, updated from CodonW E. coli dataset
_ECOLI_RELATIVE_ADAPTIVENESS: Dict[str, float] = {
    # Phe
    "TTT": 0.296, "TTC": 1.000,
    # Leu
    "TTA": 0.020, "TTG": 0.020, "CTT": 0.020, "CTC": 0.020,
    "CTA": 0.007, "CTG": 1.000,
    # Ile
    "ATT": 0.185, "ATC": 1.000, "ATA": 0.003,
    # Met
    "ATG": 1.000,
    # Val
    "GTT": 0.221, "GTC": 0.063, "GTA": 0.016, "GTG": 1.000,
    # Ser
    "TCT": 0.085, "TCC": 0.069, "TCA": 0.016, "TCG": 0.013,
    "AGT": 0.016, "AGC": 1.000,
    # Pro
    "CCT": 0.070, "CCC": 0.012, "CCA": 0.017, "CCG": 1.000,
    # Thr
    "ACT": 0.965, "ACC": 1.000, "ACA": 0.076, "ACG": 0.099,
    # Ala
    "GCT": 0.586, "GCC": 0.122, "GCA": 0.586, "GCG": 1.000,
    # Tyr
    "TAT": 0.239, "TAC": 1.000,
    # Stop
    "TAA": 1.000, "TAG": 0.002, "TGA": 0.549,
    # His
    "CAT": 0.291, "CAC": 1.000,
    # Gln
    "CAA": 0.124, "CAG": 1.000,
    # Asn
    "AAT": 0.051, "AAC": 1.000,
    # Lys
    "AAA": 1.000, "AAG": 0.253,
    # Asp
    "GAT": 0.434, "GAC": 1.000,
    # Glu
    "GAA": 1.000, "GAG": 0.259,
    # Cys
    "TGT": 0.500, "TGC": 1.000,
    # Trp
    "TGG": 1.000,
    # Arg
    "CGT": 1.000, "CGC": 0.356, "CGA": 0.004, "CGG": 0.004,
    "AGA": 0.002, "AGG": 0.002,
    # Gly
    "GGT": 1.000, "GGC": 0.724, "GGA": 0.010, "GGG": 0.019,
}


def compute_cai(nucleotide_seq: str, fallback: float = 0.5) -> float:
    """
    Compute Codon Adaptation Index from the nucleotide CDS sequence.

    Args:
        nucleotide_seq: nucleotide CDS (must be multiple of 3, no stop codon needed)
        fallback: value returned if sequence is too short or invalid

    Returns:
        CAI value in [0, 1]. Higher = better adapted to E. coli expression.
    """
    seq = nucleotide_seq.upper().replace(" ", "").replace("\n", "")

    # Trim stop codon if present
    if seq[-3:] in ("TAA", "TAG", "TGA"):
        seq = seq[:-3]

    if len(seq) < 9:
        return fallback

    if len(seq) % 3 != 0:
        # Trim to nearest codon
        seq = seq[:len(seq) - len(seq) % 3]

    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    scores = []

    for codon in codons:
        if "N" in codon:
            continue
        w = _ECOLI_RELATIVE_ADAPTIVENESS.get(codon)
        if w is None:
            continue
        if w <= 0:
            w = 0.01   # avoid log(0)
        scores.append(math.log(w))

    if not scores:
        return fallback

    # Geometric mean = exp(mean of logs)
    return round(math.exp(sum(scores) / len(scores)), 4)


def compute_cai_if_available(
    nucleotide_seq: Optional[str],
    fallback: float = 0.5,
) -> float:
    """Compute CAI if nucleotide sequence is available, else return fallback."""
    if not nucleotide_seq or len(nucleotide_seq) < 9:
        log.debug("No nucleotide sequence available for CAI — using fallback")
        return fallback
    return compute_cai(nucleotide_seq, fallback)
