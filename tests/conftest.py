"""tests/conftest.py — Shared fixtures."""
import json
import numpy as np
import pytest

from pipeline.utils.data_model import (
    EndolysínRecord, HolinRecord, SpanínRecord, LysisModule,
)


# ── Sequence fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def good_endolysin_seq():
    return "MKLSTLEKQLVDAIRAQMKADIAQLKAEIEKLKKELEEAKEQMAELRKKLENALEEQAK" * 3

@pytest.fixture
def hydrophobic_seq():
    return "LLLLVVVVIIIIFFFFWWWW" * 10

@pytest.fixture
def holin_seq():
    return "MWLLVVIIAGLLAGILSG" + "MKLEKQLVDAIREQ" * 8


# ── Candidate fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def endolysin_candidate(good_endolysin_seq):
    return EndolysínRecord(
        candidate_id   = "genome1__endo01",
        genome_id      = "genome1",
        protein_id     = "endo01",
        sequence       = good_endolysin_seq,
        nucleotide_seq = "CTG" * (len(good_endolysin_seq)),
        has_chap       = True,
        pfam_domains   = ["PF04851"],
        cds_start      = 1000,
        cds_end        = 1600,
        cds_strand     = "+",
    )

@pytest.fixture
def holin_candidate(holin_seq):
    return HolinRecord(
        candidate_id = "genome1__holin01",
        genome_id    = "genome1",
        protein_id   = "holin01",
        sequence     = holin_seq,
        pfam_domains = ["PF04531"],
        n_tm_helices = 2,
        cds_start    = 900,
        cds_end      = 1050,
        cds_strand   = "+",
    )

@pytest.fixture
def spanin_candidate():
    return SpanínRecord(
        candidate_id = "genome1__spanin01",
        genome_id    = "genome1",
        protein_id   = "spanin01",
        sequence     = "MKRNAVLLLGAVLALTACSSNADAQAQE" + "MKLEKQL" * 8,
        pfam_domains = ["PF16614"],
        spanin_type  = "i_spanin",
        cds_start    = 1650,
        cds_end      = 1900,
        cds_strand   = "+",
    )

@pytest.fixture
def lysis_module(endolysin_candidate, holin_candidate, spanin_candidate):
    return LysisModule(
        module_id    = "genome1__mod0001",
        genome_id    = "genome1",
        prophage_id  = "genome1",
        endolysin_id = endolysin_candidate.candidate_id,
        holin_id     = holin_candidate.candidate_id,
        ispanin_id   = spanin_candidate.candidate_id,
        completeness = "complete",
        genomic_span_bp = 900,
    )


# ── Embedding fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def random_embedding_matrix():
    """10 random 1280-dim embeddings (like ESM-2 output)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 1280)).astype(np.float32)

@pytest.fixture
def clusterable_embedding_matrix():
    """
    20 embeddings arranged in 4 clear clusters of 5.
    Designed so HDBSCAN / MaxMin behave predictably in tests.
    """
    rng     = np.random.default_rng(42)
    centers = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    chunks = []
    for center in centers:
        # 5 points around each center with small noise
        noise = rng.standard_normal((5, 4)).astype(np.float32) * 0.05
        chunks.append(center[np.newaxis, :] + noise)
    return np.vstack(chunks)
