"""tests/test_utils.py — Unit tests for CAI, numba kernels, PG database."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pipeline.utils.cai import compute_cai, compute_cai_if_available
from pipeline.utils.numba_kernels import (
    l2_normalize, cosine_similarity_matrix, maxmin_select,
    compute_mean_pairwise_distance, block_cosine_similarity,
)


# ── CAI ───────────────────────────────────────────────────────────────────────

class TestCAI:
    """Codon Adaptation Index — real nucleotide CDS."""

    def test_optimal_ecoli_codons_high_cai(self):
        # Use only high-adaptiveness codons: CTG (Leu), ATC (Ile), GTG (Val), GCG (Ala)
        optimal = "CTG" * 30   # all Leu-CTG (w=1.0) — should give CAI close to 1.0
        cai = compute_cai(optimal)
        assert cai > 0.9, f"Expected CAI > 0.9, got {cai}"

    def test_rare_codons_low_cai(self):
        # ATA (Ile, w=0.003), CGA (Arg, w=0.004) — should give very low CAI
        rare = "ATA" * 20 + "CGA" * 10
        cai = compute_cai(rare)
        assert cai < 0.3, f"Expected CAI < 0.3, got {cai}"

    def test_cai_range_zero_to_one(self):
        seq = "ATG" + "CTG" * 40 + "TAA"
        cai = compute_cai(seq)
        assert 0.0 <= cai <= 1.0

    def test_stop_codon_stripped(self):
        seq_with_stop    = "CTG" * 30 + "TAA"
        seq_without_stop = "CTG" * 30
        assert abs(compute_cai(seq_with_stop) - compute_cai(seq_without_stop)) < 0.01

    def test_empty_sequence_fallback(self):
        cai = compute_cai_if_available("", fallback=0.42)
        assert cai == 0.42

    def test_none_sequence_fallback(self):
        cai = compute_cai_if_available(None, fallback=0.5)
        assert cai == 0.5

    def test_short_sequence_fallback(self):
        cai = compute_cai_if_available("ATG", fallback=0.5)
        assert cai == 0.5   # too short (< 9 nt)

    def test_cai_is_float(self):
        cai = compute_cai("CTG" * 20)
        assert isinstance(cai, float)


# ── Numba kernels ─────────────────────────────────────────────────────────────

class TestL2Normalize:
    def test_unit_norm(self):
        mat  = np.random.randn(10, 128).astype(np.float32)
        norm = l2_normalize(mat)
        norms = np.linalg.norm(norm, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_zero_vector_safe(self):
        mat = np.zeros((3, 10), dtype=np.float32)
        norm = l2_normalize(mat)
        assert not np.any(np.isnan(norm))

    def test_output_dtype(self):
        mat  = np.random.randn(5, 16).astype(np.float64)
        norm = l2_normalize(mat)
        assert norm.dtype == np.float32


class TestCosineSimilarityMatrix:
    def test_symmetric(self):
        mat = np.random.randn(8, 32).astype(np.float32)
        sim = cosine_similarity_matrix(mat)
        np.testing.assert_allclose(sim, sim.T, atol=1e-5)

    def test_diagonal_one(self):
        mat = np.random.randn(8, 32).astype(np.float32)
        sim = cosine_similarity_matrix(mat)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_range_minus1_to_1(self):
        mat = np.random.randn(10, 64).astype(np.float32)
        sim = cosine_similarity_matrix(mat)
        assert np.all(sim >= -1.0 - 1e-5)
        assert np.all(sim <=  1.0 + 1e-5)

    def test_identical_vectors_similarity_1(self):
        v   = np.random.randn(1, 32).astype(np.float32)
        mat = np.tile(v, (5, 1))
        sim = cosine_similarity_matrix(mat)
        np.testing.assert_allclose(sim, 1.0, atol=1e-4)


class TestMaxMinSelect:
    def test_returns_correct_count(self):
        # Use well-separated vectors to ensure MaxMin always finds 5 distinct candidates
        rng = np.random.default_rng(0)
        # 20 vectors in 32D — add large random offsets so all are distant
        mat = rng.standard_normal((20, 32)).astype(np.float32)
        mat += rng.standard_normal((20, 1)).astype(np.float32) * 5  # spread them out
        selected, dists = maxmin_select(mat, max_n=5)
        assert len(selected) == 5, f"Expected 5, got {len(selected)}: {selected}"

    def test_no_duplicates(self):
        mat = np.random.randn(15, 32).astype(np.float32)
        selected, _ = maxmin_select(mat, max_n=8)
        assert len(set(selected)) == len(selected)

    def test_indices_in_range(self):
        n   = 12
        mat = np.random.randn(n, 16).astype(np.float32)
        selected, _ = maxmin_select(mat, max_n=6)
        assert all(0 <= i < n for i in selected)

    def test_distances_non_negative(self):
        mat = np.random.randn(10, 32).astype(np.float32)
        _, dists = maxmin_select(mat, max_n=5)
        assert all(d >= 0 for d in dists)

    def test_identical_vectors_zero_diversity(self):
        v   = np.random.randn(1, 32).astype(np.float32)
        mat = np.tile(v, (8, 1))
        _, dists = maxmin_select(mat, max_n=4)
        # After first selection, all remaining dists should be ~0
        assert all(d < 1e-4 for d in dists[1:])

    def test_max_n_gt_n_returns_all(self):
        n   = 6
        rng = np.random.default_rng(7)
        # Ensure all vectors are distinct and well-separated
        mat = np.eye(n, 32, dtype=np.float32) * 10 + rng.standard_normal((n, 32)).astype(np.float32) * 0.01
        selected, _ = maxmin_select(mat, max_n=100)
        assert len(selected) == n, f"Expected {n}, got {len(selected)}"


class TestMeanPairwiseDistance:
    def test_identical_vectors_zero_distance(self):
        v   = np.random.randn(1, 32).astype(np.float32)
        mat = np.tile(v, (4, 1))
        d   = compute_mean_pairwise_distance(mat, [0, 1, 2, 3])
        assert d < 1e-4

    def test_single_item_zero(self):
        mat = np.random.randn(4, 16).astype(np.float32)
        d   = compute_mean_pairwise_distance(mat, [0])
        assert d == 0.0

    def test_orthogonal_vectors_distance_one(self):
        # Two orthogonal vectors → cosine similarity 0 → distance 1
        mat = np.zeros((2, 4), dtype=np.float32)
        mat[0, 0] = 1.0
        mat[1, 1] = 1.0
        d = compute_mean_pairwise_distance(mat, [0, 1])
        assert abs(d - 1.0) < 1e-5


class TestBlockCosineSimilarity:
    def test_no_oom_medium_matrix(self):
        mat   = np.random.randn(50, 128).astype(np.float32)
        edges = block_cosine_similarity(mat, threshold=0.5, block_size=16)
        assert edges.shape[1] == 3

    def test_edges_above_threshold(self):
        # All-same matrix → all pairs should be edges with sim ≈ 1.0
        v   = np.random.randn(1, 32).astype(np.float32)
        mat = np.tile(v, (8, 1))
        edges = block_cosine_similarity(mat, threshold=0.99)
        # Should have C(8,2)=28 edges
        assert len(edges) == 28

    def test_distinct_vectors_no_edges(self):
        # Build 8 orthogonal vectors → cosine similarity = 0 → no edges at threshold 0.5
        mat = np.eye(8, dtype=np.float32)
        edges = block_cosine_similarity(mat, threshold=0.5)
        assert len(edges) == 0


# ── PG Database ───────────────────────────────────────────────────────────────

class TestPathogenDatabase:
    def test_load_from_yaml(self, tmp_path):
        from pipeline.utils.pg_database import PathogenDatabase
        db_yaml = tmp_path / "pathogen_db.yaml"
        db_yaml.write_text("""
pathogens:
  - id: test_vibrio
    display_name: "Test Vibrio"
    species: "Vibrio test"
    gram_stain: "negative"
    pg_chemotype: "gram_negative_dap_om_barrier"
    aquaculture_host: "flounder"
    notes: ""
""")
        db = PathogenDatabase(str(db_yaml))
        assert "test_vibrio" in db.pathogen_ids()

    def test_score_endolysin_lysozyme_vs_gramneg(self, tmp_path):
        from pipeline.utils.pg_database import PathogenDatabase
        db_yaml = tmp_path / "pathogen_db.yaml"
        db_yaml.write_text("""
pathogens:
  - id: vib_h
    display_name: "Vibrio harveyi"
    species: "Vibrio harveyi"
    gram_stain: "negative"
    pg_chemotype: "gram_negative_dap_om_barrier"
    aquaculture_host: "flounder"
    notes: ""
""")
        db    = PathogenDatabase(str(db_yaml))
        score = db.score_endolysin("vib_h", "lysozyme")
        assert score == 2

    def test_score_chap_vs_grampos(self, tmp_path):
        from pipeline.utils.pg_database import PathogenDatabase
        db_yaml = tmp_path / "pathogen_db.yaml"
        db_yaml.write_text("""
pathogens:
  - id: strep
    display_name: "Streptococcus"
    species: "Streptococcus parauberis"
    gram_stain: "positive"
    pg_chemotype: "gram_positive_lys"
    aquaculture_host: "flounder"
    notes: ""
""")
        db    = PathogenDatabase(str(db_yaml))
        score = db.score_endolysin("strep", "CHAP")
        assert score == 2

    def test_add_target_persists(self, tmp_path):
        from pipeline.utils.pg_database import PathogenDatabase
        db_yaml = tmp_path / "pathogen_db.yaml"
        db_yaml.write_text("pathogens: []\n")
        db = PathogenDatabase(str(db_yaml))
        db.add_target(
            pathogen_id     = "new_pathogen",
            display_name    = "New Pathogen",
            species         = "New sp.",
            gram_stain      = "negative",
            pg_chemotype    = "gram_negative_dap",
            aquaculture_host= "shrimp",
        )
        # Reload and check
        db2 = PathogenDatabase(str(db_yaml))
        assert "new_pathogen" in db2.pathogen_ids()

    def test_add_target_invalid_chemotype(self, tmp_path):
        from pipeline.utils.pg_database import PathogenDatabase
        db_yaml = tmp_path / "pathogen_db.yaml"
        db_yaml.write_text("pathogens: []\n")
        db = PathogenDatabase(str(db_yaml))
        with pytest.raises(ValueError, match="Unknown PG chemotype"):
            db.add_target(
                pathogen_id   = "bad",
                display_name  = "Bad",
                species       = "Bad sp.",
                gram_stain    = "negative",
                pg_chemotype  = "not_real_chemotype",
                aquaculture_host = "",
            )

    def test_uncovered_pathogens(self, tmp_path):
        from pipeline.utils.pg_database import PathogenDatabase
        db_yaml = tmp_path / "pathogen_db.yaml"
        db_yaml.write_text("""
pathogens:
  - id: p1
    display_name: P1
    species: sp1
    gram_stain: "negative"
    pg_chemotype: "gram_negative_dap"
    aquaculture_host: ""
    notes: ""
  - id: p2
    display_name: P2
    species: sp2
    gram_stain: "positive"
    pg_chemotype: "gram_positive_lys"
    aquaculture_host: ""
    notes: ""
""")
        db = PathogenDatabase(str(db_yaml))
        # Only one candidate with p1 score 2, nothing for p2
        pg_scores = [{"p1": 2, "p2": 0}]
        uncovered = db.uncovered_pathogens(pg_scores, min_coverage=2)
        assert "p1" in uncovered   # only 1, need 2
        assert "p2" in uncovered


# ── Conftest ──────────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks test as slow")
    config.addinivalue_line("markers", "gpu: marks test as requiring GPU")
