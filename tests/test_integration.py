"""
tests/test_integration.py
==========================
Integration tests using fully synthetic data.
No external databases (Pharokka, HMMER) required.

These tests mock M01 outputs and run M02–M09 sequentially on synthetic
candidate data to verify end-to-end data flow and logic.

Mark: not slow — these should complete in < 60s on any machine.
"""

from __future__ import annotations
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Add project root to path for direct import
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.utils.data_model import (
    EndolysínRecord, HolinRecord, SpanínRecord, LysisModule,
    save_candidates, save_modules, load_candidates, load_modules,
)


# ── Synthetic data factory ────────────────────────────────────────────────────

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_WEIGHTS = [
    1.28, 5.30, 3.87, 6.32, 7.08, 2.28, 5.49, 9.68, 4.97, 2.32,
    3.88, 3.87, 5.02, 3.90, 5.52, 6.71, 5.34, 6.73, 1.37, 2.76,
]


def rand_seq(length: int, rng: random.Random) -> str:
    return "".join(rng.choices(AA_LIST, weights=AA_WEIGHTS, k=length))


def rand_hydrophobic(length: int, rng: random.Random) -> str:
    return "".join(rng.choices(list("LLLVVVIIIFFFW"), k=length))


def rand_cds(protein: str, rng: random.Random) -> str:
    """Minimal back-translation: each AA → one codon, stop = TAA."""
    simple_codon = {
        'A': 'GCT', 'C': 'TGT', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT',
        'G': 'GGT', 'H': 'CAT', 'I': 'ATT', 'K': 'AAA', 'L': 'CTG',
        'M': 'ATG', 'N': 'AAT', 'P': 'CCT', 'Q': 'CAA', 'R': 'CGT',
        'S': 'TCT', 'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAT',
    }
    return "".join(simple_codon.get(aa, 'NNN') for aa in protein) + "TAA"


def make_synthetic_candidates(
    n_endolysins: int = 20,
    n_holins:     int = 10,
    n_spanins:    int = 5,
    seed:         int = 42,
) -> List:
    rng = random.Random(seed)
    candidates = []
    pos = 100

    for i in range(n_endolysins):
        seq = "M" + rand_seq(rng.randint(130, 220), rng)
        nuc = rand_cds(seq, rng)
        c = EndolysínRecord(
            candidate_id       = f"genome1__lysin_{i+1:03d}",
            genome_id          = "genome1",
            protein_id         = f"lysin_{i+1:03d}",
            sequence           = seq,
            nucleotide_seq     = nuc,
            track              = "endolysin",
            source_organism    = "genome1",
            cds_start          = pos,
            cds_end            = pos + len(seq) * 3,
            cds_strand         = "+",
            pharokka_function  = "endolysin peptidoglycan hydrolase",
            pharokka_category  = "lysis",
            pfam_domains       = ["PF04851"] if i % 3 == 0 else ["PF01520"],
            pfam_descriptions  = ["CHAP"] if i % 3 == 0 else ["Amidase_2"],
            pfam_evalues       = [1e-20],
            has_chap           = (i % 3 == 0),
            has_amidase        = (i % 3 == 1),
            has_lysozyme       = (i % 3 == 2),
            inclusion_reason   = "hmmer_domain",
            length_aa          = len(seq),
        )
        candidates.append(c)
        pos += len(seq) * 3 + rng.randint(20, 100)

    for i in range(n_holins):
        tm1 = rand_hydrophobic(20, rng)
        tm2 = rand_hydrophobic(20, rng)
        seq = "MKK" + rand_seq(8, rng) + tm1 + rand_seq(8, rng) + tm2 + rand_seq(10, rng)
        nuc = rand_cds(seq, rng)
        h = HolinRecord(
            candidate_id      = f"genome1__holin_{i+1:03d}",
            genome_id         = "genome1",
            protein_id        = f"holin_{i+1:03d}",
            sequence          = seq,
            nucleotide_seq    = nuc,
            track             = "holin",
            source_organism   = "genome1",
            cds_start         = pos,
            cds_end           = pos + len(seq) * 3,
            cds_strand        = "+",
            pharokka_function = "phage holin",
            pharokka_category = "lysis",
            pfam_domains      = ["PF04531"],
            pfam_descriptions = ["Holin_LLH"],
            pfam_evalues      = [1e-15],
            n_tm_helices      = 2,
            inclusion_reason  = "hmmer_domain",
            length_aa         = len(seq),
        )
        candidates.append(h)
        pos += len(seq) * 3 + rng.randint(20, 100)

    for i in range(n_spanins):
        tm = rand_hydrophobic(20, rng)
        seq = "MK" + tm + rand_seq(rng.randint(60, 80), rng)
        nuc = rand_cds(seq, rng)
        s = SpanínRecord(
            candidate_id      = f"genome1__spanin_{i+1:03d}",
            genome_id         = "genome1",
            protein_id        = f"spanin_{i+1:03d}",
            sequence          = seq,
            nucleotide_seq    = nuc,
            track             = "spanin",
            source_organism   = "genome1",
            cds_start         = pos,
            cds_end           = pos + len(seq) * 3,
            cds_strand        = "+",
            pharokka_function = "i-spanin Rz-like",
            pharokka_category = "lysis",
            pfam_domains      = ["PF11551"],
            pfam_descriptions = ["Rz"],
            pfam_evalues      = [1e-12],
            spanin_type       = "i_spanin",
            n_tm_helices      = 1,
            inclusion_reason  = "hmmer_domain",
            length_aa         = len(seq),
        )
        candidates.append(s)
        pos += len(seq) * 3 + rng.randint(20, 100)

    return candidates


def make_synthetic_modules(candidates: List) -> List[LysisModule]:
    """Build simple modules linking the first endolysin + holin + spanin."""
    endos  = [c for c in candidates if c.track == "endolysin"]
    holins = [c for c in candidates if c.track == "holin"]
    spans  = [c for c in candidates if c.track == "spanin"]
    mods   = []
    for i, endo in enumerate(endos):
        hol = holins[i % len(holins)] if holins else None
        sp  = spans[i % len(spans)]   if spans  else None
        mod = LysisModule(
            module_id    = f"genome1__mod{i:04d}",
            genome_id    = "genome1",
            prophage_id  = "genome1",
            endolysin_id = endo.candidate_id,
            holin_id     = hol.candidate_id if hol else None,
            ispanin_id   = sp.candidate_id  if sp  else None,
            completeness = "complete" if hol and sp else "partial",
        )
        endo.module_id     = mod.module_id
        endo.module_complete = mod.completeness == "complete"
        if hol:
            hol.module_id      = mod.module_id
            hol.module_complete = mod.completeness == "complete"
        if sp:
            sp.module_id       = mod.module_id
            sp.module_complete = mod.completeness == "complete"
        mods.append(mod)
    return mods


def make_synthetic_embeddings(
    candidates: List,
    dim:        int = 64,    # small dim for test speed (real is 1280)
    seed:       int = 42,
) -> tuple:
    """
    Generate fake embeddings with realistic cluster structure.
    Endolysins from different domain types get different embedding regions.
    """
    rng = np.random.default_rng(seed)
    index   = []
    vectors = []

    domain_centers = {
        "CHAP":              rng.standard_normal(dim).astype(np.float32) * 5,
        "amidase":           rng.standard_normal(dim).astype(np.float32) * 5,
        "lysozyme":          rng.standard_normal(dim).astype(np.float32) * 5,
        "holin_center":      rng.standard_normal(dim).astype(np.float32) * 5,
        "spanin_center":     rng.standard_normal(dim).astype(np.float32) * 5,
    }

    for c in candidates:
        if c.track == "endolysin":
            from pipeline.utils.data_model import EndolysínRecord
            if isinstance(c, EndolysínRecord):
                center_key = c.catalytic_domain_type
            else:
                center_key = "CHAP"
            center = domain_centers.get(center_key, domain_centers["CHAP"])
        elif c.track == "holin":
            center = domain_centers["holin_center"]
        else:
            center = domain_centers["spanin_center"]

        noise  = rng.standard_normal(dim).astype(np.float32) * 0.5
        vector = center + noise
        index.append(c.candidate_id)
        vectors.append(vector)

    matrix = np.vstack(vectors).astype(np.float32)
    return matrix, index


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGate1Integration:
    """Gate 1 on synthetic candidates — end-to-end with BioPython."""

    def test_gate1_runs_without_crash(self, tmp_path):
        from pipeline.modules.m03_gate1 import (
            _set_physicochemical, _evaluate_endolysin,
            _evaluate_holin, _evaluate_spanin, _apply_gate1_result,
        )
        from pipeline.utils.data_model import split_by_track

        g1_cfg = {
            "max_mw_kda": 70.0, "max_gravy_endolysin": 0.1,
            "max_instability_index": 60.0, "min_cai_score": 0.55,
            "flags_to_fail": 2, "max_mw_kda_holin": 25.0,
            "min_gravy_holin": -0.1, "max_tm_helices_holin": 4,
            "flags_to_fail_holin": 2, "max_mw_kda_spanin": 35.0,
            "flags_to_fail_spanin": 1,
        }
        candidates = make_synthetic_candidates(n_endolysins=10, n_holins=5, n_spanins=3)
        tracks = split_by_track(candidates)

        for c in tracks["endolysin"]:
            _set_physicochemical(c)
            flags = _evaluate_endolysin(c, g1_cfg)
            _apply_gate1_result(c, flags, g1_cfg["flags_to_fail"])

        for c in tracks["holin"]:
            _set_physicochemical(c)
            flags = _evaluate_holin(c, g1_cfg)
            _apply_gate1_result(c, flags, g1_cfg["flags_to_fail_holin"])

        for c in tracks["spanin"]:
            _set_physicochemical(c)
            flags = _evaluate_spanin(c, g1_cfg)
            _apply_gate1_result(c, flags, g1_cfg["flags_to_fail_spanin"])

        # All candidates must have gate1_status set
        assert all(c.gate1_status is not None for c in candidates)
        # All candidates must have physicochemical fields
        assert all(c.mw_kda is not None for c in candidates)
        # All must have length_aa
        assert all(c.length_aa is not None for c in candidates)

    def test_gate1_n_tm_helices_serialized(self, tmp_path):
        """n_tm_helices must survive JSON serialization after Gate 1."""
        from pipeline.modules.m03_gate1 import _set_physicochemical, _evaluate_endolysin, _apply_gate1_result
        from pipeline.utils.data_model import save_candidates, load_candidates

        g1_cfg = {
            "max_mw_kda": 70.0, "max_gravy_endolysin": 0.1,
            "max_instability_index": 60.0, "min_cai_score": 0.55,
            "flags_to_fail": 2,
        }
        candidates = make_synthetic_candidates(n_endolysins=5, n_holins=0, n_spanins=0)
        for c in candidates:
            _set_physicochemical(c)
            flags = _evaluate_endolysin(c, g1_cfg)
            _apply_gate1_result(c, flags, g1_cfg["flags_to_fail"])

        path = str(tmp_path / "candidates.json")
        save_candidates(candidates, path)
        loaded = load_candidates(path)

        for c in loaded:
            # n_tm_helices must be preserved (was previously lost)
            assert "n_tm_helices" in c.to_dict(), \
                "n_tm_helices field missing from serialized EndolysínRecord"


class TestRedundancyIntegration:
    """M07 redundancy collapse on synthetic embeddings."""

    def test_identical_embeddings_collapsed(self, tmp_path):
        from pipeline.utils.numba_kernels import cosine_similarity_matrix

        rng = np.random.default_rng(42)
        # 5 identical + 5 distinct vectors
        base   = rng.standard_normal(64).astype(np.float32)
        matrix = np.vstack([
            np.tile(base, (5, 1)),   # 5 identical → should all collapse to one representative
            rng.standard_normal((5, 64)).astype(np.float32) * 5,  # 5 very different
        ])

        sim = cosine_similarity_matrix(matrix)
        np.fill_diagonal(sim, 0)

        # The 5 identical rows (indices 0–4) should all be > 0.99 similar to each other
        for i in range(5):
            for j in range(i + 1, 5):
                assert sim[i, j] > 0.99, \
                    f"Identical vectors not detected as similar: sim[{i},{j}]={sim[i,j]:.4f}"

    def test_maxmin_selects_diverse_set(self):
        from pipeline.utils.numba_kernels import maxmin_select, compute_mean_pairwise_distance

        rng = np.random.default_rng(42)
        # 4 well-separated clusters in 32D space
        # Use orthogonal unit vectors as centers — cosine distance = 1.0 between them
        dim = 32
        centers = np.zeros((4, dim), dtype=np.float32)
        for i in range(4):
            centers[i, i * 8] = 10.0  # spike in different dimension per cluster

        chunks = []
        for c in centers:
            noise = rng.standard_normal((5, dim)).astype(np.float32) * 0.05
            chunks.append(c[np.newaxis] + noise)
        matrix = np.vstack(chunks)  # (20, 32)

        selected, dists = maxmin_select(matrix, max_n=4)
        assert len(selected) == 4, f"Expected 4 selected, got {len(selected)}"
        mpd = compute_mean_pairwise_distance(matrix, selected)
        # Clusters are near-orthogonal — cosine distance between them ~ 1.0
        assert mpd > 0.5, f"MaxMin selected non-diverse set: MPD={mpd:.4f}"


class TestClusteringIntegration:
    """M05 clustering on synthetic embeddings — no HMMER/Pharokka required."""

    def test_clustering_per_track(self, tmp_path):
        """UMAP+HDBSCAN assigns cluster IDs to all candidates."""
        import yaml
        from pipeline.utils.data_model import save_candidates, load_candidates, split_by_track

        candidates = make_synthetic_candidates(n_endolysins=15, n_holins=8, n_spanins=5)

        # Set up fake Gate 1 pass status
        for c in candidates:
            c.gate1_status = "pass"
            c.length_aa    = len(c.sequence)

        # Write candidates_passing.json
        cand_dir = tmp_path / "03_gate1"
        cand_dir.mkdir()
        cand_path = str(cand_dir / "candidates_passing.json")
        save_candidates(candidates, cand_path)

        # Write embedding matrix
        matrix, index = make_synthetic_embeddings(candidates, dim=64)
        emb_dir = tmp_path / "04_embeddings"
        emb_dir.mkdir()
        np.save(emb_dir / "embedding_matrix.npy", matrix)
        (emb_dir / "embedding_index.json").write_text(json.dumps(index))

        # Write minimal config
        cfg = {
            "paths": {
                "intermediate_dir": str(tmp_path),
                "output_dir":       str(tmp_path / "output"),
                "log_dir":          str(tmp_path / "logs"),
            },
            "clustering": {
                "umap_n_components":      10,   # small for test speed
                "umap_n_neighbors":        5,
                "umap_min_dist":           0.1,
                "umap_2d_min_dist":        0.1,
                "umap_metric":            "cosine",
                "umap_random_state":       42,
                "hdbscan_min_cluster_size": 2,
                "hdbscan_min_samples":      1,
                "hdbscan_metric":          "euclidean",
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg))

        # Run M05
        from pipeline.modules.m05_clustering import run
        run(cfg)

        # Verify all candidates have cluster assignments
        updated = load_candidates(cand_path)
        tracks  = split_by_track(updated)

        for track_name, track_cands in tracks.items():
            if track_cands:
                assert all(c.cluster_id is not None for c in track_cands), \
                    f"Track '{track_name}' has candidates without cluster_id"
                assert all(c.umap_x is not None for c in track_cands), \
                    f"Track '{track_name}' has candidates without UMAP coordinates"

        # Cluster summary file must exist
        assert (tmp_path / "05_clusters" / "cluster_summary.json").exists()


class TestSelectionIntegration:
    """M08 MaxMin selection end-to-end."""

    def test_selection_produces_priority_and_reserve(self, tmp_path):
        import yaml
        from pipeline.utils.data_model import save_candidates, save_modules, load_candidates

        candidates = make_synthetic_candidates(n_endolysins=30, n_holins=10, n_spanins=5)
        modules    = make_synthetic_modules(candidates)

        # Set up pre-selection state
        rng = random.Random(42)
        for i, c in enumerate(candidates):
            c.gate1_status    = "pass"
            c.gate3_status    = "representative"
            c.is_representative = True
            c.redundancy_cluster = i
            c.length_aa       = len(c.sequence)
            c.mw_kda          = len(c.sequence) * 0.11
            c.cai_score       = 0.6 + rng.random() * 0.2

        # Write state
        cand_dir = tmp_path / "03_gate1"
        cand_dir.mkdir()
        cand_path = str(cand_dir / "candidates_passing.json")
        save_candidates(candidates, cand_path)

        mod_dir = tmp_path / "02_lysis_modules"
        mod_dir.mkdir()
        mod_path = str(mod_dir / "modules.json")
        save_modules(modules, mod_path)

        # Write embeddings
        matrix, index = make_synthetic_embeddings(candidates, dim=64)
        emb_dir = tmp_path / "04_embeddings"
        emb_dir.mkdir()
        np.save(emb_dir / "embedding_matrix.npy", matrix)
        (emb_dir / "embedding_index.json").write_text(json.dumps(index))

        cfg = {
            "paths": {
                "intermediate_dir": str(tmp_path),
                "output_dir":       str(tmp_path / "output"),
                "log_dir":          str(tmp_path / "logs"),
            },
            "selection": {
                "selection_strategy":  "saturation",
                "min_expression_n":     3,
                "max_expression_n":     15,
                "saturation_threshold": 0.3,
                "run_blast_novelty":    False,
            },
            "pg_matching": {"enabled": False, "enforce_coverage": False},
        }

        # Run M08
        from pipeline.modules.m08_selection import run
        run(cfg)

        # Verify results
        updated = load_candidates(cand_path)
        endos   = [c for c in updated if c.track == "endolysin"]
        priority_endos = [c for c in endos if c.final_status == "priority"]
        reserve_endos  = [c for c in endos if c.final_status == "reserve"]

        assert len(priority_endos) >= 1, "No priority endolysins selected"
        assert len(priority_endos) <= 15, f"Too many priority endolysins: {len(priority_endos)}"
        assert len(reserve_endos) >= 1,  "No reserve endolysins"

        # Priority endolysins must have ranks
        assert all(c.final_rank is not None for c in priority_endos)
        ranks = [c.final_rank for c in priority_endos]
        assert sorted(ranks) == list(range(1, len(ranks) + 1)), \
            f"Ranks not sequential: {sorted(ranks)}"

        # Selection summary file must exist
        assert (tmp_path / "08_selection" / "selection_summary.json").exists()


class TestReportIntegration:
    """M09 report generation end-to-end."""

    def test_report_writes_all_outputs(self, tmp_path):
        import yaml
        from pipeline.utils.data_model import save_candidates, save_modules, load_candidates

        candidates = make_synthetic_candidates(n_endolysins=10, n_holins=5, n_spanins=3)
        modules    = make_synthetic_modules(candidates)

        # Set up fully-processed state
        rng = random.Random(99)
        for i, c in enumerate(candidates):
            c.gate1_status = "pass"
            c.gate1_flags  = []
            c.length_aa    = len(c.sequence)
            c.mw_kda       = len(c.sequence) * 0.11
            c.isoelectric_point = 7.0
            c.gravy        = -0.3
            c.instability_index = 35.0
            c.cai_score    = 0.65
            c.cluster_id   = i % 4
            c.umap_x       = float(rng.gauss(0, 1))
            c.umap_y       = float(rng.gauss(0, 1))
            c.gate3_status = "representative"
            c.is_representative = True
            c.redundancy_cluster = i

            if c.track == "endolysin" and i < 5:
                c.final_status    = "priority"
                c.final_rank      = i + 1
                c.diversity_rank  = i + 1
                c.novelty_flag    = "novel" if i < 2 else "known_homolog"
                c.closest_known   = "" if i < 2 else "SomeProtein [45%]"
                c.module_complete = True
                c.selection_strategy = "maxmin_saturation"
            elif c.track == "endolysin":
                c.final_status = "reserve"
            elif c.track in ("holin", "spanin") and c.module_complete:
                c.final_status = "priority"
                c.selection_strategy = "module_coselection"
            else:
                c.final_status = "reserve"

        # A few eliminated candidates
        for c in candidates[:2]:
            c.final_status     = "eliminated"
            c.elimination_gate = "gate1"
            c.elimination_reason = "MW_too_large_75.0kDa; low_CAI_0.44"
            c.gate1_flags      = ["MW_too_large_75.0kDa", "low_CAI_0.44"]

        cand_dir = tmp_path / "03_gate1"
        cand_dir.mkdir()
        cand_path = str(cand_dir / "candidates_passing.json")
        save_candidates(candidates, cand_path)

        out_dir = tmp_path / "output"

        cfg = {
            "paths": {
                "intermediate_dir": str(tmp_path),
                "output_dir":       str(out_dir),
                "log_dir":          str(tmp_path / "logs"),
            },
            "selection": {"selection_strategy": "saturation"},
            "reporting": {
                "generate_candidate_reports": True,
                "generate_umap_plot":         False,  # skip matplotlib in CI
            },
        }

        from pipeline.modules.m09_report import run
        run(cfg)

        # All expected output files must exist
        assert (out_dir / "priority_list.csv").exists(),   "priority_list.csv missing"
        assert (out_dir / "reserve_list.csv").exists(),    "reserve_list.csv missing"
        assert (out_dir / "eliminated_log.csv").exists(),  "eliminated_log.csv missing"
        assert (out_dir / "pipeline_summary.json").exists(),"pipeline_summary.json missing"

        # Summary must have reasonable counts
        import json as jsonlib
        summary = jsonlib.loads((out_dir / "pipeline_summary.json").read_text())
        assert summary["priority_count"] >= 1
        assert summary["eliminated_count"] >= 1
        assert "pipeline_version" in summary

        # Per-candidate reports for priority endolysins
        report_dir = out_dir / "candidate_reports"
        assert report_dir.exists(), "candidate_reports/ directory missing"
        priority_endos = [
            c for c in candidates
            if c.track == "endolysin" and c.final_status == "priority"
        ]
        for c in priority_endos:
            rpt = report_dir / f"{c.candidate_id}.md"
            assert rpt.exists(), f"Report missing for {c.candidate_id}"
            content = rpt.read_text()
            assert c.candidate_id in content
            assert "## Selection" in content
            assert "## Sequence" in content


class TestActivelearningIntegration:
    """M10/M11 active learning loop with synthetic wet lab data."""

    def test_classifier_trains_and_predicts(self, tmp_path):
        import csv
        from pipeline.utils.data_model import save_candidates

        rng_py = random.Random(42)

        # Synthetic candidates
        candidates = make_synthetic_candidates(n_endolysins=30, n_holins=0, n_spanins=0)
        for c in candidates:
            c.final_status    = "reserve"
            c.gate1_status    = "pass"
            c.length_aa       = len(c.sequence)

        cand_dir = tmp_path / "03_gate1"
        cand_dir.mkdir()
        save_candidates(candidates, str(cand_dir / "candidates_passing.json"))

        # Synthetic embeddings (small dim for speed)
        matrix, index = make_synthetic_embeddings(candidates, dim=32, seed=7)
        emb_dir = tmp_path / "04_embeddings"
        emb_dir.mkdir()
        np.save(emb_dir / "embedding_matrix.npy", matrix)
        (emb_dir / "embedding_index.json").write_text(json.dumps(index))

        # Write synthetic wet lab results (first 15 candidates, binary labels)
        wetlab_path = tmp_path / "round1.csv"
        with open(wetlab_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "pathogen_id", "active"])
            for i, cid in enumerate(index[:15]):
                writer.writerow([cid, "vibrio_harveyi", 1 if i < 8 else 0])
                writer.writerow([cid, "strep_parauberis", 1 if i % 3 == 0 else 0])

        # Write config
        import yaml
        cfg_path = tmp_path / "config.yaml"
        cfg = {
            "paths": {
                "intermediate_dir": str(tmp_path),
                "output_dir":       str(tmp_path / "output"),
                "log_dir":          str(tmp_path / "logs"),
            }
        }
        cfg_path.write_text(yaml.dump(cfg))

        al_dir = tmp_path / "active_learning"
        al_dir.mkdir()

        # Train
        from pipeline.active_learning.train import train
        meta = train(
            results_csv = str(wetlab_path),
            config_path = str(cfg_path),
            output_dir  = str(al_dir),
            append      = False,
        )

        assert len(meta) == 2, f"Expected 2 pathogen models, got {len(meta)}"
        assert "vibrio_harveyi" in meta
        assert "strep_parauberis" in meta
        assert (al_dir / "models" / "vibrio_harveyi.pkl").exists()

        # Predict / re-rank
        from pipeline.active_learning.predict import predict_and_rerank
        predict_and_rerank(
            config_path = str(cfg_path),
            al_dir      = str(al_dir),
            output_dir  = str(al_dir),
        )

        reranked_path = al_dir / "round2_candidates.csv"
        assert reranked_path.exists(), "round2_candidates.csv not written"

        # Read and validate
        with open(reranked_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) > 0, "No rows in round2_candidates.csv"
        assert "composite_al_score" in rows[0]
        assert "prob_vibrio_harveyi" in rows[0]

        # Scores must be sorted descending
        scores = [float(r["composite_al_score"]) for r in rows]
        assert scores == sorted(scores, reverse=True), \
            "round2_candidates.csv not sorted by composite_al_score"
