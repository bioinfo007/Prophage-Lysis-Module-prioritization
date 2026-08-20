"""
tests/test_m02_fixtures.py
===========================
Tests for M02 lysis module identification using fixture files.
No HMMER, no Pharokka — HMMER hits are synthesized directly.

These tests verify:
  - Keyword-based identification picks up the fixture lysis genes
  - Domain-based identification works with mock hits
  - Module linkage connects holin + endolysin from same genomic region
  - SAR endolysin detection does not misclassify fixture holins
  - Spanin classification from topology
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.utils.data_model import (
    EndolysínRecord, HolinRecord, SpanínRecord, LysisModule,
    split_by_track,
)
from pipeline.utils.hmmer import (
    classify_protein, is_sar_endolysin, catalytic_domain_type,
    predict_tm_helices_simple, predict_signal_peptide_heuristic,
    has_endolysin_catalytic, has_holin_domain, has_spanin_domain,
    has_cbd, ENDOLYSIN_CATALYTIC_DOMAINS, HOLIN_DOMAINS, SPANIN_DOMAINS,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ── Load fixtures ─────────────────────────────────────────────────────────────

def load_fixture_proteins():
    """Parse the fixture .faa file into a dict of id → sequence."""
    faa_path = FIXTURE_DIR / "test_phage_001.faa"
    proteins = {}
    current_id, current_seq = None, []
    for line in faa_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id:
                proteins[current_id] = "".join(current_seq)
            parts = line[1:].split()
            current_id = parts[0]
            current_seq = []
        else:
            current_seq.append(line)
    if current_id:
        proteins[current_id] = "".join(current_seq)
    return proteins


def load_fixture_annotations():
    import csv
    ann_path = FIXTURE_DIR / "test_phage_001_annotations.tsv"
    out = {}
    with open(ann_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out[row["candidate_id"]] = dict(row)
    return out


def load_fixture_nuc_lookup():
    nuc_path = FIXTURE_DIR / "test_phage_001_nuc_lookup.json"
    return json.loads(nuc_path.read_text())


# ── Mock HMMER hit builders ───────────────────────────────────────────────────

def _mock_hit(accession: str, evalue: float = 1e-20, start: int = 1, end: int = 100):
    from pipeline.utils.hmmer import _classify_accession
    return {
        "accession":   accession,
        "name":        accession,
        "description": f"test {accession}",
        "evalue":      evalue,
        "score":       80.0,
        "start":       start,
        "end":         end,
        "domain_type": _classify_accession(accession),
    }


ENDOLYSIN_HITS = [_mock_hit("PF04851")]   # CHAP
HOLIN_HITS     = [_mock_hit("PF04531")]   # Holin_LLH
SPANIN_HITS    = [_mock_hit("PF11551")]   # Rz (i-spanin)
NO_HITS        = []


# ── Fixture loading tests ─────────────────────────────────────────────────────

class TestFixtureFiles:
    def test_fixture_proteins_loadable(self):
        proteins = load_fixture_proteins()
        assert len(proteins) == 10, f"Expected 10 proteins, got {len(proteins)}"

    def test_fixture_lysis_proteins_present(self):
        proteins = load_fixture_proteins()
        assert "test_phage_001__lysin_001" in proteins
        assert "test_phage_001__holin_001" in proteins
        assert "test_phage_001__spanin_001" in proteins

    def test_fixture_annotations_loadable(self):
        ann = load_fixture_annotations()
        assert len(ann) == 10

    def test_fixture_nuc_lookup_loadable(self):
        nuc = load_fixture_nuc_lookup()
        assert "test_phage_001__lysin_001" in nuc
        cds = nuc["test_phage_001__lysin_001"]
        assert len(cds) % 3 == 0, "CDS length not multiple of 3"
        assert cds.startswith("ATG"), "CDS does not start with ATG"

    def test_fixture_endolysin_length_plausible(self):
        proteins = load_fixture_proteins()
        for cid in ["test_phage_001__lysin_001", "test_phage_001__lysin_002"]:
            seq = proteins[cid]
            assert 80 <= len(seq) <= 700, \
                f"{cid}: length {len(seq)} outside expected endolysin range"

    def test_fixture_holin_length_plausible(self):
        proteins = load_fixture_proteins()
        for cid in ["test_phage_001__holin_001", "test_phage_001__holin_002"]:
            seq = proteins[cid]
            assert 20 <= len(seq) <= 300, \
                f"{cid}: length {len(seq)} outside expected holin range"


# ── TM topology of fixture proteins ──────────────────────────────────────────

class TestFixtureTMTopology:
    def test_holins_have_tm_helices(self):
        proteins = load_fixture_proteins()
        for cid in ["test_phage_001__holin_001", "test_phage_001__holin_002"]:
            seq = proteins[cid]
            n_tm, tm_pos = predict_tm_helices_simple(seq)
            assert n_tm >= 1, \
                f"{cid}: expected ≥1 TM helix, got {n_tm} (GRAVY-based prediction)"

    def test_endolysins_mostly_no_tm(self):
        proteins = load_fixture_proteins()
        for cid in ["test_phage_001__lysin_001", "test_phage_001__lysin_002"]:
            seq = proteins[cid]
            n_tm, _ = predict_tm_helices_simple(seq)
            # Endolysins may have 0-1 TM helices (SAR endolysins have 1)
            assert n_tm <= 1, \
                f"{cid}: {n_tm} TM helices — looks like a holin, not endolysin"

    def test_ispanin_has_one_tm(self):
        proteins = load_fixture_proteins()
        seq = proteins["test_phage_001__spanin_001"]
        n_tm, _ = predict_tm_helices_simple(seq)
        # i-spanin should have 1 TM helix (the membrane anchor)
        assert n_tm >= 1, "i-spanin fixture has no TM helix — sequence may need fixing"

    def test_noise_proteins_no_tm_or_variable(self):
        """Structural proteins may have TM or not — just ensure no crash."""
        proteins = load_fixture_proteins()
        for cid in ["test_phage_001__orf_001", "test_phage_001__orf_002"]:
            seq = proteins[cid]
            n_tm, tm_pos = predict_tm_helices_simple(seq)
            assert isinstance(n_tm, int)
            assert isinstance(tm_pos, list)


# ── Domain-based classification with mock hits ────────────────────────────────

class TestMockDomainClassification:
    def setup_method(self):
        self.proteins = load_fixture_proteins()

    def test_endolysin_classified_by_chap_domain(self):
        seq = self.proteins["test_phage_001__lysin_001"]
        n_tm, tm_pos = predict_tm_helices_simple(seq)
        classification = classify_protein(ENDOLYSIN_HITS, n_tm, tm_pos, seq)
        assert classification == "endolysin", \
            f"Expected endolysin, got {classification}"

    def test_holin_classified_by_holin_domain(self):
        seq = self.proteins["test_phage_001__holin_001"]
        n_tm, tm_pos = predict_tm_helices_simple(seq)
        classification = classify_protein(HOLIN_HITS, n_tm, tm_pos, seq)
        assert classification == "holin"

    def test_spanin_classified_by_spanin_domain(self):
        seq = self.proteins["test_phage_001__spanin_001"]
        n_tm, tm_pos = predict_tm_helices_simple(seq)
        classification = classify_protein(SPANIN_HITS, n_tm, tm_pos, seq)
        assert classification == "spanin"

    def test_holin_not_misclassified_as_sar(self):
        """Holins with 2+ TM helices must NOT be SAR endolysins."""
        seq = self.proteins["test_phage_001__holin_001"]
        n_tm, tm_pos = predict_tm_helices_simple(seq)
        # Holins have holin domain, not endolysin catalytic — so SAR check fails
        sar = is_sar_endolysin(HOLIN_HITS, seq, n_tm, tm_pos)
        assert not sar, "Holin incorrectly classified as SAR endolysin"

    def test_chap_domain_type(self):
        assert catalytic_domain_type(ENDOLYSIN_HITS) == "CHAP"

    def test_no_hits_unknown(self):
        seq = self.proteins["test_phage_001__orf_001"]
        n_tm, tm_pos = predict_tm_helices_simple(seq)
        classification = classify_protein(NO_HITS, n_tm, tm_pos, seq)
        assert classification == "unknown"


# ── Keyword-based identification (as in M02) ──────────────────────────────────

class TestKeywordIdentification:
    """Verify that M02's keyword matching picks up fixture lysis genes."""

    LYSIS_KEYWORDS = {
        "lysis", "endolysin", "lysin", "lysozyme", "muramidase",
        "amidase", "peptidoglycan", "murein", "chap", "autolysin",
        "transglycosylase", "glucosaminidase", "bacteriolytic",
    }
    HOLIN_KEYWORDS = {
        "holin", "hole-forming", "phage holin", "membrane hole",
    }
    SPANIN_KEYWORDS = {
        "spanin", "rz", "rz1", "i-spanin", "o-spanin",
    }

    def _kw_match(self, func: str, keywords: set) -> bool:
        return any(kw in func.lower() for kw in keywords)

    def test_endolysins_matched_by_keyword(self):
        ann = load_fixture_annotations()
        for cid in ["test_phage_001__lysin_001", "test_phage_001__lysin_002"]:
            func = ann[cid]["function"]
            assert self._kw_match(func, self.LYSIS_KEYWORDS), \
                f"{cid}: function '{func}' not matched by any lysis keyword"

    def test_holins_matched_by_keyword(self):
        ann = load_fixture_annotations()
        for cid in ["test_phage_001__holin_001", "test_phage_001__holin_002"]:
            func = ann[cid]["function"]
            assert self._kw_match(func, self.HOLIN_KEYWORDS), \
                f"{cid}: function '{func}' not matched by any holin keyword"

    def test_spanins_matched_by_keyword(self):
        ann = load_fixture_annotations()
        for cid in ["test_phage_001__spanin_001", "test_phage_001__spanin_002"]:
            func = ann[cid]["function"]
            assert self._kw_match(func, self.SPANIN_KEYWORDS), \
                f"{cid}: function '{func}' not matched by any spanin keyword"

    def test_noise_not_matched(self):
        ann = load_fixture_annotations()
        for cid in ["test_phage_001__orf_001", "test_phage_001__orf_002"]:
            func = ann[cid]["function"]
            is_endo  = self._kw_match(func, self.LYSIS_KEYWORDS)
            is_holin = self._kw_match(func, self.HOLIN_KEYWORDS)
            is_span  = self._kw_match(func, self.SPANIN_KEYWORDS)
            assert not (is_endo or is_holin or is_span), \
                f"{cid}: noise protein '{func}' incorrectly matched as lysis gene"


# ── Module linkage logic ──────────────────────────────────────────────────────

class TestModuleLinkage:
    """Test the genomic proximity-based module linkage from M02."""

    def test_holin_endolysin_within_window(self):
        """Holin at pos 900 and endolysin at pos 1000 are within 10 ORFs."""
        ann = load_fixture_annotations()

        holin_start = int(ann["test_phage_001__holin_001"]["start"])
        endo_start  = int(ann["test_phage_001__lysin_001"]["start"])

        # Both are in the same lysis cluster — positions close
        assert abs(endo_start - holin_start) < 2000, \
            "Holin and endolysin fixture are too far apart for module linkage test"

    def test_spanin_between_holin_and_endolysin(self):
        """Spanin at pos 1990 is after endolysin at 1000 — still linkable."""
        ann = load_fixture_annotations()
        endo_end   = int(ann["test_phage_001__lysin_001"]["end"])
        span_start = int(ann["test_phage_001__spanin_001"]["start"])
        assert span_start > endo_end, \
            "Spanin should be downstream of endolysin in fixture"

    def test_module_component_ids(self):
        """LysisModule.component_ids() returns all non-None component IDs."""
        mod = LysisModule(
            module_id    = "test_phage_001__mod0001",
            genome_id    = "test_phage_001",
            prophage_id  = "test_phage_001",
            endolysin_id = "test_phage_001__lysin_001",
            holin_id     = "test_phage_001__holin_001",
            ispanin_id   = "test_phage_001__spanin_001",
            completeness = "complete",
        )
        ids = mod.component_ids()
        assert len(ids) == 3
        assert "test_phage_001__lysin_001" in ids
        assert "test_phage_001__holin_001" in ids
        assert "test_phage_001__spanin_001" in ids


# ── CAI from fixture nucleotide sequences ─────────────────────────────────────

class TestFixtureCAI:
    def test_cai_computed_from_fixture_nuc(self):
        from pipeline.utils.cai import compute_cai
        nuc = load_fixture_nuc_lookup()
        for cid, seq in nuc.items():
            cai = compute_cai(seq, fallback=0.5)
            assert 0.0 <= cai <= 1.0, \
                f"{cid}: CAI out of range: {cai}"

    def test_ctg_rich_sequence_high_cai(self):
        """CTG (Leu) is the optimal E. coli codon — should give high CAI."""
        from pipeline.utils.cai import compute_cai
        # Build a CDS from optimal E. coli codons only
        optimal_cds = "ATG" + "CTG" * 50 + "TAA"   # Met + 50×Leu(CTG) + stop
        cai = compute_cai(optimal_cds)
        assert cai > 0.85, f"Optimal E. coli codon sequence should give CAI > 0.85, got {cai}"

    def test_fixture_nuc_cai_above_zero(self):
        from pipeline.utils.cai import compute_cai
        nuc = load_fixture_nuc_lookup()
        for cid, seq in nuc.items():
            cai = compute_cai(seq, fallback=0.0)
            assert cai > 0.0, f"{cid}: CAI = 0 — likely all N codons or empty"
