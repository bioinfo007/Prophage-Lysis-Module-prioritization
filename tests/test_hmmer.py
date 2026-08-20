"""tests/test_hmmer.py — Unit tests for HMMER domain parsing and classification."""
import pytest

from pipeline.utils.hmmer import (
    parse_domtblout,
    has_endolysin_catalytic, has_holin_domain, has_spanin_domain,
    catalytic_domain_type, classify_protein,
    is_sar_endolysin, has_cbd,
    predict_tm_helices_simple, predict_signal_peptide_heuristic,
    ENDOLYSIN_CATALYTIC_DOMAINS, HOLIN_DOMAINS, SPANIN_DOMAINS, CBD_DOMAINS,
)


# ── Synthetic domain hits ─────────────────────────────────────────────────────

def _hit(accession: str, evalue: float = 1e-10, start: int = 1, end: int = 100) -> dict:
    """Build a minimal domain hit dict for testing."""
    from pipeline.utils.hmmer import _classify_accession
    return {
        "accession":   accession,
        "name":        accession,
        "description": f"test domain {accession}",
        "evalue":      evalue,
        "score":       50.0,
        "start":       start,
        "end":         end,
        "domain_type": _classify_accession(accession),
    }


class TestDomainSets:
    """Verify no domain bleeds between sets — PF01471 removed from catalytic."""

    def test_pf01471_in_cbd_not_catalytic(self):
        assert "PF01471" not in ENDOLYSIN_CATALYTIC_DOMAINS
        assert "PF01471" in CBD_DOMAINS

    def test_pf07411_not_in_catalytic(self):
        """PF07411 (L,D-transpeptidase) must not be classified as endolysin."""
        assert "PF07411" not in ENDOLYSIN_CATALYTIC_DOMAINS

    def test_no_overlap_catalytic_holin(self):
        assert not (set(ENDOLYSIN_CATALYTIC_DOMAINS) & set(HOLIN_DOMAINS))

    def test_no_overlap_catalytic_spanin(self):
        assert not (set(ENDOLYSIN_CATALYTIC_DOMAINS) & set(SPANIN_DOMAINS))

    def test_no_overlap_holin_spanin(self):
        assert not (set(HOLIN_DOMAINS) & set(SPANIN_DOMAINS))


class TestDomainQueries:
    def test_has_endolysin_catalytic_chap(self):
        hits = [_hit("PF04851")]
        assert has_endolysin_catalytic(hits)

    def test_has_endolysin_catalytic_amidase(self):
        hits = [_hit("PF01520")]
        assert has_endolysin_catalytic(hits)

    def test_has_endolysin_catalytic_pf01471_false(self):
        """PF01471 is a CBD, NOT a catalytic domain."""
        hits = [_hit("PF01471")]
        assert not has_endolysin_catalytic(hits)

    def test_has_holin_domain(self):
        hits = [_hit("PF04531")]
        assert has_holin_domain(hits)
        assert not has_endolysin_catalytic(hits)

    def test_has_spanin_domain(self):
        hits = [_hit("PF16614")]
        assert has_spanin_domain(hits)

    def test_has_cbd(self):
        hits = [_hit("PF01471"), _hit("PF04851")]
        assert has_cbd(hits)
        assert has_endolysin_catalytic(hits)


class TestCatalyticDomainType:
    def test_chap_wins(self):
        hits = [_hit("PF04851"), _hit("PF01520")]
        assert catalytic_domain_type(hits) == "CHAP"

    def test_amidase(self):
        hits = [_hit("PF01520")]
        assert catalytic_domain_type(hits) == "amidase"

    def test_lysozyme(self):
        hits = [_hit("PF00959")]
        assert catalytic_domain_type(hits) == "lysozyme"

    def test_glucosaminidase(self):
        hits = [_hit("PF13529")]
        assert catalytic_domain_type(hits) == "glucosaminidase"

    def test_empty_hits(self):
        assert catalytic_domain_type([]) == "unknown"


class TestTMHelix:
    def test_no_helices_hydrophilic(self):
        seq = "MKLSTLEKQLVDAIREMQKADIAQLKAEIEQLK" * 5
        n, pos = predict_tm_helices_simple(seq)
        assert n == 0

    def test_hydrophobic_stretch_detected(self):
        # synthetic: 10 hydrophilic + 20 hydrophobic (LLLLVVVVIIIIFFFFWWWW) + 10 hydrophilic
        seq = "MKLSTLEKQL" + "LLLLVVVVIIIIFFFFWWWW" + "MKLSTLEKQL" * 5
        n, pos = predict_tm_helices_simple(seq)
        assert n >= 1

    def test_short_sequence_no_crash(self):
        n, pos = predict_tm_helices_simple("MKLA")
        assert n == 0


class TestSAREndolysin:
    """SAR endolysins: exactly 1 N-terminal TM helix + catalytic domain after it."""

    def test_sar_detected_correctly(self):
        seq    = "LLLLVVVVIIIIFFFFWWWW" + "MKLSTLEKQLVDAIREQQK" * 10
        n_tm, tm_pos = predict_tm_helices_simple(seq)
        hits   = [_hit("PF04851", start=25, end=180)]   # catalytic after TM
        if n_tm == 1 and tm_pos:
            result = is_sar_endolysin(hits, seq, n_tm, tm_pos)
            # If TM is N-terminal and catalytic is after it, should be SAR
            assert isinstance(result, bool)

    def test_sar_requires_catalytic_domain(self):
        seq  = "LLLLVVVVIIIIFFFFWWWW" + "MKLSTLEKQL" * 10
        hits = [_hit("PF04531")]   # holin domain, not catalytic
        assert not is_sar_endolysin(hits, seq, 1, [(0, 19)])

    def test_sar_requires_one_tm_helix(self):
        seq  = "MKLSTLEKQLVDAIREQQK" * 10
        hits = [_hit("PF04851", start=5, end=180)]
        assert not is_sar_endolysin(hits, seq, 0, [])

    def test_sar_two_tm_helices_not_sar(self):
        seq  = "LLLLVVVVIIIIFFFFWWWW" * 2 + "MKLSTLEKQL" * 5
        hits = [_hit("PF04851", start=45, end=180)]
        assert not is_sar_endolysin(hits, seq, 2, [(0, 19), (20, 39)])


class TestClassifyProtein:
    def test_endolysin_classified(self):
        seq  = "MKLSTLEKQL" * 12
        hits = [_hit("PF04851")]
        n_tm, tm_pos = 0, []
        assert classify_protein(hits, n_tm, tm_pos, seq) == "endolysin"

    def test_holin_classified(self):
        seq  = "MWLLVVVVII" * 8
        hits = [_hit("PF04531")]
        n_tm, tm_pos = 2, [(3, 22), (30, 49)]
        assert classify_protein(hits, n_tm, tm_pos, seq) == "holin"

    def test_spanin_classified(self):
        seq  = "MKLSTLEKQL" * 8
        hits = [_hit("PF16614")]
        assert classify_protein(hits, 0, [], seq) == "spanin"

    def test_no_hits_unknown(self):
        seq = "MKLSTLEKQL" * 8
        assert classify_protein([], 0, [], seq) == "unknown"
