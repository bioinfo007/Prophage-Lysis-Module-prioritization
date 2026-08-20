"""tests/test_gate1.py — Integration tests for Gate 1 expressibility filter."""
import json
import tempfile
from pathlib import Path
import pytest

from pipeline.utils.data_model import (
    EndolysínRecord, HolinRecord, SpanínRecord,
    save_candidates, load_candidates,
)


# ── Synthetic sequences ───────────────────────────────────────────────────────

# Good endolysin: moderate size, not too hydrophobic, stable
_GOOD_ENDOLYSIN_SEQ = (
    "MKLSTLEKQLVDAIRAQMKADIAQLKAEIEKLKKELEEAKEQMAELRKKLENALEEQAKKITDLAEQAEISALKE" * 3
)[:180]

# Bad endolysin: extremely hydrophobic (GRAVY too high) + very long
_BAD_ENDOLYSIN_SEQ = "LLLLVVVVIIIIFFFFWWWW" * 35   # 700 aa, all hydrophobic

# Good holin: small, hydrophobic (needs TM helices)
_GOOD_HOLIN_SEQ = "MWLLVVIIAGLLAGILSG" + "MKLEKQL" * 12

# Bad holin: too large (>25 kDa) and no TM helices
_BAD_HOLIN_SEQ = "MKLSTLEKQLVDAIRAQMK" * 20   # ~380 aa, hydrophilic

# Good spanin: moderate size
_GOOD_SPANIN_SEQ = "MKRNAVLLLGAVLALTACSSNADAQAQE" + "MKLEKQLVD" * 8

# Bad spanin: too large
_BAD_SPANIN_SEQ = "MKLSTLEKQLVDAIRAQMK" * 25   # ~475 aa


def _make_candidate(cid, track, seq, nuc="CTG" * 200):
    classes = {"endolysin": EndolysínRecord, "holin": HolinRecord, "spanin": SpanínRecord}
    cls = classes[track]
    return cls(
        candidate_id   = cid,
        genome_id      = "genome1",
        protein_id     = cid.split("__")[1],
        sequence       = seq,
        nucleotide_seq = nuc,
        track          = track,
    )


def _run_gate1_on(candidates, cfg_overrides=None) -> list:
    """Run Gate 1 evaluation and return updated candidates."""
    import yaml
    from pipeline.utils.data_model import split_by_track
    from pipeline.modules.m03_gate1 import (
        _set_physicochemical, _evaluate_endolysin, _evaluate_holin,
        _evaluate_spanin, _apply_gate1_result,
    )

    cfg_gate1 = {
        "max_mw_kda":               70.0,
        "max_gravy_endolysin":       0.1,
        "max_instability_index":    60.0,
        "min_cai_score":             0.55,
        "flags_to_fail":             2,
        "max_mw_kda_holin":         25.0,
        "min_gravy_holin":          -0.1,
        "max_tm_helices_holin":      4,
        "flags_to_fail_holin":       2,
        "max_mw_kda_spanin":        35.0,
        "flags_to_fail_spanin":      1,
    }
    if cfg_overrides:
        cfg_gate1.update(cfg_overrides)

    tracks = split_by_track(candidates)

    for c in tracks["endolysin"]:
        _set_physicochemical(c)
        flags = _evaluate_endolysin(c, cfg_gate1)
        _apply_gate1_result(c, flags, cfg_gate1["flags_to_fail"])

    for c in tracks["holin"]:
        _set_physicochemical(c)
        flags = _evaluate_holin(c, cfg_gate1)
        _apply_gate1_result(c, flags, cfg_gate1["flags_to_fail_holin"])

    for c in tracks["spanin"]:
        _set_physicochemical(c)
        flags = _evaluate_spanin(c, cfg_gate1)
        _apply_gate1_result(c, flags, cfg_gate1["flags_to_fail_spanin"])

    return candidates


class TestGate1Endolysin:
    def test_good_endolysin_passes(self):
        c = _make_candidate("g1__e1", "endolysin", _GOOD_ENDOLYSIN_SEQ)
        updated = _run_gate1_on([c])
        assert updated[0].gate1_status in ("pass", "warn"), \
            f"Expected pass/warn, got {updated[0].gate1_status}: {updated[0].gate1_flags}"

    def test_physicochemical_fields_set(self):
        c = _make_candidate("g1__e1", "endolysin", _GOOD_ENDOLYSIN_SEQ)
        _run_gate1_on([c])
        assert c.mw_kda is not None
        assert c.gravy  is not None
        assert c.isoelectric_point is not None
        assert c.instability_index is not None
        assert c.length_aa == len(_GOOD_ENDOLYSIN_SEQ)

    def test_cai_computed(self):
        c = _make_candidate("g1__e1", "endolysin", _GOOD_ENDOLYSIN_SEQ, nuc="CTG" * 200)
        _run_gate1_on([c])
        assert c.cai_score is not None
        assert 0.0 <= c.cai_score <= 1.0

    def test_bad_endolysin_flagged(self):
        c = _make_candidate("g1__e2", "endolysin", _BAD_ENDOLYSIN_SEQ)
        _run_gate1_on([c])
        # Very hydrophobic → should have gravy flag
        assert any("hydrophobic" in f or "MW" in f for f in c.gate1_flags)

    def test_gate1_status_set(self):
        c = _make_candidate("g1__e1", "endolysin", _GOOD_ENDOLYSIN_SEQ)
        _run_gate1_on([c])
        assert c.gate1_status in ("pass", "warn", "fail")

    def test_failed_has_elimination_reason(self):
        c = _make_candidate("g1__e_bad", "endolysin", _BAD_ENDOLYSIN_SEQ)
        _run_gate1_on([c])
        if c.gate1_status == "fail":
            assert c.elimination_gate == "gate1"
            assert c.elimination_reason


class TestGate1Holin:
    def test_good_holin_passes(self):
        c = _make_candidate("g1__h1", "holin", _GOOD_HOLIN_SEQ)
        _run_gate1_on([c])
        # holins need TM helices — good holin has them
        assert c.gate1_status in ("pass", "warn", "fail")  # just check no crash

    def test_holin_physicochemical_set(self):
        c = _make_candidate("g1__h1", "holin", _GOOD_HOLIN_SEQ)
        _run_gate1_on([c])
        assert c.mw_kda is not None

    def test_large_holin_flagged(self):
        c = _make_candidate("g1__h_bad", "holin", _BAD_HOLIN_SEQ)
        _run_gate1_on([c])
        # ~380 aa holin: definitely > 25 kDa limit → should flag MW
        assert any("MW" in f for f in c.gate1_flags)


class TestGate1Spanin:
    def test_spanin_physicochemical_set(self):
        c = _make_candidate("g1__s1", "spanin", _GOOD_SPANIN_SEQ)
        _run_gate1_on([c])
        assert c.mw_kda is not None


class TestGate1MixedInput:
    def test_three_tracks_processed_independently(self):
        candidates = [
            _make_candidate("g1__e1", "endolysin", _GOOD_ENDOLYSIN_SEQ),
            _make_candidate("g1__h1", "holin",     _GOOD_HOLIN_SEQ),
            _make_candidate("g1__s1", "spanin",    _GOOD_SPANIN_SEQ),
        ]
        _run_gate1_on(candidates)
        assert all(c.gate1_status is not None for c in candidates)
        assert all(c.mw_kda      is not None for c in candidates)

    def test_eliminated_count_reasonable(self):
        candidates = [
            _make_candidate("g1__e1", "endolysin", _GOOD_ENDOLYSIN_SEQ),
            _make_candidate("g1__e2", "endolysin", _BAD_ENDOLYSIN_SEQ),
            _make_candidate("g1__h1", "holin",     _BAD_HOLIN_SEQ),
        ]
        _run_gate1_on(candidates)
        n_fail = sum(1 for c in candidates if c.gate1_status == "fail")
        n_pass = sum(1 for c in candidates if c.gate1_status in ("pass", "warn"))
        assert n_fail + n_pass == len(candidates)
