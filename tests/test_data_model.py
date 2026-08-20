"""tests/test_data_model.py — Unit tests for data model serialization."""
import json
import tempfile
from pathlib import Path
import pytest

from pipeline.utils.data_model import (
    EndolysínRecord, HolinRecord, SpanínRecord, LysisModule,
    save_candidates, load_candidates, save_modules, load_modules, split_by_track,
)


def _make_endolysin(cid: str = "genome1__locus01") -> EndolysínRecord:
    return EndolysínRecord(
        candidate_id   = cid,
        genome_id      = "genome1",
        protein_id     = "locus01",
        sequence       = "MKLSTLEKQLV" * 10,
        nucleotide_seq = "ATG" * 30,
    )


def _make_holin(cid: str = "genome1__locus02") -> HolinRecord:
    return HolinRecord(
        candidate_id = cid,
        genome_id    = "genome1",
        protein_id   = "locus02",
        sequence     = "MWLLAIIIAG" * 8,
        n_tm_helices = 2,
    )


def _make_spanin(cid: str = "genome1__locus03") -> SpanínRecord:
    return SpanínRecord(
        candidate_id = cid,
        genome_id    = "genome1",
        protein_id   = "locus03",
        sequence     = "MKRNAVLLLG" * 6,
        spanin_type  = "i_spanin",
    )


class TestEndolysínRecord:
    def test_default_track(self):
        c = _make_endolysin()
        assert c.track == "endolysin"

    def test_pg_compatibility_roundtrip(self):
        c = _make_endolysin()
        scores = {"vibrio_harveyi": 2, "strep_parauberis": 1}
        c.set_pg_compatibility(scores)
        assert c.get_pg_compatibility() == scores

    def test_pathogen_scores_roundtrip(self):
        c = _make_endolysin()
        scores = {"vibrio_harveyi": 0.87, "strep_parauberis": 0.43}
        c.set_pathogen_scores(scores)
        assert c.get_pathogen_scores() == scores

    def test_catalytic_domain_type_chap(self):
        c = _make_endolysin()
        c.has_chap = True
        assert c.catalytic_domain_type == "CHAP"

    def test_catalytic_domain_type_amidase(self):
        c = _make_endolysin()
        c.has_amidase = True
        assert c.catalytic_domain_type == "amidase"

    def test_to_dict_roundtrip(self):
        c = _make_endolysin()
        d = c.to_dict()
        c2 = EndolysínRecord.from_dict(d)
        assert c2.candidate_id == c.candidate_id
        assert c2.track == "endolysin"

    def test_repr(self):
        c = _make_endolysin()
        r = repr(c)
        assert "endolysin" in r
        assert "genome1__locus01" in r


class TestHolinRecord:
    def test_default_track(self):
        h = _make_holin()
        assert h.track == "holin"

    def test_fields_persist(self):
        h = _make_holin()
        h.n_tm_helices = 3
        h.holin_class  = "class_I"
        d  = h.to_dict()
        h2 = HolinRecord.from_dict(d)
        assert h2.n_tm_helices == 3
        assert h2.holin_class  == "class_I"


class TestSpanínRecord:
    def test_default_track(self):
        s = _make_spanin()
        assert s.track == "spanin"

    def test_spanin_type_persists(self):
        s = _make_spanin()
        d  = s.to_dict()
        s2 = SpanínRecord.from_dict(d)
        assert s2.spanin_type == "i_spanin"


class TestLysisModule:
    def test_component_ids_full(self):
        mod = LysisModule(
            module_id    = "genome1__mod0001",
            genome_id    = "genome1",
            prophage_id  = "genome1",
            endolysin_id = "genome1__locus01",
            holin_id     = "genome1__locus02",
            ispanin_id   = "genome1__locus03",
        )
        ids = mod.component_ids()
        assert "genome1__locus01" in ids
        assert "genome1__locus02" in ids
        assert "genome1__locus03" in ids

    def test_component_ids_partial(self):
        mod = LysisModule(
            module_id    = "genome1__mod0001",
            genome_id    = "genome1",
            prophage_id  = "genome1",
            endolysin_id = "genome1__locus01",
        )
        ids = mod.component_ids()
        assert len(ids) == 1


class TestSerialization:
    def test_save_load_candidates_mixed_tracks(self, tmp_path):
        candidates = [
            _make_endolysin("g1__e1"),
            _make_holin("g1__h1"),
            _make_spanin("g1__s1"),
        ]
        path = str(tmp_path / "candidates.json")
        save_candidates(candidates, path)
        loaded = load_candidates(path)

        assert len(loaded) == 3
        tracks = {c.track for c in loaded}
        assert tracks == {"endolysin", "holin", "spanin"}

    def test_save_load_candidates_preserves_types(self, tmp_path):
        candidates = [
            _make_endolysin("g1__e1"),
            _make_holin("g1__h1"),
        ]
        path = str(tmp_path / "candidates.json")
        save_candidates(candidates, path)
        loaded = load_candidates(path)

        endos = [c for c in loaded if c.track == "endolysin"]
        hols  = [c for c in loaded if c.track == "holin"]
        assert isinstance(endos[0], EndolysínRecord)
        assert isinstance(hols[0], HolinRecord)

    def test_save_load_modules(self, tmp_path):
        modules = [
            LysisModule(
                module_id    = "g1__mod0001",
                genome_id    = "g1",
                prophage_id  = "g1",
                endolysin_id = "g1__e1",
                holin_id     = "g1__h1",
                completeness = "partial",
            )
        ]
        path = str(tmp_path / "modules.json")
        save_modules(modules, path)
        loaded = load_modules(path)
        assert len(loaded) == 1
        assert loaded[0].module_id == "g1__mod0001"
        assert loaded[0].endolysin_id == "g1__e1"

    def test_split_by_track(self):
        candidates = [
            _make_endolysin("g1__e1"),
            _make_endolysin("g1__e2"),
            _make_holin("g1__h1"),
            _make_spanin("g1__s1"),
        ]
        tracks = split_by_track(candidates)
        assert len(tracks["endolysin"]) == 2
        assert len(tracks["holin"])     == 1
        assert len(tracks["spanin"])    == 1
