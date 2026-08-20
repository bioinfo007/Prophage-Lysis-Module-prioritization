"""
data_model.py
=============
Central data objects for the prophage lysis module discovery pipeline.

Three parallel tracks:
  - EndolysínRecord  : catalytic PG-degrading enzyme
  - HolinRecord      : inner membrane permeabilizer
  - SpanínRecord     : outer membrane disruptor (gram-negative hosts)

Linked into LysisModule objects that travel through the pipeline together.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json


# ── Base record shared across all three tracks ────────────────────────────────

@dataclass
class _BaseRecord:
    # Identity
    candidate_id:       str            # unique: {genome_id}__{locus_tag}
    genome_id:          str            # source genome/prophage stem
    protein_id:         str            # Pharokka locus tag
    sequence:           str            # amino acid sequence
    nucleotide_seq:     str    = ""    # CDS nucleotide sequence (for real CAI)
    track:              str    = ""    # "endolysin" | "holin" | "spanin"

    # Source metadata
    source_organism:    str            = ""
    cds_start:          Optional[int]  = None
    cds_end:            Optional[int]  = None
    cds_strand:         Optional[str]  = None
    prophage_quality:   Optional[str]  = None  # intact/questionable/incomplete

    # Pharokka annotation
    pharokka_function:  Optional[str]  = None
    pharokka_category:  Optional[str]  = None
    genomic_context:    Optional[str]  = None  # comma-separated neighboring functions

    # HMMER domain annotation
    pfam_domains:       List[str]      = field(default_factory=list)
    pfam_descriptions:  List[str]      = field(default_factory=list)
    pfam_evalues:       List[float]    = field(default_factory=list)
    inclusion_reason:   Optional[str]  = None

    # Module linkage (set in M02)
    module_id:          Optional[str]  = None  # links records from same prophage locus
    module_complete:    bool           = False  # True if holin+endolysin+spanin all found

    # Gate 1 (M03)
    length_aa:          Optional[int]  = None
    mw_kda:             Optional[float]= None
    isoelectric_point:  Optional[float]= None
    gravy:              Optional[float]= None
    instability_index:  Optional[float]= None
    aromaticity:        Optional[float]= None
    cai_score:          Optional[float]= None
    gate1_flags:        List[str]      = field(default_factory=list)
    gate1_status:       Optional[str]  = None  # pass/warn/fail

    # TM topology — needed by Gate 1 for all three tracks
    n_tm_helices:       Optional[int]  = None   # number of predicted TM helices

    # ESM-2 embedding (M04)
    embedding_path:     Optional[str]  = None
    embedding_dim:      Optional[int]  = None

    # Clustering (M05)
    umap_x:             Optional[float]= None
    umap_y:             Optional[float]= None
    cluster_id:         Optional[int]  = None
    cluster_enrichment: Optional[str]  = None
    is_noise:           bool           = False

    # Redundancy collapse (M07)
    redundancy_cluster: Optional[int]  = None
    is_representative:  Optional[bool] = None
    similar_to:         Optional[str]  = None
    max_similarity:     Optional[float]= None
    gate3_status:       Optional[str]  = None  # representative/collapsed

    # Selection (M08)
    final_rank:         Optional[int]  = None
    final_status:       Optional[str]  = None  # priority/reserve/eliminated
    elimination_gate:   Optional[str]  = None
    elimination_reason: Optional[str]  = None
    diversity_rank:     Optional[int]  = None
    min_dist_at_selection: Optional[float] = None
    selection_strategy: Optional[str]  = None
    novelty_flag:       Optional[str]  = None  # novel/known_homolog
    closest_known:      Optional[str]  = None

    # Active learning scores (M10/M11) — one score per target pathogen
    # Stored as JSON string for dataclass simplicity: '{"vibrio_h": 0.87, ...}'
    pathogen_scores:    Optional[str]  = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_BaseRecord":
        valid = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def get_pathogen_scores(self) -> Dict[str, float]:
        if not self.pathogen_scores:
            return {}
        return json.loads(self.pathogen_scores)

    def set_pathogen_scores(self, scores: Dict[str, float]) -> None:
        self.pathogen_scores = json.dumps(scores)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.candidate_id}, "
                f"{len(self.sequence)}aa, track={self.track}, "
                f"status={self.final_status})")


# ── Endolysin-specific fields ─────────────────────────────────────────────────

@dataclass
class EndolysínRecord(_BaseRecord):
    track:              str    = "endolysin"

    # Domain flags
    has_chap:           bool   = False
    has_amidase:        bool   = False
    has_lysozyme:       bool   = False
    has_glucosaminidase:bool   = False
    has_transglycosylase:bool  = False
    has_cbd:            bool   = False   # cell wall binding domain
    is_sar_endolysin:   bool   = False   # signal-arrest-release type
    initial_class:      Optional[str] = None

    # PG chemistry matching (M06) — set per target pathogen
    # JSON: '{"vibrio_harveyi": 2, "strep_parauberis": 1, ...}'
    pg_compatibility:   Optional[str]  = None

    def get_pg_compatibility(self) -> Dict[str, int]:
        if not self.pg_compatibility:
            return {}
        return json.loads(self.pg_compatibility)

    def set_pg_compatibility(self, scores: Dict[str, int]) -> None:
        self.pg_compatibility = json.dumps(scores)

    @property
    def catalytic_domain_type(self) -> str:
        if self.has_chap:           return "CHAP"
        if self.has_amidase:        return "amidase"
        if self.has_lysozyme:       return "lysozyme"
        if self.has_glucosaminidase:return "glucosaminidase"
        if self.has_transglycosylase:return "transglycosylase"
        return "unknown"


# ── Holin-specific fields ─────────────────────────────────────────────────────

@dataclass
class HolinRecord(_BaseRecord):
    track:              str    = "holin"

    holin_class:        Optional[str]   = None  # class_I/class_II/SAR_holin
    has_signal_peptide: Optional[bool]  = None
    pinholin_score:     Optional[float] = None  # probability of being a pinholin


# ── Spanin-specific fields ────────────────────────────────────────────────────

@dataclass
class SpanínRecord(_BaseRecord):
    track:              str    = "spanin"

    spanin_type:        Optional[str]   = None  # i_spanin/o_spanin/u_spanin
    has_lipoprotein_signal: bool        = False
    partner_id:         Optional[str]   = None  # paired i/o spanin candidate_id


# ── Lysis module — links all three track records from one prophage locus ───────

@dataclass
class LysisModule:
    module_id:          str                         # unique per prophage locus
    genome_id:          str
    prophage_id:        str
    endolysin_id:       Optional[str]  = None       # candidate_id of endolysin
    holin_id:           Optional[str]  = None
    ispanin_id:         Optional[str]  = None
    ospanin_id:         Optional[str]  = None
    uspanin_id:         Optional[str]  = None
    completeness:       str            = "partial"  # partial/complete
    genomic_span_bp:    Optional[int]  = None
    phenotypic_activity: Optional[str] = None       # JSON from wet lab W3

    def component_ids(self) -> List[str]:
        return [x for x in [
            self.endolysin_id, self.holin_id,
            self.ispanin_id, self.ospanin_id, self.uspanin_id
        ] if x is not None]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "LysisModule":
        valid = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ── Serialization helpers ─────────────────────────────────────────────────────

_TRACK_CLS = {
    "endolysin": EndolysínRecord,
    "holin":     HolinRecord,
    "spanin":    SpanínRecord,
}


def load_candidates(path: str) -> List[_BaseRecord]:
    data = json.loads(open(path).read())
    out  = []
    for d in data:
        track = d.get("track", "endolysin")
        cls   = _TRACK_CLS.get(track, EndolysínRecord)
        out.append(cls.from_dict(d))
    return out


def save_candidates(candidates: List[_BaseRecord], path: str) -> None:
    data = [c.to_dict() for c in candidates]
    open(path, "w").write(json.dumps(data, indent=2))


def load_modules(path: str) -> List[LysisModule]:
    data = json.loads(open(path).read())
    return [LysisModule.from_dict(d) for d in data]


def save_modules(modules: List[LysisModule], path: str) -> None:
    data = [m.to_dict() for m in modules]
    open(path, "w").write(json.dumps(data, indent=2))


def split_by_track(
    candidates: List[_BaseRecord],
) -> Dict[str, List[_BaseRecord]]:
    out: Dict[str, List] = {"endolysin": [], "holin": [], "spanin": []}
    for c in candidates:
        if c.track in out:
            out[c.track].append(c)
    return out
