"""
pg_database.py
==============
Peptidoglycan chemistry database for target pathogen specificity scoring.

The database is read from targets/pathogen_db.yaml.
New targets are added via CLI: prophage_lysis add-target ...
No source code changes needed to add new pathogens.

Compatibility scoring logic:
  - Each endolysin catalytic domain type has a known substrate range
  - Each pathogen has a PG chemotype (DAP-type gram-negative, Lys-type gram-positive, etc.)
  - Score 0 = incompatible, 1 = low probability, 2 = high probability

Score definitions:
  2 = catalytic domain has strong published activity against this PG chemotype
  1 = catalytic domain has reported but variable/weak activity
  0 = no known activity or chemically incompatible
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml

log = logging.getLogger("pg_database")


# ── Hardcoded compatibility table (domain_type × pg_chemotype → score) ────────
# Based on published endolysin biochemistry literature.
# This table is internal — the pathogen DB is external and user-extensible.

_COMPATIBILITY_TABLE: Dict[str, Dict[str, int]] = {
    # PG chemotype → score per catalytic domain type
    # Gram-negative DAP-type (Vibrio, Edwardsiella, Aeromonas, E. coli)
    "gram_negative_dap": {
        "lysozyme":          2,   # cleaves MurNAc-GlcNAc — active on all DAP-type
        "amidase":           2,   # cleaves stem peptide — active gram-negative
        "CHAP":              1,   # primarily gram-positive activity, variable gram-neg
        "glucosaminidase":   2,   # cleaves GlcNAc — active gram-negative
        "transglycosylase":  2,   # lytic transglycosylase — gram-negative specialist
        "endolysin_other":   1,
        "unknown":           0,
    },
    # Gram-positive Lys-type (Streptococcus, Staphylococcus — no outer membrane)
    "gram_positive_lys": {
        "lysozyme":          1,   # active but outer membrane absent — direct access
        "amidase":           2,   # very active on gram-positive PG
        "CHAP":              2,   # cysteine histidine-dep — gram-positive specialist
        "glucosaminidase":   1,
        "transglycosylase":  0,   # primarily gram-negative
        "endolysin_other":   1,
        "unknown":           0,
    },
    # Gram-positive DAP-type (Listeria, Bacillus)
    "gram_positive_dap": {
        "lysozyme":          2,
        "amidase":           2,
        "CHAP":              1,
        "glucosaminidase":   1,
        "transglycosylase":  0,
        "endolysin_other":   1,
        "unknown":           0,
    },
    # Gram-negative DAP-type with outer membrane (same as gram_negative_dap
    # but exogenous application requires OM permeabilization — flag only)
    "gram_negative_dap_om_barrier": {
        "lysozyme":          2,
        "amidase":           2,
        "CHAP":              1,
        "glucosaminidase":   2,
        "transglycosylase":  2,
        "endolysin_other":   1,
        "unknown":           0,
    },
}


# ── Pathogen database loader ──────────────────────────────────────────────────

class PathogenDatabase:
    """
    Loads and manages the target pathogen database from YAML.
    Provides compatibility scoring for endolysin candidates.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.pathogens: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.db_path.exists():
            log.warning(f"Pathogen DB not found: {self.db_path} — using empty DB")
            return
        with open(self.db_path) as f:
            data = yaml.safe_load(f)
        self.pathogens = {p["id"]: p for p in data.get("pathogens", [])}
        log.info(
            f"Loaded {len(self.pathogens)} target pathogens from "
            f"{self.db_path.name}"
        )

    def pathogen_ids(self) -> List[str]:
        return list(self.pathogens.keys())

    def get_pg_chemotype(self, pathogen_id: str) -> Optional[str]:
        p = self.pathogens.get(pathogen_id)
        return p["pg_chemotype"] if p else None

    def score_endolysin(
        self,
        pathogen_id:      str,
        catalytic_domain: str,
    ) -> int:
        """
        Return compatibility score (0/1/2) for an endolysin's catalytic
        domain against a specific target pathogen.
        0 = incompatible, 1 = possible, 2 = likely
        """
        pg_type = self.get_pg_chemotype(pathogen_id)
        if pg_type is None:
            log.debug(f"Unknown pathogen: {pathogen_id}")
            return 0
        table = _COMPATIBILITY_TABLE.get(pg_type, {})
        return table.get(catalytic_domain, 0)

    def score_all_pathogens(
        self,
        catalytic_domain: str,
    ) -> Dict[str, int]:
        """
        Return compatibility scores against all pathogens in the DB.
        Returns dict: pathogen_id → score
        """
        return {
            pid: self.score_endolysin(pid, catalytic_domain)
            for pid in self.pathogens
        }

    def add_target(
        self,
        pathogen_id:    str,
        display_name:   str,
        species:        str,
        gram_stain:     str,    # "positive" | "negative"
        pg_chemotype:   str,    # must be a key in _COMPATIBILITY_TABLE
        aquaculture_host: str,
        notes:          str = "",
    ) -> None:
        """
        Add a new target pathogen to the DB and save to disk.
        This is called by the CLI `add-target` command.
        """
        if pg_chemotype not in _COMPATIBILITY_TABLE:
            valid = list(_COMPATIBILITY_TABLE.keys())
            raise ValueError(
                f"Unknown PG chemotype '{pg_chemotype}'. "
                f"Valid options: {valid}"
            )

        entry = {
            "id":               pathogen_id,
            "display_name":     display_name,
            "species":          species,
            "gram_stain":       gram_stain,
            "pg_chemotype":     pg_chemotype,
            "aquaculture_host": aquaculture_host,
            "notes":            notes,
        }
        self.pathogens[pathogen_id] = entry
        self._save()
        log.info(f"Added target pathogen: {pathogen_id} ({display_name})")

    def _save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"pathogens": list(self.pathogens.values())}
        with open(self.db_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        log.info(f"Pathogen DB saved: {self.db_path}")

    def coverage_summary(
        self,
        candidates_pg_scores: List[Dict[str, int]],
    ) -> Dict[str, int]:
        """
        Given a list of per-candidate PG score dicts, return count of
        candidates with score >= 2 per pathogen.
        Used in M08 to check coverage saturation.
        """
        summary: Dict[str, int] = {pid: 0 for pid in self.pathogens}
        for scores in candidates_pg_scores:
            for pid, score in scores.items():
                if pid in summary and score >= 2:
                    summary[pid] += 1
        return summary

    def uncovered_pathogens(
        self,
        candidates_pg_scores: List[Dict[str, int]],
        min_coverage: int = 2,
    ) -> List[str]:
        """Return pathogen IDs with fewer than min_coverage high-score candidates."""
        cov = self.coverage_summary(candidates_pg_scores)
        return [pid for pid, count in cov.items() if count < min_coverage]
