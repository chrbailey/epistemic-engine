"""
VERITY Core Types
=================

Foundational data structures for the VERITY belief system.

NOTE: Core epistemic types (MassFunction, DecomposedUncertainty, Uncertainty,
ConflictResult) are now imported from the shared `belief_math` package.
VERITY-specific types (Evidence, Claim, ProvenanceChain) remain here.

Key Design Decisions:
1. Immutable where possible (use frozen dataclasses)
2. Full provenance - every belief traces to evidence
3. Decomposed uncertainty - epistemic vs aleatoric
4. Dempster-Shafer mass functions - uncertainty intervals, not point estimates

References:
- Dempster-Shafer: Shafer, G. (1976). A Mathematical Theory of Evidence
- Decomposed Uncertainty: ICML 2024, "Decomposing uncertainty for LLMs"
- Provenance: SIGMOD 2019, "Data Provenance: Principles and Applications"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Optional, Literal, Tuple, FrozenSet
from enum import Enum
import hashlib

# =============================================================================
# IMPORT SHARED TYPES FROM belief_math
# =============================================================================
from belief_math import (
    MassFunction,
    DecomposedUncertainty,
    Uncertainty,
    ConflictResult,
)


# =============================================================================
# ENUMS (VERITY-SPECIFIC)
# =============================================================================

class EvidenceSource(Enum):
    """Type of evidence source with associated reliability priors."""
    DOCUMENT = "document"           # Structured document (contract, filing, etc.)
    HUMAN_EXPERT = "human_expert"   # Domain expert input
    HUMAN_USER = "human_user"       # Regular user input
    MODEL_OUTPUT = "model_output"   # LLM or ML model output
    INFERENCE = "inference"         # Derived from other evidence
    SENSOR = "sensor"               # IoT/external data feed
    DATABASE = "database"           # Structured data query result

    @property
    def default_reliability(self) -> float:
        """Prior reliability weight for this source type."""
        weights = {
            EvidenceSource.DOCUMENT: 0.9,
            EvidenceSource.HUMAN_EXPERT: 0.85,
            EvidenceSource.HUMAN_USER: 0.7,
            EvidenceSource.MODEL_OUTPUT: 0.6,
            EvidenceSource.INFERENCE: 0.5,
            EvidenceSource.SENSOR: 0.8,
            EvidenceSource.DATABASE: 0.95,
        }
        return weights.get(self, 0.5)


class ExtractionMethod(Enum):
    """How the claim was extracted from source."""
    EXACT = "exact"             # Verbatim quote
    PARAPHRASE = "paraphrase"   # Rephrased but semantically equivalent
    INFERENCE = "inference"     # Derived/inferred from source
    AGGREGATION = "aggregation" # Combined from multiple sources


class VerificationStatus(Enum):
    """Human verification status."""
    UNVERIFIED = "unverified"
    VERIFIED_TRUE = "verified_true"
    VERIFIED_FALSE = "verified_false"
    DISPUTED = "disputed"
    PENDING_REVIEW = "pending_review"


# =============================================================================
# EVIDENCE (VERITY-SPECIFIC)
# =============================================================================

@dataclass(frozen=True)
class Evidence:
    """
    Atomic unit of evidence with full provenance.

    Immutable - once created, evidence cannot be modified.
    Every belief update MUST cite Evidence objects.

    Example:
        evidence = Evidence(
            id="ev_001",
            source_type=EvidenceSource.DOCUMENT,
            source_id="contract_2024_001",
            raw_text="Payment due within 30 days of invoice.",
            extraction_method=ExtractionMethod.EXACT,
            location="Section 4.2, Paragraph 1",
            timestamp=datetime.now(),
        )
    """
    id: str
    source_type: EvidenceSource
    source_id: str                          # Document ID, user ID, model version
    raw_text: str                           # Original text
    extraction_method: ExtractionMethod
    timestamp: datetime

    # Location within source (optional but recommended)
    location: Optional[str] = None          # "Page 5, Paragraph 2" or "Line 145"

    # Reliability override (if different from source type default)
    reliability_override: Optional[float] = None

    # Verification
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verifier_id: Optional[str] = None
    verification_timestamp: Optional[datetime] = None
    verification_notes: Optional[str] = None

    # Hash for integrity checking
    content_hash: Optional[str] = None

    def __post_init__(self):
        # Compute content hash if not provided
        if self.content_hash is None:
            content = f"{self.source_id}:{self.raw_text}:{self.timestamp.isoformat()}"
            object.__setattr__(self, 'content_hash',
                             hashlib.sha256(content.encode()).hexdigest()[:16])

    @property
    def reliability(self) -> float:
        """Effective reliability score."""
        if self.reliability_override is not None:
            return self.reliability_override
        return self.source_type.default_reliability

    @property
    def is_verified(self) -> bool:
        return self.verification_status in {
            VerificationStatus.VERIFIED_TRUE,
            VerificationStatus.VERIFIED_FALSE
        }

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "raw_text": self.raw_text,
            "extraction_method": self.extraction_method.value,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "reliability": self.reliability,
            "verification_status": self.verification_status.value,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Evidence:
        return cls(
            id=data["id"],
            source_type=EvidenceSource(data["source_type"]),
            source_id=data["source_id"],
            raw_text=data["raw_text"],
            extraction_method=ExtractionMethod(data["extraction_method"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            location=data.get("location"),
            reliability_override=data.get("reliability_override"),
            verification_status=VerificationStatus(data.get("verification_status", "unverified")),
            content_hash=data.get("content_hash"),
        )


# =============================================================================
# CLAIM WITH PROVENANCE (VERITY-SPECIFIC)
# =============================================================================

@dataclass
class Claim:
    """
    A single atomic claim with decomposed uncertainty and full provenance.

    Unlike simple Bayesian beliefs, Claim tracks:
    1. WHERE the belief came from (evidence chain)
    2. WHY we're uncertain (epistemic vs aleatoric)
    3. HOW certain we should be (Dempster-Shafer interval)

    Uses shared types from belief_math:
    - MassFunction: Dempster-Shafer mass assignment
    - DecomposedUncertainty: Epistemic/aleatoric split

    Example:
        claim = Claim(
            id="claim_001",
            text="Defendant accessed the database on March 15, 2024",
            category="factual",
            frame={"true", "false"},
            mass_function=MassFunction(...),
            uncertainty=DecomposedUncertainty(...),
        )
    """
    id: str
    text: str
    category: str                           # "factual", "legal", "opinion", etc.

    # Uncertainty representation (from belief_math)
    frame: FrozenSet[str]                   # Possible values (often {"true", "false"})
    mass_function: MassFunction             # Dempster-Shafer mass
    uncertainty: DecomposedUncertainty      # Decomposed variance

    # Provenance
    evidence_ids: List[str] = field(default_factory=list)
    derived_from: List[str] = field(default_factory=list)  # Parent claim IDs
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Status
    anchored: bool = False                  # Human-validated, won't be overwritten
    anchor_evidence_id: Optional[str] = None

    @property
    def probability(self) -> float:
        """Point probability estimate (pignistic transformation)."""
        if "true" in self.frame:
            return self.mass_function.pignistic_probability("true")
        # For non-boolean claims, return mean
        return self.uncertainty.mean

    @property
    def belief_interval(self) -> Tuple[float, float]:
        """[Bel(true), Pl(true)] interval."""
        if "true" in self.frame:
            return self.mass_function.uncertainty_interval({"true"})
        return (self.uncertainty.mean, self.uncertainty.mean)

    @property
    def ignorance(self) -> float:
        """Width of belief interval (Pl - Bel)."""
        bel, pl = self.belief_interval
        return pl - bel

    def should_gather_evidence(self) -> bool:
        """Should we seek more evidence for this claim?"""
        return self.uncertainty.should_gather_more_evidence()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category,
            "frame": list(self.frame),
            "mass_function": {
                "frame": list(self.mass_function.frame),
                "masses": {",".join(sorted(k)): v for k, v in self.mass_function.masses.items()},
                "evidence_ids": self.mass_function.evidence_ids,
            },
            "uncertainty": {
                "mean": self.uncertainty.mean,
                "epistemic_variance": self.uncertainty.epistemic_variance,
                "aleatoric_variance": self.uncertainty.aleatoric_variance,
                "n_observations": self.uncertainty.n_observations,
                "evidence_ids": self.uncertainty.evidence_ids,
            },
            "evidence_ids": self.evidence_ids,
            "derived_from": self.derived_from,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "anchored": self.anchored,
            "anchor_evidence_id": self.anchor_evidence_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Claim:
        # Parse mass function
        mf_data = data["mass_function"]
        frame = frozenset(mf_data["frame"])
        masses = {
            frozenset(k.split(",")) if k else frozenset(): v
            for k, v in mf_data["masses"].items()
        }
        mass_function = MassFunction(
            frame=frame,
            masses=masses,
            evidence_ids=mf_data.get("evidence_ids", []),
        )

        # Parse uncertainty
        unc_data = data["uncertainty"]
        uncertainty = DecomposedUncertainty(
            mean=unc_data["mean"],
            epistemic_variance=unc_data["epistemic_variance"],
            aleatoric_variance=unc_data["aleatoric_variance"],
            n_observations=unc_data.get("n_observations", 0),
            evidence_ids=unc_data.get("evidence_ids", []),
        )

        return cls(
            id=data["id"],
            text=data["text"],
            category=data["category"],
            frame=frozenset(data["frame"]),
            mass_function=mass_function,
            uncertainty=uncertainty,
            evidence_ids=data.get("evidence_ids", []),
            derived_from=data.get("derived_from", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            anchored=data.get("anchored", False),
            anchor_evidence_id=data.get("anchor_evidence_id"),
        )

    @classmethod
    def create_boolean(
        cls,
        id: str,
        text: str,
        category: str = "factual",
        prior_true: float = 0.5,
    ) -> Claim:
        """Create a boolean claim with uniform prior."""
        frame = frozenset({"true", "false"})

        # Vacuous prior: all mass on frame (total ignorance)
        mass_function = MassFunction(
            frame=frame,
            masses={frame: 1.0},
        )

        uncertainty = DecomposedUncertainty.uniform_prior()

        return cls(
            id=id,
            text=text,
            category=category,
            frame=frame,
            mass_function=mass_function,
            uncertainty=uncertainty,
        )


# =============================================================================
# PROVENANCE CHAIN (VERITY-SPECIFIC)
# =============================================================================

@dataclass
class ProvenanceChain:
    """
    Complete provenance chain for a claim.

    Traces a belief back through all evidence and derivations.
    """
    claim_id: str

    # Direct evidence
    direct_evidence: List[Evidence]

    # Derived from (for inferred claims)
    parent_claims: List[str]
    derivation_method: Optional[str] = None  # "inference", "aggregation", etc.

    # Update history
    update_history: List[Dict] = field(default_factory=list)

    def audit_trail(self) -> str:
        """
        Human-readable audit trail.

        Format suitable for legal/regulatory review.
        """
        lines = [f"PROVENANCE CHAIN FOR CLAIM: {self.claim_id}", "=" * 60]

        if self.direct_evidence:
            lines.append("\nDIRECT EVIDENCE:")
            for i, ev in enumerate(self.direct_evidence, 1):
                lines.append(f"  [{i}] Source: {ev.source_type.value} ({ev.source_id})")
                lines.append(f"      Text: \"{ev.raw_text[:100]}...\""
                           if len(ev.raw_text) > 100 else f"      Text: \"{ev.raw_text}\"")
                lines.append(f"      Location: {ev.location or 'N/A'}")
                lines.append(f"      Timestamp: {ev.timestamp.isoformat()}")
                lines.append(f"      Extraction: {ev.extraction_method.value}")
                lines.append(f"      Verified: {ev.is_verified}")
                lines.append("")

        if self.parent_claims:
            lines.append("\nDERIVED FROM CLAIMS:")
            for claim_id in self.parent_claims:
                lines.append(f"  - {claim_id}")
            if self.derivation_method:
                lines.append(f"  Method: {self.derivation_method}")

        if self.update_history:
            lines.append("\nUPDATE HISTORY:")
            for update in self.update_history[-5:]:  # Last 5 updates
                lines.append(f"  - {update.get('timestamp', 'N/A')}: {update.get('action', 'N/A')}")

        return "\n".join(lines)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # From belief_math (re-exported for convenience)
    "MassFunction",
    "DecomposedUncertainty",
    "Uncertainty",
    "ConflictResult",
    # VERITY-specific enums
    "EvidenceSource",
    "ExtractionMethod",
    "VerificationStatus",
    # VERITY-specific types
    "Evidence",
    "Claim",
    "ProvenanceChain",
]
