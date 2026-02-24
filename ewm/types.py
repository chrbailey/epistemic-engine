"""
Epistemic World Model — Type System
====================================

Single source of truth for ALL data structures in the Epistemic World Model.

This module defines the complete type vocabulary for a LeCun-inspired 6-module
cognitive architecture extended with principled uncertainty tracking. The core
innovation is the Uncertainty type, which implements Subjective Logic (Josang)
to represent belief, disbelief, and epistemic uncertainty as a Dirichlet/Beta
distribution — going beyond scalar confidence scores.

Type Design Principles:
    - Frozen dataclasses for value objects that should never change after
      creation (Evidence, CostViolation, Uncertainty, MassFunction).
    - Mutable dataclasses for state containers whose contents evolve over
      the lifetime of the system (WorldState, Claim, Entity, Plan).
    - Python 3.11+, with Uncertainty imported from belief_math package.
    - Full type annotations on every field and return type.

Module-to-Type Mapping (LeCun architecture):
    Perception  -> PerceptionResult (entities, claims, evidence extracted)
    Memory      -> MemoryResult (retrieved context with relevance scores)
    World Model -> WorldState (current belief graph snapshot)
    Cost        -> CostAssessment, CostViolation (ethical/safety guardrails)
    Actor       -> Action, Plan (proposed and executed actions)
    Critic      -> uses Uncertainty, MassFunction (evaluates epistemic state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any
from uuid import uuid4

# Import Uncertainty from the shared belief-math package
# This provides Subjective Logic opinion triples with Beta distribution mapping
from belief_math.types import Uncertainty


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Generate a new UUID4 string identifier."""
    return str(uuid4())


# ===========================================================================
# Enums
# ===========================================================================


class SourceType(Enum):
    """Origin of a piece of evidence.

    Each source type carries an implicit reliability prior that downstream
    modules can use when fusing evidence.  For example, DIRECT_OBSERVATION
    is generally weighted higher than HEARSAY, and SYSTEM-generated evidence
    is treated as ground truth within the pipeline.
    """

    DIRECT_OBSERVATION = "direct_observation"
    EXPERT_TESTIMONY = "expert_testimony"
    STATISTICAL = "statistical"
    DOCUMENT = "document"
    INFERENCE = "inference"
    HEARSAY = "hearsay"
    SELF_REPORT = "self_report"
    SYSTEM = "system"


class ClaimType(Enum):
    """Semantic category of a propositional claim.

    The claim type determines which uncertainty calculus and update rules
    the Critic module applies.  For instance, CAUSAL claims require
    intervention-aware reasoning, while STATISTICAL claims update via
    standard Bayesian fusion.

    Types:
        FACTUAL      — verifiable statement of fact (e.g. "Company X was founded in 2004")
        STATISTICAL  — quantitative assertion (e.g. "Revenue grew 12% YoY")
        CAUSAL       — cause-effect relationship (e.g. "Policy A led to outcome B")
        PREDICTIVE   — forward-looking forecast (e.g. "Stock will reach $200 by Q3")
        ACCUSATORY   — attribution of blame or fault (e.g. "Vendor caused the outage")
        DIAGNOSTIC   — root-cause or classification (e.g. "The failure is due to memory leak")
        PRESCRIPTIVE — recommendation or should-statement (e.g. "We should migrate to cloud")
    """

    FACTUAL = "factual"
    STATISTICAL = "statistical"
    CAUSAL = "causal"
    PREDICTIVE = "predictive"
    ACCUSATORY = "accusatory"
    DIAGNOSTIC = "diagnostic"
    PRESCRIPTIVE = "prescriptive"


class EntityCategory(Enum):
    """Broad ontological category for entities in the world model.

    Reduced from 18 to 8 categories to keep the entity graph manageable
    while still covering the domains the system reasons about (business,
    technology, people, finance, geography).
    """

    PERSON = "person"
    ORGANIZATION = "organization"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    LOCATION = "location"
    EVENT = "event"
    ARTIFACT = "artifact"
    FINANCIAL = "financial"


class RelationshipType(Enum):
    """Edge types in the entity-relationship graph.

    Reduced from 48 to 12 canonical types.  Each type is directional
    (source -> target).  Inverse relationships are inferred at query time
    rather than stored redundantly.

    Structural:
        OWNS, EMPLOYS, PART_OF, LOCATED_IN

    Functional:
        USES, PRODUCES, DEPENDS_ON

    Competitive / Regulatory:
        COMPETES_WITH, REGULATES

    Causal / Temporal:
        CAUSES, PRECEDED_BY

    Similarity:
        SIMILAR_TO
    """

    OWNS = "owns"
    EMPLOYS = "employs"
    USES = "uses"
    PRODUCES = "produces"
    DEPENDS_ON = "depends_on"
    COMPETES_WITH = "competes_with"
    REGULATES = "regulates"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    CAUSES = "causes"
    PRECEDED_BY = "preceded_by"
    SIMILAR_TO = "similar_to"


class ActionType(Enum):
    """Operations the Actor module can propose.

    Each action goes through the Cost module for ethical/safety screening
    before execution.

    Mutation actions:
        CREATE_ENTITY, UPDATE_ENTITY, CREATE_CLAIM, UPDATE_CLAIM,
        CREATE_RELATIONSHIP

    Meta actions:
        INVESTIGATE — request more information to reduce uncertainty
        REDACT      — mask sensitive content (PII, classified info)
        BLOCK       — refuse to process or surface content
    """

    CREATE_ENTITY = "create_entity"
    UPDATE_ENTITY = "update_entity"
    CREATE_CLAIM = "create_claim"
    UPDATE_CLAIM = "update_claim"
    CREATE_RELATIONSHIP = "create_relationship"
    INVESTIGATE = "investigate"
    REDACT = "redact"
    BLOCK = "block"


# ===========================================================================
# Core Uncertainty Types
# ===========================================================================
#
# Uncertainty is now imported from belief_math.types at the top of this file.
# This provides Subjective Logic opinion triples (belief, disbelief, uncertainty)
# with Beta distribution mapping, credible intervals, and factory methods.
#
# See belief_math package for implementation details.
# ===========================================================================


@dataclass(frozen=True)
class MassFunction:
    """Dempster-Shafer mass function over a frame of discernment.

    Maps subsets of hypotheses (represented as frozensets of strings) to
    mass values in [0, 1].  The masses must sum to 1.0.  This provides a
    more expressive uncertainty representation than Bayesian probabilities
    because mass can be assigned to *sets* of hypotheses, explicitly
    encoding "I know it's one of these but I don't know which."

    The full frame of discernment (Theta) is represented by the union of
    all hypothesis labels.  Mass on Theta represents total ignorance.

    Example:
        >>> mf = MassFunction(masses={
        ...     frozenset({"sunny"}): 0.3,
        ...     frozenset({"rainy"}): 0.2,
        ...     frozenset({"sunny", "rainy", "cloudy"}): 0.5,
        ... })
        >>> mf.belief(frozenset({"sunny"}))
        0.3
        >>> mf.plausibility(frozenset({"sunny"}))
        0.8
    """

    masses: dict[frozenset[str], float]

    def __post_init__(self) -> None:
        """Validate that masses sum to 1.0 and are non-negative.

        Raises:
            ValueError: If any mass is negative or masses don't sum to 1.0.
        """
        for subset, mass in self.masses.items():
            if mass < 0.0:
                raise ValueError(
                    f"Mass must be non-negative: m({subset}) = {mass}"
                )
        total = sum(self.masses.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Masses must sum to 1.0 (got {total:.10f})"
            )

    @property
    def focal_elements(self) -> list[frozenset[str]]:
        """Return all hypothesis sets with non-zero mass."""
        return [s for s, m in self.masses.items() if m > 0.0]

    def belief(self, hypothesis: frozenset[str]) -> float:
        """Belief in a hypothesis = sum of masses of all subsets.

        Bel(A) = sum{ m(B) : B is a subset of A }

        This is the minimum guaranteed support for the hypothesis.

        Args:
            hypothesis: The set of hypotheses to compute belief for.

        Returns:
            Belief value in [0, 1].
        """
        return sum(
            mass for subset, mass in self.masses.items()
            if subset <= hypothesis and len(subset) > 0
        )

    def plausibility(self, hypothesis: frozenset[str]) -> float:
        """Plausibility of a hypothesis = 1 - Bel(complement).

        Pl(A) = 1 - Bel(not A) = sum{ m(B) : B intersects A }

        This is the maximum possible support — the upper bound when all
        ambiguous mass is resolved in the hypothesis's favor.

        Args:
            hypothesis: The set of hypotheses to compute plausibility for.

        Returns:
            Plausibility value in [0, 1].
        """
        return sum(
            mass for subset, mass in self.masses.items()
            if subset & hypothesis and len(subset) > 0
        )

    def uncertainty_interval(self, hypothesis: frozenset[str]) -> tuple[float, float]:
        """Return the (belief, plausibility) interval for a hypothesis.

        The width of this interval represents the degree of ignorance
        about the hypothesis.  A narrow interval means we have strong
        evidence one way or the other; a wide interval means significant
        ambiguity remains.

        Args:
            hypothesis: The set of hypotheses to compute the interval for.

        Returns:
            (belief, plausibility) tuple.
        """
        return (self.belief(hypothesis), self.plausibility(hypothesis))

    def __repr__(self) -> str:
        items = ", ".join(
            f"{set(k)}: {v:.3f}" for k, v in self.masses.items() if v > 0.0
        )
        return f"MassFunction({{{items}}})"


# ===========================================================================
# Evidence & Claims
# ===========================================================================


@dataclass(frozen=True)
class Evidence:
    """A single piece of evidence that supports or undermines claims.

    Evidence is immutable after creation — once observed, an observation
    doesn't change.  New evidence is created, not modified.

    The reliability field captures the source's overall trustworthiness
    (0 = completely unreliable, 1 = perfectly reliable).  This is used
    as a discount factor when fusing evidence into claim uncertainty.

    Attributes:
        id: Unique identifier (UUID4).
        source_type: Origin category of the evidence.
        content: The actual evidential content (text, structured data, etc.).
        source_id: Identifier of the entity/agent that provided the evidence.
        timestamp: ISO 8601 creation timestamp.
        reliability: Source reliability weight in [0, 1].
        metadata: Arbitrary key-value pairs for domain-specific annotations.
    """

    id: str = field(default_factory=_new_id)
    source_type: SourceType = SourceType.SYSTEM
    content: str = ""
    source_id: str = ""
    timestamp: str = field(default_factory=_now_iso)
    reliability: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.reliability <= 1.0):
            raise ValueError(
                f"reliability must be in [0, 1], got {self.reliability}"
            )


@dataclass
class Claim:
    """A propositional claim with tracked uncertainty.

    Claims are the primary unit of knowledge in the world model.  Each
    claim has a natural-language text, a semantic type, and a full
    Uncertainty opinion that evolves as evidence arrives.

    Claims are mutable because their uncertainty updates over time as
    new evidence is fused, and evidence/entity links grow.

    The five_w1h dict captures the journalistic decomposition:
        who, what, when, where, why, how

    Attributes:
        id: Unique identifier (UUID4).
        text: The proposition being asserted.
        claim_type: Semantic category of the claim.
        uncertainty: Current Subjective Logic opinion.
        evidence_ids: IDs of Evidence objects supporting/undermining this claim.
        entity_ids: IDs of Entities mentioned in or relevant to this claim.
        created: ISO 8601 creation timestamp.
        updated: ISO 8601 last-update timestamp.
        tags: Free-form labels for categorization and retrieval.
        five_w1h: Journalistic decomposition of the claim.
    """

    id: str = field(default_factory=_new_id)
    text: str = ""
    claim_type: ClaimType = ClaimType.FACTUAL
    uncertainty: Uncertainty = field(default_factory=Uncertainty.uniform)
    evidence_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    created: str = field(default_factory=_now_iso)
    updated: str = field(default_factory=_now_iso)
    tags: list[str] = field(default_factory=list)
    five_w1h: dict[str, str] = field(default_factory=dict)


@dataclass
class Entity:
    """A named entity in the world model's knowledge graph.

    Entities are the nodes of the graph.  They have a canonical name,
    a broad category, optional aliases for fuzzy matching, and a
    flexible properties dict for domain-specific attributes.

    Attributes:
        id: Unique identifier (UUID4).
        name: Canonical display name.
        category: Broad ontological category.
        aliases: Alternative names, abbreviations, ticker symbols, etc.
        properties: Arbitrary key-value pairs (e.g. {"ticker": "NVDA"}).
        created: ISO 8601 creation timestamp.
        updated: ISO 8601 last-update timestamp.
    """

    id: str = field(default_factory=_new_id)
    name: str = ""
    category: EntityCategory = EntityCategory.CONCEPT
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=_now_iso)
    updated: str = field(default_factory=_now_iso)


@dataclass
class Relationship:
    """A directed edge between two entities in the knowledge graph.

    Relationships are mutable because new evidence can be linked to them
    over time and confidence updates as the system learns.

    The direction is source -> target.  For example:
        (Google, EMPLOYS, John) means "Google employs John"
        (Python, DEPENDS_ON, CPython) means "Python depends on CPython"

    Attributes:
        id: Unique identifier (UUID4).
        source_id: ID of the source Entity.
        target_id: ID of the target Entity.
        rel_type: Semantic type of the relationship.
        evidence_ids: IDs of Evidence supporting this relationship.
        confidence: Scalar confidence in [0, 1] (derived from evidence).
        created: ISO 8601 creation timestamp.
    """

    id: str = field(default_factory=_new_id)
    source_id: str = ""
    target_id: str = ""
    rel_type: RelationshipType = RelationshipType.SIMILAR_TO
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    created: str = field(default_factory=_now_iso)


# ===========================================================================
# World State — the central snapshot
# ===========================================================================


@dataclass
class WorldState:
    """Snapshot of the world model at a point in time.

    This is the object that flows through the 6-module pipeline.  Each
    module reads from and/or writes to the world state, progressively
    refining the system's understanding.

    Pipeline flow:
        Input -> Perception -> Memory -> WorldModel -> Cost -> Actor -> Output
                                            ^                   |
                                            |--- Critic --------|

    Attributes:
        entities: All known entities, keyed by ID.
        claims: All known claims, keyed by ID.
        relationships: All known relationships.
        timestamp: ISO 8601 timestamp of this snapshot.
    """

    entities: dict[str, Entity] = field(default_factory=dict)
    claims: dict[str, Claim] = field(default_factory=dict)
    red_lines: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    timestamp: str = field(default_factory=_now_iso)


# ===========================================================================
# Actions & Planning
# ===========================================================================


@dataclass
class Action:
    """A single proposed or executed action.

    Actions go through a lifecycle: proposed -> approved -> executed,
    or proposed -> blocked if the Cost module rejects them.

    Attributes:
        id: Unique identifier (UUID4).
        action_type: What kind of operation this is.
        target_id: ID of the entity/claim/relationship being acted upon.
        payload: Action-specific parameters (e.g., new field values).
        rationale: Human-readable explanation of why this action was proposed.
        status: Lifecycle state — "proposed", "approved", "executed", or "blocked".
    """

    id: str = field(default_factory=_new_id)
    action_type: ActionType = ActionType.INVESTIGATE
    target_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    status: str = "proposed"


@dataclass
class Plan:
    """An ordered sequence of actions proposed by the Actor module.

    The plan includes an estimate of how much epistemic uncertainty
    the actions are expected to reduce (expected_info_gain), which the
    Critic module uses to prioritize among competing plans.

    Attributes:
        actions: Ordered list of actions to execute.
        expected_info_gain: Estimated reduction in total uncertainty (bits).
        rationale: Human-readable explanation of the plan's strategy.
    """

    actions: list[Action] = field(default_factory=list)
    expected_info_gain: float = 0.0
    rationale: str = ""


# ===========================================================================
# Cost Module Types
# ===========================================================================


@dataclass(frozen=True)
class CostViolation:
    """A single rule violation detected by the Cost module.

    Immutable because violations are factual findings — once detected,
    they don't change.

    Severity levels:
        red_line — action MUST be blocked (e.g., PII exposure, harmful content)
        caution  — action should be flagged for review but may proceed
        info     — advisory note, no blocking

    Attributes:
        rule: Identifier of the rule that was violated (e.g., "no_pii_exposure").
        description: Human-readable explanation of the violation.
        severity: Impact level — "red_line", "caution", or "info".
    """

    rule: str = ""
    description: str = ""
    severity: str = "info"

    def __post_init__(self) -> None:
        valid_severities = {"red_line", "caution", "info"}
        if self.severity not in valid_severities:
            raise ValueError(
                f"severity must be one of {valid_severities}, "
                f"got '{self.severity}'"
            )


@dataclass(frozen=True)
class CostAssessment:
    """Result of the Cost module evaluating a proposed action.

    Contains the original action, whether it was blocked, all violations
    found, and optionally an adjusted (safer) version of the action.

    Attributes:
        action: The original proposed action.
        blocked: Whether the action was blocked by a red_line violation.
        violations: All violations detected (may be empty).
        adjusted_action: A modified version of the action that addresses
            caution-level violations, or None if no adjustment was needed
            or the action was blocked outright.
    """

    action: Action
    blocked: bool = False
    violations: tuple[CostViolation, ...] = ()
    adjusted_action: Action | None = None


# ===========================================================================
# Module Result Types
# ===========================================================================


@dataclass
class PerceptionResult:
    """Output of the Perception module.

    The Perception module is the system's sensory interface — it takes
    raw input (text, documents, API responses) and extracts structured
    knowledge: entities, claims, relationships, and evidence.

    It also handles privacy/safety at the input boundary by redacting
    sensitive content before it enters the rest of the pipeline.

    Attributes:
        entities: Entities extracted from the input.
        claims: Claims extracted from the input.
        relationships: Relationships extracted from the input.
        evidence: Evidence objects created from the input.
        redacted_content: The input text with sensitive spans replaced.
        sensitive_spans: List of (start, end, span_type) tuples identifying
            what was redacted and why (e.g., "PII", "CLASSIFIED").
    """

    entities: list[Entity] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    redacted_content: str = ""
    sensitive_spans: list[tuple[int, int, str]] = field(default_factory=list)


@dataclass
class MemoryResult:
    """Output of the Memory module.

    The Memory module retrieves previously stored knowledge that is
    relevant to the current context.  It returns entities, claims, and
    relationships along with relevance scores indicating how closely
    each item matches the current query/context.

    Attributes:
        entities: Retrieved entities relevant to the current context.
        claims: Retrieved claims relevant to the current context.
        relationships: Retrieved relationships relevant to the current context.
        relevance_scores: Mapping from item ID to relevance score in [0, 1].
    """

    entities: list[Entity] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    relevance_scores: dict[str, float] = field(default_factory=dict)
