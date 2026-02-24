"""
Epistemic World Model -- World Model Module
=============================================

The mathematical core of the Epistemic World Model, implementing the World Model
component from LeCun's 6-module cognitive architecture.  This module is responsible
for maintaining, updating, and reasoning about the system's beliefs using principled
uncertainty calculus.

Core capabilities:
    1. Dempster-Shafer belief fusion -- combining independent evidence sources via
       mass functions over a frame of discernment.
    2. Subjective Logic updates -- Bayesian-like updates of opinion triples
       (belief, disbelief, uncertainty) as new evidence arrives.
    3. Belief graph propagation -- spreading belief updates through entity
       relationships so that related claims are coherently adjusted.
    4. Prediction -- projecting the current world state forward in time by
       applying heuristic decay and stability rules to claims and relationships.
    5. Gap detection -- identifying missing knowledge (entities without claims,
       unsupported claims, isolated organizations) to guide investigation.
    6. Evidence integration -- the full pipeline from raw evidence to an updated
       world state with propagated belief changes.

Design principles:
    - Pure functions where possible: return new objects, don't mutate inputs.
    - Uses belief_math package for Subjective Logic operators.
    - All uncertainty is explicit: no silent confidence scalars or hidden defaults.
    - The Uncertainty type (Subjective Logic opinion) is the lingua franca of
      every function in this module.

Python 3.11+ required.  Uses ``from __future__ import annotations`` for deferred
evaluation of type hints throughout.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any

from ewm.types import (
    Claim,
    Entity,
    EntityCategory,
    Evidence,
    Relationship,
    RelationshipType,
    SourceType,
    Uncertainty,
    WorldState,
)

# Import Subjective Logic operators from the shared belief_math package
from belief_math.subjective_logic import (
    cumulative_fuse,
    averaging_fuse,
    trust_discount,
    trust_chain,
    deduce,
    opinion_complement,
    uncertainty_maximized,
    opinion_to_probability,
    probability_to_opinion,
    blend_uncertainty as _blend_uncertainty,
)


# ===========================================================================
# Uncertainty Operations
# ===========================================================================


def update_uncertainty(
    prior: Uncertainty,
    evidence: Evidence,
    direction: str = "support",
) -> Uncertainty:
    """Bayesian-like update of a Subjective Logic opinion given new evidence.

    Updates the belief/disbelief/uncertainty triple based on the incoming
    evidence.  Supporting evidence shifts mass from uncertainty into belief;
    contradicting evidence shifts mass from uncertainty into disbelief.

    The magnitude of the shift is proportional to:
        - The evidence source's reliability (higher = stronger update)
        - The remaining epistemic uncertainty (less room to move when
          uncertainty is already low)

    Args:
        prior: Current Uncertainty opinion to update.
        evidence: The new evidence being incorporated.
        direction: Either ``"support"`` (evidence confirms the claim) or
            ``"contradict"`` (evidence undermines the claim).

    Returns:
        A new Uncertainty opinion reflecting the updated belief state.

    Raises:
        ValueError: If direction is not ``"support"`` or ``"contradict"``.

    Note:
        This function is pure -- it does not mutate the prior or evidence.
    """
    if direction not in ("support", "contradict"):
        raise ValueError(
            f"direction must be 'support' or 'contradict', got '{direction}'"
        )

    weight = evidence.reliability
    transferable = prior.uncertainty * 0.5 * weight

    if direction == "support":
        new_belief = prior.belief + transferable
        new_disbelief = prior.disbelief * (1.0 - 0.1 * weight)
        new_uncertainty = 1.0 - new_belief - new_disbelief
    else:
        new_disbelief = prior.disbelief + transferable
        new_belief = prior.belief * (1.0 - 0.1 * weight)
        new_uncertainty = 1.0 - new_belief - new_disbelief

    # Clamp to valid ranges before normalization
    new_belief = max(0.0, new_belief)
    new_disbelief = max(0.0, new_disbelief)
    new_uncertainty = max(0.0, new_uncertainty)

    # Normalize so components sum to 1.0
    total = new_belief + new_disbelief + new_uncertainty
    if total < 1e-12:
        # Degenerate case: fall back to uniform
        return Uncertainty.uniform()

    new_belief /= total
    new_disbelief /= total
    new_uncertainty /= total

    new_sample_size = prior.sample_size + weight

    return Uncertainty(
        belief=new_belief,
        disbelief=new_disbelief,
        uncertainty=new_uncertainty,
        sample_size=new_sample_size,
    )


def expected_info_gain(
    claim: Claim,
    possible_outcomes: list[str] | None = None,
) -> float:
    """Compute expected information gain from investigating a claim.

    Higher uncertainty means more potential for learning.  This function
    combines the raw epistemic uncertainty with the Shannon entropy of the
    current belief distribution to produce a single scalar that can rank
    claims by "investigation value."

    Formula:
        gain = u * (1.0 + H)
    where:
        u = claim.uncertainty.uncertainty  (the epistemic gap)
        H = -sum(p * log2(p))  for p in [E[value], 1 - E[value]]
            (entropy of the expected probability)

    Args:
        claim: The claim to evaluate.
        possible_outcomes: Reserved for future use with multi-hypothesis
            claims.  Currently ignored.

    Returns:
        Non-negative float representing expected information gain in bits.
        A return value of 0.0 means investigating this claim would yield
        no new information (it is already fully resolved).
    """
    u = claim.uncertainty.uncertainty
    ev = claim.uncertainty.expected_value

    # Compute binary entropy of the expected value.
    # Clamp to [eps, 1-eps] to avoid log2(~0) producing -inf.
    entropy = 0.0
    _eps = 1e-15
    if _eps < ev < (1.0 - _eps):
        entropy = -(ev * math.log2(ev) + (1.0 - ev) * math.log2(1.0 - ev))

    gain = u * (1.0 + entropy)
    return max(0.0, gain)


def credible_interval(
    uncertainty: Uncertainty,
    width: float = 0.9,
) -> tuple[float, float]:
    """Compute an approximate credible interval for the expected probability.

    Convenience wrapper around ``Uncertainty.credible_interval``.

    Args:
        uncertainty: The Subjective Logic opinion.
        width: Probability mass to capture (0 < width < 1).  Default 0.9
            returns the (5th, 95th) percentile.

    Returns:
        (lower, upper) bounds clamped to [0, 1].
    """
    return uncertainty.credible_interval(width)


# ===========================================================================
# Subjective Logic Operators
# ===========================================================================
#
# All Subjective Logic operators are now imported from belief_math.subjective_logic:
#   - cumulative_fuse: Combine independent opinion sources
#   - averaging_fuse: Combine dependent opinion sources
#   - trust_discount: Propagate opinions through trust networks
#   - trust_chain: Transitive trust computation
#   - deduce: Derive opinions about conclusions from premises
#   - opinion_complement: Negate an opinion (swap belief/disbelief)
#   - uncertainty_maximized: Select most uncertain opinion
#   - opinion_to_probability: Project opinion to scalar probability
#   - probability_to_opinion: Convert probability to opinion
#   - blend_uncertainty (as _blend_uncertainty): Weighted blend of opinions
#
# See belief_math package for implementation details.
# ===========================================================================


# ===========================================================================
# Belief Graph
# ===========================================================================


class BeliefGraph:
    """A graph structure for propagating belief updates through entity relationships.

    The belief graph is built from a WorldState's entities, claims, and
    relationships.  When a claim's uncertainty changes, the update is spread
    to related claims via the entity-relationship graph, with influence
    decaying exponentially per hop.

    The graph is bipartite at its core: entities are connected to other
    entities via relationships, and claims are connected to entities via
    their ``entity_ids`` lists.  Propagation follows the path:
        claim -> entities -> relationships -> entities -> claims.
    """

    def __init__(self, world_state: WorldState) -> None:
        """Build the belief graph from a world state snapshot.

        Constructs:
            - Entity adjacency: entity_id -> set of neighbor entity_ids
            - Claim index: entity_id -> list of claim_ids about that entity
            - Claim lookup: claim_id -> Claim object
            - Entity lookup: entity_id -> Entity object

        Args:
            world_state: The world state to build the graph from.
        """
        self._state = world_state

        # Entity adjacency (undirected for propagation purposes)
        self._adjacency: dict[str, set[str]] = {}
        for entity_id in world_state.entities:
            self._adjacency[entity_id] = set()

        for rel in world_state.relationships:
            src = rel.source_id
            tgt = rel.target_id
            if src in self._adjacency:
                self._adjacency[src].add(tgt)
            else:
                self._adjacency[src] = {tgt}
            if tgt in self._adjacency:
                self._adjacency[tgt].add(src)
            else:
                self._adjacency[tgt] = {src}

        # Claims indexed by entity_id
        self._entity_claims: dict[str, list[str]] = {}
        for claim_id, claim in world_state.claims.items():
            for eid in claim.entity_ids:
                if eid not in self._entity_claims:
                    self._entity_claims[eid] = []
                self._entity_claims[eid].append(claim_id)

        # Reverse: claim_id -> set of entity_ids
        self._claim_entities: dict[str, set[str]] = {}
        for claim_id, claim in world_state.claims.items():
            self._claim_entities[claim_id] = set(claim.entity_ids)

    def propagate(
        self,
        source_claim_id: str,
        max_depth: int = 3,
        decay: float = 0.8,
    ) -> dict[str, Uncertainty]:
        """Propagate belief updates from a source claim through the graph.

        Starting from the source claim, finds the entities it mentions,
        then traverses the relationship graph outward via BFS.  At each
        hop, the influence decays by the ``decay`` factor.  For each
        entity reached, all associated claims have their uncertainty
        nudged toward the source claim's updated uncertainty.

        The nudge is computed as a weighted blend:
            new_u = (1 - influence) * current_u + influence * source_u
        where ``influence = decay^depth`` and the blend is applied to each
        component (belief, disbelief, uncertainty) independently.

        Args:
            source_claim_id: ID of the claim whose uncertainty just changed.
            max_depth: Maximum number of relationship hops to propagate.
            decay: Multiplicative decay per hop (0 < decay < 1).

        Returns:
            Dict mapping claim_id -> updated Uncertainty for every claim
            affected by the propagation (excluding the source claim itself).

        Raises:
            ValueError: If source_claim_id is not in the world state.
        """
        if source_claim_id not in self._state.claims:
            raise ValueError(
                f"Claim '{source_claim_id}' not found in world state"
            )

        source_claim = self._state.claims[source_claim_id]
        source_u = source_claim.uncertainty
        updates: dict[str, Uncertainty] = {}

        # Collect starting entities for the source claim
        start_entities = self._claim_entities.get(source_claim_id, set())
        if not start_entities:
            return updates

        # BFS through entity graph
        visited_entities: set[str] = set()
        visited_claims: set[str] = {source_claim_id}

        # Queue entries: (entity_id, current_depth)
        queue: deque[tuple[str, int]] = deque()
        for eid in start_entities:
            queue.append((eid, 0))
            visited_entities.add(eid)

        while queue:
            entity_id, depth = queue.popleft()

            if depth > max_depth:
                continue

            influence = decay ** max(depth, 1)

            # Update all claims attached to this entity (except the source)
            for claim_id in self._entity_claims.get(entity_id, []):
                if claim_id in visited_claims:
                    continue
                visited_claims.add(claim_id)

                current = self._state.claims[claim_id].uncertainty
                blended = _blend_uncertainty(current, source_u, influence)
                updates[claim_id] = blended

            # Expand to neighbor entities if within depth
            if depth < max_depth:
                for neighbor_id in self._adjacency.get(entity_id, set()):
                    if neighbor_id not in visited_entities:
                        visited_entities.add(neighbor_id)
                        queue.append((neighbor_id, depth + 1))

        return updates


# ===========================================================================
# Prediction
# ===========================================================================


def predict_state(
    current: WorldState,
    action_type: str = "",
) -> WorldState:
    """Predict the next world state given the current state.

    Applies heuristic rules to project the belief graph forward in time:

    1. **Settled claims** (uncertainty < 0.1 and belief > 0.8) are kept
       unchanged -- they represent stable, well-supported knowledge.

    2. **High-uncertainty claims** (uncertainty > 0.5) decay toward a
       more uniform distribution, modeling information loss over time
       when knowledge is not reinforced by evidence.

    3. **Low-confidence relationships** (confidence < 0.3) are dropped
       from the predicted state -- they represent weak associations that
       are unlikely to persist without further evidence.

    4. **Investigation action hint**: If ``action_type`` contains
       "investigate", the highest-uncertainty claims get a small
       uncertainty reduction, anticipating that investigation will yield
       new evidence.

    Args:
        current: The current world state to project from.
        action_type: Optional hint about what action is being taken.
            If it contains "investigate", high-uncertainty claims get
            a preemptive reduction.

    Returns:
        A new WorldState representing the predicted future state.
        The original state is not modified.

    Note:
        This is a simplified heuristic predictor.  A production system
        would use learned transition models (e.g., a neural state-space
        model) trained on historical state trajectories.
    """
    predicted = copy.deepcopy(current)

    # Information decay constant: how fast uncertain knowledge degrades
    decay_rate = 0.05

    for claim_id, claim in predicted.claims.items():
        u = claim.uncertainty

        # Settled claims: leave them alone
        if u.uncertainty < 0.1 and u.belief > 0.8:
            continue

        # High uncertainty claims: decay toward uniform
        if u.uncertainty > 0.5:
            # Shift belief and disbelief toward 1/3 each
            target_b = 1.0 / 3.0
            target_d = 1.0 / 3.0
            target_u = 1.0 / 3.0

            new_b = u.belief + decay_rate * (target_b - u.belief)
            new_d = u.disbelief + decay_rate * (target_d - u.disbelief)
            new_u = u.uncertainty + decay_rate * (target_u - u.uncertainty)

            # Normalize
            total = new_b + new_d + new_u
            claim.uncertainty = Uncertainty(
                belief=new_b / total,
                disbelief=new_d / total,
                uncertainty=new_u / total,
                sample_size=u.sample_size,
            )
        else:
            # Moderate claims: slight uncertainty increase (information loss)
            shift = decay_rate * 0.5
            new_b = max(0.0, u.belief - shift * 0.5)
            new_d = max(0.0, u.disbelief - shift * 0.5)
            new_u = u.uncertainty + shift

            total = new_b + new_d + new_u
            claim.uncertainty = Uncertainty(
                belief=new_b / total,
                disbelief=new_d / total,
                uncertainty=new_u / total,
                sample_size=u.sample_size,
            )

    # Drop low-confidence relationships
    predicted.relationships = [
        rel for rel in predicted.relationships if rel.confidence >= 0.3
    ]

    # Investigation hint: reduce uncertainty of most uncertain claims
    if "investigate" in action_type.lower():
        sorted_claims = sorted(
            predicted.claims.values(),
            key=lambda c: c.uncertainty.uncertainty,
            reverse=True,
        )
        # Reduce uncertainty of top 3 most uncertain claims
        for claim in sorted_claims[:3]:
            u = claim.uncertainty
            if u.uncertainty > 0.1:
                reduction = 0.1 * u.uncertainty
                new_u = u.uncertainty - reduction
                # Transfer reduced uncertainty to belief (optimistic about investigation)
                new_b = u.belief + reduction * 0.7
                new_d = u.disbelief + reduction * 0.3

                total = new_b + new_d + new_u
                claim.uncertainty = Uncertainty(
                    belief=new_b / total,
                    disbelief=new_d / total,
                    uncertainty=new_u / total,
                    sample_size=u.sample_size + 0.5,
                )

    return predicted


# ===========================================================================
# Fill Missing (Gap Detection)
# ===========================================================================


def fill_missing(state: WorldState) -> list[dict[str, Any]]:
    """Identify knowledge gaps in the world state.

    Scans the world state for structural deficiencies -- places where
    the knowledge graph is thin, unsupported, or incomplete.  Returns
    a prioritized list of gaps that investigation could fill.

    Gap types detected:
        - ``entity_without_claims``: An entity exists but has no claims
          about it, meaning we know *of* the entity but know nothing
          *about* it.
        - ``unsupported_claim``: A claim exists but has no linked evidence,
          meaning it is an assertion without backing.
        - ``relationship_without_context``: Two entities are connected by
          a relationship, but share no claims, meaning the relationship
          exists in a vacuum without narrative context.
        - ``isolated_org``: An organization entity has no EMPLOYS or OWNS
          relationships, which is structurally unusual for organizations
          and suggests missing knowledge.

    Args:
        state: The world state to analyze.

    Returns:
        List of gap descriptors sorted by estimated importance (descending).
        Each descriptor is a dict with keys: ``type``, relevant IDs,
        ``description``, and ``estimated_importance`` (float in [0, 1]).
    """
    gaps: list[dict[str, Any]] = []

    # Build lookup: entity_id -> set of claim_ids
    entity_claim_ids: dict[str, set[str]] = {}
    for claim_id, claim in state.claims.items():
        for eid in claim.entity_ids:
            if eid not in entity_claim_ids:
                entity_claim_ids[eid] = set()
            entity_claim_ids[eid].add(claim_id)

    # Build lookup: entity_id -> set of relationship types (as source)
    entity_rel_types: dict[str, set[RelationshipType]] = {}
    for rel in state.relationships:
        if rel.source_id not in entity_rel_types:
            entity_rel_types[rel.source_id] = set()
        entity_rel_types[rel.source_id].add(rel.rel_type)

    # 1. Entities with no claims
    for entity_id, entity in state.entities.items():
        if entity_id not in entity_claim_ids or len(entity_claim_ids[entity_id]) == 0:
            gaps.append({
                "type": "entity_without_claims",
                "entity_id": entity_id,
                "description": (
                    f"Entity '{entity.name}' ({entity.category.value}) "
                    f"has no claims. We know of it but nothing about it."
                ),
                "estimated_importance": 0.8,
            })

    # 2. Claims with no evidence
    for claim_id, claim in state.claims.items():
        if not claim.evidence_ids:
            # Higher importance for high-belief unsupported claims
            importance = 0.6 + 0.3 * claim.uncertainty.belief
            gaps.append({
                "type": "unsupported_claim",
                "claim_id": claim_id,
                "description": (
                    f"Claim '{claim.text[:80]}' has no supporting evidence. "
                    f"Current belief: {claim.uncertainty.belief:.2f}."
                ),
                "estimated_importance": min(1.0, importance),
            })

    # 3. Relationships without shared claims
    for rel in state.relationships:
        source_claims = entity_claim_ids.get(rel.source_id, set())
        target_claims = entity_claim_ids.get(rel.target_id, set())

        # Check if any claim mentions both entities
        shared = source_claims & target_claims
        if not shared:
            source_name = state.entities.get(rel.source_id)
            target_name = state.entities.get(rel.target_id)
            src_label = source_name.name if source_name else rel.source_id
            tgt_label = target_name.name if target_name else rel.target_id
            gaps.append({
                "type": "relationship_without_context",
                "relationship_id": rel.id,
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "description": (
                    f"Relationship {rel.rel_type.value} between "
                    f"'{src_label}' and '{tgt_label}' has no shared claims "
                    f"providing narrative context."
                ),
                "estimated_importance": 0.5,
            })

    # 4. Organizations without EMPLOYS or OWNS relationships
    structural_rel_types = {RelationshipType.EMPLOYS, RelationshipType.OWNS}
    for entity_id, entity in state.entities.items():
        if entity.category == EntityCategory.ORGANIZATION:
            rels = entity_rel_types.get(entity_id, set())
            if not rels & structural_rel_types:
                gaps.append({
                    "type": "isolated_org",
                    "entity_id": entity_id,
                    "description": (
                        f"Organization '{entity.name}' has no EMPLOYS or "
                        f"OWNS relationships. This is structurally unusual "
                        f"and may indicate missing knowledge."
                    ),
                    "estimated_importance": 0.7,
                })

    # Sort by estimated importance, descending
    gaps.sort(key=lambda g: g["estimated_importance"], reverse=True)

    return gaps


# ===========================================================================
# Integration Pipeline
# ===========================================================================


def integrate_evidence(
    state: WorldState,
    evidence: Evidence,
    claim_id: str,
    direction: str = "support",
) -> WorldState:
    """Full evidence integration pipeline: update, propagate, return.

    This is the primary entry point for incorporating new evidence into
    the world model.  It performs three steps:

    1. **Direct update**: Update the target claim's uncertainty using
       ``update_uncertainty`` with the new evidence.
    2. **Graph propagation**: Build a ``BeliefGraph`` and propagate the
       belief change to related claims, with influence decaying per hop.
    3. **State assembly**: Apply all updates (direct + propagated) to
       a deep copy of the world state and return it.

    Args:
        state: Current world state.
        evidence: The new evidence to incorporate.
        claim_id: ID of the claim this evidence pertains to.
        direction: ``"support"`` or ``"contradict"``.

    Returns:
        A new WorldState with all claim uncertainties updated.

    Raises:
        ValueError: If claim_id is not found in the state, or if
            direction is invalid.
    """
    if claim_id not in state.claims:
        raise ValueError(
            f"Claim '{claim_id}' not found in world state. "
            f"Available claims: {list(state.claims.keys())}"
        )

    updated_state = copy.deepcopy(state)

    # Step 1: Direct update of the target claim
    target_claim = updated_state.claims[claim_id]
    new_uncertainty = update_uncertainty(
        prior=target_claim.uncertainty,
        evidence=evidence,
        direction=direction,
    )
    target_claim.uncertainty = new_uncertainty

    # Record the evidence link
    if evidence.id not in target_claim.evidence_ids:
        target_claim.evidence_ids.append(evidence.id)

    # Step 2: Propagate through the belief graph
    graph = BeliefGraph(updated_state)
    propagated = graph.propagate(source_claim_id=claim_id)

    # Step 3: Apply propagated updates
    for propagated_claim_id, propagated_uncertainty in propagated.items():
        if propagated_claim_id in updated_state.claims:
            updated_state.claims[propagated_claim_id].uncertainty = propagated_uncertainty

    return updated_state
