"""
Epistemic World Model -- Actor Module
======================================

The Actor module is one of LeCun's 6 core modules in the cognitive architecture.
It plans investigations and proposes actions based on world state analysis,
bridging the gap between epistemic assessment and concrete state mutations.

Core capabilities:
    1. Investigation planning -- ranks claims by expected information gain and
       generates 5W1H investigation questions for the highest-value targets.
    2. Update proposal -- compares perception output to current world state and
       proposes CREATE/UPDATE actions for entities, claims, and relationships.
    3. Action execution -- applies approved actions against the database layer.
    4. 5W1H decomposition -- heuristic extraction of Who/What/When/Where/Why/How
       from claim text and metadata.

Design principles:
    - Pure functions where possible (plan_investigation, propose_updates, etc.)
    - execute() is the only function with side effects (database writes)
    - stdlib only -- no external dependencies
    - Uses expected_info_gain() from world_model.py as the ranking criterion
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from ewm.db import Database
from ewm.types import (
    Action,
    ActionType,
    Claim,
    ClaimType,
    Entity,
    PerceptionResult,
    Plan,
    Relationship,
    Uncertainty,
    WorldState,
)
from ewm.world_model import expected_info_gain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level compiled patterns for 5W1H extraction
# ---------------------------------------------------------------------------

_RE_CAPITALIZED_WORDS = re.compile(
    r"\b([A-Z][a-zA-Z&'-]+(?:\s+[A-Z][a-zA-Z&'-]+)*)\b"
)

_RE_ORG_PATTERNS = re.compile(
    r"\b(?:Corp(?:oration)?|Inc(?:orporated)?|Ltd|LLC|Co(?:mpany)?|Group|"
    r"Holdings|Technologies|Systems|Partners|Foundation|Institute)\b"
)

_RE_DATE_TIME = re.compile(
    r"\b(?:Q[1-4]\s*\d{4}|FY\s*\d{4}|\d{4}|"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
)

_RE_LOCATION = re.compile(
    r"\b(?:in|based\s+in|headquartered\s+in|located\s+in|from|near)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
)

_COUNTRY_NAMES = frozenset({
    "United States", "United Kingdom", "China", "Japan", "Germany", "France",
    "India", "Canada", "Australia", "Brazil", "South Korea", "Mexico",
    "Russia", "Italy", "Spain", "Netherlands", "Switzerland", "Sweden",
    "Singapore", "Israel", "Ireland", "Taiwan", "Norway", "Denmark",
    "Finland", "Belgium", "Austria", "Poland", "Turkey", "Saudi Arabia",
    "New Zealand", "Argentina", "Colombia", "Chile", "Thailand", "Indonesia",
    "Malaysia", "Philippines", "Vietnam", "Nigeria", "Egypt", "Kenya",
    "South Africa", "UAE", "Qatar",
})

_RE_CAUSAL_BECAUSE = re.compile(
    r"\bbecause\s+(.+?)(?:\.|$)", re.IGNORECASE
)
_RE_CAUSAL_LED_TO = re.compile(
    r"(.+?)\s+(?:led\s+to|caused|resulted\s+in)\s+", re.IGNORECASE
)

_RE_PRESCRIPTIVE_METHOD = re.compile(
    r"\b(?:should|must|recommend|need\s+to|ought\s+to)\s+(.+?)(?:\.|$)",
    re.IGNORECASE,
)


# ===========================================================================
# Public API
# ===========================================================================


def plan_investigation(
    claims: List[Claim],
    top_k: int = 5,
) -> Plan:
    """Rank claims by expected information gain and build an investigation plan.

    For the top_k highest-gain claims, creates INVESTIGATE actions whose
    payloads include the claim ID, text, computed info gain, and 5W1H
    investigation questions.

    Args:
        claims: Claims to evaluate for investigation value.
        top_k: Maximum number of claims to include in the plan.

    Returns:
        A Plan containing INVESTIGATE actions ordered by info gain (descending)
        and a total expected_info_gain across all actions.
    """
    ranked = rank_by_info_gain(claims)
    top = ranked[:top_k]

    actions: List[Action] = []
    total_gain = 0.0

    for claim, gain in top:
        if gain <= 0.0:
            continue

        questions = generate_5w1h(claim)
        question_list = [
            f"{key.upper()}: {value}"
            for key, value in questions.items()
            if value and value != "Not determined"
        ]

        action = Action(
            action_type=ActionType.INVESTIGATE,
            target_id=claim.id,
            payload={
                "claim_id": claim.id,
                "claim_text": claim.text,
                "info_gain": gain,
                "questions": question_list,
            },
            rationale=(
                f"Investigate claim with info gain {gain:.3f} bits: "
                f"'{claim.text[:80]}'"
            ),
        )
        actions.append(action)
        total_gain += gain

    rationale = (
        f"Investigation plan targeting {len(actions)} claims "
        f"with total expected info gain of {total_gain:.3f} bits."
    )

    return Plan(
        actions=actions,
        expected_info_gain=total_gain,
        rationale=rationale,
    )


def propose_updates(
    result: PerceptionResult,
    state: WorldState,
) -> List[Action]:
    """Compare perception output to current world state and propose actions.

    For each entity in the perception result:
        - If not present in the state (by name match), propose CREATE_ENTITY.
        - If present but properties differ, propose UPDATE_ENTITY.
    For each claim in the result, propose CREATE_CLAIM.
    For each relationship in the result, propose CREATE_RELATIONSHIP.

    Args:
        result: Output of the Perception module.
        state: Current world state snapshot.

    Returns:
        List of proposed Action objects, each with a rationale.
    """
    actions: List[Action] = []

    # Build a name-based lookup of existing entities for matching
    existing_by_name: dict[str, Entity] = {}
    for entity in state.entities.values():
        existing_by_name[entity.name.lower()] = entity
        for alias in entity.aliases:
            existing_by_name[alias.lower()] = entity

    # --- Entity proposals ---
    for entity in result.entities:
        key = entity.name.lower()
        existing = existing_by_name.get(key)

        if existing is None:
            actions.append(Action(
                action_type=ActionType.CREATE_ENTITY,
                target_id=entity.id,
                payload={
                    "id": entity.id,
                    "name": entity.name,
                    "category": entity.category.value
                        if hasattr(entity.category, "value")
                        else str(entity.category),
                    "aliases": entity.aliases,
                    "properties": entity.properties,
                    "created": entity.created,
                    "updated": entity.updated,
                },
                rationale=(
                    f"New entity '{entity.name}' "
                    f"({entity.category.value}) not found in world state."
                ),
            ))
        else:
            # Check if properties differ
            if entity.properties and entity.properties != existing.properties:
                merged = dict(existing.properties)
                merged.update(entity.properties)
                actions.append(Action(
                    action_type=ActionType.UPDATE_ENTITY,
                    target_id=existing.id,
                    payload={
                        "properties": merged,
                    },
                    rationale=(
                        f"Entity '{existing.name}' has new/changed properties: "
                        f"{set(entity.properties.keys()) - set(existing.properties.keys())}."
                    ),
                ))

    # --- Claim proposals ---
    for claim in result.claims:
        actions.append(Action(
            action_type=ActionType.CREATE_CLAIM,
            target_id=claim.id,
            payload={
                "id": claim.id,
                "text": claim.text,
                "claim_type": claim.claim_type.value
                    if hasattr(claim.claim_type, "value")
                    else str(claim.claim_type),
                "uncertainty": {
                    "belief": claim.uncertainty.belief,
                    "disbelief": claim.uncertainty.disbelief,
                    "uncertainty": claim.uncertainty.uncertainty,
                    "sample_size": claim.uncertainty.sample_size,
                },
                "entity_ids": claim.entity_ids,
                "evidence_ids": claim.evidence_ids,
                "created": claim.created,
                "updated": claim.updated,
                "tags": claim.tags,
                "five_w1h": claim.five_w1h,
            },
            rationale=(
                f"New claim extracted: '{claim.text[:80]}'."
            ),
        ))

    # --- Relationship proposals ---
    for rel in result.relationships:
        actions.append(Action(
            action_type=ActionType.CREATE_RELATIONSHIP,
            target_id=rel.id,
            payload={
                "id": rel.id,
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "rel_type": rel.rel_type.value
                    if hasattr(rel.rel_type, "value")
                    else str(rel.rel_type),
                "evidence_ids": rel.evidence_ids,
                "confidence": rel.confidence,
                "created": rel.created,
            },
            rationale=(
                f"New relationship ({rel.rel_type.value}) "
                f"between entities {rel.source_id[:8]}... "
                f"and {rel.target_id[:8]}..."
            ),
        ))

    return actions


def execute(action: Action, db: Database) -> Action:
    """Execute an approved action against the database.

    Based on the action_type, performs the appropriate database operation.
    Updates action.status to "executed" on success or "failed" on error.

    Args:
        action: The action to execute (should have status "proposed" or "approved").
        db: Database instance for persistence.

    Returns:
        The action with updated status (and error info in rationale if failed).
    """
    try:
        if action.action_type == ActionType.CREATE_ENTITY:
            _execute_create_entity(action, db)

        elif action.action_type == ActionType.UPDATE_ENTITY:
            _execute_update_entity(action, db)

        elif action.action_type == ActionType.CREATE_CLAIM:
            _execute_create_claim(action, db)

        elif action.action_type == ActionType.UPDATE_CLAIM:
            _execute_update_claim(action, db)

        elif action.action_type == ActionType.CREATE_RELATIONSHIP:
            _execute_create_relationship(action, db)

        elif action.action_type in (
            ActionType.INVESTIGATE,
            ActionType.REDACT,
            ActionType.BLOCK,
        ):
            db.log_action(action)

        action.status = "executed"

    except Exception as exc:
        action.status = "failed"
        action.rationale = f"{action.rationale} | ERROR: {exc}"
        logger.error("Action %s failed: %s", action.id, exc)

    return action


def generate_5w1h(claim: Claim) -> dict[str, str]:
    """Decompose a claim into 5W1H investigation questions.

    Uses heuristic pattern matching to extract Who, What, When, Where,
    Why, and How from the claim text and metadata. Falls back to
    "Not determined" for dimensions that cannot be extracted.

    Args:
        claim: The claim to decompose.

    Returns:
        Dict with keys "who", "what", "when", "where", "why", "how".
    """
    text = claim.text

    # --- WHO: Extract entity names (capitalized phrases, org patterns) ---
    who_parts: List[str] = []
    for match in _RE_CAPITALIZED_WORDS.finditer(text):
        candidate = match.group(1)
        # Accept multi-word capitalized phrases or org-pattern words
        if len(candidate.split()) >= 2 or _RE_ORG_PATTERNS.search(candidate):
            who_parts.append(candidate)
    who = ", ".join(who_parts) if who_parts else "Not determined"

    # --- WHAT: The claim text itself (first sentence or truncated) ---
    what = text.split(".")[0].strip() if "." in text else text.strip()

    # --- WHEN: Date/time patterns ---
    when_matches = _RE_DATE_TIME.findall(text)
    when = ", ".join(when_matches) if when_matches else "Not determined"

    # --- WHERE: Location patterns and known country names ---
    where_parts: List[str] = []
    for match in _RE_LOCATION.finditer(text):
        where_parts.append(match.group(1))
    # Also check for country names directly in text
    for country in _COUNTRY_NAMES:
        if country in text:
            if country not in where_parts:
                where_parts.append(country)
    where = ", ".join(where_parts) if where_parts else "Not determined"

    # --- WHY: For CAUSAL claims, extract the cause ---
    why = "Not determined"
    if claim.claim_type == ClaimType.CAUSAL:
        match = _RE_CAUSAL_BECAUSE.search(text)
        if match:
            why = match.group(1).strip()
        else:
            match = _RE_CAUSAL_LED_TO.search(text)
            if match:
                why = match.group(1).strip()

    # --- HOW: For PRESCRIPTIVE claims, extract the method ---
    how = "Not determined"
    if claim.claim_type == ClaimType.PRESCRIPTIVE:
        match = _RE_PRESCRIPTIVE_METHOD.search(text)
        if match:
            how = match.group(1).strip()

    return {
        "who": who,
        "what": what,
        "when": when,
        "where": where,
        "why": why,
        "how": how,
    }


def rank_by_info_gain(claims: List[Claim]) -> List[Tuple[Claim, float]]:
    """Return claims sorted by expected information gain, descending.

    Args:
        claims: Claims to rank.

    Returns:
        List of (claim, info_gain) tuples, highest gain first.
    """
    scored = [(claim, expected_info_gain(claim)) for claim in claims]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def summarize_plan(plan: Plan) -> str:
    """Generate a human-readable summary of an investigation plan.

    Args:
        plan: The plan to summarize.

    Returns:
        Formatted multi-line string describing the plan.
    """
    lines = [
        f"Investigation Plan "
        f"({len(plan.actions)} actions, "
        f"expected gain: {plan.expected_info_gain:.2f} bits)"
    ]
    for i, action in enumerate(plan.actions, start=1):
        gain = action.payload.get("info_gain", 0.0)
        lines.append(
            f"  {i}. [{action.action_type.value}] "
            f"{action.rationale} (gain: {gain:.3f})"
        )
    return "\n".join(lines) + "\n"


# ===========================================================================
# Private execution helpers
# ===========================================================================


def _execute_create_entity(action: Action, db: Database) -> None:
    """Create a new entity from action payload."""
    from ewm.types import Entity, EntityCategory

    payload = action.payload
    category = payload.get("category", "concept")
    if isinstance(category, str):
        category = EntityCategory(category)

    entity = Entity(
        id=payload.get("id", ""),
        name=payload.get("name", ""),
        category=category,
        aliases=payload.get("aliases", []),
        properties=payload.get("properties", {}),
        created=payload.get("created", ""),
        updated=payload.get("updated", ""),
    )
    db.save_entity(entity)
    db.log_action(action)


def _execute_update_entity(action: Action, db: Database) -> None:
    """Update an existing entity's fields from action payload."""
    entity = db.get_entity(action.target_id)
    if entity is None:
        raise ValueError(f"Entity '{action.target_id}' not found for update")

    payload = action.payload
    if "name" in payload:
        entity.name = payload["name"]
    if "properties" in payload:
        entity.properties.update(payload["properties"])
    if "aliases" in payload:
        for alias in payload["aliases"]:
            if alias not in entity.aliases:
                entity.aliases.append(alias)

    db.save_entity(entity)
    db.log_action(action)


def _execute_create_claim(action: Action, db: Database) -> None:
    """Create a new claim from action payload."""
    from ewm.types import Claim, ClaimType, Uncertainty

    payload = action.payload
    claim_type = payload.get("claim_type", "factual")
    if isinstance(claim_type, str):
        claim_type = ClaimType(claim_type)

    # Reconstruct uncertainty from payload dict
    unc_data = payload.get("uncertainty", {})
    if isinstance(unc_data, dict) and unc_data:
        uncertainty = Uncertainty(
            belief=unc_data.get("belief", 1 / 3),
            disbelief=unc_data.get("disbelief", 1 / 3),
            uncertainty=unc_data.get("uncertainty", 1 / 3),
            sample_size=unc_data.get("sample_size", 2.0),
        )
    else:
        uncertainty = Uncertainty.uniform()

    claim = Claim(
        id=payload.get("id", ""),
        text=payload.get("text", ""),
        claim_type=claim_type,
        uncertainty=uncertainty,
        entity_ids=payload.get("entity_ids", []),
        evidence_ids=payload.get("evidence_ids", []),
        created=payload.get("created", ""),
        updated=payload.get("updated", ""),
        tags=payload.get("tags", []),
        five_w1h=payload.get("five_w1h", {}),
    )
    db.save_claim(claim)
    db.log_action(action)


def _execute_update_claim(action: Action, db: Database) -> None:
    """Update an existing claim's fields from action payload."""
    claim = db.get_claim(action.target_id)
    if claim is None:
        raise ValueError(f"Claim '{action.target_id}' not found for update")

    payload = action.payload
    if "text" in payload:
        claim.text = payload["text"]
    if "tags" in payload:
        claim.tags = payload["tags"]
    if "five_w1h" in payload:
        claim.five_w1h = payload["five_w1h"]
    if "uncertainty" in payload:
        unc_data = payload["uncertainty"]
        if isinstance(unc_data, dict):
            claim.uncertainty = Uncertainty(
                belief=unc_data.get("belief", claim.uncertainty.belief),
                disbelief=unc_data.get("disbelief", claim.uncertainty.disbelief),
                uncertainty=unc_data.get("uncertainty", claim.uncertainty.uncertainty),
                sample_size=unc_data.get("sample_size", claim.uncertainty.sample_size),
            )

    db.save_claim(claim)
    db.log_action(action)


def _execute_create_relationship(action: Action, db: Database) -> None:
    """Create a new relationship from action payload."""
    from ewm.types import Relationship, RelationshipType

    payload = action.payload
    rel_type = payload.get("rel_type", "similar_to")
    if isinstance(rel_type, str):
        rel_type = RelationshipType(rel_type)

    rel = Relationship(
        id=payload.get("id", ""),
        source_id=payload.get("source_id", ""),
        target_id=payload.get("target_id", ""),
        rel_type=rel_type,
        evidence_ids=payload.get("evidence_ids", []),
        confidence=payload.get("confidence", 0.5),
        created=payload.get("created", ""),
    )
    db.save_relationship(rel)
    db.log_action(action)
