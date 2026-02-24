"""
Epistemic World Model — Cost Module
=====================================

The Cost module is the ethical and safety gatekeeper of the 6-module LeCun
architecture.  Every proposed Action passes through this module before
execution, and the module's assessment determines whether the action proceeds,
is modified, or is blocked outright.

The module implements a DUAL evaluation architecture:

1. **Red Lines** (immutable safety boundary)
   Hard-coded pattern matchers that detect dangerous content: leaked
   credentials, PII exposure, destructive commands, code injection.  These
   are compiled once at module load and CANNOT be relaxed, overridden, or
   disabled by any code path — including the learning system.  A red-line
   violation always blocks the action.

2. **Trainable Critic** (calibratable quality thresholds)
   Soft thresholds that check whether claims have sufficient evidence for
   their type and confidence level.  These thresholds can be tightened or
   loosened via feedback (see ``record_feedback`` and ``calibrate_thresholds``),
   but the adjustment is bounded: confidence caps never exceed 0.95 and
   minimum evidence never drops below 0.

The single entry point for other modules is ``assess_action_cost``, which
returns a ``CostAssessment`` containing the original action, whether it was
blocked, and all violations found.

Design principle: the critic is STATELESS.  Feedback stores are passed in
and out as plain dicts, never held in module-level mutable state.  This
makes the module easy to test, serialize, and reason about.
"""

from __future__ import annotations

import re
from copy import deepcopy
import logging
from datetime import datetime, timezone

from ewm.types import (
    Action,
    ActionType,
    Claim,
    ClaimType,
    CostAssessment,
    CostViolation,
    Uncertainty,
    WorldState,
)

# ---------------------------------------------------------------------------
# Red Lines — IMMUTABLE safety patterns
# ---------------------------------------------------------------------------
# These patterns detect content that must NEVER appear in system output.
# They are compiled once at import time for performance and cannot be
# modified at runtime.  Any match results in a hard block.

DEFAULT_RED_LINES: list[dict[str, str]] = [
    {"pattern": r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S+", "description": "Exposed password"},
    {"pattern": r"(?i)\b(api[_-]?key|apikey)\s*[:=]\s*\S+", "description": "Exposed API key"},
    {"pattern": r"(?i)\b(secret|token)\s*[:=]\s*\S+", "description": "Exposed secret/token"},
    {"pattern": r"AKIA[0-9A-Z]{16}", "description": "AWS access key"},
    {"pattern": r"-----BEGIN\s+(RSA\s+|EC\s+)?PRIVATE\s+KEY-----", "description": "Private key material"},
    {"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "description": "Social Security Number pattern"},
    {"pattern": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "description": "Credit card number pattern"},
    {"pattern": r"(?i)(drop\s+table|delete\s+from|truncate\s+table)", "description": "SQL destructive operation"},
    {"pattern": r"(?i)(rm\s+-rf\s+/|format\s+c:)", "description": "Destructive system command"},
    {"pattern": r"(?i)exec\s*\(\s*['\"]", "description": "Code injection attempt"},
]

_COMPILED_RED_LINES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r["pattern"]), r["description"]) for r in DEFAULT_RED_LINES
]

# ---------------------------------------------------------------------------
# Trainable Critic — claim type requirements
# ---------------------------------------------------------------------------
# These thresholds determine how much evidence a claim needs before the
# system accepts it at a given confidence level.  They are the "soft"
# counterpart to the hard red lines and can be calibrated via feedback.
#
# Keys:
#   min_evidence                  — minimum evidence items needed
#   max_confidence_without_evidence — ceiling on confidence when below min_evidence
#   requires_source (optional)    — whether at least one source is mandatory
#   confidence_cap (optional)     — absolute ceiling on confidence for this type

CLAIM_REQUIREMENTS: dict[str, dict] = {
    "factual": {"min_evidence": 1, "max_confidence_without_evidence": 0.5},
    "statistical": {"min_evidence": 1, "max_confidence_without_evidence": 0.3, "requires_source": True},
    "causal": {"min_evidence": 2, "max_confidence_without_evidence": 0.3},
    "predictive": {"min_evidence": 1, "max_confidence_without_evidence": 0.4, "confidence_cap": 0.85},
    "accusatory": {"min_evidence": 2, "max_confidence_without_evidence": 0.2, "confidence_cap": 0.75},
    "diagnostic": {"min_evidence": 1, "max_confidence_without_evidence": 0.4},
    "prescriptive": {"min_evidence": 0, "max_confidence_without_evidence": 0.6},
}


# ===========================================================================
# Public API
# ===========================================================================


def check_red_lines(
    text: str,
    extra_patterns: list[dict[str, str]] | None = None,
) -> list[CostViolation]:
    """Scan text against all red-line patterns and return any violations.

    This function is the hard safety boundary of the entire system.  It
    CANNOT be relaxed by learning, feedback, or configuration.

    Args:
        text: The content to scan for dangerous patterns.
        extra_patterns: Additional patterns to check beyond the defaults.
            Each dict must have ``"pattern"`` and ``"description"`` keys.

    Returns:
        A list of ``CostViolation`` objects with ``severity="red_line"``
        for each match found.  An empty list means the text is clean.
    """
    violations: list[CostViolation] = []

    # Check all compiled default patterns
    for compiled_pattern, description in _COMPILED_RED_LINES:
        if compiled_pattern.search(text):
            violations.append(
                CostViolation(
                    rule="red_line",
                    description=description,
                    severity="red_line",
                )
            )

    # Check any extra patterns provided at call time
    if extra_patterns:
        for pattern_def in extra_patterns:
            try:
                compiled = re.compile(pattern_def["pattern"])
            except re.error:
                # Invalid regex — skip it but don't crash the safety check
                continue
            if compiled.search(text):
                violations.append(
                    CostViolation(
                        rule="red_line",
                        description=pattern_def.get("description", "Custom red-line pattern matched"),
                        severity="red_line",
                    )
                )

    return violations


def check_claim_requirements(
    claim: Claim,
    evidence_count: int,
) -> list[CostViolation]:
    """Check whether a claim meets the evidence requirements for its type.

    This implements the trainable critic — thresholds that can be adjusted
    via ``calibrate_thresholds`` based on user feedback.

    Args:
        claim: The claim to evaluate.
        evidence_count: Number of evidence items supporting this claim.

    Returns:
        A list of ``CostViolation`` objects with ``severity="caution"``
        for each requirement that is not met.  Empty means the claim
        passes all checks.
    """
    violations: list[CostViolation] = []

    # Resolve claim type to its string key for lookup
    claim_type_key = (
        claim.claim_type.value
        if isinstance(claim.claim_type, ClaimType)
        else str(claim.claim_type)
    )

    requirements = CLAIM_REQUIREMENTS.get(claim_type_key)
    if requirements is None:
        # Unknown claim type — no requirements to check
        return violations

    min_evidence: int = requirements.get("min_evidence", 0)
    max_conf_no_evidence: float = requirements.get("max_confidence_without_evidence", 1.0)
    requires_source: bool = requirements.get("requires_source", False)
    confidence_cap: float | None = requirements.get("confidence_cap")

    claim_confidence = claim.uncertainty.confidence

    # Check 1: insufficient evidence with high confidence
    if evidence_count < min_evidence and claim_confidence > max_conf_no_evidence:
        violations.append(
            CostViolation(
                rule="insufficient_evidence",
                description=(
                    f"{claim_type_key.capitalize()} claim has confidence "
                    f"{claim_confidence:.2f} but only {evidence_count} evidence item(s) "
                    f"(requires {min_evidence}, max confidence without evidence: "
                    f"{max_conf_no_evidence:.2f})"
                ),
                severity="caution",
            )
        )

    # Check 2: confidence exceeds the hard cap for this claim type
    if confidence_cap is not None and claim_confidence > confidence_cap:
        violations.append(
            CostViolation(
                rule="confidence_cap_exceeded",
                description=(
                    f"{claim_type_key.capitalize()} claim has confidence "
                    f"{claim_confidence:.2f} exceeding cap of {confidence_cap:.2f}"
                ),
                severity="caution",
            )
        )

    # Check 3: source required but no evidence at all
    if requires_source and evidence_count == 0:
        violations.append(
            CostViolation(
                rule="source_required",
                description=(
                    f"{claim_type_key.capitalize()} claim requires at least one source, "
                    f"but no evidence was provided"
                ),
                severity="caution",
            )
        )

    return violations


def assess_action_cost(
    action: Action,
    state: WorldState,
    extra_red_lines: list[dict[str, str]] | None = None,
) -> CostAssessment:
    """Evaluate whether a proposed action should proceed.

    This is the **single entry point** other modules should call.  It
    combines red-line safety checks with trainable critic thresholds to
    produce a holistic assessment.

    Steps:
        1. Scan the action's rationale and payload for red-line violations.
        2. If the action creates or updates a claim, check claim requirements.
        3. Safety actions (BLOCK, REDACT) are always allowed.
        4. Aggregate violations and determine whether the action is blocked.

    Args:
        action: The proposed action to evaluate.
        state: Current world state (used to look up existing claims).
        extra_red_lines: Additional red-line patterns beyond the defaults.

    Returns:
        A ``CostAssessment`` with the action, blocked flag, and all
        violations found.
    """
    violations: list[CostViolation] = []

    # --- Step 1: Red-line checks on textual content -------------------------
    # Scan the action's rationale
    if action.rationale:
        violations.extend(check_red_lines(action.rationale, extra_red_lines))

    # Scan the action's payload (converted to string for pattern matching)
    payload_text = str(action.payload) if action.payload else ""
    if payload_text:
        violations.extend(check_red_lines(payload_text, extra_red_lines))

    # --- Step 2: Claim requirement checks -----------------------------------
    if action.action_type in (ActionType.CREATE_CLAIM, ActionType.UPDATE_CLAIM):
        claim = _resolve_claim(action, state)
        if claim is not None:
            evidence_count = len(claim.evidence_ids)
            violations.extend(check_claim_requirements(claim, evidence_count))

    # --- Step 3: Safety actions always pass ----------------------------------
    if action.action_type in (ActionType.BLOCK, ActionType.REDACT):
        # Safety actions cannot be blocked — they ARE the safety mechanism.
        # Clear any violations that might have been found (e.g., if the
        # rationale mentions the dangerous content being blocked).
        return CostAssessment(
            action=action,
            blocked=False,
            violations=tuple(violations),
        )

    # --- Step 4: Aggregate and decide ----------------------------------------
    blocked = any(v.severity == "red_line" for v in violations)

    return CostAssessment(
        action=action,
        blocked=blocked,
        violations=tuple(violations),
    )


def record_feedback(
    claim_type: str,
    was_appropriate: bool,
    feedback_store: dict | None = None,
) -> dict:
    """Record user feedback on a cost decision for critic calibration.

    This is how the trainable critic learns.  Each piece of feedback
    increments a counter for the given claim type, tracking whether the
    system's cost assessment was appropriate or not.

    The feedback store is passed in and returned — this function does NOT
    maintain any module-level mutable state.

    Note: feedback can NEVER relax red lines.  Red lines are structural,
    not statistical.

    Args:
        claim_type: The claim type the feedback applies to (e.g. "factual").
        was_appropriate: Whether the cost decision was appropriate (True)
            or not (False).
        feedback_store: Existing feedback data to update, or None to
            create a new store.

    Returns:
        The updated feedback store dict with structure::

            {
                "factual": {"appropriate": 5, "inappropriate": 1},
                "causal": {"appropriate": 12, "inappropriate": 0},
                ...
            }
    """
    if feedback_store is None:
        feedback_store = {}

    if claim_type not in feedback_store:
        feedback_store[claim_type] = {"appropriate": 0, "inappropriate": 0}

    if was_appropriate:
        feedback_store[claim_type]["appropriate"] += 1
    else:
        feedback_store[claim_type]["inappropriate"] += 1

    return feedback_store


def calibrate_thresholds(feedback_store: dict) -> dict[str, dict]:
    """Use accumulated feedback to adjust claim requirement thresholds.

    This implements the critic's learning loop.  Based on the ratio of
    appropriate to inappropriate feedback for each claim type, thresholds
    are tightened (if the system is too permissive) or loosened (if it is
    too restrictive).

    Rules:
        - >70% inappropriate feedback: tighten (reduce confidence caps by 0.05,
          reduce max_confidence_without_evidence by 0.05).
        - >90% appropriate feedback AND >10 total samples: loosen slightly
          (increase confidence caps by 0.02, increase
          max_confidence_without_evidence by 0.02).
        - ``min_evidence`` never drops below 0.
        - ``confidence_cap`` and ``max_confidence_without_evidence`` never
          exceed 0.95.

    This function returns a NEW requirements dict — the module-level
    ``CLAIM_REQUIREMENTS`` constant is never modified.

    Args:
        feedback_store: Feedback data as produced by ``record_feedback``.

    Returns:
        A new requirements dict with adjusted thresholds.
    """
    # Deep copy so we never mutate the module constant
    adjusted = deepcopy(CLAIM_REQUIREMENTS)

    for claim_type, counters in feedback_store.items():
        if claim_type not in adjusted:
            # Feedback for a claim type we don't have requirements for — skip
            continue

        appropriate = counters.get("appropriate", 0)
        inappropriate = counters.get("inappropriate", 0)
        total = appropriate + inappropriate

        if total == 0:
            continue

        inappropriate_ratio = inappropriate / total
        appropriate_ratio = appropriate / total

        reqs = adjusted[claim_type]

        # Tighten: too many inappropriate assessments
        if inappropriate_ratio > 0.7:
            # Reduce confidence thresholds
            if "confidence_cap" in reqs:
                reqs["confidence_cap"] = max(0.0, reqs["confidence_cap"] - 0.05)
            if "max_confidence_without_evidence" in reqs:
                reqs["max_confidence_without_evidence"] = max(
                    0.0, reqs["max_confidence_without_evidence"] - 0.05
                )

        # Loosen: consistently appropriate AND sufficient sample size
        elif appropriate_ratio > 0.9 and total > 10:
            if "confidence_cap" in reqs:
                reqs["confidence_cap"] = min(0.95, reqs["confidence_cap"] + 0.02)
            if "max_confidence_without_evidence" in reqs:
                reqs["max_confidence_without_evidence"] = min(
                    0.95, reqs["max_confidence_without_evidence"] + 0.02
                )

        # Enforce floor on min_evidence
        if "min_evidence" in reqs:
            reqs["min_evidence"] = max(0, reqs["min_evidence"])

    return adjusted


def cost_summary(
    violations: list[CostViolation] | tuple[CostViolation, ...],
) -> str:
    """Produce a human-readable summary of cost violations.

    Format: ``"N violation(s): X red line(s), Y caution(s), Z info(s)"``

    For red-line violations, the description is included in the output
    because these represent critical safety issues that must be surfaced
    to the operator.

    Args:
        violations: Sequence of violations to summarize.

    Returns:
        A formatted summary string.  Returns ``"No violations"`` if
        the sequence is empty.
    """
    if not violations:
        return "No violations"

    red_lines: list[CostViolation] = []
    cautions: list[CostViolation] = []
    infos: list[CostViolation] = []

    for v in violations:
        if v.severity == "red_line":
            red_lines.append(v)
        elif v.severity == "caution":
            cautions.append(v)
        else:
            infos.append(v)

    total = len(violations)
    parts = []
    if red_lines:
        parts.append(f"{len(red_lines)} red line(s)")
    if cautions:
        parts.append(f"{len(cautions)} caution(s)")
    if infos:
        parts.append(f"{len(infos)} info(s)")

    summary = f"{total} violation(s): {', '.join(parts)}"

    # Include details for red-line violations — these are critical
    if red_lines:
        details = "; ".join(v.description for v in red_lines)
        summary += f" [RED LINES: {details}]"

    return summary


# ===========================================================================
# Internal helpers
# ===========================================================================


def _resolve_claim(action: Action, state: WorldState) -> Claim | None:
    """Resolve the claim associated with an action.

    For CREATE_CLAIM actions, the claim data is expected in the action's
    payload.  For UPDATE_CLAIM actions, we look up the existing claim in
    the world state by ``target_id`` and then overlay any payload updates.

    Args:
        action: The action whose claim we want to resolve.
        state: Current world state for looking up existing claims.

    Returns:
        The resolved ``Claim`` object, or ``None`` if the claim cannot
        be found or constructed.
    """
    # Try to find an existing claim by target_id
    if action.target_id and action.target_id in state.claims:
        claim = state.claims[action.target_id]
        # For updates, overlay any payload fields onto the existing claim
        if action.action_type == ActionType.UPDATE_CLAIM and action.payload:
            claim = _apply_claim_payload(claim, action.payload)
        return claim

    # For CREATE_CLAIM, construct a claim from payload
    if action.action_type == ActionType.CREATE_CLAIM and action.payload:
        return _claim_from_payload(action.payload)

    logging.getLogger(__name__).debug(
        "_resolve_claim: could not resolve claim for action %s (type=%s, target=%s)",
        action.id, action.action_type, action.target_id,
    )
    return None


def _claim_from_payload(payload: dict) -> Claim | None:
    """Construct a Claim from an action payload dict.

    Handles the common payload shapes:
        - ``{"claim": <Claim>}`` — the claim is already built
        - ``{"text": ..., "claim_type": ..., ...}`` — raw fields

    Args:
        payload: The action payload dict.

    Returns:
        A ``Claim`` object, or ``None`` if the payload cannot be parsed.
    """
    # Case 1: payload contains a pre-built Claim object
    if "claim" in payload and isinstance(payload["claim"], Claim):
        return payload["claim"]

    # Case 2: payload contains raw claim fields
    if "text" in payload:
        claim_type = ClaimType.FACTUAL
        raw_type = payload.get("claim_type")
        if isinstance(raw_type, ClaimType):
            claim_type = raw_type
        elif isinstance(raw_type, str):
            try:
                claim_type = ClaimType(raw_type)
            except ValueError:
                pass

        uncertainty = payload.get("uncertainty", Uncertainty.uniform())
        if not isinstance(uncertainty, Uncertainty):
            uncertainty = Uncertainty.uniform()

        evidence_ids = payload.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            evidence_ids = []

        return Claim(
            text=str(payload["text"]),
            claim_type=claim_type,
            uncertainty=uncertainty,
            evidence_ids=evidence_ids,
        )

    return None


def _apply_claim_payload(claim: Claim, payload: dict) -> Claim:
    """Apply payload updates to an existing claim, returning the modified claim.

    Only updates fields that are present in the payload.  The original
    claim object may be mutated (claims are mutable dataclasses).

    Args:
        claim: The existing claim to update.
        payload: Fields to overlay.

    Returns:
        The updated claim (same object, mutated in place).
    """
    if "uncertainty" in payload and isinstance(payload["uncertainty"], Uncertainty):
        claim.uncertainty = payload["uncertainty"]

    if "evidence_ids" in payload and isinstance(payload["evidence_ids"], list):
        claim.evidence_ids = payload["evidence_ids"]

    if "claim_type" in payload:
        raw_type = payload["claim_type"]
        if isinstance(raw_type, ClaimType):
            claim.claim_type = raw_type
        elif isinstance(raw_type, str):
            try:
                claim.claim_type = ClaimType(raw_type)
            except ValueError:
                pass

    if "text" in payload and isinstance(payload["text"], str):
        claim.text = payload["text"]

    return claim
