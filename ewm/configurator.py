"""
Epistemic World Model -- Configurator Module
==============================================

The Configurator is LeCun Module 1 -- executive control.  It orchestrates the
full 6-module pipeline and adjusts module behavior based on the task type.
This is the main entry point for all operations.

Pipeline flow:
    Input text -> Perception -> Memory.store -> WorldModel.integrate
               -> Cost.assess -> Actor.propose -> Output

Task types:
    ingest        -- Full pipeline: perceive, store, integrate, assess
    investigate   -- Load world state, rank claims by info gain, plan
    query         -- Load world state, recall from memory
    audit         -- Load world state, check claims for cost violations
    session_sync  -- Perceive input, store, generate context

Design principles:
    - Single entry point: run() handles all task types
    - Convenience wrappers for each task type
    - Configuration is per-task with overridable defaults
    - Errors are caught and returned as status="error" results
    - stdlib only, Python 3.9+ compatible
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ewm.actor import plan_investigation, propose_updates, summarize_plan
from ewm.cost import assess_action_cost, check_claim_requirements
from ewm.db import Database
from ewm.memory import context_for_session, forget_stale, recall, store
from ewm.perception import perceive
from ewm.types import SourceType, WorldState
from ewm.world_model import integrate_evidence


# ---------------------------------------------------------------------------
# Task configuration defaults
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ingest": {
        "source_type": "document",
        "store_results": True,
        "assess_cost": True,
        "integrate_evidence": True,
    },
    "investigate": {
        "top_k": 5,
        "min_info_gain": 0.1,
    },
    "query": {
        "limit": 20,
        "include_context": False,
    },
    "audit": {
        "check_red_lines": True,
        "check_requirements": True,
        "include_stale": True,
    },
    "session_sync": {
        "source_type": "direct_observation",
        "generate_context": True,
        "session_topic": "",
    },
}


# ===========================================================================
# Configuration
# ===========================================================================


def get_task_config(task: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the configuration for a given task type.

    Starts from the default config for the task and merges any overrides
    on top.  Unknown task types return an empty dict (with overrides applied
    if provided).

    Args:
        task: Task type name (e.g. "ingest", "query").
        overrides: Optional dict of values to merge into the defaults.

    Returns:
        Merged configuration dict.
    """
    config = dict(TASK_CONFIGS.get(task, {}))
    if overrides:
        config.update(overrides)
    return config


# ===========================================================================
# Main orchestrator
# ===========================================================================


def run(
    task: str,
    input_text: str,
    db: Database,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full pipeline for a given task type.

    This is the single entry point for all EWM operations.  It dispatches
    to the appropriate task handler based on the task string.

    Args:
        task: One of "ingest", "investigate", "query", "audit", "session_sync".
        input_text: The input text to process (may be empty for some tasks).
        db: Database instance for persistence and retrieval.
        config: Optional overrides merged into the task's default config.

    Returns:
        A dict with keys: task, status, result, stats, config_used.
        On error, status is "error" and an "error" key contains the message.
    """
    merged_config = get_task_config(task, config)

    try:
        if task == "ingest":
            result = _run_ingest(input_text, db, merged_config)
        elif task == "investigate":
            result = _run_investigate(db, merged_config)
        elif task == "query":
            result = _run_query(input_text, db, merged_config)
        elif task == "audit":
            result = _run_audit(db, merged_config)
        elif task == "session_sync":
            result = _run_session_sync(input_text, db, merged_config)
        else:
            return {
                "task": task,
                "status": "error",
                "error": f"Unknown task type: {task}",
                "stats": db.stats(),
                "config_used": merged_config,
            }

        return {
            "task": task,
            "status": "completed",
            "result": result,
            "stats": db.stats(),
            "config_used": merged_config,
        }

    except Exception as exc:
        return {
            "task": task,
            "status": "error",
            "error": str(exc),
            "stats": db.stats(),
            "config_used": merged_config,
        }


# ===========================================================================
# Convenience wrappers
# ===========================================================================


def ingest(
    input_text: str,
    db: Database,
    source_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for the ingest task.

    Args:
        input_text: Text to perceive and store.
        db: Database instance.
        source_type: Optional source type override (e.g. "expert_testimony").

    Returns:
        Result of run("ingest", ...).
    """
    config = None
    if source_type is not None:
        config = {"source_type": source_type}
    return run("ingest", input_text, db, config)


def investigate(db: Database, top_k: int = 5) -> Dict[str, Any]:
    """Convenience wrapper for the investigate task.

    Args:
        db: Database instance.
        top_k: Maximum number of claims to investigate.

    Returns:
        Result of run("investigate", ...).
    """
    return run("investigate", "", db, {"top_k": top_k})


def query(query_text: str, db: Database, limit: int = 20) -> Dict[str, Any]:
    """Convenience wrapper for the query task.

    Args:
        query_text: Free-text search query.
        db: Database instance.
        limit: Maximum number of results per category.

    Returns:
        Result of run("query", ...).
    """
    return run("query", query_text, db, {"limit": limit})


def audit(db: Database) -> Dict[str, Any]:
    """Convenience wrapper for the audit task.

    Args:
        db: Database instance.

    Returns:
        Result of run("audit", ...).
    """
    return run("audit", "", db)


def session_sync(transcript: str, db: Database) -> Dict[str, Any]:
    """Convenience wrapper for the session_sync task.

    Args:
        transcript: Session transcript text to process.
        db: Database instance.

    Returns:
        Result of run("session_sync", ...).
    """
    return run("session_sync", transcript, db)


# ===========================================================================
# Task implementations
# ===========================================================================


def _run_ingest(
    input_text: str,
    db: Database,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the ingest pipeline.

    Steps:
        1. Perceive entities, claims, evidence from input text
        2. Store perception result to database
        3. Optionally integrate evidence into world model
        4. Optionally assess proposed actions for cost violations

    Args:
        input_text: Text to process.
        db: Database instance.
        config: Merged task configuration.

    Returns:
        Dict with perception summary, store summary, and assessments.
    """
    source_type = SourceType(config["source_type"])
    result = perceive(input_text, source_type=source_type, source_id="configurator")
    summary = store(db, result)

    integration_info: Optional[str] = None
    if config.get("integrate_evidence", False):
        state = db.load_world_state()
        for claim, evidence in zip(result.claims, result.evidence):
            if claim.id in state.claims:
                state = integrate_evidence(state, evidence, claim.id, "support")
        integration_info = "evidence integrated"

    assessments: List[Dict[str, Any]] = []
    if config.get("assess_cost", False):
        state = db.load_world_state()
        actions = propose_updates(result, state)
        for action in actions:
            assessment = assess_action_cost(action, state)
            assessments.append({
                "action": assessment.action.action_type.value,
                "blocked": assessment.blocked,
                "violations": len(assessment.violations),
            })

    return {
        "entities": len(result.entities),
        "claims": len(result.claims),
        "evidence": len(result.evidence),
        "relationships": len(result.relationships),
        "store_summary": summary,
        "integration": integration_info,
        "assessments": assessments,
    }


def _run_investigate(
    db: Database,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the investigate pipeline.

    Steps:
        1. Load current world state
        2. Rank all claims by information gain
        3. Build investigation plan for top-k claims
        4. Summarize the plan

    Args:
        db: Database instance.
        config: Merged task configuration.

    Returns:
        Dict with plan summary, action count, and total expected gain.
    """
    state = db.load_world_state()
    claims = list(state.claims.values())
    top_k = config.get("top_k", 5)
    plan = plan_investigation(claims, top_k=top_k)
    summary = summarize_plan(plan)

    return {
        "plan": summary,
        "actions": len(plan.actions),
        "total_gain": plan.expected_info_gain,
    }


def _run_query(
    input_text: str,
    db: Database,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the query pipeline.

    Steps:
        1. Recall matching entities and claims from memory
        2. Optionally generate session context

    Args:
        input_text: Search query text.
        db: Database instance.
        config: Merged task configuration.

    Returns:
        Dict with matched entities, claims, total count, and optional context.
    """
    limit = config.get("limit", 20)
    mem = recall(db, input_text, limit=limit)

    result: Dict[str, Any] = {
        "entities": [e.name for e in mem.entities],
        "claims": [c.text[:100] for c in mem.claims],
        "total": len(mem.entities) + len(mem.claims),
    }

    if config.get("include_context", False):
        result["context"] = context_for_session(db, input_text)

    return result


def _run_audit(
    db: Database,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the audit pipeline.

    Steps:
        1. Load world state
        2. Check each claim for requirement violations
        3. Optionally prune stale uncertain claims

    Args:
        db: Database instance.
        config: Merged task configuration.

    Returns:
        Dict with total claims, violation count, stale count, and details.
    """
    state = db.load_world_state()
    violations: List[Dict[str, Any]] = []

    if config.get("check_requirements", True):
        for claim in state.claims.values():
            evidence_count = len(claim.evidence_ids)
            claim_violations = check_claim_requirements(claim, evidence_count)
            for v in claim_violations:
                violations.append({
                    "claim_id": claim.id,
                    "claim_text": claim.text[:80],
                    "rule": v.rule,
                    "severity": v.severity,
                    "description": v.description,
                })

    stale_forgotten = 0
    if config.get("include_stale", True):
        stale_result = forget_stale(db, max_age_days=90)
        stale_forgotten = stale_result.get("claims_forgotten", 0)

    return {
        "total_claims": len(state.claims),
        "violations": len(violations),
        "stale_forgotten": stale_forgotten,
        "details": violations,
    }


def _run_session_sync(
    input_text: str,
    db: Database,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the session_sync pipeline.

    Steps:
        1. Perceive entities and claims from session transcript
        2. Store perception result to database
        3. Generate session context block

    Args:
        input_text: Session transcript text.
        db: Database instance.
        config: Merged task configuration.

    Returns:
        Dict with extraction summary, context text, and context length.
    """
    result = perceive(
        input_text,
        source_type=SourceType.DIRECT_OBSERVATION,
        source_id="session_sync",
    )
    summary = store(db, result)
    session_topic = config.get("session_topic", "")
    context = context_for_session(db, session_topic)

    return {
        "extracted": summary,
        "context": context,
        "context_length": len(context),
    }
