"""One-shot migration tool for the Epistemic World Model.

Migrates data from legacy systems into the EWM database:
  1. PromptSpeak symbols (SQLite) -> entities + claims
  2. Knowledge insights (JSONL)   -> claims + evidence
  3. Grounded learnings (JSONL)   -> claims with 5W1H evidence

Run once, then old systems become read-only archives.

Usage:
    python -m ewm.migrate
    python -m ewm.migrate --db-path /custom/path.db
    python -m ewm.migrate --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional

from ewm.db import Database
from ewm.types import (
    Claim,
    ClaimType,
    Entity,
    EntityCategory,
    Evidence,
    SourceType,
    Uncertainty,
)

# ---------------------------------------------------------------------------
# Default source locations
# ---------------------------------------------------------------------------

PROMPTSPEAK_DB = "/Volumes/OWC drive/Dev/promptspeak/mcp-server/promptspeak.db"
INSIGHTS_JSONL = "/Volumes/OWC drive/Knowledge/extracted/captured_insights.jsonl"
LEARNINGS_JSONL = "/Volumes/OWC drive/Knowledge/extracted/grounded_learnings.jsonl"

DEFAULT_DB_PATH = os.path.expanduser("~/.ewm/world_model.db")

# ---------------------------------------------------------------------------
# Category and claim-type mapping helpers
# ---------------------------------------------------------------------------

_CATEGORY_MAP: Dict[str, EntityCategory] = {
    "company": EntityCategory.ORGANIZATION,
    "technology": EntityCategory.TECHNOLOGY,
    "person": EntityCategory.PERSON,
    "project": EntityCategory.CONCEPT,
    "concept": EntityCategory.CONCEPT,
    "location": EntityCategory.LOCATION,
    "event": EntityCategory.EVENT,
}

# Patterns for _guess_claim_type, evaluated in order.
# Each entry is (compiled regex, ClaimType).
_CLAIM_PATTERNS: List[tuple] = [
    (re.compile(r"\bshould\b|\brecommend\b|\bmust\b", re.IGNORECASE), ClaimType.PRESCRIPTIVE),
    (re.compile(r"\bbecause\b|\bcaused by\b|\bled to\b", re.IGNORECASE), ClaimType.CAUSAL),
    (re.compile(r"\bpredict\b|\bforecast\b|\bwill\b", re.IGNORECASE), ClaimType.PREDICTIVE),
    (re.compile(r"\d+%|\d+\.\d+", re.IGNORECASE), ClaimType.STATISTICAL),
]


def _map_category(symbol_type: str) -> EntityCategory:
    """Map a PromptSpeak symbol type string to an EWM EntityCategory."""
    return _CATEGORY_MAP.get(symbol_type.lower(), EntityCategory.CONCEPT)


def _guess_claim_type(text: str) -> ClaimType:
    """Guess the ClaimType from claim text using simple heuristics."""
    for pattern, claim_type in _CLAIM_PATTERNS:
        if pattern.search(text):
            return claim_type
    return ClaimType.FACTUAL


# ---------------------------------------------------------------------------
# Migration: PromptSpeak symbols
# ---------------------------------------------------------------------------


def migrate_promptspeak_symbols(
    db: Database,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Migrate PromptSpeak symbols into EWM entities and claims.

    Opens the PromptSpeak SQLite database read-only, reads all symbols,
    and creates corresponding Entity and Claim records in the EWM database.
    """
    if source_path is None:
        source_path = PROMPTSPEAK_DB

    if not os.path.exists(source_path):
        return {"status": "source_not_found", "path": source_path}

    symbols_found = 0
    entities_created = 0
    claims_created = 0
    errors = 0

    try:
        src_uri = "file:{}?mode=ro".format(source_path)
        src_conn = sqlite3.connect(src_uri, uri=True)
        src_conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return {"status": "source_not_found", "path": source_path}

    try:
        # Verify the symbols table exists with expected columns
        try:
            rows = src_conn.execute("SELECT * FROM symbols").fetchall()
        except sqlite3.OperationalError:
            src_conn.close()
            return {
                "status": "completed",
                "symbols_found": 0,
                "entities_created": 0,
                "claims_created": 0,
                "errors": 0,
                "note": "symbols table not found or unexpected schema",
            }

        symbols_found = len(rows)

        for row in rows:
            try:
                row_dict = dict(row)

                symbol_type = row_dict.get("type", "")
                identifier = row_dict.get("identifier", "")
                display_name = row_dict.get("display_name", identifier)
                description = row_dict.get("description", "")
                metadata_raw = row_dict.get("metadata", "{}")

                # Parse metadata JSON
                if isinstance(metadata_raw, str):
                    try:
                        properties = json.loads(metadata_raw)
                    except (json.JSONDecodeError, TypeError):
                        properties = {}
                else:
                    properties = {}

                if not isinstance(properties, dict):
                    properties = {}

                category = _map_category(symbol_type)

                # Build aliases list from the identifier
                aliases = [identifier] if identifier else []

                entity = Entity(
                    name=display_name or identifier,
                    category=category,
                    aliases=aliases,
                    properties=properties,
                )
                db.save_entity(entity)
                entities_created += 1

                # Create a claim from the description if present
                if description:
                    claim = Claim(
                        text=description,
                        claim_type=ClaimType.FACTUAL,
                        uncertainty=Uncertainty.from_confidence(0.7),
                        entity_ids=[entity.id],
                    )
                    db.save_claim(claim)
                    claims_created += 1

            except Exception:
                errors += 1

    finally:
        src_conn.close()

    return {
        "status": "completed",
        "symbols_found": symbols_found,
        "entities_created": entities_created,
        "claims_created": claims_created,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Migration: Knowledge insights
# ---------------------------------------------------------------------------


def migrate_knowledge_insights(
    db: Database,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Migrate captured insights from JSONL into EWM claims and evidence.

    Each JSON line may contain: content/text, type/category, source,
    timestamp, confidence, entities (list).
    """
    if source_path is None:
        source_path = INSIGHTS_JSONL

    if not os.path.exists(source_path):
        return {"status": "source_not_found", "path": source_path}

    insights_found = 0
    claims_created = 0
    evidence_created = 0
    entities_created = 0
    errors = 0

    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            insights_found += 1

            try:
                data = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                errors += 1
                continue

            if not isinstance(data, dict):
                errors += 1
                continue

            # Extract text from content or text field
            insight_text = data.get("content", "") or data.get("text", "")
            if not insight_text:
                errors += 1
                continue

            confidence = data.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            confidence = max(0.0, min(1.0, float(confidence)))

            # Create evidence
            evidence = Evidence(
                source_type=SourceType.INFERENCE,
                content=insight_text,
                reliability=0.6,
            )
            db.save_evidence(evidence)
            evidence_created += 1

            # Create entity objects for listed entities
            entity_ids = []  # type: List[str]
            raw_entities = data.get("entities", [])
            if isinstance(raw_entities, list):
                for ent_name in raw_entities:
                    if isinstance(ent_name, str) and ent_name.strip():
                        entity = Entity(
                            name=ent_name.strip(),
                            category=EntityCategory.CONCEPT,
                        )
                        db.save_entity(entity)
                        entities_created += 1
                        entity_ids.append(entity.id)

            # Create claim
            claim_type = _guess_claim_type(insight_text)
            claim = Claim(
                text=insight_text,
                claim_type=claim_type,
                uncertainty=Uncertainty.from_confidence(confidence),
                evidence_ids=[evidence.id],
                entity_ids=entity_ids,
            )
            db.save_claim(claim)
            claims_created += 1

    return {
        "status": "completed",
        "insights_found": insights_found,
        "claims_created": claims_created,
        "evidence_created": evidence_created,
        "entities_created": entities_created,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Migration: Grounded learnings
# ---------------------------------------------------------------------------


def migrate_grounded_learnings(
    db: Database,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Migrate grounded learnings from JSONL into EWM claims with 5W1H.

    Each JSON line may contain: learning/text, who, what, when, where,
    why, how, source, confidence, timestamp.
    """
    if source_path is None:
        source_path = LEARNINGS_JSONL

    if not os.path.exists(source_path):
        return {"status": "source_not_found", "path": source_path}

    learnings_found = 0
    claims_created = 0
    evidence_created = 0
    errors = 0

    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            learnings_found += 1

            try:
                data = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                errors += 1
                continue

            if not isinstance(data, dict):
                errors += 1
                continue

            learning_text = data.get("learning", "") or data.get("text", "")
            if not learning_text:
                errors += 1
                continue

            confidence = data.get("confidence", 0.7)
            if not isinstance(confidence, (int, float)):
                confidence = 0.7
            confidence = max(0.0, min(1.0, float(confidence)))

            # Create evidence
            evidence = Evidence(
                source_type=SourceType.DIRECT_OBSERVATION,
                content=learning_text,
                reliability=0.7,
            )
            db.save_evidence(evidence)
            evidence_created += 1

            # Build 5W1H dict from available fields
            five_w1h = {}  # type: Dict[str, str]
            for key in ("who", "what", "when", "where", "why", "how"):
                value = data.get(key, "")
                if value:
                    five_w1h[key] = str(value)

            # Create claim
            claim = Claim(
                text=learning_text,
                claim_type=ClaimType.FACTUAL,
                uncertainty=Uncertainty.from_confidence(confidence),
                evidence_ids=[evidence.id],
                five_w1h=five_w1h,
            )
            db.save_claim(claim)
            claims_created += 1

    return {
        "status": "completed",
        "learnings_found": learnings_found,
        "claims_created": claims_created,
        "evidence_created": evidence_created,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Verification and full migration
# ---------------------------------------------------------------------------


def verify_migration(db: Database) -> Dict[str, Any]:
    """Verify the migration by checking database counts."""
    stats = db.stats()
    counts = stats.get("counts", {})

    notes = []  # type: List[str]
    entity_count = counts.get("entities", 0)
    claim_count = counts.get("claims", 0)
    evidence_count = counts.get("evidence", 0)
    relationship_count = counts.get("relationships", 0)

    if entity_count == 0:
        notes.append("No entities found -- source data may have been empty or missing")
    if claim_count == 0:
        notes.append("No claims found -- source data may have been empty or missing")
    if evidence_count == 0:
        notes.append("No evidence records -- insights/learnings sources may be missing")

    return {
        "entities": entity_count,
        "claims": claim_count,
        "evidence": evidence_count,
        "relationships": relationship_count,
        "status": "verified",
        "notes": notes,
    }


def run_full_migration(db: Database) -> Dict[str, Any]:
    """Run all three migrations in sequence, then verify.

    Uses default source paths. Each migration handles missing files
    gracefully, so this always completes even if sources are absent.
    """
    promptspeak_result = migrate_promptspeak_symbols(db)
    insights_result = migrate_knowledge_insights(db)
    learnings_result = migrate_grounded_learnings(db)
    verification = verify_migration(db)

    return {
        "promptspeak": promptspeak_result,
        "insights": insights_result,
        "learnings": learnings_result,
        "verification": verification,
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# Dry-run helper
# ---------------------------------------------------------------------------


def _dry_run_report() -> Dict[str, Any]:
    """Check source files and report what would be migrated, without writing."""
    report = {}  # type: Dict[str, Any]

    # PromptSpeak symbols
    if os.path.exists(PROMPTSPEAK_DB):
        try:
            src_uri = "file:{}?mode=ro".format(PROMPTSPEAK_DB)
            conn = sqlite3.connect(src_uri, uri=True)
            count = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
            conn.close()
            report["promptspeak"] = {"found": True, "symbols": count}
        except sqlite3.Error as e:
            report["promptspeak"] = {"found": True, "error": str(e)}
    else:
        report["promptspeak"] = {"found": False, "path": PROMPTSPEAK_DB}

    # Insights JSONL
    if os.path.exists(INSIGHTS_JSONL):
        line_count = 0
        with open(INSIGHTS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    line_count += 1
        report["insights"] = {"found": True, "lines": line_count}
    else:
        report["insights"] = {"found": False, "path": INSIGHTS_JSONL}

    # Learnings JSONL
    if os.path.exists(LEARNINGS_JSONL):
        line_count = 0
        with open(LEARNINGS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    line_count += 1
        report["learnings"] = {"found": True, "lines": line_count}
    else:
        report["learnings"] = {"found": False, "path": LEARNINGS_JSONL}

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for running the migration as a script."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy data into the Epistemic World Model database."
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Path to the EWM SQLite database (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check sources and report what would be migrated without writing",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN: Checking migration sources ===\n")
        report = _dry_run_report()
        print(json.dumps(report, indent=2))
        return

    # Ensure the database directory exists
    db_dir = os.path.dirname(args.db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    print("=== Epistemic World Model Migration ===\n")
    print("Database: {}\n".format(args.db_path))

    db = Database(args.db_path)
    try:
        results = run_full_migration(db)
        print(json.dumps(results, indent=2))

        overall_status = results.get("status", "unknown")
        print("\nMigration {}.".format(overall_status))
    finally:
        db.close()


if __name__ == "__main__":
    main()
