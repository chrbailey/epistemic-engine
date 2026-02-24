"""
Epistemic World Model -- Memory Module
=======================================

Implements LeCun's Memory module with three tiers:

    1. Working memory  -- in-process dict, ephemeral within a session
    2. Episodic memory  -- audit_log table in SQLite (read via db.get_audit_log)
    3. Semantic memory  -- entities, claims, relationships in SQLite

Optional Pinecone integration provides semantic vector search via the
Pinecone REST API (stdlib only -- uses urllib.request, no SDK).

Public API:
    store()               -- persist a PerceptionResult, deduplicating entities
    recall()              -- search across entities, claims, relationships
    context_for_session() -- generate an LLM-injectable context block
    forget_stale()        -- prune uncertain claims that never resolved
    sync_to_pinecone()    -- bulk-sync all records to Pinecone
    configure_pinecone()  -- set up Pinecone connectivity
    working_get/set/clear/keys -- working memory accessors

All database interaction goes through the Database class (ewm.db).
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ewm.db import Database
from ewm.types import (
    Claim,
    Entity,
    Evidence,
    MemoryResult,
    PerceptionResult,
    Relationship,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pinecone Configuration
# ---------------------------------------------------------------------------

PINECONE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "api_key": "",
    "index_name": "claude-knowledge-base",
    "namespace": "ewm",
    "host": "",
    "embed_model": "llama-text-embed-v2",
    "content_field": "content",  # fieldMap text field name in the index
}


# ---------------------------------------------------------------------------
# Working Memory (Tier 1) -- module-level ephemeral cache
# ---------------------------------------------------------------------------

_working_memory: Dict[str, Any] = {}


def working_get(key: str) -> Any:
    """Retrieve a value from working memory, or None if absent."""
    return _working_memory.get(key)


def working_set(key: str, value: Any) -> None:
    """Store a value in working memory."""
    _working_memory[key] = value


def working_clear() -> None:
    """Clear all working memory."""
    _working_memory.clear()


def working_keys() -> List[str]:
    """Return all keys currently in working memory."""
    return list(_working_memory.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_overlap_score(query: str, text: str) -> float:
    """Compute a simple relevance score based on overlapping words.

    Normalises the count of shared words by the total number of unique
    words across both strings, producing a value in [0, 1].

    Args:
        query: The search query.
        text: The text to compare against.

    Returns:
        A float in [0, 1] where 1.0 means identical word sets.
    """
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    if not query_words or not text_words:
        return 0.0
    intersection = query_words & text_words
    union = query_words | text_words
    return len(intersection) / len(union)


def _entity_text(entity: Entity) -> str:
    """Build a searchable text representation of an entity."""
    parts = [entity.name, entity.category.value]
    parts.extend(entity.aliases)
    return " ".join(parts)


def _claim_text(claim: Claim) -> str:
    """Build a searchable text representation of a claim."""
    return claim.text


# ---------------------------------------------------------------------------
# Pinecone REST helpers (stdlib only -- urllib.request)
# ---------------------------------------------------------------------------

def _pinecone_request(
    url: str,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    api_key: str = "",
    raw_body: Optional[bytes] = None,
    content_type: str = "application/json",
) -> Dict[str, Any]:
    """Make an HTTP request to the Pinecone REST API.

    Args:
        url: Full URL to request.
        method: HTTP method (GET, POST).
        body: JSON body for POST requests (encoded as application/json).
        api_key: Pinecone API key for authentication.
        raw_body: Pre-encoded body bytes (takes precedence over body).
        content_type: Content-Type header value.

    Returns:
        Parsed JSON response as a dict, or an error dict on failure.
    """
    headers = {
        "Api-Key": api_key or PINECONE_CONFIG["api_key"],
        "Content-Type": content_type,
        "X-Pinecone-API-Version": "2025-04",
    }
    if raw_body is not None:
        data = raw_body
    elif body is not None:
        data = json.dumps(body).encode("utf-8")
    else:
        data = None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            resp_body = resp.read().decode("utf-8")
            if resp_body:
                return json.loads(resp_body)
            return {"status": "ok"}
    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        _log.warning("Pinecone HTTP %s: %s %s", exc.code, exc.reason, error_body[:200])
        return {"status": "error", "code": exc.code, "message": error_body[:500]}
    except urllib.error.URLError as exc:
        _log.warning("Pinecone URL error: %s", exc.reason)
        return {"status": "error", "message": str(exc.reason)}
    except Exception as exc:
        _log.warning("Pinecone request failed: %s", exc)
        return {"status": "error", "message": str(exc)}


def configure_pinecone(
    api_key: Optional[str] = None,
    index_name: str = "claude-knowledge-base",
    namespace: str = "ewm",
) -> Dict[str, Any]:
    """Configure Pinecone connectivity by testing the API key and locating the index host.

    If api_key is not provided, falls back to the PINECONE_API_KEY environment
    variable.  Tests connectivity by listing indexes and finding the host URL
    for the requested index.

    Args:
        api_key: Pinecone API key. Falls back to env var if not provided.
        index_name: Name of the Pinecone index to use.
        namespace: Namespace within the index.

    Returns:
        Status dict with connection result details.
    """
    key = api_key or os.environ.get("PINECONE_API_KEY", "")
    if not key:
        return {"status": "no_api_key"}

    resp = _pinecone_request(
        "https://api.pinecone.io/indexes",
        method="GET",
        api_key=key,
    )

    if resp.get("status") == "error":
        return {"status": "connection_failed", "detail": resp.get("message", "")}

    # Find the requested index in the response
    indexes = resp.get("indexes", [])
    host = ""
    embed_info: Dict[str, Any] = {}
    for idx in indexes:
        if idx.get("name") == index_name:
            host = idx.get("host", "")
            embed_info = idx.get("embed", {})
            break

    if not host:
        available = [idx.get("name", "?") for idx in indexes]
        return {
            "status": "index_not_found",
            "index_name": index_name,
            "available_indexes": available,
        }

    # Ensure host has https:// prefix
    if not host.startswith("https://"):
        host = "https://" + host

    # Detect the content field name from the index's fieldMap
    field_map = embed_info.get("fieldMap", {})
    content_field = field_map.get("text", "content")
    embed_model = embed_info.get("model", "llama-text-embed-v2")

    PINECONE_CONFIG["enabled"] = True
    PINECONE_CONFIG["api_key"] = key
    PINECONE_CONFIG["index_name"] = index_name
    PINECONE_CONFIG["namespace"] = namespace
    PINECONE_CONFIG["host"] = host
    PINECONE_CONFIG["embed_model"] = embed_model
    PINECONE_CONFIG["content_field"] = content_field

    return {
        "status": "configured",
        "index_name": index_name,
        "namespace": namespace,
        "host": host,
        "embed_model": embed_model,
        "content_field": content_field,
    }


def _pinecone_upsert(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Upsert records into Pinecone via the integrated inference API.

    Uses NDJSON format (one JSON object per line) as required by the
    Pinecone records upsert endpoint.  Each record must include the
    content field matching the index's fieldMap (default: ``content``).

    Args:
        records: List of record dicts, each with ``_id``, content field, and metadata.

    Returns:
        Status dict with upsert result.
    """
    if not PINECONE_CONFIG["enabled"]:
        return {"status": "pinecone_not_configured"}

    host = PINECONE_CONFIG["host"]
    namespace = PINECONE_CONFIG["namespace"]
    url = "{host}/records/namespaces/{ns}/upsert".format(host=host, ns=namespace)

    # Build NDJSON body (one JSON object per line)
    ndjson_lines = [json.dumps(record) for record in records]
    ndjson_body = "\n".join(ndjson_lines)

    resp = _pinecone_request(
        url, method="POST", raw_body=ndjson_body.encode("utf-8"),
        content_type="application/x-ndjson",
    )

    if resp.get("status") == "error":
        return resp

    return {"status": "ok", "upserted": len(records)}


def _pinecone_search(
    query: str,
    top_k: int = 20,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Search Pinecone for records similar to the query text.

    Uses the index's integrated inference model for query embedding.

    Args:
        query: Free-text search query.
        top_k: Maximum number of results to return.
        filter_dict: Optional metadata filter (MongoDB-style operators).

    Returns:
        List of hit dicts with ``id``, ``score``, and ``fields`` keys.
    """
    if not PINECONE_CONFIG["enabled"]:
        return []

    host = PINECONE_CONFIG["host"]
    namespace = PINECONE_CONFIG["namespace"]
    url = "{host}/records/namespaces/{ns}/search".format(host=host, ns=namespace)

    query_body: Dict[str, Any] = {
        "inputs": {"text": query},
        "top_k": top_k,
    }
    if filter_dict:
        query_body["filter"] = filter_dict

    body: Dict[str, Any] = {"query": query_body}

    resp = _pinecone_request(url, method="POST", body=body)

    if resp.get("status") == "error":
        _log.warning("Pinecone search failed: %s", resp.get("message", ""))
        return []

    # Parse hits from the response
    hits: List[Dict[str, Any]] = []
    result_list = resp.get("result", resp.get("matches", []))
    if isinstance(result_list, dict):
        result_list = result_list.get("hits", result_list.get("matches", []))

    for item in result_list:
        hit_id = item.get("_id", item.get("id", ""))
        score = item.get("_score", item.get("score", 0.0))
        fields = item.get("fields", {})
        hits.append({"id": hit_id, "score": score, "fields": fields})

    return hits


# ===========================================================================
# store -- persist a PerceptionResult
# ===========================================================================

def store(db: Database, result: PerceptionResult) -> Dict[str, int]:
    """Persist a PerceptionResult to the database.

    Handles deduplication for entities: if an entity with the same name
    already exists, the existing record is updated rather than creating a
    duplicate.  Evidence, claims, and relationships are saved directly.

    When Pinecone is configured, also upserts entity and claim records
    for semantic vector search.

    Args:
        db: Database instance.
        result: Output from the Perception module.

    Returns:
        Summary dict with counts of stored items and merged duplicates.
    """
    entities_stored = 0
    duplicates_merged = 0
    evidence_stored = 0
    claims_stored = 0
    relationships_stored = 0

    # Map from original entity id -> persisted entity id (for remapping)
    id_map: Dict[str, str] = {}

    # Track persisted entities/claims for Pinecone upsert
    persisted_entities: List[Entity] = []
    persisted_claims: List[Claim] = []

    # --- Entities (with deduplication) ---
    for entity in result.entities:
        existing = db.find_entities(name=entity.name)
        # Look for an exact name match (case-insensitive)
        matched: Optional[Entity] = None
        for candidate in existing:
            if candidate.name.lower() == entity.name.lower():
                matched = candidate
                break

        if matched is not None:
            # Update the existing entity's timestamp and merge aliases
            new_aliases = list(set(matched.aliases + entity.aliases))
            merged_props = dict(matched.properties)
            merged_props.update(entity.properties)
            merged = Entity(
                id=matched.id,
                name=matched.name,
                category=matched.category,
                aliases=new_aliases,
                properties=merged_props,
                created=matched.created,
                updated=entity.updated,
            )
            db.save_entity(merged)
            id_map[entity.id] = matched.id
            persisted_entities.append(merged)
            duplicates_merged += 1
        else:
            db.save_entity(entity)
            id_map[entity.id] = entity.id
            persisted_entities.append(entity)
            entities_stored += 1

    # --- Evidence ---
    for evidence in result.evidence:
        db.save_evidence(evidence)
        evidence_stored += 1

    # --- Claims (remap entity_ids through id_map) ---
    for claim in result.claims:
        remapped_entity_ids = [id_map.get(eid, eid) for eid in claim.entity_ids]
        claim.entity_ids = remapped_entity_ids
        db.save_claim(claim)
        persisted_claims.append(claim)
        claims_stored += 1

    # --- Relationships (remap source_id and target_id) ---
    for rel in result.relationships:
        remapped_rel = Relationship(
            id=rel.id,
            source_id=id_map.get(rel.source_id, rel.source_id),
            target_id=id_map.get(rel.target_id, rel.target_id),
            rel_type=rel.rel_type,
            evidence_ids=rel.evidence_ids,
            confidence=rel.confidence,
            created=rel.created,
        )
        db.save_relationship(remapped_rel)
        relationships_stored += 1

    # --- Pinecone upsert (fire-and-forget, errors logged but not raised) ---
    if PINECONE_CONFIG["enabled"]:
        _store_to_pinecone(persisted_entities, persisted_claims)

    return {
        "entities_stored": entities_stored,
        "claims_stored": claims_stored,
        "evidence_stored": evidence_stored,
        "relationships_stored": relationships_stored,
        "duplicates_merged": duplicates_merged,
    }


def _store_to_pinecone(
    entities: List[Entity], claims: List[Claim]
) -> None:
    """Upsert entities and claims to Pinecone after a store() call.

    Builds Pinecone-compatible records and upserts them in batches of 96
    (Pinecone's per-request limit). Errors are logged but never raised --
    Pinecone is optional infrastructure.

    Args:
        entities: Persisted entities to upsert.
        claims: Persisted claims to upsert.
    """
    content_field = PINECONE_CONFIG.get("content_field", "content")
    records: List[Dict[str, Any]] = []

    for entity in entities:
        props_text = " ".join(
            "{k}: {v}".format(k=k, v=v) for k, v in entity.properties.items()
        )
        content = "{name} ({cat}). {props}".format(
            name=entity.name,
            cat=entity.category.value,
            props=props_text,
        ).strip()
        records.append({
            "_id": "entity:{eid}".format(eid=entity.id),
            content_field: content,
            "category": entity.category.value,
            "type": "entity",
            "confidence": "1.0",
            "source": "ewm",
        })

    for claim in claims:
        records.append({
            "_id": "claim:{cid}".format(cid=claim.id),
            content_field: claim.text,
            "category": claim.claim_type.value,
            "type": "claim",
            "confidence": str(claim.uncertainty.expected_value),
            "source": "ewm",
        })

    # Batch upsert in groups of 96 (Pinecone's per-request limit)
    batch_size = 96
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        _pinecone_upsert(batch)


# ===========================================================================
# recall -- search across the knowledge base
# ===========================================================================

def recall(db: Database, query: str, limit: int = 20) -> MemoryResult:
    """Search across entities, claims, and relationships.

    When Pinecone is configured, uses vector similarity search for
    semantically relevant results.  Falls back to the database's
    LIKE-based search with word-overlap scoring when Pinecone is
    unavailable.

    Args:
        db: Database instance.
        query: Free-text search query.
        limit: Maximum number of items to return per category.

    Returns:
        A MemoryResult with relevance_scores populated.
    """
    if PINECONE_CONFIG["enabled"]:
        try:
            result = _recall_from_pinecone(db, query, limit)
            if result is not None:
                return result
        except Exception as exc:
            _log.warning("Pinecone recall failed, falling back to SQL: %s", exc)

    return _recall_from_sql(db, query, limit)


def _recall_from_pinecone(
    db: Database, query: str, limit: int
) -> Optional[MemoryResult]:
    """Attempt recall via Pinecone vector search.

    Args:
        db: Database instance for hydrating entities/claims by ID.
        query: Free-text search query.
        limit: Maximum results.

    Returns:
        A MemoryResult if hits were found, or None to signal fallback.
    """
    hits = _pinecone_search(query, top_k=limit)
    if not hits:
        return None

    entities: List[Entity] = []
    claims: List[Claim] = []
    relevance_scores: Dict[str, float] = {}

    for hit in hits:
        item_id = hit["id"]
        score = hit["score"]

        if item_id.startswith("entity:"):
            eid = item_id[7:]
            entity = db.get_entity(eid)
            if entity is not None:
                entities.append(entity)
                relevance_scores[eid] = score
        elif item_id.startswith("claim:"):
            cid = item_id[6:]
            claim = db.get_claim(cid)
            if claim is not None:
                claims.append(claim)
                relevance_scores[cid] = score

    # Gather relationships for matched entities
    relationships: List[Relationship] = []
    for entity in entities:
        for rel in db.get_relationships(entity.id):
            if rel.id not in relevance_scores:
                relationships.append(rel)
                relevance_scores[rel.id] = max(
                    relevance_scores.get(rel.source_id, 0.0),
                    relevance_scores.get(rel.target_id, 0.0),
                )

    return MemoryResult(
        entities=entities,
        claims=claims,
        relationships=relationships,
        relevance_scores=relevance_scores,
    )


def _recall_from_sql(db: Database, query: str, limit: int) -> MemoryResult:
    """Recall using SQL LIKE search with word-overlap scoring.

    This is the original recall implementation, preserved as the fallback
    when Pinecone is not available.

    Args:
        db: Database instance.
        query: Free-text search query.
        limit: Maximum results per category.

    Returns:
        A MemoryResult with relevance_scores populated.
    """
    relevance_scores: Dict[str, float] = {}

    # Search entities
    entities = db.find_entities(name=query, limit=limit)
    for entity in entities:
        score = _word_overlap_score(query, _entity_text(entity))
        relevance_scores[entity.id] = score

    # Search claims
    claims = db.find_claims(text=query, limit=limit)
    for claim in claims:
        score = _word_overlap_score(query, _claim_text(claim))
        relevance_scores[claim.id] = score

    # Find relationships connected to any matched entity
    entity_ids = {e.id for e in entities}
    all_relationships = db.find_relationships(limit=limit * 5)
    relationships: List[Relationship] = []
    for rel in all_relationships:
        if rel.source_id in entity_ids or rel.target_id in entity_ids:
            relationships.append(rel)
            relevance_scores[rel.id] = max(
                relevance_scores.get(rel.source_id, 0.0),
                relevance_scores.get(rel.target_id, 0.0),
            )
        if len(relationships) >= limit:
            break

    return MemoryResult(
        entities=entities,
        claims=claims,
        relationships=relationships,
        relevance_scores=relevance_scores,
    )


# ===========================================================================
# context_for_session -- generate injectable context
# ===========================================================================

def context_for_session(db: Database, session_topic: str = "") -> str:
    """Generate a text block suitable for injecting into an LLM session.

    Includes top entities (sorted by relationship count), recent claims
    with high confidence, and system statistics.  When a session_topic is
    provided, items matching that topic are prioritised.

    Args:
        db: Database instance.
        session_topic: Optional topic to prioritise matching items.

    Returns:
        A structured text block with sections.
    """
    stats = db.stats()
    counts = stats.get("counts", {})

    # Load entities and relationships
    all_entities = db.find_entities(limit=200)
    all_relationships = db.find_relationships(limit=500)

    # Count relationships per entity
    rel_counts: Dict[str, int] = {}
    for rel in all_relationships:
        rel_counts[rel.source_id] = rel_counts.get(rel.source_id, 0) + 1
        rel_counts[rel.target_id] = rel_counts.get(rel.target_id, 0) + 1

    # Score entities: relationship count + topic bonus
    entity_scores: List[tuple] = []
    for entity in all_entities:
        score = rel_counts.get(entity.id, 0)
        if session_topic:
            overlap = _word_overlap_score(session_topic, _entity_text(entity))
            score += overlap * 10  # boost topic-relevant entities
        entity_scores.append((score, entity))
    entity_scores.sort(key=lambda x: x[0], reverse=True)
    top_entities = entity_scores[:10]

    # Load high-confidence claims
    all_claims = db.find_claims(limit=100)
    claim_scored: List[tuple] = []
    for claim in all_claims:
        confidence = claim.uncertainty.confidence
        score = confidence
        if session_topic:
            overlap = _word_overlap_score(session_topic, claim.text)
            score += overlap * 5
        claim_scored.append((score, claim))
    claim_scored.sort(key=lambda x: x[0], reverse=True)
    top_claims = claim_scored[:10]

    # Recent audit log entries
    recent_actions = db.get_audit_log(limit=5)

    # Build output
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("EPISTEMIC WORLD MODEL -- SESSION CONTEXT")
    lines.append("=" * 60)
    lines.append("")

    # Stats section
    lines.append("## System Stats")
    lines.append("  Entities: {n}".format(n=counts.get("entities", 0)))
    lines.append("  Claims:   {n}".format(n=counts.get("claims", 0)))
    lines.append("  Evidence: {n}".format(n=counts.get("evidence", 0)))
    lines.append("  Relationships: {n}".format(n=counts.get("relationships", 0)))
    if stats.get("last_activity"):
        lines.append("  Last activity: {ts}".format(ts=stats["last_activity"]))
    lines.append("")

    # Top entities
    if top_entities:
        lines.append("## Key Entities")
        for score, entity in top_entities:
            rel_count = rel_counts.get(entity.id, 0)
            lines.append(
                "  - {name} [{cat}] ({n} relationships)".format(
                    name=entity.name,
                    cat=entity.category.value,
                    n=rel_count,
                )
            )
        lines.append("")

    # Top claims
    if top_claims:
        lines.append("## Key Claims")
        for score, claim in top_claims:
            conf = claim.uncertainty.confidence
            lines.append(
                "  - [{ct}] {text} (confidence: {conf:.0%})".format(
                    ct=claim.claim_type.value,
                    text=claim.text,
                    conf=conf,
                )
            )
        lines.append("")

    # Recent activity
    if recent_actions:
        lines.append("## Recent Activity")
        for action in recent_actions:
            lines.append(
                "  - {ts} | {at} | {rat}".format(
                    ts=action.get("timestamp", "?"),
                    at=action.get("action_type", "?"),
                    rat=action.get("rationale", ""),
                )
            )
        lines.append("")

    if session_topic:
        lines.append("## Session Topic: {topic}".format(topic=session_topic))
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ===========================================================================
# forget_stale -- prune claims that stayed uncertain too long
# ===========================================================================

def forget_stale(db: Database, max_age_days: int = 90) -> Dict[str, int]:
    """Delete claims whose uncertainty remained high past a deadline.

    Finds claims where uncertainty > 0.6 AND last updated more than
    max_age_days ago, then deletes them from the database.

    Args:
        db: Database instance.
        max_age_days: Maximum age in days before uncertain claims are pruned.

    Returns:
        Summary dict with counts of reviewed and forgotten claims.
    """
    all_claims = db.find_claims(limit=10000)
    now = datetime.now(timezone.utc)

    claims_reviewed = 0
    stale_ids: list[str] = []

    for claim in all_claims:
        claims_reviewed += 1

        # Check uncertainty threshold
        if claim.uncertainty.uncertainty <= 0.6:
            continue

        # Parse the claim's updated timestamp and compute age
        updated_dt = _parse_iso_timestamp(claim.updated)
        if updated_dt is None:
            continue

        age_days = (now - updated_dt).total_seconds() / 86400.0
        if age_days < max_age_days:
            continue

        stale_ids.append(claim.id)

    # Delete all stale claims in a single transaction
    if stale_ids:
        db.conn.execute("BEGIN")
        try:
            for cid in stale_ids:
                db.conn.execute(
                    "DELETE FROM claim_evidence WHERE claim_id = ?", (cid,)
                )
                db.conn.execute("DELETE FROM claims WHERE id = ?", (cid,))
            db.conn.commit()
        except Exception:
            db.conn.rollback()
            raise
    claims_forgotten = len(stale_ids)

    return {
        "claims_reviewed": claims_reviewed,
        "claims_forgotten": claims_forgotten,
    }


def _parse_iso_timestamp(ts: str) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp string into a timezone-aware datetime.

    Handles both with and without timezone info.  Returns None on failure.

    Args:
        ts: ISO 8601 timestamp string.

    Returns:
        A datetime object, or None if parsing fails.
    """
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ===========================================================================
# sync_to_pinecone -- bulk-sync all records
# ===========================================================================

def sync_to_pinecone(
    db: Database,
    index_name: str = "claude-knowledge-base",
    namespace: str = "ewm",
) -> Dict[str, Any]:
    """Bulk-sync all database entities and claims to Pinecone.

    Requires Pinecone to be configured via configure_pinecone() first.
    Records are batched in groups of 100 (Pinecone's per-request limit).

    NEVER deletes from Pinecone -- append/upsert only per project policy.

    Args:
        db: Database instance.
        index_name: Target Pinecone index name (informational).
        namespace: Target namespace (informational; uses PINECONE_CONFIG).

    Returns:
        A dict with sync status and record counts.
    """
    if not PINECONE_CONFIG["enabled"]:
        return {
            "status": "pinecone_not_configured",
            "hint": "Call configure_pinecone() first",
        }

    entities = db.find_entities(limit=10000)
    claims = db.find_claims(limit=10000)
    content_field = PINECONE_CONFIG.get("content_field", "content")

    records: List[Dict[str, Any]] = []

    for entity in entities:
        props_text = " ".join(
            "{k}: {v}".format(k=k, v=v) for k, v in entity.properties.items()
        )
        content = "{name} ({cat}). {props}".format(
            name=entity.name,
            cat=entity.category.value,
            props=props_text,
        ).strip()
        records.append({
            "_id": "entity:{eid}".format(eid=entity.id),
            content_field: content,
            "category": entity.category.value,
            "type": "entity",
            "confidence": "1.0",
            "source": "ewm",
        })

    for claim in claims:
        records.append({
            "_id": "claim:{cid}".format(cid=claim.id),
            content_field: claim.text,
            "category": claim.claim_type.value,
            "type": "claim",
            "confidence": str(claim.uncertainty.expected_value),
            "source": "ewm",
        })

    # Batch upsert in groups of 96 (Pinecone's per-request limit)
    total_upserted = 0
    batch_size = 96
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        result = _pinecone_upsert(batch)
        if result.get("status") == "ok":
            total_upserted += result.get("upserted", 0)

    return {
        "status": "synced",
        "records": total_upserted,
        "entities": len(entities),
        "claims": len(claims),
    }
