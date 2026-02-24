"""SQLite persistence layer for the Epistemic World Model.

Uses WAL mode, foreign keys, and JSON serialization for complex fields.
Stdlib only: sqlite3, json, pathlib, datetime.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ewm.types import (
    Action,
    ActionType,
    Claim,
    ClaimType,
    Entity,
    EntityCategory,
    Evidence,
    Relationship,
    RelationshipType,
    SourceType,
    Uncertainty,
    WorldState,
)


def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class Database:
    """SQLite-backed persistence for the Epistemic World Model.

    Manages entities, claims, evidence, relationships, audit logs, and
    red-line constraints. All complex fields (aliases, properties, tags,
    metadata, five_w1h) are stored as JSON text columns.
    """

    def __init__(self, path: str | Path = "ewm.db") -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create all tables if they do not already exist."""
        stmts = [
            """
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                aliases TEXT DEFAULT '[]',
                properties TEXT DEFAULT '{}',
                created TEXT NOT NULL,
                updated TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                content TEXT NOT NULL,
                source_id TEXT DEFAULT '',
                timestamp TEXT NOT NULL,
                reliability REAL DEFAULT 0.7,
                metadata TEXT DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                claim_type TEXT NOT NULL,
                belief REAL NOT NULL,
                disbelief REAL NOT NULL,
                uncertainty_val REAL NOT NULL,
                sample_size REAL DEFAULT 2.0,
                entity_ids TEXT DEFAULT '[]',
                created TEXT NOT NULL,
                updated TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                five_w1h TEXT DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS claim_evidence (
                claim_id TEXT NOT NULL REFERENCES claims(id),
                evidence_id TEXT NOT NULL REFERENCES evidence(id),
                PRIMARY KEY (claim_id, evidence_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES entities(id),
                target_id TEXT NOT NULL REFERENCES entities(id),
                rel_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                created TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS relationship_evidence (
                relationship_id TEXT NOT NULL REFERENCES relationships(id),
                evidence_id TEXT NOT NULL REFERENCES evidence(id),
                PRIMARY KEY (relationship_id, evidence_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action_type TEXT NOT NULL,
                target_id TEXT DEFAULT '',
                payload TEXT DEFAULT '{}',
                rationale TEXT DEFAULT '',
                status TEXT DEFAULT 'executed'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS red_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                description TEXT NOT NULL,
                created TEXT NOT NULL
            )
            """,
        ]
        cur = self.conn.cursor()
        for stmt in stmts:
            cur.execute(stmt)
        self.conn.commit()

    # ==================================================================
    # Entity CRUD
    # ==================================================================

    def save_entity(self, entity: Entity) -> Entity:
        """Insert or replace an entity. Returns the saved entity."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO entities
                (id, name, category, aliases, properties, created, updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity.id,
                entity.name,
                entity.category.value if isinstance(entity.category, EntityCategory) else str(entity.category),
                json.dumps(entity.aliases),
                json.dumps(entity.properties),
                entity.created,
                entity.updated,
            ),
        )
        self.conn.commit()
        return entity

    def get_entity(self, id: str) -> Entity | None:
        """Fetch a single entity by id, or None if not found."""
        row = self.conn.execute(
            "SELECT * FROM entities WHERE id = ?", (id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entity(row)

    def find_entities(
        self,
        name: str = "",
        category: EntityCategory | None = None,
        limit: int = 50,
    ) -> list[Entity]:
        """Search entities by name (LIKE) and/or category."""
        clauses: list[str] = []
        params: list[object] = []

        if name:
            clauses.append("name LIKE ?")
            params.append(f"%{name}%")
        if category is not None:
            cat_val = category.value if isinstance(category, EntityCategory) else str(category)
            clauses.append("category = ?")
            params.append(cat_val)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM entities {where} ORDER BY updated DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def delete_entity(self, id: str) -> bool:
        """Delete an entity. Returns True if a row was deleted."""
        cur = self.conn.execute("DELETE FROM entities WHERE id = ?", (id,))
        self.conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            name=row["name"],
            category=EntityCategory(row["category"]),
            aliases=json.loads(row["aliases"]),
            properties=json.loads(row["properties"]),
            created=row["created"],
            updated=row["updated"],
        )

    # ==================================================================
    # Evidence CRUD
    # ==================================================================

    def save_evidence(self, evidence: Evidence) -> Evidence:
        """Insert or replace an evidence record."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO evidence
                (id, source_type, content, source_id, timestamp, reliability, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence.id,
                evidence.source_type.value if isinstance(evidence.source_type, SourceType) else str(evidence.source_type),
                evidence.content,
                evidence.source_id,
                evidence.timestamp,
                evidence.reliability,
                json.dumps(evidence.metadata),
            ),
        )
        self.conn.commit()
        return evidence

    def get_evidence(self, id: str) -> Evidence | None:
        """Fetch a single evidence record by id."""
        row = self.conn.execute(
            "SELECT * FROM evidence WHERE id = ?", (id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_evidence(row)

    def find_evidence(
        self,
        source_type: SourceType | None = None,
        limit: int = 50,
    ) -> list[Evidence]:
        """List evidence, optionally filtered by source_type."""
        if source_type is not None:
            st_val = source_type.value if isinstance(source_type, SourceType) else str(source_type)
            rows = self.conn.execute(
                "SELECT * FROM evidence WHERE source_type = ? ORDER BY timestamp DESC LIMIT ?",
                (st_val, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM evidence ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_evidence(r) for r in rows]

    @staticmethod
    def _row_to_evidence(row: sqlite3.Row) -> Evidence:
        return Evidence(
            id=row["id"],
            source_type=SourceType(row["source_type"]),
            content=row["content"],
            source_id=row["source_id"],
            timestamp=row["timestamp"],
            reliability=row["reliability"],
            metadata=json.loads(row["metadata"]),
        )

    # ==================================================================
    # Claim CRUD
    # ==================================================================

    def save_claim(self, claim: Claim) -> Claim:
        """Insert or replace a claim and its evidence links."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO claims
                (id, text, claim_type, belief, disbelief, uncertainty_val,
                 sample_size, entity_ids, created, updated, tags, five_w1h)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.id,
                claim.text,
                claim.claim_type.value if isinstance(claim.claim_type, ClaimType) else str(claim.claim_type),
                claim.uncertainty.belief,
                claim.uncertainty.disbelief,
                claim.uncertainty.uncertainty,
                claim.uncertainty.sample_size,
                json.dumps(claim.entity_ids),
                claim.created,
                claim.updated,
                json.dumps(claim.tags),
                json.dumps(claim.five_w1h),
            ),
        )

        # Rebuild claim_evidence links
        self.conn.execute(
            "DELETE FROM claim_evidence WHERE claim_id = ?", (claim.id,)
        )
        if claim.evidence_ids:
            self.conn.executemany(
                "INSERT OR IGNORE INTO claim_evidence (claim_id, evidence_id) VALUES (?, ?)",
                [(claim.id, eid) for eid in claim.evidence_ids],
            )

        self.conn.commit()
        return claim

    def get_claim(self, id: str) -> Claim | None:
        """Fetch a claim by id, including its evidence links."""
        row = self.conn.execute(
            "SELECT * FROM claims WHERE id = ?", (id,)
        ).fetchone()
        if row is None:
            return None

        evidence_ids = [
            r["evidence_id"]
            for r in self.conn.execute(
                "SELECT evidence_id FROM claim_evidence WHERE claim_id = ?",
                (id,),
            ).fetchall()
        ]
        return self._row_to_claim(row, evidence_ids)

    def find_claims(
        self,
        text: str = "",
        claim_type: ClaimType | None = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[Claim]:
        """Search claims by text (LIKE), type, and minimum confidence (belief)."""
        clauses: list[str] = []
        params: list[object] = []

        if text:
            clauses.append("text LIKE ?")
            params.append(f"%{text}%")
        if claim_type is not None:
            ct_val = claim_type.value if isinstance(claim_type, ClaimType) else str(claim_type)
            clauses.append("claim_type = ?")
            params.append(ct_val)
        if min_confidence > 0.0:
            clauses.append("belief >= ?")
            params.append(min_confidence)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM claims {where} ORDER BY updated DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        claims: list[Claim] = []
        for row in rows:
            evidence_ids = [
                r["evidence_id"]
                for r in self.conn.execute(
                    "SELECT evidence_id FROM claim_evidence WHERE claim_id = ?",
                    (row["id"],),
                ).fetchall()
            ]
            claims.append(self._row_to_claim(row, evidence_ids))
        return claims

    def update_claim_uncertainty(self, claim_id: str, uncertainty: Uncertainty) -> bool:
        """Update just the uncertainty columns on an existing claim."""
        cur = self.conn.execute(
            """
            UPDATE claims
            SET belief = ?, disbelief = ?, uncertainty_val = ?,
                sample_size = ?, updated = ?
            WHERE id = ?
            """,
            (
                uncertainty.belief,
                uncertainty.disbelief,
                uncertainty.uncertainty,
                uncertainty.sample_size,
                _now_iso(),
                claim_id,
            ),
        )
        self.conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row_to_claim(row: sqlite3.Row, evidence_ids: list[str]) -> Claim:
        return Claim(
            id=row["id"],
            text=row["text"],
            claim_type=ClaimType(row["claim_type"]),
            uncertainty=Uncertainty(
                belief=row["belief"],
                disbelief=row["disbelief"],
                uncertainty=row["uncertainty_val"],
                sample_size=row["sample_size"],
            ),
            entity_ids=json.loads(row["entity_ids"]),
            evidence_ids=evidence_ids,
            created=row["created"],
            updated=row["updated"],
            tags=json.loads(row["tags"]),
            five_w1h=json.loads(row["five_w1h"]),
        )

    # ==================================================================
    # Relationship CRUD
    # ==================================================================

    def save_relationship(self, rel: Relationship) -> Relationship:
        """Insert or replace a relationship and its evidence links."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO relationships
                (id, source_id, target_id, rel_type, confidence, created)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                rel.id,
                rel.source_id,
                rel.target_id,
                rel.rel_type.value if isinstance(rel.rel_type, RelationshipType) else str(rel.rel_type),
                rel.confidence,
                rel.created,
            ),
        )

        # Rebuild relationship_evidence links if evidence_ids present
        if hasattr(rel, "evidence_ids") and rel.evidence_ids:
            self.conn.execute(
                "DELETE FROM relationship_evidence WHERE relationship_id = ?",
                (rel.id,),
            )
            self.conn.executemany(
                "INSERT OR IGNORE INTO relationship_evidence (relationship_id, evidence_id) VALUES (?, ?)",
                [(rel.id, eid) for eid in rel.evidence_ids],
            )

        self.conn.commit()
        return rel

    def get_relationships(self, entity_id: str) -> list[Relationship]:
        """Get all relationships where the entity is source or target."""
        rows = self.conn.execute(
            """
            SELECT * FROM relationships
            WHERE source_id = ? OR target_id = ?
            ORDER BY created DESC
            """,
            (entity_id, entity_id),
        ).fetchall()
        return [self._row_to_relationship(r) for r in rows]

    def find_relationships(
        self,
        rel_type: RelationshipType | None = None,
        limit: int = 50,
    ) -> list[Relationship]:
        """List relationships, optionally filtered by type."""
        if rel_type is not None:
            rt_val = rel_type.value if isinstance(rel_type, RelationshipType) else str(rel_type)
            rows = self.conn.execute(
                "SELECT * FROM relationships WHERE rel_type = ? ORDER BY created DESC LIMIT ?",
                (rt_val, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM relationships ORDER BY created DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_relationship(r) for r in rows]

    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        evidence_ids = [
            r["evidence_id"]
            for r in self.conn.execute(
                "SELECT evidence_id FROM relationship_evidence WHERE relationship_id = ?",
                (row["id"],),
            ).fetchall()
        ]
        return Relationship(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            rel_type=RelationshipType(row["rel_type"]),
            confidence=row["confidence"],
            created=row["created"],
            evidence_ids=evidence_ids,
        )

    # ==================================================================
    # Red Lines
    # ==================================================================

    def save_red_line(self, pattern: str, description: str) -> int:
        """Insert a new red-line constraint. Returns the row id.

        Returns:
            The integer row ID of the newly inserted red-line.
            Falls back to -1 if the database driver returns None
            (should not happen with SQLite in practice).
        """
        cur = self.conn.execute(
            "INSERT INTO red_lines (pattern, description, created) VALUES (?, ?, ?)",
            (pattern, description, _now_iso()),
        )
        self.conn.commit()
        return cur.lastrowid if cur.lastrowid is not None else -1

    def get_red_lines(self) -> list[dict]:
        """Return all red-line constraints as plain dicts."""
        rows = self.conn.execute(
            "SELECT id, pattern, description, created FROM red_lines ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    # ==================================================================
    # Audit Log
    # ==================================================================

    def log_action(self, action: Action) -> None:
        """Write an action to the audit log."""
        self.conn.execute(
            """
            INSERT INTO audit_log
                (timestamp, action_type, target_id, payload, rationale, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                _now_iso(),
                action.action_type.value if isinstance(action.action_type, ActionType) else str(action.action_type),
                action.target_id,
                json.dumps(action.payload) if isinstance(action.payload, dict) else str(action.payload),
                action.rationale,
                action.status,
            ),
        )
        self.conn.commit()

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        """Return recent audit log entries as plain dicts."""
        rows = self.conn.execute(
            """
            SELECT id, timestamp, action_type, target_id, payload,
                   rationale, status
            FROM audit_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        results: list[dict] = []
        for r in rows:
            d = dict(r)
            # Parse payload back to dict if valid JSON
            try:
                d["payload"] = json.loads(d["payload"])
            except (json.JSONDecodeError, TypeError):
                pass
            results.append(d)
        return results

    # ==================================================================
    # World State
    # ==================================================================

    def load_world_state(
        self,
        entity_limit: int = 1000,
        claim_limit: int = 1000,
    ) -> WorldState:
        """Build a WorldState snapshot from the current database contents."""
        entities = self.find_entities(limit=entity_limit)
        claims = self.find_claims(limit=claim_limit)
        relationships = self.find_relationships(limit=5000)
        red_lines = self.get_red_lines()

        return WorldState(
            entities={e.id: e for e in entities},
            claims={c.id: c for c in claims},
            relationships=relationships,
            red_lines=red_lines,
        )

    # ==================================================================
    # Stats
    # ==================================================================

    # Tables known to this schema — used by stats() for safe iteration.
    _KNOWN_TABLES: tuple[str, ...] = (
        "entities",
        "evidence",
        "claims",
        "claim_evidence",
        "relationships",
        "relationship_evidence",
        "audit_log",
        "red_lines",
    )

    def stats(self) -> dict:
        """Return table counts, DB file size, and last activity timestamp."""
        counts: dict[str, int] = {}
        for table in self._KNOWN_TABLES:
            # Use parameterized-safe query: table names come from the frozen
            # _KNOWN_TABLES tuple above (never from user input).  SQLite does
            # not support parameterized table names, so we validate membership.
            assert table in self._KNOWN_TABLES  # belt-and-suspenders guard
            row = self.conn.execute(
                "SELECT COUNT(*) AS cnt FROM " + table  # noqa: S608
            ).fetchone()
            counts[table] = row["cnt"]

        # Database file size
        try:
            file_size = os.path.getsize(self.path)
        except OSError:
            file_size = 0

        # Last activity: most recent audit_log entry or most recent entity/claim update
        last_activity: str | None = None
        row = self.conn.execute(
            "SELECT timestamp FROM audit_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            last_activity = row["timestamp"]
        else:
            # Fall back to most recent entity or claim update
            row = self.conn.execute(
                "SELECT updated FROM entities ORDER BY updated DESC LIMIT 1"
            ).fetchone()
            if row:
                last_activity = row["updated"]
            else:
                row = self.conn.execute(
                    "SELECT updated FROM claims ORDER BY updated DESC LIMIT 1"
                ).fetchone()
                if row:
                    last_activity = row["updated"]

        return {
            "counts": counts,
            "file_size_bytes": file_size,
            "last_activity": last_activity,
        }
