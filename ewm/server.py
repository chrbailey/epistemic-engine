"""
Epistemic World Model -- MCP Server Interface
===============================================

Exposes the EWM system as 12 MCP (Model Context Protocol) tools callable
from Claude Code.  The ``mcp`` package is an optional dependency -- when
unavailable the module still defines TOOL_DEFINITIONS and helper functions
for testing, and ``main()`` prints an error message and exits.

Tool manifest (12 tools):
    ewm_ingest           -- Ingest new information into the world model
    ewm_query            -- Query entities, claims, and relationships
    ewm_claim_create     -- Create a claim with uncertainty tracking
    ewm_claim_update     -- Add evidence to an existing claim
    ewm_claim_investigate-- Generate investigation plan for uncertain claims
    ewm_entity_create    -- Create or update an entity
    ewm_entity_relate    -- Create a relationship between two entities
    ewm_audit            -- Run a full system audit
    ewm_session_sync     -- Extract knowledge from a session transcript
    ewm_stats            -- Get system statistics and health metrics
    ewm_self_inspect     -- Inspect a module's source code and API
    ewm_self_propose     -- Propose a modification to a module

Design principles:
    - Tool definitions are plain dicts in TOOL_DEFINITIONS (testable without mcp).
    - Each handler wraps its body in try/except to avoid crashing the server.
    - Results are JSON-serialized via _to_json() with a default=str fallback.
    - The mcp import is guarded: absent mcp gracefully degrades at main().
    - Python 3.9+ compatible (no match statements, no X | Y unions).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Database path configuration
# ---------------------------------------------------------------------------

DB_PATH: str = os.environ.get(
    "EWM_DB_PATH",
    os.path.expanduser("~/.ewm/world_model.db"),
)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _to_json(obj: Any) -> str:
    """Convert a result to a JSON string.

    Handles dataclasses, enums, and other non-serializable types via
    the default=str fallback.
    """
    return json.dumps(obj, indent=2, default=str)


def _safe_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances to plain dicts."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _safe_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_dict(item) for item in obj]
    return obj


# ===========================================================================
# Tool Definitions (plain dicts -- no mcp dependency)
# ===========================================================================

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "ewm_ingest",
        "description": "Ingest new information into the world model",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text content to ingest into the world model",
                },
                "source_type": {
                    "type": "string",
                    "description": (
                        "Source type of the text (e.g. 'document', "
                        "'expert_testimony', 'statistical'). Default: 'document'"
                    ),
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "ewm_query",
        "description": "Query the world model for entities, claims, and relationships",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results per category (default: 20)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ewm_claim_create",
        "description": "Create a new claim with uncertainty tracking",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The propositional claim text",
                },
                "claim_type": {
                    "type": "string",
                    "description": (
                        "Semantic type: factual, statistical, causal, "
                        "predictive, accusatory, diagnostic, prescriptive "
                        "(default: 'factual')"
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": "Initial confidence in [0, 1] (default: 0.5)",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "ewm_claim_update",
        "description": "Add evidence to an existing claim, updating its uncertainty",
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim_id": {
                    "type": "string",
                    "description": "ID of the claim to update",
                },
                "evidence_text": {
                    "type": "string",
                    "description": "Text of the new evidence",
                },
                "direction": {
                    "type": "string",
                    "description": "'support' or 'contradict' (default: 'support')",
                },
            },
            "required": ["claim_id", "evidence_text"],
        },
    },
    {
        "name": "ewm_claim_investigate",
        "description": "Generate an investigation plan for uncertain claims",
        "inputSchema": {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "description": "Number of most uncertain claims to investigate (default: 5)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "ewm_entity_create",
        "description": "Create or update an entity in the world model",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entity name",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Category: person, organization, technology, concept, "
                        "location, event, artifact, financial (default: 'concept')"
                    ),
                },
                "aliases": {
                    "type": "string",
                    "description": "Comma-separated list of aliases (optional)",
                },
                "properties": {
                    "type": "string",
                    "description": "JSON string of additional properties (optional)",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "ewm_entity_relate",
        "description": "Create a relationship between two entities",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_name": {
                    "type": "string",
                    "description": "Name of the source entity",
                },
                "target_name": {
                    "type": "string",
                    "description": "Name of the target entity",
                },
                "relationship": {
                    "type": "string",
                    "description": (
                        "Relationship type: owns, employs, uses, produces, "
                        "depends_on, competes_with, regulates, part_of, "
                        "located_in, causes, preceded_by, similar_to"
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in [0, 1] (default: 0.5)",
                },
            },
            "required": ["source_name", "target_name", "relationship"],
        },
    },
    {
        "name": "ewm_audit",
        "description": "Run a full system audit checking all claims for issues",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "ewm_session_sync",
        "description": "Extract knowledge from a session transcript",
        "inputSchema": {
            "type": "object",
            "properties": {
                "transcript": {
                    "type": "string",
                    "description": "Session transcript text to process",
                },
                "topic": {
                    "type": "string",
                    "description": "Optional topic hint for context generation",
                },
            },
            "required": ["transcript"],
        },
    },
    {
        "name": "ewm_stats",
        "description": "Get system statistics and health metrics",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "ewm_self_inspect",
        "description": "Inspect a module's source code and API",
        "inputSchema": {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": (
                        "Module name without extension (e.g. 'perception', "
                        "'types', 'world_model')"
                    ),
                },
            },
            "required": ["module"],
        },
    },
    {
        "name": "ewm_self_propose",
        "description": "Propose a modification to a module (returns proposal, never auto-applies)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "description": "Module name without extension",
                },
                "description": {
                    "type": "string",
                    "description": "What should change",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why the change is warranted",
                },
            },
            "required": ["module", "description", "rationale"],
        },
    },
]


# ===========================================================================
# Tool Handlers
# ===========================================================================

def _handle_ingest(arguments: Dict[str, Any], db: Any) -> str:
    """Handle the ewm_ingest tool call."""
    from ewm.configurator import ingest

    text = arguments["text"]
    source_type = arguments.get("source_type", "document")
    result = ingest(text, db, source_type=source_type)
    return _to_json(result)


def _handle_query(arguments: Dict[str, Any], db: Any) -> str:
    """Handle the ewm_query tool call."""
    from ewm.configurator import query

    query_text = arguments["query"]
    limit = arguments.get("limit", 20)
    result = query(query_text, db, limit=limit)
    return _to_json(result)


def _handle_claim_create(arguments: Dict[str, Any], db: Any) -> str:
    """Handle the ewm_claim_create tool call."""
    from ewm.types import Claim, ClaimType, Uncertainty

    text = arguments["text"]
    claim_type_str = arguments.get("claim_type", "factual")
    confidence = arguments.get("confidence", 0.5)

    claim_type = ClaimType(claim_type_str)
    uncertainty = Uncertainty.from_confidence(confidence)
    claim = Claim(text=text, claim_type=claim_type, uncertainty=uncertainty)
    db.save_claim(claim)

    return _to_json({
        "claim_id": claim.id,
        "text": claim.text,
        "claim_type": claim.claim_type.value,
        "uncertainty": {
            "belief": uncertainty.belief,
            "disbelief": uncertainty.disbelief,
            "uncertainty": uncertainty.uncertainty,
            "expected_value": uncertainty.expected_value,
        },
    })


def _handle_claim_update(arguments: Dict[str, Any], db: Any) -> str:
    """Handle the ewm_claim_update tool call."""
    from ewm.types import Evidence, SourceType
    from ewm.world_model import integrate_evidence

    claim_id = arguments["claim_id"]
    evidence_text = arguments["evidence_text"]
    direction = arguments.get("direction", "support")

    claim = db.get_claim(claim_id)
    if claim is None:
        return _to_json({"error": "Claim not found", "claim_id": claim_id})

    evidence = Evidence(
        source_type=SourceType.DOCUMENT,
        content=evidence_text,
        source_id="mcp_server",
    )
    db.save_evidence(evidence)

    state = db.load_world_state()
    updated_state = integrate_evidence(state, evidence, claim_id, direction)

    updated_claim = updated_state.claims.get(claim_id)
    if updated_claim is not None:
        db.update_claim_uncertainty(claim_id, updated_claim.uncertainty)
        # Link evidence to claim
        if evidence.id not in claim.evidence_ids:
            claim.evidence_ids.append(evidence.id)
            claim.uncertainty = updated_claim.uncertainty
            db.save_claim(claim)

        u = updated_claim.uncertainty
        return _to_json({
            "claim_id": claim_id,
            "evidence_id": evidence.id,
            "direction": direction,
            "updated_uncertainty": {
                "belief": u.belief,
                "disbelief": u.disbelief,
                "uncertainty": u.uncertainty,
                "expected_value": u.expected_value,
                "confidence": u.confidence,
            },
        })

    return _to_json({"error": "Failed to update claim", "claim_id": claim_id})


def _handle_claim_investigate(arguments: Dict[str, Any], db: Any) -> str:
    """Handle the ewm_claim_investigate tool call."""
    from ewm.configurator import investigate

    top_k = arguments.get("top_k", 5)
    result = investigate(db, top_k=top_k)
    return _to_json(result)


def _handle_entity_create(arguments: Dict[str, Any], db: Any) -> str:
    """Handle the ewm_entity_create tool call."""
    from ewm.types import Entity, EntityCategory

    name = arguments["name"]
    category_str = arguments.get("category", "concept")
    aliases_str = arguments.get("aliases", "")
    properties_str = arguments.get("properties", "{}")

    category = EntityCategory(category_str)
    aliases = [a.strip() for a in aliases_str.split(",") if a.strip()] if aliases_str else []

    try:
        properties = json.loads(properties_str) if properties_str else {}
    except json.JSONDecodeError:
        properties = {}

    entity = Entity(
        name=name,
        category=category,
        aliases=aliases,
        properties=properties,
    )
    db.save_entity(entity)

    return _to_json({
        "entity_id": entity.id,
        "name": entity.name,
        "category": entity.category.value,
        "aliases": entity.aliases,
    })


def _handle_entity_relate(arguments: Dict[str, Any], db: Any) -> str:
    """Handle the ewm_entity_relate tool call."""
    from ewm.types import Relationship, RelationshipType

    source_name = arguments["source_name"]
    target_name = arguments["target_name"]
    rel_type_str = arguments["relationship"]
    confidence = arguments.get("confidence", 0.5)

    # Look up entities by name
    source_entities = db.find_entities(name=source_name, limit=1)
    target_entities = db.find_entities(name=target_name, limit=1)

    if not source_entities:
        return _to_json({"error": "Source entity not found", "source_name": source_name})
    if not target_entities:
        return _to_json({"error": "Target entity not found", "target_name": target_name})

    rel_type = RelationshipType(rel_type_str)
    relationship = Relationship(
        source_id=source_entities[0].id,
        target_id=target_entities[0].id,
        rel_type=rel_type,
        confidence=confidence,
    )
    db.save_relationship(relationship)

    return _to_json({
        "relationship_id": relationship.id,
        "source": source_entities[0].name,
        "target": target_entities[0].name,
        "relationship": rel_type.value,
        "confidence": confidence,
    })


def _handle_audit(arguments: Dict[str, Any], db: Any) -> str:
    """Handle the ewm_audit tool call."""
    from ewm.configurator import audit

    result = audit(db)
    return _to_json(result)


def _handle_session_sync(arguments: Dict[str, Any], db: Any) -> str:
    """Handle the ewm_session_sync tool call."""
    from ewm.configurator import session_sync

    transcript = arguments["transcript"]
    # topic is accepted but session_sync uses internal config
    result = session_sync(transcript, db)
    return _to_json(result)


def _handle_stats(arguments: Dict[str, Any], db: Any) -> str:
    """Handle the ewm_stats tool call."""
    from ewm.self_modify import system_health

    db_stats = db.stats()
    health = system_health()
    return _to_json({
        "database": db_stats,
        "system_health": {
            "total_modules": health.get("total_modules", 0),
            "total_lines": health.get("total_lines", 0),
            "total_size_bytes": health.get("total_size_bytes", 0),
            "import_status": health.get("import_status", {}),
        },
    })


def _handle_self_inspect(arguments: Dict[str, Any], db: Any) -> str:
    """Handle the ewm_self_inspect tool call."""
    from ewm.self_modify import inspect_module

    module_name = arguments["module"]
    info = inspect_module(module_name)

    # Exclude full source for brevity -- return metadata only
    return _to_json({
        "name": info.get("name", module_name),
        "exists": info.get("exists", False),
        "path": info.get("path", ""),
        "lines": info.get("lines", 0),
        "size_bytes": info.get("size_bytes", 0),
        "docstring": info.get("docstring", ""),
        "functions": info.get("functions", []),
        "classes": info.get("classes", []),
        "imports": info.get("imports", []),
    })


def _handle_self_propose(arguments: Dict[str, Any], db: Any) -> str:
    """Handle the ewm_self_propose tool call."""
    from ewm.self_modify import propose_change

    module_name = arguments["module"]
    description = arguments["description"]
    rationale = arguments["rationale"]
    proposal = propose_change(module_name, description, rationale)

    # Exclude full source for brevity
    return _to_json({
        "module": proposal.get("module", module_name),
        "description": proposal.get("description", description),
        "rationale": proposal.get("rationale", rationale),
        "current_lines": proposal.get("current_lines", 0),
        "status": proposal.get("status", "proposed"),
        "note": proposal.get("note", ""),
        "timestamp": proposal.get("timestamp", ""),
    })


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS: Dict[str, Any] = {
    "ewm_ingest": _handle_ingest,
    "ewm_query": _handle_query,
    "ewm_claim_create": _handle_claim_create,
    "ewm_claim_update": _handle_claim_update,
    "ewm_claim_investigate": _handle_claim_investigate,
    "ewm_entity_create": _handle_entity_create,
    "ewm_entity_relate": _handle_entity_relate,
    "ewm_audit": _handle_audit,
    "ewm_session_sync": _handle_session_sync,
    "ewm_stats": _handle_stats,
    "ewm_self_inspect": _handle_self_inspect,
    "ewm_self_propose": _handle_self_propose,
}


# ===========================================================================
# MCP Server Setup (guarded import)
# ===========================================================================

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    import mcp.types as mcp_types

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False


def _setup_server(db: Any) -> Any:
    """Create and configure the MCP server with all tool handlers.

    Only called when the mcp package is available.

    Args:
        db: Database instance to pass to all tool handlers.

    Returns:
        Configured mcp.server.Server instance.
    """
    server = Server("ewm")

    @server.list_tools()
    async def list_tools() -> List[Any]:
        return [
            mcp_types.Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["inputSchema"],
            )
            for tool in TOOL_DEFINITIONS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[Any]:
        handler = _HANDLERS.get(name)
        if handler is None:
            raise ValueError("Unknown tool: {}".format(name))

        try:
            result_text = handler(arguments, db)
        except Exception as exc:
            # Let the MCP framework handle tool errors properly.
            # Re-raising ensures the client receives an error response,
            # not a success payload containing an error message.
            raise RuntimeError(
                "Tool '{}' failed: {}".format(name, exc)
            ) from exc

        return [mcp_types.TextContent(type="text", text=result_text)]

    return server


# ===========================================================================
# Entry Point
# ===========================================================================


async def _async_main() -> None:
    """Async entry point: initialize DB and run MCP server."""
    from ewm.db import Database

    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    db = Database(DB_PATH)
    try:
        server = _setup_server(db)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        db.close()


def main() -> None:
    """Entry point for the MCP server.

    If the mcp package is not installed, prints an error and exits.
    Otherwise, starts the async MCP server loop.
    """
    if not _HAS_MCP:
        print(
            "ERROR: The 'mcp' package is required to run the EWM MCP server.\n"
            "Install it with: pip install mcp\n"
            "The ewm package works without mcp for direct Python usage."
        )
        raise SystemExit(1)

    import asyncio
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
