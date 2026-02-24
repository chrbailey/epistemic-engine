#!/usr/bin/env python3
"""
EWM Session Stop Hook — replaces knowledge-sync-hook.py

Fires on Stop/SubagentStop to process session transcript through EWM pipeline.

Input:  JSON on stdin from Claude Code hook system
        Expected keys: session_id, transcript_path, hook_event_name
Output: None (silent). Writes to EWM database, Pinecone, and log files.

Pipeline:
    1. Read transcript JSONL and flatten to text
    2. Call configurator.session_sync() which runs:
       - perception.perceive() — extract entities, claims, relationships
       - memory.store() — persist to SQLite
       - memory.context_for_session() — rebuild context
    3. Sync entities/claims to Pinecone (if PINECONE_API_KEY is set)
       - Index: claude-knowledge-base, Namespace: ewm
       - Uses Pinecone SDK with integrated inference
       - Batches in groups of 96
    4. Log summary to sync_summary.log (includes pinecone_synced count)

Environment:
    PINECONE_API_KEY — If set, enables Pinecone sync after SQLite storage
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Safety: exit silently if OWC drive not mounted
if not Path("/Volumes/OWC drive").exists():
    sys.exit(0)

# Disable flag
DISABLE_FLAG = Path("/Volumes/OWC drive/Knowledge/hooks/.disabled")
if DISABLE_FLAG.exists():
    sys.exit(0)

# EWM project path
EWM_PROJECT = Path("/Volumes/OWC drive/Dev/epistemic-world-model")
sys.path.insert(0, str(EWM_PROJECT))

# Unified database path (shared between hooks and MCP server)
EWM_DB_PATH = Path("/Volumes/OWC drive/Dev/epistemic-world-model/ewm_data.db")

# Paths
HOOKS_DIR = Path("/Volumes/OWC drive/Knowledge/hooks")
SYNC_LOG = HOOKS_DIR / "sync_summary.log"
ERROR_LOG = HOOKS_DIR / "error.log"
SESSION_TRACKER = HOOKS_DIR / "ewm_session_captured.json"

# Security: allowed transcript directories
ALLOWED_TRANSCRIPT_DIRS = [
    Path.home() / ".claude" / "projects",
    Path("/tmp"),
    Path("/var/folders"),
    Path("/private/tmp"),
]


def log_error(message: str):
    """Append error to log file."""
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG, "a") as f:
            f.write(f"{datetime.utcnow().isoformat()} - EWM stop hook: {message}\n")
    except IOError:
        pass


def is_safe_transcript_path(path_str: str) -> bool:
    """Validate transcript path is in an allowed directory."""
    try:
        path = Path(path_str).resolve()
        for allowed in ALLOWED_TRANSCRIPT_DIRS:
            try:
                path.relative_to(allowed.resolve())
                return True
            except ValueError:
                continue
        return False
    except (OSError, ValueError):
        return False


def read_transcript_as_text(transcript_path: str, max_chars: int = 500_000) -> str:
    """Read a JSONL transcript file and flatten to plain text.

    Extracts text content from each message, joining assistant and user
    messages into a single string suitable for EWM perception.

    For large transcripts, returns only the most recent content up to max_chars.
    This ensures valuable recent context is captured even for long sessions.
    """
    parts = []
    try:
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = _extract_message_text(msg)
                if text and len(text) > 20:
                    parts.append(text)
    except (FileNotFoundError, IOError) as e:
        log_error(f"Failed to read transcript: {e}")
        return ""

    full_text = "\n\n".join(parts)

    # Truncate to most recent content if too large
    if len(full_text) > max_chars:
        # Keep the end (most recent) content
        full_text = full_text[-max_chars:]
        # Find first paragraph break to avoid mid-sentence start
        first_break = full_text.find("\n\n")
        if first_break > 0 and first_break < 5000:
            full_text = full_text[first_break + 2:]

    return full_text


def _extract_message_text(msg: dict) -> str:
    """Extract text content from a single transcript message."""
    content = ""

    # Standard format: msg.message.content = [{type: "text", text: "..."}]
    if "message" in msg and isinstance(msg["message"], dict):
        blocks = msg["message"].get("content", [])
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    content += block.get("text", "") + " "

    # Legacy format: msg.content = "..."
    elif "content" in msg and isinstance(msg["content"], str):
        content = msg["content"]

    return content.strip()


def get_session_captured(session_id: str) -> set:
    """Check if this session was already processed."""
    try:
        if SESSION_TRACKER.exists():
            data = json.loads(SESSION_TRACKER.read_text())
            return set(data.get(session_id, []))
    except (json.JSONDecodeError, IOError):
        pass
    return set()


def mark_session_captured(session_id: str, summary_hash: str):
    """Record that this session was processed."""
    try:
        data = {}
        if SESSION_TRACKER.exists():
            data = json.loads(SESSION_TRACKER.read_text())

        existing = set(data.get(session_id, []))
        existing.add(summary_hash)
        data[session_id] = list(existing)

        # Keep only last 50 sessions
        if len(data) > 50:
            sorted_sessions = sorted(data.keys())
            data = {k: data[k] for k in sorted_sessions[-50:]}

        SESSION_TRACKER.parent.mkdir(parents=True, exist_ok=True)
        SESSION_TRACKER.write_text(json.dumps(data, indent=2))
    except (json.JSONDecodeError, IOError) as e:
        log_error(f"Failed to update session tracker: {e}")


def sync_to_pinecone_direct(db, session_id: str, api_key: str) -> int:
    """Sync extracted entities and claims to Pinecone.

    Uses the Pinecone Python SDK for direct upsert to the claude-knowledge-base
    index in the 'ewm' namespace. Records are batched in groups of 96.

    Args:
        db: EWM Database instance with entities/claims.
        session_id: Current session ID for metadata.
        api_key: Pinecone API key.

    Returns:
        Number of records successfully upserted.
    """
    try:
        from pinecone import Pinecone
    except ImportError:
        log_error("Pinecone SDK not installed (pip install pinecone)")
        return 0

    # Initialize client
    pc = Pinecone(api_key=api_key)

    # Get index host
    index_name = "claude-knowledge-base"
    namespace = "ewm"

    try:
        index_info = pc.describe_index(index_name)
        host = index_info.host
    except Exception as exc:
        log_error(f"Failed to describe index {index_name}: {exc}")
        return 0

    index = pc.Index(host=host)

    # Fetch recent entities and claims from database
    # Use last 100 of each to avoid too large batches
    entities = db.find_entities(limit=100)
    claims = db.find_claims(limit=100)

    if not entities and not claims:
        return 0

    # Build records for upsert (using integrated embedding)
    records = []
    timestamp = datetime.utcnow().isoformat()

    for entity in entities:
        # Build content text for embedding
        props_text = " ".join(f"{k}: {v}" for k, v in entity.properties.items())
        content = f"{entity.name} ({entity.category.value}). {props_text}".strip()

        records.append({
            "_id": f"entity:{entity.id}",
            "content": content,  # Field used for embedding
            "type": "entity",
            "category": entity.category.value,
            "session_id": session_id,
            "timestamp": timestamp,
            "source": "ewm-hook",
        })

    for claim in claims:
        records.append({
            "_id": f"claim:{claim.id}",
            "content": claim.text,  # Field used for embedding
            "type": "claim",
            "category": claim.claim_type.value,
            "confidence": str(claim.uncertainty.expected_value),
            "session_id": session_id,
            "timestamp": timestamp,
            "source": "ewm-hook",
        })

    # Batch upsert in groups of 96 (Pinecone's per-request limit)
    batch_size = 96
    total_upserted = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            # Use upsert_records for integrated inference indexes
            index.upsert_records(namespace=namespace, records=batch)
            total_upserted += len(batch)
        except Exception as exc:
            log_error(f"Pinecone batch upsert failed at {i}: {exc}")
            # Continue with remaining batches

    return total_upserted


def main():
    hook_input = {}
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        pass

    session_id = hook_input.get("session_id", "unknown")
    transcript_path = hook_input.get("transcript_path", "")
    event_name = hook_input.get("hook_event_name", "unknown")

    # --- Validation ---
    if not transcript_path or not Path(transcript_path).exists():
        sys.exit(0)

    if not is_safe_transcript_path(transcript_path):
        log_error(f"SECURITY: Rejected transcript path: {transcript_path}")
        sys.exit(0)

    try:
        file_size = Path(transcript_path).stat().st_size
        if file_size > 200_000_000:  # 200 MB - generous limit, truncation handles memory
            log_error(f"SECURITY: Transcript too large ({file_size} bytes)")
            sys.exit(0)
    except OSError:
        sys.exit(0)

    # --- Read and flatten transcript ---
    transcript_text = read_transcript_as_text(transcript_path)
    if not transcript_text or len(transcript_text) < 200:
        sys.exit(0)

    # --- Session dedup ---
    text_hash = hashlib.sha256(transcript_text[:5000].encode()).hexdigest()[:16]
    already_captured = get_session_captured(session_id)
    if text_hash in already_captured:
        sys.exit(0)

    # --- Run EWM pipeline ---
    try:
        from ewm.db import Database
        from ewm.configurator import session_sync

        EWM_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        db = Database(str(EWM_DB_PATH))

        try:
            # Optionally configure Pinecone
            pinecone_key = os.environ.get("PINECONE_API_KEY", "")
            if pinecone_key:
                from ewm.memory import configure_pinecone
                configure_pinecone(api_key=pinecone_key)

            # Run: perceive() -> store() -> context_for_session()
            result = session_sync(transcript_text, db)

            status = result.get("status", "unknown")
            extracted = result.get("result", {}).get("extracted", {})

            # --- Direct Pinecone sync ---
            pinecone_synced = 0
            pinecone_error = None
            if pinecone_key and status == "completed":
                try:
                    pinecone_synced = sync_to_pinecone_direct(
                        db, session_id, pinecone_key
                    )
                except Exception as pc_exc:
                    pinecone_error = str(pc_exc)
                    log_error(f"Pinecone sync failed: {pc_exc}")

            # --- Log summary ---
            summary = {
                "session_id": session_id,
                "event": event_name,
                "engine": "ewm",
                "status": status,
                "entities_stored": extracted.get("entities_stored", 0),
                "claims_stored": extracted.get("claims_stored", 0),
                "evidence_stored": extracted.get("evidence_stored", 0),
                "relationships_stored": extracted.get("relationships_stored", 0),
                "duplicates_merged": extracted.get("duplicates_merged", 0),
                "pinecone_synced": pinecone_synced,
                "pinecone_error": pinecone_error,
                "context_length": result.get("result", {}).get("context_length", 0),
                "transcript_chars": len(transcript_text),
                "timestamp": datetime.utcnow().isoformat(),
            }

            SYNC_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(SYNC_LOG, "a") as f:
                f.write(json.dumps(summary) + "\n")

            mark_session_captured(session_id, text_hash)

        finally:
            db.close()

    except Exception as exc:
        log_error(f"EWM pipeline failed: {exc}")

        failure_summary = {
            "session_id": session_id,
            "event": event_name,
            "engine": "ewm",
            "status": "error",
            "error": str(exc),
            "transcript_chars": len(transcript_text),
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            SYNC_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(SYNC_LOG, "a") as f:
                f.write(json.dumps(failure_summary) + "\n")
        except IOError:
            pass

    sys.exit(0)


if __name__ == "__main__":
    main()
