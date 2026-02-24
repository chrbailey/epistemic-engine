#!/usr/bin/env python3
"""
EWM Session Start Hook — replaces session-context-loader.py

Fires on SessionStart to inject EWM world model context into Claude Code.

Input:  JSON on stdin from Claude Code hook system
Output: JSON on stdout with {"additionalContext": "..."}

Uses EWM's memory.context_for_session() to generate live context
from the SQLite database instead of reading flat files.
During transition, also reads legacy pending_learnings.json.
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

# Safety: exit silently if OWC drive not mounted
if not Path("/Volumes/OWC drive").exists():
    sys.exit(0)

# EWM project path — add to sys.path so we can import ewm
EWM_PROJECT = Path("/Volumes/OWC drive/Dev/epistemic-world-model")
sys.path.insert(0, str(EWM_PROJECT))

# Unified database path (shared between hooks and MCP server)
EWM_DB_PATH = Path("/Volumes/OWC drive/Dev/epistemic-world-model/ewm_data.db")

# Legacy paths for backward compatibility during transition
LEGACY_PENDING = Path("/Volumes/OWC drive/Knowledge/hooks/pending_learnings.json")
LEGACY_PRESENTED = Path("/Volumes/OWC drive/Knowledge/hooks/presented_learnings.json")
ERROR_LOG = Path("/Volumes/OWC drive/Knowledge/hooks/error.log")


def log_error(message: str):
    """Append error to log file."""
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG, "a") as f:
            f.write(f"{datetime.utcnow().isoformat()} - EWM start hook: {message}\n")
    except IOError:
        pass


def load_legacy_pending() -> list:
    """Load pending learnings from the old hook system (transition period)."""
    try:
        if LEGACY_PENDING.exists():
            data = json.loads(LEGACY_PENDING.read_text())
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        pass
    return []


def format_legacy_learning(learning: dict) -> str:
    """Format a legacy 5W1H learning for presentation."""
    lines = []
    confidence = learning.get("confidence", 0)
    conf_bar = "●" * int(confidence * 5) + "○" * (5 - int(confidence * 5))

    lines.append(f"Learning (confidence: [{conf_bar}] {confidence:.0%})")
    lines.append(f"  WHAT: {learning.get('what', 'N/A')}")

    where = learning.get("where", "")
    if where and where != "Unknown":
        lines.append(f"  WHERE: {where}")

    ai_insight = learning.get("ai_insight", "")
    if ai_insight and ai_insight != "No synthesis available":
        lines.append(f"  INSIGHT: {ai_insight}")

    return "\n".join(lines)


def compute_hash(learning: dict) -> str:
    """Hash a learning for dedup."""
    text = learning.get("what", "") + learning.get("session_id", "")
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def get_presented_hashes() -> set:
    """Get set of already-presented learning hashes."""
    try:
        if LEGACY_PRESENTED.exists():
            data = json.loads(LEGACY_PRESENTED.read_text())
            return set(data.get("presented", []))
    except (json.JSONDecodeError, IOError):
        pass
    return set()


def mark_presented(hashes: list):
    """Mark learnings as presented in legacy tracker."""
    try:
        data = {"presented": [], "last_updated": None}
        if LEGACY_PRESENTED.exists():
            data = json.loads(LEGACY_PRESENTED.read_text())
        existing = set(data.get("presented", []))
        existing.update(hashes)
        data["presented"] = list(existing)[-100:]
        data["last_updated"] = datetime.utcnow().isoformat()
        LEGACY_PRESENTED.parent.mkdir(parents=True, exist_ok=True)
        LEGACY_PRESENTED.write_text(json.dumps(data, indent=2))
    except (json.JSONDecodeError, IOError):
        pass


def clear_legacy_pending():
    """Clear the legacy pending queue after presentation."""
    try:
        if LEGACY_PENDING.exists():
            LEGACY_PENDING.write_text("[]")
    except IOError:
        pass


def main():
    # Parse hook input (consumed from stdin)
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    output_parts = []

    # --- Phase 1: Present legacy pending learnings (transition period) ---
    pending = load_legacy_pending()
    already_presented = get_presented_hashes()

    new_learnings = []
    for learning in pending:
        h = compute_hash(learning)
        if h not in already_presented:
            new_learnings.append((learning, h))

    if new_learnings:
        new_learnings.sort(key=lambda x: x[0].get("confidence", 0), reverse=True)
        presented_hashes = []
        for learning, h in new_learnings[:3]:
            if learning.get("confidence", 0) >= 0.3:
                output_parts.append(format_legacy_learning(learning))
                presented_hashes.append(h)
        if presented_hashes:
            mark_presented(presented_hashes)

    if pending:
        clear_legacy_pending()

    # --- Phase 2: Generate EWM context from live database ---
    try:
        from ewm.db import Database
        from ewm.memory import context_for_session

        if EWM_DB_PATH.exists():
            db = Database(str(EWM_DB_PATH))
            try:
                # Detect session topic from hook input
                session_topic = ""
                workspace = hook_input.get("workspace", {})
                if isinstance(workspace, dict):
                    cwd = workspace.get("current_dir", "")
                    if cwd:
                        session_topic = Path(cwd).name

                context = context_for_session(db, session_topic)
                if context:
                    output_parts.append(context)
            finally:
                db.close()
    except Exception as exc:
        log_error(f"EWM context generation failed: {exc}")

    # --- Output for Claude Code ---
    if output_parts:
        full_output = "\n\n".join(output_parts)
        # Truncate to ~800 tokens (~3200 chars)
        max_chars = 3200
        if len(full_output) > max_chars:
            full_output = full_output[:max_chars] + "\n...[truncated]"
        print(json.dumps({"additionalContext": full_output}))

    sys.exit(0)


if __name__ == "__main__":
    main()
