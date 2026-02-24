#!/usr/bin/env python3
"""
Batch reprocess large transcripts that were rejected by the old 10MB limit.

This script:
1. Finds all transcripts >10MB in ~/.claude/projects
2. Processes each through the EWM pipeline (with truncation)
3. Logs results to sync_summary.log

Run: python batch_reprocess_transcripts.py [--dry-run]
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add paths
EWM_PROJECT = Path("/Volumes/OWC drive/Dev/epistemic-world-model")
sys.path.insert(0, str(EWM_PROJECT))
sys.path.insert(0, str(EWM_PROJECT / "hooks"))

from ewm_session_stop_hook import read_transcript_as_text

# Paths
HOOKS_DIR = Path("/Volumes/OWC drive/Knowledge/hooks")
SYNC_LOG = HOOKS_DIR / "sync_summary.log"
REPROCESS_LOG = HOOKS_DIR / "batch_reprocess.log"
EWM_DB_PATH = EWM_PROJECT / "ewm_data.db"

# Track what we've already processed
PROCESSED_TRACKER = HOOKS_DIR / "batch_processed.json"


def get_processed_hashes() -> set:
    """Load set of already-processed transcript hashes."""
    try:
        if PROCESSED_TRACKER.exists():
            return set(json.loads(PROCESSED_TRACKER.read_text()))
    except (json.JSONDecodeError, IOError):
        pass
    return set()


def save_processed_hash(hash_val: str):
    """Add a hash to the processed set."""
    processed = get_processed_hashes()
    processed.add(hash_val)
    PROCESSED_TRACKER.write_text(json.dumps(list(processed), indent=2))


def find_large_transcripts(min_size_mb: int = 10) -> list:
    """Find all transcript files larger than min_size_mb."""
    claude_projects = Path.home() / ".claude" / "projects"
    if not claude_projects.exists():
        return []

    large_files = []
    min_bytes = min_size_mb * 1_000_000

    for jsonl_file in claude_projects.rglob("*.jsonl"):
        try:
            size = jsonl_file.stat().st_size
            if size > min_bytes:
                large_files.append((jsonl_file, size))
        except OSError:
            continue

    # Sort by size descending
    large_files.sort(key=lambda x: x[1], reverse=True)
    return large_files


def process_transcript(transcript_path: Path, dry_run: bool = False) -> dict:
    """Process a single transcript through EWM pipeline."""
    result = {
        "path": str(transcript_path),
        "size_mb": transcript_path.stat().st_size / 1_000_000,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Read and truncate transcript
    transcript_text = read_transcript_as_text(str(transcript_path))

    if not transcript_text or len(transcript_text) < 200:
        result["status"] = "skipped"
        result["reason"] = "empty or too short"
        return result

    # Check if already processed (by content hash)
    text_hash = hashlib.sha256(transcript_text[:10000].encode()).hexdigest()[:16]
    processed = get_processed_hashes()

    if text_hash in processed:
        result["status"] = "skipped"
        result["reason"] = "already processed"
        return result

    result["transcript_chars"] = len(transcript_text)
    result["content_hash"] = text_hash

    if dry_run:
        result["status"] = "dry_run"
        return result

    # Run EWM pipeline
    try:
        from ewm.db import Database
        from ewm.configurator import session_sync

        db = Database(str(EWM_DB_PATH))

        try:
            # Configure Pinecone if available
            pinecone_key = os.environ.get("PINECONE_API_KEY", "")
            if pinecone_key:
                from ewm.memory import configure_pinecone
                configure_pinecone(api_key=pinecone_key)

            # Run pipeline
            sync_result = session_sync(transcript_text, db)

            result["status"] = sync_result.get("status", "unknown")
            extracted = sync_result.get("result", {}).get("extracted", {})
            result["entities_stored"] = extracted.get("entities_stored", 0)
            result["claims_stored"] = extracted.get("claims_stored", 0)
            result["evidence_stored"] = extracted.get("evidence_stored", 0)
            result["relationships_stored"] = extracted.get("relationships_stored", 0)
            result["duplicates_merged"] = extracted.get("duplicates_merged", 0)

            # Mark as processed
            save_processed_hash(text_hash)

        finally:
            db.close()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 70)
    print("BATCH REPROCESS LARGE TRANSCRIPTS")
    print("=" * 70)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    # Find large transcripts
    large_files = find_large_transcripts(min_size_mb=10)

    if not large_files:
        print("No large transcripts found.")
        return

    print(f"Found {len(large_files)} transcripts >10MB:")
    print("-" * 70)

    total_size = sum(size for _, size in large_files)
    print(f"Total size: {total_size / 1_000_000_000:.2f} GB")
    print()

    # Process each
    results = []
    for i, (path, size) in enumerate(large_files, 1):
        size_mb = size / 1_000_000
        print(f"[{i}/{len(large_files)}] Processing {path.name} ({size_mb:.1f} MB)...")

        result = process_transcript(path, dry_run=dry_run)
        results.append(result)

        status = result.get("status", "unknown")
        if status == "completed":
            claims = result.get("claims_stored", 0)
            entities = result.get("entities_stored", 0)
            print(f"         ✓ {status}: {claims} claims, {entities} entities")
        elif status == "skipped":
            print(f"         → {status}: {result.get('reason', '')}")
        elif status == "dry_run":
            chars = result.get("transcript_chars", 0)
            print(f"         → would process {chars:,} chars")
        else:
            print(f"         ✗ {status}: {result.get('error', '')[:50]}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    completed = sum(1 for r in results if r.get("status") == "completed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    errors = sum(1 for r in results if r.get("status") == "error")

    total_claims = sum(r.get("claims_stored", 0) for r in results)
    total_entities = sum(r.get("entities_stored", 0) for r in results)

    print(f"Processed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")
    print(f"Total claims stored:   {total_claims}")
    print(f"Total entities stored: {total_entities}")

    # Log results
    if not dry_run:
        REPROCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(REPROCESS_LOG, "a") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        print(f"\nResults logged to: {REPROCESS_LOG}")


if __name__ == "__main__":
    main()
