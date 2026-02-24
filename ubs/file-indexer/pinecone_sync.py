#!/usr/bin/env python3
"""
Pinecone Sync for Knowledge Index
=================================

Syncs the knowledge.yaml file to Pinecone for semantic search.

Features:
- Incremental sync via content hash (only changed records upserted)
- Batch upserts of 50 records
- Dry-run mode for testing
- Graceful handling of missing OWC drive

Usage:
    python pinecone_sync.py                    # Full sync
    python pinecone_sync.py --dry-run          # Preview without syncing
    python pinecone_sync.py --force            # Force full re-sync

Namespace: local-files in claude-knowledge-base index
State: ~/.cache/file-indexer/pinecone_state.json
"""

import json
import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ============ CONFIGURATION ============

KNOWLEDGE_FILE = Path.home() / "knowledge.yaml"
STATE_FILE = Path.home() / ".cache/file-indexer/pinecone_state.json"
PINECONE_INDEX = "claude-knowledge-base"
PINECONE_NAMESPACE = "local-files"
BATCH_SIZE = 50

# Record type limits (to avoid overwhelming the index)
MAX_PROJECTS = 100
MAX_DOCUMENTS = 200
MAX_CODE_PATTERNS = 100
MAX_CONFIGS = 50  # configs are numerous but less valuable for search


# ============ SAFETY CHECKS ============

def check_prerequisites() -> bool:
    """Check if prerequisites are met."""
    if not KNOWLEDGE_FILE.exists():
        log.error(f"Knowledge file not found: {KNOWLEDGE_FILE}")
        log.info("Run the indexer first: python indexer.py")
        return False
    return True


def get_pinecone_client():
    """Initialize Pinecone client from environment."""
    try:
        from pinecone import Pinecone
    except ImportError:
        log.error("Pinecone SDK not installed. Run: pip3 install pinecone")
        return None

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        # Try loading from common locations
        env_file = Path.home() / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("PINECONE_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"\'')
                    break

    if not api_key:
        log.error("PINECONE_API_KEY not found in environment or ~/.env")
        return None

    try:
        return Pinecone(api_key=api_key)
    except Exception as e:
        log.error(f"Failed to initialize Pinecone: {e}")
        return None


# ============ STATE MANAGEMENT ============

def load_state() -> Dict:
    """Load previous sync state."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception as e:
            log.warning(f"Failed to load state: {e}")
    return {"record_hashes": {}, "last_sync": None}


def save_state(state: Dict):
    """Save sync state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def compute_hash(content: str) -> str:
    """Compute content hash for change detection."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ============ RECORD CONVERSION ============

def project_to_record(project: Dict, index: int) -> Dict:
    """Convert a project to a Pinecone record."""
    name = project.get("name", "unknown")
    path = project.get("path", "")
    proj_type = project.get("type", "unknown")
    description = (project.get("description") or "")[:500] or "No description"
    dependencies = project.get("dependencies", [])
    key_files = project.get("key_files", [])
    last_modified = project.get("last_modified", "")

    # Build searchable content
    key_file_list = "\n".join([
        f"  - {kf.get('path', '')}: {kf.get('purpose', '')[:100]}"
        for kf in key_files[:5]
    ])

    content = f"""PROJECT: {name}
Type: {proj_type}
Path: {path}

Description:
{description}

Key Files:
{key_file_list}

Dependencies: {', '.join(dependencies[:10]) if dependencies else 'None listed'}

Last Modified: {last_modified}
"""

    record_id = f"project_{name.lower().replace(' ', '_').replace('/', '_')[:50]}"

    return {
        "_id": record_id,
        "content": content,
        "record_type": "project",
        "name": name,
        "project_type": proj_type,
        "path": path,
        "last_modified": last_modified,
        "source": "file-indexer",
    }


def document_to_record(doc: Dict, category: str, index: int) -> Dict:
    """Convert a document to a Pinecone record."""
    path = doc.get("path", "")
    preview = (doc.get("preview") or "")[:400] or "No preview available"
    doc_format = doc.get("format", "unknown")
    modified = doc.get("modified", "")

    # Extract filename for ID
    filename = Path(path).stem[:30]

    content = f"""DOCUMENT: {Path(path).name}
Category: {category}
Format: {doc_format}
Path: {path}

Content Preview:
{preview}

Modified: {modified}
"""

    # Create unique ID from path hash
    path_hash = hashlib.md5(path.encode()).hexdigest()[:8]
    record_id = f"doc_{category}_{filename}_{path_hash}"

    return {
        "_id": record_id,
        "content": content,
        "record_type": "document",
        "category": category,
        "doc_format": doc_format,
        "path": path,
        "modified": modified,
        "source": "file-indexer",
    }


def code_pattern_to_record(pattern: Dict, index: int) -> Dict:
    """Convert a code pattern to a Pinecone record."""
    file_path = pattern.get("file", "")
    classes = pattern.get("classes", [])
    language = pattern.get("language", "unknown")

    # Extract project/module name from path
    parts = Path(file_path).parts
    project_name = ""
    for i, part in enumerate(parts):
        if part in ("Dev", "Projects", "Archive AI Projects"):
            if i + 1 < len(parts):
                project_name = parts[i + 1]
            break

    content = f"""CODE PATTERN: {Path(file_path).name}
Project: {project_name}
Language: {language}
Path: {file_path}

Classes/Components:
{chr(10).join(f'  - {cls}' for cls in classes)}

Use this file when you need:
- Classes: {', '.join(classes[:5])}
- {language.upper()} implementation patterns from {project_name}
"""

    filename = Path(file_path).stem[:30]
    path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    record_id = f"codepattern_{filename}_{path_hash}"

    return {
        "_id": record_id,
        "content": content,
        "record_type": "code_pattern",
        "language": language,
        "classes": ",".join(classes[:20]),
        "path": file_path,
        "project": project_name,
        "source": "file-indexer",
    }


# ============ MAIN SYNC LOGIC ============

def load_knowledge() -> Optional[Dict]:
    """Load and parse knowledge.yaml."""
    try:
        import yaml
        return yaml.safe_load(KNOWLEDGE_FILE.read_text())
    except ImportError:
        log.error("PyYAML not installed. Run: pip3 install pyyaml")
        return None
    except Exception as e:
        log.error(f"Failed to load knowledge.yaml: {e}")
        return None


def build_records(knowledge: Dict) -> List[Dict]:
    """Build Pinecone records from knowledge data."""
    records = []

    # Projects
    projects = knowledge.get("projects", [])
    log.info(f"Processing {len(projects)} projects (max {MAX_PROJECTS})")
    for i, proj in enumerate(projects[:MAX_PROJECTS]):
        records.append(project_to_record(proj, i))

    # Documents
    documents = knowledge.get("documents", {})
    doc_count = 0
    for category, docs in documents.items():
        if doc_count >= MAX_DOCUMENTS:
            break
        for doc in docs:
            if doc_count >= MAX_DOCUMENTS:
                break
            records.append(document_to_record(doc, category, doc_count))
            doc_count += 1
    log.info(f"Processed {doc_count} documents (max {MAX_DOCUMENTS})")

    # Code patterns
    patterns = knowledge.get("code_patterns", [])
    log.info(f"Processing {len(patterns)} code patterns (max {MAX_CODE_PATTERNS})")
    for i, pattern in enumerate(patterns[:MAX_CODE_PATTERNS]):
        records.append(code_pattern_to_record(pattern, i))

    return records


def filter_changed_records(
    records: List[Dict],
    state: Dict,
    force: bool = False
) -> Tuple[List[Dict], Dict]:
    """Filter to only records that have changed."""
    if force:
        log.info("Force mode: syncing all records")
        new_hashes = {r["_id"]: compute_hash(r["content"]) for r in records}
        return records, new_hashes

    old_hashes = state.get("record_hashes", {})
    changed = []
    new_hashes = {}

    for record in records:
        record_id = record["_id"]
        content_hash = compute_hash(record["content"])
        new_hashes[record_id] = content_hash

        if old_hashes.get(record_id) != content_hash:
            changed.append(record)

    log.info(f"Changed records: {len(changed)} of {len(records)} total")
    return changed, new_hashes


def upsert_to_pinecone(pc, records: List[Dict], dry_run: bool = False) -> bool:
    """Upsert records to Pinecone."""
    if dry_run:
        log.info(f"DRY RUN: Would upsert {len(records)} records")
        for record in records[:5]:
            log.info(f"  - {record['_id']}: {record['record_type']}")
        if len(records) > 5:
            log.info(f"  ... and {len(records) - 5} more")
        return True

    try:
        index = pc.Index(PINECONE_INDEX)

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            index.upsert_records(PINECONE_NAMESPACE, batch)
            log.info(f"Upserted batch {i // BATCH_SIZE + 1}: {len(batch)} records")

        return True
    except Exception as e:
        log.error(f"Failed to upsert to Pinecone: {e}")
        return False


def run_sync(dry_run: bool = False, force: bool = False) -> bool:
    """Main sync function."""
    log.info(f"Starting Pinecone sync...")
    log.info(f"Index: {PINECONE_INDEX}, Namespace: {PINECONE_NAMESPACE}")
    log.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}{' (FORCE)' if force else ''}")
    print()

    # Load knowledge
    knowledge = load_knowledge()
    if not knowledge:
        return False

    # Show stats
    meta = knowledge.get("meta", {})
    stats = meta.get("stats", {})
    log.info(f"Knowledge file stats:")
    log.info(f"  Total files: {stats.get('total_files', 'unknown')}")
    log.info(f"  Generated: {meta.get('generated_at', 'unknown')}")
    print()

    # Build records
    records = build_records(knowledge)
    log.info(f"Built {len(records)} total records")
    print()

    # Load state and filter to changed
    state = load_state()
    changed_records, new_hashes = filter_changed_records(records, state, force)

    if not changed_records:
        log.info("No changes detected. Nothing to sync.")
        return True

    # Connect to Pinecone
    if not dry_run:
        pc = get_pinecone_client()
        if not pc:
            return False
    else:
        pc = None

    # Upsert
    success = upsert_to_pinecone(pc, changed_records, dry_run)

    if success and not dry_run:
        # Update state
        state["record_hashes"] = new_hashes
        state["last_sync"] = datetime.utcnow().isoformat()
        state["records_synced"] = len(changed_records)
        save_state(state)
        log.info(f"State saved. Last sync: {state['last_sync']}")

    return success


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(description="Sync knowledge.yaml to Pinecone")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without syncing")
    parser.add_argument("--force", action="store_true",
                        help="Force full re-sync (ignore state)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not check_prerequisites():
        sys.exit(1)

    success = run_sync(dry_run=args.dry_run, force=args.force)

    if success:
        print("\n✓ Sync complete!")
    else:
        print("\n✗ Sync failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
