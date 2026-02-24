#!/bin/bash
# Sync knowledge index from Mac Mini (run on MacBook Pro)
# This pulls the remote partial index and consolidates

set -e

REMOTE_HOST="${1:-mac-mini.local}"
REMOTE_USER="${2:-$USER}"
LOCAL_INDEX="$HOME/Dropbox/knowledge.yaml"
REMOTE_INDEX="/tmp/remote-knowledge.yaml"
MERGED_INDEX="$HOME/Dropbox/knowledge-merged.yaml"

echo "=== Knowledge Index Sync ==="
echo "Remote: $REMOTE_USER@$REMOTE_HOST"
echo ""

# First, run indexer on remote machine
echo "Running indexer on remote..."
ssh "$REMOTE_USER@$REMOTE_HOST" \
    "python3 ~/Desktop/unified-belief-system/file-indexer/indexer.py \
     --output /tmp/knowledge-partial.yaml 2>&1" || {
    echo "Remote indexer failed. Using existing remote index if available."
}

# Pull remote index
echo "Pulling remote index..."
scp "$REMOTE_USER@$REMOTE_HOST:/tmp/knowledge-partial.yaml" "$REMOTE_INDEX" || {
    echo "No remote index found."
    exit 0
}

# Merge indices (simple concatenation for now)
# A proper merge would dedupe and combine
echo "Merging indices..."

python3 << 'EOF'
import yaml
from pathlib import Path
import sys

local_path = Path.home() / "Dropbox" / "knowledge.yaml"
remote_path = Path("/tmp/remote-knowledge.yaml")
merged_path = Path.home() / "Dropbox" / "knowledge-merged.yaml"

def load_yaml(path):
    if path.exists():
        return yaml.safe_load(path.read_text())
    return {}

local = load_yaml(local_path)
remote = load_yaml(remote_path)

if not local:
    print("No local index found")
    sys.exit(1)

# Merge sources
local_sources = local.get("meta", {}).get("sources", [])
remote_sources = remote.get("meta", {}).get("sources", [])
local["meta"]["sources"] = local_sources + remote_sources

# Merge projects
local_projects = local.get("projects", [])
remote_projects = remote.get("projects", [])
seen_names = {p["name"] for p in local_projects}
for p in remote_projects:
    if p["name"] not in seen_names:
        local_projects.append(p)
local["projects"] = local_projects

# Update stats
local["meta"]["stats"]["total_files"] += remote.get("meta", {}).get("stats", {}).get("total_files", 0)

# Write merged
merged_path.write_text(yaml.dump(local, default_flow_style=False, sort_keys=False))
print(f"Merged index: {merged_path}")
print(f"  Local sources: {len(local_sources)}")
print(f"  Remote sources: {len(remote_sources)}")
print(f"  Total projects: {len(local_projects)}")
EOF

echo ""
echo "=== Sync Complete ==="
echo "Merged index: $MERGED_INDEX"
