# Personal Knowledge Indexer

Automatically scans your files across multiple Macs and cloud storage, creating a structured YAML document optimized for LLM context loading.

## Features

- **Multi-source scanning**: Local files, Dropbox, Box, iCloud, Google Drive
- **Remote machine support**: SSH/rsync from Mac Mini or other machines
- **Content extraction**: PDF, DOCX, Markdown, code files
- **Project detection**: Recognizes Python, Node, Rust, Go, etc.
- **Semantic indexing**: Auto-categorizes by topic
- **Privacy-aware**: Never indexes credential file contents
- **Incremental updates**: Only re-scans changed files
- **Hands-off operation**: Runs daily via launchd

## Quick Start

```bash
# Install (one-time setup)
cd ~/Desktop/unified-belief-system/file-indexer
chmod +x install.sh
./install.sh

# Manual run
python3 indexer.py

# Quick scan (modified files only)
python3 indexer.py --quick

# Include Mac Mini
python3 indexer.py --remote mac-mini

# Custom output location
python3 indexer.py --output ~/my-knowledge.yaml
```

## Output Structure

The generated YAML includes:

```yaml
meta:
  version: "1.0"
  generated_at: "2025-12-07T..."
  sources: [...]        # Where we scanned
  stats:
    total_files: 27576
    documents: 1234
    code_files: 8456

projects:              # Detected repos/projects
  - name: "my-project"
    path: "~/Projects/..."
    type: "python"
    key_files: [...]
    dependencies: [...]

documents:             # PDFs, markdown, notes
  research: [...]
  notes: [...]

code_patterns:         # Reusable code patterns detected
  - name: "API Client"
    language: "python"
    pattern: "..."

configs:               # Shell, git, editor configs
  shell: [...]
  git: [...]

sensitive_paths: [...]  # Listed but never indexed

recent_activity:       # What you've been working on
  last_7_days: {...}

semantic_index:        # Topic-based lookup
  topics:
    "machine learning": [file1, file2]
    "api": [file3, file4]
```

## Using with LLMs

### Load Full Context
```bash
# Copy to clipboard
cat ~/Dropbox/knowledge.yaml | pbcopy

# Then paste into Claude/ChatGPT with:
# "Here's my knowledge index. Help me find..."
```

### Load Partial Context
```python
import yaml

with open("~/knowledge.yaml") as f:
    knowledge = yaml.safe_load(f)

# Just projects
projects = knowledge["projects"]

# Just recent activity
recent = knowledge["recent_activity"]

# Search by topic
ml_files = knowledge["semantic_index"]["topics"].get("machine learning", [])
```

### Example Prompts

With the index loaded, you can ask:

- "What projects am I working on that involve API clients?"
- "Find all my notes about machine learning"
- "Which files have I modified this week?"
- "What dependencies does my unified-belief-system project use?"
- "Show me code patterns I've used for database access"

## Configuration

Edit `indexer.py` to customize:

```python
@dataclass
class IndexConfig:
    # Paths to scan
    scan_paths: List[str] = [
        "~/Desktop",
        "~/Documents",
        "~/Projects",
    ]

    # Cloud storage
    cloud_paths: Dict[str, str] = {
        "dropbox": "~/Dropbox",
        "box": "~/Box",
    }

    # Remote machines
    remote_hosts: Dict[str, dict] = {
        "mac-mini": {
            "host": "mac-mini.local",
            "user": "chris",
            "paths": ["~/Documents", "~/Projects"],
        }
    }

    # File types to index
    code_extensions: Set[str] = {".py", ".js", ".ts", ...}
    doc_extensions: Set[str] = {".md", ".pdf", ".docx", ...}

    # Skip patterns
    skip_patterns: List[str] = ["node_modules", ".git", ...]
```

## Automation

The indexer runs automatically via macOS launchd:

- **Schedule**: Daily at 6:00 AM
- **Output**: `~/Dropbox/knowledge.yaml` (syncs to all devices)
- **Logs**: `~/Library/Logs/knowledge-indexer.log`

### Control Commands

```bash
# Check status
launchctl list | grep knowledge

# Stop automation
launchctl unload ~/Library/LaunchAgents/com.user.knowledge-indexer.plist

# Restart automation
launchctl load ~/Library/LaunchAgents/com.user.knowledge-indexer.plist

# View logs
tail -f ~/Library/Logs/knowledge-indexer.log
```

## Multi-Machine Setup

### On MacBook Pro (primary)
```bash
./install.sh
```

### On Mac Mini (secondary)
```bash
# Copy indexer files
scp -r file-indexer/ mac-mini.local:~/Desktop/unified-belief-system/

# SSH to mini and install
ssh mac-mini.local
cd ~/Desktop/unified-belief-system/file-indexer
./install.sh
```

### Sync Both Machines
```bash
# From MacBook Pro, pull Mac Mini's index and merge
./sync-from-remote.sh mac-mini.local
```

## Dependencies

Required:
- Python 3.8+
- PyYAML (`pip install pyyaml`)

Optional (for content extraction):
- PyPDF2 (`pip install PyPDF2`) - PDF text extraction
- python-docx (`pip install python-docx`) - Word doc extraction

## Privacy & Security

The indexer is designed with privacy in mind:

1. **Sensitive paths are listed but never read**
   - `.ssh/`, `.aws/`, credentials files
   - Only the path is recorded, never contents

2. **Local processing only**
   - Nothing is sent to external services
   - YAML stays on your machines

3. **Configurable exclusions**
   - Add patterns to `skip_patterns` or `sensitive_patterns`

## Troubleshooting

### "Permission denied" errors
```bash
# Grant Full Disk Access to Terminal
# System Preferences > Security > Privacy > Full Disk Access
```

### SSH to Mac Mini fails
```bash
# Enable Remote Login on Mac Mini
# System Preferences > Sharing > Remote Login

# Test connection
ssh mac-mini.local
```

### Index is too large
Adjust in `IndexConfig`:
```python
max_files_per_project: int = 50  # Reduce from 100
max_preview_chars: int = 200     # Reduce from 500
```
