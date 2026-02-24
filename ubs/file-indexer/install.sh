#!/bin/bash
# Knowledge Indexer Installation Script
# Run this once to set up automated indexing

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_NAME="com.user.knowledge-indexer.plist"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs"

echo "=== Knowledge Indexer Installation ==="
echo ""

# Create directories
mkdir -p "$LAUNCHD_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$HOME/.cache/file-indexer"

# Install Python dependencies (optional, for PDF/DOCX support)
echo "Installing optional dependencies..."
pip3 install --user PyPDF2 python-docx pyyaml 2>/dev/null || {
    echo "  Note: Some dependencies couldn't be installed."
    echo "  PDF/DOCX extraction may be limited."
}

# Copy plist to LaunchAgents
echo "Installing launchd service..."
cp "$SCRIPT_DIR/$PLIST_NAME" "$LAUNCHD_DIR/"

# Update paths in plist to match actual location
sed -i '' "s|/Users/christopherbailey|$HOME|g" "$LAUNCHD_DIR/$PLIST_NAME"

# Make indexer executable
chmod +x "$SCRIPT_DIR/indexer.py"

# Unload if already loaded
launchctl unload "$LAUNCHD_DIR/$PLIST_NAME" 2>/dev/null || true

# Load the service
launchctl load "$LAUNCHD_DIR/$PLIST_NAME"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "The indexer will run:"
echo "  - Daily at 6:00 AM"
echo "  - Output: ~/Dropbox/knowledge.yaml"
echo "  - Logs: ~/Library/Logs/knowledge-indexer.log"
echo ""
echo "Manual commands:"
echo "  Run now:     python3 $SCRIPT_DIR/indexer.py"
echo "  Quick scan:  python3 $SCRIPT_DIR/indexer.py --quick"
echo "  With remote: python3 $SCRIPT_DIR/indexer.py --remote mac-mini"
echo ""
echo "Service control:"
echo "  Stop:   launchctl unload ~/Library/LaunchAgents/$PLIST_NAME"
echo "  Start:  launchctl load ~/Library/LaunchAgents/$PLIST_NAME"
echo "  Status: launchctl list | grep knowledge"
echo ""
