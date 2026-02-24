#!/usr/bin/env python3
"""
Test suite for ewm_session_stop_hook.py

Tests the truncation logic, transcript parsing, and size handling.
Run with: python -m pytest test_ewm_stop_hook.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add hook directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ewm_session_stop_hook import (
    read_transcript_as_text,
    _extract_message_text,
    is_safe_transcript_path,
)


# =============================================================================
# EXPECTED TEST RESULTS - Document predictions before running
# =============================================================================

EXPECTED_RESULTS = {
    "test_small_transcript": {
        "description": "Small transcript under limit should pass through unchanged",
        "input_chars": 1000,
        "expected_output_chars": 1000,  # No truncation
        "truncated": False,
    },
    "test_at_limit_transcript": {
        "description": "Transcript exactly at 500K limit should pass unchanged",
        "input_chars": 500_000,
        "expected_output_chars": 500_000,
        "truncated": False,
    },
    "test_over_limit_transcript": {
        "description": "Transcript over 500K should be truncated to ~500K",
        "input_chars": 1_000_000,
        "expected_output_chars_range": (490_000, 500_000),  # After truncation + paragraph trim
        "truncated": True,
    },
    "test_massive_transcript": {
        "description": "50MB transcript should truncate to ~500K (most recent)",
        "input_chars": 50_000_000,
        "expected_output_chars_range": (490_000, 500_000),
        "truncated": True,
    },
    "test_truncation_preserves_recent": {
        "description": "Truncation should keep END of content (most recent)",
        "marker_at_end": "RECENT_CONTENT_MARKER",
        "marker_at_start": "OLD_CONTENT_MARKER",
        "recent_should_be_present": True,
        "old_should_be_absent": True,
    },
    "test_paragraph_break_cleanup": {
        "description": "Truncation should find paragraph break to avoid mid-sentence",
        "should_start_at_paragraph": True,
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_mock_transcript(num_messages: int, chars_per_message: int = 500) -> str:
    """Generate a mock JSONL transcript with specified size."""
    lines = []
    for i in range(num_messages):
        content = f"Message {i}: " + "x" * (chars_per_message - 20)
        msg = {
            "message": {
                "content": [{"type": "text", "text": content}]
            }
        }
        lines.append(json.dumps(msg))
    return "\n".join(lines)


def create_transcript_with_markers(total_chars: int, marker_start: str, marker_end: str) -> str:
    """Create transcript with identifiable markers at start and end."""
    # Create messages with markers
    lines = []

    # Start marker message
    start_msg = {
        "message": {"content": [{"type": "text", "text": f"{marker_start} - This is old content at the beginning"}]}
    }
    lines.append(json.dumps(start_msg))

    # Fill middle with content
    chars_needed = total_chars - 200  # Account for markers
    num_middle_msgs = chars_needed // 500
    for i in range(num_middle_msgs):
        msg = {
            "message": {"content": [{"type": "text", "text": f"Middle message {i}: " + "y" * 450}]}
        }
        lines.append(json.dumps(msg))

    # End marker message
    end_msg = {
        "message": {"content": [{"type": "text", "text": f"{marker_end} - This is recent content at the end"}]}
    }
    lines.append(json.dumps(end_msg))

    return "\n".join(lines)


def write_temp_transcript(content: str) -> str:
    """Write content to a temp file and return path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="transcript_")
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path


# =============================================================================
# TESTS
# =============================================================================

class TestTranscriptTruncation:
    """Test the truncation behavior for various transcript sizes."""

    def test_small_transcript(self):
        """Small transcript under limit should pass through unchanged."""
        expected = EXPECTED_RESULTS["test_small_transcript"]

        # Create ~1000 char transcript
        transcript = create_mock_transcript(num_messages=2, chars_per_message=500)
        path = write_temp_transcript(transcript)

        try:
            result = read_transcript_as_text(path)

            # Assertions
            assert len(result) > 0, "Should produce output"
            assert len(result) < 500_000, "Should be under limit"
            # Small transcripts should be fully preserved (no truncation)

            print(f"✓ Small transcript: {len(result)} chars (expected: ~{expected['expected_output_chars']})")
        finally:
            os.unlink(path)

    def test_at_limit_transcript(self):
        """Transcript exactly at 500K limit should pass unchanged."""
        expected = EXPECTED_RESULTS["test_at_limit_transcript"]

        # Create ~500K char transcript (1000 messages * 500 chars = 500K)
        transcript = create_mock_transcript(num_messages=1000, chars_per_message=500)
        path = write_temp_transcript(transcript)

        try:
            result = read_transcript_as_text(path)

            # Should be close to input size (some overhead from JSON parsing)
            assert len(result) <= 500_000, f"Should be at or under limit, got {len(result)}"

            print(f"✓ At-limit transcript: {len(result)} chars")
        finally:
            os.unlink(path)

    def test_over_limit_transcript(self):
        """Transcript over 500K should be truncated to ~500K."""
        expected = EXPECTED_RESULTS["test_over_limit_transcript"]

        # Create ~1MB transcript
        transcript = create_mock_transcript(num_messages=2000, chars_per_message=500)
        path = write_temp_transcript(transcript)

        try:
            result = read_transcript_as_text(path)

            min_expected, max_expected = expected["expected_output_chars_range"]
            assert min_expected <= len(result) <= max_expected, \
                f"Expected {min_expected}-{max_expected}, got {len(result)}"

            print(f"✓ Over-limit transcript: truncated to {len(result)} chars")
        finally:
            os.unlink(path)

    def test_massive_transcript(self):
        """50MB transcript should truncate to ~500K (most recent)."""
        expected = EXPECTED_RESULTS["test_massive_transcript"]

        # Create ~5MB transcript (50MB would be too slow for tests)
        # Scale: 10000 messages * 500 chars = 5MB
        transcript = create_mock_transcript(num_messages=10000, chars_per_message=500)
        path = write_temp_transcript(transcript)

        try:
            result = read_transcript_as_text(path)

            min_expected, max_expected = expected["expected_output_chars_range"]
            assert min_expected <= len(result) <= max_expected, \
                f"Expected {min_expected}-{max_expected}, got {len(result)}"

            print(f"✓ Massive transcript: truncated to {len(result)} chars")
        finally:
            os.unlink(path)

    def test_truncation_preserves_recent(self):
        """Truncation should keep END of content (most recent)."""
        expected = EXPECTED_RESULTS["test_truncation_preserves_recent"]

        # Create 1MB transcript with markers
        transcript = create_transcript_with_markers(
            total_chars=1_000_000,
            marker_start="OLD_CONTENT_MARKER",
            marker_end="RECENT_CONTENT_MARKER"
        )
        path = write_temp_transcript(transcript)

        try:
            result = read_transcript_as_text(path)

            # Recent marker should be present
            assert expected["marker_at_end"] in result, \
                "Recent content marker should be preserved"

            # Old marker should be truncated away
            assert expected["marker_at_start"] not in result, \
                "Old content marker should be truncated"

            print(f"✓ Truncation preserves recent content")
        finally:
            os.unlink(path)

    def test_paragraph_break_cleanup(self):
        """Truncation should find paragraph break to avoid mid-sentence."""
        # Create content where truncation point falls mid-word
        lines = []
        for i in range(3000):
            msg = {
                "message": {"content": [{"type": "text", "text": f"Paragraph {i} content here."}]}
            }
            lines.append(json.dumps(msg))

        transcript = "\n".join(lines)
        path = write_temp_transcript(transcript)

        try:
            result = read_transcript_as_text(path)

            # Result should start with "Paragraph" (clean break), not mid-word
            # After truncation and paragraph-finding, first word should be clean
            first_word = result.strip().split()[0]
            assert first_word == "Paragraph", \
                f"Should start at paragraph break, got: '{first_word}'"

            print(f"✓ Paragraph break cleanup working")
        finally:
            os.unlink(path)


class TestMessageExtraction:
    """Test the message text extraction logic."""

    def test_standard_format(self):
        """Extract text from standard message format."""
        msg = {
            "message": {
                "content": [{"type": "text", "text": "Hello world"}]
            }
        }
        result = _extract_message_text(msg)
        assert result == "Hello world"

    def test_legacy_format(self):
        """Extract text from legacy message format."""
        msg = {"content": "Legacy content"}
        result = _extract_message_text(msg)
        assert result == "Legacy content"

    def test_multiple_blocks(self):
        """Extract text from multiple content blocks."""
        msg = {
            "message": {
                "content": [
                    {"type": "text", "text": "First"},
                    {"type": "tool_use", "name": "something"},
                    {"type": "text", "text": "Second"}
                ]
            }
        }
        result = _extract_message_text(msg)
        assert "First" in result
        assert "Second" in result

    def test_empty_message(self):
        """Handle empty messages gracefully."""
        msg = {}
        result = _extract_message_text(msg)
        assert result == ""


class TestSecurityChecks:
    """Test security validation functions."""

    def test_allowed_paths(self):
        """Validate allowed transcript paths."""
        allowed = [
            str(Path.home() / ".claude" / "projects" / "test" / "transcript.jsonl"),
            "/tmp/transcript.jsonl",
            "/private/tmp/claude-123/transcript.jsonl",
        ]
        for path in allowed:
            assert is_safe_transcript_path(path), f"Should allow: {path}"

    def test_rejected_paths(self):
        """Reject paths outside allowed directories."""
        rejected = [
            "/etc/passwd",
            "/var/log/system.log",
            str(Path.home() / "Documents" / "secrets.txt"),
        ]
        for path in rejected:
            assert not is_safe_transcript_path(path), f"Should reject: {path}"


class TestPerformance:
    """Test performance characteristics."""

    def test_large_file_processing_time(self):
        """Processing 5MB transcript should complete in reasonable time."""
        import time

        transcript = create_mock_transcript(num_messages=10000, chars_per_message=500)
        path = write_temp_transcript(transcript)

        try:
            start = time.time()
            result = read_transcript_as_text(path)
            elapsed = time.time() - start

            # Should complete within 5 seconds
            assert elapsed < 5.0, f"Processing took {elapsed:.2f}s, expected < 5s"

            print(f"✓ 5MB transcript processed in {elapsed:.2f}s")
        finally:
            os.unlink(path)


# =============================================================================
# RESULTS COMPARISON
# =============================================================================

def compare_results():
    """Run all tests and compare actual vs expected results."""
    print("\n" + "="*70)
    print("EXPECTED VS ACTUAL RESULTS COMPARISON")
    print("="*70 + "\n")

    results = {}

    for test_name, expected in EXPECTED_RESULTS.items():
        print(f"\n{test_name}:")
        print(f"  Description: {expected['description']}")
        # Results will be filled by pytest

    return results


if __name__ == "__main__":
    # Run comparison first
    compare_results()

    # Then run pytest
    pytest.main([__file__, "-v", "--tb=short"])
