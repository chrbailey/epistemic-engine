"""
Truth Validator - Catch LLM Drift
=================================

Integrated from your truth-validator skill. Flags claims that:
- Need record citation in legal context
- Are inferred vs exact from source
- Contain capability verbs not in record
- Make numeric claims without source

This module provides validation functions that can be used with the Guardian
system to catch LLM hallucinations before they cause problems.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class FactType(Enum):
    """Classification of factual claims."""
    EXACT = "FACT_EXACT"              # Verbatim or near-verbatim from record
    PARAPHRASE = "FACT_PARAPHRASE"    # Reasonable rephrasing
    INFERRED = "FACT_INFERRED"        # Added characterization not in record
    UNKNOWN = "UNKNOWN"               # Can't determine


class ValidationMode(Enum):
    """Operating modes with different strictness."""
    STRICT_CLOSED_RECORD = "strict"   # Legal briefs - no inference allowed
    NORMAL_LEGAL = "normal_legal"     # Background - paraphrase OK
    GENERAL = "general"               # Non-legal - more permissive


class ValidationStatus(Enum):
    """Result of validation check."""
    PASS = "PASS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAIL = "FAIL"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validating a claim or text."""
    status: str                       # ValidationStatus value
    fact_type: str                    # FactType value
    flags: List[str] = field(default_factory=list)
    explanation: str = ""
    suggested_fix: str = ""
    confidence: float = 0.0


@dataclass
class RecordContext:
    """Context from source record for validation."""
    text: str                         # The source record text
    capability_verbs: Set[str] = field(default_factory=set)  # Allowed capability verbs
    numeric_values: Set[str] = field(default_factory=set)    # Numbers in record
    entity_names: Set[str] = field(default_factory=set)      # Named entities


# =============================================================================
# VALIDATORS
# =============================================================================

class TruthValidator:
    """
    Validates claims against source records and flags potential drift.

    Usage:
        validator = TruthValidator(mode=ValidationMode.STRICT_CLOSED_RECORD)
        result = validator.validate_claim(claim_text, record_context)
    """

    # Capability verbs that indicate what something can/does do
    DEFAULT_CAPABILITY_VERBS = {
        'access', 'display', 'reproduce', 'copy', 'store', 'transmit',
        'process', 'analyze', 'generate', 'create', 'modify', 'delete',
        'extract', 'transform', 'enable', 'allow', 'permit', 'support'
    }

    # Words that suggest inference rather than fact
    INFERENCE_MARKERS = {
        'likely', 'probably', 'possibly', 'suggests', 'implies', 'indicates',
        'appears', 'seems', 'might', 'could', 'may', 'potentially',
        'essentially', 'effectively', 'virtually', 'arguably'
    }

    # Commercial/capability characterizations that need record support
    SENSITIVE_CHARACTERIZATIONS = {
        'commercial', 'profitable', 'successful', 'leading', 'innovative',
        'revolutionary', 'unique', 'superior', 'best', 'only', 'first',
        'largest', 'fastest', 'most', 'primary', 'main', 'key'
    }

    def __init__(self, mode: ValidationMode = ValidationMode.GENERAL):
        self.mode = mode

    def set_mode(self, mode: ValidationMode):
        """Change validation mode."""
        self.mode = mode

    def validate_claim(self, claim: str, record: Optional[RecordContext] = None) -> ValidationResult:
        """
        Validate a single claim.

        Args:
            claim: The text to validate
            record: Optional source record context

        Returns:
            ValidationResult with status and flags
        """
        flags = []
        fact_type = FactType.UNKNOWN
        status = ValidationStatus.PASS

        # Check for inference markers
        inference_markers_found = self._find_inference_markers(claim)
        if inference_markers_found:
            flags.append(f"INFERENCE_MARKERS: {inference_markers_found}")
            fact_type = FactType.INFERRED

        # Check for sensitive characterizations
        sensitive_found = self._find_sensitive_characterizations(claim)
        if sensitive_found:
            flags.append(f"SENSITIVE_CHARACTERIZATION: {sensitive_found}")
            if self.mode == ValidationMode.STRICT_CLOSED_RECORD:
                status = ValidationStatus.REVIEW_REQUIRED

        # Check for numeric claims
        numerics = self._find_numerics(claim)
        if numerics:
            if record and not self._numerics_in_record(numerics, record):
                flags.append(f"NEW_NUMERICS: {numerics}")
                if self.mode == ValidationMode.STRICT_CLOSED_RECORD:
                    status = ValidationStatus.FAIL

        # Check capability verbs against record
        if record:
            capability_issues = self._check_capability_verbs(claim, record)
            if capability_issues:
                flags.extend(capability_issues)
                if self.mode == ValidationMode.STRICT_CLOSED_RECORD:
                    status = ValidationStatus.FAIL

        # Check if claim matches record (if provided)
        if record:
            fact_type, match_explanation = self._classify_fact_type(claim, record)
            if fact_type == FactType.INFERRED and self.mode == ValidationMode.STRICT_CLOSED_RECORD:
                status = ValidationStatus.FAIL
                flags.append(f"FACT_INFERRED_IN_STRICT_MODE")

        # Build explanation
        explanation = self._build_explanation(flags, fact_type, status)

        # Build suggested fix
        suggested_fix = self._suggest_fix(flags, claim)

        return ValidationResult(
            status=status.value,
            fact_type=fact_type.value,
            flags=flags,
            explanation=explanation,
            suggested_fix=suggested_fix,
            confidence=self._calculate_confidence(flags, record)
        )

    def validate_text(self, text: str, record: Optional[RecordContext] = None) -> List[ValidationResult]:
        """
        Validate a longer text by splitting into claims.

        Returns a list of results, one per detected claim.
        """
        claims = self._split_into_claims(text)
        return [self.validate_claim(claim, record) for claim in claims]

    # -------------------------------------------------------------------------
    # Detection Helpers
    # -------------------------------------------------------------------------

    def _find_inference_markers(self, text: str) -> List[str]:
        """Find words that suggest inference."""
        text_lower = text.lower()
        return [m for m in self.INFERENCE_MARKERS if m in text_lower]

    def _find_sensitive_characterizations(self, text: str) -> List[str]:
        """Find characterizations that need record support."""
        text_lower = text.lower()
        return [c for c in self.SENSITIVE_CHARACTERIZATIONS if c in text_lower]

    def _find_numerics(self, text: str) -> List[str]:
        """Find numeric values in text."""
        # Match numbers with optional units
        pattern = r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent|million|billion|thousand|dollars?|years?|months?|days?|hours?)?\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [f"{m[0]} {m[1]}".strip() for m in matches if m[0]]

    def _numerics_in_record(self, numerics: List[str], record: RecordContext) -> bool:
        """Check if numerics appear in record."""
        for num in numerics:
            # Extract just the number part
            num_value = re.sub(r'[^\d.]', '', num.split()[0])
            if num_value not in record.text and num_value not in record.numeric_values:
                return False
        return True

    def _check_capability_verbs(self, claim: str, record: RecordContext) -> List[str]:
        """Check if capability verbs are supported by record."""
        issues = []
        claim_lower = claim.lower()

        # Find capability verbs in claim
        for verb in self.DEFAULT_CAPABILITY_VERBS:
            if verb in claim_lower:
                # Check if verb is in record's allowed list or text
                if verb not in record.capability_verbs and verb not in record.text.lower():
                    issues.append(f"CAPABILITY_VERB_NOT_IN_RECORD: '{verb}'")

        return issues

    def _classify_fact_type(self, claim: str, record: RecordContext) -> Tuple[FactType, str]:
        """
        Classify whether claim is exact, paraphrase, or inferred.

        This is a simplified version - a full implementation would use
        semantic similarity.
        """
        claim_lower = claim.lower().strip()
        record_lower = record.text.lower()

        # Check for exact match (substring)
        if claim_lower in record_lower or claim_lower[:50] in record_lower:
            return FactType.EXACT, "Claim appears verbatim in record"

        # Check for high word overlap (simple paraphrase detection)
        claim_words = set(claim_lower.split())
        record_words = set(record_lower.split())
        overlap = len(claim_words & record_words) / max(len(claim_words), 1)

        if overlap > 0.7:
            return FactType.PARAPHRASE, f"High word overlap ({overlap:.0%})"
        elif overlap > 0.4:
            return FactType.PARAPHRASE, f"Moderate word overlap ({overlap:.0%})"
        else:
            return FactType.INFERRED, f"Low word overlap ({overlap:.0%}) - may be inference"

    def _split_into_claims(self, text: str) -> List[str]:
        """Split text into individual claims/sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _build_explanation(self, flags: List[str], fact_type: FactType,
                          status: ValidationStatus) -> str:
        """Build human-readable explanation."""
        parts = []

        parts.append(f"Classification: {fact_type.value}")
        parts.append(f"Status: {status.value}")

        if flags:
            parts.append(f"Issues found: {len(flags)}")
            for flag in flags[:3]:  # Limit to first 3
                parts.append(f"  - {flag}")

        return "\n".join(parts)

    def _suggest_fix(self, flags: List[str], claim: str) -> str:
        """Suggest how to fix flagged issues."""
        suggestions = []

        for flag in flags:
            if "INFERENCE_MARKERS" in flag:
                suggestions.append("Remove hedging language or add citation")
            elif "SENSITIVE_CHARACTERIZATION" in flag:
                suggestions.append("Add record citation or use 'record does not specify' template")
            elif "NEW_NUMERICS" in flag:
                suggestions.append("Remove numeric or add source citation")
            elif "CAPABILITY_VERB" in flag:
                suggestions.append("Use only capability verbs from record")
            elif "FACT_INFERRED" in flag:
                suggestions.append("Rephrase to match record language or mark as inference")

        return "; ".join(suggestions) if suggestions else ""

    def _calculate_confidence(self, flags: List[str], record: Optional[RecordContext]) -> float:
        """Calculate confidence in validation result."""
        confidence = 0.5  # Base

        if record:
            confidence += 0.3  # Having record improves confidence

        # More flags = more certain about problems
        if flags:
            confidence += min(0.2, len(flags) * 0.05)

        return min(1.0, confidence)


# =============================================================================
# INTEGRATION WITH GUARDIAN
# =============================================================================

def create_validation_rules_for_router():
    """
    Create context rules for truth validation.

    These can be added to ContextRouter to trigger validation.
    """
    from context_router import ContextRule

    rules = [
        ContextRule(
            name="legal_fact_validation",
            domain="legal",
            description="Validate factual claims in legal context",
            response_level=4,  # INTERRUPT
            significance_weight=1.0,
            trigger_conditions={
                "content_type": "factual_claim",
                "source": "llm"
            },
            response_template="[REQUIRES VALIDATION AGAINST RECORD]"
        ),
        ContextRule(
            name="capability_claim_validation",
            domain="*",  # All domains
            description="Validate capability claims",
            response_level=3,  # ALERT
            significance_weight=0.8,
            trigger_conditions={
                "claim_contains": ["can", "able to", "capability", "enables"]
            },
            response_template="Verify capability against source"
        )
    ]

    return rules


# =============================================================================
# TESTS
# =============================================================================

def self_test():
    """Test truth validator."""
    print("Testing truth_validator.py...")

    validator = TruthValidator(mode=ValidationMode.STRICT_CLOSED_RECORD)

    # Test inference detection
    result = validator.validate_claim("The system likely processes data quickly")
    assert "INFERENCE_MARKERS" in str(result.flags)
    print("  ✓ Inference marker detection works")

    # Test sensitive characterization
    result = validator.validate_claim("This is the most innovative solution")
    assert "SENSITIVE_CHARACTERIZATION" in str(result.flags)
    print("  ✓ Sensitive characterization detection works")

    # Test numeric detection
    result = validator.validate_claim("Revenue was $50 million in 2023")
    numerics = validator._find_numerics("Revenue was $50 million in 2023")
    assert len(numerics) > 0
    print(f"  ✓ Numeric detection works: found {numerics}")

    # Test with record context
    record = RecordContext(
        text="The software can access and display files. Revenue was $10 million.",
        capability_verbs={'access', 'display'},
        numeric_values={'10'}
    )

    # Claim matching record
    result = validator.validate_claim("The software can access files", record)
    assert result.status != ValidationStatus.FAIL.value
    print("  ✓ Record-matching claim passes")

    # Claim with new capability verb
    result = validator.validate_claim("The software can transform files", record)
    assert "CAPABILITY_VERB_NOT_IN_RECORD" in str(result.flags)
    print("  ✓ New capability verb flagged")

    # Claim with wrong numeric
    validator_strict = TruthValidator(mode=ValidationMode.STRICT_CLOSED_RECORD)
    result = validator_strict.validate_claim("Revenue was $50 million", record)
    assert result.status == ValidationStatus.FAIL.value or "NEW_NUMERICS" in str(result.flags)
    print("  ✓ Wrong numeric flagged in strict mode")

    # Test mode switching
    validator.set_mode(ValidationMode.GENERAL)
    result = validator.validate_claim("This is probably the best approach")
    assert result.status != ValidationStatus.FAIL.value  # More permissive in general mode
    print("  ✓ Mode switching works")

    print("All truth_validator tests passed! ✓")


if __name__ == "__main__":
    self_test()
