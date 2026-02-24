"""
Provenance Tracking Module
==========================

This module implements full provenance tracking for beliefs.

PROVENANCE = "Where did this belief come from?"

In a DARPA-grade system, every number must be traceable to its source.
This isn't just for debugging - it's for:
1. AUDIT: Regulatory/legal review of AI decisions
2. TRUST: Users can verify claims independently
3. DEBUGGING: Find where bad beliefs entered the system
4. LEARNING: Track which sources are reliable over time

Key Concepts:
    - Evidence: Atomic unit with source attribution
    - ProvenanceChain: Full history of a claim
    - ProvenanceGraph: All evidence and derivation relationships

References:
    - "Data Provenance: Principles and Applications" SIGMOD 2019
    - W3C PROV-O: The PROV Ontology
    - "Explaining AI Decisions" EU GDPR Article 22
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import hashlib
from collections import defaultdict

from .types import (
    Evidence,
    EvidenceSource,
    ExtractionMethod,
    VerificationStatus,
    Claim,
    ProvenanceChain,
)


# =============================================================================
# PROVENANCE GRAPH
# =============================================================================

class ProvenanceGraph:
    """
    Graph tracking all evidence and derivation relationships.

    Structure:
        - Evidence nodes: Atomic evidence with source attribution
        - Claim nodes: Beliefs derived from evidence
        - Derivation edges: How claims were derived from evidence/other claims

    Key Operations:
        - add_evidence(): Register new evidence
        - add_claim(): Register new claim with evidence
        - trace(): Get full provenance chain for a claim
        - audit_trail(): Human-readable audit report
    """

    def __init__(self, path: Optional[str] = None):
        # Evidence storage
        self.evidence: Dict[str, Evidence] = {}

        # Claim → Evidence relationships
        self.claim_evidence: Dict[str, List[str]] = defaultdict(list)

        # Claim → Parent Claims (for derived claims)
        self.claim_parents: Dict[str, List[str]] = defaultdict(list)

        # Update history
        self.update_history: Dict[str, List[Dict]] = defaultdict(list)

        # Source reliability tracking
        self.source_reliability: Dict[str, Dict] = {}

        # Persistence
        self.path = path
        if path and Path(path).exists():
            self._load()

    # -------------------------------------------------------------------------
    # Evidence Management
    # -------------------------------------------------------------------------

    def add_evidence(
        self,
        source_type: EvidenceSource,
        source_id: str,
        raw_text: str,
        extraction_method: ExtractionMethod = ExtractionMethod.EXACT,
        location: Optional[str] = None,
        reliability_override: Optional[float] = None,
        evidence_id: Optional[str] = None,
    ) -> Evidence:
        """
        Register new evidence in the graph.

        Args:
            source_type: Type of source (document, human, model, etc.)
            source_id: Identifier for the source
            raw_text: Actual evidence text
            extraction_method: How it was extracted from source
            location: Location within source (page, line, etc.)
            reliability_override: Override default source reliability
            evidence_id: Optional custom ID (auto-generated if not provided)

        Returns:
            Evidence object
        """
        # Generate ID if not provided
        if evidence_id is None:
            content_hash = hashlib.sha256(
                f"{source_id}:{raw_text}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
            evidence_id = f"ev_{content_hash}"

        evidence = Evidence(
            id=evidence_id,
            source_type=source_type,
            source_id=source_id,
            raw_text=raw_text,
            extraction_method=extraction_method,
            timestamp=datetime.now(),
            location=location,
            reliability_override=reliability_override,
        )

        self.evidence[evidence_id] = evidence

        # Update source reliability tracking
        self._track_source(source_type, source_id)

        self._save()
        return evidence

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """Get evidence by ID."""
        return self.evidence.get(evidence_id)

    def get_evidence_by_source(
        self,
        source_type: Optional[EvidenceSource] = None,
        source_id: Optional[str] = None,
    ) -> List[Evidence]:
        """Get all evidence from a specific source."""
        results = []
        for ev in self.evidence.values():
            if source_type and ev.source_type != source_type:
                continue
            if source_id and ev.source_id != source_id:
                continue
            results.append(ev)
        return results

    def verify_evidence(
        self,
        evidence_id: str,
        verified_true: bool,
        verifier_id: str,
        notes: Optional[str] = None,
    ) -> Evidence:
        """
        Mark evidence as verified.

        Note: Evidence is immutable, so this creates a new Evidence object
        with updated verification status.
        """
        old_evidence = self.evidence.get(evidence_id)
        if not old_evidence:
            raise ValueError(f"Evidence {evidence_id} not found")

        # Create new evidence with verification
        new_evidence = Evidence(
            id=old_evidence.id,
            source_type=old_evidence.source_type,
            source_id=old_evidence.source_id,
            raw_text=old_evidence.raw_text,
            extraction_method=old_evidence.extraction_method,
            timestamp=old_evidence.timestamp,
            location=old_evidence.location,
            reliability_override=old_evidence.reliability_override,
            verification_status=(
                VerificationStatus.VERIFIED_TRUE if verified_true
                else VerificationStatus.VERIFIED_FALSE
            ),
            verifier_id=verifier_id,
            verification_timestamp=datetime.now(),
            verification_notes=notes,
            content_hash=old_evidence.content_hash,
        )

        self.evidence[evidence_id] = new_evidence
        self._save()
        return new_evidence

    # -------------------------------------------------------------------------
    # Claim-Evidence Relationships
    # -------------------------------------------------------------------------

    def link_evidence(self, claim_id: str, evidence_id: str):
        """Link evidence to a claim."""
        if evidence_id not in self.evidence:
            raise ValueError(f"Evidence {evidence_id} not found")

        if evidence_id not in self.claim_evidence[claim_id]:
            self.claim_evidence[claim_id].append(evidence_id)

            # Record update
            self._record_update(claim_id, "link_evidence", {
                "evidence_id": evidence_id,
            })

        self._save()

    def link_parent_claim(
        self,
        child_claim_id: str,
        parent_claim_id: str,
        derivation_method: str = "inference",
    ):
        """Link a derived claim to its parent claim."""
        if parent_claim_id not in self.claim_parents[child_claim_id]:
            self.claim_parents[child_claim_id].append(parent_claim_id)

            self._record_update(child_claim_id, "link_parent", {
                "parent_claim_id": parent_claim_id,
                "derivation_method": derivation_method,
            })

        self._save()

    def get_claim_evidence(self, claim_id: str) -> List[Evidence]:
        """Get all evidence supporting a claim."""
        evidence_ids = self.claim_evidence.get(claim_id, [])
        return [self.evidence[eid] for eid in evidence_ids if eid in self.evidence]

    # -------------------------------------------------------------------------
    # Provenance Tracing
    # -------------------------------------------------------------------------

    def trace(self, claim_id: str, max_depth: int = 10) -> ProvenanceChain:
        """
        Get full provenance chain for a claim.

        Traces back through all evidence and parent claims.
        """
        direct_evidence = self.get_claim_evidence(claim_id)
        parent_claims = self.claim_parents.get(claim_id, [])

        # Determine derivation method
        derivation_method = None
        for update in self.update_history.get(claim_id, []):
            if update.get("action") == "link_parent":
                derivation_method = update.get("details", {}).get("derivation_method")
                break

        return ProvenanceChain(
            claim_id=claim_id,
            direct_evidence=direct_evidence,
            parent_claims=parent_claims,
            derivation_method=derivation_method,
            update_history=self.update_history.get(claim_id, []),
        )

    def trace_recursive(
        self,
        claim_id: str,
        max_depth: int = 10,
        visited: Optional[Set[str]] = None,
    ) -> Dict:
        """
        Recursively trace provenance through parent claims.

        Returns nested structure showing full derivation tree.
        """
        if visited is None:
            visited = set()

        if claim_id in visited or max_depth <= 0:
            return {"claim_id": claim_id, "truncated": True}

        visited.add(claim_id)

        chain = self.trace(claim_id)

        result = {
            "claim_id": claim_id,
            "direct_evidence": [ev.to_dict() for ev in chain.direct_evidence],
            "derivation_method": chain.derivation_method,
            "parent_chains": [
                self.trace_recursive(parent_id, max_depth - 1, visited)
                for parent_id in chain.parent_claims
            ],
        }

        return result

    def audit_trail(self, claim_id: str) -> str:
        """
        Generate human-readable audit trail for a claim.

        Suitable for legal/regulatory review.
        """
        chain = self.trace(claim_id)
        recursive = self.trace_recursive(claim_id)

        lines = [
            "=" * 70,
            f"AUDIT TRAIL FOR CLAIM: {claim_id}",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 70,
        ]

        # Direct evidence
        if chain.direct_evidence:
            lines.append("\n📄 DIRECT EVIDENCE:")
            lines.append("-" * 50)
            for i, ev in enumerate(chain.direct_evidence, 1):
                lines.append(f"\n  [{i}] Evidence ID: {ev.id}")
                lines.append(f"      Source Type: {ev.source_type.value}")
                lines.append(f"      Source ID: {ev.source_id}")
                lines.append(f"      Timestamp: {ev.timestamp.isoformat()}")
                lines.append(f"      Location: {ev.location or 'N/A'}")
                lines.append(f"      Extraction: {ev.extraction_method.value}")
                lines.append(f"      Reliability: {ev.reliability:.2f}")

                # Text (truncated if long)
                text = ev.raw_text
                if len(text) > 200:
                    text = text[:200] + "..."
                lines.append(f"      Text: \"{text}\"")

                # Verification
                lines.append(f"      Verified: {ev.verification_status.value}")
                if ev.verifier_id:
                    lines.append(f"      Verifier: {ev.verifier_id}")
                    lines.append(f"      Verification Time: {ev.verification_timestamp}")
        else:
            lines.append("\n📄 DIRECT EVIDENCE: None")

        # Parent claims
        if chain.parent_claims:
            lines.append("\n\n🔗 DERIVED FROM CLAIMS:")
            lines.append("-" * 50)
            for parent_id in chain.parent_claims:
                lines.append(f"  → {parent_id}")
            if chain.derivation_method:
                lines.append(f"  Method: {chain.derivation_method}")
        else:
            lines.append("\n\n🔗 DERIVED FROM CLAIMS: None (root claim)")

        # Update history
        if chain.update_history:
            lines.append("\n\n📜 UPDATE HISTORY (last 10):")
            lines.append("-" * 50)
            for update in chain.update_history[-10:]:
                ts = update.get("timestamp", "N/A")
                action = update.get("action", "N/A")
                details = update.get("details", {})
                lines.append(f"  [{ts}] {action}")
                for k, v in details.items():
                    lines.append(f"      {k}: {v}")

        lines.append("\n" + "=" * 70)
        lines.append("END AUDIT TRAIL")
        lines.append("=" * 70)

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Source Reliability Tracking
    # -------------------------------------------------------------------------

    def _track_source(self, source_type: EvidenceSource, source_id: str):
        """Track a source for reliability analysis."""
        key = f"{source_type.value}:{source_id}"
        if key not in self.source_reliability:
            self.source_reliability[key] = {
                "source_type": source_type.value,
                "source_id": source_id,
                "evidence_count": 0,
                "verified_true": 0,
                "verified_false": 0,
                "first_seen": datetime.now().isoformat(),
            }
        self.source_reliability[key]["evidence_count"] += 1

    def update_source_reliability(self, source_type: EvidenceSource, source_id: str,
                                   verified_true: bool):
        """Update source reliability based on verification outcomes."""
        key = f"{source_type.value}:{source_id}"
        if key in self.source_reliability:
            if verified_true:
                self.source_reliability[key]["verified_true"] += 1
            else:
                self.source_reliability[key]["verified_false"] += 1
        self._save()

    def get_source_reliability(
        self,
        source_type: EvidenceSource,
        source_id: str,
    ) -> float:
        """Get computed reliability for a source."""
        key = f"{source_type.value}:{source_id}"
        if key not in self.source_reliability:
            return source_type.default_reliability

        stats = self.source_reliability[key]
        verified = stats["verified_true"] + stats["verified_false"]
        if verified == 0:
            return source_type.default_reliability

        # Bayesian estimate with prior
        prior = source_type.default_reliability
        alpha = stats["verified_true"] + prior * 2
        beta = stats["verified_false"] + (1 - prior) * 2
        return alpha / (alpha + beta)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _record_update(self, claim_id: str, action: str, details: Dict):
        """Record an update in history."""
        self.update_history[claim_id].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
        })

    def stats(self) -> Dict:
        """Get statistics about the provenance graph."""
        verified_count = sum(
            1 for ev in self.evidence.values()
            if ev.verification_status in {
                VerificationStatus.VERIFIED_TRUE,
                VerificationStatus.VERIFIED_FALSE
            }
        )

        return {
            "evidence_count": len(self.evidence),
            "verified_count": verified_count,
            "claim_count": len(self.claim_evidence),
            "source_count": len(self.source_reliability),
            "update_count": sum(len(h) for h in self.update_history.values()),
        }

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _save(self):
        """Save to disk if path is set."""
        if not self.path:
            return

        data = {
            "evidence": {k: v.to_dict() for k, v in self.evidence.items()},
            "claim_evidence": dict(self.claim_evidence),
            "claim_parents": dict(self.claim_parents),
            "update_history": dict(self.update_history),
            "source_reliability": self.source_reliability,
        }

        Path(self.path).write_text(json.dumps(data, indent=2))

    def _load(self):
        """Load from disk."""
        if not self.path or not Path(self.path).exists():
            return

        try:
            data = json.loads(Path(self.path).read_text())

            self.evidence = {
                k: Evidence.from_dict(v)
                for k, v in data.get("evidence", {}).items()
            }

            self.claim_evidence = defaultdict(list, data.get("claim_evidence", {}))
            self.claim_parents = defaultdict(list, data.get("claim_parents", {}))
            self.update_history = defaultdict(list, data.get("update_history", {}))
            self.source_reliability = data.get("source_reliability", {})

        except Exception as e:
            print(f"Warning: Could not load provenance from {self.path}: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_evidence(
    source_type: str,
    source_id: str,
    raw_text: str,
    extraction_method: str = "exact",
    location: Optional[str] = None,
) -> Evidence:
    """
    Convenience function to create evidence.

    Args:
        source_type: "document", "human_expert", "human_user", "model_output", etc.
        source_id: Identifier for the source
        raw_text: The evidence text
        extraction_method: "exact", "paraphrase", "inference"
        location: Location within source

    Returns:
        Evidence object
    """
    return Evidence(
        id=f"ev_{hashlib.sha256(f'{source_id}:{raw_text}'.encode()).hexdigest()[:12]}",
        source_type=EvidenceSource(source_type),
        source_id=source_id,
        raw_text=raw_text,
        extraction_method=ExtractionMethod(extraction_method),
        timestamp=datetime.now(),
        location=location,
    )


def trace_claim(graph: ProvenanceGraph, claim_id: str) -> str:
    """Get human-readable trace for a claim."""
    return graph.audit_trail(claim_id)


def audit_trail(graph: ProvenanceGraph, claim_id: str) -> str:
    """Alias for trace_claim."""
    return graph.audit_trail(claim_id)


# =============================================================================
# SELF-TEST
# =============================================================================

def self_test():
    """Test provenance module."""
    print("Testing verity/provenance.py...")

    # Create graph
    graph = ProvenanceGraph()

    # Test 1: Add evidence
    ev1 = graph.add_evidence(
        source_type=EvidenceSource.DOCUMENT,
        source_id="contract_2024_001",
        raw_text="Payment is due within 30 days of invoice receipt.",
        extraction_method=ExtractionMethod.EXACT,
        location="Section 4.2, Paragraph 1",
    )
    assert ev1.id.startswith("ev_")
    assert ev1.reliability == 0.9  # Document default
    print(f"  ✓ Evidence created: {ev1.id}")

    # Test 2: Add more evidence
    ev2 = graph.add_evidence(
        source_type=EvidenceSource.HUMAN_EXPERT,
        source_id="lawyer_jsmith",
        raw_text="Based on my review, the payment terms are standard.",
        extraction_method=ExtractionMethod.PARAPHRASE,
    )
    print(f"  ✓ Second evidence: {ev2.id}")

    # Test 3: Link evidence to claim
    claim_id = "claim_payment_30_days"
    graph.link_evidence(claim_id, ev1.id)
    graph.link_evidence(claim_id, ev2.id)

    evidence = graph.get_claim_evidence(claim_id)
    assert len(evidence) == 2
    print(f"  ✓ Linked evidence: {len(evidence)} pieces")

    # Test 4: Trace provenance
    chain = graph.trace(claim_id)
    assert len(chain.direct_evidence) == 2
    print(f"  ✓ Provenance chain: {len(chain.direct_evidence)} direct evidence")

    # Test 5: Derived claims
    derived_claim = "claim_contract_valid"
    graph.link_parent_claim(derived_claim, claim_id, derivation_method="inference")

    chain2 = graph.trace(derived_claim)
    assert claim_id in chain2.parent_claims
    assert chain2.derivation_method == "inference"
    print(f"  ✓ Derived claim linked: parent={chain2.parent_claims[0]}")

    # Test 6: Verify evidence
    verified = graph.verify_evidence(
        ev1.id,
        verified_true=True,
        verifier_id="auditor_001",
        notes="Confirmed against original contract PDF",
    )
    assert verified.verification_status == VerificationStatus.VERIFIED_TRUE
    print(f"  ✓ Evidence verified: {verified.verification_status.value}")

    # Test 7: Audit trail
    trail = graph.audit_trail(claim_id)
    assert "AUDIT TRAIL" in trail
    assert "Payment is due" in trail
    print(f"  ✓ Audit trail generated: {len(trail)} chars")

    # Test 8: Recursive trace
    recursive = graph.trace_recursive(derived_claim)
    assert recursive["claim_id"] == derived_claim
    assert len(recursive["parent_chains"]) == 1
    print(f"  ✓ Recursive trace: depth={len(recursive['parent_chains'])}")

    # Test 9: Source reliability
    reliability = graph.get_source_reliability(
        EvidenceSource.DOCUMENT,
        "contract_2024_001"
    )
    assert reliability > 0.8  # Should be high after verification
    print(f"  ✓ Source reliability: {reliability:.3f}")

    # Test 10: Stats
    stats = graph.stats()
    assert stats["evidence_count"] == 2
    assert stats["verified_count"] == 1
    print(f"  ✓ Stats: {stats['evidence_count']} evidence, "
          f"{stats['verified_count']} verified")

    # Test 11: Evidence by source
    doc_evidence = graph.get_evidence_by_source(source_type=EvidenceSource.DOCUMENT)
    assert len(doc_evidence) == 1
    print(f"  ✓ Filter by source: {len(doc_evidence)} document evidence")

    # Test 12: Convenience function
    ev3 = create_evidence(
        source_type="model_output",
        source_id="gpt4_v1",
        raw_text="The contract appears to be a standard service agreement.",
        extraction_method="inference",
    )
    assert ev3.source_type == EvidenceSource.MODEL_OUTPUT
    assert ev3.reliability == 0.6  # Model output default
    print(f"  ✓ Convenience create: reliability={ev3.reliability}")

    print("\nAll provenance tests passed! ✓")


if __name__ == "__main__":
    self_test()
