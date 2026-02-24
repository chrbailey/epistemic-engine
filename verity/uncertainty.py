"""
VERITY Uncertainty Module - Thin Wrapper
==========================================

This module provides backwards compatibility for VERITY's uncertainty functions.
Core epistemic math is now provided by the `belief_math` package.

VERITY-specific functionality retained:
- update_with_evidence(): Updates uncertainty with VERITY Evidence objects

All other functions are re-exported from belief_math.
"""

from __future__ import annotations

from typing import Tuple

# =============================================================================
# RE-EXPORT FROM belief_math
# =============================================================================
from belief_math import (
    DecomposedUncertainty,
    MassFunction,
    Uncertainty,
    # Core decomposition
    decompose_from_ensemble,
    decompose_from_clarifications,
    decompose_from_beta,
    decompose_from_mass_function,
    decompose_from_opinion,
    # Information gain
    expected_information_gain,
    should_investigate,
    # Confidence intervals
    credible_interval,
    epistemic_interval,
    # Combination
    combine_uncertainties,
)

# =============================================================================
# VERITY-SPECIFIC IMPORTS
# =============================================================================
from .types import Evidence


# =============================================================================
# VERITY-SPECIFIC: update_with_evidence
# =============================================================================

def update_with_evidence(
    current: DecomposedUncertainty,
    new_evidence: Evidence,
    evidence_supports: bool,
    evidence_strength: float = 1.0,
) -> DecomposedUncertainty:
    """
    Update uncertainty given new evidence.

    This is VERITY-specific because it uses the Evidence type with its
    reliability property.

    New evidence:
    - REDUCES epistemic uncertainty (we learned something)
    - Shifts mean toward evidence direction
    - Aleatoric stays roughly constant (or increases if conflicting)

    Args:
        current: Current uncertainty state
        new_evidence: New evidence object (VERITY Evidence type)
        evidence_supports: True if evidence supports the claim
        evidence_strength: Strength of evidence (0-1)

    Returns:
        Updated DecomposedUncertainty
    """
    reliability = new_evidence.reliability * evidence_strength

    # Direction of update
    direction = 1 if evidence_supports else -1

    # Update mean (Bayesian-like)
    # Stronger evidence, higher reliability -> larger shift
    shift = direction * reliability * 0.1 * (1 - current.mean if direction > 0 else current.mean)
    new_mean = max(0.001, min(0.999, current.mean + shift))

    # Reduce epistemic uncertainty (we learned something)
    # Reduction proportional to reliability and inverse of current observations
    epistemic_reduction = reliability * 0.1 / (1 + current.n_observations * 0.1)
    new_epistemic = max(0.001, current.epistemic_variance * (1 - epistemic_reduction))

    # Aleatoric stays roughly constant
    # (Might increase slightly if evidence is conflicting)
    conflict_signal = abs(new_mean - 0.5) < abs(current.mean - 0.5)
    if conflict_signal:
        # Evidence pushed us toward 0.5 = possibly conflicting
        new_aleatoric = current.aleatoric_variance * 1.05
    else:
        new_aleatoric = current.aleatoric_variance

    return DecomposedUncertainty(
        mean=float(new_mean),
        epistemic_variance=float(new_epistemic),
        aleatoric_variance=float(new_aleatoric),
        n_observations=current.n_observations + 1,
        evidence_ids=current.evidence_ids + [new_evidence.id],
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # From belief_math
    "DecomposedUncertainty",
    "MassFunction",
    "Uncertainty",
    "decompose_from_ensemble",
    "decompose_from_clarifications",
    "decompose_from_beta",
    "decompose_from_mass_function",
    "decompose_from_opinion",
    "expected_information_gain",
    "should_investigate",
    "credible_interval",
    "epistemic_interval",
    "combine_uncertainties",
    # VERITY-specific
    "update_with_evidence",
]
