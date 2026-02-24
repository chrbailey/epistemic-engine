"""
VERITY Dempster-Shafer Module - Thin Wrapper
==============================================

This module provides backwards compatibility for VERITY's Dempster-Shafer functions.
Core epistemic math is now provided by the `belief_math` package.

VERITY-specific functionality retained:
- from_likelihood_ratio(): Creates mass function from likelihood ratio evidence

All other functions are re-exported from belief_math.

NOTE: New code should import directly from belief_math.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

# =============================================================================
# RE-EXPORT FROM belief_math
# =============================================================================
from belief_math import (
    # Types
    MassFunction,
    ConflictResult,
    # Creation
    create_mass_function,
    create_vacuous,
    create_certain,
    create_bayesian,
    # Combination
    dempster_combine,
    combine_multiple,
    # Belief functions
    belief,
    plausibility,
    uncertainty_interval,
    ignorance,
    # Pignistic
    pignistic_probability,
    pignistic_distribution,
    # Operations
    discount,
    condition,
    specificity,
    nonspecificity,
    strife,
    # Utilities
    is_consistent,
    summarize,
    # Exceptions
    ConflictingEvidenceError,
    InvalidMassFunctionError,
)


# =============================================================================
# VERITY-SPECIFIC: from_likelihood_ratio
# =============================================================================

def from_likelihood_ratio(
    frame: Set[str],
    hypothesis: str,
    likelihood_ratio: float,
    prior_mass: Optional[MassFunction] = None,
) -> MassFunction:
    """
    Create mass function from likelihood ratio evidence.

    This bridges Bayesian and D-S reasoning.

    Args:
        frame: Frame of discernment
        hypothesis: Hypothesis being supported/opposed
        likelihood_ratio: LR > 1 supports hypothesis, LR < 1 opposes
        prior_mass: Prior mass function (default: vacuous)

    Returns:
        Updated mass function

    Example:
        # Evidence with LR=10 strongly supports hypothesis
        mf = from_likelihood_ratio(
            frame={"true", "false"},
            hypothesis="true",
            likelihood_ratio=10.0,
        )
        # BetP(true) will be around 0.9
    """
    if hypothesis not in frame:
        raise ValueError(f"{hypothesis} not in frame {frame}")

    frame_frozen = frozenset(frame)

    if prior_mass is None:
        prior_mass = create_vacuous(frame)

    # Convert LR to mass assignment
    # Mass on hypothesis = LR / (1 + LR) when LR > 1
    # Otherwise, mass on complement
    if likelihood_ratio >= 1:
        support = likelihood_ratio / (1 + likelihood_ratio)
        focal = frozenset({hypothesis})
    else:
        support = 1 / (1 + likelihood_ratio)
        focal = frame_frozen - frozenset({hypothesis})

    # Remaining mass goes to frame (uncertainty)
    evidence_mass = create_mass_function(
        frame,
        {focal: support, "FRAME": 1 - support}
    )

    # Combine with prior
    combined, _ = dempster_combine(prior_mass, evidence_mass)
    return combined


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Types
    "MassFunction",
    "ConflictResult",
    # Creation
    "create_mass_function",
    "create_vacuous",
    "create_certain",
    "create_bayesian",
    "from_likelihood_ratio",  # VERITY-specific
    # Combination
    "dempster_combine",
    "combine_multiple",
    # Belief functions
    "belief",
    "plausibility",
    "uncertainty_interval",
    "ignorance",
    # Pignistic
    "pignistic_probability",
    "pignistic_distribution",
    # Operations
    "discount",
    "condition",
    "specificity",
    "nonspecificity",
    "strife",
    # Utilities
    "is_consistent",
    "summarize",
    # Exceptions
    "ConflictingEvidenceError",
    "InvalidMassFunctionError",
]
