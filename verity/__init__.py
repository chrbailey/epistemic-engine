"""
VERITY: Verifiable Evidence-based Reasoning with Interpretable Trust Yields
============================================================================

A DARPA-grade belief system combining:
- Dempster-Shafer evidence theory for uncertainty ranges
- Decomposed uncertainty (epistemic vs aleatoric)
- Circular belief propagation with loop correction
- Full provenance tracking
- Conformal prediction for calibrated guarantees

This is not textbook Bayesian updating. This is defensible IP.

NOTE: Core epistemic math (Dempster-Shafer, Subjective Logic, uncertainty
decomposition, and calibration) is now provided by the shared `belief_math`
package. VERITY re-exports these for backwards compatibility, but new code
should import directly from `belief_math`.

See: /Volumes/OWC drive/Dev/belief-math/
"""

import warnings

# =============================================================================
# RE-EXPORT FROM belief_math (source of truth for epistemic math)
# =============================================================================
from belief_math import (
    # Core types
    MassFunction,
    Uncertainty,
    DecomposedUncertainty,
    ConflictResult,
    # D-S Creation
    create_mass_function,
    create_vacuous,
    create_certain,
    create_bayesian,
    # D-S Combination
    dempster_combine,
    combine_multiple,
    # D-S Belief functions
    belief,
    plausibility,
    uncertainty_interval,
    ignorance,
    # D-S Pignistic
    pignistic_probability,
    pignistic_distribution,
    # D-S Operations
    discount,
    condition,
    specificity,
    nonspecificity,
    strife,
    # D-S Utilities
    is_consistent,
    summarize,
    # D-S Exceptions
    ConflictingEvidenceError,
    InvalidMassFunctionError,
    # Subjective Logic Fusion
    cumulative_fuse,
    averaging_fuse,
    # Subjective Logic Trust
    trust_discount,
    trust_chain,
    # Subjective Logic Deduction
    deduce,
    # Subjective Logic Complement
    opinion_complement,
    uncertainty_maximized,
    # Subjective Logic Probability
    opinion_to_probability,
    probability_to_opinion,
    # Subjective Logic Blending
    blend_uncertainty,
    # Decomposition
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
    # Calibration metrics
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    brier_skill_score,
    brier_decomposition,
    # Recalibration
    recalibrate,
    temperature_scale,
    apply_temperature,
    # Calibration diagnostics
    reliability_diagram,
    calibration_summary,
)

# =============================================================================
# VERITY-SPECIFIC TYPES (not in belief_math)
# =============================================================================
from .types import (
    Evidence,
    EvidenceSource,
    ExtractionMethod,
    VerificationStatus,
    Claim,
    ProvenanceChain,
)

# =============================================================================
# VERITY-SPECIFIC MODULES
# =============================================================================

# Uncertainty module - keep update_with_evidence (VERITY-specific, uses Evidence type)
from .uncertainty import (
    update_with_evidence,
)

# Provenance tracking (VERITY-specific)
from .provenance import (
    ProvenanceGraph,
    create_evidence,
    trace_claim,
    audit_trail,
)

# Belief propagation networks (VERITY-specific)
from .belief_propagation import (
    BeliefNetwork,
    Edge,
    EdgeType,
    Message,
    PropagationState,
    PropagationStatus,
    PropagationResult,
    compare_lbp_vs_circular,
)

__version__ = "0.2.0"  # Updated: now uses belief_math as backend
__author__ = "Christopher Bailey"

__all__ = [
    # ==========================================================================
    # Types (from belief_math)
    # ==========================================================================
    "MassFunction",
    "Uncertainty",
    "DecomposedUncertainty",
    "ConflictResult",
    # ==========================================================================
    # Dempster-Shafer (from belief_math)
    # ==========================================================================
    # Creation
    "create_mass_function",
    "create_vacuous",
    "create_certain",
    "create_bayesian",
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
    # ==========================================================================
    # Subjective Logic (from belief_math)
    # ==========================================================================
    "cumulative_fuse",
    "averaging_fuse",
    "trust_discount",
    "trust_chain",
    "deduce",
    "opinion_complement",
    "uncertainty_maximized",
    "opinion_to_probability",
    "probability_to_opinion",
    "blend_uncertainty",
    # ==========================================================================
    # Decomposition (from belief_math)
    # ==========================================================================
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
    # ==========================================================================
    # Calibration (from belief_math)
    # ==========================================================================
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "brier_skill_score",
    "brier_decomposition",
    "recalibrate",
    "temperature_scale",
    "apply_temperature",
    "reliability_diagram",
    "calibration_summary",
    # ==========================================================================
    # VERITY-SPECIFIC Types
    # ==========================================================================
    "Evidence",
    "EvidenceSource",
    "ExtractionMethod",
    "VerificationStatus",
    "Claim",
    "ProvenanceChain",
    # ==========================================================================
    # VERITY-SPECIFIC Uncertainty
    # ==========================================================================
    "update_with_evidence",
    # ==========================================================================
    # VERITY-SPECIFIC Provenance
    # ==========================================================================
    "ProvenanceGraph",
    "create_evidence",
    "trace_claim",
    "audit_trail",
    # ==========================================================================
    # VERITY-SPECIFIC Belief Propagation
    # ==========================================================================
    "BeliefNetwork",
    "Edge",
    "EdgeType",
    "Message",
    "PropagationState",
    "PropagationStatus",
    "PropagationResult",
    "compare_lbp_vs_circular",
]
