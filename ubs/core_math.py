"""
Core Mathematical Functions for Belief System
==============================================

Pure functions, no state. These are the building blocks.

All math is standard Bayesian - likelihood ratios, entropy, Beta distributions.
"""

import math
from typing import Tuple


# =============================================================================
# ENTROPY
# =============================================================================

def entropy(p: float) -> float:
    """
    Binary Shannon entropy.

    H(p) = -p·log₂(p) - (1-p)·log₂(1-p)

    Returns:
        0.0 at p=0 or p=1 (certainty)
        1.0 at p=0.5 (maximum uncertainty)
    """
    if p <= 0.01 or p >= 0.99:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def beta_entropy(alpha: float, beta: float) -> float:
    """
    Entropy approximation for Beta distribution.
    Uses the mean as point estimate.
    """
    if alpha + beta == 0:
        return 1.0
    p = alpha / (alpha + beta)
    return entropy(p)


# =============================================================================
# BAYESIAN UPDATES
# =============================================================================

def bayes_lr(prior: float, lr: float, bounds: Tuple[float, float] = (0.01, 0.99)) -> float:
    """
    Bayesian update via likelihood ratio.

    posterior_odds = prior_odds × LR

    Args:
        prior: P(H) before evidence
        lr: Likelihood ratio P(E|H)/P(E|¬H)
            - LR > 1: evidence supports H
            - LR < 1: evidence against H
            - LR = 1: uninformative
        bounds: (min, max) to prevent extreme values

    Returns:
        P(H|E) bounded by bounds
    """
    if lr <= 0:
        raise ValueError("Likelihood ratio must be positive")

    # Clamp prior to avoid division by zero
    prior = max(bounds[0], min(bounds[1], prior))

    odds = prior / (1 - prior)
    posterior_odds = odds * lr
    posterior = posterior_odds / (1 + posterior_odds)

    return max(bounds[0], min(bounds[1], posterior))


def bayes_beta(alpha: float, beta: float, supports: bool,
               strength: float = 1.0, competency: float = 1.0) -> Tuple[float, float]:
    """
    Bayesian update for Beta distribution parameters.

    This is the conjugate update: add pseudo-counts.

    Args:
        alpha, beta: Current Beta parameters
        supports: True if evidence supports the hypothesis
        strength: How much evidence (pseudo-count to add)
        competency: Source reliability [0, 1]. 1.0 = perfect source

    Returns:
        (new_alpha, new_beta)
    """
    effective = strength * competency
    noise = strength * (1 - competency)

    if supports:
        return (alpha + effective, beta + noise)
    else:
        return (alpha + noise, beta + effective)


# =============================================================================
# CORRELATION TO LIKELIHOOD RATIO
# =============================================================================

def correlation_to_lr(base_lr: float, correlation: float, damping: float = 0.5) -> float:
    """
    Convert correlation to effective likelihood ratio for propagation.

    LR_effective = base_lr^(correlation × damping)

    Args:
        base_lr: The LR from direct evidence
        correlation: [-1, 1] relationship strength
            - +1: perfectly correlated (A true → B likely true)
            - -1: anti-correlated (A true → B likely false)
            - 0: independent (no effect)
        damping: [0, 1] how much to attenuate propagation

    Returns:
        Effective LR for the related claim. Always positive.

    Examples:
        correlation_to_lr(10, 0.8, 0.5) → 10^0.4 ≈ 2.51 (moderate increase)
        correlation_to_lr(10, -0.8, 0.5) → 10^-0.4 ≈ 0.40 (moderate decrease)
        correlation_to_lr(10, 0, 0.5) → 10^0 = 1.0 (no change)
    """
    dampened = correlation * damping
    return base_lr ** dampened


# =============================================================================
# RESPONSE MAPPING
# =============================================================================

# Human response → Likelihood Ratio
RESPONSE_LR = {
    'confirm': 10.0,      # Strong positive evidence
    'strong_confirm': 50.0,  # Very strong (e.g., with documentation)
    'reject': 0.1,        # Strong negative (1/10)
    'strong_reject': 0.02,   # Very strong negative (1/50)
    'modify': 1.5,        # Weak positive (partial truth)
    'uncertain': 1.0,     # No information
}


def response_to_lr(response: str) -> float:
    """Convert human response string to likelihood ratio."""
    return RESPONSE_LR.get(response, 1.0)


# =============================================================================
# SIGNIFICANCE SCORING
# =============================================================================

def significance_score(deviation: float, context_weight: float,
                       base_rate: float = 0.5) -> float:
    """
    Score how significant a deviation is in a given context.

    Args:
        deviation: How far from expected (0 = expected, 1 = max deviation)
        context_weight: How much this context cares about this type of deviation
        base_rate: Prior probability of this type of event

    Returns:
        Significance score [0, 1]
    """
    # Rare events with high deviation in important contexts = high significance
    rarity = 1 - base_rate
    return min(1.0, deviation * context_weight * (1 + rarity))


# =============================================================================
# TESTS
# =============================================================================

def self_test():
    """Run self-tests to verify math is correct."""
    print("Testing core_math.py...")

    # Entropy tests
    assert abs(entropy(0.5) - 1.0) < 0.001, "entropy(0.5) should be 1.0"
    assert entropy(0.01) == 0.0, "entropy at boundary should be 0"
    assert entropy(0.99) == 0.0, "entropy at boundary should be 0"
    assert abs(entropy(0.25) - entropy(0.75)) < 0.001, "entropy should be symmetric"
    print("  ✓ entropy tests passed")

    # Bayes LR tests
    assert abs(bayes_lr(0.5, 10) - 0.909) < 0.01, "confirm should → ~0.91"
    assert abs(bayes_lr(0.5, 0.1) - 0.091) < 0.01, "reject should → ~0.09"
    assert abs(bayes_lr(0.5, 1.0) - 0.5) < 0.001, "LR=1 should not change"
    assert bayes_lr(0.01, 100) <= 0.99, "should respect upper bound"
    assert bayes_lr(0.99, 0.01) >= 0.01, "should respect lower bound"
    print("  ✓ bayes_lr tests passed")

    # Beta update tests
    a, b = bayes_beta(5, 5, supports=True, strength=10, competency=1.0)
    assert a == 15 and b == 5, "perfect confirm should add to alpha only"
    a, b = bayes_beta(5, 5, supports=False, strength=10, competency=1.0)
    assert a == 5 and b == 15, "perfect reject should add to beta only"
    a, b = bayes_beta(5, 5, supports=True, strength=10, competency=0.5)
    assert a == 10 and b == 10, "50% competency should split evenly"
    print("  ✓ bayes_beta tests passed")

    # Correlation to LR tests
    assert correlation_to_lr(10, 0) == 1.0, "zero correlation → no change"
    assert correlation_to_lr(10, 0.5) > 1.0, "positive → LR > 1"
    assert correlation_to_lr(10, -0.5) < 1.0, "negative → LR < 1"
    assert correlation_to_lr(10, -1.0) > 0, "LR must always be positive"
    print("  ✓ correlation_to_lr tests passed")

    print("All core_math tests passed! ✓")


if __name__ == "__main__":
    self_test()
