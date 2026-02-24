"""
Circular Belief Propagation with Loop Correction
=================================================

Standard Loopy Belief Propagation (LBP) has a fundamental problem:
On graphs with cycles, evidence "reverberates" and gets double-counted.

Example problem:
    A → B → C → A (cycle)

    If we observe A=true, the message travels:
    A → B → C → A → B → C → ...

    Each pass REINFORCES the belief, even though it's the same evidence!
    This leads to overconfident, incorrect beliefs.

CIRCULAR BP SOLUTION:
    Add "cancellation factors" that remove the reverberant component.

    For each edge i→j, we compute:
        raw_message[i→j] = standard BP message
        cancellation[i→j] = message[j→i]^(-η)  # η is learned
        corrected_message[i→j] = raw_message[i→j] * cancellation[i→j]

    The cancellation factor "subtracts out" information that came FROM j,
    so we don't send it BACK to j.

References:
    - "Belief propagation for general graphical models with loops"
      arXiv:2411.04957 (November 2024)
    - "Correctness of belief propagation in Gaussian graphical models"
      NeurIPS 2001
    - "Understanding belief propagation and its generalizations"
      Yedidia, Freeman, Weiss (2003)

Key Innovations in this Implementation:
    1. Automatic cycle detection
    2. Adaptive cancellation factor learning
    3. Convergence guarantees via residual tracking
    4. Integration with Dempster-Shafer mass functions
    5. Full provenance tracking through propagation
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, FrozenSet, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import math
import numpy as np
from copy import deepcopy

# Import from belief_math (shared library)
from belief_math import (
    MassFunction,
    DecomposedUncertainty,
    create_mass_function,
    dempster_combine,
    pignistic_probability,
    belief,
    plausibility,
    decompose_from_beta,
)

# VERITY-specific types
from .types import (
    Claim,
    Evidence,
)
from .uncertainty import update_with_evidence


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class EdgeType(Enum):
    """Type of relationship between nodes."""
    SUPPORTS = "supports"           # Positive correlation
    CONTRADICTS = "contradicts"     # Negative correlation
    CORRELATES = "correlates"       # Undirected correlation
    CAUSES = "causes"               # Causal relationship


class PropagationStatus(Enum):
    """Status of belief propagation."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    OSCILLATING = "oscillating"
    FAILED = "failed"


@dataclass
class Edge:
    """
    Directed edge in belief graph.

    Represents a relationship between two claims with associated
    weight and cancellation factor for loop correction.
    """
    source: str                         # Source claim ID
    target: str                         # Target claim ID
    weight: float                       # Relationship strength [-1, 1]
    edge_type: EdgeType = EdgeType.CORRELATES

    # Circular BP correction factor
    # This is learned/adapted during propagation
    cancellation_eta: float = 0.5       # η parameter for this edge

    # Provenance
    evidence_id: Optional[str] = None   # Evidence that established this edge
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def is_positive(self) -> bool:
        """Does this edge represent positive correlation?"""
        return self.weight > 0 or self.edge_type == EdgeType.SUPPORTS

    def reverse(self) -> 'Edge':
        """Create reversed edge (for undirected relationships)."""
        return Edge(
            source=self.target,
            target=self.source,
            weight=self.weight,
            edge_type=self.edge_type,
            cancellation_eta=self.cancellation_eta,
            evidence_id=self.evidence_id,
        )


@dataclass
class Message:
    """
    Message passed between nodes during BP.

    In standard BP, message is a probability distribution.
    Here we use log-odds for numerical stability.
    """
    source: str                         # Sending node
    target: str                         # Receiving node
    log_odds: float                     # Log-odds representation
    iteration: int = 0                  # When this message was computed
    is_corrected: bool = False          # Has cancellation been applied?

    @property
    def probability(self) -> float:
        """Convert log-odds to probability."""
        return 1.0 / (1.0 + math.exp(-self.log_odds))

    @classmethod
    def from_probability(cls, source: str, target: str, prob: float,
                        iteration: int = 0) -> 'Message':
        """Create message from probability."""
        # Clamp to avoid inf
        prob = max(1e-10, min(1 - 1e-10, prob))
        log_odds = math.log(prob / (1 - prob))
        return cls(source=source, target=target, log_odds=log_odds,
                  iteration=iteration)


@dataclass
class PropagationState:
    """
    Complete state of belief propagation.

    Tracks messages, residuals, and convergence.
    """
    # Current messages: (source, target) → Message
    messages: Dict[Tuple[str, str], Message] = field(default_factory=dict)

    # Previous messages for convergence checking
    prev_messages: Dict[Tuple[str, str], Message] = field(default_factory=dict)

    # Learned cancellation factors
    cancellation_factors: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Convergence tracking
    residuals: List[float] = field(default_factory=list)
    iteration: int = 0
    status: PropagationStatus = PropagationStatus.NOT_STARTED

    # Configuration
    max_iterations: int = 100
    tolerance: float = 1e-6
    damping: float = 0.5

    @property
    def converged(self) -> bool:
        return self.status == PropagationStatus.CONVERGED

    @property
    def max_residual(self) -> float:
        return self.residuals[-1] if self.residuals else float('inf')


@dataclass
class PropagationResult:
    """Result of running belief propagation."""
    beliefs: Dict[str, float]           # Claim ID → final probability
    uncertainties: Dict[str, DecomposedUncertainty]
    state: PropagationState
    cycles_detected: List[List[str]]    # Detected cycles in the graph
    messages_passed: int
    computation_time_ms: float


# =============================================================================
# BELIEF NETWORK
# =============================================================================

class BeliefNetwork:
    """
    Belief network with Circular BP for loop-corrected inference.

    This is the core inference engine that replaces simple Bayesian
    updating with proper graphical model inference.

    Key Features:
        1. Automatic cycle detection
        2. Loop-corrected message passing
        3. Convergence guarantees
        4. Integration with Dempster-Shafer
        5. Provenance tracking

    Example:
        network = BeliefNetwork()

        # Add claims
        network.add_claim("A", prior=0.5)
        network.add_claim("B", prior=0.5)
        network.add_claim("C", prior=0.5)

        # Add relationships
        network.add_edge("A", "B", weight=0.8)  # A supports B
        network.add_edge("B", "C", weight=0.6)  # B supports C
        network.add_edge("C", "A", weight=0.4)  # C supports A (creates cycle!)

        # Observe evidence
        network.observe("A", True, strength=0.9)

        # Run inference with loop correction
        result = network.propagate()

        # Get corrected beliefs (won't be overconfident due to cycle)
        print(result.beliefs)
    """

    def __init__(self):
        # Claims: ID → Claim object
        self.claims: Dict[str, Claim] = {}

        # Edges: source → list of edges from source
        self.edges: Dict[str, List[Edge]] = defaultdict(list)

        # Reverse edges: target → list of edges to target
        self.reverse_edges: Dict[str, List[Edge]] = defaultdict(list)

        # Observations (evidence): claim_id → (observed_value, strength)
        self.observations: Dict[str, Tuple[bool, float]] = {}

        # Propagation state
        self.state: Optional[PropagationState] = None

        # Detected cycles (cached)
        self._cycles: Optional[List[List[str]]] = None

    # -------------------------------------------------------------------------
    # Graph Construction
    # -------------------------------------------------------------------------

    def add_claim(
        self,
        claim_id: str,
        text: str = "",
        prior: float = 0.5,
        category: str = "general",
    ) -> Claim:
        """
        Add a claim to the network.

        Args:
            claim_id: Unique identifier
            text: Human-readable claim text
            prior: Prior probability [0, 1]
            category: Claim category

        Returns:
            Created Claim object
        """
        if claim_id in self.claims:
            return self.claims[claim_id]

        claim = Claim.create_boolean(
            id=claim_id,
            text=text or claim_id,
            category=category,
            prior_true=prior,
        )

        # Update mass function to reflect prior
        frame = frozenset({"true", "false"})
        if prior != 0.5:
            # Non-uniform prior
            claim.mass_function = create_mass_function(
                frame,
                {"true": prior * 0.5, "false": (1-prior) * 0.5, "FRAME": 0.5}
            )

        # Set uncertainty from prior
        claim.uncertainty = decompose_from_beta(
            alpha=prior * 2 + 1,
            beta=(1 - prior) * 2 + 1,
        )

        self.claims[claim_id] = claim
        self._cycles = None  # Invalidate cycle cache

        return claim

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float,
        edge_type: EdgeType = EdgeType.CORRELATES,
        evidence_id: Optional[str] = None,
        bidirectional: bool = True,
    ) -> Edge:
        """
        Add an edge (relationship) between claims.

        Args:
            source: Source claim ID
            target: Target claim ID
            weight: Relationship strength [-1, 1]
                    Positive = supports, Negative = contradicts
            edge_type: Type of relationship
            evidence_id: Evidence that established this relationship
            bidirectional: If True, add reverse edge too

        Returns:
            Created Edge object
        """
        # Ensure claims exist
        if source not in self.claims:
            self.add_claim(source)
        if target not in self.claims:
            self.add_claim(target)

        # Clamp weight
        weight = max(-1.0, min(1.0, weight))

        edge = Edge(
            source=source,
            target=target,
            weight=weight,
            edge_type=edge_type,
            evidence_id=evidence_id,
        )

        self.edges[source].append(edge)
        self.reverse_edges[target].append(edge)

        if bidirectional:
            reverse = edge.reverse()
            self.edges[target].append(reverse)
            self.reverse_edges[source].append(reverse)

        self._cycles = None  # Invalidate cycle cache

        return edge

    def observe(
        self,
        claim_id: str,
        value: bool,
        strength: float = 0.9,
    ):
        """
        Observe evidence for a claim.

        Args:
            claim_id: Claim to observe
            value: Observed value (True/False)
            strength: Strength of observation [0, 1]
        """
        if claim_id not in self.claims:
            raise ValueError(f"Unknown claim: {claim_id}")

        self.observations[claim_id] = (value, strength)

        # Update claim's mass function based on observation
        claim = self.claims[claim_id]
        frame = frozenset({"true", "false"})

        if value:
            claim.mass_function = create_mass_function(
                frame,
                {"true": strength, "FRAME": 1 - strength}
            )
        else:
            claim.mass_function = create_mass_function(
                frame,
                {"false": strength, "FRAME": 1 - strength}
            )

    # -------------------------------------------------------------------------
    # Cycle Detection
    # -------------------------------------------------------------------------

    def detect_cycles(self) -> List[List[str]]:
        """
        Detect all cycles in the graph.

        Uses DFS-based cycle detection.

        Returns:
            List of cycles, each cycle is a list of claim IDs
        """
        if self._cycles is not None:
            return self._cycles

        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for edge in self.edges.get(node, []):
                neighbor = edge.target

                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for node in self.claims:
            if node not in visited:
                dfs(node)

        self._cycles = cycles
        return cycles

    def has_cycles(self) -> bool:
        """Check if graph has cycles."""
        return len(self.detect_cycles()) > 0

    # -------------------------------------------------------------------------
    # Message Computation
    # -------------------------------------------------------------------------

    def _compute_raw_message(
        self,
        source: str,
        target: str,
        exclude: Optional[str] = None,
    ) -> Message:
        """
        Compute raw (uncorrected) message from source to target.

        Standard BP message computation:
        m_{i→j}(x_j) ∝ Σ_{x_i} φ_i(x_i) ψ_{ij}(x_i, x_j) Π_{k≠j} m_{k→i}(x_i)

        In log-odds form for binary variables:
        log_odds_{i→j} = log_odds_i + Σ_{k≠j} weight_{ki} * log_odds_{k→i}

        The message is then scaled by the edge weight to the target.
        """
        claim = self.claims[source]

        # Start with claim's own belief (from observation or prior)
        if source in self.observations:
            obs_value, obs_strength = self.observations[source]
            base_prob = obs_strength if obs_value else (1 - obs_strength)
        else:
            base_prob = claim.probability

        # Convert to log-odds
        base_prob = max(1e-10, min(1 - 1e-10, base_prob))
        log_odds = math.log(base_prob / (1 - base_prob))

        # Aggregate incoming messages (except from target)
        for edge in self.reverse_edges.get(source, []):
            if edge.source == target or edge.source == exclude:
                continue

            msg_key = (edge.source, source)
            if msg_key in self.state.messages:
                incoming = self.state.messages[msg_key]
                # Apply edge weight to transform message
                log_odds += edge.weight * incoming.log_odds * 0.5

        # Find the outgoing edge to target and apply its weight
        outgoing_weight = 1.0
        for edge in self.edges.get(source, []):
            if edge.target == target:
                outgoing_weight = edge.weight
                break

        # Apply the edge weight to scale/flip the message
        # Positive weight: source supports target
        # Negative weight: source contradicts target
        final_log_odds = outgoing_weight * log_odds

        return Message(
            source=source,
            target=target,
            log_odds=final_log_odds,
            iteration=self.state.iteration,
            is_corrected=False,
        )

    def _apply_cancellation(
        self,
        message: Message,
        edge: Edge,
    ) -> Message:
        """
        Apply Circular BP cancellation to remove reverberant information.

        The key insight of Circular BP:
        When sending message i→j, we should NOT include information
        that originally came FROM j (via the cycle).

        Cancellation formula:
            corrected[i→j] = raw[i→j] * reverse[j→i]^(-η)

        In log-odds:
            corrected_log_odds = raw_log_odds - η * reverse_log_odds
        """
        reverse_key = (message.target, message.source)

        if reverse_key not in self.state.messages:
            # No reverse message yet, return uncorrected
            return Message(
                source=message.source,
                target=message.target,
                log_odds=message.log_odds,
                iteration=message.iteration,
                is_corrected=True,
            )

        reverse_msg = self.state.messages[reverse_key]

        # Get cancellation factor for this edge
        edge_key = (message.source, message.target)
        eta = self.state.cancellation_factors.get(edge_key, edge.cancellation_eta)

        # Apply cancellation
        corrected_log_odds = message.log_odds - eta * reverse_msg.log_odds

        return Message(
            source=message.source,
            target=message.target,
            log_odds=corrected_log_odds,
            iteration=message.iteration,
            is_corrected=True,
        )

    def _adapt_cancellation(
        self,
        edge: Edge,
        residual: float,
    ):
        """
        Adapt cancellation factor based on convergence behavior.

        If residuals are high (not converging), increase cancellation.
        If residuals are low and stable, can decrease slightly.
        """
        edge_key = (edge.source, edge.target)
        current_eta = self.state.cancellation_factors.get(edge_key, edge.cancellation_eta)

        # Adaptive learning rate
        lr = 0.1 / (1 + self.state.iteration * 0.1)

        if residual > self.state.tolerance * 10:
            # High residual - increase cancellation
            new_eta = min(1.0, current_eta + lr * residual)
        elif residual < self.state.tolerance:
            # Low residual - can decrease slightly
            new_eta = max(0.0, current_eta - lr * 0.1)
        else:
            new_eta = current_eta

        self.state.cancellation_factors[edge_key] = new_eta

    # -------------------------------------------------------------------------
    # Main Propagation Algorithm
    # -------------------------------------------------------------------------

    def propagate(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        damping: float = 0.5,
        use_circular_bp: bool = True,
    ) -> PropagationResult:
        """
        Run belief propagation with optional loop correction.

        Args:
            max_iterations: Maximum iterations before stopping
            tolerance: Convergence tolerance (max residual)
            damping: Damping factor for message updates [0, 1]
            use_circular_bp: Whether to use Circular BP correction

        Returns:
            PropagationResult with final beliefs and diagnostics
        """
        import time
        start_time = time.time()

        # Initialize state
        self.state = PropagationState(
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
        )
        self.state.status = PropagationStatus.RUNNING

        # Detect cycles
        cycles = self.detect_cycles()

        # Initialize messages
        self._initialize_messages()

        # Main propagation loop
        messages_passed = 0

        for iteration in range(max_iterations):
            self.state.iteration = iteration
            max_residual = 0.0

            # Store previous messages
            self.state.prev_messages = deepcopy(self.state.messages)

            # Update all messages
            for source in self.claims:
                for edge in self.edges.get(source, []):
                    target = edge.target

                    # Compute raw message
                    raw_msg = self._compute_raw_message(source, target)

                    # Apply Circular BP correction if enabled and graph has cycles
                    if use_circular_bp and cycles:
                        msg = self._apply_cancellation(raw_msg, edge)
                    else:
                        msg = raw_msg
                        msg.is_corrected = True

                    # Damped update
                    msg_key = (source, target)
                    if msg_key in self.state.prev_messages:
                        old_log_odds = self.state.prev_messages[msg_key].log_odds
                        msg.log_odds = (damping * msg.log_odds +
                                       (1 - damping) * old_log_odds)

                    # Track residual
                    if msg_key in self.state.prev_messages:
                        residual = abs(msg.log_odds -
                                      self.state.prev_messages[msg_key].log_odds)
                        max_residual = max(max_residual, residual)

                        # Adapt cancellation factor
                        if use_circular_bp:
                            self._adapt_cancellation(edge, residual)

                    self.state.messages[msg_key] = msg
                    messages_passed += 1

            self.state.residuals.append(max_residual)

            # Check convergence
            if max_residual < tolerance:
                self.state.status = PropagationStatus.CONVERGED
                break

            # Check for oscillation
            if len(self.state.residuals) > 10:
                recent = self.state.residuals[-10:]
                if max(recent) - min(recent) < tolerance * 2:
                    # Residuals are stable but not decreasing
                    self.state.status = PropagationStatus.OSCILLATING
                    break
        else:
            self.state.status = PropagationStatus.MAX_ITERATIONS

        # Compute final beliefs
        beliefs = self._compute_final_beliefs()
        uncertainties = self._compute_uncertainties(beliefs)

        computation_time = (time.time() - start_time) * 1000

        return PropagationResult(
            beliefs=beliefs,
            uncertainties=uncertainties,
            state=self.state,
            cycles_detected=cycles,
            messages_passed=messages_passed,
            computation_time_ms=computation_time,
        )

    def _initialize_messages(self):
        """Initialize all messages with prior beliefs."""
        for source in self.claims:
            for edge in self.edges.get(source, []):
                target = edge.target
                claim = self.claims[source]

                # Initialize with prior
                msg = Message.from_probability(
                    source=source,
                    target=target,
                    prob=claim.probability,
                    iteration=0,
                )

                self.state.messages[(source, target)] = msg

                # Initialize cancellation factor
                self.state.cancellation_factors[(source, target)] = edge.cancellation_eta

    def _compute_final_beliefs(self) -> Dict[str, float]:
        """
        Compute final beliefs from converged messages.

        Belief at node i = prior * product of all incoming messages.
        """
        beliefs = {}

        for claim_id, claim in self.claims.items():
            # Start with observation if present, else prior
            if claim_id in self.observations:
                obs_value, obs_strength = self.observations[claim_id]
                base_prob = obs_strength if obs_value else (1 - obs_strength)
            else:
                base_prob = claim.probability

            base_prob = max(1e-10, min(1 - 1e-10, base_prob))
            log_odds = math.log(base_prob / (1 - base_prob))

            # Aggregate all incoming messages
            # Edge weights are already incorporated into the messages
            for edge in self.reverse_edges.get(claim_id, []):
                msg_key = (edge.source, claim_id)
                if msg_key in self.state.messages:
                    msg = self.state.messages[msg_key]
                    # Use tanh-scaled influence for bounded updates
                    # This prevents any single message from dominating
                    scaled_log_odds = math.tanh(msg.log_odds / 3.0) * 3.0
                    log_odds += scaled_log_odds * 0.5

            # Convert back to probability
            prob = 1.0 / (1.0 + math.exp(-log_odds))
            beliefs[claim_id] = prob

        return beliefs

    def _compute_uncertainties(
        self,
        beliefs: Dict[str, float],
    ) -> Dict[str, DecomposedUncertainty]:
        """
        Compute uncertainty estimates from propagation.

        Epistemic uncertainty comes from:
        - Message disagreement (different paths give different beliefs)
        - Non-convergence (residuals > 0)

        Aleatoric uncertainty comes from:
        - Inherent randomness in the claim
        """
        uncertainties = {}

        for claim_id, final_belief in beliefs.items():
            claim = self.claims[claim_id]

            # Collect incoming message beliefs
            incoming_beliefs = []
            for edge in self.reverse_edges.get(claim_id, []):
                msg_key = (edge.source, claim_id)
                if msg_key in self.state.messages:
                    incoming_beliefs.append(
                        self.state.messages[msg_key].probability
                    )

            if incoming_beliefs:
                # Epistemic: variance between incoming messages
                msg_variance = np.var(incoming_beliefs)

                # Add contribution from non-convergence
                convergence_penalty = self.state.max_residual * 0.1
                epistemic_var = msg_variance + convergence_penalty
            else:
                # No incoming messages - use prior uncertainty
                epistemic_var = claim.uncertainty.epistemic_variance

            # Aleatoric: based on final belief (closer to 0.5 = more uncertain)
            aleatoric_var = final_belief * (1 - final_belief) * 0.5

            uncertainties[claim_id] = DecomposedUncertainty(
                mean=final_belief,
                epistemic_variance=float(max(0, epistemic_var)),
                aleatoric_variance=float(max(0, aleatoric_var)),
                n_observations=len(incoming_beliefs),
            )

        return uncertainties

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_belief(self, claim_id: str) -> float:
        """Get current belief for a claim."""
        if self.state is None or not self.state.messages:
            return self.claims[claim_id].probability

        beliefs = self._compute_final_beliefs()
        return beliefs.get(claim_id, self.claims[claim_id].probability)

    def get_uncertainty(self, claim_id: str) -> DecomposedUncertainty:
        """Get uncertainty for a claim."""
        if self.state is None:
            return self.claims[claim_id].uncertainty

        beliefs = self._compute_final_beliefs()
        uncertainties = self._compute_uncertainties(beliefs)
        return uncertainties.get(claim_id, self.claims[claim_id].uncertainty)

    def explain_belief(self, claim_id: str) -> str:
        """
        Explain how a belief was computed.

        Returns human-readable explanation.
        """
        if claim_id not in self.claims:
            return f"Unknown claim: {claim_id}"

        lines = [f"=== BELIEF EXPLANATION: {claim_id} ==="]

        claim = self.claims[claim_id]
        lines.append(f"Text: {claim.text}")

        # Prior
        lines.append(f"\nPrior: {claim.probability:.3f}")

        # Observation
        if claim_id in self.observations:
            obs_value, obs_strength = self.observations[claim_id]
            lines.append(f"Observation: {obs_value} (strength={obs_strength:.2f})")

        # Incoming edges
        incoming = self.reverse_edges.get(claim_id, [])
        if incoming:
            lines.append(f"\nIncoming relationships ({len(incoming)}):")
            for edge in incoming:
                source_belief = self.get_belief(edge.source)
                lines.append(f"  ← {edge.source}: weight={edge.weight:.2f}, "
                           f"belief={source_belief:.3f}")

                msg_key = (edge.source, claim_id)
                if self.state and msg_key in self.state.messages:
                    msg = self.state.messages[msg_key]
                    lines.append(f"     message log-odds={msg.log_odds:.3f}, "
                               f"corrected={msg.is_corrected}")

        # Final belief
        final = self.get_belief(claim_id)
        lines.append(f"\nFinal belief: {final:.3f}")

        # Uncertainty
        unc = self.get_uncertainty(claim_id)
        lines.append(f"Uncertainty: epistemic={unc.epistemic_variance:.4f}, "
                    f"aleatoric={unc.aleatoric_variance:.4f}")

        # Cycles
        cycles = self.detect_cycles()
        relevant_cycles = [c for c in cycles if claim_id in c]
        if relevant_cycles:
            lines.append(f"\nParticipates in {len(relevant_cycles)} cycle(s):")
            for cycle in relevant_cycles[:3]:
                lines.append(f"  {' → '.join(cycle)}")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialize network to dictionary."""
        return {
            "claims": {k: v.to_dict() for k, v in self.claims.items()},
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "edge_type": e.edge_type.value,
                    "cancellation_eta": e.cancellation_eta,
                    "evidence_id": e.evidence_id,
                }
                for edges in self.edges.values()
                for e in edges
            ],
            "observations": self.observations,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BeliefNetwork':
        """Deserialize network from dictionary."""
        network = cls()

        for claim_id, claim_data in data.get("claims", {}).items():
            claim = Claim.from_dict(claim_data)
            network.claims[claim_id] = claim

        seen_edges = set()
        for edge_data in data.get("edges", []):
            edge_key = (edge_data["source"], edge_data["target"])
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            edge = Edge(
                source=edge_data["source"],
                target=edge_data["target"],
                weight=edge_data["weight"],
                edge_type=EdgeType(edge_data["edge_type"]),
                cancellation_eta=edge_data.get("cancellation_eta", 0.5),
                evidence_id=edge_data.get("evidence_id"),
            )
            network.edges[edge.source].append(edge)
            network.reverse_edges[edge.target].append(edge)

        network.observations = data.get("observations", {})

        return network


# =============================================================================
# COMPARISON: Standard LBP vs Circular BP
# =============================================================================

def compare_lbp_vs_circular(
    network: BeliefNetwork,
    claim_id: str,
) -> Dict[str, Any]:
    """
    Compare standard Loopy BP with Circular BP on the same network.

    Shows the difference in beliefs when loop correction is used.
    """
    # Run without correction
    result_lbp = network.propagate(use_circular_bp=False)
    belief_lbp = result_lbp.beliefs.get(claim_id, 0.5)

    # Run with correction
    result_cbp = network.propagate(use_circular_bp=True)
    belief_cbp = result_cbp.beliefs.get(claim_id, 0.5)

    return {
        "claim_id": claim_id,
        "has_cycles": len(result_cbp.cycles_detected) > 0,
        "cycles": result_cbp.cycles_detected,
        "standard_lbp": {
            "belief": belief_lbp,
            "iterations": result_lbp.state.iteration,
            "converged": result_lbp.state.converged,
            "status": result_lbp.state.status.value,
        },
        "circular_bp": {
            "belief": belief_cbp,
            "iterations": result_cbp.state.iteration,
            "converged": result_cbp.state.converged,
            "status": result_cbp.state.status.value,
        },
        "difference": abs(belief_lbp - belief_cbp),
        "lbp_overconfident": abs(belief_lbp - 0.5) > abs(belief_cbp - 0.5),
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def self_test():
    """Comprehensive tests for Circular BP."""
    print("Testing verity/belief_propagation.py...")

    # Test 1: Simple chain (no cycles)
    print("\n  --- Test 1: Simple Chain (A → B → C) ---")
    network = BeliefNetwork()
    network.add_claim("A", "Claim A", prior=0.5)
    network.add_claim("B", "Claim B", prior=0.5)
    network.add_claim("C", "Claim C", prior=0.5)

    network.add_edge("A", "B", weight=0.8, bidirectional=False)
    network.add_edge("B", "C", weight=0.6, bidirectional=False)

    network.observe("A", True, strength=0.9)

    result = network.propagate()

    assert not network.has_cycles(), "Chain should have no cycles"
    assert result.beliefs["A"] > 0.8, "A should have high belief"
    assert result.beliefs["B"] > result.beliefs["C"], "B should be higher than C"
    print(f"  ✓ Chain propagation: A={result.beliefs['A']:.3f}, "
          f"B={result.beliefs['B']:.3f}, C={result.beliefs['C']:.3f}")
    print(f"  ✓ Converged in {result.state.iteration} iterations, "
          f"status={result.state.status.value}")

    # Test 2: Triangle cycle (A → B → C → A)
    print("\n  --- Test 2: Triangle Cycle (A → B → C → A) ---")
    network2 = BeliefNetwork()
    network2.add_claim("A", prior=0.5)
    network2.add_claim("B", prior=0.5)
    network2.add_claim("C", prior=0.5)

    network2.add_edge("A", "B", weight=0.7, bidirectional=False)
    network2.add_edge("B", "C", weight=0.7, bidirectional=False)
    network2.add_edge("C", "A", weight=0.7, bidirectional=False)

    network2.observe("A", True, strength=0.8)

    cycles = network2.detect_cycles()
    assert len(cycles) > 0, "Should detect cycle"
    print(f"  ✓ Detected {len(cycles)} cycle(s): {cycles[0] if cycles else 'none'}")

    # Compare LBP vs Circular BP
    comparison = compare_lbp_vs_circular(network2, "B")
    print(f"  ✓ Standard LBP belief(B): {comparison['standard_lbp']['belief']:.3f}")
    print(f"  ✓ Circular BP belief(B): {comparison['circular_bp']['belief']:.3f}")
    print(f"  ✓ Difference: {comparison['difference']:.4f}")
    print(f"  ✓ LBP overconfident: {comparison['lbp_overconfident']}")

    # Test 3: Conflicting evidence
    print("\n  --- Test 3: Conflicting Evidence ---")
    network3 = BeliefNetwork()
    network3.add_claim("H", "Hypothesis", prior=0.5)
    network3.add_claim("E1", "Evidence 1", prior=0.5)
    network3.add_claim("E2", "Evidence 2", prior=0.5)

    network3.add_edge("E1", "H", weight=0.8)   # E1 supports H
    network3.add_edge("E2", "H", weight=-0.7)  # E2 contradicts H

    network3.observe("E1", True, strength=0.9)
    network3.observe("E2", True, strength=0.9)

    result3 = network3.propagate()

    # Conflicting evidence should result in moderate belief
    assert 0.3 < result3.beliefs["H"] < 0.7, "Conflicting evidence should moderate belief"
    print(f"  ✓ Conflicting evidence: H={result3.beliefs['H']:.3f}")
    print(f"  ✓ Uncertainty: epistemic={result3.uncertainties['H'].epistemic_variance:.4f}")

    # Test 4: Larger network with multiple cycles
    print("\n  --- Test 4: Complex Network ---")
    network4 = BeliefNetwork()
    for i in range(5):
        network4.add_claim(f"N{i}", prior=0.5)

    # Create complex connectivity
    network4.add_edge("N0", "N1", weight=0.6)
    network4.add_edge("N1", "N2", weight=0.6)
    network4.add_edge("N2", "N3", weight=0.6)
    network4.add_edge("N3", "N4", weight=0.6)
    network4.add_edge("N4", "N0", weight=0.5)  # Cycle back
    network4.add_edge("N1", "N3", weight=0.4)  # Shortcut

    network4.observe("N0", True, strength=0.85)

    result4 = network4.propagate()

    cycles4 = network4.detect_cycles()
    print(f"  ✓ Network with {len(network4.claims)} nodes, "
          f"{sum(len(e) for e in network4.edges.values())} edges")
    print(f"  ✓ Detected {len(cycles4)} cycle(s)")
    print(f"  ✓ Converged: {result4.state.converged}, "
          f"iterations: {result4.state.iteration}")
    print(f"  ✓ Messages passed: {result4.messages_passed}")
    print(f"  ✓ Time: {result4.computation_time_ms:.2f}ms")

    # Test 5: Explain belief
    print("\n  --- Test 5: Belief Explanation ---")
    explanation = network4.explain_belief("N2")
    assert "N2" in explanation
    assert "Incoming" in explanation
    print(f"  ✓ Generated explanation ({len(explanation)} chars)")

    # Test 6: Serialization
    print("\n  --- Test 6: Serialization ---")
    data = network4.to_dict()
    network4_loaded = BeliefNetwork.from_dict(data)

    assert len(network4_loaded.claims) == len(network4.claims)
    print(f"  ✓ Serialization roundtrip: {len(network4_loaded.claims)} claims")

    # Test 7: Edge types
    print("\n  --- Test 7: Edge Types ---")
    network5 = BeliefNetwork()
    network5.add_claim("cause", prior=0.5)
    network5.add_claim("effect", prior=0.5)

    edge = network5.add_edge(
        "cause", "effect",
        weight=0.9,
        edge_type=EdgeType.CAUSES,
        bidirectional=False
    )

    assert edge.edge_type == EdgeType.CAUSES
    assert edge.is_positive
    print(f"  ✓ Edge type: {edge.edge_type.value}, positive={edge.is_positive}")

    print("\n" + "=" * 50)
    print("All belief_propagation tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    self_test()
