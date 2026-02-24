"""
Steering Vectors - Mathematical Persona Layer
==============================================

Based on Anthropic's research on activation engineering and representation engineering.

Key insight: Model behavior can be steered by adding vectors in activation space
without changing weights. This is:
- Fast (no retraining)
- Reversible (just remove the vector)
- Adjustable (scale the vector strength)
- Composable (add multiple vectors)

Architecture:
    base_output = model(input)
    steered_output = base_output + (alpha * steering_vector)

Where:
- steering_vector: Direction in activation space (e.g., "protective" vs "accommodating")
- alpha: Strength of steering (-1 to 1, can go beyond for stronger effects)

This module provides:
1. SteeringVector - A direction in high-dimensional space
2. PersonaProfile - Collection of steering vectors with weights
3. SteeringEngine - Applies steering to model outputs
4. VectorExtractor - Learns steering vectors from contrastive examples

Mathematical Foundation:
- Vectors are learned from contrastive pairs (positive vs negative examples)
- Mean difference gives the steering direction
- Projection onto this direction measures alignment
- Adding/subtracting moves along the dimension
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path
from enum import Enum
import hashlib


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class SteeringVector:
    """
    A direction in activation space that represents a behavioral dimension.

    Examples:
        - "protective" vs "accommodating"
        - "skeptical" vs "trusting"
        - "verbose" vs "terse"
        - "formal" vs "casual"

    The vector is normalized (unit length) for consistent steering strength.
    """
    name: str
    description: str
    positive_pole: str      # What high values mean (e.g., "protective")
    negative_pole: str      # What low values mean (e.g., "accommodating")
    dimension: int          # Size of the vector
    values: List[float]     # The actual vector values
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        # Normalize on creation
        if self.values:
            self._normalize()

    def _normalize(self):
        """Normalize to unit length."""
        magnitude = math.sqrt(sum(v * v for v in self.values))
        if magnitude > 0:
            self.values = [v / magnitude for v in self.values]

    @property
    def magnitude(self) -> float:
        """Current magnitude (should be 1.0 if normalized)."""
        return math.sqrt(sum(v * v for v in self.values))

    def dot(self, other: 'SteeringVector') -> float:
        """Dot product with another vector (measures alignment)."""
        if len(self.values) != len(other.values):
            raise ValueError("Vectors must have same dimension")
        return sum(a * b for a, b in zip(self.values, other.values))

    def scale(self, alpha: float) -> List[float]:
        """Return scaled version of this vector."""
        return [v * alpha for v in self.values]

    def add(self, other: 'SteeringVector', other_weight: float = 1.0) -> 'SteeringVector':
        """Add another vector (for composing steering)."""
        if len(self.values) != len(other.values):
            raise ValueError("Vectors must have same dimension")
        new_values = [a + b * other_weight for a, b in zip(self.values, other.values)]
        return SteeringVector(
            name=f"{self.name}+{other.name}",
            description=f"Composite: {self.description} + {other.description}",
            positive_pole=f"{self.positive_pole}/{other.positive_pole}",
            negative_pole=f"{self.negative_pole}/{other.negative_pole}",
            dimension=self.dimension,
            values=new_values
        )

    def project(self, point: List[float]) -> float:
        """
        Project a point onto this vector direction.

        Returns scalar indicating how far along this dimension the point is.
        Positive = toward positive_pole, Negative = toward negative_pole.
        """
        if len(point) != len(self.values):
            raise ValueError("Point must have same dimension as vector")
        return sum(p * v for p, v in zip(point, self.values))

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SteeringVector':
        return cls(**data)


@dataclass
class PersonaProfile:
    """
    A collection of steering vectors with their active weights.

    Think of this as a "personality configuration" - multiple dimensions
    each set to a specific strength.

    Example:
        protective_professional = PersonaProfile(
            name="Protective Professional",
            vectors={
                "protective_accommodating": 0.8,   # High protection
                "skeptical_trusting": 0.5,         # Moderate skepticism
                "formal_casual": 0.3,              # Slightly formal
            }
        )
    """
    name: str
    description: str = ""
    vector_weights: Dict[str, float] = field(default_factory=dict)  # vector_name → weight
    active: bool = True
    metadata: Dict = field(default_factory=dict)

    def set_weight(self, vector_name: str, weight: float):
        """Set the weight for a steering vector. Weight typically in [-1, 1]."""
        self.vector_weights[vector_name] = max(-2.0, min(2.0, weight))

    def get_weight(self, vector_name: str) -> float:
        """Get current weight for a vector (0 if not set)."""
        return self.vector_weights.get(vector_name, 0.0)

    def adjust(self, vector_name: str, delta: float):
        """Adjust a weight by delta (for gradual steering changes)."""
        current = self.get_weight(vector_name)
        self.set_weight(vector_name, current + delta)


class MoodState(Enum):
    """Quick presets for common states."""
    NEUTRAL = "neutral"
    PROTECTIVE = "protective"
    ACCOMMODATING = "accommodating"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    FORMAL = "formal"
    CASUAL = "casual"


# =============================================================================
# VECTOR EXTRACTION
# =============================================================================

class VectorExtractor:
    """
    Learns steering vectors from contrastive examples.

    Method:
    1. Collect examples of positive pole behavior (text → embedding)
    2. Collect examples of negative pole behavior (text → embedding)
    3. Compute mean of each set
    4. Steering vector = positive_mean - negative_mean
    5. Normalize

    This gives us a direction in embedding space that represents
    the difference between the two behaviors.
    """

    def __init__(self, embedding_fn: Optional[Callable[[str], List[float]]] = None):
        """
        Args:
            embedding_fn: Function that takes text and returns embedding vector.
                         If None, uses a simple hash-based pseudo-embedding.
        """
        if embedding_fn is None:
            import warnings
            warnings.warn("No embedding function provided - using pseudo-embeddings (test mode only)")
        self.embedding_fn = embedding_fn or self._pseudo_embedding
        self.dimension = 768  # Default dimension, will adjust based on actual embeddings

    def _pseudo_embedding(self, text: str) -> List[float]:
        """
        Simple hash-based pseudo-embedding for testing.

        NOT for production - just allows testing the math without a real model.
        Uses deterministic hash so same text → same vector.
        """
        # Create seed from text
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)

        # Generate pseudo-random vector
        vec = np.random.randn(self.dimension).tolist()

        # Add some structure based on text properties
        # (This makes the pseudo-embeddings slightly meaningful)
        words = text.lower().split()

        # Protective words push in one direction
        protective_words = {'protect', 'guard', 'careful', 'verify', 'check', 'confirm', 'wait'}
        protective_count = sum(1 for w in words if w in protective_words)
        if protective_count > 0:
            for i in range(min(50, self.dimension)):
                vec[i] += 0.3 * protective_count

        # Accommodating words push in opposite direction
        accommodating_words = {'yes', 'sure', 'okay', 'help', 'happy', 'glad', 'absolutely'}
        accommodating_count = sum(1 for w in words if w in accommodating_words)
        if accommodating_count > 0:
            for i in range(min(50, self.dimension)):
                vec[i] -= 0.3 * accommodating_count

        return vec

    def extract_vector(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        name: str,
        positive_pole: str,
        negative_pole: str,
        description: str = ""
    ) -> SteeringVector:
        """
        Extract a steering vector from contrastive examples.

        Args:
            positive_examples: Text examples of positive pole behavior
            negative_examples: Text examples of negative pole behavior
            name: Name for this steering vector
            positive_pole: Label for positive direction
            negative_pole: Label for negative direction
            description: Human-readable description

        Returns:
            SteeringVector representing the direction from negative to positive
        """
        if not positive_examples or not negative_examples:
            raise ValueError("Need at least one example of each pole")

        # Embed all examples
        positive_embeddings = [self.embedding_fn(text) for text in positive_examples]
        negative_embeddings = [self.embedding_fn(text) for text in negative_examples]

        # Compute means
        dim = len(positive_embeddings[0])
        self.dimension = dim

        positive_mean = [
            sum(emb[i] for emb in positive_embeddings) / len(positive_embeddings)
            for i in range(dim)
        ]
        negative_mean = [
            sum(emb[i] for emb in negative_embeddings) / len(negative_embeddings)
            for i in range(dim)
        ]

        # Steering vector = positive - negative
        direction = [p - n for p, n in zip(positive_mean, negative_mean)]

        return SteeringVector(
            name=name,
            description=description or f"Steering from {negative_pole} toward {positive_pole}",
            positive_pole=positive_pole,
            negative_pole=negative_pole,
            dimension=dim,
            values=direction,
            metadata={
                'positive_example_count': len(positive_examples),
                'negative_example_count': len(negative_examples),
            }
        )

    def measure_alignment(self, text: str, vector: SteeringVector) -> float:
        """
        Measure how aligned a piece of text is with a steering vector.

        Returns:
            Score from roughly -1 to 1
            Positive = aligned with positive_pole
            Negative = aligned with negative_pole
        """
        embedding = self.embedding_fn(text)
        raw_projection = vector.project(embedding)

        # Normalize to roughly [-1, 1] range
        # (This is approximate - would need calibration with real embeddings)
        return max(-1.0, min(1.0, raw_projection / 10.0))


# =============================================================================
# STEERING ENGINE
# =============================================================================

class SteeringEngine:
    """
    Applies steering vectors to model outputs.

    In a real implementation, this would:
    1. Hook into model activations at specific layers
    2. Add steering vectors during forward pass
    3. Return modified outputs

    This implementation provides the interface and math,
    with hooks for integration with actual models.
    """

    def __init__(self, path: Optional[str] = None):
        self.vectors: Dict[str, SteeringVector] = {}
        self.profiles: Dict[str, PersonaProfile] = {}
        self.active_profile: Optional[str] = None
        self.path = path
        self.extractor = VectorExtractor()

        # Load default vectors
        self._create_default_vectors()

        if path:
            self._load()

    def _create_default_vectors(self):
        """Create default steering vectors for common dimensions."""

        # Protective vs Accommodating
        protective_examples = [
            "Let me verify that before I commit to anything.",
            "I need to check with my team first.",
            "What's the full scope of this request?",
            "I'll need more details before proceeding.",
            "Let me review our current commitments.",
            "I want to make sure we can deliver before I promise.",
            "Can you send that in writing so I can review it?",
        ]
        accommodating_examples = [
            "Sure, I can help with that!",
            "Absolutely, no problem at all.",
            "Yes, we can make that work.",
            "Happy to help, what do you need?",
            "Of course, I'll get right on it.",
            "No worries, I'll handle it.",
            "Yes, that sounds fine to me.",
        ]
        self.vectors['protective_accommodating'] = self.extractor.extract_vector(
            protective_examples, accommodating_examples,
            name='protective_accommodating',
            positive_pole='protective',
            negative_pole='accommodating',
            description='Guards resources vs readily agrees to requests'
        )

        # Skeptical vs Trusting
        skeptical_examples = [
            "What evidence supports that claim?",
            "I'd like to verify that independently.",
            "That seems inconsistent with what I've seen before.",
            "Can you show me the source for that?",
            "I'm not sure that's accurate.",
            "Let's double-check those numbers.",
            "Who told you that, and how do they know?",
        ]
        trusting_examples = [
            "That makes sense, I trust your judgment.",
            "If you say so, I believe you.",
            "That sounds right to me.",
            "I'll take your word for it.",
            "You're the expert here.",
            "That matches what I expected.",
            "Good enough for me.",
        ]
        self.vectors['skeptical_trusting'] = self.extractor.extract_vector(
            skeptical_examples, trusting_examples,
            name='skeptical_trusting',
            positive_pole='skeptical',
            negative_pole='trusting',
            description='Questions and verifies vs accepts at face value'
        )

        # Analytical vs Intuitive
        analytical_examples = [
            "Let's break this down systematically.",
            "What does the data show?",
            "We need to consider all the variables.",
            "Step by step, first we need to...",
            "The logical conclusion is...",
            "Based on the evidence, we should...",
            "Let me calculate the probability.",
        ]
        intuitive_examples = [
            "My gut says we should...",
            "This feels right to me.",
            "I have a hunch that...",
            "Something tells me...",
            "Let's go with our instincts.",
            "I just know this is the way.",
            "It doesn't feel right.",
        ]
        self.vectors['analytical_intuitive'] = self.extractor.extract_vector(
            analytical_examples, intuitive_examples,
            name='analytical_intuitive',
            positive_pole='analytical',
            negative_pole='intuitive',
            description='Data-driven reasoning vs gut feelings'
        )

        # Formal vs Casual
        formal_examples = [
            "I would like to formally request...",
            "Please find attached the documentation.",
            "Per our previous discussion...",
            "I appreciate your consideration of this matter.",
            "Kindly advise at your earliest convenience.",
            "Thank you for your attention to this issue.",
            "I look forward to your response.",
        ]
        casual_examples = [
            "Hey, quick question...",
            "Just wanted to check in.",
            "FYI - this happened.",
            "Thoughts?",
            "Sounds good!",
            "Cool, thanks!",
            "No worries.",
        ]
        self.vectors['formal_casual'] = self.extractor.extract_vector(
            formal_examples, casual_examples,
            name='formal_casual',
            positive_pole='formal',
            negative_pole='casual',
            description='Professional/formal tone vs relaxed/casual tone'
        )

        # Verbose vs Terse
        verbose_examples = [
            "To provide you with a comprehensive understanding of the situation, I would like to explain that there are several interconnected factors we need to consider.",
            "Allow me to elaborate on the various aspects of this matter in detail.",
            "The complete picture includes multiple dimensions that warrant careful examination.",
            "I want to make sure I give you all the relevant context and background information.",
        ]
        terse_examples = [
            "No.",
            "Done.",
            "See attached.",
            "Confirmed.",
            "Will do.",
            "Noted.",
            "TL;DR: it works.",
        ]
        self.vectors['verbose_terse'] = self.extractor.extract_vector(
            verbose_examples, terse_examples,
            name='verbose_terse',
            positive_pole='verbose',
            negative_pole='terse',
            description='Detailed explanations vs minimal responses'
        )

        # Create default profiles
        self._create_default_profiles()

    def _create_default_profiles(self):
        """Create preset persona profiles."""

        # Protective Professional
        self.profiles['protective_professional'] = PersonaProfile(
            name='Protective Professional',
            description='Guards resources, skeptical of requests, formal tone',
            vector_weights={
                'protective_accommodating': 0.8,
                'skeptical_trusting': 0.5,
                'analytical_intuitive': 0.4,
                'formal_casual': 0.3,
                'verbose_terse': 0.0,
            }
        )

        # Accommodating Helper
        self.profiles['accommodating_helper'] = PersonaProfile(
            name='Accommodating Helper',
            description='Readily helps, trusting, casual and friendly',
            vector_weights={
                'protective_accommodating': -0.7,
                'skeptical_trusting': -0.5,
                'analytical_intuitive': -0.2,
                'formal_casual': -0.4,
                'verbose_terse': 0.0,
            }
        )

        # Analytical Reviewer
        self.profiles['analytical_reviewer'] = PersonaProfile(
            name='Analytical Reviewer',
            description='Thorough analysis, skeptical, detailed explanations',
            vector_weights={
                'protective_accommodating': 0.3,
                'skeptical_trusting': 0.7,
                'analytical_intuitive': 0.8,
                'formal_casual': 0.2,
                'verbose_terse': 0.5,
            }
        )

        # Quick Responder
        self.profiles['quick_responder'] = PersonaProfile(
            name='Quick Responder',
            description='Fast, terse, intuitive responses',
            vector_weights={
                'protective_accommodating': 0.0,
                'skeptical_trusting': 0.0,
                'analytical_intuitive': -0.5,
                'formal_casual': -0.3,
                'verbose_terse': -0.7,
            }
        )

        # Neutral baseline
        self.profiles['neutral'] = PersonaProfile(
            name='Neutral',
            description='No steering applied',
            vector_weights={}
        )

    # -------------------------------------------------------------------------
    # Vector Management
    # -------------------------------------------------------------------------

    def add_vector(self, vector: SteeringVector):
        """Add or update a steering vector."""
        self.vectors[vector.name] = vector
        if self.path:
            self._save()

    def get_vector(self, name: str) -> Optional[SteeringVector]:
        """Get a steering vector by name."""
        return self.vectors.get(name)

    def list_vectors(self) -> List[str]:
        """List all available steering vector names."""
        return list(self.vectors.keys())

    def learn_vector(
        self,
        name: str,
        positive_examples: List[str],
        negative_examples: List[str],
        positive_pole: str,
        negative_pole: str,
        description: str = ""
    ) -> SteeringVector:
        """Learn a new steering vector from examples."""
        vector = self.extractor.extract_vector(
            positive_examples, negative_examples,
            name, positive_pole, negative_pole, description
        )
        self.add_vector(vector)
        return vector

    # -------------------------------------------------------------------------
    # Profile Management
    # -------------------------------------------------------------------------

    def add_profile(self, profile: PersonaProfile):
        """Add or update a persona profile."""
        self.profiles[profile.name] = profile
        if self.path:
            self._save()

    def get_profile(self, name: str) -> Optional[PersonaProfile]:
        """Get a persona profile by name."""
        return self.profiles.get(name)

    def list_profiles(self) -> List[str]:
        """List all available profile names."""
        return list(self.profiles.keys())

    def set_active_profile(self, name: str) -> bool:
        """Set the active persona profile."""
        if name in self.profiles:
            self.active_profile = name
            if self.path:
                self._save()
            return True
        return False

    def get_active_profile(self) -> Optional[PersonaProfile]:
        """Get the currently active profile."""
        if self.active_profile:
            return self.profiles.get(self.active_profile)
        return None

    # -------------------------------------------------------------------------
    # Steering Application
    # -------------------------------------------------------------------------

    def compute_steering_vector(self, profile: Optional[PersonaProfile] = None) -> Optional[List[float]]:
        """
        Compute the combined steering vector for a profile.

        Sums all weighted vectors in the profile.
        """
        profile = profile or self.get_active_profile()
        if not profile:
            return None

        if not profile.vector_weights:
            return None

        # Start with zeros
        first_vec = next(iter(self.vectors.values()), None)
        if not first_vec:
            return None

        dim = first_vec.dimension
        combined = [0.0] * dim

        # Add each weighted vector
        for vec_name, weight in profile.vector_weights.items():
            vec = self.vectors.get(vec_name)
            if vec and weight != 0:
                scaled = vec.scale(weight)
                combined = [c + s for c, s in zip(combined, scaled)]

        return combined

    def steer_embedding(
        self,
        embedding: List[float],
        profile: Optional[PersonaProfile] = None,
        strength: float = 1.0
    ) -> List[float]:
        """
        Apply steering to an embedding vector.

        Args:
            embedding: The original embedding
            profile: Profile to use (or active profile)
            strength: Overall steering strength multiplier

        Returns:
            Steered embedding
        """
        steering = self.compute_steering_vector(profile)
        if not steering:
            return embedding

        if len(embedding) != len(steering):
            # Dimension mismatch - can't steer
            return embedding

        return [e + s * strength for e, s in zip(embedding, steering)]

    def measure_text_alignment(self, text: str, vector_name: str) -> float:
        """
        Measure how aligned a piece of text is with a steering dimension.

        Returns:
            Score roughly in [-1, 1]
            Positive = aligned with positive pole
            Negative = aligned with negative pole
        """
        vector = self.vectors.get(vector_name)
        if not vector:
            return 0.0

        return self.extractor.measure_alignment(text, vector)

    def analyze_text_profile(self, text: str) -> Dict[str, float]:
        """
        Analyze text alignment across all steering dimensions.

        Returns dict of vector_name → alignment_score
        """
        return {
            name: self.measure_text_alignment(text, name)
            for name in self.vectors
        }

    def suggest_response_adjustment(
        self,
        original_response: str,
        target_profile: Optional[PersonaProfile] = None
    ) -> Dict:
        """
        Analyze how to adjust a response to match target profile.

        Returns adjustment suggestions.
        """
        profile = target_profile or self.get_active_profile()
        if not profile:
            return {'adjustments': [], 'message': 'No active profile'}

        current = self.analyze_text_profile(original_response)
        adjustments = []

        for vec_name, target_weight in profile.vector_weights.items():
            current_alignment = current.get(vec_name, 0.0)
            difference = target_weight - current_alignment

            if abs(difference) > 0.3:  # Significant difference
                vec = self.vectors.get(vec_name)
                if vec:
                    if difference > 0:
                        adjustments.append({
                            'dimension': vec_name,
                            'direction': f'more {vec.positive_pole}',
                            'current': current_alignment,
                            'target': target_weight,
                            'gap': difference
                        })
                    else:
                        adjustments.append({
                            'dimension': vec_name,
                            'direction': f'more {vec.negative_pole}',
                            'current': current_alignment,
                            'target': target_weight,
                            'gap': difference
                        })

        return {
            'adjustments': adjustments,
            'current_profile': current,
            'target_profile': profile.vector_weights
        }

    # -------------------------------------------------------------------------
    # Mood / Quick Adjustments
    # -------------------------------------------------------------------------

    def set_mood(self, mood: MoodState):
        """Quick preset for common moods."""
        mood_profiles = {
            MoodState.NEUTRAL: 'neutral',
            MoodState.PROTECTIVE: 'protective_professional',
            MoodState.ACCOMMODATING: 'accommodating_helper',
            MoodState.ANALYTICAL: 'analytical_reviewer',
            MoodState.FORMAL: 'protective_professional',  # Reuse
            MoodState.CASUAL: 'accommodating_helper',     # Reuse
            MoodState.CREATIVE: 'neutral',                # Would need custom
        }
        profile_name = mood_profiles.get(mood, 'neutral')
        self.set_active_profile(profile_name)

    def adjust_dimension(self, dimension: str, delta: float):
        """Adjust a single dimension of the active profile."""
        profile = self.get_active_profile()
        if profile:
            profile.adjust(dimension, delta)
            if self.path:
                self._save()

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _save(self):
        if not self.path:
            return

        data = {
            'vectors': {k: v.to_dict() for k, v in self.vectors.items()},
            'profiles': {k: asdict(v) for k, v in self.profiles.items()},
            'active_profile': self.active_profile
        }
        Path(self.path).write_text(json.dumps(data, indent=2))

    def _load(self):
        if not self.path or not Path(self.path).exists():
            return

        try:
            data = json.loads(Path(self.path).read_text())

            for k, v in data.get('vectors', {}).items():
                self.vectors[k] = SteeringVector.from_dict(v)

            for k, v in data.get('profiles', {}).items():
                self.profiles[k] = PersonaProfile(**v)

            self.active_profile = data.get('active_profile')
        except Exception as e:
            print(f"Warning: Could not load {self.path}: {e}")

    # -------------------------------------------------------------------------
    # Export for LLM Context
    # -------------------------------------------------------------------------

    def context(self) -> str:
        """Generate context string for LLM consumption."""
        lines = ["=== ACTIVE STEERING ===\n"]

        profile = self.get_active_profile()
        if profile:
            lines.append(f"Profile: {profile.name}")
            lines.append(f"Description: {profile.description}")
            lines.append("\nActive Steering Dimensions:")

            for vec_name, weight in profile.vector_weights.items():
                if weight != 0:
                    vec = self.vectors.get(vec_name)
                    if vec:
                        if weight > 0:
                            lines.append(f"  → {vec.positive_pole}: {weight:+.1f}")
                        else:
                            lines.append(f"  → {vec.negative_pole}: {abs(weight):.1f}")
        else:
            lines.append("No active profile (neutral)")

        lines.append("\n=== END STEERING ===")
        return "\n".join(lines)


# =============================================================================
# INTEGRATION WITH GUARDIAN
# =============================================================================

def create_guardian_steering_rules():
    """
    Create steering adjustments based on Guardian context.

    Returns functions that adjust steering based on:
    - Entity threat level
    - Current context
    - Historical patterns
    """

    def adjust_for_threat(engine: SteeringEngine, threat_level: int):
        """Adjust steering based on entity threat level."""
        profile = engine.get_active_profile()
        if not profile:
            engine.set_active_profile('neutral')
            profile = engine.get_active_profile()

        # Higher threat = more protective
        protective_weight = min(1.0, threat_level * 0.25)
        profile.set_weight('protective_accommodating', protective_weight)

        # Higher threat = more skeptical
        skeptical_weight = min(0.8, threat_level * 0.2)
        profile.set_weight('skeptical_trusting', skeptical_weight)

    def adjust_for_context(engine: SteeringEngine, domain: str):
        """Adjust steering based on context domain."""
        profile = engine.get_active_profile()
        if not profile:
            engine.set_active_profile('neutral')
            profile = engine.get_active_profile()

        if domain == 'work_executive':
            profile.set_weight('formal_casual', 0.4)
            profile.set_weight('protective_accommodating', 0.6)
        elif domain == 'legal':
            profile.set_weight('analytical_intuitive', 0.8)
            profile.set_weight('skeptical_trusting', 0.7)
            profile.set_weight('formal_casual', 0.5)
        elif domain == 'personal':
            profile.set_weight('formal_casual', -0.3)
            profile.set_weight('protective_accommodating', -0.2)

    return {
        'adjust_for_threat': adjust_for_threat,
        'adjust_for_context': adjust_for_context,
    }


# =============================================================================
# TESTS
# =============================================================================

def self_test():
    """Test steering vectors system."""
    print("Testing steering_vectors.py...")

    # Test SteeringVector basics
    vec1 = SteeringVector(
        name="test",
        description="Test vector",
        positive_pole="positive",
        negative_pole="negative",
        dimension=3,
        values=[1.0, 0.0, 0.0]
    )
    assert abs(vec1.magnitude - 1.0) < 0.01, "Should be normalized"
    print("  ✓ SteeringVector normalization works")

    # Test dot product
    vec2 = SteeringVector(
        name="test2",
        description="Test vector 2",
        positive_pole="positive",
        negative_pole="negative",
        dimension=3,
        values=[1.0, 0.0, 0.0]
    )
    assert abs(vec1.dot(vec2) - 1.0) < 0.01, "Parallel vectors should have dot=1"
    print("  ✓ Dot product works")

    # Test VectorExtractor
    extractor = VectorExtractor()
    vector = extractor.extract_vector(
        positive_examples=["I need to verify that", "Let me check first"],
        negative_examples=["Sure, no problem", "Happy to help"],
        name="test_vector",
        positive_pole="cautious",
        negative_pole="eager"
    )
    assert vector.dimension > 0, "Should have dimension"
    assert len(vector.values) == vector.dimension, "Values should match dimension"
    print(f"  ✓ Vector extraction works (dim={vector.dimension})")

    # Test SteeringEngine
    engine = SteeringEngine()
    assert len(engine.vectors) > 0, "Should have default vectors"
    assert len(engine.profiles) > 0, "Should have default profiles"
    print(f"  ✓ SteeringEngine created with {len(engine.vectors)} vectors and {len(engine.profiles)} profiles")

    # Test profile activation
    engine.set_active_profile('protective_professional')
    profile = engine.get_active_profile()
    assert profile is not None, "Should have active profile"
    assert profile.name == 'Protective Professional'
    print(f"  ✓ Profile activation works: {profile.name}")

    # Test steering computation
    steering = engine.compute_steering_vector()
    assert steering is not None, "Should compute steering vector"
    assert len(steering) > 0, "Steering should have values"
    print(f"  ✓ Steering computation works (dim={len(steering)})")

    # Test text alignment
    protective_text = "Let me verify that before I commit to anything."
    accommodating_text = "Sure, I can help with that right away!"

    prot_score = engine.measure_text_alignment(protective_text, 'protective_accommodating')
    acc_score = engine.measure_text_alignment(accommodating_text, 'protective_accommodating')

    assert prot_score > acc_score, f"Protective text should score higher: {prot_score} vs {acc_score}"
    print(f"  ✓ Text alignment works: protective={prot_score:.2f}, accommodating={acc_score:.2f}")

    # Test mood setting
    engine.set_mood(MoodState.ANALYTICAL)
    profile = engine.get_active_profile()
    assert profile.name == 'Analytical Reviewer'
    print(f"  ✓ Mood setting works: {profile.name}")

    # Test context generation
    ctx = engine.context()
    assert "STEERING" in ctx
    print("  ✓ Context generation works")

    # Test profile analysis
    analysis = engine.analyze_text_profile("I'll need to check with my team before committing.")
    assert 'protective_accommodating' in analysis
    print(f"  ✓ Profile analysis works: {len(analysis)} dimensions analyzed")

    print("All steering_vectors tests passed! ✓")


if __name__ == "__main__":
    self_test()
