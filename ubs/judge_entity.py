"""
Judge Entity Integration
========================

Emergent feature: Treat judges as entities in the Guardian system.

This combines:
- Judicial Analyzer: Pattern extraction from opinions
- Entity Registry: Track interactions (cases) with the judge
- TruthLayer: Bayesian beliefs about judge behavior
- Steering Vectors: Adjust brief writing style per judge

Use case: You're a lawyer preparing a brief.
1. System loads judge's profile into Guardian
2. Entity threat level = how likely to rule against your position
3. Patterns = procedural preferences, ruling tendencies
4. Steering = adjust writing style to match judge's preferences
"""

from typing import Dict, List, Optional
from datetime import datetime

from truth_layer import TruthLayer
from entity_registry import EntityRegistry, Entity, ThreatLevel
from steering_vectors import SteeringEngine, PersonaProfile
from judicial_analyzer import (
    JudicialProfileGenerator,
    JudgeProfile,
    format_profile_summary
)


class JudgeEntity:
    """
    Integrate a judge into the Guardian entity system.

    Converts judicial patterns into:
    - Entity threat levels (how hostile to your position)
    - Belief claims (what we know about the judge)
    - Steering profiles (how to write for this judge)
    """

    def __init__(self, truth_layer: TruthLayer, entity_registry: EntityRegistry,
                 steering: SteeringEngine):
        self.truth_layer = truth_layer
        self.entities = entity_registry
        self.steering = steering
        self.generator = JudicialProfileGenerator(truth_layer=truth_layer)

    def add_judge(self, judge_name: str, your_side: str = "defendant",
                  case_type: str = None, max_opinions: int = 20) -> Entity:
        """
        Add a judge to the entity registry based on their judicial profile.

        Args:
            judge_name: Name of the judge
            your_side: 'plaintiff' or 'defendant' - determines threat assessment
            case_type: Optional case type filter (copyright, patent, etc.)
            max_opinions: Number of opinions to analyze

        Returns:
            Entity with judge data
        """
        # Generate profile
        profile = self.generator.generate_profile(
            judge_name,
            case_type=case_type,
            max_opinions=max_opinions
        )

        # Create entity ID
        entity_id = f"judge_{judge_name.lower().replace(' ', '_').replace('.', '')}"

        # Calculate threat level based on your side
        threat = self._calculate_threat(profile, your_side)

        # Create entity
        entity = self.entities.add_entity(
            id=entity_id,
            name=f"Judge {judge_name}",
            role="Federal Judge",
            relationship="adjudicator",
            threat_level=threat,
            tags=["judge", profile.court or "federal"]
        )

        # Add patterns from profile
        for factor in profile.procedural_factors + profile.substantive_factors:
            entity.patterns.append({
                'pattern_type': factor.name.lower().replace(' ', '_'),
                'confidence': factor.confidence,
                'evidence_count': factor.evidence_count,
                'description': factor.description,
                'detected_date': datetime.now().isoformat()
            })

        # Record the "interaction" (analyzing their opinions)
        self.entities.record_interaction(
            entity_id,
            "research",
            f"Analyzed {profile.opinions_analyzed} opinions",
            you_initiated=True
        )

        # Create steering profile for this judge
        self._create_judge_steering(entity_id, profile)

        # Add detailed beliefs
        self._add_judge_beliefs(entity_id, judge_name, profile)

        return entity

    def _calculate_threat(self, profile: JudgeProfile, your_side: str) -> int:
        """
        Calculate threat level based on judge's patterns and your side.

        Returns: 0-4 (NONE to CRITICAL)
        """
        threat = 1  # Default LOW

        # Summary judgment tendencies
        if your_side == "defendant":
            # Defendants want SJ granted
            if profile.grant_rate_sj < 0.3:
                threat += 2  # Judge rarely grants SJ - bad for defendant
            elif profile.grant_rate_sj > 0.6:
                threat -= 1  # Judge grants SJ often - good for defendant
        else:
            # Plaintiffs want SJ denied
            if profile.grant_rate_sj > 0.6:
                threat += 2  # Judge grants SJ often - bad for plaintiff
            elif profile.grant_rate_sj < 0.3:
                threat -= 1  # Judge rarely grants - good for plaintiff

        # Procedural strictness increases threat for everyone
        for factor in profile.procedural_factors:
            if 'deadline' in factor.description.lower():
                threat += 1
                break

        return max(0, min(4, threat))

    def _create_judge_steering(self, entity_id: str, profile: JudgeProfile):
        """Create a steering profile tailored to this judge."""
        # Analyze profile to set steering weights
        weights = {
            'protective_accommodating': 0.0,  # Not relevant for briefs
            'skeptical_trusting': 0.0,
            'analytical_intuitive': 0.7,      # Judges want analytical
            'formal_casual': 0.8,             # Very formal
            'verbose_terse': 0.3,             # Concise but thorough
        }

        # Adjust based on patterns
        for factor in profile.procedural_factors + profile.substantive_factors:
            name_lower = factor.name.lower()

            if 'technical' in name_lower:
                weights['analytical_intuitive'] = min(1.0, weights['analytical_intuitive'] + 0.2)

            if 'summary judgment' in name_lower:
                # Judge who grants SJ wants concise, clear briefs
                weights['verbose_terse'] = max(-0.5, weights['verbose_terse'] - 0.2)

        # Create profile
        judge_profile = PersonaProfile(
            name=f"judge_{entity_id}",
            description=f"Writing style optimized for {profile.judge_name}",
            vector_weights=weights
        )

        self.steering.profiles[judge_profile.name] = judge_profile

    def _add_judge_beliefs(self, entity_id: str, judge_name: str,
                          profile: JudgeProfile):
        """Add detailed beliefs about the judge to TruthLayer."""

        # Core beliefs
        beliefs = [
            (f"{entity_id}_technical",
             f"Judge {judge_name} has technical expertise",
             "judge_trait"),
            (f"{entity_id}_strict_procedure",
             f"Judge {judge_name} strictly enforces procedural requirements",
             "judge_trait"),
            (f"{entity_id}_fair_hearing",
             f"Judge {judge_name} gives fair hearings to both sides",
             "judge_trait"),
            (f"{entity_id}_grants_sj",
             f"Judge {judge_name} readily grants summary judgment",
             "judge_pattern"),
            (f"{entity_id}_favors_plaintiff",
             f"Judge {judge_name} tends to favor plaintiffs",
             "judge_pattern"),
        ]

        for claim_id, text, category in beliefs:
            self.truth_layer.add_claim(claim_id, text, category)

        # Link related beliefs
        self.truth_layer.add_relationship(
            f"{entity_id}_strict_procedure",
            f"{entity_id}_grants_sj",
            weight=-4.0  # Strict on procedure → less likely to shortcut with SJ
        )

        self.truth_layer.add_relationship(
            f"{entity_id}_technical",
            f"{entity_id}_fair_hearing",
            weight=3.0  # Technical judges tend to be fair
        )

        # Update based on profile data
        if profile.grant_rate_sj > 0.5:
            self.truth_layer.validate(f"{entity_id}_grants_sj", "confirm")
        else:
            self.truth_layer.validate(f"{entity_id}_grants_sj", "reject")

    def get_brief_guidance(self, entity_id: str) -> Dict:
        """
        Get guidance for writing a brief for this judge.

        Returns dict with:
        - threat_level: How careful to be
        - do_list: Things to do
        - dont_list: Things to avoid
        - steering_profile: Which profile to use
        - model_feel: Intuitive description
        """
        entity = self.entities.get(entity_id)
        if not entity:
            return {'error': f'No judge entity {entity_id}'}

        # Get patterns
        patterns = entity.patterns if entity.patterns else []

        do_list = [
            "File complete record with ALL exhibits",
            "Meet all deadlines without exception",
            "Cite controlling authority accurately",
        ]

        dont_list = [
            "Miss filing deadlines",
            "Submit incomplete briefing",
            "Misstate facts or law",
        ]

        # Add pattern-specific guidance
        for p in patterns:
            ptype = p.get('pattern_type', '')
            if 'summary_judgment' in ptype:
                do_list.append("Support motions with detailed declarations")
            if 'technical' in ptype:
                do_list.append("Explain technology clearly but accurately")
                dont_list.append("Hand-wave technical details")

        return {
            'judge': entity.name,
            'threat_level': ThreatLevel(entity.threat_level).name,
            'do_list': do_list,
            'dont_list': dont_list,
            'steering_profile': f"judge_{entity_id}",
            'patterns': [p.get('description') for p in patterns[:5]]
        }


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demo the judge entity integration."""
    print("=" * 60)
    print("JUDGE ENTITY INTEGRATION - Emergent Feature Demo")
    print("=" * 60)

    # Initialize components
    truth_layer = TruthLayer("judge_entity_demo.json")
    entity_registry = EntityRegistry(belief_graph=truth_layer)
    steering = SteeringEngine()

    # Create judge entity handler
    judge_handler = JudgeEntity(truth_layer, entity_registry, steering)

    # Add Judge Alsup as defendant's counsel
    print("\n📋 Adding Judge Alsup (you represent DEFENDANT)...")
    entity = judge_handler.add_judge(
        "William Alsup",
        your_side="defendant",
        max_opinions=5
    )

    print(f"\n📊 JUDGE ENTITY CREATED:")
    print(f"   Name: {entity.name}")
    print(f"   Threat Level: {ThreatLevel(entity.threat_level).name}")
    print(f"   Patterns: {len(entity.patterns)}")

    # Get brief guidance
    print("\n📝 BRIEF WRITING GUIDANCE:")
    guidance = judge_handler.get_brief_guidance(entity.id)

    print(f"   Threat: {guidance['threat_level']}")
    print(f"\n   DO:")
    for item in guidance['do_list']:
        print(f"   ✓ {item}")
    print(f"\n   DON'T:")
    for item in guidance['dont_list']:
        print(f"   ✗ {item}")

    # Show truth layer state
    print("\n📊 BELIEFS ABOUT JUDGE:")
    print(truth_layer.get_truth_context())

    # Show entity registry
    print("\n📊 ENTITY REGISTRY:")
    print(entity_registry.context())

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
