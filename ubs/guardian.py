"""
Guardian - Unified Runtime
==========================

The "AI Secretary" that protects you from resource extraction, tracks patterns,
and learns from your feedback.

This is the main entry point that connects:
- BeliefGraph: Track beliefs and propagate updates
- EntityRegistry: Track people and their patterns
- ContextRouter: Determine what rules apply and how to respond

Core loop:
1. DETECT - Something happens (contact, message, meeting)
2. IDENTIFY - Who is this? What context?
3. SCORE - How significant based on patterns and context?
4. RESPOND - Alert, flag, or ignore
5. LEARN - Record outcome, update beliefs

Usage:
    guardian = Guardian()
    guardian.process_event({
        'type': 'contact',
        'entity_name': 'Mike Chen',
        'interaction_type': 'ask_resource',
        'description': 'Asked for dev help',
        'stated_cost': '2 days'
    })
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from truth_layer import TruthLayer
from entity_registry import EntityRegistry, Entity, Interaction, ThreatLevel
from context_router import ContextRouter, Context, Deviation, ResponseLevel
from steering_vectors import SteeringEngine, PersonaProfile, MoodState


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Event:
    """An incoming event to process."""
    event_type: str             # contact, message, meeting, etc.
    timestamp: str = ""
    entity_name: str = ""       # Who is involved
    entity_id: str = ""         # If known
    interaction_type: str = ""  # ask_resource, inform, etc.
    description: str = ""
    stated_cost: str = ""
    context_signals: Dict = field(default_factory=dict)
    raw_data: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ProcessingResult:
    """Result of processing an event."""
    event: Event
    entity: Optional[Entity]
    context: Context
    significance_score: float
    response_level: str
    action: Dict
    matched_rules: List[str]
    explanation: str


# =============================================================================
# GUARDIAN
# =============================================================================

class Guardian:
    """
    The unified protection system.

    Combines belief tracking, entity monitoring, and context-aware routing.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize Guardian with optional data directory for persistence.

        Args:
            data_dir: Directory to store JSON files. If None, runs in memory only.
        """
        self.data_dir = Path(data_dir) if data_dir else None
        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        belief_path = str(self.data_dir / "truth_layer.json") if self.data_dir else "truth_layer.json"
        entity_path = str(self.data_dir / "entities.json") if self.data_dir else None
        router_path = str(self.data_dir / "rules.json") if self.data_dir else None
        steering_path = str(self.data_dir / "steering.json") if self.data_dir else None

        self.beliefs = TruthLayer(path=belief_path)
        self.entities = EntityRegistry(path=entity_path, belief_graph=self.beliefs)
        self.router = ContextRouter(path=router_path)
        self.steering = SteeringEngine(path=steering_path)

        # Set default protective profile
        self.steering.set_active_profile('protective_professional')

        # Event log
        self.event_log: List[Dict] = []

    # -------------------------------------------------------------------------
    # Main Processing Loop
    # -------------------------------------------------------------------------

    def process_event(self, event_data: Dict) -> ProcessingResult:
        """
        Process an incoming event through the full pipeline.

        Args:
            event_data: Dictionary with event details

        Returns:
            ProcessingResult with action to take
        """
        # 1. Parse event
        event = Event(**{k: v for k, v in event_data.items()
                        if k in Event.__dataclass_fields__})

        # 2. Identify entity
        entity = self._identify_entity(event)

        # 3. Detect context
        context_signals = event.context_signals.copy()
        if entity:
            context_signals['entity_relationship'] = entity.relationship
            context_signals['entity_id'] = entity.id
        context = self.router.detect_context(context_signals)

        # 4. Create deviation
        deviation = Deviation(
            deviation_type=event.interaction_type or event.event_type,
            source=entity.id if entity else event.entity_name,
            magnitude=self._estimate_magnitude(event, entity),
            description=event.description,
            raw_data={
                'interaction_type': event.interaction_type,
                'stated_cost': event.stated_cost,
                'event_type': event.event_type
            }
        )

        # 5. Adjust steering based on context and entity
        self._adjust_steering(entity, context)

        # 6. Score significance
        entity_data = self._get_entity_data(entity) if entity else {}
        sig_result = self.router.score_deviation(deviation, context, entity_data)

        # 7. Generate response
        action = self.router.get_response(sig_result)

        # 8. Add steering context to response
        action['steering_profile'] = self.steering.get_active_profile().name if self.steering.get_active_profile() else 'neutral'

        # 7. Enhance response with entity-specific info
        if entity and action.get('show_context'):
            alert_info = self.entities.get_alert_info(entity.id)
            if alert_info:
                action['entity_context'] = alert_info

        # 8. Record interaction (if entity known)
        if entity and event.interaction_type:
            self.entities.record_interaction(
                entity.id,
                event.interaction_type,
                event.description,
                stated_cost=event.stated_cost,
                you_initiated=False
            )

        # 9. Log event
        result = ProcessingResult(
            event=event,
            entity=entity,
            context=context,
            significance_score=sig_result.score,
            response_level=ResponseLevel(sig_result.response_level).name,
            action=action,
            matched_rules=sig_result.matched_rules,
            explanation=sig_result.explanation
        )
        self._log_event(result)

        return result

    def _identify_entity(self, event: Event) -> Optional[Entity]:
        """Find or create entity for this event."""
        # First try by ID
        if event.entity_id:
            entity = self.entities.get(event.entity_id)
            if entity:
                return entity

        # Then try by name
        if event.entity_name:
            matches = self.entities.find_by_name(event.entity_name)
            if matches:
                return matches[0]

            # Create new entity if we have a name but no match
            entity_id = event.entity_name.lower().replace(' ', '_')
            return self.entities.add_entity(
                id=entity_id,
                name=event.entity_name,
                relationship='peer'  # Default, can be updated
            )

        return None

    def _estimate_magnitude(self, event: Event, entity: Optional[Entity]) -> float:
        """Estimate how significant this deviation is (0-1)."""
        magnitude = 0.5  # Default

        # Asks are more significant than informs
        if event.interaction_type in {'ask_resource', 'ask_commitment'}:
            magnitude = 0.7
        elif event.interaction_type in {'ask_time', 'ask_coverage'}:
            magnitude = 0.6
        elif event.interaction_type == 'inform':
            magnitude = 0.2

        # High threat entities increase magnitude
        if entity and entity.threat_level >= 3:
            magnitude = min(1.0, magnitude + 0.2)

        # Patterns increase magnitude
        if entity and entity.patterns:
            pattern_types = [p.get('pattern_type') for p in entity.patterns]
            if 'understates_cost' in pattern_types:
                magnitude = min(1.0, magnitude + 0.1)
            if 'commits_your_resources' in pattern_types:
                magnitude = min(1.0, magnitude + 0.15)

        return magnitude

    def _get_entity_data(self, entity: Entity) -> Dict:
        """Extract entity data for rule matching."""
        return {
            'threat_level': entity.threat_level,
            'relationship': entity.relationship,
            'patterns': entity.patterns,
            'tags': entity.tags,
            'ask_count': entity.ask_count,
            'negative_outcomes': entity.negative_outcome_count
        }

    def _adjust_steering(self, entity: Optional[Entity], context: Context):
        """
        Dynamically adjust steering based on entity and context.

        This is where the persona layer adapts to the situation:
        - Higher threat → more protective
        - Executive context → more formal
        - Known bad patterns → more skeptical
        """
        profile = self.steering.get_active_profile()
        if not profile:
            self.steering.set_active_profile('protective_professional')
            profile = self.steering.get_active_profile()

        # Adjust based on entity threat level
        if entity:
            threat = entity.threat_level

            # Scale protective dimension with threat
            protective_weight = min(1.0, 0.3 + threat * 0.2)
            profile.set_weight('protective_accommodating', protective_weight)

            # Scale skepticism with threat
            skeptical_weight = min(0.8, threat * 0.2)
            profile.set_weight('skeptical_trusting', skeptical_weight)

            # If entity has pattern of understating costs, increase analytical
            if entity.patterns:
                pattern_types = [p.get('pattern_type') for p in entity.patterns]
                if 'understates_cost' in pattern_types:
                    profile.set_weight('analytical_intuitive', 0.6)

        # Adjust based on context domain
        domain = context.domain
        if domain == 'work_executive':
            profile.set_weight('formal_casual', 0.4)
        elif domain == 'legal':
            profile.set_weight('analytical_intuitive', 0.8)
            profile.set_weight('skeptical_trusting', 0.7)
            profile.set_weight('formal_casual', 0.5)
        elif domain == 'work_peer':
            profile.set_weight('formal_casual', 0.1)
        elif domain == 'personal':
            profile.set_weight('formal_casual', -0.3)
            profile.set_weight('protective_accommodating', 0.0)

    def _log_event(self, result: ProcessingResult):
        """Log processed event."""
        log_entry = {
            'timestamp': result.event.timestamp,
            'entity': result.entity.name if result.entity else None,
            'event_type': result.event.event_type,
            'interaction_type': result.event.interaction_type,
            'significance': result.significance_score,
            'response_level': result.response_level,
            'rules': result.matched_rules
        }
        self.event_log.append(log_entry)

        if self.data_dir:
            log_path = self.data_dir / "event_log.json"
            with open(log_path, 'w') as f:
                json.dump(self.event_log, f, indent=2)

    # -------------------------------------------------------------------------
    # Feedback Loop
    # -------------------------------------------------------------------------

    def record_outcome(self, entity_id: str, outcome: str, actual_cost: str = "",
                       notes: str = ""):
        """
        Record the outcome of an interaction for learning.

        Args:
            entity_id: Which entity
            outcome: 'positive', 'neutral', 'negative', 'very_negative'
            actual_cost: What it actually cost (vs stated)
            notes: Any additional notes
        """
        entity = self.entities.get(entity_id)
        if not entity or not entity.interactions:
            return

        # Update most recent interaction
        self.entities.record_outcome(
            entity_id,
            len(entity.interactions) - 1,
            outcome,
            actual_cost
        )

        # If negative, strengthen the "not reasonable" belief
        if outcome in {'negative', 'very_negative'}:
            claim_id = f"{entity_id}_reasonable"
            if claim_id in self.beliefs.net.beliefs:
                self.beliefs.validate(claim_id, 'reject',
                                      f"Negative outcome: {notes}")

    def validate_belief(self, claim_id: str, response: str,
                        correction: str = ""):
        """
        Directly validate a belief in the truth layer.

        Args:
            claim_id: ID of the claim to validate
            response: 'confirm', 'reject', or 'modify'
            correction: New text if modifying
        """
        self.beliefs.validate(claim_id, response, correction)

    # -------------------------------------------------------------------------
    # Query Interface
    # -------------------------------------------------------------------------

    def get_alert(self, entity_name: str) -> Optional[Dict]:
        """
        Get alert card for an entity (for real-time display).

        This is what the "secretary" shows when detecting someone.
        """
        matches = self.entities.find_by_name(entity_name)
        if not matches:
            return None
        return self.entities.get_alert_info(matches[0].id)

    def get_high_risk_entities(self) -> List[Dict]:
        """Get list of high-risk entities to watch."""
        return [
            self.entities.get_alert_info(e.id)
            for e in self.entities.get_by_threat_level(2)
        ]

    def get_context(self) -> str:
        """Get full context for LLM consumption."""
        parts = [
            self.beliefs.get_truth_context(),
            "",
            self.entities.context(),
            "",
            self.steering.context()
        ]
        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Steering Interface
    # -------------------------------------------------------------------------

    def set_mood(self, mood: MoodState):
        """Quick mood adjustment."""
        self.steering.set_mood(mood)

    def set_persona(self, profile_name: str) -> bool:
        """Set active persona profile."""
        return self.steering.set_active_profile(profile_name)

    def adjust_steering(self, dimension: str, delta: float):
        """Manually adjust a steering dimension."""
        self.steering.adjust_dimension(dimension, delta)

    def get_steering_analysis(self, text: str) -> Dict:
        """Analyze how a text aligns with current steering dimensions."""
        return self.steering.analyze_text_profile(text)

    def suggest_response_adjustment(self, response: str) -> Dict:
        """Get suggestions for adjusting a response to match active persona."""
        return self.steering.suggest_response_adjustment(response)

    def list_personas(self) -> List[str]:
        """List available persona profiles."""
        return self.steering.list_profiles()

    def get_current_steering(self) -> Dict:
        """Get current steering state."""
        profile = self.steering.get_active_profile()
        if not profile:
            return {'profile': 'none', 'weights': {}}
        return {
            'profile': profile.name,
            'description': profile.description,
            'weights': profile.vector_weights
        }

    def get_next_question(self) -> Optional[Dict]:
        """Get the next belief to validate for maximum information gain."""
        # Find highest variance (most uncertain) belief
        beliefs = self.beliefs.net.beliefs
        if not beliefs:
            return None

        # Sort by variance (highest uncertainty first)
        sorted_beliefs = sorted(
            beliefs.items(),
            key=lambda x: x[1].variance,
            reverse=True
        )

        if not sorted_beliefs:
            return None

        claim_id, claim = sorted_beliefs[0]
        return {
            'claim_id': claim_id,
            'text': claim.text,
            'current_probability': claim.probability,
            'variance': claim.variance,
            'category': claim.category
        }

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def stats(self) -> Dict:
        """Get system statistics."""
        return {
            'beliefs': self.beliefs.stats(),
            'entities': {
                'total': len(self.entities.entities),
                'high_risk': len(self.entities.get_by_threat_level(3)),
                'medium_risk': len([e for e in self.entities.entities.values()
                                   if e.threat_level == 2])
            },
            'rules': len(self.router.rules),
            'events_processed': len(self.event_log)
        }

    def add_belief(self, cid: str, text: str, category: str = "general"):
        """Add a belief to the truth layer."""
        self.beliefs.add_claim(cid, text, category)

    def link_beliefs(self, parent: str, child: str, weight: float):
        """Link two beliefs with a relationship weight.

        Args:
            parent: Source belief ID
            child: Target belief ID
            weight: Positive = support, negative = contradiction
        """
        self.beliefs.add_relationship(parent, child, weight)


# =============================================================================
# DEMO & TESTS
# =============================================================================

def demo():
    """Interactive demo of Guardian system."""
    print("=" * 60)
    print("GUARDIAN - AI Protection System Demo")
    print("=" * 60)

    # Create guardian with persistence
    guardian = Guardian(data_dir="/tmp/guardian_demo")

    # Add known entity with history
    mike = guardian.entities.add_entity(
        "mike_chen",
        "Mike Chen",
        role="VP Operations",
        relationship="superior"
    )

    # Add some history
    for i in range(3):
        guardian.entities.record_interaction(
            "mike_chen",
            "ask_resource",
            f"Asked for dev help #{i+1}",
            stated_cost="2 days",
            actual_cost="2 weeks" if i > 0 else "",
            outcome="negative" if i > 0 else "unknown"
        )

    print(f"\n📊 Entity created: {mike.name}")
    print(f"   Threat level: {ThreatLevel(mike.threat_level).name}")
    print(f"   Patterns: {[p.get('pattern_type') for p in mike.patterns]}")

    # Process a new event
    print("\n" + "-" * 40)
    print("📨 NEW EVENT: Mike Chen contacts you")
    print("-" * 40)

    result = guardian.process_event({
        'event_type': 'contact',
        'entity_name': 'Mike Chen',
        'interaction_type': 'ask_resource',
        'description': 'Need to borrow a dev for "quick" server migration',
        'stated_cost': '2 days'
    })

    print(f"\n🎯 SIGNIFICANCE: {result.significance_score:.2f}")
    print(f"📍 RESPONSE LEVEL: {result.response_level}")
    print(f"📋 RULES MATCHED: {result.matched_rules}")

    if result.action.get('notify'):
        print(f"\n⚠️  ALERT!")
        print(f"   Message: {result.action.get('message')}")

        if result.action.get('entity_context'):
            ctx = result.action['entity_context']
            print(f"\n📇 ENTITY CARD:")
            print(f"   Name: {ctx['name']} ({ctx['role']})")
            print(f"   Threat: {ctx['threat_level']}")
            print(f"   Ask count: {ctx['ask_count']}")
            print(f"   Negative outcomes: {ctx['negative_outcomes']}")
            if ctx['patterns']:
                print(f"   Patterns: {ctx['patterns']}")

    # Show system state
    print("\n" + "-" * 40)
    print("📈 SYSTEM STATE")
    print("-" * 40)
    print(guardian.get_context())

    # Stats
    print("\n📊 STATS:")
    for k, v in guardian.stats().items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def self_test():
    """Run self-tests."""
    print("Testing guardian.py...")

    guardian = Guardian()

    # Test event processing without known entity
    result = guardian.process_event({
        'event_type': 'contact',
        'entity_name': 'New Person',
        'interaction_type': 'inform',
        'description': 'Just sharing info'
    })
    assert result.entity is not None, "Should create entity"
    assert result.entity.name == 'New Person'
    print("  ✓ Unknown entity handling works")

    # Test high-threat scenario
    guardian.entities.add_entity(
        "threat_person",
        "Threat Person",
        relationship="superior"
    )
    # Build up threat history
    for i in range(5):
        guardian.entities.record_interaction(
            "threat_person",
            "ask_resource",
            f"Urgent ask #{i}",
            stated_cost="1 day",
            actual_cost="1 week",
            outcome="negative"
        )

    result = guardian.process_event({
        'event_type': 'contact',
        'entity_id': 'threat_person',
        'entity_name': 'Threat Person',
        'interaction_type': 'ask_resource',
        'description': 'Another ask'
    })

    assert result.significance_score > 0.5, "High threat should have high significance"
    assert result.response_level in {'ALERT', 'INTERRUPT', 'CRITICAL'}
    print(f"  ✓ High threat handling: score={result.significance_score:.2f}, level={result.response_level}")

    # Test outcome recording
    guardian.record_outcome("threat_person", "very_negative", "3 weeks", "Way over estimate")
    entity = guardian.entities.get("threat_person")
    assert entity.threat_level >= 3, "Should be high threat after negative outcomes"
    print(f"  ✓ Outcome recording: threat_level={entity.threat_level}")

    # Test context generation
    ctx = guardian.get_context()
    assert "BELIEF" in ctx or "ENTITY" in ctx
    print("  ✓ Context generation works")

    print("All guardian tests passed! ✓")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo()
    else:
        self_test()
