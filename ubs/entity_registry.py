"""
Entity Registry - Track People, Projects, and Patterns
======================================================

This is the "secretary's memory" - who does what, what they typically want,
how interactions have gone historically.

Integrates with TruthLayer to create claims about entities automatically.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path
from enum import Enum

from truth_layer import TruthLayer


# =============================================================================
# ENUMS
# =============================================================================

class ThreatLevel(Enum):
    """How much to guard against this entity."""
    NONE = 0        # Ally, trusted
    LOW = 1         # Generally fine, occasional asks
    MEDIUM = 2      # Regular asks, need to watch
    HIGH = 3        # Frequent resource extraction, protect actively
    CRITICAL = 4    # Known bad actor, maximum defense


class InteractionType(Enum):
    """Types of interactions to track."""
    ASK_RESOURCE = "ask_resource"       # Wants your people/budget
    ASK_TIME = "ask_time"               # Wants your time
    ASK_COVERAGE = "ask_coverage"       # Wants you to cover for them
    ASK_COMMITMENT = "ask_commitment"   # Wants you to commit to something
    PROVIDE_HELP = "provide_help"       # They helped you
    COLLABORATE = "collaborate"         # Genuine two-way work
    INFORM = "inform"                   # Just information sharing
    ESCALATE = "escalate"               # Escalated to/from


class OutcomeType(Enum):
    """How an interaction turned out."""
    POSITIVE = "positive"       # Good for you
    NEUTRAL = "neutral"         # No impact
    NEGATIVE = "negative"       # Cost you something
    VERY_NEGATIVE = "very_negative"  # Significant cost
    UNKNOWN = "unknown"         # Haven't seen outcome yet


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Interaction:
    """A single interaction with an entity."""
    timestamp: str
    interaction_type: str   # InteractionType value
    description: str
    stated_cost: str = ""   # What they said it would cost
    actual_cost: str = ""   # What it actually cost
    outcome: str = "unknown"  # OutcomeType value
    you_initiated: bool = False
    context: str = ""       # Meeting, hallway, slack, etc.


@dataclass
class Pattern:
    """A detected pattern in entity behavior."""
    pattern_type: str       # e.g., "understates_cost", "last_minute_asks"
    confidence: float       # 0-1 how confident in this pattern
    evidence_count: int     # How many interactions support this
    description: str
    detected_date: str


@dataclass
class Entity:
    """
    A person, team, or external party you interact with.

    Tracks:
    - Basic info (name, role, relationship)
    - Threat level (how much to guard)
    - Interaction history
    - Detected patterns
    - Suggested responses
    """
    id: str
    name: str
    role: str = ""
    organization: str = ""
    relationship: str = ""  # peer, superior, report, external

    # Threat assessment
    threat_level: int = 1   # ThreatLevel value
    threat_notes: str = ""

    # History
    interactions: List[Dict] = field(default_factory=list)
    patterns: List[Dict] = field(default_factory=list)

    # Response suggestions
    default_response: str = ""  # e.g., "Let me check my team's capacity"
    delay_phrases: List[str] = field(default_factory=list)

    # Metadata
    created: str = ""
    updated: str = ""
    tags: List[str] = field(default_factory=list)

    def add_interaction(self, interaction: Interaction):
        self.interactions.append(asdict(interaction))
        self.updated = datetime.now().isoformat()

    def add_pattern(self, pattern: Pattern):
        self.patterns.append(asdict(pattern))
        self.updated = datetime.now().isoformat()

    @property
    def ask_count(self) -> int:
        """How many times they've asked for something."""
        ask_types = {'ask_resource', 'ask_time', 'ask_coverage', 'ask_commitment'}
        return sum(1 for i in self.interactions
                   if i.get('interaction_type') in ask_types)

    @property
    def negative_outcome_count(self) -> int:
        """How many interactions had negative outcomes."""
        return sum(1 for i in self.interactions
                   if i.get('outcome') in {'negative', 'very_negative'})

    @property
    def cost_accuracy(self) -> Optional[float]:
        """How accurate are their cost estimates? None if insufficient data."""
        with_costs = [i for i in self.interactions
                      if i.get('stated_cost') and i.get('actual_cost')]
        if len(with_costs) < 2:
            return None
        # Simple heuristic: count how many times actual exceeded stated
        understated = sum(1 for i in with_costs
                         if 'week' in i['actual_cost'].lower() and 'day' in i['stated_cost'].lower())
        return 1.0 - (understated / len(with_costs))


# =============================================================================
# ENTITY REGISTRY
# =============================================================================

class EntityRegistry:
    """
    Registry of all entities you interact with.

    Provides:
    - CRUD for entities
    - Pattern detection across interactions
    - Threat level calculation
    - Integration with TruthLayer
    """

    def __init__(self, path: Optional[str] = None, belief_graph: Optional[TruthLayer] = None):
        self.entities: Dict[str, Entity] = {}
        self.path = path
        self.belief_graph = belief_graph  # Actually TruthLayer now
        if path:
            self._load()

    # -------------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------------

    def add_entity(self, id: str, name: str, role: str = "",
                   relationship: str = "peer", **kwargs) -> Entity:
        """Add a new entity to track."""
        entity = Entity(
            id=id,
            name=name,
            role=role,
            relationship=relationship,
            created=datetime.now().isoformat(),
            updated=datetime.now().isoformat(),
            **kwargs
        )
        self.entities[id] = entity

        # Create beliefs about this entity
        if self.belief_graph:
            self._create_entity_beliefs(entity)

        if self.path:
            self._save()

        return entity

    def get(self, id: str) -> Optional[Entity]:
        return self.entities.get(id)

    def find_by_name(self, name: str) -> List[Entity]:
        """Fuzzy match by name."""
        name_lower = name.lower()
        return [e for e in self.entities.values()
                if name_lower in e.name.lower()]

    def get_by_threat_level(self, min_level: int = 2) -> List[Entity]:
        """Get entities at or above a threat level."""
        return sorted(
            [e for e in self.entities.values() if e.threat_level >= min_level],
            key=lambda e: -e.threat_level
        )

    # -------------------------------------------------------------------------
    # Interaction Recording
    # -------------------------------------------------------------------------

    def record_interaction(self, entity_id: str, interaction_type: str,
                          description: str, **kwargs) -> Optional[Interaction]:
        """Record an interaction with an entity."""
        entity = self.entities.get(entity_id)
        if not entity:
            return None

        interaction = Interaction(
            timestamp=datetime.now().isoformat(),
            interaction_type=interaction_type,
            description=description,
            **kwargs
        )
        entity.add_interaction(interaction)

        # Update threat level based on new data
        self._update_threat_level(entity)

        # Detect patterns
        self._detect_patterns(entity)

        # Update beliefs
        if self.belief_graph:
            self._update_entity_beliefs(entity, interaction)

        if self.path:
            self._save()

        return interaction

    def record_outcome(self, entity_id: str, interaction_index: int,
                       outcome: str, actual_cost: str = ""):
        """Update an interaction with its outcome."""
        entity = self.entities.get(entity_id)
        if not entity or interaction_index >= len(entity.interactions):
            return

        entity.interactions[interaction_index]['outcome'] = outcome
        if actual_cost:
            entity.interactions[interaction_index]['actual_cost'] = actual_cost
        entity.updated = datetime.now().isoformat()

        self._update_threat_level(entity)
        self._detect_patterns(entity)

        if self.path:
            self._save()

    # -------------------------------------------------------------------------
    # Threat Assessment
    # -------------------------------------------------------------------------

    def _update_threat_level(self, entity: Entity):
        """Recalculate threat level based on interaction history.

        NOTE: Threat levels only increase based on negative patterns - there is
        no decay over time. Once an entity reaches a high threat level, it stays
        there permanently. This is intentional: trust is hard to rebuild.
        """
        if not entity.interactions:
            return

        # Factors that increase threat level
        score = 0

        # Ask frequency (more asks = higher threat)
        ask_count = entity.ask_count
        if ask_count >= 10:
            score += 2
        elif ask_count >= 5:
            score += 1

        # Negative outcome ratio
        if len(entity.interactions) >= 3:
            neg_ratio = entity.negative_outcome_count / len(entity.interactions)
            if neg_ratio > 0.5:
                score += 2
            elif neg_ratio > 0.25:
                score += 1

        # Cost accuracy (do they understate costs?)
        accuracy = entity.cost_accuracy
        if accuracy is not None and accuracy < 0.5:
            score += 1

        # Pattern penalties
        for pattern in entity.patterns:
            if pattern.get('pattern_type') in {'understates_cost', 'last_minute_asks', 'commits_your_resources'}:
                score += 1

        # Cap at CRITICAL (4)
        entity.threat_level = min(4, max(0, score))

    def _detect_patterns(self, entity: Entity):
        """Detect behavioral patterns from interaction history."""
        if len(entity.interactions) < 3:
            return

        existing_types = {p.get('pattern_type') for p in entity.patterns}

        # Pattern: Understates costs
        with_costs = [i for i in entity.interactions
                      if i.get('stated_cost') and i.get('actual_cost')]
        if len(with_costs) >= 2:
            understated = sum(1 for i in with_costs
                             if 'week' in i['actual_cost'].lower() and 'day' in i['stated_cost'].lower())
            if understated >= 2 and 'understates_cost' not in existing_types:
                entity.add_pattern(Pattern(
                    pattern_type='understates_cost',
                    confidence=min(0.9, understated / len(with_costs)),
                    evidence_count=understated,
                    description=f"Stated costs are often lower than actual ({understated} instances)",
                    detected_date=datetime.now().isoformat()
                ))

        # Pattern: Last minute asks
        # (Would need timestamp analysis - simplified here)
        last_minute = sum(1 for i in entity.interactions
                         if 'urgent' in i.get('description', '').lower() or
                         'asap' in i.get('description', '').lower())
        if last_minute >= 3 and 'last_minute_asks' not in existing_types:
            entity.add_pattern(Pattern(
                pattern_type='last_minute_asks',
                confidence=min(0.9, last_minute / len(entity.interactions)),
                evidence_count=last_minute,
                description=f"Frequently makes urgent/ASAP requests ({last_minute} instances)",
                detected_date=datetime.now().isoformat()
            ))

        # Pattern: Commits your resources without asking
        commits = sum(1 for i in entity.interactions
                     if 'already' in i.get('description', '').lower() and
                     'commit' in i.get('description', '').lower())
        if commits >= 2 and 'commits_your_resources' not in existing_types:
            entity.add_pattern(Pattern(
                pattern_type='commits_your_resources',
                confidence=min(0.9, commits / len(entity.interactions)),
                evidence_count=commits,
                description=f"Commits your resources before asking ({commits} instances)",
                detected_date=datetime.now().isoformat()
            ))

    # -------------------------------------------------------------------------
    # Belief Graph Integration
    # -------------------------------------------------------------------------

    def _create_entity_beliefs(self, entity: Entity):
        """Create initial beliefs about an entity."""
        if not self.belief_graph:
            return

        # Claim: This person makes reasonable asks
        self.belief_graph.add_claim(
            f"{entity.id}_reasonable",
            f"{entity.name} makes reasonable asks",
            category='entity_trait'
        )

        # Claim: This person's cost estimates are accurate
        self.belief_graph.add_claim(
            f"{entity.id}_accurate_costs",
            f"{entity.name}'s cost estimates are accurate",
            category='entity_trait'
        )

        # Claim: This person reciprocates help
        self.belief_graph.add_claim(
            f"{entity.id}_reciprocates",
            f"{entity.name} reciprocates when you help them",
            category='entity_trait'
        )

        # Link them - positive weights = support, negative = contradiction
        # If reasonable, more likely to reciprocate
        self.belief_graph.add_relationship(f"{entity.id}_reasonable", f"{entity.id}_reciprocates", 6.0)
        # If accurate costs, more likely to be reasonable
        self.belief_graph.add_relationship(f"{entity.id}_accurate_costs", f"{entity.id}_reasonable", 5.0)

    def _update_entity_beliefs(self, entity: Entity, interaction: Interaction):
        """Update beliefs based on new interaction."""
        if not self.belief_graph:
            return

        # If negative outcome, reject "reasonable" claim
        if interaction.outcome in {'negative', 'very_negative'}:
            claim_id = f"{entity.id}_reasonable"
            if claim_id in self.belief_graph.net.beliefs:
                self.belief_graph.validate(claim_id, 'reject')

        # If they understated costs, reject "accurate_costs"
        if (interaction.stated_cost and interaction.actual_cost and
            'week' in interaction.actual_cost.lower() and
            'day' in interaction.stated_cost.lower()):
            claim_id = f"{entity.id}_accurate_costs"
            if claim_id in self.belief_graph.net.beliefs:
                self.belief_graph.validate(claim_id, 'reject')

    # -------------------------------------------------------------------------
    # Query & Export
    # -------------------------------------------------------------------------

    def get_alert_info(self, entity_id: str) -> Optional[Dict]:
        """
        Get information for real-time alert when this entity is detected.

        This is what the "secretary" shows you.
        """
        entity = self.entities.get(entity_id)
        if not entity:
            return None

        # Build alert card
        alert = {
            'name': entity.name,
            'role': entity.role,
            'threat_level': ThreatLevel(entity.threat_level).name,
            'ask_count': entity.ask_count,
            'negative_outcomes': entity.negative_outcome_count,
            'patterns': [p.get('description') for p in entity.patterns],
            'suggested_response': entity.default_response or self._get_default_response(entity),
            'recent_interactions': entity.interactions[-3:] if entity.interactions else []
        }

        return alert

    def _get_default_response(self, entity: Entity) -> str:
        """Generate a default delay phrase based on threat level."""
        responses = {
            0: "",  # No delay needed for allies
            1: "Let me check my calendar.",
            2: "Let me check what my team has committed this sprint.",
            3: "I need to review our capacity and get back to you.",
            4: "I'll need to discuss this with my manager first."
        }
        return responses.get(entity.threat_level, "Let me get back to you on that.")

    def context(self) -> str:
        """Generate LLM-ready context about entities."""
        lines = ["=== ENTITY REGISTRY ===\n"]

        # High threat entities
        high_threat = self.get_by_threat_level(3)
        if high_threat:
            lines.append("HIGH THREAT (protect actively):")
            for e in high_threat:
                patterns = ", ".join(p.get('pattern_type', '') for p in e.patterns)
                lines.append(f"  ⚠️ {e.name} ({e.role}): {e.ask_count} asks, "
                           f"patterns: {patterns or 'none detected'}")

        # Medium threat
        medium_threat = [e for e in self.entities.values() if e.threat_level == 2]
        if medium_threat:
            lines.append("\nMEDIUM THREAT (watch):")
            for e in medium_threat:
                lines.append(f"  ⚡ {e.name} ({e.role}): {e.ask_count} asks")

        lines.append("\n=== END REGISTRY ===")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _save(self):
        if not self.path:
            return
        data = {
            'entities': {k: asdict(v) for k, v in self.entities.items()}
        }
        Path(self.path).write_text(json.dumps(data, indent=2))

    def _load(self):
        if not self.path or not Path(self.path).exists():
            return
        try:
            data = json.loads(Path(self.path).read_text())
            for k, v in data.get('entities', {}).items():
                self.entities[k] = Entity(**v)
        except Exception as e:
            print(f"Warning: Could not load {self.path}: {e}")


# =============================================================================
# TESTS
# =============================================================================

def self_test():
    """Test entity registry."""
    print("Testing entity_registry.py...")

    # Create with TruthLayer integration
    tl = TruthLayer()
    reg = EntityRegistry(belief_graph=tl)

    # Add an entity
    mike = reg.add_entity(
        "mike_chen",
        "Mike Chen",
        role="VP Operations",
        relationship="peer"
    )
    assert mike.threat_level == 1, "Initial threat should be LOW"
    print("  ✓ Entity creation works")

    # Record interactions
    reg.record_interaction(
        "mike_chen",
        "ask_resource",
        "Asked for dev help on server migration",
        stated_cost="2 days",
        actual_cost="",
        you_initiated=False
    )
    print("  ✓ Interaction recording works")

    # Record more interactions to trigger pattern detection
    for i in range(4):
        reg.record_interaction(
            "mike_chen",
            "ask_resource",
            f"Asked for urgent help #{i+2}",
            stated_cost="1 day",
            actual_cost="2 weeks" if i > 1 else "",
            outcome="negative" if i > 1 else "unknown"
        )

    # Check threat level increased
    assert mike.threat_level >= 2, f"Threat should increase, got {mike.threat_level}"
    print(f"  ✓ Threat level updated to {ThreatLevel(mike.threat_level).name}")

    # Check patterns detected
    assert len(mike.patterns) > 0, "Should detect patterns"
    print(f"  ✓ Patterns detected: {[p.get('pattern_type') for p in mike.patterns]}")

    # Check TruthLayer integration
    assert f"mike_chen_reasonable" in tl.net.beliefs, "Should create entity beliefs"
    print("  ✓ TruthLayer integration works")

    # Check alert info
    alert = reg.get_alert_info("mike_chen")
    assert alert is not None
    assert 'suggested_response' in alert
    print(f"  ✓ Alert info: threat={alert['threat_level']}, response='{alert['suggested_response']}'")

    print("All entity_registry tests passed! ✓")


if __name__ == "__main__":
    self_test()
