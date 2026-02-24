"""
Context Router - Determines What Rules Apply
============================================

The same deviation means different things in different contexts:
- Particle in clean room = CRITICAL
- Particle on beach = IGNORE
- Unexpected ask from known threat = ALERT
- Unexpected ask from ally = PROBABLY FINE

This module:
1. Detects current context (domain, situation)
2. Loads appropriate rules
3. Scores significance of deviations
4. Routes to appropriate response
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class ResponseLevel(Enum):
    """What action to take based on significance."""
    IGNORE = 0          # Log only, no action
    NOTE = 1            # Log with note for later review
    FLAG = 2            # Flag for batch review
    ALERT = 3           # Subtle alert (vibration)
    INTERRUPT = 4       # Interrupt with full context
    CRITICAL = 5        # Automated response + human notification


class Domain(Enum):
    """Known domains with pre-built rules."""
    WORK_EXECUTIVE = "work_executive"       # Interactions with executives
    WORK_PEER = "work_peer"                 # Peer interactions
    WORK_REPORT = "work_report"             # Managing reports
    WORK_EXTERNAL = "work_external"         # External parties (vendors, etc)
    PERSONAL = "personal"
    FINANCIAL = "financial"
    HEALTH = "health"
    LEGAL = "legal"
    UNKNOWN = "unknown"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ContextRule:
    """
    A rule that applies in a specific context.

    Example:
        name: "resource_ask_from_threat"
        domain: "work_executive"
        trigger: entity.threat_level >= 2 AND interaction_type == "ask_resource"
        response_level: ALERT
        significance_weight: 0.8
    """
    name: str
    domain: str
    description: str
    response_level: int = 2  # ResponseLevel value
    significance_weight: float = 0.5  # How much this context cares about matches
    trigger_conditions: Dict = field(default_factory=dict)
    response_template: str = ""


@dataclass
class Context:
    """
    Current detected context.

    Represents "where we are" and "what rules apply".
    """
    domain: str
    sub_domain: str = ""
    active_rules: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    confidence: float = 1.0  # How confident in this context detection


@dataclass
class Deviation:
    """
    A detected deviation from expected behavior.

    This is the "movement" that the predator eye catches.
    """
    deviation_type: str         # e.g., "unexpected_contact", "resource_ask"
    source: str                 # What/who triggered this
    magnitude: float = 0.5      # How far from expected [0, 1]
    description: str = ""
    raw_data: Dict = field(default_factory=dict)


@dataclass
class SignificanceResult:
    """
    Result of scoring a deviation in context.
    """
    deviation: Deviation
    context: Context
    score: float                    # Final significance [0, 1]
    response_level: int             # ResponseLevel to use
    matched_rules: List[str]        # Which rules fired
    suggested_response: str = ""    # What to say/do
    explanation: str = ""           # Why this score


# =============================================================================
# CONTEXT ROUTER
# =============================================================================

class ContextRouter:
    """
    Routes deviations through context-specific rules to determine response.

    Core operations:
    1. detect_context() - Figure out what domain we're in
    2. score_deviation() - How significant is this deviation in this context
    3. get_response() - What action to take
    """

    def __init__(self, path: Optional[str] = None):
        self.rules: Dict[str, ContextRule] = {}
        self.current_context: Optional[Context] = None
        self.path = path

        # Load default rules
        self._load_default_rules()

        if path:
            self._load()

    # -------------------------------------------------------------------------
    # Default Rules
    # -------------------------------------------------------------------------

    def _load_default_rules(self):
        """Load built-in rules for common scenarios."""

        # Work Executive Rules
        self.add_rule(ContextRule(
            name="exec_resource_ask",
            domain="work_executive",
            description="Executive asking for your resources",
            response_level=ResponseLevel.ALERT.value,
            significance_weight=0.9,
            trigger_conditions={
                "interaction_type": ["ask_resource", "ask_time", "ask_coverage"],
                "entity_relationship": "superior"
            },
            response_template="Let me check what we have committed this sprint."
        ))

        self.add_rule(ContextRule(
            name="exec_unscheduled_contact",
            domain="work_executive",
            description="Unscheduled contact from executive",
            response_level=ResponseLevel.FLAG.value,
            significance_weight=0.6,
            trigger_conditions={
                "contact_type": "unscheduled",
                "entity_relationship": "superior"
            },
            response_template=""
        ))

        self.add_rule(ContextRule(
            name="exec_already_committed",
            domain="work_executive",
            description="Executive already committed your resources",
            response_level=ResponseLevel.INTERRUPT.value,
            significance_weight=1.0,
            trigger_conditions={
                "pattern": "commits_your_resources"
            },
            response_template="I'll need to verify our capacity before I can confirm that commitment."
        ))

        # Work Peer Rules
        self.add_rule(ContextRule(
            name="peer_high_threat_ask",
            domain="work_peer",
            description="High-threat peer making an ask",
            response_level=ResponseLevel.ALERT.value,
            significance_weight=0.8,
            trigger_conditions={
                "entity_threat_level": [3, 4],
                "interaction_type": ["ask_resource", "ask_time"]
            },
            response_template="Let me review my team's priorities and get back to you."
        ))

        self.add_rule(ContextRule(
            name="peer_pattern_understates",
            domain="work_peer",
            description="Peer with pattern of understating costs",
            response_level=ResponseLevel.ALERT.value,
            significance_weight=0.85,
            trigger_conditions={
                "entity_pattern": "understates_cost"
            },
            response_template="What's the full scope? I want to make sure we plan appropriately."
        ))

        # External Rules
        self.add_rule(ContextRule(
            name="external_salesperson",
            domain="work_external",
            description="External salesperson contact",
            response_level=ResponseLevel.FLAG.value,
            significance_weight=0.4,
            trigger_conditions={
                "entity_tags": ["sales", "vendor"]
            },
            response_template="Please send details to my email and I'll review when I have time."
        ))

        # Legal Domain Rules
        self.add_rule(ContextRule(
            name="legal_fact_claim",
            domain="legal",
            description="Factual claim in legal context",
            response_level=ResponseLevel.INTERRUPT.value,
            significance_weight=1.0,
            trigger_conditions={
                "claim_type": "fact",
                "source": "llm_generated"
            },
            response_template="[REQUIRES RECORD CITATION]"
        ))

    # -------------------------------------------------------------------------
    # Rule Management
    # -------------------------------------------------------------------------

    def add_rule(self, rule: ContextRule):
        """Add or update a rule."""
        self.rules[rule.name] = rule
        if self.path:
            self._save()

    def get_rules_for_domain(self, domain: str) -> List[ContextRule]:
        """Get all rules that apply to a domain."""
        return [r for r in self.rules.values() if r.domain == domain]

    # -------------------------------------------------------------------------
    # Context Detection
    # -------------------------------------------------------------------------

    def detect_context(self, signals: Dict) -> Context:
        """
        Detect current context from available signals.

        Args:
            signals: Dictionary with keys like:
                - entity_id: Who we're interacting with
                - entity_relationship: Their relationship to us
                - location: Where we are (office, meeting room, etc)
                - calendar_event: Current meeting if any
                - application: What app is active
                - keywords: Detected keywords in conversation

        Returns:
            Detected Context with domain and active rules
        """
        domain = Domain.UNKNOWN.value
        sub_domain = ""
        confidence = 0.5

        # Entity-based detection
        relationship = signals.get('entity_relationship', '')
        if relationship == 'superior':
            domain = Domain.WORK_EXECUTIVE.value
            confidence = 0.9
        elif relationship == 'peer':
            domain = Domain.WORK_PEER.value
            confidence = 0.8
        elif relationship == 'report':
            domain = Domain.WORK_REPORT.value
            confidence = 0.8
        elif relationship == 'external':
            domain = Domain.WORK_EXTERNAL.value
            confidence = 0.7

        # Keyword-based refinement
        keywords = signals.get('keywords', [])
        if any(k in keywords for k in ['legal', 'court', 'judge', 'lawsuit']):
            domain = Domain.LEGAL.value
            confidence = 0.95
        elif any(k in keywords for k in ['health', 'medical', 'doctor']):
            domain = Domain.HEALTH.value
            confidence = 0.9

        # Application-based refinement
        app = signals.get('application', '')
        if 'slack' in app.lower() or 'teams' in app.lower():
            sub_domain = 'messaging'
        elif 'zoom' in app.lower() or 'meet' in app.lower():
            sub_domain = 'video_call'
        elif 'mail' in app.lower():
            sub_domain = 'email'

        # Get active rules for this domain
        active_rules = [r.name for r in self.get_rules_for_domain(domain)]

        context = Context(
            domain=domain,
            sub_domain=sub_domain,
            active_rules=active_rules,
            metadata=signals,
            confidence=confidence
        )

        self.current_context = context
        return context

    # -------------------------------------------------------------------------
    # Significance Scoring
    # -------------------------------------------------------------------------

    def score_deviation(self, deviation: Deviation, context: Optional[Context] = None,
                        entity_data: Optional[Dict] = None) -> SignificanceResult:
        """
        Score how significant a deviation is in the given context.

        Args:
            deviation: The detected deviation
            context: Context to use (or current_context if None)
            entity_data: Additional data about the entity involved

        Returns:
            SignificanceResult with score, response level, and suggestions
        """
        context = context or self.current_context or Context(domain=Domain.UNKNOWN.value)
        entity_data = entity_data or {}

        matched_rules = []
        max_significance = 0.0
        max_response_level = ResponseLevel.IGNORE.value
        suggested_response = ""

        # Check each rule for this domain
        for rule in self.get_rules_for_domain(context.domain):
            if self._rule_matches(rule, deviation, entity_data):
                matched_rules.append(rule.name)

                # Calculate significance for this rule
                rule_sig = deviation.magnitude * rule.significance_weight
                if rule_sig > max_significance:
                    max_significance = rule_sig
                    max_response_level = rule.response_level
                    if rule.response_template:
                        suggested_response = rule.response_template

        # Also check domain-agnostic rules
        for rule in self.get_rules_for_domain("*"):
            if self._rule_matches(rule, deviation, entity_data):
                matched_rules.append(rule.name)
                rule_sig = deviation.magnitude * rule.significance_weight
                if rule_sig > max_significance:
                    max_significance = rule_sig
                    max_response_level = rule.response_level

        # Build explanation
        if matched_rules:
            explanation = f"Matched rules: {', '.join(matched_rules)}. " \
                         f"Domain: {context.domain}, Deviation: {deviation.deviation_type}"
        else:
            explanation = f"No rules matched. Domain: {context.domain}"

        return SignificanceResult(
            deviation=deviation,
            context=context,
            score=min(1.0, max_significance),
            response_level=max_response_level,
            matched_rules=matched_rules,
            suggested_response=suggested_response,
            explanation=explanation
        )

    def _rule_matches(self, rule: ContextRule, deviation: Deviation,
                      entity_data: Dict) -> bool:
        """Check if a rule's conditions are met."""
        conditions = rule.trigger_conditions

        # Check interaction type
        if 'interaction_type' in conditions:
            allowed = conditions['interaction_type']
            if isinstance(allowed, str):
                allowed = [allowed]
            if deviation.deviation_type not in allowed and \
               deviation.raw_data.get('interaction_type') not in allowed:
                return False

        # Check entity threat level
        if 'entity_threat_level' in conditions:
            allowed_levels = conditions['entity_threat_level']
            if isinstance(allowed_levels, int):
                allowed_levels = [allowed_levels]
            if entity_data.get('threat_level', 0) not in allowed_levels:
                return False

        # Check entity relationship
        if 'entity_relationship' in conditions:
            if entity_data.get('relationship') != conditions['entity_relationship']:
                return False

        # Check entity patterns
        if 'entity_pattern' in conditions:
            patterns = entity_data.get('patterns', [])
            pattern_types = [p.get('pattern_type') if isinstance(p, dict) else p for p in patterns]
            if conditions['entity_pattern'] not in pattern_types:
                return False

        # Check entity tags
        if 'entity_tags' in conditions:
            required_tags = conditions['entity_tags']
            entity_tags = entity_data.get('tags', [])
            if not any(t in entity_tags for t in required_tags):
                return False

        # Check contact type
        if 'contact_type' in conditions:
            if deviation.raw_data.get('contact_type') != conditions['contact_type']:
                return False

        # If we get here, all conditions passed
        return True

    # -------------------------------------------------------------------------
    # Response Generation
    # -------------------------------------------------------------------------

    def get_response(self, result: SignificanceResult) -> Dict:
        """
        Generate appropriate response based on significance result.

        Returns action instructions for the system.
        """
        level = ResponseLevel(result.response_level)

        response = {
            'action': level.name.lower(),
            'level': result.response_level,
            'score': result.score,
            'rules': result.matched_rules,
        }

        if level == ResponseLevel.IGNORE:
            response['message'] = None
            response['notify'] = False

        elif level == ResponseLevel.NOTE:
            response['message'] = None
            response['notify'] = False
            response['log'] = True

        elif level == ResponseLevel.FLAG:
            response['message'] = None
            response['notify'] = False
            response['flag_for_review'] = True

        elif level == ResponseLevel.ALERT:
            response['notify'] = True
            response['vibrate'] = True
            response['message'] = result.suggested_response
            response['show_context'] = False

        elif level == ResponseLevel.INTERRUPT:
            response['notify'] = True
            response['vibrate'] = True
            response['message'] = result.suggested_response
            response['show_context'] = True
            response['explanation'] = result.explanation

        elif level == ResponseLevel.CRITICAL:
            response['notify'] = True
            response['vibrate'] = True
            response['sound'] = True
            response['message'] = result.suggested_response
            response['show_context'] = True
            response['explanation'] = result.explanation
            response['auto_response'] = result.suggested_response

        return response

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _save(self):
        if not self.path:
            return
        data = {
            'rules': {k: asdict(v) for k, v in self.rules.items()}
        }
        Path(self.path).write_text(json.dumps(data, indent=2))

    def _load(self):
        if not self.path or not Path(self.path).exists():
            return
        try:
            data = json.loads(Path(self.path).read_text())
            for k, v in data.get('rules', {}).items():
                self.rules[k] = ContextRule(**v)
        except Exception as e:
            print(f"Warning: Could not load {self.path}: {e}")


# =============================================================================
# TESTS
# =============================================================================

def self_test():
    """Test context router."""
    print("Testing context_router.py...")

    router = ContextRouter()

    # Test context detection
    context = router.detect_context({
        'entity_relationship': 'superior',
        'entity_id': 'mike_chen'
    })
    assert context.domain == 'work_executive'
    print(f"  ✓ Context detection: {context.domain}")

    # Test rule matching with high-threat ask
    deviation = Deviation(
        deviation_type="ask_resource",
        source="mike_chen",
        magnitude=0.8,
        description="Asked for dev resources",
        raw_data={'interaction_type': 'ask_resource'}
    )

    entity_data = {
        'threat_level': 3,
        'relationship': 'superior',
        'patterns': [{'pattern_type': 'understates_cost'}]
    }

    result = router.score_deviation(deviation, context, entity_data)
    print(f"  ✓ Significance score: {result.score:.2f}")
    print(f"  ✓ Response level: {ResponseLevel(result.response_level).name}")
    print(f"  ✓ Matched rules: {result.matched_rules}")
    assert result.response_level >= ResponseLevel.ALERT.value, "Should trigger alert"

    # Test response generation
    response = router.get_response(result)
    assert response['notify'] == True
    assert response.get('message') is not None
    print(f"  ✓ Response: notify={response['notify']}, message='{response.get('message', '')[:50]}...'")

    # Test low-significance deviation in different context
    personal_context = router.detect_context({
        'entity_relationship': 'peer',
        'keywords': []
    })
    low_deviation = Deviation(
        deviation_type="inform",
        source="ally_person",
        magnitude=0.2,
        description="FYI message"
    )
    low_result = router.score_deviation(low_deviation, personal_context, {'threat_level': 0})
    # With no matching rules and low threat, should be low response
    print(f"  ✓ Low deviation handled: level={ResponseLevel(low_result.response_level).name}, score={low_result.score:.2f}")

    print("All context_router tests passed! ✓")


if __name__ == "__main__":
    self_test()
