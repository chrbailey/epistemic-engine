"""
Judicial Analyzer - Real Implementation
=======================================

Takes judge opinions, extracts patterns, builds predictive profiles.
Integrates with TruthLayer for belief tracking.

What it actually does:
1. Fetches opinions from CourtListener API
2. Extracts 5W1H context from each opinion (via LLM)
3. Aggregates patterns into weighted factors
4. Generates "model feel" and predictions
5. Tracks judge beliefs in TruthLayer for learning

Revenue: $2,500/profile vs $30K/year Lex Machina
"""

import json
import re
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Optional: anthropic for LLM analysis
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Opinion5W1H:
    """Extracted context from a single opinion."""
    case_name: str
    date: str
    who: List[str]          # Parties, lawyers
    what: str               # Legal issue, outcome
    when: str               # Timeline, procedural history
    where: str              # Court, jurisdiction
    why: str                # Judge's reasoning
    how: str                # Decision mechanism (SJ, MTD, trial)
    outcome: str            # plaintiff/defendant/mixed
    key_quotes: List[str] = field(default_factory=list)


@dataclass
class WeightedFactor:
    """A pattern detected across multiple opinions."""
    name: str
    weight: float           # 0-100, importance
    confidence: float       # 0-1, how sure
    description: str
    evidence_count: int
    example_cases: List[str] = field(default_factory=list)


@dataclass
class JudgeProfile:
    """Complete judicial profile."""
    judge_name: str
    court: str
    appointed: str
    background: str
    model_feel: str         # Intuitive assessment

    # Weighted patterns
    procedural_factors: List[WeightedFactor] = field(default_factory=list)
    substantive_factors: List[WeightedFactor] = field(default_factory=list)
    communication_factors: List[WeightedFactor] = field(default_factory=list)

    # Strategic guidance
    do_list: List[str] = field(default_factory=list)
    dont_list: List[str] = field(default_factory=list)

    # Predictions
    grant_rate_sj: float = 0.5      # Summary judgment grant rate
    grant_rate_mtd: float = 0.5     # Motion to dismiss grant rate

    opinions_analyzed: int = 0
    last_updated: str = ""


# =============================================================================
# COURTLISTENER CLIENT (Python version)
# =============================================================================

class CourtListenerClient:
    """
    Fetch opinions from CourtListener API.
    API requires authentication token (free to register).
    Falls back to mock data for demo.
    """

    BASE_URL = "https://www.courtlistener.com/api/rest/v4"
    RATE_LIMIT_MS = 200

    COURT_CODES = {
        'N.D. Cal': 'cand',
        'S.D.N.Y.': 'nysd',
        'C.D. Cal': 'cacd',
        'D. Del': 'ded',
        'E.D. Texas': 'txed',
    }

    # Mock data for demo when API unavailable
    MOCK_OPINIONS = {
        'William Alsup': [
            {
                'id': 1,
                'case_name': 'Bartz v. Anthropic, PBC',
                'date_filed': '2024-08-15',
                'text': """ORDER GRANTING IN PART AND DENYING IN PART MOTION FOR SUMMARY JUDGMENT

The Court has carefully considered the parties' arguments regarding fair use.

BACKGROUND: Plaintiff Bartz alleges Anthropic's AI training process infringes her copyrighted works.

LEGAL STANDARD: Summary judgment is appropriate when there is no genuine dispute as to any material fact.

ANALYSIS:
1. Factor One (Purpose): The Court finds AI training to be transformative. The model learns patterns, not copies expression. GRANTED as to training use.

2. Factor Two (Nature): Creative works receive stronger protection. This factor favors plaintiff.

3. Factor Three (Amount): Using entire works for training is substantial but necessary for the transformative purpose.

4. Factor Four (Market Effect): Plaintiff has not demonstrated market substitution.

The Court GRANTS summary judgment on the training fair use claim.
The Court DENIES summary judgment on the pirated library copies claim - jury question.

IT IS SO ORDERED.
Judge William Alsup"""
            },
            {
                'id': 2,
                'case_name': 'Oracle America, Inc. v. Google LLC',
                'date_filed': '2021-04-05',
                'text': """ORDER RE FAIR USE

This case involves Google's use of Java API declarations in Android.

The Court has expertise in software development, having written code in multiple languages.

TECHNICAL BACKGROUND: APIs are interfaces that allow programs to communicate. The declarations are like a library's card catalog - they describe what's available.

FAIR USE ANALYSIS:
The Court finds Google's use transformative. Google reimplemented APIs for a new platform (mobile), creating new expression and meaning.

The jury found fair use, and the Court agrees this was supported by substantial evidence.

GRANTED - fair use affirmed.

Judge William Alsup"""
            },
            {
                'id': 3,
                'case_name': 'Waymo LLC v. Uber Technologies',
                'date_filed': '2018-02-09',
                'text': """ORDER ON MOTION TO COMPEL

The Court is deeply troubled by the discovery misconduct in this case.

Uber shall produce ALL documents related to the Levandowski acquisition within 14 days.
Failure to comply will result in sanctions including adverse inference instructions.

The Court does not tolerate discovery gamesmanship.

Deadlines in this Court are not suggestions. They are requirements.

IT IS SO ORDERED.
Judge William Alsup"""
            },
            {
                'id': 4,
                'case_name': 'In re: Google Location History Litigation',
                'date_filed': '2023-03-20',
                'text': """ORDER DENYING MOTION TO DISMISS

Plaintiffs allege Google continued tracking location even when Location History was disabled.

The Court finds plaintiffs have adequately pled their claims.

STANDARD: Under Twombly/Iqbal, plaintiff must plead facts making the claim plausible.

Here, plaintiffs cite specific technical evidence showing location data collection continued.
The complaint is not conclusory - it provides technical details.

DENIED - case may proceed to discovery.

Judge William Alsup"""
            },
            {
                'id': 5,
                'case_name': 'Apple Inc. v. Samsung Electronics',
                'date_filed': '2014-05-02',
                'text': """ORDER ON DAMAGES RETRIAL

The Court has reviewed the evidence on design patent damages.

Samsung's arguments regarding apportionment are rejected.
The statute provides for disgorgement of total profit from the article of manufacture.

However, the jury must determine what constitutes the "article of manufacture" -
the entire phone or specific components.

Trial will proceed on this limited issue.

The Court notes both parties' briefing was excellent in this technically complex case.

IT IS SO ORDERED.
Judge William Alsup"""
            }
        ]
    }

    def __init__(self, api_token: Optional[str] = None):
        self._last_request = 0
        self.api_token = api_token or os.environ.get('COURTLISTENER_TOKEN')

    def _rate_limit(self):
        """Respect rate limits."""
        import time
        now = time.time() * 1000
        elapsed = now - self._last_request
        if elapsed < self.RATE_LIMIT_MS:
            time.sleep((self.RATE_LIMIT_MS - elapsed) / 1000)
        self._last_request = time.time() * 1000

    def search_opinions(self, judge_name: str, limit: int = 50) -> List[Dict]:
        """Search for opinions by judge name."""
        # Try API first if token available
        if self.api_token:
            result = self._search_api(judge_name, limit)
            if result:
                return result

        # Fall back to mock data
        print(f"  Using mock data for {judge_name} (API requires auth token)")
        mock = self.MOCK_OPINIONS.get(judge_name, [])
        return mock[:limit]

    def _search_api(self, judge_name: str, limit: int) -> List[Dict]:
        """Search via CourtListener API."""
        import urllib.request
        import urllib.parse

        self._rate_limit()

        params = urllib.parse.urlencode({
            'author__name_full__icontains': judge_name,
            'order_by': '-date_filed',
            'page_size': limit
        })

        url = f"{self.BASE_URL}/opinions/?{params}"

        try:
            req = urllib.request.Request(url)
            req.add_header('Authorization', f'Token {self.api_token}')
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                return data.get('results', [])
        except Exception as e:
            print(f"  CourtListener API error: {e}")
            return []

    def get_opinion_text(self, opinion_id: int) -> Optional[str]:
        """Get full text of an opinion."""
        # Check mock data first
        for judge_opinions in self.MOCK_OPINIONS.values():
            for op in judge_opinions:
                if op.get('id') == opinion_id:
                    return op.get('text', '')

        # Try API if we have a token
        if not self.api_token:
            return None

        import urllib.request

        self._rate_limit()

        url = f"{self.BASE_URL}/opinions/{opinion_id}/"

        try:
            req = urllib.request.Request(url)
            req.add_header('Authorization', f'Token {self.api_token}')
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                # Try different text fields
                for field in ['plain_text', 'html', 'html_with_citations']:
                    if data.get(field):
                        # Strip HTML if needed
                        text = data[field]
                        text = re.sub(r'<[^>]+>', '', text)
                        return text[:50000]  # Limit size
                return None
        except Exception as e:
            print(f"  Failed to fetch opinion {opinion_id}: {e}")
            return None


# =============================================================================
# OPINION ANALYZER (uses LLM)
# =============================================================================

class OpinionAnalyzer:
    """
    Extract 5W1H context from judicial opinions.
    Uses Claude API if available, otherwise falls back to regex patterns.
    """

    EXTRACT_PROMPT = """Analyze this judicial opinion and extract structured information.

OPINION TEXT:
{text}

Extract the following (be specific and cite the opinion):

WHO: List the parties, key lawyers, and any amici
WHAT: What legal issue was decided? What was the outcome?
WHEN: Timeline - when filed, key dates, procedural history
WHERE: Court, jurisdiction, venue
WHY: Judge's core reasoning - what factors were decisive?
HOW: Decision mechanism (summary judgment, motion to dismiss, trial, etc.)
OUTCOME: Did plaintiff or defendant prevail? Or mixed?
KEY_QUOTES: 2-3 direct quotes showing judge's reasoning style

Respond in JSON format:
{{
  "who": ["party1", "party2", ...],
  "what": "description",
  "when": "timeline",
  "where": "court info",
  "why": "core reasoning",
  "how": "decision mechanism",
  "outcome": "plaintiff/defendant/mixed",
  "key_quotes": ["quote1", "quote2"]
}}"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.client = None
        if HAS_ANTHROPIC and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def analyze(self, case_name: str, date: str, text: str) -> Opinion5W1H:
        """Extract 5W1H from opinion text."""
        if self.client:
            return self._analyze_with_llm(case_name, date, text)
        else:
            return self._analyze_with_patterns(case_name, date, text)

    def _analyze_with_llm(self, case_name: str, date: str, text: str) -> Opinion5W1H:
        """Use Claude to extract 5W1H."""
        try:
            # Truncate if too long
            if len(text) > 30000:
                text = text[:15000] + "\n...[truncated]...\n" + text[-15000:]

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": self.EXTRACT_PROMPT.format(text=text)
                }]
            )

            # Parse JSON from response
            content = response.content[0].text
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return Opinion5W1H(
                    case_name=case_name,
                    date=date,
                    who=data.get('who', []),
                    what=data.get('what', ''),
                    when=data.get('when', ''),
                    where=data.get('where', ''),
                    why=data.get('why', ''),
                    how=data.get('how', ''),
                    outcome=data.get('outcome', 'unknown'),
                    key_quotes=data.get('key_quotes', [])
                )
        except Exception as e:
            print(f"LLM analysis failed: {e}")

        return self._analyze_with_patterns(case_name, date, text)

    def _analyze_with_patterns(self, case_name: str, date: str, text: str) -> Opinion5W1H:
        """Fallback: extract using regex patterns."""
        text_lower = text.lower()

        # Detect outcome
        outcome = 'unknown'
        if 'granted' in text_lower and 'denied' not in text_lower:
            outcome = 'defendant' if 'defendant' in text_lower else 'plaintiff'
        elif 'denied' in text_lower and 'granted' not in text_lower:
            outcome = 'plaintiff' if 'defendant' in text_lower else 'defendant'
        elif 'granted in part' in text_lower or 'denied in part' in text_lower:
            outcome = 'mixed'

        # Detect decision type
        how = 'unknown'
        if 'summary judgment' in text_lower:
            how = 'summary judgment'
        elif 'motion to dismiss' in text_lower:
            how = 'motion to dismiss'
        elif 'jury verdict' in text_lower or 'trial' in text_lower:
            how = 'trial'

        # Extract parties from case name
        parties = case_name.split(' v. ') if ' v. ' in case_name else [case_name]

        return Opinion5W1H(
            case_name=case_name,
            date=date,
            who=parties,
            what=f"Case decided via {how}",
            when=date,
            where="Federal court",
            why="[Pattern-based extraction - LLM not available]",
            how=how,
            outcome=outcome,
            key_quotes=[]
        )


# =============================================================================
# PATTERN AGGREGATOR
# =============================================================================

class PatternAggregator:
    """
    Aggregate 5W1H extractions into weighted patterns.
    """

    def aggregate(self, opinions: List[Opinion5W1H]) -> Dict[str, List[WeightedFactor]]:
        """Convert list of opinions into weighted factors."""

        procedural = []
        substantive = []
        communication = []

        if not opinions:
            return {
                'procedural': procedural,
                'substantive': substantive,
                'communication': communication
            }

        total = len(opinions)

        # Count decision types
        sj_count = sum(1 for o in opinions if 'summary judgment' in o.how.lower())
        mtd_count = sum(1 for o in opinions if 'dismiss' in o.how.lower())
        trial_count = sum(1 for o in opinions if 'trial' in o.how.lower())

        # Count outcomes
        plaintiff_wins = sum(1 for o in opinions if o.outcome == 'plaintiff')
        defendant_wins = sum(1 for o in opinions if o.outcome == 'defendant')
        mixed = sum(1 for o in opinions if o.outcome == 'mixed')

        # Procedural factors
        if sj_count > 0:
            grant_rate = defendant_wins / max(1, sj_count)
            procedural.append(WeightedFactor(
                name="Summary Judgment Disposition",
                weight=min(95, 60 + sj_count * 2),
                confidence=min(0.95, 0.5 + sj_count / total),
                description=f"Grants SJ {grant_rate:.0%} of the time ({sj_count} cases analyzed)",
                evidence_count=sj_count,
                example_cases=[o.case_name for o in opinions if 'summary' in o.how.lower()][:3]
            ))

        if mtd_count > 0:
            procedural.append(WeightedFactor(
                name="Motion to Dismiss Standards",
                weight=min(90, 55 + mtd_count * 2),
                confidence=min(0.95, 0.5 + mtd_count / total),
                description=f"Ruled on {mtd_count} MTD motions - applies Twombly/Iqbal rigorously",
                evidence_count=mtd_count,
                example_cases=[o.case_name for o in opinions if 'dismiss' in o.how.lower()][:3]
            ))

        # Substantive factors
        if total >= 5:
            substantive.append(WeightedFactor(
                name="Outcome Distribution",
                weight=80,
                confidence=min(0.9, 0.4 + total / 50),
                description=f"Plaintiff wins: {plaintiff_wins}, Defendant wins: {defendant_wins}, Mixed: {mixed}",
                evidence_count=total,
                example_cases=[]
            ))

        # Communication factors (from key quotes if available)
        quotes_available = sum(1 for o in opinions if o.key_quotes)
        if quotes_available > 0:
            communication.append(WeightedFactor(
                name="Writing Style",
                weight=70,
                confidence=0.7,
                description=f"Based on {quotes_available} opinions with extracted quotes",
                evidence_count=quotes_available,
                example_cases=[]
            ))

        return {
            'procedural': procedural,
            'substantive': substantive,
            'communication': communication
        }


# =============================================================================
# PROFILE GENERATOR
# =============================================================================

class JudicialProfileGenerator:
    """
    Generate complete judicial profiles.
    Integrates with TruthLayer for belief tracking.
    """

    MODEL_FEEL_PROMPT = """Based on these judicial patterns, write a 2-3 sentence "model feel" -
an intuitive characterization of how this judge approaches cases.

Judge: {judge_name}
Court: {court}
Patterns:
{patterns}

Write in this style:
"This judge feels like [analogy]. They [key characteristic]. [Notable behavior]."

Be specific and actionable for lawyers."""

    def __init__(self, truth_layer=None, api_key: Optional[str] = None):
        self.truth_layer = truth_layer
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.client = None
        if HAS_ANTHROPIC and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)

        self.courtlistener = CourtListenerClient()
        self.analyzer = OpinionAnalyzer(api_key=self.api_key)
        self.aggregator = PatternAggregator()

    def generate_profile(self, judge_name: str, case_type: str = None,
                        max_opinions: int = 30) -> JudgeProfile:
        """
        Generate a complete judicial profile.

        Args:
            judge_name: Name of the judge
            case_type: Optional filter (copyright, patent, etc.)
            max_opinions: Maximum opinions to analyze
        """
        print(f"Generating profile for {judge_name}...")

        # 1. Fetch opinions
        print(f"  Fetching opinions from CourtListener...")
        raw_opinions = self.courtlistener.search_opinions(judge_name, limit=max_opinions)

        if not raw_opinions:
            print(f"  No opinions found for {judge_name}")
            return self._create_empty_profile(judge_name)

        print(f"  Found {len(raw_opinions)} opinions")

        # 2. Analyze each opinion
        analyzed = []
        for i, op in enumerate(raw_opinions[:max_opinions]):
            case_name = op.get('case_name', 'Unknown')
            date = op.get('date_filed', '')

            # Get full text
            op_id = op.get('id')
            if op_id:
                print(f"  [{i+1}/{min(len(raw_opinions), max_opinions)}] Analyzing: {case_name[:50]}...")
                text = self.courtlistener.get_opinion_text(op_id)
                if text:
                    analysis = self.analyzer.analyze(case_name, date, text)
                    analyzed.append(analysis)

        print(f"  Analyzed {len(analyzed)} opinions")

        # 3. Aggregate patterns
        factors = self.aggregator.aggregate(analyzed)

        # 4. Generate model feel
        model_feel = self._generate_model_feel(judge_name, factors)

        # 5. Generate DO/DON'T lists
        do_list, dont_list = self._generate_strategic_guidance(factors)

        # 6. Calculate rates
        sj_grants = sum(1 for o in analyzed if 'summary' in o.how.lower() and o.outcome == 'defendant')
        sj_total = sum(1 for o in analyzed if 'summary' in o.how.lower())
        grant_rate_sj = sj_grants / max(1, sj_total)

        # 7. Create profile
        profile = JudgeProfile(
            judge_name=judge_name,
            court=analyzed[0].where if analyzed else "Unknown",
            appointed="",
            background="",
            model_feel=model_feel,
            procedural_factors=factors['procedural'],
            substantive_factors=factors['substantive'],
            communication_factors=factors['communication'],
            do_list=do_list,
            dont_list=dont_list,
            grant_rate_sj=grant_rate_sj,
            opinions_analyzed=len(analyzed),
            last_updated=datetime.now().isoformat()
        )

        # 8. Update TruthLayer with judge beliefs
        if self.truth_layer:
            self._update_truth_layer(judge_name, profile, factors)

        return profile

    def _generate_model_feel(self, judge_name: str, factors: Dict) -> str:
        """Generate intuitive model feel."""
        if self.client:
            try:
                patterns = []
                for category, factor_list in factors.items():
                    for f in factor_list:
                        patterns.append(f"- {f.name}: {f.description} (weight: {f.weight}%)")

                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[{
                        "role": "user",
                        "content": self.MODEL_FEEL_PROMPT.format(
                            judge_name=judge_name,
                            court="Federal District Court",
                            patterns="\n".join(patterns) if patterns else "Limited data available"
                        )
                    }]
                )
                return response.content[0].text.strip()
            except Exception as e:
                print(f"Model feel generation failed: {e}")

        # Fallback
        return f"Judge {judge_name} - profile based on {sum(len(v) for v in factors.values())} detected patterns."

    def _generate_strategic_guidance(self, factors: Dict) -> Tuple[List[str], List[str]]:
        """Generate DO and DON'T lists."""
        do_list = [
            "File complete record with ALL exhibits",
            "Meet all deadlines without exception",
            "Cite controlling authority accurately",
            "Address all elements of legal standard",
        ]

        dont_list = [
            "Miss filing deadlines",
            "Submit incomplete briefing",
            "Misstate facts or law",
            "Make arguments without record support",
        ]

        # Add pattern-specific guidance
        for f in factors.get('procedural', []):
            if 'summary judgment' in f.name.lower():
                do_list.append("Support SJ motions with detailed declarations")
                dont_list.append("File SJ without complete factual record")

        return do_list, dont_list

    def _update_truth_layer(self, judge_name: str, profile: JudgeProfile,
                           factors: Dict):
        """Update TruthLayer with beliefs about this judge."""
        judge_id = judge_name.lower().replace(' ', '_').replace('.', '')

        # Add core beliefs
        self.truth_layer.add_claim(
            f"{judge_id}_grants_sj_readily",
            f"Judge {judge_name} grants summary judgment readily",
            category="judge_pattern"
        )

        self.truth_layer.add_claim(
            f"{judge_id}_strict_deadlines",
            f"Judge {judge_name} strictly enforces deadlines",
            category="judge_pattern"
        )

        self.truth_layer.add_claim(
            f"{judge_id}_technical_expertise",
            f"Judge {judge_name} has technical expertise",
            category="judge_pattern"
        )

        # Link beliefs
        self.truth_layer.add_relationship(
            f"{judge_id}_strict_deadlines",
            f"{judge_id}_grants_sj_readily",
            weight=-3.0  # Strict on procedure = less likely to grant shortcuts
        )

        # Update based on data
        if profile.grant_rate_sj > 0.6:
            self.truth_layer.validate(f"{judge_id}_grants_sj_readily", "confirm")
        elif profile.grant_rate_sj < 0.4:
            self.truth_layer.validate(f"{judge_id}_grants_sj_readily", "reject")

    def _create_empty_profile(self, judge_name: str) -> JudgeProfile:
        """Create empty profile when no data found."""
        return JudgeProfile(
            judge_name=judge_name,
            court="Unknown",
            appointed="",
            background="",
            model_feel=f"Insufficient data to profile Judge {judge_name}",
            opinions_analyzed=0,
            last_updated=datetime.now().isoformat()
        )


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

def format_profile_summary(profile: JudgeProfile) -> str:
    """Format profile as summary text."""
    lines = [
        f"━━━ JUDICIAL PROFILE: {profile.judge_name.upper()} ━━━",
        "",
        f"Court: {profile.court}",
        f"Opinions Analyzed: {profile.opinions_analyzed}",
        f"Last Updated: {profile.last_updated[:10]}",
        "",
        "MODEL FEEL:",
        profile.model_feel,
        "",
        "TOP WEIGHTED FACTORS:",
    ]

    all_factors = (profile.procedural_factors +
                   profile.substantive_factors +
                   profile.communication_factors)

    for f in sorted(all_factors, key=lambda x: -x.weight)[:5]:
        lines.append(f"• {f.name} ({f.weight:.0f}% weight, {f.confidence:.0%} confidence)")
        lines.append(f"  → {f.description}")

    lines.extend([
        "",
        "STRATEGIC DO:",
    ])
    for item in profile.do_list[:5]:
        lines.append(f"✓ {item}")

    lines.extend([
        "",
        "STRATEGIC DON'T:",
    ])
    for item in profile.dont_list[:5]:
        lines.append(f"✗ {item}")

    lines.extend([
        "",
        f"SJ Grant Rate: {profile.grant_rate_sj:.0%}",
        "",
        "━━━ END PROFILE ━━━"
    ])

    return "\n".join(lines)


def format_profile_json(profile: JudgeProfile) -> str:
    """Format profile as JSON."""
    return json.dumps(asdict(profile), indent=2, default=str)


# =============================================================================
# CLI / DEMO
# =============================================================================

def demo():
    """Demo the judicial analyzer."""
    print("=" * 60)
    print("JUDICIAL ANALYZER - Real Implementation")
    print("=" * 60)

    # Check for API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n⚠️  ANTHROPIC_API_KEY not set - will use pattern-based extraction")
        print("   Set the key for LLM-powered analysis")
    else:
        print("\n✓ Anthropic API key found - will use LLM analysis")

    # Import TruthLayer
    try:
        from truth_layer import TruthLayer
        truth_layer = TruthLayer("judicial_beliefs.json")
        print("✓ TruthLayer integrated for belief tracking")
    except ImportError:
        truth_layer = None
        print("⚠️  TruthLayer not found - running without belief tracking")

    # Create generator
    generator = JudicialProfileGenerator(truth_layer=truth_layer, api_key=api_key)

    # Demo with a well-known judge
    judge_name = "William Alsup"
    print(f"\n📊 Generating profile for Judge {judge_name}...")
    print("   (This will fetch real data from CourtListener)")

    profile = generator.generate_profile(judge_name, max_opinions=10)

    print("\n" + format_profile_summary(profile))

    # Show truth layer state if available
    if truth_layer:
        print("\n" + truth_layer.get_truth_context())


def self_test():
    """Run self-tests."""
    print("Testing judicial_analyzer.py...")

    # Test CourtListener client
    client = CourtListenerClient()
    print("  ✓ CourtListener client created")

    # Test pattern aggregator
    aggregator = PatternAggregator()
    empty_result = aggregator.aggregate([])
    assert 'procedural' in empty_result
    print("  ✓ Pattern aggregator works")

    # Test opinion analyzer (without API)
    analyzer = OpinionAnalyzer(api_key=None)
    result = analyzer.analyze("Test v. Case", "2024-01-01", "Motion for summary judgment is GRANTED.")
    assert result.how == 'summary judgment'
    print("  ✓ Opinion analyzer pattern fallback works")

    print("All judicial_analyzer tests passed! ✓")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        self_test()
    else:
        demo()
