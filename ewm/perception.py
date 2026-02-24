"""
Epistemic World Model — Perception Module
==========================================

The Perception module is the sensory interface of the LeCun 6-module
cognitive architecture.  It takes raw unstructured text and extracts
structured knowledge: entities, claims, relationships, and evidence.

Two extraction backends are available:

    1. **Regex** (default) — pure pattern matching with ZERO external
       dependencies.  Fast, deterministic, and fully auditable.  Lower
       recall compared to ML-based NER.

    2. **LLM** (optional) — when configured via ``configure_llm()``, the
       module calls a local Ollama instance or any OpenAI-compatible
       endpoint to perform entity/claim/relationship extraction.  Falls
       back to regex automatically on any failure.

Pipeline within perceive():
    1. If LLM enabled, attempt LLM-based extraction (fall back to regex on failure)
    2. Detect and redact sensitive data (PII, secrets, credentials) — always regex
    3. Extract entities (organizations, people, tech, financial, locations, events)
    4. Extract claims (sentence-level propositions with semantic typing)
    5. Extract relationships (directed edges between co-occurring entities)
    6. Create evidence record from the raw input
    7. Assemble and return PerceptionResult

Design principles:
    - Sensitive data detection ALWAYS uses regex (never send PII to an LLM)
    - LLM path is fully optional — the system works identically without it
    - stdlib only — LLM calls use urllib.request, no third-party HTTP libraries
    - Conservative extraction: prefer precision over recall
    - Regexes compiled at module level for performance
    - Private helpers prefixed with underscore
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from ewm.types import (
    Claim,
    ClaimType,
    Entity,
    EntityCategory,
    Evidence,
    PerceptionResult,
    Relationship,
    RelationshipType,
    SourceType,
    Uncertainty,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Generate a new UUID4 string identifier."""
    return str(uuid4())


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Configuration & Extraction Stats
# ---------------------------------------------------------------------------

LLM_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "provider": "ollama",              # "ollama" or "openai_compatible"
    "base_url": "http://localhost:11434",  # Ollama default
    "model": "llama3.2",              # or any installed model
    "api_key": "",                     # for openai-compatible endpoints
    "timeout": 30,                     # seconds
    "temperature": 0.1,               # low for extraction
}

_EXTRACTION_STATS: Dict[str, Any] = {
    "last_method": "none",
    "last_entity_count": 0,
    "last_claim_count": 0,
    "last_relationship_count": 0,
    "llm_calls": 0,
    "llm_failures": 0,
    "regex_calls": 0,
}


# ---------------------------------------------------------------------------
# Module-level compiled regex patterns
# ---------------------------------------------------------------------------

# -- Entity extraction patterns --

_ORG_SUFFIXES = (
    r"(?:Corp(?:oration)?|Inc(?:orporated)?|Ltd|LLC|Co(?:mpany)?|Group|Holdings|"
    r"Technologies|Systems|Partners|Associates|Foundation|Institute)"
)
_RE_ORG = re.compile(
    r"\b([A-Z][a-zA-Z&'-]+(?:\s+[A-Z][a-zA-Z&'-]+)*\s+" + _ORG_SUFFIXES + r")\b"
)

_PERSON_STOP = frozenset({
    "The", "And", "Inc", "Corp", "Ltd", "LLC", "Co", "Group", "Holdings",
    "Technologies", "Systems", "Partners", "Associates", "Foundation",
    "Institute", "Electronics", "Software", "Solutions", "Services",
    "Industries", "Enterprises", "Communications", "Pharmaceuticals",
    "Semiconductor", "Dynamics", "Instruments", "Networks", "Capital",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October", "November", "December",
    "North", "South", "East", "West", "New", "San", "Los", "Las", "United",
    "States", "National", "International", "Global", "American", "European",
    "Pacific", "Atlantic", "Revenue", "Growth", "Quarter", "Annual",
    "Report", "Fiscal", "Stock", "Market", "API", "SDK",
})

_ROLE_WORDS = frozenset({
    "ceo", "cto", "cfo", "coo", "cio", "ciso", "vp", "svp", "evp",
    "founder", "cofounder", "co-founder", "director", "manager", "president",
    "chairman", "chairwoman", "chairperson", "professor", "dr", "mr", "mrs",
    "ms", "chief", "head", "lead", "officer", "executive", "partner",
    "analyst", "engineer", "architect", "scientist", "researcher",
    "secretary", "treasurer", "governor", "senator", "representative",
    "minister", "ambassador",
})
_RE_PERSON_CANDIDATE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)

_TECH_KEYWORDS = frozenset({
    "Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C++",
    "React", "Vue", "Angular", "Node.js", "Docker", "Kubernetes",
    "AWS", "Azure", "GCP", "PostgreSQL", "MySQL", "MongoDB", "Redis",
    "GraphQL", "REST", "API", "SDK", "ML", "AI", "NLP", "LLM", "GPT",
    "Claude", "SAP", "ERP", "HANA", "ABAP", "Terraform", "Linux",
    "Windows", "macOS", "iOS", "Android",
})
# Build a regex that matches any tech keyword as a whole word.
# Sort by length descending so longer matches are preferred (e.g. "Node.js" before "Node").
_tech_sorted = sorted(_TECH_KEYWORDS, key=len, reverse=True)
_RE_TECH = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _tech_sorted) + r")(?:\b|(?=\s|$|[,;:!?)\]]))"
)

_RE_CURRENCY = re.compile(
    r"(?:[$\u20ac\u00a3])[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|[MBT]))?|"
    r"[\d,]+(?:\.\d+)?\s*(?:USD|EUR|GBP|JPY|CHF|CAD|AUD)"
)
_RE_PERCENTAGE = re.compile(r"\b\d+(?:\.\d+)?%")
_RE_TICKER_CONTEXT = re.compile(
    r"(?:[Tt]icker|[Ss]tock|[Ss]hares?|[Ss]ymbol|NYSE|NASDAQ|[Ll]isted)\s*(?:[:=])?\s*([A-Z]{1,5})\b"
)
_RE_CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")

_LOCATION_PREPS = re.compile(
    r"\b(?:in|based\s+in|headquartered\s+in|located\s+in|from|near)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
)

_RE_EVENT_QUARTER = re.compile(r"\bQ[1-4]\s*\d{4}\b")
_RE_EVENT_FY = re.compile(r"\bFY\s*\d{4}\b")
_RE_EVENT_KEYWORDS = re.compile(
    r"\b(conference|summit|IPO|acquisition|merger|bankruptcy|layoff|"
    r"restructuring|spinoff|spin-off|divestiture|launch|announcement)\b",
    re.IGNORECASE,
)
_RE_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4})\b"
)


# -- Claim classification patterns --

_CLAIM_STATISTICAL = re.compile(
    r"\d+|%|\$|revenue|growth|increased|decreased|million|billion|trillion|"
    r"percent|average|median|total|quarterly|annually",
    re.IGNORECASE,
)
_CLAIM_CAUSAL = re.compile(
    r"\bbecause\b|\bcaused\b|\bled\s+to\b|\bresulted\s+in\b|\bdue\s+to\b|"
    r"\bas\s+a\s+result\b",
    re.IGNORECASE,
)
_CLAIM_PREDICTIVE = re.compile(
    r"\bwill\b|\bforecast\b|\bexpect(?:s|ed)?\b|\bpredict(?:s|ed)?\b|"
    r"\banticipate[ds]?\b|\bproject(?:s|ed)?\b|\b20[3-9]\d\b",
    re.IGNORECASE,
)
_CLAIM_ACCUSATORY = re.compile(
    r"\bblame[ds]?\b|\bfault\b|\bresponsible\s+for\b|\bcaused\s+the\b|"
    r"\bnegligent\b|\bnegligence\b",
    re.IGNORECASE,
)
_CLAIM_DIAGNOSTIC = re.compile(
    r"\broot\s+cause\b|\bdiagnos(?:is|ed|tic)\b|\bidentified\s+as\b|"
    r"\bdetermined\s+to\s+be\b|\bthe\s+issue\s+is\b|\bthe\s+problem\s+is\b",
    re.IGNORECASE,
)
_CLAIM_PRESCRIPTIVE = re.compile(
    r"\bshould\b|\bmust\b|\brecommend[s]?\b|\bneed\s+to\b|\bought\s+to\b",
    re.IGNORECASE,
)


# -- Relationship signal patterns --
# Each tuple: (compiled_regex, RelationshipType)

_RELATIONSHIP_SIGNALS: list[tuple[re.Pattern[str], RelationshipType]] = [
    (re.compile(r"\bacquired\b|\bbought\b|\bmerged\s+with\b", re.IGNORECASE), RelationshipType.OWNS),
    (re.compile(r"\bworks\s+at\b|\bjoined\b|\bhired\b|\bemployed\s+by\b|\bemploys\b", re.IGNORECASE), RelationshipType.EMPLOYS),
    (re.compile(r"\buses\b|\bbuilt\s+with\b|\bpowered\s+by\b|\bruns\s+on\b|\badopted\b", re.IGNORECASE), RelationshipType.USES),
    (re.compile(r"\bmakes\b|\bproduces\b|\bsells\b|\boffers\b|\blaunched\b|\bdeveloped\b", re.IGNORECASE), RelationshipType.PRODUCES),
    (re.compile(r"\bdepends\s+on\b|\brelies\s+on\b|\brequires\b|\bneeds\b", re.IGNORECASE), RelationshipType.DEPENDS_ON),
    (re.compile(r"\bcompetes\s+with\b|\brival\b|\bcompetitor\b", re.IGNORECASE), RelationshipType.COMPETES_WITH),
    (re.compile(r"\bregulates\b|\boversees\b|\bgoverns\b|\bapproved\s+by\b", re.IGNORECASE), RelationshipType.REGULATES),
    (re.compile(r"\bpart\s+of\b|\bdivision\s+of\b|\bsubsidiary\s+of\b|\bwithin\b", re.IGNORECASE), RelationshipType.PART_OF),
    (re.compile(r"\blocated\s+in\b|\bbased\s+in\b|\bheadquartered\s+in\b", re.IGNORECASE), RelationshipType.LOCATED_IN),
    (re.compile(r"\bcaused\b|\bled\s+to\b|\btriggered\b", re.IGNORECASE), RelationshipType.CAUSES),
    (re.compile(r"\bbefore\b|\bpreceded\b|\bfollowed\s+by\b|\bafter\b", re.IGNORECASE), RelationshipType.PRECEDED_BY),
    (re.compile(r"\bsimilar\s+to\b|\blike\b|\bcomparable\s+to\b", re.IGNORECASE), RelationshipType.SIMILAR_TO),
]


# -- Sensitive data detection patterns --
# Each tuple: (compiled_regex, type_label)

_SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Order matters: more specific patterns first to avoid partial matches
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "EMAIL"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    (re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b"), "CREDIT_CARD"),
    (re.compile(r"(?:AKIA)[A-Za-z0-9]{16}"), "AWS_KEY"),
    (re.compile(r"-----BEGIN\s+(?:RSA\s+|EC\s+)?PRIVATE\s+KEY-----"), "PRIVATE_KEY"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9_\-\.]{20,}\b"), "BEARER_TOKEN"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"), "JWT"),
    (re.compile(
        r"(?i)\b(?:password)\s*[:=]\s*[\"']?([^\s\"',;]{3,})[\"']?"
    ), "PASSWORD"),
    (re.compile(
        r"(?i)\b(?:secret|token)\s*[:=]\s*[\"']?([^\s\"',;]{3,})[\"']?"
    ), "SECRET"),
    (re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"), "API_KEY"),
    (re.compile(r"\b(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b"), "PHONE"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "IP_ADDRESS"),
    (re.compile(
        r"(?i)(?:DOB|date\s+of\s+birth)\s*[:=]?\s*"
        r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4})"
    ), "DATE_OF_BIRTH"),
    (re.compile(
        r"(?i)\bpassport\s*(?:number|no|#|:)?\s*[A-Za-z0-9]{6,12}\b"
    ), "PASSPORT"),
    (re.compile(
        r"(?i)\bdriver(?:'?s)?\s+licen[sc]e\s*(?:number|no|#|:)?\s*[A-Za-z0-9]{5,15}\b"
    ), "DRIVER_LICENSE"),
    (re.compile(
        r"(?i)\b(?:account|acct)\s*(?:number|no|#|:)?\s*\d{8,17}\b"
    ), "BANK_ACCOUNT"),
]


# -- Sentence splitting --

_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])(?:\s+|$)")


# ===========================================================================
# LLM Extraction Prompt
# ===========================================================================

_LLM_EXTRACTION_PROMPT = """\
You are an entity and claim extraction system. Extract structured information from the following text.

Return a JSON object with:
- "entities": [{"name": str, "category": "person"|"organization"|"technology"|"concept"|"location"|"event"|"artifact"|"financial", "aliases": [str]}]
- "claims": [{"text": str, "claim_type": "factual"|"statistical"|"causal"|"predictive"|"accusatory"|"diagnostic"|"prescriptive"}]
- "relationships": [{"source": str, "target": str, "type": "owns"|"employs"|"uses"|"produces"|"depends_on"|"competes_with"|"regulates"|"part_of"|"located_in"|"causes"|"preceded_by"|"similar_to"}]

Text: {text}

Respond with ONLY valid JSON, no explanation."""


# ===========================================================================
# Public API
# ===========================================================================


def configure_llm(**kwargs: Any) -> None:
    """Configure the optional LLM extraction backend.

    Updates LLM_CONFIG from the provided keyword arguments, then tests
    connectivity by issuing a lightweight request to the endpoint.
    Sets ``LLM_CONFIG["enabled"]`` to ``True`` only if the connectivity
    test succeeds.

    Args:
        **kwargs: Any keys matching LLM_CONFIG (e.g. provider, base_url,
            model, api_key, timeout, temperature).
    """
    for key, value in kwargs.items():
        if key in LLM_CONFIG:
            LLM_CONFIG[key] = value

    # Test connectivity
    LLM_CONFIG["enabled"] = False
    provider = LLM_CONFIG["provider"]
    base_url = LLM_CONFIG["base_url"].rstrip("/")
    timeout = LLM_CONFIG.get("timeout", 30)

    if provider == "ollama":
        test_url = base_url + "/api/tags"
    else:
        test_url = base_url + "/v1/models"

    try:
        req = urllib.request.Request(test_url, method="GET")
        if provider == "openai_compatible" and LLM_CONFIG["api_key"]:
            req.add_header("Authorization", "Bearer " + LLM_CONFIG["api_key"])

        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=min(timeout, 10), context=ctx) as resp:
            if resp.status == 200:
                LLM_CONFIG["enabled"] = True
                _log.info(
                    "LLM extraction enabled: provider=%s, model=%s",
                    provider, LLM_CONFIG["model"],
                )
    except Exception as exc:
        _log.debug("LLM connectivity test failed: %s", exc)
        LLM_CONFIG["enabled"] = False


def extraction_stats() -> Dict[str, Any]:
    """Return statistics about the most recent extraction run.

    Returns:
        A dict with keys: last_method ("llm", "regex", or "none"),
        last_entity_count, last_claim_count, last_relationship_count,
        llm_calls, llm_failures, regex_calls.
    """
    return dict(_EXTRACTION_STATS)


def perceive(
    text: str,
    source_type: SourceType = SourceType.DOCUMENT,
    source_id: str = "",
    known_entities: list[Entity] | None = None,
) -> PerceptionResult:
    """Perceive structured knowledge from unstructured text.

    This is the main entry point of the Perception module. It orchestrates
    all extraction steps and returns a PerceptionResult containing entities,
    claims, relationships, evidence, and any redacted sensitive content.

    Args:
        text: Raw input text to process.
        source_type: The origin category of this text.
        source_id: Identifier of the source that provided the text.
        known_entities: Previously known entities for deduplication.
            If an extracted entity matches a known one (by name or alias,
            case-insensitive), the known entity is returned instead.

    Returns:
        A PerceptionResult with all extracted knowledge.
    """
    if not text or not text.strip():
        return PerceptionResult(redacted_content=text)

    known = known_entities or []

    # --- LLM extraction path (optional, falls back to regex on failure) ---
    if LLM_CONFIG["enabled"]:
        llm_result = _llm_extract(text, source_type)
        if llm_result is not None:
            entities = _llm_entities_to_types(llm_result.get("entities", []))
            claims = _llm_claims_to_types(llm_result.get("claims", []), entities)
            relationships = _llm_rels_to_types(
                llm_result.get("relationships", []), entities
            )
            # Sensitive data detection always uses regex (never send PII to LLM)
            redacted_text, sensitive_spans = _detect_sensitive(text)
            evidence = _create_evidence(redacted_text, source_type, source_id)
            for claim in claims:
                claim.evidence_ids.append(evidence.id)

            _EXTRACTION_STATS["last_method"] = "llm"
            _EXTRACTION_STATS["last_entity_count"] = len(entities)
            _EXTRACTION_STATS["last_claim_count"] = len(claims)
            _EXTRACTION_STATS["last_relationship_count"] = len(relationships)

            return PerceptionResult(
                entities=entities,
                claims=claims,
                relationships=relationships,
                evidence=[evidence],
                redacted_content=redacted_text,
                sensitive_spans=sensitive_spans,
            )

    # --- Regex extraction path (default / fallback) ---
    _EXTRACTION_STATS["regex_calls"] += 1

    # Step 1: Detect and redact sensitive data
    redacted_text, sensitive_spans = _detect_sensitive(text)

    # Step 2: Extract entities (from original text for better matching)
    entities = _extract_entities(text, known)

    # Step 3: Extract claims (from original text, linked to extracted entities)
    claims = _extract_claims(text, entities)

    # Step 4: Extract relationships between co-occurring entities
    relationships = _extract_relationships(text, entities)

    # Step 5: Create evidence record
    evidence = _create_evidence(redacted_text, source_type, source_id)

    # Step 6: Link evidence to all claims
    for claim in claims:
        claim.evidence_ids.append(evidence.id)

    _EXTRACTION_STATS["last_method"] = "regex"
    _EXTRACTION_STATS["last_entity_count"] = len(entities)
    _EXTRACTION_STATS["last_claim_count"] = len(claims)
    _EXTRACTION_STATS["last_relationship_count"] = len(relationships)

    return PerceptionResult(
        entities=entities,
        claims=claims,
        relationships=relationships,
        evidence=[evidence],
        redacted_content=redacted_text,
        sensitive_spans=sensitive_spans,
    )


# ===========================================================================
# Private extraction functions
# ===========================================================================


def _extract_entities(
    text: str, known_entities: list[Entity]
) -> list[Entity]:
    """Extract entities from text using pattern matching.

    Applies regex patterns for organizations, people, technology keywords,
    financial terms, locations, and events.  Deduplicates against known
    entities by case-insensitive name/alias matching.

    Args:
        text: Input text to scan.
        known_entities: Entities to match against for deduplication.

    Returns:
        List of Entity objects (mix of known and newly created).
    """
    seen_names: dict[str, Entity] = {}  # lowercase name -> Entity

    # Build lookup from known entities (name + aliases)
    known_lookup: dict[str, Entity] = {}
    for ent in known_entities:
        known_lookup[ent.name.lower()] = ent
        for alias in ent.aliases:
            known_lookup[alias.lower()] = ent

    def _add_entity(
        name: str, category: EntityCategory
    ) -> None:
        """Register an entity by name, deduplicating against known and seen."""
        name = name.strip()
        if not name or len(name) < 2:
            return
        key = name.lower()
        if key in seen_names:
            return
        # Check known entities
        if key in known_lookup:
            seen_names[key] = known_lookup[key]
            return
        # Create new entity
        entity = Entity(
            id=_new_id(),
            name=name,
            category=category,
            created=_now_iso(),
            updated=_now_iso(),
        )
        seen_names[key] = entity

    # --- Organizations ---
    for match in _RE_ORG.finditer(text):
        _add_entity(match.group(1), EntityCategory.ORGANIZATION)

    # --- People ---
    # Find capitalized multi-word names near role words
    words = text.split()
    word_lower = [w.lower().rstrip(".,;:!?'\"") for w in words]
    role_positions: set[int] = set()
    for i, wl in enumerate(word_lower):
        if wl in _ROLE_WORDS:
            role_positions.add(i)

    for match in _RE_PERSON_CANDIDATE.finditer(text):
        candidate = match.group(1)
        parts = candidate.split()
        # Skip if any part is in the stop list
        if any(p in _PERSON_STOP for p in parts):
            continue
        # Check proximity to role words: find the candidate's word index
        start_pos = match.start()
        char_count = 0
        word_idx = 0
        for wi, w in enumerate(words):
            if char_count >= start_pos:
                word_idx = wi
                break
            char_count += len(w) + 1  # +1 for space
        # Check within a window of 5 words before/after the candidate
        window = set(range(max(0, word_idx - 5), min(len(words), word_idx + len(parts) + 5)))
        if role_positions & window:
            _add_entity(candidate, EntityCategory.PERSON)

    # --- Technology ---
    for match in _RE_TECH.finditer(text):
        _add_entity(match.group(1), EntityCategory.TECHNOLOGY)

    # --- Financial ---
    for match in _RE_CURRENCY.finditer(text):
        _add_entity(match.group(0), EntityCategory.FINANCIAL)
    for match in _RE_PERCENTAGE.finditer(text):
        _add_entity(match.group(0), EntityCategory.FINANCIAL)
    for match in _RE_TICKER_CONTEXT.finditer(text):
        _add_entity(match.group(1), EntityCategory.FINANCIAL)
    for match in _RE_CASHTAG.finditer(text):
        _add_entity(match.group(1), EntityCategory.FINANCIAL)

    # --- Locations ---
    for match in _LOCATION_PREPS.finditer(text):
        _add_entity(match.group(1), EntityCategory.LOCATION)

    # --- Events ---
    for match in _RE_EVENT_QUARTER.finditer(text):
        _add_entity(match.group(0), EntityCategory.EVENT)
    for match in _RE_EVENT_FY.finditer(text):
        _add_entity(match.group(0), EntityCategory.EVENT)
    for match in _RE_EVENT_KEYWORDS.finditer(text):
        _add_entity(match.group(1), EntityCategory.EVENT)
    for match in _RE_DATE_PATTERN.finditer(text):
        _add_entity(match.group(0), EntityCategory.EVENT)

    return list(seen_names.values())


def _extract_claims(
    text: str, entities: list[Entity]
) -> list[Claim]:
    """Extract claims from text by splitting into sentences and classifying.

    Each sentence becomes a Claim with a semantic type determined by keyword
    matching.  Entity IDs are linked when the sentence mentions an entity
    by name.

    Args:
        text: Input text to process.
        entities: Previously extracted entities for linking.

    Returns:
        List of Claim objects with uniform initial uncertainty.
    """
    sentences = _split_sentences(text)
    claims: list[Claim] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue

        # Classify claim type (check in order of specificity)
        claim_type = _classify_claim(sentence)

        # Link entities mentioned in this sentence
        entity_ids: list[str] = []
        sentence_lower = sentence.lower()
        for entity in entities:
            if entity.name.lower() in sentence_lower:
                entity_ids.append(entity.id)
            else:
                # Check aliases
                for alias in entity.aliases:
                    if alias.lower() in sentence_lower:
                        entity_ids.append(entity.id)
                        break

        now = _now_iso()
        claim = Claim(
            id=_new_id(),
            text=sentence,
            claim_type=claim_type,
            uncertainty=Uncertainty.uniform(),
            entity_ids=entity_ids,
            created=now,
            updated=now,
        )
        claims.append(claim)

    return claims


def _classify_claim(sentence: str) -> ClaimType:
    """Classify a sentence into a ClaimType based on keyword patterns.

    Checks patterns in order of specificity: accusatory and diagnostic
    are checked before causal (since they are more specific subtypes),
    and prescriptive before predictive.

    Args:
        sentence: A single sentence string.

    Returns:
        The most appropriate ClaimType.
    """
    # Most specific first
    if _CLAIM_ACCUSATORY.search(sentence):
        return ClaimType.ACCUSATORY
    if _CLAIM_DIAGNOSTIC.search(sentence):
        return ClaimType.DIAGNOSTIC
    if _CLAIM_CAUSAL.search(sentence):
        return ClaimType.CAUSAL
    if _CLAIM_PRESCRIPTIVE.search(sentence):
        return ClaimType.PRESCRIPTIVE
    if _CLAIM_PREDICTIVE.search(sentence):
        return ClaimType.PREDICTIVE
    if _CLAIM_STATISTICAL.search(sentence):
        return ClaimType.STATISTICAL
    return ClaimType.FACTUAL


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on sentence-ending punctuation.

    Splits on `.`, `!`, or `?` followed by whitespace or end-of-string.
    Preserves the punctuation on the sentence that precedes it.

    Args:
        text: Input text.

    Returns:
        List of sentence strings.
    """
    parts = _RE_SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _extract_relationships(
    text: str, entities: list[Entity]
) -> list[Relationship]:
    """Extract relationships between entities co-occurring in sentences.

    For each sentence, finds all pairs of entities mentioned and checks
    for directional relationship signals.  Only creates a relationship
    when a clear signal is detected.

    Args:
        text: Input text.
        entities: Extracted entities to look for.

    Returns:
        List of Relationship objects.
    """
    sentences = _split_sentences(text)
    relationships: list[Relationship] = []
    seen_pairs: set[tuple[str, str, RelationshipType]] = set()

    for sentence in sentences:
        sentence_lower = sentence.lower()

        # Find which entities appear in this sentence
        present: list[Entity] = []
        for entity in entities:
            if entity.name.lower() in sentence_lower:
                present.append(entity)
            else:
                for alias in entity.aliases:
                    if alias.lower() in sentence_lower:
                        present.append(entity)
                        break

        if len(present) < 2:
            continue

        # Determine which relationship signal is present in the sentence
        detected_type: RelationshipType | None = None
        for pattern, rel_type in _RELATIONSHIP_SIGNALS:
            if pattern.search(sentence):
                detected_type = rel_type
                break

        if detected_type is None:
            continue

        # Create relationships for all entity pairs in the sentence
        for i in range(len(present)):
            for j in range(len(present)):
                if i == j:
                    continue
                source = present[i]
                target = present[j]
                key = (source.id, target.id, detected_type)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                rel = Relationship(
                    id=_new_id(),
                    source_id=source.id,
                    target_id=target.id,
                    rel_type=detected_type,
                    confidence=0.5,
                    created=_now_iso(),
                )
                relationships.append(rel)

    return relationships


def _detect_sensitive(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Detect and redact sensitive data in text.

    Scans the text for 16 categories of sensitive information (PII,
    credentials, secrets) and replaces detected spans with
    ``[REDACTED:{type}]`` markers.

    The function processes matches from the end of the string backward
    so that span offsets remain valid during replacement.

    Args:
        text: Input text to scan for sensitive data.

    Returns:
        A tuple of (redacted_text, sensitive_spans) where sensitive_spans
        is a list of (start, end, type_label) tuples describing what was
        found in the *original* text.
    """
    # Collect all matches across all patterns
    all_matches: list[tuple[int, int, str]] = []

    for pattern, label in _SENSITIVE_PATTERNS:
        for match in pattern.finditer(text):
            all_matches.append((match.start(), match.end(), label))

    if not all_matches:
        return text, []

    # Sort by start position, then by length descending (prefer longer matches)
    all_matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))

    # Remove overlapping matches: keep the first (longest at each position)
    filtered: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, label in all_matches:
        if start >= last_end:
            filtered.append((start, end, label))
            last_end = end

    # Build redacted text by replacing spans from end to start
    redacted = text
    for start, end, label in reversed(filtered):
        replacement = f"[REDACTED:{label}]"
        redacted = redacted[:start] + replacement + redacted[end:]

    return redacted, filtered


def _create_evidence(
    text: str, source_type: SourceType, source_id: str
) -> Evidence:
    """Create an Evidence record from the input text.

    This is a simple wrapper that packages the raw (redacted) text and
    source metadata into an Evidence object.  The reliability is set to
    a default based on the source type.

    Args:
        text: The (potentially redacted) input text.
        source_type: Origin category of the evidence.
        source_id: Identifier of the source entity/agent.

    Returns:
        A new Evidence record.
    """
    # Source-type reliability priors
    reliability_priors: dict[SourceType, float] = {
        SourceType.DIRECT_OBSERVATION: 0.9,
        SourceType.EXPERT_TESTIMONY: 0.85,
        SourceType.STATISTICAL: 0.8,
        SourceType.DOCUMENT: 0.7,
        SourceType.INFERENCE: 0.6,
        SourceType.SELF_REPORT: 0.5,
        SourceType.HEARSAY: 0.3,
        SourceType.SYSTEM: 1.0,
    }

    reliability = reliability_priors.get(source_type, 0.7)

    return Evidence(
        id=_new_id(),
        source_type=source_type,
        content=text,
        source_id=source_id,
        timestamp=_now_iso(),
        reliability=reliability,
    )


# ===========================================================================
# LLM extraction internals
# ===========================================================================


def _llm_extract(text: str, source_type: SourceType) -> Optional[dict]:
    """Call the configured LLM to extract entities, claims, and relationships.

    Constructs a prompt from ``_LLM_EXTRACTION_PROMPT``, sends it to the
    configured provider (Ollama or OpenAI-compatible), and parses the JSON
    response.  Returns ``None`` on any failure (network, timeout, malformed
    JSON), allowing the caller to fall back to regex extraction.

    Args:
        text: Raw input text (not redacted — the caller handles PII separately).
        source_type: Passed for future prompt tuning; currently unused.

    Returns:
        A parsed dict with keys "entities", "claims", "relationships",
        or None if the LLM call failed.
    """
    _EXTRACTION_STATS["llm_calls"] += 1

    prompt = _LLM_EXTRACTION_PROMPT.format(text=text)
    provider = LLM_CONFIG["provider"]
    base_url = LLM_CONFIG["base_url"].rstrip("/")
    model = LLM_CONFIG["model"]
    temperature = LLM_CONFIG["temperature"]
    timeout = LLM_CONFIG["timeout"]

    try:
        if provider == "ollama":
            response_text = _llm_call_ollama(
                base_url, model, prompt, temperature, timeout
            )
        else:
            response_text = _llm_call_openai_compatible(
                base_url, model, prompt, temperature, timeout
            )

        if response_text is None:
            _EXTRACTION_STATS["llm_failures"] += 1
            return None

        # Strip markdown code fences if the LLM wraps its response
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (possibly ```json)
            first_newline = cleaned.find("\n")
            if first_newline == -1:
                # Fence with no newline — discard as unparseable
                _EXTRACTION_STATS["llm_failures"] += 1
                return None
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            _EXTRACTION_STATS["llm_failures"] += 1
            return None

        # Validate required keys exist (gracefully default to empty lists)
        for key in ("entities", "claims", "relationships"):
            if key not in parsed:
                parsed[key] = []

        return parsed

    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        _log.debug("LLM response parse failed: %s", exc)
        _EXTRACTION_STATS["llm_failures"] += 1
        return None
    except Exception as exc:
        _log.debug("LLM extraction failed: %s", exc)
        _EXTRACTION_STATS["llm_failures"] += 1
        return None


def _llm_call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
) -> Optional[str]:
    """Send a generation request to an Ollama endpoint.

    Args:
        base_url: Ollama API base (e.g. ``http://localhost:11434``).
        model: Model name (e.g. ``llama3.2``).
        prompt: The full prompt string.
        temperature: Sampling temperature.
        timeout: HTTP timeout in seconds.

    Returns:
        The generated text, or None on failure.
    """
    url = base_url + "/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )

    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "")


def _llm_call_openai_compatible(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
) -> Optional[str]:
    """Send a chat completion request to an OpenAI-compatible endpoint.

    Args:
        base_url: API base URL (e.g. ``http://localhost:8080``).
        model: Model identifier.
        prompt: The full prompt string (sent as a single user message).
        temperature: Sampling temperature.
        timeout: HTTP timeout in seconds.

    Returns:
        The assistant's reply text, or None on failure.
    """
    url = base_url + "/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    api_key = LLM_CONFIG.get("api_key", "")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    req = urllib.request.Request(url, data=payload, method="POST", headers=headers)

    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return None


# ---------------------------------------------------------------------------
# LLM output -> ewm type converters
# ---------------------------------------------------------------------------

_ENTITY_CATEGORY_MAP: Dict[str, EntityCategory] = {
    "person": EntityCategory.PERSON,
    "organization": EntityCategory.ORGANIZATION,
    "technology": EntityCategory.TECHNOLOGY,
    "concept": EntityCategory.CONCEPT,
    "location": EntityCategory.LOCATION,
    "event": EntityCategory.EVENT,
    "artifact": EntityCategory.ARTIFACT,
    "financial": EntityCategory.FINANCIAL,
}

_CLAIM_TYPE_MAP: Dict[str, ClaimType] = {
    "factual": ClaimType.FACTUAL,
    "statistical": ClaimType.STATISTICAL,
    "causal": ClaimType.CAUSAL,
    "predictive": ClaimType.PREDICTIVE,
    "accusatory": ClaimType.ACCUSATORY,
    "diagnostic": ClaimType.DIAGNOSTIC,
    "prescriptive": ClaimType.PRESCRIPTIVE,
}

_REL_TYPE_MAP: Dict[str, RelationshipType] = {
    "owns": RelationshipType.OWNS,
    "employs": RelationshipType.EMPLOYS,
    "uses": RelationshipType.USES,
    "produces": RelationshipType.PRODUCES,
    "depends_on": RelationshipType.DEPENDS_ON,
    "competes_with": RelationshipType.COMPETES_WITH,
    "regulates": RelationshipType.REGULATES,
    "part_of": RelationshipType.PART_OF,
    "located_in": RelationshipType.LOCATED_IN,
    "causes": RelationshipType.CAUSES,
    "preceded_by": RelationshipType.PRECEDED_BY,
    "similar_to": RelationshipType.SIMILAR_TO,
}


def _llm_entities_to_types(raw: list) -> List[Entity]:
    """Convert LLM JSON entity list to Entity dataclasses.

    Args:
        raw: List of dicts with keys "name", "category", and optionally
            "aliases".

    Returns:
        List of Entity objects.  Entries with missing or invalid data are
        silently skipped.
    """
    entities: List[Entity] = []
    seen_names: set = set()

    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "").strip()
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())

        cat_str = item.get("category", "concept").lower()
        category = _ENTITY_CATEGORY_MAP.get(cat_str, EntityCategory.CONCEPT)

        raw_aliases = item.get("aliases", [])
        aliases: List[str] = []
        if isinstance(raw_aliases, list):
            aliases = [str(a) for a in raw_aliases if a]

        now = _now_iso()
        entities.append(Entity(
            id=_new_id(),
            name=name,
            category=category,
            aliases=aliases,
            created=now,
            updated=now,
        ))

    return entities


def _llm_claims_to_types(
    raw: list, entities: List[Entity]
) -> List[Claim]:
    """Convert LLM JSON claim list to Claim dataclasses.

    Links entity IDs by checking whether each entity's name appears
    in the claim text (case-insensitive).

    Args:
        raw: List of dicts with keys "text" and "claim_type".
        entities: Entities to link against by name.

    Returns:
        List of Claim objects.
    """
    claims: List[Claim] = []

    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "").strip()
        if not text:
            continue

        ct_str = item.get("claim_type", "factual").lower()
        claim_type = _CLAIM_TYPE_MAP.get(ct_str, ClaimType.FACTUAL)

        # Link entities mentioned in this claim
        entity_ids: List[str] = []
        text_lower = text.lower()
        for entity in entities:
            if entity.name.lower() in text_lower:
                entity_ids.append(entity.id)
            else:
                for alias in entity.aliases:
                    if alias.lower() in text_lower:
                        entity_ids.append(entity.id)
                        break

        now = _now_iso()
        claims.append(Claim(
            id=_new_id(),
            text=text,
            claim_type=claim_type,
            uncertainty=Uncertainty.uniform(),
            entity_ids=entity_ids,
            created=now,
            updated=now,
        ))

    return claims


def _llm_rels_to_types(
    raw: list, entities: List[Entity]
) -> List[Relationship]:
    """Convert LLM JSON relationship list to Relationship dataclasses.

    Resolves entity names from the LLM output to entity IDs by
    case-insensitive name/alias lookup.

    Args:
        raw: List of dicts with keys "source", "target", "type".
        entities: Entities to resolve names against.

    Returns:
        List of Relationship objects.  Entries whose source or target
        cannot be resolved are silently skipped.
    """
    # Build name -> entity ID lookup
    name_to_id: Dict[str, str] = {}
    for entity in entities:
        name_to_id[entity.name.lower()] = entity.id
        for alias in entity.aliases:
            name_to_id[alias.lower()] = entity.id

    relationships: List[Relationship] = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        source_name = item.get("source", "").strip().lower()
        target_name = item.get("target", "").strip().lower()
        rel_str = item.get("type", "").lower()

        source_id = name_to_id.get(source_name)
        target_id = name_to_id.get(target_name)

        if not source_id or not target_id:
            continue
        if source_id == target_id:
            continue

        rel_type = _REL_TYPE_MAP.get(rel_str)
        if rel_type is None:
            continue

        relationships.append(Relationship(
            id=_new_id(),
            source_id=source_id,
            target_id=target_id,
            rel_type=rel_type,
            confidence=0.7,  # higher confidence than regex (0.5)
            created=_now_iso(),
        ))

    return relationships
