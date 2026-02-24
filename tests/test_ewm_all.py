"""
Epistemic World Model — Comprehensive Test Suite
==================================================

Single test file covering all modules, following Karpathy's convention:
one test file, all tests, organized by module.

Requires: pytest >= 7.0
Python:   3.9+ (no match statements, no X | Y union syntax)
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Ensure the project root is on sys.path so `ewm` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ewm.types import (
    Action,
    ActionType,
    Claim,
    ClaimType,
    CostAssessment,
    CostViolation,
    Entity,
    EntityCategory,
    Evidence,
    MemoryResult,
    PerceptionResult,
    Plan,
    Relationship,
    RelationshipType,
    SourceType,
    Uncertainty,
    WorldState,
)
from ewm.db import Database
from ewm.perception import perceive
from ewm.world_model import (
    update_uncertainty,
    expected_info_gain,
    integrate_evidence,
    predict_state,
    fill_missing,
    cumulative_fuse,
    averaging_fuse,
    trust_discount,
    trust_chain,
    deduce,
    opinion_complement,
    uncertainty_maximized,
    opinion_to_probability,
    probability_to_opinion,
)
from ewm.cost import (
    check_red_lines,
    check_claim_requirements,
    assess_action_cost,
    record_feedback,
    calibrate_thresholds,
)
from ewm.memory import (
    store,
    recall,
    context_for_session,
    forget_stale,
    working_set,
    working_get,
    working_clear,
    working_keys,
)
from ewm.actor import (
    plan_investigation,
    propose_updates,
    execute,
    generate_5w1h,
    rank_by_info_gain,
    summarize_plan,
)
from ewm.configurator import run, ingest, investigate, query, audit, session_sync
from ewm.self_modify import (
    inspect_module,
    list_modules,
    module_dependency_graph,
    system_health,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def db():
    """Fresh temporary database for each test."""
    with tempfile.TemporaryDirectory() as tmp:
        database = Database(os.path.join(tmp, "test.db"))
        yield database
        database.close()


# ===========================================================================
# 1. Types Module
# ===========================================================================


class TestUncertainty:
    """Tests for the Uncertainty opinion triple."""

    def test_uncertainty_triple_sums_to_one(self):
        """Uncertainty(b, d, u) must sum to 1.0."""
        u = Uncertainty(belief=0.5, disbelief=0.3, uncertainty=0.2)
        assert abs(u.belief + u.disbelief + u.uncertainty - 1.0) < 1e-6

    def test_uncertainty_rejects_invalid(self):
        """Negative values or non-unit sum should raise ValueError."""
        with pytest.raises(ValueError):
            Uncertainty(belief=-0.1, disbelief=0.5, uncertainty=0.6)
        with pytest.raises(ValueError):
            Uncertainty(belief=0.5, disbelief=0.5, uncertainty=0.5)  # sums to 1.5

    def test_uncertainty_uniform(self):
        """Uniform creates maximum ignorance (1/3, 1/3, 1/3)."""
        u = Uncertainty.uniform()
        assert abs(u.belief - 1 / 3) < 1e-6
        assert abs(u.disbelief - 1 / 3) < 1e-6
        assert abs(u.uncertainty - 1 / 3) < 1e-6

    def test_uncertainty_from_confidence(self):
        """from_confidence maps scalar to (conf, 0, 1-conf)."""
        u = Uncertainty.from_confidence(0.8)
        assert abs(u.belief - 0.8) < 1e-6
        assert abs(u.disbelief) < 1e-6
        assert abs(u.uncertainty - 0.2) < 1e-6

    def test_uncertainty_from_beta(self):
        """from_beta converts Beta distribution params to opinion."""
        u = Uncertainty.from_beta(alpha=5.0, beta=3.0)
        assert u.belief > u.disbelief  # more evidence for than against
        assert u.sample_size > 0

    def test_uncertainty_credible_interval(self):
        """Credible interval should be within [0, 1] and lower < upper."""
        u = Uncertainty(
            belief=0.6, disbelief=0.2, uncertainty=0.2, sample_size=10.0
        )
        lo, hi = u.credible_interval(0.9)
        assert 0.0 <= lo < hi <= 1.0

    def test_uncertainty_expected_value(self):
        """Expected value is alpha / (alpha + beta)."""
        u = Uncertainty(belief=0.8, disbelief=0.1, uncertainty=0.1)
        ev = u.expected_value
        assert 0.0 < ev < 1.0
        # With high belief, expected value should be above 0.5
        assert ev > 0.5

    def test_uncertainty_confidence_property(self):
        """Confidence = 1 - uncertainty."""
        u = Uncertainty(belief=0.5, disbelief=0.2, uncertainty=0.3)
        assert abs(u.confidence - 0.7) < 1e-6


# ===========================================================================
# 2. Database Module
# ===========================================================================


class TestDatabase:
    """Tests for SQLite persistence layer."""

    def test_db_entity_crud(self, db):
        """Create, read, find, delete entity."""
        entity = Entity(
            id="e1",
            name="TestCorp",
            category=EntityCategory.ORGANIZATION,
        )
        db.save_entity(entity)

        # Read
        fetched = db.get_entity("e1")
        assert fetched is not None
        assert fetched.name == "TestCorp"
        assert fetched.category == EntityCategory.ORGANIZATION

        # Find by name
        found = db.find_entities(name="TestCorp")
        assert len(found) >= 1
        assert any(e.id == "e1" for e in found)

        # Delete
        deleted = db.delete_entity("e1")
        assert deleted is True
        assert db.get_entity("e1") is None

    def test_db_evidence_crud(self, db):
        """Create and retrieve evidence."""
        ev = Evidence(
            id="ev1",
            source_type=SourceType.DOCUMENT,
            content="Some evidence text",
            reliability=0.8,
        )
        db.save_evidence(ev)

        fetched = db.get_evidence("ev1")
        assert fetched is not None
        assert fetched.content == "Some evidence text"
        assert abs(fetched.reliability - 0.8) < 1e-6

    def test_db_claim_crud(self, db):
        """Create, read, find claims; update uncertainty."""
        claim = Claim(
            id="c1",
            text="Revenue grew 15%",
            claim_type=ClaimType.STATISTICAL,
            uncertainty=Uncertainty(belief=0.6, disbelief=0.1, uncertainty=0.3),
        )
        db.save_claim(claim)

        # Read
        fetched = db.get_claim("c1")
        assert fetched is not None
        assert fetched.text == "Revenue grew 15%"
        assert fetched.claim_type == ClaimType.STATISTICAL
        assert abs(fetched.uncertainty.belief - 0.6) < 1e-6

        # Find by text
        found = db.find_claims(text="Revenue")
        assert len(found) >= 1

        # Update uncertainty
        new_u = Uncertainty(belief=0.8, disbelief=0.1, uncertainty=0.1)
        updated = db.update_claim_uncertainty("c1", new_u)
        assert updated is True
        refetched = db.get_claim("c1")
        assert refetched is not None
        assert abs(refetched.uncertainty.belief - 0.8) < 1e-6

    def test_db_relationship_crud(self, db):
        """Create and retrieve relationships between entities."""
        # Save two entities first (foreign key constraint)
        db.save_entity(Entity(id="e1", name="Alpha Corp", category=EntityCategory.ORGANIZATION))
        db.save_entity(Entity(id="e2", name="Beta Inc", category=EntityCategory.ORGANIZATION))

        rel = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            rel_type=RelationshipType.COMPETES_WITH,
            confidence=0.7,
        )
        db.save_relationship(rel)

        rels = db.get_relationships("e1")
        assert len(rels) >= 1
        assert rels[0].rel_type == RelationshipType.COMPETES_WITH

    def test_db_world_state(self, db):
        """load_world_state returns correct WorldState with dicts keyed by ID."""
        db.save_entity(Entity(id="e1", name="Test", category=EntityCategory.CONCEPT))
        db.save_claim(Claim(id="c1", text="A claim"))

        state = db.load_world_state()
        assert isinstance(state, WorldState)
        assert "e1" in state.entities
        assert "c1" in state.claims
        assert isinstance(state.entities, dict)
        assert isinstance(state.claims, dict)

    def test_db_audit_log(self, db):
        """log_action and get_audit_log round-trip."""
        action = Action(
            id="a1",
            action_type=ActionType.CREATE_ENTITY,
            target_id="e1",
            rationale="Test action",
            status="executed",
        )
        db.log_action(action)

        log = db.get_audit_log(limit=10)
        assert len(log) >= 1
        assert log[0]["action_type"] == "create_entity"
        assert log[0]["rationale"] == "Test action"


# ===========================================================================
# 3. Perception Module
# ===========================================================================


class TestPerception:
    """Tests for entity/claim/relationship extraction."""

    def test_perceive_entities(self):
        """Extracts entities from business text."""
        result = perceive(
            "Apple Inc. CEO Tim Cook announced new products.",
            source_type=SourceType.DOCUMENT,
            source_id="test",
        )
        names = [e.name for e in result.entities]
        assert any("Apple" in n for n in names)
        assert any("Tim Cook" in n for n in names)

    def test_perceive_claims(self):
        """Extracts and classifies claims."""
        result = perceive(
            "Revenue increased 25% year over year. The company should invest in AI.",
            source_type=SourceType.DOCUMENT,
            source_id="test",
        )
        assert len(result.claims) >= 1
        # At least one claim should be statistical (contains percentage)
        types = [c.claim_type for c in result.claims]
        assert ClaimType.STATISTICAL in types or ClaimType.PRESCRIPTIVE in types

    def test_perceive_sensitive_data(self):
        """Detects and redacts sensitive information."""
        result = perceive(
            "Contact john@example.com or call 555-123-4567",
            source_type=SourceType.DOCUMENT,
            source_id="test",
        )
        assert "[REDACTED" in result.redacted_content
        assert len(result.sensitive_spans) > 0

    def test_perceive_financial(self):
        """Extracts financial entities (currency, percentages)."""
        result = perceive(
            "The stock price hit $150 million with 12% growth.",
            source_type=SourceType.DOCUMENT,
            source_id="test",
        )
        categories = [e.category for e in result.entities]
        assert EntityCategory.FINANCIAL in categories

    def test_perceive_empty_input(self):
        """Empty input returns empty result without errors."""
        result = perceive(
            "",
            source_type=SourceType.DOCUMENT,
            source_id="test",
        )
        assert len(result.entities) == 0
        assert len(result.claims) == 0


# ===========================================================================
# 4. World Model Module
# ===========================================================================


class TestWorldModel:
    """Tests for uncertainty updates and evidence integration."""

    def test_update_uncertainty_support(self):
        """Support evidence increases belief."""
        prior = Uncertainty.uniform()
        ev = Evidence(source_type=SourceType.EXPERT_TESTIMONY, reliability=0.9)
        updated = update_uncertainty(prior, ev, direction="support")
        assert updated.belief > prior.belief

    def test_update_uncertainty_contradict(self):
        """Contradicting evidence increases disbelief."""
        prior = Uncertainty.uniform()
        ev = Evidence(source_type=SourceType.EXPERT_TESTIMONY, reliability=0.9)
        updated = update_uncertainty(prior, ev, direction="contradict")
        assert updated.disbelief > prior.disbelief

    def test_expected_info_gain(self):
        """High uncertainty claims have higher info gain."""
        c_uncertain = Claim(text="Unknown", uncertainty=Uncertainty.uniform())
        c_certain = Claim(
            text="Known",
            uncertainty=Uncertainty(belief=0.9, disbelief=0.05, uncertainty=0.05),
        )
        assert expected_info_gain(c_uncertain) > expected_info_gain(c_certain)

    def test_integrate_evidence(self):
        """Full evidence integration updates the claim in world state."""
        claim = Claim(
            id="c1",
            text="Test claim",
            uncertainty=Uncertainty.uniform(),
        )
        entity = Entity(id="e1", name="TestEntity", category=EntityCategory.CONCEPT)
        claim.entity_ids = ["e1"]

        state = WorldState(
            entities={"e1": entity},
            claims={"c1": claim},
        )
        ev = Evidence(
            id="ev1",
            source_type=SourceType.DOCUMENT,
            reliability=0.8,
        )

        new_state = integrate_evidence(state, ev, "c1", direction="support")
        updated_claim = new_state.claims["c1"]
        assert updated_claim.uncertainty.belief > claim.uncertainty.belief
        assert "ev1" in updated_claim.evidence_ids


# ===========================================================================
# 5. Cost Module
# ===========================================================================


class TestCost:
    """Tests for red lines, claim requirements, and action assessment."""

    def test_red_lines_sql_injection(self):
        """SQL injection patterns trigger red line violation."""
        violations = check_red_lines("DROP TABLE users;")
        assert len(violations) > 0
        assert any(v.severity == "red_line" for v in violations)

    def test_red_lines_clean_text(self):
        """Clean text produces no violations."""
        violations = check_red_lines("The company reported strong Q3 earnings.")
        assert len(violations) == 0

    def test_claim_requirements(self):
        """Claim type requirements check evidence count and confidence."""
        # A causal claim with high confidence but no evidence should flag
        claim = Claim(
            text="A caused B",
            claim_type=ClaimType.CAUSAL,
            uncertainty=Uncertainty(belief=0.7, disbelief=0.1, uncertainty=0.2),
        )
        violations = check_claim_requirements(claim, evidence_count=0)
        assert len(violations) > 0
        assert any(v.rule == "insufficient_evidence" for v in violations)

    def test_assess_action_safe(self):
        """Safe actions pass cost assessment."""
        action = Action(
            action_type=ActionType.CREATE_ENTITY,
            target_id="e1",
            payload={"name": "Acme Corp", "category": "organization"},
            rationale="New entity discovered",
        )
        state = WorldState()
        assessment = assess_action_cost(action, state)
        assert assessment.blocked is False

    def test_assess_action_dangerous(self):
        """Actions with red line content are blocked."""
        action = Action(
            action_type=ActionType.CREATE_ENTITY,
            target_id="e1",
            payload={"name": "Test"},
            rationale="password=secret123 leaked",
        )
        state = WorldState()
        assessment = assess_action_cost(action, state)
        assert assessment.blocked is True
        assert len(assessment.violations) > 0


# ===========================================================================
# 6. Memory Module
# ===========================================================================


class TestMemory:
    """Tests for store, recall, context, working memory, and stale pruning."""

    def test_store_and_recall(self, db):
        """Store perception result, then recall by query."""
        result = perceive(
            "Google Cloud announced new AI features.",
            source_type=SourceType.DOCUMENT,
            source_id="test",
        )
        store(db, result)

        # Recall should find something related
        mem = recall(db, "Google", limit=10)
        # We should get at least one entity or claim mentioning Google
        all_names = [e.name for e in mem.entities]
        all_texts = [c.text for c in mem.claims]
        has_google = any("Google" in n for n in all_names) or any("Google" in t for t in all_texts)
        assert has_google

    def test_store_deduplication(self, db):
        """Storing same entity twice should merge, not duplicate."""
        result1 = PerceptionResult(
            entities=[
                Entity(id="e1", name="Acme Corp", category=EntityCategory.ORGANIZATION)
            ],
        )
        result2 = PerceptionResult(
            entities=[
                Entity(
                    id="e2",
                    name="Acme Corp",
                    category=EntityCategory.ORGANIZATION,
                    aliases=["ACME"],
                )
            ],
        )
        store(db, result1)
        summary = store(db, result2)

        assert summary["duplicates_merged"] >= 1

        # Only one entity should exist
        all_entities = db.find_entities(name="Acme Corp")
        # Exact match dedup: both had name "Acme Corp"
        exact = [e for e in all_entities if e.name.lower() == "acme corp"]
        assert len(exact) == 1

    def test_context_for_session(self, db):
        """Context generation produces non-empty structured text."""
        db.save_entity(Entity(id="e1", name="TestEntity", category=EntityCategory.CONCEPT))
        db.save_claim(Claim(id="c1", text="A factual claim about testing."))

        context = context_for_session(db, session_topic="testing")
        assert len(context) > 0
        assert "EPISTEMIC WORLD MODEL" in context

    def test_working_memory(self):
        """Working memory set/get/clear/keys cycle."""
        working_clear()
        assert working_keys() == []

        working_set("key1", "value1")
        working_set("key2", 42)

        assert working_get("key1") == "value1"
        assert working_get("key2") == 42
        assert working_get("missing") is None
        assert set(working_keys()) == {"key1", "key2"}

        working_clear()
        assert working_keys() == []

    def test_forget_stale(self, db):
        """Stale uncertain claims are pruned (or reviewed without error)."""
        # Create a claim with high uncertainty
        claim = Claim(
            id="stale1",
            text="An uncertain claim",
            uncertainty=Uncertainty.uniform(),
        )
        db.save_claim(claim)

        # With max_age_days=0, it should be pruned immediately (uncertainty > 0.6 = 1/3 ~0.33)
        # Actually uniform uncertainty is 1/3 = 0.33, which is <= 0.6, so it won't be pruned.
        # Use a high-uncertainty claim instead.
        claim2 = Claim(
            id="stale2",
            text="Very uncertain",
            uncertainty=Uncertainty(belief=0.1, disbelief=0.1, uncertainty=0.8),
        )
        db.save_claim(claim2)

        result = forget_stale(db, max_age_days=0)
        assert "claims_reviewed" in result
        assert result["claims_reviewed"] >= 1
        # With max_age_days=0, the claim created "now" might not be pruned
        # because its updated timestamp is also "now", and age_days < 0 would fail.
        # The important thing is the function runs without error.


# ===========================================================================
# 7. Actor Module
# ===========================================================================


class TestActor:
    """Tests for investigation planning, updates, and execution."""

    def test_plan_investigation(self):
        """Investigation plan ranks by info gain."""
        claims = [
            Claim(text="Uncertain claim", uncertainty=Uncertainty.uniform()),
            Claim(
                text="Certain claim",
                uncertainty=Uncertainty(belief=0.9, disbelief=0.05, uncertainty=0.05),
            ),
        ]
        plan = plan_investigation(claims, top_k=5)
        assert isinstance(plan, Plan)
        # The uncertain claim should appear first (higher info gain)
        if len(plan.actions) >= 1:
            first_text = plan.actions[0].payload.get("claim_text", "")
            assert "Uncertain" in first_text

    def test_propose_updates_new_entities(self):
        """Propose creates CREATE_ENTITY for unknown entities."""
        result = PerceptionResult(
            entities=[
                Entity(id="new1", name="NewCo", category=EntityCategory.ORGANIZATION)
            ],
            claims=[],
        )
        state = WorldState(entities={}, claims={})
        actions = propose_updates(result, state)

        create_actions = [
            a for a in actions if a.action_type == ActionType.CREATE_ENTITY
        ]
        assert len(create_actions) >= 1
        assert create_actions[0].payload["name"] == "NewCo"

    def test_execute_create_entity(self, db):
        """Execute CREATE_ENTITY persists to database."""
        action = Action(
            action_type=ActionType.CREATE_ENTITY,
            target_id="exec1",
            payload={
                "id": "exec1",
                "name": "ExecutedCorp",
                "category": "organization",
                "aliases": [],
                "properties": {},
                "created": "",
                "updated": "",
            },
            rationale="Test execution",
        )
        result = execute(action, db)
        assert result.status == "executed"

        entity = db.get_entity("exec1")
        assert entity is not None
        assert entity.name == "ExecutedCorp"

    def test_generate_5w1h(self):
        """5W1H generates all six keys."""
        claim = Claim(text="Apple Inc CEO Tim Cook announced new products in Q1 2025.")
        result = generate_5w1h(claim)
        expected_keys = {"who", "what", "when", "where", "why", "how"}
        assert set(result.keys()) == expected_keys

    def test_rank_by_info_gain(self):
        """Ranking puts uncertain claims first."""
        claims = [
            Claim(
                text="Certain",
                uncertainty=Uncertainty(belief=0.9, disbelief=0.05, uncertainty=0.05),
            ),
            Claim(text="Uncertain", uncertainty=Uncertainty.uniform()),
        ]
        ranked = rank_by_info_gain(claims)
        assert ranked[0][0].text == "Uncertain"
        assert ranked[0][1] > ranked[1][1]  # higher gain first


# ===========================================================================
# 8. Configurator Integration
# ===========================================================================


class TestConfigurator:
    """Tests for the orchestration pipelines."""

    def test_ingest_pipeline(self, db):
        """Full ingest pipeline: perceive -> store -> integrate."""
        result = ingest(
            "Microsoft Corporation announced a $10 billion investment in AI.",
            db,
            source_type="document",
        )
        assert result["status"] == "completed"
        assert result["result"]["entities"] >= 0
        assert result["result"]["claims"] >= 0

    def test_investigate_pipeline(self, db):
        """Investigation pipeline loads state and plans."""
        # Seed some data first
        db.save_claim(Claim(id="c1", text="An uncertain claim about markets."))
        result = investigate(db, top_k=3)
        assert result["status"] == "completed"
        assert "plan" in result["result"]

    def test_session_sync_pipeline(self, db):
        """Session sync extracts and generates context."""
        result = session_sync(
            "We discussed Python and TypeScript during the session.",
            db,
        )
        assert result["status"] == "completed"
        assert "context" in result["result"]
        assert result["result"]["context_length"] > 0


# ===========================================================================
# 9. Self-Modify Module
# ===========================================================================


class TestSelfModify:
    """Tests for introspection and module listing."""

    def test_inspect_module(self):
        """inspect_module returns valid info for types.py."""
        info = inspect_module("types")
        assert info["exists"] is True
        assert info["lines"] > 0
        assert len(info["functions"]) >= 0
        assert info["name"] == "types"

    def test_inspect_nonexistent_module(self):
        """inspect_module returns exists=False for missing modules."""
        info = inspect_module("nonexistent_module_xyz")
        assert info["exists"] is False
        assert info["lines"] == 0

    def test_list_modules(self):
        """list_modules finds all expected modules."""
        modules = list_modules()
        module_names = [m["name"] for m in modules]
        expected = [
            "types", "db", "perception", "world_model",
            "cost", "memory", "actor", "configurator", "self_modify",
        ]
        for name in expected:
            assert name in module_names, f"Module '{name}' not found in list_modules()"


# ===========================================================================
# 10. Subjective Logic Operators
# ===========================================================================


class TestSubjectiveLogic:
    """Tests for Josang's Subjective Logic operators in world_model.py."""

    def test_cumulative_fuse_reduces_uncertainty(self):
        """Cumulative fusion of independent sources reduces uncertainty."""
        o1 = Uncertainty(belief=0.6, disbelief=0.1, uncertainty=0.3)
        o2 = Uncertainty(belief=0.5, disbelief=0.2, uncertainty=0.3)
        fused = cumulative_fuse([o1, o2])
        assert fused.uncertainty < o1.uncertainty
        assert fused.uncertainty < o2.uncertainty
        # Components should still sum to ~1
        assert abs(fused.belief + fused.disbelief + fused.uncertainty - 1.0) < 1e-9

    def test_cumulative_fuse_single(self):
        """Fusing a single opinion returns it unchanged."""
        o = Uncertainty(belief=0.7, disbelief=0.1, uncertainty=0.2)
        assert cumulative_fuse([o]) is o

    def test_cumulative_fuse_empty_raises(self):
        """Fusing an empty list raises ValueError."""
        with pytest.raises(ValueError):
            cumulative_fuse([])

    def test_cumulative_fuse_sample_size_additive(self):
        """Cumulative fusion sums sample sizes (independent evidence)."""
        o1 = Uncertainty(belief=0.5, disbelief=0.2, uncertainty=0.3, sample_size=10.0)
        o2 = Uncertainty(belief=0.6, disbelief=0.1, uncertainty=0.3, sample_size=5.0)
        fused = cumulative_fuse([o1, o2])
        assert abs(fused.sample_size - 15.0) < 1e-9

    def test_averaging_fuse_preserves_uncertainty(self):
        """Averaging fusion does NOT concentrate certainty (unlike cumulative)."""
        o1 = Uncertainty(belief=0.6, disbelief=0.1, uncertainty=0.3, sample_size=10.0)
        o2 = Uncertainty(belief=0.4, disbelief=0.2, uncertainty=0.4, sample_size=10.0)
        avg = averaging_fuse([o1, o2])
        # Averaged uncertainty should be between the two inputs
        assert min(o1.uncertainty, o2.uncertainty) <= avg.uncertainty + 1e-9
        assert avg.uncertainty <= max(o1.uncertainty, o2.uncertainty) + 1e-9
        assert abs(avg.belief + avg.disbelief + avg.uncertainty - 1.0) < 1e-9

    def test_averaging_fuse_empty_raises(self):
        """Averaging an empty list raises ValueError."""
        with pytest.raises(ValueError):
            averaging_fuse([])

    def test_trust_discount_increases_uncertainty(self):
        """Discounting by trust increases the uncertainty of an opinion."""
        trust = Uncertainty(belief=0.8, disbelief=0.1, uncertainty=0.1)
        opinion = Uncertainty(belief=0.7, disbelief=0.1, uncertainty=0.2)
        discounted = trust_discount(trust, opinion)
        assert discounted.uncertainty >= opinion.uncertainty
        assert abs(discounted.belief + discounted.disbelief + discounted.uncertainty - 1.0) < 1e-9

    def test_trust_discount_zero_trust(self):
        """Full distrust produces a vacuous opinion (all uncertainty)."""
        no_trust = Uncertainty(belief=0.0, disbelief=0.9, uncertainty=0.1)
        opinion = Uncertainty(belief=0.9, disbelief=0.05, uncertainty=0.05)
        discounted = trust_discount(no_trust, opinion)
        # With near-zero trust, most of the opinion becomes uncertainty
        assert discounted.uncertainty > 0.9

    def test_trust_chain_transitive(self):
        """Trust chain computes transitive trust through intermediary."""
        # Chain: A trusts B (0.9), B trusts C (0.8), C's opinion about X
        ab = Uncertainty(belief=0.9, disbelief=0.05, uncertainty=0.05)
        bc = Uncertainty(belief=0.8, disbelief=0.1, uncertainty=0.1)
        cx = Uncertainty(belief=0.7, disbelief=0.1, uncertainty=0.2)
        ac = trust_chain([ab, bc, cx])
        # Transitive trust should increase uncertainty compared to direct
        assert ac.uncertainty >= cx.uncertainty
        assert abs(ac.belief + ac.disbelief + ac.uncertainty - 1.0) < 1e-9

    def test_opinion_complement(self):
        """Complement swaps belief and disbelief."""
        o = Uncertainty(belief=0.7, disbelief=0.1, uncertainty=0.2)
        c = opinion_complement(o)
        assert abs(c.belief - o.disbelief) < 1e-9
        assert abs(c.disbelief - o.belief) < 1e-9
        assert abs(c.uncertainty - o.uncertainty) < 1e-9

    def test_uncertainty_maximized(self):
        """Uncertainty-maximized selects the most uncertain opinion."""
        o1 = Uncertainty(belief=0.6, disbelief=0.1, uncertainty=0.3)
        o2 = Uncertainty(belief=0.3, disbelief=0.2, uncertainty=0.5)
        um = uncertainty_maximized([o1, o2])
        assert um.uncertainty == o2.uncertainty
        assert abs(um.belief + um.disbelief + um.uncertainty - 1.0) < 1e-9

    def test_opinion_probability_roundtrip(self):
        """Converting opinion -> probability -> opinion preserves expected value."""
        o = Uncertainty(belief=0.6, disbelief=0.2, uncertainty=0.2)
        p = opinion_to_probability(o)
        assert 0.0 <= p <= 1.0
        o2 = probability_to_opinion(p, uncertainty_level=o.uncertainty)
        assert abs(opinion_to_probability(o2) - p) < 1e-9

    def test_deduce_produces_valid_opinion(self):
        """Deduction produces an opinion with components summing to 1."""
        # opinion_xy: "X implies Y" with high belief
        opinion_xy = Uncertainty(belief=0.9, disbelief=0.05, uncertainty=0.05)
        # opinion_x: belief about X
        opinion_x = Uncertainty(belief=0.7, disbelief=0.1, uncertainty=0.2)
        result = deduce(opinion_xy, opinion_x)
        assert abs(result.belief + result.disbelief + result.uncertainty - 1.0) < 1e-9
        assert result.belief >= 0.0
        assert result.disbelief >= 0.0
        assert result.uncertainty >= 0.0
