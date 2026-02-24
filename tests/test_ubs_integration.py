"""
Integration Tests - Verify Everything Works Together
====================================================

Tests the full pipeline:
1. Event comes in
2. Entity identified/created
3. Context detected
4. Significance scored
5. Response generated
6. Beliefs updated
7. Patterns detected
8. Learning happens
"""

import sys
import json
from pathlib import Path


def test_full_pipeline():
    """Test the complete event processing pipeline."""
    print("=" * 60)
    print("INTEGRATION TEST: Full Pipeline")
    print("=" * 60)

    from guardian import Guardian
    from entity_registry import ThreatLevel
    from context_router import ResponseLevel

    # Create fresh guardian
    guardian = Guardian()

    # -------------------------------------------------------------------------
    # Scenario 1: First contact from unknown person
    # -------------------------------------------------------------------------
    print("\n📌 SCENARIO 1: First contact from unknown person")

    result = guardian.process_event({
        'event_type': 'contact',
        'entity_name': 'Alice Smith',
        'interaction_type': 'inform',
        'description': 'FYI - quarterly report is ready'
    })

    assert result.entity is not None, "Should create entity"
    assert result.entity.name == 'Alice Smith', "Should have correct name"
    assert result.response_level in {'IGNORE', 'NOTE', 'FLAG'}, \
        f"First contact info should be low priority, got {result.response_level}"
    print(f"  ✓ New entity created: {result.entity.name}")
    print(f"  ✓ Response level: {result.response_level}")

    # -------------------------------------------------------------------------
    # Scenario 2: Building threat history
    # -------------------------------------------------------------------------
    print("\n📌 SCENARIO 2: Building threat history over time")

    # Create a known entity
    guardian.entities.add_entity(
        "bob_jones",
        "Bob Jones",
        role="Director of Engineering",
        relationship="peer"
    )

    # Simulate history of bad interactions
    for i in range(5):
        guardian.entities.record_interaction(
            "bob_jones",
            "ask_resource",
            f"Urgent request #{i+1} - need your dev for 'small' task",
            stated_cost="2 days",
            actual_cost="2 weeks" if i >= 2 else "",
            outcome="negative" if i >= 2 else "unknown"
        )

    bob = guardian.entities.get("bob_jones")
    assert bob.threat_level >= 2, f"Should be elevated threat, got {bob.threat_level}"
    assert len(bob.patterns) > 0, "Should detect patterns"
    print(f"  ✓ Threat level elevated to: {ThreatLevel(bob.threat_level).name}")
    print(f"  ✓ Patterns detected: {[p.get('pattern_type') for p in bob.patterns]}")

    # -------------------------------------------------------------------------
    # Scenario 3: High-threat person makes new ask
    # -------------------------------------------------------------------------
    print("\n📌 SCENARIO 3: High-threat person makes new ask")

    result = guardian.process_event({
        'event_type': 'contact',
        'entity_name': 'Bob Jones',
        'interaction_type': 'ask_resource',
        'description': 'Quick favor - can I borrow Sarah for a day?',
        'stated_cost': '1 day',
        'context_signals': {'entity_relationship': 'peer'}
    })

    assert result.significance_score > 0.5, f"Should be significant, got {result.significance_score}"
    assert result.response_level in {'ALERT', 'INTERRUPT', 'CRITICAL'}, \
        f"High threat ask should trigger alert, got {result.response_level}"
    assert result.action.get('notify') == True, "Should notify"
    print(f"  ✓ Significance: {result.significance_score:.2f}")
    print(f"  ✓ Response level: {result.response_level}")
    print(f"  ✓ Suggested response: '{result.action.get('message', '')[:50]}...'")

    # -------------------------------------------------------------------------
    # Scenario 4: Belief propagation
    # -------------------------------------------------------------------------
    print("\n📌 SCENARIO 4: Belief propagation")

    # Add related claims (using TruthLayer API now)
    guardian.add_belief("project_a_success", "Project A was successful")
    guardian.add_belief("project_b_success", "Project B was successful")
    guardian.add_belief("team_effective", "Team is effective")

    # Link them - positive weights = support
    guardian.link_beliefs("project_a_success", "team_effective", 7.0)
    guardian.link_beliefs("project_b_success", "team_effective", 6.0)
    guardian.link_beliefs("project_a_success", "project_b_success", 4.0)

    # Get initial probability
    initial_team = guardian.beliefs.get_probability("team_effective")

    # Validate one
    guardian.validate_belief("project_a_success", "confirm")

    # Check propagation happened
    team_prob = guardian.beliefs.get_probability("team_effective")
    project_b_prob = guardian.beliefs.get_probability("project_b_success")

    assert team_prob > initial_team, f"Team effective should increase, got {team_prob}"

    print(f"  ✓ Initial team_effective: {initial_team:.2f}")
    print(f"  ✓ After confirm project_a: team_effective = {team_prob:.2f}")
    print(f"  ✓ project_b_success propagated to: {project_b_prob:.2f}")

    # -------------------------------------------------------------------------
    # Scenario 5: Outcome learning
    # -------------------------------------------------------------------------
    print("\n📌 SCENARIO 5: Outcome learning")

    initial_threat = bob.threat_level
    guardian.record_outcome("bob_jones", "very_negative", "3 weeks", "Way over estimate again")

    bob = guardian.entities.get("bob_jones")
    # Check belief was updated
    bob_reasonable_prob = guardian.beliefs.get_probability("bob_jones_reasonable")
    assert bob_reasonable_prob < 0.5, f"Should believe Bob is not reasonable, got {bob_reasonable_prob:.2f}"
    print(f"  ✓ Belief updated: P(Bob reasonable) = {bob_reasonable_prob:.2f}")

    print(f"  ✓ Threat level: {ThreatLevel(bob.threat_level).name}")

    # -------------------------------------------------------------------------
    # Scenario 6: Executive context triggers stronger response
    # -------------------------------------------------------------------------
    print("\n📌 SCENARIO 6: Executive context")

    guardian.entities.add_entity(
        "vp_carol",
        "Carol Williams",
        role="VP Operations",
        relationship="superior"
    )

    result = guardian.process_event({
        'event_type': 'contact',
        'entity_name': 'Carol Williams',
        'interaction_type': 'ask_resource',
        'description': 'I told the CEO we can handle the migration',
        'context_signals': {'entity_relationship': 'superior'}
    })

    assert result.context.domain == 'work_executive', f"Should detect exec context, got {result.context.domain}"
    assert result.response_level in {'ALERT', 'INTERRUPT', 'CRITICAL'}, \
        f"Exec ask should trigger alert, got {result.response_level}"
    print(f"  ✓ Context: {result.context.domain}")
    print(f"  ✓ Response level: {result.response_level}")
    print(f"  ✓ Rules matched: {result.matched_rules}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)

    stats = guardian.stats()
    print(f"\n📊 Final Stats:")
    print(f"   Beliefs tracked: {stats['beliefs']['total_claims']}")
    print(f"   Beliefs anchored: {stats['beliefs']['anchored']}")
    print(f"   Entities tracked: {stats['entities']['total']}")
    print(f"   High-risk entities: {stats['entities']['high_risk']}")
    print(f"   Events processed: {stats['events_processed']}")

    print("\n✅ All integration tests passed!")
    return True


def test_truth_validator_integration():
    """Test truth validator with the system."""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: Truth Validator")
    print("=" * 60)

    from truth_validator import TruthValidator, ValidationMode, RecordContext

    validator = TruthValidator(mode=ValidationMode.STRICT_CLOSED_RECORD)

    # Simulate validating an LLM-generated claim against a record
    record = RecordContext(
        text="""
        The defendant's software provides users with the ability to access
        and display copyrighted content. The plaintiff's damages report
        indicates losses of $10 million from 2020-2023.
        """,
        capability_verbs={'access', 'display'},
        numeric_values={'10', '2020', '2023'}
    )

    # Test good claim
    good_claim = "The software can access and display content"
    result = validator.validate_claim(good_claim, record)
    assert result.status != "FAIL", "Matching claim should pass"
    print(f"  ✓ Good claim: status={result.status}")

    # Test claim with wrong numeric
    bad_numeric = "Losses were $50 million"
    result = validator.validate_claim(bad_numeric, record)
    assert "NEW_NUMERICS" in str(result.flags), "Should flag new numeric"
    print(f"  ✓ Wrong numeric flagged: {result.flags}")

    # Test claim with inference
    inference = "The software likely violates copyright law"
    result = validator.validate_claim(inference, record)
    assert "INFERENCE_MARKERS" in str(result.flags), "Should flag inference"
    print(f"  ✓ Inference flagged: {result.flags}")

    # Test claim with new capability
    new_capability = "The software can transform and redistribute content"
    result = validator.validate_claim(new_capability, record)
    assert "CAPABILITY_VERB_NOT_IN_RECORD" in str(result.flags), "Should flag new capability"
    print(f"  ✓ New capability flagged: {result.flags}")

    print("\n✅ Truth validator integration tests passed!")
    return True


def test_persistence():
    """Test that data persists correctly."""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: Persistence")
    print("=" * 60)

    import tempfile
    import shutil
    from guardian import Guardian

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create guardian with persistence
        guardian1 = Guardian(data_dir=str(temp_dir))

        # Add data (using new TruthLayer API)
        guardian1.entities.add_entity("test_entity", "Test Person", role="Tester")
        guardian1.add_belief("test_claim", "Test claim is true")
        guardian1.process_event({
            'event_type': 'test',
            'entity_name': 'Test Person',
            'interaction_type': 'inform',
            'description': 'Test event'
        })

        # Trigger belief save by validating
        guardian1.validate_belief("test_claim", "confirm")

        # Check files exist
        assert (temp_dir / "entities.json").exists(), "Entities file should exist"
        assert (temp_dir / "truth_layer.json").exists(), "Truth layer file should exist"
        print("  ✓ Data files created")

        # Create new guardian from same directory
        guardian2 = Guardian(data_dir=str(temp_dir))

        # Verify data loaded
        assert "test_entity" in guardian2.entities.entities, "Entity should persist"
        assert "test_claim" in guardian2.beliefs.net.beliefs, "Belief should persist"
        print("  ✓ Data loaded correctly")

        # Verify values
        assert guardian2.entities.get("test_entity").name == "Test Person"
        # After validation with 'confirm', probability should be high
        prob = guardian2.beliefs.get_probability("test_claim")
        assert prob > 0.5, f"Probability should be > 0.5, got {prob}"
        print(f"  ✓ Values correct after reload (claim prob={prob:.2f})")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

    print("\n✅ Persistence tests passed!")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL INTEGRATION TESTS")
    print("=" * 60)

    # Change to the unified-belief-system directory
    import os
    os.chdir(Path(__file__).parent)

    results = []

    try:
        results.append(("Full Pipeline", test_full_pipeline()))
    except Exception as e:
        print(f"❌ Full Pipeline FAILED: {e}")
        results.append(("Full Pipeline", False))

    try:
        results.append(("Truth Validator", test_truth_validator_integration()))
    except Exception as e:
        print(f"❌ Truth Validator FAILED: {e}")
        results.append(("Truth Validator", False))

    try:
        results.append(("Persistence", test_persistence()))
    except Exception as e:
        print(f"❌ Persistence FAILED: {e}")
        results.append(("Persistence", False))

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
