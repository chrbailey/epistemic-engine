"""
Complex Tests for TruthLayer Integration
=========================================

Real-world scenarios to verify the system actually works:

1. Contradiction collapse - confirming one claim should reject contradictions
2. Belief chains - multi-hop propagation should work correctly
3. Entity threat escalation - repeated bad outcomes should lower beliefs
4. Mixed evidence - some positive, some negative outcomes
5. SAP-style scenario - the original use case
"""

import sys
from pathlib import Path


def test_contradiction_collapse():
    """
    SCENARIO: Two contradictory claims - confirming one should reject the other.

    This is the SAP example from the original code.
    """
    print("=" * 60)
    print("TEST 1: Contradiction Collapse")
    print("=" * 60)

    from truth_layer import TruthLayer

    tl = TruthLayer("test_contradiction.json")

    # Fresh start
    tl.net.beliefs.clear()
    tl.net.edges.clear()
    tl.net.anchored.clear()

    # Add contradictory claims
    tl.add_claim("project_frozen", "The project was frozen by executive mandate")
    tl.add_claim("project_completed", "The project was completed on schedule")
    tl.add_claim("had_authority", "Chris had full authority over the project")

    # Strong contradictions
    tl.add_relationship("project_frozen", "project_completed", weight=-10.0)
    tl.add_relationship("project_frozen", "had_authority", weight=-8.0)

    # Initial state - all 50/50
    print("\nINITIAL STATE:")
    for cid, belief in tl.net.beliefs.items():
        print(f"  {cid}: {belief.probability:.1%}")

    # Confirm the project was frozen
    print("\n→ VALIDATING: Project WAS frozen")
    tl.validate("project_frozen", "confirm")

    print("\nAFTER PROPAGATION:")
    frozen_prob = tl.get_probability("project_frozen")
    completed_prob = tl.get_probability("project_completed")
    authority_prob = tl.get_probability("had_authority")

    print(f"  project_frozen: {frozen_prob:.1%}")
    print(f"  project_completed: {completed_prob:.1%}")
    print(f"  had_authority: {authority_prob:.1%}")

    # Verify: frozen should be high, contradictions should be low
    assert frozen_prob > 0.9, f"Frozen should be high, got {frozen_prob:.1%}"
    assert completed_prob < 0.3, f"Completed should be low (contradiction), got {completed_prob:.1%}"
    assert authority_prob < 0.3, f"Authority should be low (contradiction), got {authority_prob:.1%}"

    print("\n✅ Contradiction collapse works correctly!")

    # Cleanup
    Path("test_contradiction.json").unlink(missing_ok=True)
    return True


def test_multi_hop_propagation():
    """
    SCENARIO: A → B → C chain. Confirming A should affect C.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Hop Propagation")
    print("=" * 60)

    from truth_layer import TruthLayer

    tl = TruthLayer("test_multihop.json")
    tl.net.beliefs.clear()
    tl.net.edges.clear()
    tl.net.anchored.clear()

    # Chain: rain → wet_ground → slippery
    tl.add_claim("rain", "It is raining")
    tl.add_claim("wet_ground", "The ground is wet")
    tl.add_claim("slippery", "The ground is slippery")

    # Rain causes wet ground, wet ground causes slippery
    tl.add_relationship("rain", "wet_ground", weight=8.0)
    tl.add_relationship("wet_ground", "slippery", weight=7.0)

    print("\nINITIAL (all 50%):")
    for cid, belief in tl.net.beliefs.items():
        print(f"  {cid}: {belief.probability:.1%}")

    # Confirm rain
    print("\n→ VALIDATING: It IS raining")
    tl.validate("rain", "confirm")

    print("\nAFTER PROPAGATION:")
    rain_prob = tl.get_probability("rain")
    wet_prob = tl.get_probability("wet_ground")
    slip_prob = tl.get_probability("slippery")

    print(f"  rain: {rain_prob:.1%}")
    print(f"  wet_ground: {wet_prob:.1%}")
    print(f"  slippery: {slip_prob:.1%}")

    # All should be elevated
    assert rain_prob > 0.9, f"Rain should be high, got {rain_prob:.1%}"
    assert wet_prob > 0.6, f"Wet should increase, got {wet_prob:.1%}"
    assert slip_prob > 0.5, f"Slippery should increase (multi-hop), got {slip_prob:.1%}"

    print("\n✅ Multi-hop propagation works!")

    Path("test_multihop.json").unlink(missing_ok=True)
    return True


def test_entity_belief_degradation():
    """
    SCENARIO: Entity has repeated negative outcomes - beliefs should degrade.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Entity Belief Degradation")
    print("=" * 60)

    from guardian import Guardian

    g = Guardian()

    # Add entity
    g.entities.add_entity(
        "bob_jones",
        "Bob Jones",
        role="Director",
        relationship="peer"
    )

    # Initial beliefs should be neutral
    reasonable_prob = g.beliefs.get_probability("bob_jones_reasonable")
    print(f"\nINITIAL: bob_jones_reasonable = {reasonable_prob:.1%}")

    # Simulate 5 negative interactions
    print("\n→ Recording 5 negative interactions...")
    for i in range(5):
        g.entities.record_interaction(
            "bob_jones",
            "ask_resource",
            f"Urgent request #{i+1} - need dev for 'small' task",
            stated_cost="2 days",
            actual_cost="2 weeks",
            outcome="negative"
        )

    # Check beliefs degraded
    reasonable_prob = g.beliefs.get_probability("bob_jones_reasonable")
    accurate_prob = g.beliefs.get_probability("bob_jones_accurate_costs")
    reciprocates_prob = g.beliefs.get_probability("bob_jones_reciprocates")

    print(f"\nAFTER 5 NEGATIVE OUTCOMES:")
    print(f"  bob_jones_reasonable: {reasonable_prob:.1%}")
    print(f"  bob_jones_accurate_costs: {accurate_prob:.1%}")
    print(f"  bob_jones_reciprocates: {reciprocates_prob:.1%}")

    # All should be low
    assert reasonable_prob < 0.2, f"Reasonable should be very low, got {reasonable_prob:.1%}"
    assert accurate_prob < 0.2, f"Accurate costs should be very low, got {accurate_prob:.1%}"

    # Check threat level elevated
    bob = g.entities.get("bob_jones")
    print(f"\n  Threat level: {bob.threat_level} (should be ≥3)")
    assert bob.threat_level >= 3, f"Threat should be HIGH or CRITICAL, got {bob.threat_level}"

    print("\n✅ Entity belief degradation works!")
    return True


def test_mixed_evidence():
    """
    SCENARIO: Some positive, some negative evidence - beliefs should balance.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Mixed Evidence Handling")
    print("=" * 60)

    from truth_layer import TruthLayer

    tl = TruthLayer("test_mixed.json")
    tl.net.beliefs.clear()
    tl.net.edges.clear()
    tl.net.anchored.clear()

    tl.add_claim("product_quality", "The product is high quality")

    print("\nINITIAL: product_quality = 50%")

    # Positive evidence
    print("\n→ 3 confirmations...")
    for _ in range(3):
        tl.validate("product_quality", "confirm")

    prob_after_confirms = tl.get_probability("product_quality")
    print(f"  After 3 confirms: {prob_after_confirms:.1%}")
    assert prob_after_confirms > 0.9, f"Should be very high, got {prob_after_confirms:.1%}"

    # Now negative evidence
    print("\n→ 2 rejections...")
    for _ in range(2):
        tl.validate("product_quality", "reject")

    prob_after_rejects = tl.get_probability("product_quality")
    print(f"  After 2 rejects: {prob_after_rejects:.1%}")

    # Should still be positive but less certain
    assert 0.4 < prob_after_rejects < 0.9, f"Should be moderate, got {prob_after_rejects:.1%}"

    print("\n✅ Mixed evidence handling works!")

    Path("test_mixed.json").unlink(missing_ok=True)
    return True


def test_real_work_scenario():
    """
    SCENARIO: Full workplace protection scenario.

    Mike Chen (VP Ops) repeatedly asks for resources, understates costs.
    Over time, system should:
    1. Track his pattern
    2. Lower belief he's reasonable
    3. Raise threat level
    4. Generate appropriate alerts
    """
    print("\n" + "=" * 60)
    print("TEST 5: Full Workplace Scenario")
    print("=" * 60)

    from guardian import Guardian
    from entity_registry import ThreatLevel

    g = Guardian()

    # Week 1: First contact - seems fine
    print("\n📅 WEEK 1: First contact")
    g.entities.add_entity(
        "mike_chen",
        "Mike Chen",
        role="VP Operations",
        relationship="superior"
    )

    g.entities.record_interaction(
        "mike_chen",
        "ask_resource",
        "Can I borrow a dev for a quick project?",
        stated_cost="2 days"
    )

    mike = g.entities.get("mike_chen")
    print(f"  Threat: {ThreatLevel(mike.threat_level).name}")
    print(f"  P(reasonable): {g.beliefs.get_probability('mike_chen_reasonable'):.1%}")

    # Week 2: That "2 day" project took 2 weeks
    print("\n📅 WEEK 2: First disappointment")
    g.record_outcome("mike_chen", "negative", "2 weeks", "Way over estimate")

    mike = g.entities.get("mike_chen")
    print(f"  Threat: {ThreatLevel(mike.threat_level).name}")
    print(f"  P(reasonable): {g.beliefs.get_probability('mike_chen_reasonable'):.1%}")
    print(f"  P(accurate_costs): {g.beliefs.get_probability('mike_chen_accurate_costs'):.1%}")

    # Week 3-6: It happens again. And again.
    print("\n📅 WEEKS 3-6: Pattern emerges")
    for week in range(3, 7):
        g.entities.record_interaction(
            "mike_chen",
            "ask_resource",
            f"URGENT: Need help with migration (week {week})",
            stated_cost="3 days",
            actual_cost="3 weeks",
            outcome="negative"
        )

    mike = g.entities.get("mike_chen")
    print(f"  Threat: {ThreatLevel(mike.threat_level).name}")
    print(f"  P(reasonable): {g.beliefs.get_probability('mike_chen_reasonable'):.1%}")
    print(f"  P(accurate_costs): {g.beliefs.get_probability('mike_chen_accurate_costs'):.1%}")
    print(f"  Patterns: {[p.get('pattern_type') for p in mike.patterns]}")

    # Week 7: He asks again - what does system say?
    print("\n📅 WEEK 7: New ask - what does system recommend?")
    result = g.process_event({
        'event_type': 'contact',
        'entity_name': 'Mike Chen',
        'interaction_type': 'ask_resource',
        'description': 'Quick favor - just need someone for a day',
        'stated_cost': '1 day'
    })

    print(f"\n  📊 SYSTEM ANALYSIS:")
    print(f"     Significance: {result.significance_score:.2f}")
    print(f"     Response level: {result.response_level}")
    print(f"     Notify: {result.action.get('notify')}")
    print(f"     Suggested response: {result.action.get('message', '')[:60]}...")

    # Verify system learned
    assert mike.threat_level >= 3, f"Should be HIGH threat, got {mike.threat_level}"
    assert result.response_level in {'ALERT', 'INTERRUPT', 'CRITICAL'}, \
        f"Should trigger alert, got {result.response_level}"
    assert result.action.get('notify') == True, "Should notify"

    reasonable = g.beliefs.get_probability('mike_chen_reasonable')
    assert reasonable < 0.1, f"Should believe Mike NOT reasonable, got {reasonable:.1%}"

    print("\n✅ Full workplace scenario works!")
    print(f"\n📋 FINAL TRUTH CONTEXT:\n{g.beliefs.get_truth_context()}")

    return True


def test_belief_persistence():
    """
    SCENARIO: Save and reload - beliefs should persist.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Belief Persistence")
    print("=" * 60)

    import tempfile
    import shutil
    from guardian import Guardian

    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create and populate
        g1 = Guardian(data_dir=str(temp_dir))

        g1.add_belief("test_claim", "This is a test claim")
        g1.validate_belief("test_claim", "confirm")

        prob_before = g1.beliefs.get_probability("test_claim")
        print(f"\nBefore save: P(test_claim) = {prob_before:.1%}")

        # Create new guardian from same directory
        g2 = Guardian(data_dir=str(temp_dir))

        prob_after = g2.beliefs.get_probability("test_claim")
        print(f"After reload: P(test_claim) = {prob_after:.1%}")

        assert abs(prob_before - prob_after) < 0.01, \
            f"Probability should match: {prob_before:.1%} vs {prob_after:.1%}"

        print("\n✅ Persistence works!")

    finally:
        shutil.rmtree(temp_dir)

    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL TRUTH LAYER TESTS")
    print("=" * 60)

    results = []

    tests = [
        ("Contradiction Collapse", test_contradiction_collapse),
        ("Multi-Hop Propagation", test_multi_hop_propagation),
        ("Entity Belief Degradation", test_entity_belief_degradation),
        ("Mixed Evidence", test_mixed_evidence),
        ("Full Workplace Scenario", test_real_work_scenario),
        ("Belief Persistence", test_belief_persistence),
    ]

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

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
