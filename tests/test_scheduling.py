import copy

import pytest

from src.scheduling import action_scores, handoff_count, reactive_pause, replay, schedule, score, select


def action(action_id, robot, duration, deps=()):
    return {"id": action_id, "robot": robot, "duration": duration,
            "deps": list(deps), "objects": [], "skill": "abstract"}


def slack_pair():
    # The same action set, assignment, and explicit dependency count. Only robot
    # 0's order differs, so the early transfer has seven ticks of receiver slack.
    source = action("source", 0, 1)
    long = action("long", 0, 7)
    tail = action("tail", 0, 2)
    filler = action("filler", 1, 8)
    consume = action("consume", 1, 1, ["source"])
    return ({"actions": [source, long, tail, filler, consume]},
            {"actions": [long, source, tail, filler, consume]})


def test_equal_makespan_and_handoffs_have_different_finite_delay_propagation():
    early, late = slack_pair()
    assert schedule(early)["makespan"] == schedule(late)["makespan"] == 10
    assert handoff_count(early) == handoff_count(late) == 1
    early_fault = replay(early, {"source": 4})
    late_fault = replay(late, {"source": 4})
    assert early_fault["peer_end_shift"] == 0
    assert late_fault["peer_end_shift"] == 4
    assert late_fault["impacted_peer_actions"] == 1
    assert action_scores(early)["source"] == pytest.approx(1 / 3)
    assert action_scores(late)["source"] == pytest.approx(14 / 3)
    # A reachability-only ablation cannot see the slack for this source action.
    assert action_scores(early, "structural")["source"] == action_scores(late, "structural")["source"]
    assert score(early) < score(late)
    assert select([late, early], "risk")["index"] == 1


def test_delay_leaves_independent_peer_branch_unaffected():
    plan = {"actions": [action("fault", 0, 2), action("peer", 1, 3, ["fault"]),
                        action("independent", 2, 9), action("peer_tail", 1, 1)]}
    result = replay(plan, {"fault": 4})
    assert result["end_shifts"] == {"fault": 4, "peer": 4, "independent": 0, "peer_tail": 4}
    assert result["peer_end_shift"] == 8
    assert result["impacted_peer_actions"] == 2
    assert result["makespan_increase"] == 1


def test_robot_order_edges_serialize_even_without_explicit_dependencies():
    plan = {"actions": [action("a", 0, 2), action("b", 1, 5), action("c", 0, 3)]}
    result = schedule(plan)
    assert result["starts"] == {"a": 0, "b": 0, "c": 2}
    assert result["ends"] == {"a": 2, "b": 5, "c": 5}
    assert handoff_count(plan) == 0


def test_rejects_cycle_created_by_robot_order_and_explicit_dependency():
    plan = {"actions": [action("a", 0, 1, ["b"]), action("b", 0, 1)]}
    with pytest.raises(ValueError, match="cycle"):
        schedule(plan)


def test_replay_is_deterministic_and_does_not_mutate_input():
    plan = slack_pair()[1]
    original = copy.deepcopy(plan)
    assert replay(plan, {"source": 8}) == replay(plan, {"source": 8})
    assert plan == original


def test_outage_waits_for_full_action_then_propagates_to_peers():
    plan = {"actions": [action("before", 0, 2), action("overlap", 0, 4),
                        action("peer", 1, 1, ["overlap"]), action("alone", 2, 20)]}
    result = replay(plan, outage={"robot": 0, "start": 3, "duration": 4})
    assert result["perturbed"]["starts"]["overlap"] == 7
    assert result["perturbed"]["ends"]["overlap"] == 11
    assert result["peer_end_shift"] == 5
    assert result["impacted_peer_actions"] == 1
    assert result["makespan_increase"] == 0


def test_outage_half_open_boundaries_do_not_delay_unaffected_actions():
    plan = {"actions": [action("ends_at_start", 0, 3), action("next", 0, 2)]}
    result = schedule(plan, outage={"robot": 0, "start": 3, "duration": 2})
    assert result["ends"]["ends_at_start"] == 3
    assert result["starts"]["next"] == 5
    assert schedule(plan, outage={"robot": 0, "start": 3, "duration": 0}) == schedule(plan)


def test_all_alternatives_share_nominal_budget_and_seeded_random():
    # Candidate 2 is just outside the 10% allowance; it must never be selected.
    plans = [{"actions": [action("a", 0, duration)]} for duration in (10, 11, 11.01)]
    for method in ("nominal", "risk", "time_risk", "time_total", "exposure_risk", "total_risk",
                   "slack", "handoffs", "structural", "random"):
        result = select(plans, method, seed=23)
        assert result["eligible_indices"] == [0, 1]
        assert result["index"] in (0, 1)
        assert result["nominal_limit"] == 11
    assert select(plans, "random", seed=23) == select(plans, "random", seed=23)
    for method in ("risk", "handoffs", "structural"):
        assert select(plans, method)["index"] == 0


def test_score_ties_use_nominal_then_pool_index():
    plans = [{"actions": [action("a", 0, duration)]} for duration in (11, 10, 10)]
    assert select(plans, "risk")["index"] == 1
    assert select(plans, "nominal")["index"] == 1


def test_handoffs_count_unique_explicit_cross_robot_edges():
    plan = {"actions": [action("a", 0, 1), action("b", 0, 1, ["a"]),
                        action("c", 1, 1, ["a", "a", "b"])]}
    assert handoff_count(plan) == 2


def test_ambiguous_peer_reference_requires_fault_robot():
    plan = {"actions": [action("a", 0, 1), action("b", 1, 1)]}
    with pytest.raises(ValueError, match="Multiple fault robots"):
        replay(plan, {"a": 2, "b": 4})
    assert replay(plan, {"a": 2, "b": 4}, fault_robot=0)["peer_end_shift"] == 4


@pytest.mark.parametrize("delay", [-1, float("inf"), float("nan")])
def test_invalid_delay_is_rejected(delay):
    with pytest.raises(ValueError, match="finite nonnegative"):
        replay({"actions": [action("a", 0, 1)]}, {"a": delay})


def test_unknown_dependencies_and_empty_grid_are_rejected():
    with pytest.raises(ValueError, match="Unknown dependency"):
        schedule({"actions": [action("a", 0, 1, ["missing"])]})
    with pytest.raises(ValueError, match="delta_grid"):
        score({"actions": [action("a", 0, 1)]}, delta_grid=())


def test_empty_plan_has_zero_schedule_and_scores():
    plan = {"actions": []}
    assert schedule(plan) == {"starts": {}, "ends": {}, "makespan": 0.0}
    assert score(plan) == score(plan, "structural") == 0.0
    assert replay(plan)["peer_end_shift"] == 0.0


def test_free_slack_is_direct_successor_gap_plus_terminal_slack():
    plan = {"actions": [action("a", 0, 2), action("b", 1, 5),
                        action("c", 0, 1, ["b"])]}
    # a -> c by serialization: free slack(a)=5-2=3; b and c have zero.
    assert score(plan, "slack") == 3
    separate_terminal = copy.deepcopy(plan)
    separate_terminal["actions"][2]["robot"] = 2
    # Now a is terminal, so its free slack is makespan(6) - end(a)(2).
    assert score(separate_terminal, "slack") == 4
    choice = select([plan, separate_terminal], "slack")
    assert choice["index"] == 1
    assert choice["score"] == 4
    assert schedule(choice["plan"])["makespan"] == 6


def test_total_risk_includes_source_robot_and_can_choose_differently():
    local_chain = {"actions": [action("a", 0, 1), action("b", 0, 1),
                               action("c", 0, 1, ["a"]), action("long", 2, 3)]}
    peer_branch = copy.deepcopy(local_chain)
    peer_branch["actions"][2]["robot"] = 1
    assert schedule(local_chain)["makespan"] == schedule(peer_branch)["makespan"] == 3
    assert score(local_chain, "risk", (2,)) == 0
    assert score(peer_branch, "risk", (2,)) == 0.5
    assert score(local_chain, "total_risk", (2,)) == 3.5
    assert score(peer_branch, "total_risk", (2,)) == 3
    assert select([local_chain, peer_branch], "risk")["index"] == 0
    assert select([local_chain, peer_branch], "total_risk")["index"] == 1


def test_reactive_active_pause_preserves_start_and_adds_duration_only_once():
    plan = {"actions": [action("active", 0, 5), action("tail", 0, 2),
                        action("peer", 1, 1, ["tail"])]}
    result = reactive_pause(plan, robot=0, start=2, duration=3)
    assert result["fault_action"] == "active"
    assert result["additional_duration"] == 3
    assert result["perturbed"]["starts"]["active"] == 0
    assert result["perturbed"]["ends"] == {"active": 8, "tail": 10, "peer": 11}
    assert result["peer_end_shift"] == 3


def test_reactive_idle_gap_only_adds_remaining_unavailability():
    plan = {"actions": [action("before", 0, 2), action("gate", 1, 5),
                        action("future", 0, 2, ["gate"]),
                        action("after", 0, 1), action("peer", 2, 1, ["after"])]}
    result = reactive_pause(plan, robot=0, start=3, duration=4)
    assert result["fault_action"] == "future"
    assert result["additional_duration"] == 2
    assert result["end_shifts"] == {"before": 0, "gate": 0, "future": 2, "after": 2, "peer": 2}
    assert result["perturbed"]["ends"]["future"] == 9
    assert result["peer_end_shift"] == 2


def test_reactive_pause_after_robot_done_or_before_next_start_has_no_effect():
    plan = {"actions": [action("done", 0, 2), action("gate", 1, 8),
                        action("later", 2, 2, ["gate"])]}
    for robot, start, duration in [(0, 2, 10), (2, 3, 4), (2, 3, 5), (9, 1, 4)]:
        result = reactive_pause(plan, robot, start, duration)
        assert result["fault_action"] is None
        assert result["additional_duration"] == 0
        assert result["baseline"] == result["perturbed"]


def test_reactive_pause_never_changes_completed_actions_or_advances_any_action():
    plan = {"actions": [action("done", 0, 2), action("active", 0, 4),
                        action("peer", 1, 2, ["active"]), action("independent", 2, 9)]}
    nominal = schedule(plan)
    for robot in range(3):
        for fraction in (0.25, 0.5, 0.75):
            start = fraction * nominal["makespan"]
            for duration in (3, 6, 12, 24):
                result = reactive_pause(plan, robot, start, duration)
                assert result == reactive_pause(plan, robot, start, duration)
                for action_id, original_end in nominal["ends"].items():
                    assert result["perturbed"]["starts"][action_id] >= nominal["starts"][action_id]
                    assert result["perturbed"]["ends"][action_id] >= original_end
                    if original_end <= start:
                        assert result["end_shifts"][action_id] == 0


@pytest.mark.parametrize("method", ["time_risk", "time_total"])
def test_time_scores_match_manual_reactive_pause_grid(method):
    plan = {"robot_ids": [0, 1, 2],
            "actions": [action("a", 0, 3), action("b", 1, 2, ["a"]),
                        action("c", 2, 4), action("d", 0, 1, ["b", "c"])]}
    horizon = 7.0
    manual_results = [reactive_pause(plan, robot, fraction * horizon, duration)
                      for robot in (0, 1, 2) for fraction in (0.125, 0.375, 0.625, 0.875)
                      for duration in (2, 4, 8)]
    manual = [sum(result["end_shifts"].values()) if method == "time_total"
              else result["peer_end_shift"] for result in manual_results]
    assert score(plan, method, horizon=horizon) == pytest.approx(sum(manual) / 36)
    assert score(plan, method) == score(plan, method, horizon=schedule(plan)["makespan"])


@pytest.mark.parametrize("method", ["time_risk", "time_total", "exposure_risk"])
def test_clock_based_selection_uses_shared_pool_best_horizon(method):
    def candidate(tail_duration):
        return {"robot_ids": [0, 1, 2],
                "actions": [action("early_transfer", 0, 1.3), action("tail", 0, tail_duration),
                            action("receiver", 1, 1, ["early_transfer"]), action("third", 2, 0.1)]}

    slow, fast = candidate(9.7), candidate(8.7)
    # Own-horizon normalization rewards a slower candidate by moving the first
    # time probe from 1.25 to 1.375, past the 1.3-tick transfer; for exposure it
    # changes the denominator without changing the risky action's exposure.
    assert score(slow, method) < score(fast, method)
    common_horizon = min(schedule(plan)["makespan"] for plan in (slow, fast))
    expected_scores = [score(plan, method, horizon=common_horizon) for plan in (slow, fast)]
    assert expected_scores[0] == pytest.approx(expected_scores[1])
    result = select([slow, fast], method)
    assert result["eligible_indices"] == [0, 1]
    assert result["index"] == 1
    assert result["score"] == pytest.approx(expected_scores[1])


def test_exposure_risk_weights_only_busy_overlap_with_common_clock():
    plan = {"robot_ids": [0, 1, 2],
            "actions": [action("short_source", 0, 2),
                        action("long_receiver", 1, 8, ["short_source"])]}
    per_action = action_scores(plan, "risk")
    assert per_action["short_source"] == pytest.approx(14 / 3)
    assert per_action["long_receiver"] == 0
    assert score(plan, "risk") == pytest.approx(7 / 3)
    assert score(plan, "exposure_risk", horizon=10) == pytest.approx((2 * 14 / 3) / (3 * 10))
    assert score(plan, "exposure_risk", horizon=1) == pytest.approx((1 * 14 / 3) / (3 * 1))
    assert select([plan], "exposure_risk")["score"] == score(plan, "exposure_risk")
    # A late source has only its portion inside the shared clock as exposure.
    late = {"robot_ids": [0, 1, 2],
            "actions": [action("prefix", 0, 8), action("source", 0, 4),
                        action("receiver", 1, 1, ["source"])]}
    values = action_scores(late)
    assert score(late, "exposure_risk", horizon=10) == pytest.approx(
        (8 * values["prefix"] + 2 * values["source"]) / 30)


@pytest.mark.parametrize("method", ["time_risk", "time_total", "exposure_risk"])
def test_clock_scores_keep_unused_robot_probability_mass(method):
    inferred_team = {"actions": [action("source", 0, 2), action("peer", 1, 3, ["source"])]}
    fixed_team = dict(inferred_team, robot_ids=[0, 1, 2])
    assert score(inferred_team, method) > 0
    assert score(fixed_team, method) == pytest.approx(score(inferred_team, method) * 2 / 3)


def test_time_total_retains_same_robot_effects_when_peer_effect_is_zero():
    plan = {"robot_ids": [0, 1, 2], "actions": [action("only", 0, 10)]}
    assert score(plan, "time_risk") == 0
    assert score(plan, "time_total") == pytest.approx((2 + 4 + 8) / (3 * 3))


def test_clock_scores_handle_empty_and_zero_duration_plans():
    for plan in ({"actions": []}, {"robot_ids": [0, 1, 2], "actions": [action("instant", 0, 0)]}):
        for method in ("time_risk", "time_total", "exposure_risk"):
            assert score(plan, method) == 0


def test_clock_scores_reject_incomplete_robot_metadata():
    with pytest.raises(ValueError, match="include every action"):
        score({"robot_ids": [0], "actions": [action("a", 1, 1)]}, "time_risk")
