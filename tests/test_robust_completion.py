"""Independent replay checks for the new terminal scoring and safe pruning."""
import random

import pytest

from src.robust_completion import (build_profile, pause_increase, score_by_plan,
    score_profile, continuous_score, lower_bound, continuous_lower_bound,
    continuous_nominal_cutoff, can_prune)
from src.scheduling import reactive_pause, replay


def action(i, robot, duration, deps=()):
    return {"id": str(i), "robot": robot, "duration": duration,
            "deps": [str(v) for v in deps]}


def random_plan(seed):
    rng = random.Random(seed)
    actions = []
    for i in range(9):
        deps = [j for j in range(i) if rng.random() < .15]
        actions.append(action(i, rng.randrange(3), rng.randrange(6), deps))
    return {"actions": actions, "robot_ids": [0, 1, 2, 3]}


def test_terminal_slack_matches_full_dag_replay():
    for seed in range(30):
        plan = random_plan(seed)
        profile = build_profile(plan)
        for i, aid in enumerate(profile.action_ids):
            for extra in (0, .25, 3, 17):
                expected = max(0, extra-profile.terminal_slack[i])
                assert replay(plan, {aid: extra})["makespan_increase"] == pytest.approx(expected)


def test_pause_score_matches_replay_at_boundaries_and_idle_times():
    for seed in range(20):
        plan = random_plan(seed)
        p = build_profile(plan)
        onsets = set(p.starts+p.ends+(0., p.makespan*1.5))
        onsets.update(t+.125 for t in list(onsets))
        for robot in p.robot_ids:
            for onset in onsets:
                for duration in (0, .5, 8):
                    actual = pause_increase(p, robot, onset, duration)
                    expected = reactive_pause(plan, robot, onset, duration)["makespan_increase"]
                    assert actual == pytest.approx(expected)


def test_discrete_and_continuous_bounds_cover_horizons_beyond_completion():
    fractions, durations = (0., .25, .5, .75, 1.), (1., 3., 7.)
    for seed in range(20):
        p = build_profile(random_plan(seed))
        for ratio in (.5, 1., 1.5):
            T = max(.1, p.makespan*ratio)
            assert score_profile(p, T, fractions, durations) >= lower_bound(
                p.makespan, len(p.robot_ids), T, fractions, durations)-1e-9
            assert continuous_score(p, T, durations) >= continuous_lower_bound(
                p.makespan, len(p.robot_ids), T, durations)-1e-9


def test_continuous_formula_against_independent_numerical_replay():
    plan = {"robot_ids": [0, 1, 2], "actions": [
        action('a', 0, 4), action('b', 1, 3, ['a']),
        action('c', 2, 12), action('d', 1, 2)]}
    p = build_profile(plan)
    T, durations, n = 15., (2., 5.), 2000
    numerical = sum(reactive_pause(plan, r, T*(i+.5)/n, d)["perturbed"]["makespan"]
                    for r in p.robot_ids for d in durations for i in range(n))/(3*2*n)
    assert continuous_score(p, T, durations) == pytest.approx(numerical, abs=.005)


def test_floor_is_tight_and_nominal_cutoff_preserves_pool_optimum():
    serial = {"robot_ids": [0, 1, 2], "actions": [action('a', 0, 10)]}
    p = build_profile(serial)
    assert continuous_score(p, 10., (3.,)) == 11.
    assert continuous_score(p, 20., (3.,)) == 10.5
    fractions, durations, T = (.125, .375, .625, .875), (2., 4., 8.), 20.
    rows = [score_by_plan(random_plan(seed), T, fractions, durations) for seed in range(40)]
    incumbent = rows[0]['J']
    kept = [rows[0]]
    for row in rows[1:]:
        if not can_prune(row['C'], incumbent, 4, T, fractions, durations):
            kept.append(row)
            incumbent = min(incumbent, row['J'])
    assert min(r['J'] for r in kept) == min(r['J'] for r in rows)
    bound = lower_bound(10, 3, 10, fractions, durations)
    assert not can_prune(10, bound, 3, 10, fractions, durations)


def test_invalid_configuration_is_rejected():
    p = build_profile(random_plan(0))
    with pytest.raises(ValueError):
        score_profile(p, 10., (), (1.,))
    with pytest.raises(ValueError):
        score_profile(p, 10., (1.1,), (1.,))
    with pytest.raises(ValueError):
        continuous_score(p, 0., (1.,))
    with pytest.raises(ValueError):
        build_profile(random_plan(0), [0, 1])


def test_continuous_cutoff_is_exact_inverse_on_both_branches():
    for incumbent in (0., 4., 10., 11., 12., 50.):
        cutoff = continuous_nominal_cutoff(incumbent, 3, 10., (3., 6.))
        assert continuous_lower_bound(cutoff, 3, 10., (3., 6.)) == pytest.approx(incumbent)
        assert continuous_lower_bound(cutoff+.001, 3, 10., (3., 6.)) > incumbent
