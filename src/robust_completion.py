"""Exact fixed-plan completion cost and certified lower bounds for one pause.

The canonical graph and earliest schedule come from ``scheduling`` unchanged.
These functions use the same preempt-and-resume pause convention. Costs are in
the duration units supplied by the plan; this module makes no physics claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

try:
    from . import scheduling
except ImportError:
    import scheduling


@dataclass(frozen=True)
class NominalProfile:
    graph: Any
    starts: tuple[float, ...]
    ends: tuple[float, ...]
    makespan: float
    robot_ids: tuple[int, ...]


@dataclass(frozen=True)
class CompletionProfile:
    makespan: float
    robot_ids: tuple[int, ...]
    action_ids: tuple[str, ...]
    robots: tuple[int, ...]
    starts: tuple[float, ...]
    ends: tuple[float, ...]
    tail: tuple[float, ...]
    terminal_slack: tuple[float, ...]
    by_robot: tuple[tuple[int, ...], ...]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _values(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(_number(v, name) for v in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def prepare_nominal(plan: Mapping[str, Any], robots: Sequence[int] | None = None
                    ) -> NominalProfile:
    """Compile and schedule once; terminal tails can wait until after pruning."""
    graph = scheduling._compile(plan)
    starts, ends, makespan = scheduling._run(graph)
    if robots is None:
        robot_ids = scheduling._robot_ids(plan, graph)
    else:
        robot_ids = tuple(robots)
        if (any(type(r) is not int for r in robot_ids)
                or len(set(robot_ids)) != len(robot_ids)
                or not set(graph.robots).issubset(robot_ids)):
            raise ValueError("robots must uniquely include every action's robot")
    return NominalProfile(graph, tuple(starts), tuple(ends), makespan, robot_ids)


def build_profile_from_nominal(nominal: NominalProfile) -> CompletionProfile:
    """Add terminal tails without repeating graph compilation or scheduling."""
    graph = nominal.graph
    starts, ends, makespan, robot_ids = (nominal.starts, nominal.ends,
                                        nominal.makespan, nominal.robot_ids)
    tail = [0.0] * len(graph.ids)
    for node in reversed(graph.order):
        tail[node] = max((graph.durations[child] + tail[child]
                          for child in graph.successors[node]), default=0.0)
    # Roundoff can only make an analytically nonnegative slack slightly negative.
    slack = tuple(max(0.0, makespan - ends[i] - tail[i]) for i in range(len(tail)))
    by_robot = tuple(tuple(i for i in graph.order if graph.robots[i] == r)
                     for r in robot_ids)
    return CompletionProfile(makespan, robot_ids, graph.ids, graph.robots,
                             tuple(starts), tuple(ends), tuple(tail), slack, by_robot)


def build_profile(plan: Mapping[str, Any], robots: Sequence[int] | None = None
                  ) -> CompletionProfile:
    return build_profile_from_nominal(prepare_nominal(plan, robots))


def pause_increase(profile: CompletionProfile, robot: int, start: float,
                   duration: float) -> float:
    """Exact terminal shift; includes idle pauses and half-open boundaries."""
    start = _number(start, "start")
    duration = _number(duration, "duration")
    if duration == 0 or robot not in profile.robot_ids:
        return 0.0
    end = start + duration
    if not math.isfinite(end):
        raise ValueError("pause end must be finite")
    for node in profile.by_robot[profile.robot_ids.index(robot)]:
        action_start, action_end = profile.starts[node], profile.ends[node]
        if action_start <= start < action_end:
            return max(0.0, duration - profile.terminal_slack[node])
        if start <= action_start < end:
            return max(0.0, end - action_start - profile.terminal_slack[node])
    return 0.0


def score_profile(profile: CompletionProfile, T: float,
                  fractions: Sequence[float], durations: Sequence[float]) -> float:
    """Absolute completion averaged over uniform robots and finite probe grids."""
    T = _number(T, "T")
    fractions = _values(fractions, "fractions")
    durations = _values(durations, "durations")
    if any(f > 1 for f in fractions):
        raise ValueError("fractions must lie in [0, 1]")
    if not profile.robot_ids:
        return profile.makespan
    increase = sum(pause_increase(profile, r, T*f, d)
                   for r in profile.robot_ids for f in fractions for d in durations)
    return profile.makespan + increase / (len(profile.robot_ids)*len(fractions)*len(durations))


def score_by_plan(plan: Mapping[str, Any], T: float,
                  fractions: Sequence[float], durations: Sequence[float],
                  robots: Sequence[int] | None = None) -> dict[str, float]:
    """Return C, J, expected increase, and the certified lower bound on J."""
    profile = build_profile(plan, robots)
    J = score_profile(profile, T, fractions, durations)
    bound = lower_bound(profile.makespan, len(profile.robot_ids), T, fractions, durations)
    return {"C": profile.makespan, "J": J, "increase": J-profile.makespan,
            "lower_bound": bound}


def continuous_score(profile: CompletionProfile, T: float,
                     durations: Sequence[float]) -> float:
    """Exact uniform-clock mean by integrating constant/linear response pieces."""
    T = _number(T, "T")
    durations = _values(durations, "durations")
    if T == 0:
        raise ValueError("continuous scoring requires T > 0")
    if not profile.robot_ids:
        return profile.makespan
    integral = 0.0
    for nodes in profile.by_robot:
        previous_end = 0.0
        for node in nodes:
            s, e, slack = profile.starts[node], profile.ends[node], profile.terminal_slack[node]
            busy = max(0.0, min(T, e)-min(T, s))
            lo, hi = min(T, previous_end), min(T, s)
            for d in durations:
                integral += busy * max(0.0, d-slack)
                if hi > lo:
                    integral += (max(0.0, hi+d-s-slack)**2
                                 - max(0.0, lo+d-s-slack)**2) / 2
            previous_end = e
    return profile.makespan + integral/(len(profile.robot_ids)*T*len(durations))


def lower_bound(C: float, m: int, T: float, fractions: Sequence[float],
                durations: Sequence[float]) -> float:
    """C + E[duration]/m * Pr(onset < C), valid also when search finds C < T.

    At every onset strictly before nominal completion, some positive-duration
    critical-path action is active. Pausing that robot adds the full duration.
    The onset-at-completion endpoint is intentionally excluded.
    """
    C, T = _number(C, "C"), _number(T, "T")
    fractions, durations = _values(fractions, "fractions"), _values(durations, "durations")
    if any(f > 1 for f in fractions):
        raise ValueError("fractions must lie in [0, 1]")
    if type(m) is not int or m < 0:
        raise ValueError("m must be a nonnegative integer")
    if m == 0:
        if C != 0:
            raise ValueError("positive makespan needs at least one robot")
        return 0.0
    probability = sum(T*f < C for f in fractions)/len(fractions)
    return C + sum(durations)/len(durations)/m * probability


def continuous_lower_bound(C: float, m: int, T: float,
                           durations: Sequence[float]) -> float:
    C, T = _number(C, "C"), _number(T, "T")
    durations = _values(durations, "durations")
    if T == 0 or type(m) is not int or m <= 0:
        raise ValueError("continuous lower bound requires T > 0 and m > 0")
    return C + sum(durations)/len(durations)/m * min(1.0, C/T)


def continuous_nominal_cutoff(incumbent_J: float, m: int, T: float,
                              durations: Sequence[float]) -> float:
    """Largest nominal C not rejected by the continuous-clock lower bound.

    Use the worst retained beam objective as incumbent_J to preserve top-b
    search trajectories. A candidate must exceed this cutoff strictly (plus
    numerical tolerance) to be rejected; equality can affect tie-breaking.
    """
    incumbent_J, T = _number(incumbent_J, "incumbent_J"), _number(T, "T")
    durations = _values(durations, "durations")
    if T == 0 or type(m) is not int or m <= 0:
        raise ValueError("continuous cutoff requires T > 0 and m > 0")
    b = sum(durations)/len(durations)/m
    return incumbent_J/(1+b/T) if incumbent_J <= T+b else incumbent_J-b


def can_prune(C: float, incumbent_J: float, m: int, T: float,
              fractions: Sequence[float], durations: Sequence[float],
              tolerance: float = 1e-9) -> bool:
    """Prune only a strictly inferior lower bound, preserving equal-score ties."""
    incumbent_J = _number(incumbent_J, "incumbent_J")
    tolerance = _number(tolerance, "tolerance")
    return lower_bound(C, m, T, fractions, durations) > incumbent_J + tolerance
