"""Deterministic fixed-plan scheduling and finite-delay propagation in abstract ticks.

The module models precedence and robot serialization, not robot physics, collision
avoidance, repair, or measured wall-clock time. It requires only the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import random
from typing import Any, Mapping, Sequence


DEFAULT_DELTA_GRID = (2.0, 4.0, 8.0)
TIME_PAUSE_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
_EPS = 1e-9


@dataclass(frozen=True)
class _Graph:
    ids: tuple[str, ...]
    robots: tuple[int, ...]
    durations: tuple[float, ...]
    predecessors: tuple[tuple[int, ...], ...]
    successors: tuple[tuple[int, ...], ...]
    order: tuple[int, ...]
    handoffs: int


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _compile(plan: Mapping[str, Any]) -> _Graph:
    actions = plan["actions"]
    ids = tuple(action["id"] for action in actions)
    if any(not isinstance(action_id, str) or not action_id for action_id in ids):
        raise ValueError("Every action needs a nonempty string id")
    if len(set(ids)) != len(ids):
        raise ValueError("Action ids must be unique")
    indices = {action_id: i for i, action_id in enumerate(ids)}
    robots = tuple(action["robot"] for action in actions)
    if any(not isinstance(robot, int) or isinstance(robot, bool) for robot in robots):
        raise ValueError("Robot ids must be integers")
    durations = tuple(_number(action["duration"], "duration") for action in actions)
    predecessors = [set() for _ in actions]
    last_by_robot: dict[int, int] = {}
    handoffs = 0
    for i, action in enumerate(actions):
        for dep in set(action.get("deps", [])):
            if dep not in indices:
                raise ValueError(f"Unknown dependency {dep!r} for {ids[i]!r}")
            predecessor = indices[dep]
            predecessors[i].add(predecessor)
            handoffs += robots[predecessor] != robots[i]
        if robots[i] in last_by_robot:
            predecessors[i].add(last_by_robot[robots[i]])
        last_by_robot[robots[i]] = i
    successors = [[] for _ in actions]
    for child, parents in enumerate(predecessors):
        for parent in sorted(parents):
            successors[parent].append(child)
    indegree = [len(parents) for parents in predecessors]
    ready = [i for i, degree in enumerate(indegree) if not degree]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for child in successors[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(actions):
        raise ValueError("Plan contains a cycle after adding robot-order edges")
    return _Graph(ids, robots, durations,
                  tuple(tuple(sorted(parents)) for parents in predecessors),
                  tuple(tuple(children) for children in successors),
                  tuple(order), handoffs)


def _outage(value: Mapping[str, Any] | None) -> tuple[int, float, float] | None:
    if value is None:
        return None
    robot = value["robot"]
    if not isinstance(robot, int) or isinstance(robot, bool):
        raise ValueError("Outage robot must be an integer")
    start = _number(value["start"], "outage start")
    duration = _number(value["duration"], "outage duration")
    return robot, start, start + duration


def _run(graph: _Graph, additions: Sequence[float] | None = None,
         outage: tuple[int, float, float] | None = None
         ) -> tuple[list[float], list[float], float]:
    starts = [0.0] * len(graph.ids)
    ends = [0.0] * len(graph.ids)
    for node in graph.order:
        start = max((ends[parent] for parent in graph.predecessors[node]), default=0.0)
        duration = graph.durations[node] + (additions[node] if additions else 0.0)
        if outage is not None:
            robot, window_start, window_end = outage
            # Half-open intervals: boundary contact and zero duration do not overlap.
            if (graph.robots[node] == robot and duration > 0
                    and window_end > window_start and start < window_end
                    and start + duration > window_start):
                start = window_end
        starts[node] = start
        ends[node] = start + duration
    return starts, ends, max(ends, default=0.0)


def _as_schedule(graph: _Graph, result: tuple[list[float], list[float], float]
                 ) -> dict[str, Any]:
    starts, ends, makespan = result
    return {"starts": dict(zip(graph.ids, starts)),
            "ends": dict(zip(graph.ids, ends)), "makespan": makespan}


def _additions(graph: _Graph, extra_durations: Mapping[str, float] | None) -> list[float]:
    values = extra_durations or {}
    unknown = set(values).difference(graph.ids)
    if unknown:
        raise ValueError(f"Unknown delayed action ids: {sorted(unknown)}")
    return [_number(values.get(action_id, 0.0), "extra duration") for action_id in graph.ids]


def schedule(plan: Mapping[str, Any],
             extra_durations: Mapping[str, float] | None = None,
             outage: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Earliest-start fixed-plan schedule; all times are abstract ticks.

    Explicit dependencies and consecutive actions of each robot both create edges.
    An outage is ``{'robot': int, 'start': float, 'duration': float}``. An action
    intersecting that window waits in full until it ends (no partial execution).
    """
    graph = _compile(plan)
    return _as_schedule(graph, _run(graph, _additions(graph, extra_durations), _outage(outage)))


def replay(plan: Mapping[str, Any],
           extra_durations: Mapping[str, float] | None = None,
           fault_robot: int | None = None,
           outage: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compare a deterministic perturbed schedule to its nominal schedule.

    ``peer_end_shift`` sums positive end-time shifts on robots OTHER than the
    fault robot. ``impacted_peer_actions`` counts those shifts above 1e-9 ticks.
    Infer the fault robot from positive delays / outage when unique. Multiple
    fault robots require an explicit reference ``fault_robot`` for peer metrics.
    """
    graph = _compile(plan)
    additions = _additions(graph, extra_durations)
    window = _outage(outage)
    fault_robots = {graph.robots[i] for i, delay in enumerate(additions) if delay > 0}
    if window is not None and window[2] > window[1]:
        fault_robots.add(window[0])
    if fault_robot is None:
        if len(fault_robots) > 1:
            raise ValueError("Multiple fault robots: specify fault_robot for peer metrics")
        fault_robot = next(iter(fault_robots), None)
    elif not isinstance(fault_robot, int) or isinstance(fault_robot, bool):
        raise ValueError("fault_robot must be an integer")
    nominal = _run(graph)
    perturbed = _run(graph, additions, window)
    shifts = [max(0.0, end - original)
              for end, original in zip(perturbed[1], nominal[1])]
    peer_shifts = [shift for i, shift in enumerate(shifts) if graph.robots[i] != fault_robot]
    return {"baseline": _as_schedule(graph, nominal),
            "perturbed": _as_schedule(graph, perturbed),
            "fault_robot": fault_robot,
            "makespan_increase": max(0.0, perturbed[2] - nominal[2]),
            "peer_end_shift": sum(peer_shifts),
            "impacted_peer_actions": sum(shift > _EPS for shift in peer_shifts),
            "end_shifts": dict(zip(graph.ids, shifts))}


def reactive_pause(plan: Mapping[str, Any], robot: int, start: float,
                   duration: float) -> dict[str, Any]:
    """Replay an exogenous robot availability loss with preempt-and-resume.

    At ``start``, an active action retains its state and remaining work while
    paused, so its end is delayed by the full pause duration. If the robot is
    idle, only the first future action starting before the pause ends receives
    the remaining unavailability. A single duration addition then propagates
    along the fixed DAG; robot serialization prevents double-counting the pause.

    There is no anticipatory waiting before ``start``. In the idle case, initial
    waiting is folded into the affected action's duration, so the returned start
    is its dispatch time, not its first productive execution time. All ticks are
    abstract; retained state is an assumption, not a simulated physical outcome.
    Extra result fields: ``fault_action`` (id or None), ``additional_duration``.
    """
    if not isinstance(robot, int) or isinstance(robot, bool):
        raise ValueError("robot must be an integer")
    pause_start = _number(start, "pause start")
    pause_duration = _number(duration, "pause duration")
    pause_end = pause_start + pause_duration
    if not math.isfinite(pause_end):
        raise ValueError("pause end must be finite")
    graph = _compile(plan)
    starts, ends, _ = _run(graph)
    fault_action = None
    additional_duration = 0.0
    if pause_duration > 0:
        for node in graph.order:
            if graph.robots[node] != robot:
                continue
            if starts[node] <= pause_start < ends[node]:
                fault_action = graph.ids[node]
                additional_duration = pause_duration
                break
            if pause_start <= starts[node] < pause_end:
                fault_action = graph.ids[node]
                additional_duration = pause_end - starts[node]
                break
    additions = {fault_action: additional_duration} if fault_action is not None else None
    result = replay(plan, extra_durations=additions, fault_robot=robot)
    result["fault_action"] = fault_action
    result["additional_duration"] = additional_duration
    return result


def _grid(delta_grid: Sequence[float]) -> tuple[float, ...]:
    deltas = tuple(_number(delta, "delta") for delta in delta_grid)
    if not deltas:
        raise ValueError("delta_grid must not be empty")
    return deltas


def _action_scores(graph: _Graph, method: str, deltas: tuple[float, ...]) -> dict[str, float]:
    if method not in {"risk", "total_risk", "structural"}:
        raise ValueError("Action scores support 'risk', 'total_risk', and 'structural'")
    nominal_ends = _run(graph)[1]
    scores = {}
    for source, action_id in enumerate(graph.ids):
        if method == "structural":
            descendants: set[int] = set()
            pending = list(graph.successors[source])
            while pending:
                node = pending.pop()
                if node not in descendants:
                    descendants.add(node)
                    pending.extend(graph.successors[node])
            peer_count = sum(graph.robots[node] != graph.robots[source] for node in descendants)
            scores[action_id] = peer_count * sum(deltas) / len(deltas)
        else:
            total = 0.0
            additions = [0.0] * len(graph.ids)
            for delta in deltas:
                additions[source] = delta
                ends = _run(graph, additions)[1]
                total += sum(max(0.0, end - nominal_ends[node])
                             for node, end in enumerate(ends)
                             if method == "total_risk" or graph.robots[node] != graph.robots[source])
            scores[action_id] = total / len(deltas)
    return scores


def action_scores(plan: Mapping[str, Any], method: str = "risk",
                  delta_grid: Sequence[float] = DEFAULT_DELTA_GRID) -> dict[str, float]:
    """Per-action mean end-time shift for a fixed, predeclared delay grid.

    ``risk`` counts peers; ``total_risk`` counts all robots, including the source.
    """
    return _action_scores(_compile(plan), method, _grid(delta_grid))


def _free_slack(graph: _Graph) -> float:
    starts, ends, makespan = _run(graph)
    return sum(max(0.0, min((starts[child] for child in graph.successors[node]),
                           default=makespan) - ends[node])
               for node in range(len(graph.ids)))


def _robot_ids(plan: Mapping[str, Any], graph: _Graph) -> tuple[int, ...]:
    robots = tuple(plan.get("robot_ids", sorted(set(graph.robots))))
    if any(not isinstance(robot, int) or isinstance(robot, bool) for robot in robots):
        raise ValueError("robot_ids must contain integers")
    if len(set(robots)) != len(robots):
        raise ValueError("robot_ids must be unique")
    if not set(graph.robots).issubset(robots):
        raise ValueError("robot_ids must include every action's robot")
    return tuple(sorted(robots))


def _time_score(plan: Mapping[str, Any], graph: _Graph, method: str,
                horizon: float, deltas: tuple[float, ...]) -> float:
    robots = _robot_ids(plan, graph)
    if not robots or horizon == 0:
        return 0.0
    total = 0.0
    for robot in robots:
        for fraction in TIME_PAUSE_FRACTIONS:
            for delta in deltas:
                result = reactive_pause(plan, robot, fraction * horizon, delta)
                total += (sum(result["end_shifts"].values()) if method == "time_total"
                          else result["peer_end_shift"])
    return total / (len(robots) * len(TIME_PAUSE_FRACTIONS) * len(deltas))


def score(plan: Mapping[str, Any], method: str = "risk",
          delta_grid: Sequence[float] = DEFAULT_DELTA_GRID,
          horizon: float | None = None) -> float:
    """Mean over action scores; never consumes held-out evaluation faults.

    'structural' counts reachable peer actions times delay, ignoring slack.
    'total_risk' averages end-time shifts over all robots, including the source.
    'slack' returns SUM free slack (larger is better), with no inserted buffers.
    'time_risk' averages peer effects of reactive pauses at TIME_PAUSE_FRACTIONS
    of horizon; 'time_total' uses the identical probes and counts all robots.
    'exposure_risk' weights action risk by busy time within [0, horizon] and
    divides by team size times horizon, retaining idle/unused robot mass.
    These three methods use plan['robot_ids'] when present and otherwise infer
    robots from actions; their omitted horizon defaults to nominal makespan.
    'nominal' returns makespan; 'handoffs' counts explicit cross-robot edges.
    """
    graph = _compile(plan)
    if method == "nominal":
        return _run(graph)[2]
    if method == "handoffs":
        return float(graph.handoffs)
    if method == "slack":
        return _free_slack(graph)
    deltas = _grid(delta_grid)
    if method in {"time_risk", "time_total", "exposure_risk"}:
        common_horizon = _run(graph)[2] if horizon is None else _number(horizon, "horizon")
        if method in {"time_risk", "time_total"}:
            return _time_score(plan, graph, method, common_horizon, deltas)
        robots = _robot_ids(plan, graph)
        if not robots or common_horizon == 0:
            return 0.0
        values = _action_scores(graph, "risk", deltas)
        starts, ends, _ = _run(graph)
        exposure_sum = sum(max(0.0, min(ends[i], common_horizon) - max(starts[i], 0.0))
                           * values[action_id] for i, action_id in enumerate(graph.ids))
        return exposure_sum / (len(robots) * common_horizon)
    values = _action_scores(graph, method, deltas).values()
    return sum(values) / len(graph.ids) if graph.ids else 0.0


def handoff_count(plan: Mapping[str, Any]) -> int:
    """Count unique explicit cross-robot dependency edges, excluding serial edges."""
    return _compile(plan).handoffs


def select(plans: Sequence[Mapping[str, Any]], method: str,
           nominal_slack: float = 0.10, seed: int = 0,
           delta_grid: Sequence[float] = DEFAULT_DELTA_GRID) -> dict[str, Any]:
    """Select from a shared pool with a nominal makespan budget.

    Methods: nominal/risk/time_risk/time_total/exposure_risk/total_risk/slack/
    handoffs/structural/random. Clock-based scores use the same pool-best nominal
    makespan as their horizon for EVERY candidate, not each candidate's duration.
    ``slack`` maximizes total free slack; other scored methods minimize.
    Non-nominal methods use
    makespan <= (1 + nominal_slack) * best makespan. Score ties break by nominal
    makespan then original pool index. Random samples eligible indices using a
    local seeded generator. Returned ``plan`` is the original object, unmodified.
    """
    methods = {"nominal", "risk", "time_risk", "time_total", "exposure_risk", "total_risk",
               "slack", "handoffs", "structural", "random"}
    if method not in methods:
        raise ValueError(f"Unknown selection method: {method!r}")
    if not plans:
        raise ValueError("Candidate pool must not be empty")
    slack = _number(nominal_slack, "nominal_slack")
    graphs = [_compile(plan) for plan in plans]
    makespans = [_run(graph)[2] for graph in graphs]
    best = min(makespans)
    threshold = (1.0 + slack) * best
    eligible = [i for i, makespan in enumerate(makespans) if makespan <= threshold + _EPS]
    scores: dict[int, float] = {}
    if method == "nominal":
        index = min(range(len(plans)), key=lambda i: (makespans[i], i))
        chosen_score: float | None = makespans[index]
    elif method == "random":
        index = random.Random(seed).choice(eligible)
        chosen_score = None
    else:
        deltas = (_grid(delta_grid) if method in {
            "risk", "time_risk", "time_total", "exposure_risk", "total_risk", "structural"
        } else ())
        for i in eligible:
            graph = graphs[i]
            if method == "handoffs":
                scores[i] = float(graph.handoffs)
            elif method == "slack":
                scores[i] = _free_slack(graph)
            elif method in {"time_risk", "time_total", "exposure_risk"}:
                # The common clock prevents changing the fault distribution
                # merely by stretching a candidate's own nominal makespan.
                scores[i] = score(plans[i], method, delta_grid=deltas, horizon=best)
            else:
                values = _action_scores(graph, method, deltas).values()
                scores[i] = sum(values) / len(graph.ids) if graph.ids else 0.0
        direction = -1 if method == "slack" else 1
        index = min(eligible, key=lambda i: (direction * scores[i], makespans[i], i))
        chosen_score = scores[index]
    return {"index": index, "plan": plans[index], "makespan": makespans[index],
            "score": chosen_score, "eligible_indices": eligible,
            "nominal_best": best, "nominal_limit": threshold, "method": method}
