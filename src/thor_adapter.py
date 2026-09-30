"""Small, synchronous AI2-THOR multi-agent skill adapter.

Navigation is an oracle teleport abstraction following Scale_Plan's public
GoToObject_teleport, not a physical navigation experiment. Baseline forceAction
is explicit and recorded. No recovery, goal scoring, or metadata mutation occurs.
Importing/constructing this class does not start Unity; call initialize().
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random
import threading
from typing import Any


@dataclass(frozen=True)
class ActionResult:
    success: bool
    error: str
    action: str
    agent_id: int
    object_id: str | None
    trace_index: int

    def to_dict(self) -> dict:
        return asdict(self)


class ThorAdapter:
    SKILLS = {
        "PickupObject": "PickupObject", "PutObject": "PutObject",
        "OpenObject": "OpenObject", "CloseObject": "CloseObject",
        "SwitchOn": "ToggleObjectOn", "SwitchOff": "ToggleObjectOff",
        "ToggleObjectOn": "ToggleObjectOn", "ToggleObjectOff": "ToggleObjectOff",
        "SliceObject": "SliceObject", "BreakObject": "BreakObject",
        "CleanObject": "CleanObject", "DirtyObject": "DirtyObject",
        "DropHandObject": "DropHandObject", "ThrowObject": "ThrowObject",
        "PushObject": "PushObject", "PullObject": "PullObject",
    }
    FORCE_ACTIONS = set(SKILLS.values()) | {"Teleport"}

    def __init__(self, scene: str | int, agent_count: int = 2, seed: int = 0,
                 baseline_force: bool = True, controller=None,
                 visibility_distance: float = 100.0, width: int = 300,
                 height: int = 300):
        if agent_count < 1:
            raise ValueError("agent_count must be positive")
        self.scene = f"FloorPlan{scene}" if isinstance(scene, int) else scene
        self.agent_count, self.seed = agent_count, seed
        self.baseline_force = baseline_force
        self.visibility_distance = visibility_distance
        self.width, self.height = width, height
        self.controller = controller
        self.trace: list[dict[str, Any]] = []
        self.reachable_positions: list[dict] = []
        self._lock = threading.RLock()
        self.initialized = False

    def initialize(self):
        try:
            return self._initialize_impl()
        except Exception:
            # Context-manager entry failures do not invoke __exit__.
            self.close()
            raise

    def _initialize_impl(self):
        with self._lock:
            if self.initialized:
                return self
            if self.controller is None:
                from ai2thor.controller import Controller
                from ai2thor.platform import CloudRendering
                import ai2thor.build
                # Public Unity artifact endpoint supports HTTPS; avoid HTTP403.
                ai2thor.build.base_url = ai2thor.build.base_url.replace("http://", "https://")
                self.controller = Controller(platform=CloudRendering,
                    scene=self.scene, width=self.width, height=self.height,
                    agentMode="default", agentCount=self.agent_count,
                    gridSize=0.25, snapToGrid=True, fieldOfView=90,
                    visibilityDistance=self.visibility_distance,
                    renderDepthImage=False, renderInstanceSegmentation=False)
            else:
                self.controller.reset(self.scene)
                result = self.step("Initialize", agentCount=self.agent_count,
                    agentMode="default", gridSize=0.25, snapToGrid=True,
                    fieldOfView=90, visibilityDistance=self.visibility_distance)
                if not result.success:
                    raise RuntimeError(result.error)
            self.initialized = True
            positions = self.step("GetReachablePositions")
            if not positions.success:
                raise RuntimeError(f"GetReachablePositions: {positions.error}")
            self.reachable_positions = deepcopy(self.metadata(0).get("actionReturn") or [])
            if not self.reachable_positions:
                raise RuntimeError("Scene returned no reachable positions")
            # Spread agents deterministically; report rather than hide collisions.
            candidates = sorted(self.reachable_positions,
                key=lambda p: (p["x"], p["z"], p["y"]))
            random.Random(self.seed).shuffle(candidates)
            assigned = []
            for agent_id in range(self.agent_count):
                placed = False
                for position in candidates:
                    if any(self._dist(position, other) < 0.55 for other in assigned):
                        continue
                    result = self.step("Teleport", agent_id, position=position,
                        rotation={"x": 0, "y": (agent_id * 90) % 360, "z": 0},
                        horizon=0, standing=True, forceAction=False)
                    if result.success:
                        assigned.append(position)
                        placed = True
                        break
                if not placed:
                    raise RuntimeError(f"No valid separated spawn for agent {agent_id}")
            return self

    @staticmethod
    def _dist(a, b):
        return math.hypot(a["x"] - b["x"], a["z"] - b["z"])

    def _check_agent(self, agent_id):
        if not isinstance(agent_id, int) or not 0 <= agent_id < self.agent_count:
            raise ValueError(f"Invalid agent id: {agent_id}")

    def metadata(self, agent_id: int = 0) -> dict:
        self._check_agent(agent_id)
        with self._lock:
            if self.controller is None:
                raise RuntimeError("Call initialize() first")
            event = self.controller.last_event
            events = getattr(event, "events", None)
            if events is not None:
                if len(events) <= agent_id:
                    raise RuntimeError("Simulator event is missing requested agent")
                return deepcopy(events[agent_id].metadata)
            if self.agent_count != 1:
                raise RuntimeError("Expected multi-agent event")
            return deepcopy(event.metadata)

    def objects(self, agent_id: int = 0) -> list[dict]:
        return self.metadata(agent_id).get("objects", [])

    def resolve_object(self, target: str, agent_id: int = 0) -> dict:
        objects = self.objects(agent_id)
        exact = [obj for obj in objects if obj["objectId"] == target]
        if exact:
            return exact[0]
        matches = [obj for obj in objects
                   if obj.get("objectType", "").lower() == target.lower()]
        if not matches:
            raise LookupError(f"Object not found: {target}")
        pos = self.metadata(agent_id)["agent"]["position"]
        return min(matches, key=lambda o: (self._dist(o["position"], pos), o["objectId"]))

    def state(self) -> dict:
        with self._lock:
            return {"scene": self.scene, "agents": [
                {"agent_id": i, "agent": self.metadata(i).get("agent"),
                 "inventoryObjects": self.metadata(i).get("inventoryObjects", [])}
                for i in range(self.agent_count)], "objects": self.objects(0)}

    def _record(self, action, agent_id, object_id, success, error, parameters, extra=None):
        index = len(self.trace)
        row = {"index": index, "action": action, "agent_id": agent_id,
               "object_id": object_id, "success": bool(success), "error": error,
               "parameters": deepcopy(parameters)}
        if extra:
            row.update(extra)
        self.trace.append(row)
        return ActionResult(bool(success), error, action, agent_id, object_id, index)

    def step(self, action: str, agent_id: int = 0, object_id: str | None = None,
             **kwargs) -> ActionResult:
        self._check_agent(agent_id)
        with self._lock:
            if self.controller is None:
                raise RuntimeError("Call initialize() first")
            parameters = dict(kwargs)
            if object_id is not None:
                parameters["objectId"] = object_id
            if action in self.FORCE_ACTIONS:
                parameters.setdefault("forceAction", self.baseline_force)
            try:
                event = self.controller.step(action=action, agentId=agent_id, **parameters)
                events = getattr(event, "events", None)
                md = events[agent_id].metadata if events is not None else event.metadata
                success = md.get("lastActionSuccess") is True
                error = md.get("errorMessage", "")
                if "lastActionSuccess" not in md:
                    error = "Simulator omitted lastActionSuccess"
                return self._record(action, agent_id, object_id, success, error, parameters,
                    {"actionReturn": deepcopy(md.get("actionReturn")),
                     "agent": deepcopy(md.get("agent")),
                     "inventoryObjects": deepcopy(md.get("inventoryObjects", []))})
            except Exception as exc:
                return self._record(action, agent_id, object_id, False,
                    f"{type(exc).__name__}: {exc}", parameters)

    def goto(self, agent_id: int, target: str, max_tries: int = 30) -> ActionResult:
        """Try nearest reachable approaches; every Unity attempt stays in trace."""
        self._check_agent(agent_id)
        with self._lock:
            try:
                obj = self.resolve_object(target, agent_id)
            except LookupError as exc:
                return self._record("GoToObject", agent_id, target, False, str(exc), {})
            center = obj.get("axisAlignedBoundingBox", {}).get("center") or obj["position"]
            other_positions = [self.metadata(i)["agent"]["position"]
                for i in range(self.agent_count) if i != agent_id]
            candidates = sorted(self.reachable_positions,
                key=lambda p: (self._dist(p, center) +
                    (5.0 if any(self._dist(p, other) < .55 for other in other_positions) else 0),
                    p["x"], p["z"]))
            for position in candidates[:max_tries]:
                yaw = math.degrees(math.atan2(center["x"]-position["x"],
                    center["z"]-position["z"])) % 360
                result = self.step("Teleport", agent_id, position=position,
                    rotation={"x": 0.0, "y": yaw, "z": 0.0}, horizon=30,
                    standing=True)
                if result.success:
                    return self._record("GoToObject", agent_id, obj["objectId"], True, "",
                        {"max_tries": max_tries, "mode": "teleport"},
                        {"distance_xz": self._dist(position, center)})
            return self._record("GoToObject", agent_id, obj["objectId"], False,
                "No teleport candidate succeeded", {"max_tries": max_tries})

    def act(self, skill: str, agent_id: int, target: str | None = None,
            **kwargs) -> ActionResult:
        if skill == "GoToObject":
            return self.goto(agent_id, target, **kwargs)
        if skill not in self.SKILLS:
            raise ValueError(f"Unsupported skill: {skill}")
        with self._lock:
            object_id = None
            if target is not None:
                try:
                    object_id = self.resolve_object(target, agent_id)["objectId"]
                except LookupError as exc:
                    return self._record(skill, agent_id, target, False, str(exc), kwargs)
            if skill == "ThrowObject":
                kwargs.setdefault("moveMagnitude", 7)
            return self.step(self.SKILLS[skill], agent_id, object_id, **kwargs)

    def close(self):
        with self._lock:
            controller, self.controller = self.controller, None
            self.initialized = False
            if controller is not None:
                controller.stop()

    def __enter__(self):
        return self.initialize()

    def __exit__(self, *_):
        self.close()
