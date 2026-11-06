"""Slot-aware reinforcement-learning environment for an ODMatrix."""
from __future__ import annotations

import heapq
import math
from functools import lru_cache

from busroute.routing.orienteering import ODMatrix
from busroute.utils.timeslots import slot_of


class ODRouteEnv:
    """Choose point IDs; all travel and POI accounting follows ODMatrix.evaluate.

    The return-time mask uses shortest paths with the arrival slot frozen. It is
    only an estimate: subsequent legs can enter different slots or become invalid.
    """

    def __init__(self, matrix: ODMatrix, start: str, end: str, start_minute: int,
                 budget_s: float, time_penalty: float = 0.01,
                 arrival_bonus: float = 10.0, fail_penalty: float = -20.0):
        if start == end or start not in matrix.points or end not in matrix.points:
            raise ValueError("Start and end must be distinct known points")
        if not 0 <= start_minute < 1440 or not math.isfinite(budget_s) or budget_s <= 0:
            raise ValueError("Invalid departure time or budget")
        self.matrix, self.start, self.end = matrix, start, end
        self.start_minute, self.budget_s = start_minute, budget_s
        self.time_penalty, self.arrival_bonus, self.fail_penalty = time_penalty, arrival_bonus, fail_penalty
        self.points = tuple(sorted(matrix.points))
        self.reset()

    @lru_cache(maxsize=5)
    def _return_times(self, slot: str) -> dict[str, float]:
        reverse: dict[str, list[tuple[str, float]]] = {}
        for (u, v, s), leg in self.matrix.legs.items():
            if s == slot:
                reverse.setdefault(v, []).append((u, leg.duration_s))
        distances = {self.end: 0.0}
        queue = [(0.0, self.end)]
        while queue:
            distance, point = heapq.heappop(queue)
            if distance != distances[point]:
                continue
            for prev, cost in reverse.get(point, []):
                candidate = distance + cost
                if candidate < distances.get(prev, math.inf):
                    distances[prev] = candidate
                    heapq.heappush(queue, (candidate, prev))
        return distances

    def reset(self) -> dict:
        self.current, self.elapsed_s = self.start, 0.0
        self.collected = {self.start} & self.matrix.scores.keys()
        self.route = [self.start]
        self.done = False
        return self.observation()

    def slot(self, elapsed_s: float | None = None) -> str:
        elapsed = self.elapsed_s if elapsed_s is None else elapsed_s
        return slot_of(int(self.start_minute + elapsed // 60) % 1440)

    def action_mask(self) -> dict[str, bool]:
        mask = dict.fromkeys(self.points, False)
        if self.done:
            return mask
        slot = self.slot()
        for point in self.points:
            if point == self.current or (point != self.end and point not in self.matrix.scores):
                continue
            if point in self.collected:
                continue
            leg = self.matrix.legs.get((self.current, point, slot))
            if leg is None or self.elapsed_s + leg.duration_s > self.budget_s:
                continue
            remaining = self.budget_s - self.elapsed_s - leg.duration_s
            estimate = self._return_times(self.slot(self.elapsed_s + leg.duration_s)).get(point, math.inf)
            if estimate <= remaining:
                mask[point] = True
        return mask

    def observation(self) -> dict:
        return {"current": self.current, "destination": self.end,
                "elapsed_s": self.elapsed_s, "remaining_s": self.budget_s - self.elapsed_s,
                "minute_of_day": int(self.start_minute + self.elapsed_s // 60) % 1440,
                "slot": self.slot(), "collected": frozenset(self.collected),
                "mask": self.action_mask()}

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if self.done or not self.action_mask().get(action, False):
            raise ValueError(f"Invalid or masked action: {action}")
        leg = self.matrix.legs[self.current, action, self.slot()]
        new_pois = (set(leg.passed) | ({action} & self.matrix.scores.keys())) - self.collected
        self.collected.update(new_pois)
        self.elapsed_s += leg.duration_s
        self.current = action
        self.route.append(action)
        reward = sum(self.matrix.scores[p] for p in new_pois) - self.time_penalty * leg.duration_s / 60
        success = action == self.end
        self.done = success
        if success:
            reward += self.arrival_bonus
        elif not any(self.action_mask().values()):
            reward += self.fail_penalty
            self.done = True
        return self.observation(), reward, self.done, {"success": success, "route": tuple(self.route)}
