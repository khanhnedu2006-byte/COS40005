"""Môi trường RL: xe đi từ A tới B, chọn cạnh tiếp theo, tối đa điểm POI trong ngân sách thời gian."""
import numpy as np

from busroute.utils.timeslots import slot_of

POI_PASS_RADIUS = 0          # TODO: cho phép tính POI khi đi qua node lân cận (theo pass_road)
TIME_PENALTY_PER_MIN = 0.01  # phạt nhỏ theo thời gian để tránh đi vòng vô ích
ARRIVAL_BONUS = 10.0         # thưởng khi tới B trong ngân sách
FAIL_PENALTY = -20.0         # phạt khi hết giờ hoặc kẹt


class TourRouteEnv:
    """State: node hiện tại, thời gian đã đi, giờ trong ngày, POI đã ghé. Action: chỉ số cạnh đi ra."""

    def __init__(self, nodes, edges, start_node, end_node, budget_min, start_minute, max_degree=8):
        self.nodes, self.edges = nodes, edges
        self.start, self.end = start_node, end_node
        self.budget_s, self.start_minute = budget_min * 60, start_minute
        self.max_degree = max_degree
        # Danh sách cạnh đi ra của từng node (theo chỉ số dòng trong edges)
        self.out_edges = edges.groupby("from_idx").groups
        self.poi_score = dict(zip(nodes.node_idx, nodes.poi_score.fillna(0)))
        self.reset()

    def reset(self):
        self.current, self.elapsed_s = self.start, 0.0
        self.visited_pois, self.path = set(), [self.start]
        return self._obs()

    def minute_of_day(self):
        return int(self.start_minute + self.elapsed_s // 60) % (24 * 60)

    def candidate_edges(self):
        return list(self.out_edges.get(self.current, []))[: self.max_degree]

    def action_mask(self):
        # Che cạnh bị cấm ở khung giờ hiện tại (đã lọc whitelist từ trước)
        slot = slot_of(self.minute_of_day())
        mask = np.zeros(self.max_degree, dtype=bool)
        for i, e in enumerate(self.candidate_edges()):
            mask[i] = not bool(self.edges.at[e, f"blocked_{slot}"])
        return mask

    def step(self, action):
        cands = self.candidate_edges()
        assert self.action_mask()[action], "Agent chọn cạnh bị che"
        e = cands[action]
        slot = slot_of(self.minute_of_day())
        dt = float(self.edges.at[e, f"travel_time_{slot}"])
        self.elapsed_s += dt
        self.current = int(self.edges.at[e, "to_idx"])
        self.path.append(self.current)
        reward = -TIME_PENALTY_PER_MIN * dt / 60
        if self.poi_score.get(self.current, 0) and self.current not in self.visited_pois:
            reward += self.poi_score[self.current]       # mỗi POI chỉ tính 1 lần
            self.visited_pois.add(self.current)
        done = False
        if self.current == self.end:
            reward += ARRIVAL_BONUS
            done = True
        elif self.elapsed_s > self.budget_s or not self.action_mask().any():
            reward += FAIL_PENALTY
            done = True
        return self._obs(), reward, done, {"elapsed_min": self.elapsed_s / 60}

    def _obs(self):
        return {
            "current": self.current,
            "time_left_ratio": max(0.0, 1 - self.elapsed_s / self.budget_s),
            "minute_of_day": self.minute_of_day(),
            "visited": sorted(self.visited_pois),
            "mask": self.action_mask(),
        }
