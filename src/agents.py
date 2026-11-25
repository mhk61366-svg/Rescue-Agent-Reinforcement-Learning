"""Baseline and learning agents for the Rescue-RL environment.
Stage 5 implements RandomAgent and GreedyAgent per DESIGN.md sections 9.1 and 9.2.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Any
import numpy as np

import config
from src.geometry import Coord
from src.state import compute_slack

if TYPE_CHECKING:
    from src.env import RescueEnv

# Cardinal movement actions paired with coordinate deltas in action-ID order
_MOVE_DELTAS: tuple[tuple[int, int, int], ...] = (
    (config.UP, -1, 0),
    (config.DOWN, 1, 0),
    (config.LEFT, 0, -1),
    (config.RIGHT, 0, 1),
)

ALL_ACTIONS: tuple[int, ...] = (
    config.UP,
    config.DOWN,
    config.LEFT,
    config.RIGHT,
    config.RESCUE,
    config.DROP,
)


class RandomAgent:
    """Uniform random baseline over the 6 discrete actions (DESIGN 9.1)."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize agent-owned random number generator."""
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def act(
        self,
        state: tuple[int, ...],
        env: RescueEnv | None = None,
        explore: bool = True,
    ) -> int:
        """Sample a uniform random action regardless of state."""
        del state, env, explore
        return int(self.rng.choice(ALL_ACTIONS))


class GreedyAgent:
    """Hand-coded shortest-path baseline avoiding fire (DESIGN 9.2)."""

    def _can_drop(self, env: RescueEnv) -> bool:
        """Return True if DROP is currently valid at agent_pos."""
        if env.carrying is None:
            return False
        victim = env.victims[env.carrying]
        # DESIGN 6 step 1: injured drop at A, non-injured drop at S
        if victim["injured"]:
            return env.agent_pos == env.layout.A
        return env.agent_pos == env.layout.S

    def _can_rescue(self, env: RescueEnv) -> bool:
        """Return True if RESCUE is currently valid at agent_pos."""
        if env.carrying is not None:
            return False
        return any(
            v["pos"] == env.agent_pos and v["status"] == "waiting"
            for v in env.victims
        )

    def _select_target(self, env: RescueEnv) -> Coord:
        """Choose target coordinate following triage, drop-off, and oxygen safety rules."""
        if env.carrying is not None:
            victim = env.victims[env.carrying]
            return env.layout.A if victim["injured"] else env.layout.S

        # DESIGN 9.2: return to S first if empty-handed and slack <= GREEDY_SAFETY_MARGIN
        if (
            env.agent_pos != env.layout.S
            and compute_slack(env) <= config.GREEDY_SAFETY_MARGIN
        ):
            return env.layout.S

        goal = env.get_goal_pos()
        return goal if goal is not None else env.layout.S

    def _next_action_dijkstra(self, env: RescueEnv, target: Coord) -> int | None:
        """Return first action on cheapest fire-free path to target, or None if unreachable."""
        start = env.agent_pos
        if start == target:
            return None

        rows, cols = env.layout.rows, env.layout.cols
        blocked = env.layout.walls | env.fire
        dist: dict[Coord, int] = {start: 0}
        pq: list[tuple[int, int, Coord]] = []

        # Seed priority queue with valid neighbor steps tagged by their initial action
        for action_id, dr, dc in _MOVE_DELTAS:
            nr, nc = start[0] + dr, start[1] + dc
            nbr = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and nbr not in blocked:
                step_cost = config.RUBBLE_COST if nbr in env.layout.rubble else 1
                dist[nbr] = step_cost
                heapq.heappush(pq, (step_cost, action_id, nbr))

        while pq:
            cost_so_far, first_act, pos = heapq.heappop(pq)
            if pos == target:
                return first_act
            if cost_so_far > dist[pos]:
                continue

            for _, dr, dc in _MOVE_DELTAS:
                nr, nc = pos[0] + dr, pos[1] + dc
                nbr = (nr, nc)
                if 0 <= nr < rows and 0 <= nc < cols and nbr not in blocked:
                    # DESIGN 9.2: Dijkstra uses real oxygen costs (rubble=5, free=1)
                    step_cost = config.RUBBLE_COST if nbr in env.layout.rubble else 1
                    new_cost = cost_so_far + step_cost
                    if new_cost < dist.get(nbr, float("inf")):
                        dist[nbr] = new_cost
                        heapq.heappush(pq, (new_cost, first_act, nbr))

        return None

    def _fallback_safe_action(self, env: RescueEnv) -> int:
        """Pick any non-fatal adjacent move if target is unreachable or already at S."""
        r, c = env.agent_pos
        blocked = env.layout.walls | env.fire
        best_rubble: int | None = None
        for action_id, dr, dc in _MOVE_DELTAS:
            nr, nc = r + dr, c + dc
            nbr = (nr, nc)
            if 0 <= nr < env.layout.rows and 0 <= nc < env.layout.cols and nbr not in blocked:
                if nbr not in env.layout.rubble:
                    return action_id
                if best_rubble is None:
                    best_rubble = action_id
        return best_rubble if best_rubble is not None else config.UP

    def act(
        self,
        state: tuple[int, ...],
        env: RescueEnv | None = None,
        explore: bool = True,
    ) -> int:
        """Choose greedy shortest-path action using live environment state."""
        del state, explore
        if env is None:
            raise ValueError("GreedyAgent requires env to plan paths.")

        # DESIGN 9.2: take rescue and drop whenever valid
        if self._can_drop(env):
            return config.DROP
        if self._can_rescue(env):
            return config.RESCUE

        target = self._select_target(env)
        action = self._next_action_dijkstra(env, target)
        if action is not None:
            return action

        # If target is blocked by fire, fall back to returning to S if possible
        if target != env.layout.S:
            retreat_action = self._next_action_dijkstra(env, env.layout.S)
            if retreat_action is not None:
                return retreat_action

        return self._fallback_safe_action(env)
