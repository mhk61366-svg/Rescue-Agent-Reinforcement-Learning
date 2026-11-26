"""Baseline and learning agents for the Rescue-RL environment.

Implements RandomAgent, GreedyAgent, QLearningAgent, and SarsaAgent
per DESIGN.md section 9.
"""

from __future__ import annotations

import heapq
import pickle
from pathlib import Path
from typing import TYPE_CHECKING
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


class _TabularTDLearner:
    """Shared sparse Q-table, epsilon-greedy policy, and persistence for TD learners."""

    def __init__(
        self,
        alpha: float = config.ALPHA,
        gamma: float = config.GAMMA,
        eps_start: float = config.EPS_START,
        eps_end: float = config.EPS_END,
        eps_decay: float = config.EPS_DECAY,
        seed: int | None = None,
    ) -> None:
        """Initialize tabular learner hyperparameters, Q-table, and RNG."""
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.eps_start: float = eps_start
        self.eps_end: float = eps_end
        self.eps_decay: float = eps_decay
        self.epsilon: float = eps_start
        self.rng: np.random.Generator = np.random.default_rng(seed)
        # DESIGN 9.5: sparse dict mapping 9-int state tuples to 6 zero-initialized Q-values
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}

    def _q_values(self, state: tuple[int, ...]) -> np.ndarray:
        """Return mutable Q-value vector for state, initializing to 0.0 on first visit."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ALL_ACTIONS), dtype=np.float64)
        return self.q_table[state]

    def act(
        self,
        state: tuple[int, ...],
        env: RescueEnv | None = None,
        explore: bool = True,
    ) -> int:
        """Choose an action with epsilon-greedy exploration and random tie-breaking."""
        del env
        if explore and float(self.rng.random()) < self.epsilon:
            return int(self.rng.choice(ALL_ACTIONS))

        q_vals = self._q_values(state)
        best_actions = np.flatnonzero(q_vals == np.max(q_vals))
        return int(self.rng.choice(best_actions))

    def end_episode(self) -> None:
        """Apply multiplicative per-episode epsilon decay down to eps_end."""
        # DESIGN 9.5: epsilon decays multiplicatively each episode with floor EPS_END
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def save(self, path: str | Path) -> None:
        """Save sparse Q-table and epsilon state to a pickle file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "algo": self.__class__.__name__,
        }
        with out_path.open("wb") as fh:
            pickle.dump(payload, fh)

    def load(self, path: str | Path) -> None:
        """Load sparse Q-table (and optional epsilon) from a pickle file."""
        in_path = Path(path)
        with in_path.open("rb") as fh:
            payload = pickle.load(fh)
        if isinstance(payload, dict) and "q_table" in payload:
            self.q_table = payload["q_table"]
            self.epsilon = float(payload.get("epsilon", self.eps_end))
        else:
            self.q_table = payload


class QLearningAgent(_TabularTDLearner):
    """Off-policy tabular Q-learning agent (DESIGN 9.3)."""

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        next_action: int,
        terminated: bool,
    ) -> None:
        """Update Q(s, a) toward max_a' Q(s', a'), bootstrapping on truncation."""
        del next_action
        q_vals = self._q_values(state)
        # DESIGN 8: terminal transitions have target=reward; truncated still bootstraps
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self._q_values(next_state)))
        q_vals[action] += self.alpha * (target - q_vals[action])


class SarsaAgent(_TabularTDLearner):
    """On-policy tabular SARSA agent (DESIGN 9.4)."""

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        next_action: int,
        terminated: bool,
    ) -> None:
        """Update Q(s, a) toward Q(s', a') for chosen next_action, bootstrapping on truncation."""
        q_vals = self._q_values(state)
        # DESIGN 8 & 9.4: use chosen next_action; bootstrap on truncation, not termination
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * float(self._q_values(next_state)[next_action])
        q_vals[action] += self.alpha * (target - q_vals[action])
