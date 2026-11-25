"""Rescue environment core implementation.

Follows DESIGN.md sections 3.3, 3.4, 4, 6, and 7: agent movement, rubble, bumps,
rescue, drop, oxygen mechanics, dynamic fire spreading, triage targeting,
and ASCII rendering.
"""

from __future__ import annotations

from typing import Any
import numpy as np

import config
from src.geometry import Coord, Layout, get_default_layout
from src.state import encode

# Action name mappings for logging and rendering
ACTION_NAMES: dict[int, str] = {
    config.UP: "UP",
    config.DOWN: "DOWN",
    config.LEFT: "LEFT",
    config.RIGHT: "RIGHT",
    config.RESCUE: "RESCUE",
    config.DROP: "DROP",
}

# Coordinate delta for cardinal movement actions
_ACTION_DELTAS: dict[int, tuple[int, int]] = {
    config.UP: (-1, 0),
    config.DOWN: (1, 0),
    config.LEFT: (0, -1),
    config.RIGHT: (0, 1),
}


class RescueEnv:
    """Disaster rescue grid environment."""

    def __init__(
        self,
        seed: int | None = None,
        shaping: bool = False,
        fire_scale: float = config.FIRE_SCALE,
        layout: Layout | None = None,
    ) -> None:
        """Initialize environment with layout, configuration, and random generator."""
        self.layout: Layout = layout if layout is not None else get_default_layout()
        self.shaping: bool = shaping
        self.fire_scale: float = fire_scale
        self.rng: np.random.Generator = np.random.default_rng(seed)

        self.agent_pos: Coord = self.layout.S
        self.oxygen: int = config.OXYGEN_MAX
        self.carrying: int | None = None
        self.victims: list[dict[str, Any]] = []
        self.fire: set[Coord] = set()
        self.steps: int = 0

        self.reset(seed=seed)

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        """Reset environment to initial state per DESIGN 6."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = self.layout.S
        self.oxygen = config.OXYGEN_MAX
        self.carrying = None
        self.steps = 0
        # DESIGN 3.3 & 6: fire restored to 5 initial cells
        self.fire = set(self.layout.initial_fire_cells)

        # DESIGN 2 & 3.3: choose 2 of 4 victims randomly to be injured
        n_victims = len(self.layout.victim_cells)
        injured_indices = set(
            self.rng.choice(n_victims, size=config.N_INJURED, replace=False)
        )

        self.victims = [
            {
                "pos": pos,
                "injured": i in injured_indices,
                "status": "waiting",
            }
            for i, pos in enumerate(self.layout.victim_cells)
        ]

        return self._observe(), self._get_info(death_cause=None)

    def _observe(self) -> tuple[int, ...]:
        """Return 9-int tabular state tuple encoded per DESIGN 5."""
        return encode(self)

    def _get_info(self, death_cause: str | None = None) -> dict[str, Any]:
        """Construct info dictionary with current step metrics."""
        delivered = sum(1 for v in self.victims if v["status"] == "delivered")
        dead_victims = sum(1 for v in self.victims if v["status"] == "dead")
        return {
            "death_cause": death_cause,
            "delivered": delivered,
            "dead_victims": dead_victims,
            "oxygen": self.oxygen,
            "step": self.steps,
        }

    def get_triage_target(self) -> int | None:
        """Return index of triage target victim per DESIGN 3.3, or None if none remain."""
        waiting = [
            (i, v) for i, v in enumerate(self.victims) if v["status"] == "waiting"
        ]
        if not waiting:
            return None

        # Nearest alive uncollected injured first; if none, nearest non-injured
        injured = [item for item in waiting if item[1]["injured"]]
        pool = injured if injured else waiting

        # DESIGN 3.3: Nearest by static cost. Ties break by lowest row, then lowest col
        return min(
            pool,
            key=lambda item: (
                self.layout.cost(self.agent_pos, item[1]["pos"]),
                item[1]["pos"][0],
                item[1]["pos"][1],
            ),
        )[0]

    def get_goal_pos(self) -> Coord | None:
        """Return navigation goal coordinate per DESIGN 5."""
        if self.carrying is not None:
            # Carrying injured -> A; carrying non-injured -> S
            return self.layout.A if self.victims[self.carrying]["injured"] else self.layout.S
        target_idx = self.get_triage_target()
        return self.victims[target_idx]["pos"] if target_idx is not None else None

    def _apply_movement(
        self, action: int
    ) -> tuple[float, int, bool, str | None]:
        """Apply directional movement and return (reward, oxygen_cost, dead, death_cause)."""
        dr, dc = _ACTION_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        target = (nr, nc)

        # DESIGN 6 step 1: bounds and wall bump check
        if not (0 <= nr < self.layout.rows and 0 <= nc < self.layout.cols) or target in self.layout.walls:
            return float(config.R_BUMP), 1, False, None

        # DESIGN 6 step 1: stepping into fire kills agent immediately
        if target in self.fire:
            self.agent_pos = target
            return float(config.R_DEATH), 1, True, "fire"

        # Valid grid move
        self.agent_pos = target
        if self.carrying is not None:
            self.victims[self.carrying]["pos"] = self.agent_pos

        # DESIGN 6 step 1 & DESIGN 7: rubble move vs free move
        if target in self.layout.rubble:
            return float(config.R_RUBBLE), config.RUBBLE_COST, False, None

        return float(config.R_STEP), 1, False, None

    def _apply_rescue(self) -> tuple[float, int]:
        """Execute rescue pickup attempt per DESIGN 6 step 1."""
        if self.carrying is None:
            for idx, v in enumerate(self.victims):
                if v["pos"] == self.agent_pos and v["status"] == "waiting":
                    self.carrying = idx
                    v["status"] = "carried"
                    return float(config.R_RESCUE), 1
        return float(config.R_RESCUE_INVALID), 1

    def _apply_drop(self) -> tuple[float, int]:
        """Execute drop-off delivery attempt per DESIGN 6 step 1."""
        if self.carrying is not None:
            v = self.victims[self.carrying]
            if v["injured"] and self.agent_pos == self.layout.A:
                v["status"] = "delivered"
                v["pos"] = self.layout.A
                self.carrying = None
                return float(config.R_DROP_INJURED), 1
            if (not v["injured"]) and self.agent_pos == self.layout.S:
                v["status"] = "delivered"
                v["pos"] = self.layout.S
                self.carrying = None
                return float(config.R_DROP_NON_INJURED), 1
        return float(config.R_DROP_INVALID), 1

    def _spread_fire(self) -> list[Coord]:
        """Spread fire to eligible neighbor cells per DESIGN 3.4."""
        old_fire = set(self.fire)
        if len(old_fire) >= config.FIRE_CAP:
            return []

        # DESIGN 3.4: Eligible cells are non-burning free cells outside protected zone
        candidates = [
            pos
            for pos in sorted(self.layout.free_cells)
            if pos not in old_fire and pos not in self.layout.protected_cells
        ]

        newly_ignited: list[Coord] = []
        for pos in candidates:
            # DESIGN 3.4: Count burning 4-neighbors in old fire set (simultaneous update)
            k = sum(1 for nbr in self.layout.neighbors4(pos) if nbr in old_fire)
            if k == 0:
                continue

            # DESIGN 3.4: p = P_BASE + P_SLOPE * (d / D_MAX), scaled by FIRE_SCALE
            d = self.layout.manhattan(pos, self.layout.S)
            p = (config.P_BASE + config.P_SLOPE * (d / self.layout.d_max)) * self.fire_scale
            p = min(1.0, max(0.0, p))
            p_ignite = 1.0 - (1.0 - p) ** k

            if self.rng.random() < p_ignite:
                newly_ignited.append(pos)

        # DESIGN 3.4: Cap at 18 burning cells; keep random subset if over cap
        space_left = config.FIRE_CAP - len(old_fire)
        if len(newly_ignited) > space_left:
            chosen_indices = self.rng.choice(
                len(newly_ignited), size=space_left, replace=False
            )
            newly_ignited = [newly_ignited[i] for i in sorted(chosen_indices)]

        self.fire.update(newly_ignited)
        return newly_ignited

    def _check_termination(self, is_dead: bool) -> tuple[bool, bool, float]:
        """Check episode termination and completion bonus per DESIGN 6 step 5."""
        if is_dead:
            return True, False, 0.0

        delivered = sum(1 for v in self.victims if v["status"] == "delivered")
        dead_v = sum(1 for v in self.victims if v["status"] == "dead")
        if delivered + dead_v == len(self.victims):
            # DESIGN 6 step 5: +100 bonus added only if all 4 were delivered alive
            bonus = float(config.R_ALL_DELIVERED) if delivered == len(self.victims) else 0.0
            return True, False, bonus

        if self.steps >= config.STEP_CAP:
            return False, True, 0.0

        return False, False, 0.0

    def step(
        self, action: int
    ) -> tuple[tuple[int, ...], float, bool, bool, dict[str, Any]]:
        """Advance simulation by one action step per DESIGN 6."""
        if action not in ACTION_NAMES:
            raise ValueError(f"Invalid action ID: {action}")

        self.steps += 1
        old_goal = self.get_goal_pos() if self.shaping else None
        old_pos = self.agent_pos

        # DESIGN 6 step 1: Apply action
        if action in _ACTION_DELTAS:
            reward, oxygen_cost, is_dead, death_cause = self._apply_movement(action)
        elif action == config.RESCUE:
            reward, oxygen_cost = self._apply_rescue()
            is_dead, death_cause = False, None
        else:
            reward, oxygen_cost = self._apply_drop()
            is_dead, death_cause = False, None

        # DESIGN 6 step 2: Oxygen consumption and refill priority
        if not is_dead:
            self.oxygen -= oxygen_cost
            if self.agent_pos == self.layout.S:
                # DESIGN 6 step 2: refill beats death check
                self.oxygen = config.OXYGEN_MAX
            elif self.oxygen <= 0:
                is_dead = True
                death_cause = "oxygen"
                reward = float(config.R_DEATH)

        # DESIGN 6 step 3 & 4: Fire spreading and consequences
        if not is_dead:
            self._spread_fire()
            if self.agent_pos in self.fire:
                # DESIGN 6 step 4: fire ignited agent's cell
                is_dead = True
                death_cause = "fire"
                reward = float(config.R_DEATH)
            else:
                # DESIGN 6 step 4: fire ignited waiting victim's cell
                for v in self.victims:
                    if v["status"] == "waiting" and v["pos"] in self.fire:
                        v["status"] = "dead"
                        reward += float(config.R_VICTIM_DIED)

        # DESIGN 6 step 5: Termination checks
        terminated, truncated, bonus = self._check_termination(is_dead)
        reward += bonus

        # DESIGN 7: Potential-based shaping (ablation only)
        if self.shaping and not is_dead:
            phi_old = -float(self.layout.cost(old_pos, old_goal)) if old_goal else 0.0
            new_goal = self.get_goal_pos()
            phi_new = 0.0 if terminated else (-float(self.layout.cost(self.agent_pos, new_goal)) if new_goal else 0.0)
            reward += config.GAMMA * phi_new - phi_old

        return self._observe(), reward, terminated, truncated, self._get_info(death_cause)

    def render_ascii(self) -> str:
        """Render 10x10 ASCII grid showing current environment state."""
        waiting_cells = {
            v["pos"] for v in self.victims if v["status"] == "waiting"
        }
        lines: list[str] = []
        for r in range(self.layout.rows):
            tokens: list[str] = []
            for c in range(self.layout.cols):
                pos = (r, c)
                if pos == self.agent_pos:
                    tokens.append("@")
                elif pos in self.fire:
                    tokens.append("F")
                elif pos in waiting_cells:
                    tokens.append("V")
                elif pos in self.layout.walls:
                    tokens.append("#")
                elif pos in self.layout.rubble:
                    tokens.append("R")
                elif pos == self.layout.S:
                    tokens.append("S")
                elif pos == self.layout.A:
                    tokens.append("A")
                else:
                    tokens.append(".")
            lines.append(" ".join(tokens))
        return "\n".join(lines)


def demo_scripted_actions() -> None:
    """Run scripted core actions demo from Stage 2."""
    print("=== Demo 1: Scripted Core Actions ===")
    demo_seed = 42
    env = RescueEnv(seed=demo_seed)
    obs, info = env.reset(seed=demo_seed)

    print("\nInitial ASCII Map:")
    print(env.render_ascii())
    print(f"\nInitial State: pos={env.agent_pos}, oxygen={env.oxygen}, carrying={env.carrying}")
    print(f"Victims: {env.victims}")
    triage_v = env.get_triage_target()
    print(f"Triage target: V{triage_v + 1 if triage_v is not None else None}")

    scripted_actions = [
        config.RIGHT,   # Step 1: Bump boundary from S (6,9) -> tests bump and oxygen refill on S
        config.DOWN,    # Step 2: Move to (7,9)
        config.DOWN,    # Step 3: Move to (8,9)
        config.LEFT,    # Step 4: Move to (8,8)
        config.LEFT,    # Step 5: Move to (8,7) -> tests rubble move (costs 5 oxygen)
        config.LEFT,    # Step 6: Move to (8,6) -> reach V4
        config.RESCUE,  # Step 7: Rescue V4 -> tests rescue (+20 reward)
        config.RIGHT,   # Step 8: Move to (8,7) -> rubble with carried victim
        config.RIGHT,   # Step 9: Move to (8,8)
        config.DOWN,    # Step 10: Move to (9,8)
        config.RIGHT,   # Step 11: Move to (9,9) -> Ambulance A
        config.DROP,    # Step 12: Drop injured V4 at A -> tests drop (+100 reward)
        config.LEFT,    # Step 13: Move to (9,8)
        config.LEFT,    # Step 14: Move to (9,7)
        config.UP,      # Step 15: Move to (8,7) rubble (costs 5 oxygen)
        config.LEFT,    # Step 16: Move to (8,6)
        config.RIGHT,   # Step 17: Move into rubble (8,7) with < 5 oxygen -> tests death by oxygen
    ]

    print("\nRunning scripted action sequence:")
    print(f"{'Step':<5} | {'Action':<7} | {'Reward':<7} | {'Oxygen':<6} | {'Terminated':<10} | {'Truncated':<9} | {'Info'}")
    print("-" * 75)

    for i, act in enumerate(scripted_actions, start=1):
        obs, reward, term, trunc, inf = env.step(act)
        act_name = ACTION_NAMES[act]
        print(
            f"{i:<5} | {act_name:<7} | {reward:<7.0f} | {env.oxygen:<6} | "
            f"{str(term):<10} | {str(trunc):<9} | death={inf['death_cause']}, deliv={inf['delivered']}"
        )
        if term or trunc:
            break

    print("\nFinal ASCII Map:")
    print(env.render_ascii())
    print(f"\nFinal outcome: terminated={term}, truncated={trunc}, info={inf}")


def demo_fire_spread() -> None:
    """Run 30 steps of harmless invalid actions to demonstrate fire spreading per Stage 3."""
    print("=== Demo 2: Fire Spreading (30 steps) ===")
    demo_seed = 42
    env = RescueEnv(seed=demo_seed)
    env.reset(seed=demo_seed)

    print(f"\nStep  0: burning cells = {len(env.fire)}")
    print(env.render_ascii())

    for step in range(1, 31):
        # DESIGN 4 & 7: DROP at safe zone with empty hands is harmless and invalid (-2 reward, oxygen refilled at S)
        obs, reward, term, trunc, info = env.step(config.DROP)
        if step in (10, 20, 30):
            print(f"\nStep {step:2d}: burning cells = {len(env.fire)}")
            print(env.render_ascii())
        if term or trunc:
            print(f"Terminated early at step {step}: term={term}, trunc={trunc}")
            break


if __name__ == "__main__":
    demo_scripted_actions()
    print("\n" + "=" * 75 + "\n")
    demo_fire_spread()
