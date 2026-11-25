"""Tabular state encoding for the Rescue-RL environment.
Encodes the environment into a compact 9-integer egocentric tuple following
DESIGN.md section 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import config
from src.geometry import Coord

if TYPE_CHECKING:
    from src.env import RescueEnv

# DESIGN 5: Neighbor cell feature codes (features 1-4)
NEIGHBOR_FREE = 0
NEIGHBOR_WALL = 1
NEIGHBOR_RUBBLE = 2
NEIGHBOR_FIRE = 3

# DESIGN 5: Agent task mode codes (feature 5)
MODE_EMPTY_INJURED = 0
MODE_EMPTY_NON_INJURED = 1
MODE_CARRY_INJURED = 2
MODE_CARRY_NON_INJURED = 3

# Human-readable labels for the 9 state features in DESIGN 5 order
FEATURE_NAMES: tuple[str, ...] = (
    "up_neighbor",
    "down_neighbor",
    "left_neighbor",
    "right_neighbor",
    "mode",
    "slack_bucket",
    "goal_dcol_sign",
    "goal_drow_sign",
    "goal_dist_bucket",
)


def neighbor_code(env: RescueEnv, pos: Coord) -> int:
    """Return DESIGN 5 neighbor code (0=free, 1=wall/edge, 2=rubble, 3=fire)."""
    r, c = pos
    # DESIGN 5: Grid boundary or wall cell maps to 1
    if not (0 <= r < env.layout.rows and 0 <= c < env.layout.cols) or pos in env.layout.walls:
        return NEIGHBOR_WALL
    # DESIGN 5: Fire must be checked before free ground because burning cells are free cells
    if pos in env.fire:
        return NEIGHBOR_FIRE
    if pos in env.layout.rubble:
        return NEIGHBOR_RUBBLE
    return NEIGHBOR_FREE


def resolve_mode_and_goal(env: RescueEnv) -> tuple[int, Coord | None]:
    """Return (mode, goal_pos) based on carrying status and triage target."""
    if env.carrying is not None:
        injured = env.victims[env.carrying]["injured"]
        # DESIGN 5: carrying injured goes to A (mode 2); non-injured goes to S (mode 3)
        if injured:
            return MODE_CARRY_INJURED, env.layout.A
        return MODE_CARRY_NON_INJURED, env.layout.S

    target_idx = env.get_triage_target()
    if target_idx is None:
        # All victims resolved (terminal step); no waiting target remains
        return MODE_EMPTY_NON_INJURED, None

    target = env.victims[target_idx]
    mode = MODE_EMPTY_INJURED if target["injured"] else MODE_EMPTY_NON_INJURED
    return mode, target["pos"]


def compute_slack(env: RescueEnv) -> int:
    """Compute oxygen slack relative to the required static return cost."""
    # DESIGN 5: carrying injured requires reaching A then returning from A to S
    if env.carrying is not None and env.victims[env.carrying]["injured"]:
        required = env.layout.cost(env.agent_pos, env.layout.A) + env.layout.cost(
            env.layout.A, env.layout.S
        )
    else:
        required = env.layout.cost(env.agent_pos, env.layout.S)
    return env.oxygen - required


def bucketize(value: int, cutoffs: tuple[int, ...]) -> int:
    """Map an integer value to a discrete bucket using inclusive upper cutoffs."""
    # DESIGN 5: cutoffs define upper bounds for buckets 0..len(cutoffs)-1
    for idx, upper in enumerate(cutoffs):
        if value <= upper:
            return idx
    return len(cutoffs)


def _sign(delta: int) -> int:
    """Return -1, 0, or +1 indicating the sign of an integer difference."""
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def encode(env: RescueEnv) -> tuple[int, ...]:
    """Encode environment into the 9-int state tuple specified in DESIGN 5."""
    r, c = env.agent_pos

    # DESIGN 5 features 1-4: up, down, left, right neighbor codes
    up = neighbor_code(env, (r - 1, c))
    down = neighbor_code(env, (r + 1, c))
    left = neighbor_code(env, (r, c - 1))
    right = neighbor_code(env, (r, c + 1))

    # DESIGN 5 feature 5 & 6: mode and oxygen slack bucket
    mode, goal = resolve_mode_and_goal(env)
    slack_bucket = bucketize(compute_slack(env), config.SLACK_BINS)

    # DESIGN 5 features 7-9: goal column sign, goal row sign, goal distance bucket
    if goal is None:
        d_col, d_row, dist_bucket = 0, 0, 0
    else:
        d_col = _sign(goal[1] - c)
        d_row = _sign(goal[0] - r)
        goal_dist = env.layout.cost(env.agent_pos, goal)
        dist_bucket = bucketize(goal_dist, config.DIST_BINS)

    return (up, down, left, right, mode, slack_bucket, d_col, d_row, dist_bucket)


def _print_state_details(label: str, env: RescueEnv, state: tuple[int, ...]) -> None:
    """Print labeled state tuple along with underlying goal and slack values."""
    _, goal = resolve_mode_and_goal(env)
    slack = compute_slack(env)
    goal_dist = env.layout.cost(env.agent_pos, goal) if goal is not None else 0

    print(f"\n--- {label} ---")
    print(
        f"pos={env.agent_pos}, oxygen={env.oxygen}, carrying={env.carrying}, "
        f"goal={goal}, goal_dist={goal_dist}, raw_slack={slack}"
    )
    print(f"state={state}")
    for name, val in zip(FEATURE_NAMES, state):
        print(f"  {name:<18}: {val}")


def demo_state_encoding() -> None:
    """Demonstrate state encoding at reset and across scripted moves."""
    from src.env import ACTION_NAMES, RescueEnv

    demo_seed = 42
    env = RescueEnv(seed=demo_seed)
    state, _ = env.reset(seed=demo_seed)

    print("=== Stage 4 Demo: State Encoding ===")
    print(env.render_ascii())
    _print_state_details("Reset (step 0)", env, state)

    # Scripted route from S (6,9) to injured V4 at (8,6) via rubble (8,7), then rescue toward A
    scripted_actions = [
        config.DOWN,    # (7,9)
        config.DOWN,    # (8,9)
        config.LEFT,    # (8,8) -> next to rubble (8,7)
        config.LEFT,    # (8,7) -> step onto rubble (-5 oxygen)
        config.LEFT,    # (8,6) -> reach V4
        config.RESCUE,  # pick up V4 -> mode switches to carrying injured, goal becomes A
        config.RIGHT,   # (8,7) -> back through rubble toward A
    ]

    for step_idx, act in enumerate(scripted_actions, start=1):
        state, reward, term, trunc, _ = env.step(act)
        _print_state_details(
            f"Step {step_idx}: {ACTION_NAMES[act]} (reward={reward:.0f})", env, state
        )
        if term or trunc:
            break


if __name__ == "__main__":
    demo_state_encoding()
