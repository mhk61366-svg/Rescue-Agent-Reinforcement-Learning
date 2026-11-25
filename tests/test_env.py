"""Rule tests for RescueEnv (Stage 2 rules). Run: python -m pytest tests/test_env.py -q

All tests use fire_scale=0.0 so they stay valid after fire spreading is added in Stage 3.
Expected numbers are written as literals on purpose: they come from DESIGN.md, not from config.py.
"""
import config
from src.env import RescueEnv

UP, DOWN, LEFT, RIGHT = config.UP, config.DOWN, config.LEFT, config.RIGHT
RESCUE, DROP = config.RESCUE, config.DROP
S = (6, 9)  # safe zone
A = (9, 9)  # ambulance

def fresh(seed=0):
    env = RescueEnv(seed=seed, fire_scale=0.0)
    env.reset()
    return env

def place(env, pos, oxygen=30):
    env.agent_pos = pos
    env.oxygen = oxygen

def carry(env, idx, injured):
    env.victims[idx]["injured"] = injured
    env.victims[idx]["status"] = "carried"
    env.carrying = idx

def set_flags(env, injured_flags):
    for v, flag in zip(env.victims, injured_flags):
        v["injured"] = flag

# ---------- reset ----------
def test_reset_state():
    env = fresh()
    assert env.agent_pos == S
    assert env.oxygen == 30
    assert env.carrying is None
    assert len(env.victims) == 4
    assert all(v["status"] == "waiting" for v in env.victims)

def test_exactly_two_injured_every_reset():
    for seed in range(30):
        env = fresh(seed)
        assert sum(v["injured"] for v in env.victims) == 2

def test_injured_flags_vary_and_are_reproducible():
    patterns = {tuple(v["injured"] for v in fresh(s).victims) for s in range(20)}
    assert len(patterns) > 1                      # flags are random across seeds
    a = tuple(v["injured"] for v in fresh(5).victims)
    b = tuple(v["injured"] for v in fresh(5).victims)
    assert a == b                                 # same seed, same flags

# ---------- movement, bump, rubble ----------
def test_move_into_free_cell():
    env = fresh()
    _, r, term, trunc, _ = env.step(DOWN)
    assert (r, env.oxygen, env.agent_pos, term, trunc) == (-1, 29, (7, 9), False, False)

def test_bump_off_grid_costs_oxygen_away_from_s():
    env = fresh()
    place(env, (7, 9))
    _, r, term, _, _ = env.step(RIGHT)
    assert (r, env.oxygen, env.agent_pos, term) == (-2, 29, (7, 9), False)

def test_bump_into_wall():
    env = fresh()
    place(env, (5, 3))                            # (4,3) is a wall
    _, r, _, _, _ = env.step(UP)
    assert (r, env.oxygen, env.agent_pos) == (-2, 29, (5, 3))

def test_rubble_move_costs_5_reward_and_5_oxygen():
    env = fresh()
    place(env, (8, 8))                            # (8,7) is rubble
    _, r, term, _, _ = env.step(LEFT)
    assert (r, env.oxygen, env.agent_pos, term) == (-5, 25, (8, 7), False)

def test_rubble_with_5_or_less_oxygen_is_fatal():
    env = fresh()
    place(env, (8, 8), oxygen=5)                  # 5 - 5 = 0 outside S -> death
    _, r, term, _, info = env.step(LEFT)
    assert (r, term, info["death_cause"]) == (-100, True, "oxygen")

def test_rubble_with_6_oxygen_survives():
    env = fresh()
    place(env, (8, 8), oxygen=6)
    _, r, term, _, _ = env.step(LEFT)
    assert (r, env.oxygen, term) == (-5, 1, False)

# ---------- oxygen: refill priority, ambulance ----------
def test_arriving_on_s_with_one_oxygen_is_safe_and_refills():
    env = fresh()
    place(env, (7, 9), oxygen=1)
    _, r, term, _, _ = env.step(UP)               # onto S
    assert (env.agent_pos, env.oxygen, term, r) == (S, 30, False, -1)

def test_running_out_one_step_from_s_is_death():
    env = fresh()
    place(env, (7, 9), oxygen=1)
    _, r, term, _, info = env.step(DOWN)          # away from S, oxygen hits 0
    assert (r, term, info["death_cause"]) == (-100, True, "oxygen")

def test_ambulance_does_not_refill():
    env = fresh()
    place(env, (9, 8), oxygen=10)
    env.step(RIGHT)                               # onto A
    assert (env.agent_pos, env.oxygen) == (A, 9)

# ---------- fire (initial cells only in Stage 2) ----------
def test_entering_fire_is_death():
    env = fresh()
    place(env, (5, 5))                            # (5,4) starts on fire
    _, r, term, _, info = env.step(LEFT)
    assert (r, term, info["death_cause"]) == (-100, True, "fire")

# ---------- rescue ----------
def test_rescue_valid():
    env = fresh()
    idx = 3                                       # victim at (8,6)
    place(env, env.victims[idx]["pos"])
    _, r, _, _, _ = env.step(RESCUE)
    assert (r, env.oxygen, env.carrying) == (20, 29, idx)
    assert env.victims[idx]["status"] == "carried"

def test_rescue_invalid_when_no_victim_here():
    env = fresh()
    place(env, (7, 9))
    _, r, _, _, _ = env.step(RESCUE)
    assert (r, env.oxygen, env.carrying) == (-2, 29, None)

def test_rescue_invalid_when_already_carrying():
    env = fresh()
    place(env, env.victims[3]["pos"])
    env.step(RESCUE)
    _, r, _, _, _ = env.step(RESCUE)              # same cell, victim already picked up
    assert r == -2 and env.carrying == 3

# ---------- drop ----------
def test_drop_injured_at_ambulance():
    env = fresh()
    carry(env, 0, injured=True)
    place(env, A)
    _, r, term, _, info = env.step(DROP)
    assert (r, term, env.carrying, info["delivered"]) == (100, False, None, 1)
    assert env.victims[0]["status"] == "delivered"

def test_drop_non_injured_at_safe_zone():
    env = fresh()
    carry(env, 1, injured=False)
    place(env, S)
    _, r, term, _, _ = env.step(DROP)
    assert (r, term, env.carrying) == (50, False, None)

def test_drop_in_wrong_place_is_invalid():
    env = fresh()
    carry(env, 0, injured=True)
    place(env, S)                                 # injured must go to A
    _, r, _, _, _ = env.step(DROP)
    assert r == -2 and env.carrying == 0
    carry(env, 1, injured=False)
    place(env, A)                                 # non-injured must go to S
    _, r, _, _, _ = env.step(DROP)
    assert r == -2 and env.carrying == 1


def test_drop_with_nothing_carried_is_invalid():
    env = fresh()
    place(env, S)
    _, r, _, _, _ = env.step(DROP)
    assert r == -2

# ---------- ending the episode ----------
def test_all_delivered_gives_completion_bonus():
    env = fresh()
    for i in range(3):
        env.victims[i]["status"] = "delivered"
    carry(env, 3, injured=False)
    place(env, S)
    _, r, term, _, info = env.step(DROP)
    assert (r, term, info["delivered"]) == (150, True, 4)   # +50 drop, +100 bonus

def test_no_bonus_if_a_victim_died():
    env = fresh()
    env.victims[0]["status"] = "dead"
    env.victims[1]["status"] = "delivered"
    env.victims[2]["status"] = "delivered"
    carry(env, 3, injured=False)
    place(env, S)
    _, r, term, _, _ = env.step(DROP)
    assert (r, term) == (50, True)                # episode ends, but no +100

def test_step_cap_truncates_without_extra_reward():
    env = fresh()
    for _ in range(149):                          # invalid rescues on S: no death, oxygen refills
        _, _, term, trunc, _ = env.step(RESCUE)
        assert not term and not trunc
    _, r, term, trunc, _ = env.step(RESCUE)       # step 150
    assert (r, term, trunc) == (-2, False, True)

def test_info_keys():
    env = fresh()
    _, _, _, _, info = env.step(DOWN)
    for key in ("death_cause", "delivered", "dead_victims", "oxygen", "step"):
        assert key in info

# ---------- triage ----------
def test_triage_prefers_injured_even_when_farther():
    env = fresh()
    set_flags(env, [False, False, False, True])   # only V4 (8,6) injured; V1 (1,8) is closer
    assert env.get_triage_target() == 3
    env.victims[3]["status"] = "delivered"        # no injured left -> nearest non-injured
    assert env.get_triage_target() == 0           # V1, S->V cost 6

def test_triage_ignores_dead_and_delivered_victims():
    env = fresh()
    set_flags(env, [True, False, False, True])
    env.victims[0]["status"] = "dead"
    assert env.get_triage_target() == 3