"""Fire spreading tests (Stage 3). Run: python -m pytest tests/test_fire.py -q
Cell sets below are copied from the validated map in DESIGN.md 3.1 on purpose, so these tests
do not trust the code under test to tell them where walls, rubble or the safe zone are.
"""
import config
from src.env import RescueEnv

RESCUE = config.RESCUE
S, A = (6, 9), (9, 9)
PROTECTED = {S, (5, 9), (7, 9), (6, 8), A, (8, 9), (9, 8)}
RUBBLE = {(4, 6), (8, 7)}
WALLS = {(4, 3), (4, 4), (6, 2), (6, 3), (7, 5), (7, 6), (8, 4), (9, 5), (9, 6)}
INITIAL_FIRE = {(0, 6), (2, 3), (4, 1), (5, 4), (9, 3)}


def fresh(seed=0, scale=1.0):
    env = RescueEnv(seed=seed, fire_scale=scale)
    env.reset()
    return env

def test_initial_fire_cells():
    assert set(fresh().fire) == INITIAL_FIRE

def test_zero_scale_never_spreads():
    env = fresh(0, scale=0.0)
    for _ in range(60):                           # invalid rescues on S: agent stays safe
        env.step(RESCUE)
        assert set(env.fire) == INITIAL_FIRE

def test_fire_rules_hold_over_many_episodes():
    """Permanent, capped at 18, never on walls, rubble, S, A or their neighbors."""
    sizes = []
    for seed in range(20):
        env = fresh(seed, scale=5.0)              # fast fire so the rules get stressed
        prev = set(env.fire)
        for _ in range(150):
            _, _, term, trunc, _ = env.step(RESCUE)
            cur = set(env.fire)
            assert prev <= cur                    # fire never goes out
            assert len(cur) <= 18                 # cap
            assert not (cur & (PROTECTED | RUBBLE | WALLS))
            sizes.append(len(cur))
            prev = cur
            if term or trunc:
                break
    assert max(sizes) == 18                       # the cap is actually reached

def test_same_seed_same_fire_history():
    a, b = fresh(3), fresh(3)
    for _ in range(40):
        a.step(RESCUE)
        b.step(RESCUE)
        assert set(a.fire) == set(b.fire)

def test_different_seeds_give_different_fire():
    outcomes = set()
    for seed in range(10):
        env = fresh(seed, scale=1.0)
        for _ in range(15):
            env.step(RESCUE)
        outcomes.add(frozenset(env.fire))
    assert len(outcomes) > 1

def test_far_cells_ignite_more_often_than_near_cells():
    """DESIGN 3.4: ignition probability grows with distance from S (single burning neighbor)."""
    env = fresh(0, scale=1.0)
    n = 3000
    near = far = 0
    for _ in range(n):
        env.reset()
        env.fire = {(5, 6)}                       # (6,6) is 3 from S: p about 0.036
        env.step(RESCUE)
        near += (6, 6) in env.fire
        env.reset()
        env.fire = {(0, 1)}                       # (0,0) is 15 from S: p = 0.10
        env.step(RESCUE)
        far += (0, 0) in env.fire
    assert far > 1.8 * near

def test_victim_burns_when_surrounded_by_fire():
    env = fresh(0, scale=5.0)
    env.fire = {(3, 4), (2, 5), (4, 5), (3, 6)}   # all 4 neighbors of V2 at (3,5)
    for _ in range(20):
        _, r, term, _, info = env.step(RESCUE)
        if env.victims[1]["status"] == "dead":
            assert r == -52                       # -2 invalid rescue + -50 victim burned
            assert info["dead_victims"] == 1
            break
    else:
        raise AssertionError("victim never burned")

def test_agent_dies_when_fire_reaches_its_cell():
    env = fresh(0, scale=5.0)
    env.agent_pos = (2, 6)
    env.oxygen = 30
    env.fire = {(1, 6), (3, 6), (2, 5), (2, 7)}   # all 4 neighbors of the agent
    for _ in range(30):
        _, r, term, _, info = env.step(RESCUE)
        if term:
            break
    assert term and r == -100 and info["death_cause"] == "fire"