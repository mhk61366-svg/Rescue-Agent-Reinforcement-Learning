"""Policy evaluation utility and baseline comparison demo.
Follows DESIGN.md sections 8 and 9.1-9.2.
"""

from __future__ import annotations

from typing import Any

import config
from src.agents import GreedyAgent, RandomAgent
from src.env import RescueEnv


def evaluate(
    agent: Any,
    n_episodes: int = config.EVAL_EPISODES,
    seed: int = 0,
    shaping: bool = False,
    fire_scale: float = config.FIRE_SCALE,
) -> dict[str, float]:
    """Run n_episodes greedily (explore=False) and return summary metrics."""
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")

    env = RescueEnv(seed=seed, shaping=shaping, fire_scale=fire_scale)
    successes = 0
    total_return = 0.0
    total_delivered = 0
    fire_deaths = 0
    oxygen_deaths = 0
    total_steps = 0

    for ep in range(n_episodes):
        # Deterministic per-episode seed so different agents see identical layouts/fires
        state, info = env.reset(seed=seed + ep)
        ep_return = 0.0
        done = False

        while not done:
            action = agent.act(state, env=env, explore=False)
            state, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        # DESIGN 8: success is all 4 victims delivered alive
        if info["delivered"] == config.N_VICTIMS:
            successes += 1
        if info["death_cause"] == "fire":
            fire_deaths += 1
        elif info["death_cause"] == "oxygen":
            oxygen_deaths += 1

        total_return += ep_return
        total_delivered += int(info["delivered"])
        total_steps += int(info["step"])

    return {
        "success_rate": successes / n_episodes,
        "mean_return": total_return / n_episodes,
        "mean_delivered": total_delivered / n_episodes,
        "fire_death_rate": fire_deaths / n_episodes,
        "oxygen_death_rate": oxygen_deaths / n_episodes,
        "mean_steps": total_steps / n_episodes,
    }


def _print_comparison_table(results: dict[str, dict[str, float]], n_episodes: int) -> None:
    """Print formatted baseline evaluation metrics table."""
    print(f"=== Baseline Evaluation ({n_episodes} episodes each) ===")
    header = (
        f"{'Agent':<10} | {'Success':>8} | {'Mean Ret':>9} | "
        f"{'Delivered':>9} | {'Fire Death':>10} | {'O2 Death':>8} | {'Steps':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<10} | {m['success_rate']*100:>7.1f}% | {m['mean_return']:>9.2f} | "
            f"{m['mean_delivered']:>9.2f} | {m['fire_death_rate']*100:>9.1f}% | "
            f"{m['oxygen_death_rate']*100:>7.1f}% | {m['mean_steps']:>7.1f}"
        )


if __name__ == "__main__":
    eval_episodes = 200
    eval_seed = 1000

    baselines = {
        "Random": RandomAgent(seed=eval_seed),
        "Greedy": GreedyAgent(),
    }
    metrics = {
        name: evaluate(agent, n_episodes=eval_episodes, seed=eval_seed)
        for name, agent in baselines.items()
    }
    _print_comparison_table(metrics, eval_episodes)
