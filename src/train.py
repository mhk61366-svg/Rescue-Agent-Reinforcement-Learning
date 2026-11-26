"""Training loop for tabular Q-learning and SARSA agents.
Implements DESIGN.md sections 8 and 9.3-9.5: episodic TD updates with
truncation bootstrapping, periodic greedy evaluation, CSV logging, and
saved Q-tables in results/raw/.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Any
import config
from src.agents import QLearningAgent, SarsaAgent
from src.env import RescueEnv
from src.evaluate import evaluate

RAW_RESULTS_DIR = Path("results") / "raw"
EVAL_SEED_BASE = 100_000
PROGRESS_INTERVAL = 1000

CSV_FIELDS: tuple[str, ...] = (
    "episode",
    "return",
    "steps",
    "delivered",
    "death_cause",
    "epsilon",
    "eval_success_rate",
    "eval_mean_return",
    "eval_mean_delivered",
    "eval_fire_death_rate",
    "eval_oxygen_death_rate",
    "eval_mean_steps",
)


def build_learner(
    algo: str,
    seed: int,
    alpha: float = config.ALPHA,
    eps_decay: float = config.EPS_DECAY,
) -> QLearningAgent | SarsaAgent:
    """Instantiate QLearningAgent or SarsaAgent by algorithm name."""
    if algo == "qlearning":
        return QLearningAgent(alpha=alpha, eps_decay=eps_decay, seed=seed)
    if algo == "sarsa":
        return SarsaAgent(alpha=alpha, eps_decay=eps_decay, seed=seed)
    raise ValueError(f"Unsupported algorithm: {algo}")


def run_episode(
    env: RescueEnv,
    agent: QLearningAgent | SarsaAgent,
) -> tuple[float, dict[str, Any]]:
    """Run one training episode, picking next_action before TD update."""
    state, info = env.reset()
    action = agent.act(state, env=env, explore=True)
    ep_return = 0.0
    done = False

    while not done:
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Pick next_action before update so SARSA uses the actual next action
        next_action = 0 if terminated else agent.act(next_state, env=env, explore=True)
        agent.update(state, action, reward, next_state, next_action, terminated)
        ep_return += reward
        state = next_state
        action = next_action

    agent.end_episode()
    return ep_return, info


def _make_log_row(
    episode: int,
    ep_return: float,
    info: dict[str, Any],
    epsilon: float,
    eval_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    """Format one CSV row combining training episode stats and optional eval metrics."""
    row: dict[str, Any] = {
        "episode": episode,
        "return": round(ep_return, 4),
        "steps": int(info["step"]),
        "delivered": int(info["delivered"]),
        "death_cause": info["death_cause"] or "none",
        "epsilon": round(epsilon, 6),
        "eval_success_rate": "",
        "eval_mean_return": "",
        "eval_mean_delivered": "",
        "eval_fire_death_rate": "",
        "eval_oxygen_death_rate": "",
        "eval_mean_steps": "",
    }
    if eval_metrics is not None:
        row["eval_success_rate"] = round(eval_metrics["success_rate"], 4)
        row["eval_mean_return"] = round(eval_metrics["mean_return"], 4)
        row["eval_mean_delivered"] = round(eval_metrics["mean_delivered"], 4)
        row["eval_fire_death_rate"] = round(eval_metrics["fire_death_rate"], 4)
        row["eval_oxygen_death_rate"] = round(eval_metrics["oxygen_death_rate"], 4)
        row["eval_mean_steps"] = round(eval_metrics["mean_steps"], 4)
    return row


def _print_progress(
    algo: str,
    seed: int,
    episode: int,
    episodes: int,
    epsilon: float,
    recent_returns: list[float],
    recent_fire_deaths: int,
    latest_eval: dict[str, float] | None,
) -> None:
    """Print periodic training and evaluation summary every PROGRESS_INTERVAL episodes."""
    mean_ret = sum(recent_returns) / len(recent_returns)
    fire_rate = recent_fire_deaths / len(recent_returns)
    eval_str = (
        f" | eval_succ={latest_eval['success_rate']*100:.1f}%"
        f" eval_ret={latest_eval['mean_return']:.1f}"
        if latest_eval is not None
        else ""
    )
    print(
        f"[{algo} seed={seed}] ep={episode}/{episodes}"
        f" eps={epsilon:.4f} train_ret={mean_ret:.1f}"
        f" fire_death={fire_rate*100:.1f}%{eval_str}"
    )


def train(
    algo: str,
    seed: int = 0,
    episodes: int = config.N_EPISODES,
    eval_every: int = config.EVAL_EVERY,
    eval_episodes: int = config.EVAL_EPISODES,
    shaping: bool = False,
    fire_scale: float = config.FIRE_SCALE,
    alpha: float = config.ALPHA,
    eps_decay: float = config.EPS_DECAY,
    out_dir: Path = RAW_RESULTS_DIR,
) -> tuple[Path, Path]:
    """Train a tabular RL agent, log CSV metrics, and save the Q-table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{algo}_seed{seed}.csv"
    qtable_path = out_dir / f"{algo}_seed{seed}.pkl"

    env = RescueEnv(seed=seed, shaping=shaping, fire_scale=fire_scale)
    agent = build_learner(algo, seed=seed, alpha=alpha, eps_decay=eps_decay)
    recent_returns: list[float] = []
    recent_fire_deaths = 0
    latest_eval: dict[str, float] | None = None

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for ep in range(1, episodes + 1):
            ep_return, info = run_episode(env, agent)
            recent_returns.append(ep_return)
            recent_fire_deaths += int(info["death_cause"] == "fire")

            eval_metrics: dict[str, float] | None = None
            if eval_every > 0 and ep % eval_every == 0:
                # DESIGN 8: evaluate greedily on separate evaluation seeds
                eval_metrics = evaluate(
                    agent, n_episodes=eval_episodes, seed=EVAL_SEED_BASE,
                    shaping=False, fire_scale=fire_scale,
                )
                latest_eval = eval_metrics

            writer.writerow(_make_log_row(ep, ep_return, info, agent.epsilon, eval_metrics))

            if ep % PROGRESS_INTERVAL == 0:
                _print_progress(
                    algo, seed, ep, episodes, agent.epsilon,
                    recent_returns, recent_fire_deaths, latest_eval,
                )
                recent_returns.clear()
                recent_fire_deaths = 0

    agent.save(qtable_path)
    print(f"Saved CSV log to {csv_path} and Q-table ({len(agent.q_table)} states) to {qtable_path}")
    return csv_path, qtable_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for tabular RL training."""
    parser = argparse.ArgumentParser(description="Train Q-learning or SARSA on RescueEnv.")
    parser.add_argument(
        "--algo",
        choices=("qlearning", "sarsa"),
        required=True,
        help="RL algorithm to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for training environment and agent (default: 0).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=config.N_EPISODES,
        help=f"Number of training episodes (default: {config.N_EPISODES}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for training a tabular Rescue-RL agent."""
    args = parse_args(argv)
    train(algo=args.algo, seed=args.seed, episodes=args.episodes)


if __name__ == "__main__":
    main()
