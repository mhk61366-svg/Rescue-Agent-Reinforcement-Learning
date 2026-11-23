# Rescue-RL: Autonomous Rescue Agent with Reinforcement Learning

An agent that learns, by trial and error, to rescue victims in a simulated disaster zone: it navigates a 10x10 grid, avoids spreading fire and rubble, manages a limited oxygen tank, and delivers injured victims to an ambulance and non-injured victims to a safe zone.

Built for the Reinforcement Learning Complex Computing Problem (BS AI, Bahria University Islamabad, Fall 2026).

## Status

Design is complete (see [DESIGN.md](DESIGN.md)). Implementation is in progress.

- [x] Design and specification
- [ ] Environment
- [ ] Baselines (random, greedy shortest-path)
- [ ] Q-learning and SARSA
- [ ] Experiments
- [ ] Pygame replay viewer
- [ ] Report

## The problem

The world is a fixed 10x10 map with walls, rubble, a safe zone, an ambulance, 4 victims and 5 starting fire cells. The agent has 30 oxygen, which refills only at the safe zone. Fire spreads randomly and spreads faster farther from the safe zone. Each episode, 2 of the 4 victims are randomly injured. The agent must carry them one at a time to the right drop-off before it runs out of oxygen or walks into fire.

## What this project does

- Custom environment with a Gymnasium-style `reset` / `step` interface, written from scratch.
- Compact egocentric state, so tabular methods can learn it.
- Tabular Q-learning (off-policy) compared with SARSA (on-policy).
- Two non-learning baselines: random agent and greedy shortest-path agent.
- Experiments on hyperparameters, reward shaping, and fire spread rate.
- Pygame replay viewer for watching a trained agent.

## Project structure

```
rescue-rl/
├── README.md
├── DESIGN.md
├── AGENTS.md
├── requirements.txt
├── config.py
├── map.txt
├── src/
│   ├── geometry.py
│   ├── env.py
│   ├── state.py
│   ├── agents.py
│   ├── train.py
│   ├── evaluate.py
│   ├── experiments.py
│   ├── plots.py
│   └── viewer.py
├── tests/
├── results/
└── screenshots/
```

See [DESIGN.md](DESIGN.md) section 12 for what each file does.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/mhk61366-svg/Rescue-Agent-Reinforcement-Learning.git
cd Rescue-Agent-Reinforcement-Learning
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Planned commands (run from the repo root). They will be confirmed as each part is built.

```bash
python -m src.train          # train an agent
python -m src.experiments    # run the experiments
python -m src.viewer         # watch a trained agent in pygame
pytest                       # run tests
```

## Results

To be added after the experiments are complete.

## Documentation

- [DESIGN.md](DESIGN.md): full specification (environment, state, rewards, agents, experiments).

## Course information

- Course: Reinforcement Learning, BS(AI) 5(A), Fall 2026
- University: Bahria University, Islamabad
- Author: Hamad