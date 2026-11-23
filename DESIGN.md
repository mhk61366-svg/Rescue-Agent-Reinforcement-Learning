# DESIGN.md: Intelligent Autonomous Rescue Agent (RL)

**Course:** Reinforcement Learning, BS(AI) 5(A), Fall 2026, Bahria University Islamabad
**Author:** Hamad
**Status:** v1.0, design frozen. Any change after this point goes in the change log (section 16).

---

## 1. Problem summary

A rescue agent operates in a 10x10 grid disaster zone. It must find victims, carry them to the correct drop-off point, avoid fire and rubble, and manage a limited oxygen tank, while fire spreads randomly around it. The agent learns this by trial and error using tabular reinforcement learning (Q-learning and SARSA), and its results are compared against two hand-coded baselines.

The grade-relevant parts are the problem formulation (state, action, reward and why), the algorithm comparison, the experiments, and the analysis of why things worked or failed. The pygame viewer is for demonstration only.

---

## 2. Locked decisions

| Topic | Decision |
|---|---|
| Grid | 10x10, fixed layout of walls, rubble, safe zone, ambulance, victim spots |
| Victims | 4 at fixed positions; 2 randomly marked injured each episode, 2 non-injured |
| Delivery | Injured go to the ambulance (A). Non-injured go to the safe zone (S). Carry capacity 1 |
| Oxygen | 30 units, refilled to full whenever the agent is on S. The ambulance does not refill |
| Rubble | Passable, costs 5 oxygen and -5 reward. Agent should prefer going around |
| Fire | 5 initial cells, spreads probabilistically, more likely far from S. Cap of 18 burning cells |
| Movement | Deterministic (no slipping) |
| Actions | up, down, left, right, rescue, drop |
| Failure | Terminal state with a large penalty. No respawn |
| Episode cap | 150 steps (truncation, not failure) |
| Victim priority | Rule-based triage: nearest injured first, else nearest non-injured |
| Algorithms | Tabular Q-learning and SARSA, plus random and greedy shortest-path baselines |
| Stretch (only if time) | Randomized victim positions, function-approximation Q-learning |

---

## 3. Environment

### 3.1 Map

Coordinates are `(row, col)` with the origin at the top-left.

```
      0 1 2 3 4 5 6 7 8 9
 0    . . . . . . F . . .
 1    . . . . . . . . V .
 2    . . . F . . . . . .
 3    . . . . . V . . . .
 4    . F . # # . R . . .
 5    . . . . F . . . . .
 6    . . # # . . . . . S
 7    . V . . . # # . . .
 8    . . . . # . V R . .
 9    . . . F . # # . . A
```

| Symbol | Meaning |
|---|---|
| `.` | Free cell |
| `#` | Wall (impassable) |
| `R` | Rubble (passable, costs 5 oxygen) |
| `S` | Safe zone: oxygen refill, non-injured drop-off, agent start |
| `A` | Ambulance: injured drop-off only |
| `V` | Victim spot (4 fixed positions) |
| `F` | Initial fire cell (5 cells) |

**Contents of `map.txt`** (the file the code reads, no coordinate header):

```
. . . . . . F . . .
. . . . . . . . V .
. . . F . . . . . .
. . . . . V . . . .
. F . # # . R . . .
. . . . F . . . . .
. . # # . . . . . S
. V . . . # # . . .
. . . . # . V R . .
. . . F . # # . . A
```

### 3.2 Map validation

Computed with static Dijkstra: free cells cost 1, rubble costs 5, walls blocked, fire ignored. These are the numbers `geometry.py` must reproduce.

| Victim | Cell | Cost S to V | Injured trip S to V to A to S | Non-injured trip S to V to S |
|---|---|---|---|---|
| V1 | (1,8) | 6 | 18 | 12 |
| V2 | (3,5) | 7 | 20 | 14 |
| V3 | (7,1) | 9 | 24 | 18 |
| V4 | (8,6) | 9 (through the rubble door) | 20 | 18 |

- Worst case is an injured V3: 24 plus 2 oxygen for the rescue and drop actions is 26 out of 30, leaving 4 spare. Every victim is reachable on one tank, but V3 allows almost no detours.
- S to A costs 3. S is at (6,9) and A is at (9,9). No wall, rubble, victim or fire cell is within one cell (including diagonals) of S or A.
- Rubble at (8,7) is the **door** of a walled pocket holding V4. Walls at (7,5), (7,6), (8,4), (9,5), (9,6) close the rest of it. Blocking the door makes V4 unreachable, so reaching V4 means paying 5 oxygen to cross, and 5 again on the way back.
- Rubble at (4,6) is the **avoid** case: it sits in open ground and removing it changes no route cost. The agent should walk around it.
- Initial fire cells (cost from S): (0,6)=9, (2,3)=10, (4,1)=10, (5,4)=6, (9,3)=9. All are at least 6 away from S and at least 3 (Manhattan) from every victim. The fire at (5,4) sits in the middle of the map, on the way to V2 and V3.
- The fire-proof zone (S, A and their 4 neighbors) contains no initial fire.
- Every non-wall cell is reachable from S.
- Because S is on the right edge, cells with a cost above about 9 from S are effectively out of reach on one tank. The left part of the map is mostly unused, except as a place where fire spreads.

### 3.3 Entities and rules of the world

- **Agent:** starts on S, empty-handed, oxygen 30. Carries at most one victim.
- **Victims:** stationary until rescued. A victim is either alive and uncollected, carried, delivered, or dead (burned).
- **Injured flags:** at every reset, 2 of the 4 victims are chosen at random to be injured. Positions never change.
- **Triage target:** when the agent is empty-handed, its target is the nearest alive uncollected injured victim; if none remain, the nearest alive uncollected non-injured victim. "Nearest" uses the static cost table (section 3.2). Ties break by lowest row, then lowest column.
- **Oxygen:** see section 6. Tank is refilled to 30 instantly whenever the agent stands on S.
- **Fire:** permanent once lit. Never ignites walls, rubble, S, A, or the 4 neighbors of S and A.

### 3.4 Fire spread

Each step, after the agent acts, fire spreads. All cells update simultaneously from the old fire set.

- **Eligible cells:** non-burning free cells (cells with an uncollected victim included) outside the fire-proof zone.
- **Ignition probability per burning 4-neighbor:** `p = P_BASE + P_SLOPE * (d / D_MAX)`, where `d` is the Manhattan distance from the cell to S and `D_MAX = 15` (the largest Manhattan distance from S on this map).
- With `k` burning neighbors, the cell ignites with probability `1 - (1 - p)^k`.
- Defaults: `P_BASE = 0.02`, `P_SLOPE = 0.08`, so `p` ranges from 0.02 next to S to 0.10 at the far corner. A global `FIRE_SCALE` multiplier (default 1.0) scales `p` for the fire-rate experiment.
- **Cap:** at most 18 cells burning at once. If more cells would ignite in one step than the cap allows, a random subset is kept.

---

## 4. Actions

| ID | Action | Effect |
|---|---|---|
| 0 | up | move to (row-1, col) |
| 1 | down | move to (row+1, col) |
| 2 | left | move to (row, col-1) |
| 3 | right | move to (row, col+1) |
| 4 | rescue | pick up the victim on this cell (alive, uncollected, hands empty) |
| 5 | drop | deliver the carried victim (injured on A, non-injured on S) |

Directions are absolute (not relative to a heading). Invalid actions have no effect and are penalized (section 7).

---

## 5. State representation (tabular)

The full world (fire map, victim set, oxygen, position) is far too large for a table, so the agent sees a compact, egocentric summary. The state is a tuple of 9 small integers.

| # | Feature | Values | Count |
|---|---|---|---|
| 1-4 | Up / down / left / right neighbor: 0 free, 1 wall or edge, 2 rubble, 3 fire | 4 each | 256 |
| 5 | Mode: 0 empty and target injured, 1 empty and target non-injured, 2 carrying injured, 3 carrying non-injured | 4 | 4 |
| 6 | Slack bucket (below) | 4 | 4 |
| 7 | Sign of goal column difference (-1, 0, +1) | 3 | 3 |
| 8 | Sign of goal row difference (-1, 0, +1) | 3 | 3 |
| 9 | Goal distance bucket (below) | 3 | 3 |

Theoretical size is 256 x 4 x 4 x 3 x 3 x 3 = 110,592 states, times 6 actions. The Q-table is stored sparsely (a dict of state tuple to array of 6 floats). Far fewer states are reachable because the layout is fixed.

**Goal** (used by features 7-9):
- empty-handed: the triage target victim's cell
- carrying injured: A
- carrying non-injured: S

**Slack** = `oxygen - required`, where `required` is:
- carrying injured: `cost(pos, A) + cost(A, S)`
- otherwise: `cost(pos, S)`

Buckets: slack <= 0 is 0, 1 to 3 is 1, 4 to 8 is 2, 9 or more is 3.

**Goal distance bucket:** static cost from the agent to the goal. 3 or less is 0, 4 to 8 is 1, 9 or more is 2.

All costs come from the precomputed static all-pairs cost table (free 1, rubble 5, walls blocked, fire ignored), built once in `geometry.py`.

---

## 6. Step logic

Order of operations inside `env.step(action)`:

1. **Apply the action.**
   - Move into a free cell: agent moves.
   - Move into rubble: agent moves.
   - Move into a wall or off the grid: agent stays (bump).
   - Move into a burning cell: agent dies (step ends, reward -100).
   - Rescue: valid only if the agent's cell holds an alive, uncollected victim and the agent is empty-handed. The victim becomes carried. Otherwise invalid.
   - Drop: valid only if carrying an injured victim and standing on A, or carrying a non-injured victim and standing on S. The victim becomes delivered. Otherwise invalid.
2. **Oxygen.** Subtract the action cost (1 for every action, 5 for a rubble move). Then:
   - if the agent is on S, oxygen is reset to 30 (this takes priority over the death check);
   - otherwise, if oxygen is 0 or less, the agent dies. A rubble move with fewer than 5 oxygen left is therefore fatal.
3. **Fire spreads** (section 3.4).
4. **Fire consequences.**
   - Fire on the agent's cell: the agent dies.
   - Fire on an uncollected victim's cell: that victim dies.
5. **Termination checks.**
   - Agent death (fire or oxygen): `terminated = True`, reward is exactly -100.
   - All victims resolved (each delivered or dead): `terminated = True`. A +100 bonus is added only if all 4 were delivered alive.
   - Step count reaches 150: `truncated = True`, no extra reward.
6. **Return** `(state, reward, terminated, truncated, info)`.

If several rewards fall in one step (for example a drop plus the completion bonus), they are summed. If the agent dies, the reward is -100 and nothing else.

`info` carries: `death_cause` (`"fire"`, `"oxygen"` or `None`), `delivered`, `dead_victims`, `oxygen`, `step`.

**Reset:** agent to S, oxygen 30, fire restored to the 5 initial cells, injured flags re-rolled, all victims alive and uncollected.

---

## 7. Rewards

Reward values are totals for the action, not additive with a separate step cost.

| Event | Reward | Oxygen | Terminal? |
|---|---|---|---|
| Move into free cell | -1 | -1 | No |
| Move into rubble | -5 | -5 | No |
| Bump wall or edge | -2 | -1 | No |
| Rescue succeeds | +20 | -1 | No |
| Rescue invalid | -2 | -1 | No |
| Drop injured at ambulance | +100 | -1 | No |
| Drop non-injured at safe zone | +50 | -1 | No |
| Drop invalid | -2 | -1 | No |
| Enter fire, or fire ignites the agent's cell | -100 | n/a | Yes |
| Oxygen 0 or less outside S | -100 | n/a | Yes |
| Fire burns an uncollected victim | -50 | n/a | No |
| All 4 victims delivered alive | +100 | n/a | Yes |
| Step cap (150) reached | 0 | n/a | Truncated |

**Why these magnitudes:**
- Death (-100) must exceed anything a risky shortcut can gain, otherwise the agent gambles.
- The step cost (-1) must be small next to delivery rewards, otherwise doing nothing looks attractive.
- Pickup (+20) is small next to delivery (+50 or +100), so the agent cannot profit by grabbing victims and stalling.
- Rubble reward (-5) matches its oxygen cost (5), so the reward and the oxygen budget agree: the agent crosses rubble only when the detour costs more than 5.
- A perfect run earns roughly 480 minus around 100 in step costs, so dying early is never the better option.

**Reward shaping (ablation only, off by default):** potential-based shaping `F = gamma * Phi(s') - Phi(s)` with `Phi(s) = -(static cost from agent to current goal)`. Potential-based shaping does not change the optimal policy and avoids reward-farming loops.

---

## 8. Episodes and evaluation protocol

- **Training episodes:** 30,000 by default (configurable).
- **Terminated vs truncated:** on termination the Q-update target is the reward alone. On truncation (step cap) the update still bootstraps from the next state, because the cap is a time limit and not a real end of the task.
- **Greedy evaluation:** every 500 training episodes, run 100 episodes with epsilon = 0 on separate evaluation seeds. Log these curves.
- **Seeds:** 5 training seeds per configuration (3 if time is short). Report mean and standard deviation across seeds.
- **Metrics:** success rate (all 4 delivered alive), mean return, mean victims delivered, deaths by cause (fire vs oxygen), mean steps per episode, and training-time fire-death rate.

---

## 9. Agents

All agents share one interface: `act(state, env)` for baselines, and `act(state)` plus `update(...)` for learners.

### 9.1 Random baseline
Uniform random over the 6 actions. Establishes the floor.

### 9.2 Greedy shortest-path baseline
Hand-coded, no learning. At each step:
- goal follows the same triage rule and the same drop-off rule as the learners;
- path is the cheapest path (Dijkstra with real oxygen costs) with burning cells treated as blocked, recomputed every step;
- when empty-handed and slack is at or below a safety margin (2), it returns to S first;
- takes rescue and drop whenever they are valid.

It is the benchmark the learners must beat or approach.

### 9.3 Q-learning (off-policy)
`Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`

### 9.4 SARSA (on-policy)
`Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))`, where `a'` is the action actually chosen by the epsilon-greedy policy in `s'`.

### 9.5 Exploration and initial hyperparameters
Starting values, all to be tuned in experiments:

| Parameter | Value |
|---|---|
| alpha (learning rate) | 0.1 |
| gamma | 0.99 |
| epsilon start / floor | 1.0 / 0.05 |
| epsilon decay | multiplicative per episode, 0.9995 |
| Q-table initial value | 0.0 |

**Hypothesis H1 (to be tested, not assumed):** fire plays the role of the cliff in the classic cliff-walking problem. SARSA, which accounts for its own exploratory mistakes, should die to fire less often during training and keep more distance from fire, while Q-learning may learn riskier, shorter routes. The results may disagree. Report whichever way they come out.

---

## 10. Experiments

| ID | Question | What varies |
|---|---|---|
| E1 | Do learners beat the baselines? Q-learning vs SARSA | Agent type (random, greedy, Q-learning, SARSA) |
| E2 | How sensitive is learning to hyperparameters? | alpha in {0.05, 0.1, 0.3}; epsilon decay in {0.999, 0.9995, 0.9999} |
| E3 | Does reward shaping help? | Shaping off vs on |
| E4 | How does performance degrade as fire gets worse? | `FIRE_SCALE` in {0.5, 1, 2, 3} |

Each experiment produces a CSV in `results/` and a figure from `plots.py`. Also produce a visit-count heatmap near fire cells for Q-learning vs SARSA to support or refute H1.

---

## 11. Visualization (pygame)

A replay viewer, not the training loop (training inside pygame is far too slow).

- Loads a saved Q-table, runs the greedy policy in the environment, and draws each step.
- Plain 2D drawing: colored squares for cell types, a circle for the agent, markers for victims (injured vs non-injured), fire cells, oxygen bar, carrying indicator.
- HUD: step count, oxygen, carried victim, cumulative reward, episode outcome.
- Controls: space pause/resume, R new episode, up/down arrows to change speed.
- No assets or images required.

---

## 12. Repository structure

```
rescue-rl/
├── README.md              project overview, how to run, results summary
├── DESIGN.md              this file
├── AGENTS.md              instructions for the coding agent
├── requirements.txt       dependencies
├── .gitignore
├── config.py              every tunable number in one place
├── map.txt                the 10x10 layout (section 3.1)
├── src/
│   ├── __init__.py
│   ├── geometry.py        parse map.txt, static all-pairs cost table (Dijkstra)
│   ├── env.py             RescueEnv: reset/step, oxygen, fire, rewards
│   ├── state.py           encode the environment into the 9-int state tuple
│   ├── agents.py          RandomAgent, GreedyAgent, QLearningAgent, SarsaAgent
│   ├── train.py           training loop, periodic greedy eval, logging
│   ├── evaluate.py        run N greedy episodes and return metrics
│   ├── experiments.py     runs E1 to E4 over seeds, writes CSVs
│   ├── plots.py           turns CSVs into figures
│   └── viewer.py          pygame replay viewer
├── tests/
│   └── test_env.py        rule tests for env, state, geometry
├── results/               CSVs, figures, saved Q-tables
├── screenshots/           viewer screenshots for the README and report
└── (report PDF in repo root when finished)
```

Run everything from the repo root, for example `python -m src.train`.

---

## 13. Testing plan

Tests are written stage by stage, not all up front. The rules most worth testing:

- map validation numbers in section 3.2 are reproduced by `geometry.py`;
- oxygen refill takes priority over the death check on S;
- a rubble move costs 5 oxygen and -5 reward, and kills the agent if oxygen is under 5;
- rescue and drop validity (right place, right victim type, hands empty or full);
- fire never ignites walls, rubble, S, A, or their neighbors; never exceeds 18 burning cells;
- triage target selection, including tie-breaks;
- slack and goal-distance buckets, including the carrying-injured case;
- the same seed produces the same episode.

---

## 14. Known limitations (state these in the report)

- Victim priority is a hard-coded triage rule. The agent learns navigation, hazard avoidance and oxygen management, not which victim to choose.
- The state is hand-engineered. Its quality limits what the agent can learn.
- Distances ignore fire (static table), so slack can be optimistic when fire blocks a route.
- The agent sees fire only in the 4 adjacent cells, so it cannot react to a fire front early.
- One fixed map with fixed victim positions: the policy partly memorizes the layout. Only injured flags and fire randomness vary.
- Tabular methods cannot scale to larger grids or richer state.

---

## 15. Stretch goals (only after everything above is done and reported)

1. Randomized victim positions per episode.
2. Function-approximation Q-learning with both nearest-injured and nearest-non-injured directions exposed, so the agent learns the priority instead of following the triage rule.
3. Injured-victim deadlines.

---

## 16. Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-22 | v1.0 created | Initial design freeze |
| 2026-11-22 | S moved to (6,9), A to (9,9); map redesigned and re-validated; D_MAX 13 to 15; rubble door pocket replaces the detour chokepoint | Requested: S and A away from rubble, walls and victims |
