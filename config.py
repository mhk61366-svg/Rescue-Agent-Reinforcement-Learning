"""Configuration constants for Rescue-RL disaster simulation.

All parameters and constants follow DESIGN.md specifications.
"""

# DESIGN 4: Action IDs
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
RESCUE = 4
DROP = 5

# DESIGN 2 & 3.3: Environment parameters
OXYGEN_MAX = 30           # DESIGN 2: maximum oxygen capacity
RUBBLE_COST = 5          # DESIGN 2: oxygen and step penalty for traversing rubble
STEP_CAP = 150           # DESIGN 2: maximum steps before episode truncation
FIRE_CAP = 18            # DESIGN 2: maximum simultaneous burning cells
P_BASE = 0.02            # DESIGN 3.4: base ignition probability per burning neighbor
P_SLOPE = 0.08           # DESIGN 3.4: distance-scaled slope for ignition probability
FIRE_SCALE = 1.0         # DESIGN 3.4: global multiplier on fire spread rate
N_VICTIMS = 4            # DESIGN 2: total number of victims on map
N_INJURED = 2            # DESIGN 2: number of injured victims per episode

# DESIGN 7: Reward table
R_STEP = -1              # Move into free cell
R_RUBBLE = -5            # Move into rubble
R_BUMP = -2              # Bump wall or grid edge
R_RESCUE = 20            # Successful rescue pickup
R_RESCUE_INVALID = -2    # Invalid rescue attempt
R_DROP_INJURED = 100     # Successful drop of injured victim at ambulance
R_DROP_NON_INJURED = 50  # Successful drop of non-injured victim at safe zone
R_DROP_INVALID = -2      # Invalid drop attempt
R_DEATH = -100           # Fatal event (fire or oxygen depletion)
R_VICTIM_DIED = -50      # Victim burned by fire while uncollected
R_ALL_DELIVERED = 100    # Completion bonus when all 4 victims delivered alive

# DESIGN 5: State bucketing cutoffs
# Slack = oxygen - required. Buckets: <= 0 -> 0, 1..3 -> 1, 4..8 -> 2, >= 9 -> 3
SLACK_BINS = (0, 3, 8)
# Goal distance: <= 3 -> 0, 4..8 -> 1, >= 9 -> 2
DIST_BINS = (3, 8)

# DESIGN 9.2: Baseline parameters
GREEDY_SAFETY_MARGIN = 2  # Return to S when empty-handed and slack <= margin

# DESIGN 8 & 9.5: Learning hyperparameters and evaluation protocol
ALPHA = 0.1              # Learning rate
GAMMA = 0.99             # Discount factor
EPS_START = 1.0          # Initial exploration rate
EPS_END = 0.05           # Minimum exploration rate floor
EPS_DECAY = 0.9995       # Multiplicative epsilon decay per episode
N_EPISODES = 30000       # Default training episode count
EVAL_EVERY = 500         # Evaluation frequency in episodes
EVAL_EPISODES = 100      # Greedy evaluation episode count
N_SEEDS = 5              # Number of independent training seeds

# Map layout file
MAP_PATH = "map.txt"
