"""Map geometry and static all-pairs shortest path calculations.

Loads map.txt, validates the disaster grid, and computes static Dijkstra costs
per DESIGN.md sections 3.1, 3.2, 3.3, and 3.4.
"""

from __future__ import annotations

import heapq
from pathlib import Path
import config

Coord = tuple[int, int]

# DESIGN 3.1: Permitted map character symbols
VALID_SYMBOLS = {".", "#", "R", "S", "A", "V", "F"}


def manhattan(a: Coord, b: Coord) -> int:
    """Compute Manhattan distance between two coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def dijkstra_from(
    src: Coord,
    rows: int,
    cols: int,
    walls: set[Coord],
    rubble: set[Coord],
) -> dict[Coord, int]:
    """Compute shortest path costs from src to all reachable non-wall cells."""
    dist: dict[Coord, int] = {src: 0}
    pq: list[tuple[int, Coord]] = [(0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        r, c = u
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            # Walls and boundaries block movement
            if 0 <= nr < rows and 0 <= nc < cols and neighbor not in walls:
                # DESIGN 3.2: free cells cost 1, rubble costs RUBBLE_COST
                step_cost = config.RUBBLE_COST if neighbor in rubble else 1
                new_d = d + step_cost
                if new_d < dist.get(neighbor, float("inf")):
                    dist[neighbor] = new_d
                    heapq.heappush(pq, (new_d, neighbor))

    return dist


class Layout:
    """Disaster zone grid layout and static routing costs."""

    def __init__(self, map_path: str = config.MAP_PATH) -> None:
        """Initialize layout from map file and precompute static cost table."""
        self.rows: int = 10
        self.cols: int = 10
        self.walls: set[Coord] = set()
        self.rubble: set[Coord] = set()
        self.free_cells: set[Coord] = set()
        self.victim_cells: list[Coord] = []
        self.initial_fire_cells: list[Coord] = []
        self.S: Coord = (0, 0)
        self.A: Coord = (0, 0)

        self._load_and_validate(map_path)

        # DESIGN 3.2: protected zone is S, A and their 4-neighbors
        s_nbrs = set(self.neighbors4(self.S))
        a_nbrs = set(self.neighbors4(self.A))
        self.protected_cells: set[Coord] = {self.S, self.A} | s_nbrs | a_nbrs

        # DESIGN 3.4: largest Manhattan distance from S
        self.d_max: int = max(
            self.manhattan(self.S, (r, c))
            for r in range(self.rows)
            for c in range(self.cols)
        )

        # DESIGN 3.2: static all-pairs cost table (fire ignored)
        self.cost_table: dict[tuple[Coord, Coord], int] = self._build_cost_table()

    @property
    def victims(self) -> list[Coord]:
        """Alias for victim_cells list."""
        return self.victim_cells

    @property
    def initial_fire(self) -> set[Coord]:
        """Set of initial fire cells."""
        return set(self.initial_fire_cells)

    def _load_and_validate(self, map_path: str) -> None:
        """Load map.txt and validate layout against DESIGN specifications."""
        path = Path(map_path)
        if not path.is_file():
            # Try resolving relative to repository root if called from elsewhere
            alt = Path(__file__).resolve().parent.parent / map_path
            if alt.is_file():
                path = alt
            else:
                raise FileNotFoundError(f"Map file not found: {map_path}")

        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if len(lines) != self.rows:
            raise ValueError(f"Expected {self.rows} rows, found {len(lines)}")

        grid: list[list[str]] = []
        for r, line in enumerate(lines):
            tokens = line.split()
            if len(tokens) != self.cols:
                raise ValueError(f"Row {r} has {len(tokens)} cols, expected {self.cols}")
            for token in tokens:
                if token not in VALID_SYMBOLS:
                    raise ValueError(f"Invalid symbol '{token}' at row {r}")
            grid.append(tokens)

        self._parse_cells(grid)

    def _parse_cells(self, grid: list[list[str]]) -> None:
        """Parse grid tokens into typed cell sets and validate entity counts."""
        s_cells: list[Coord] = []
        a_cells: list[Coord] = []

        for r in range(self.rows):
            for c in range(self.cols):
                token = grid[r][c]
                pos = (r, c)
                if token == "#":
                    self.walls.add(pos)
                elif token == "R":
                    self.rubble.add(pos)
                else:
                    # DESIGN 3.1 & 3.2: '.', 'S', 'A', 'V', 'F' are all free ground
                    self.free_cells.add(pos)
                    if token == "S":
                        s_cells.append(pos)
                    elif token == "A":
                        a_cells.append(pos)
                    elif token == "V":
                        self.victim_cells.append(pos)
                    elif token == "F":
                        self.initial_fire_cells.append(pos)

        # DESIGN 3.1: Exactly one S and one A
        if len(s_cells) != 1:
            raise ValueError(f"Expected exactly 1 'S', found {len(s_cells)}")
        if len(a_cells) != 1:
            raise ValueError(f"Expected exactly 1 'A', found {len(a_cells)}")

        # DESIGN 2 & 3.1: Exactly 4 victims and 5 initial fires
        if len(self.victim_cells) != config.N_VICTIMS:
            raise ValueError(f"Expected {config.N_VICTIMS} 'V', found {len(self.victim_cells)}")
        if len(self.initial_fire_cells) != 5:
            raise ValueError(f"Expected 5 'F', found {len(self.initial_fire_cells)}")

        self.S = s_cells[0]
        self.A = a_cells[0]

    def _build_cost_table(self) -> dict[tuple[Coord, Coord], int]:
        """Compute all-pairs static Dijkstra costs between non-wall cells."""
        table: dict[tuple[Coord, Coord], int] = {}
        non_walls = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.walls
        ]
        for src in non_walls:
            dists = dijkstra_from(src, self.rows, self.cols, self.walls, self.rubble)
            for dst, d in dists.items():
                table[(src, dst)] = d
        return table

    def neighbors4(self, pos: Coord) -> list[Coord]:
        """Return in-bounds cardinal 4-neighbors in UP, DOWN, LEFT, RIGHT order."""
        r, c = pos
        nbrs: list[Coord] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                nbrs.append((nr, nc))
        return nbrs

    def manhattan(self, a: Coord, b: Coord) -> int:
        """Compute Manhattan distance between two coordinates."""
        return manhattan(a, b)

    def cost(self, a: Coord, b: Coord) -> int:
        """Return static shortest-path cost from cell a to cell b."""
        if a == b:
            return 0
        if (a, b) not in self.cost_table:
            raise ValueError(f"No path or invalid non-wall cell between {a} and {b}")
        return self.cost_table[(a, b)]


# Default layout instance for module-level convenience
_default_layout: Layout | None = None


def get_default_layout() -> Layout:
    """Get or lazily create default Layout instance."""
    global _default_layout
    if _default_layout is None:
        _default_layout = Layout()
    return _default_layout


def cost(a: Coord, b: Coord) -> int:
    """Module-level static cost query using default layout."""
    return get_default_layout().cost(a, b)


if __name__ == "__main__":
    lay = Layout()

    print("=== Map Validation (DESIGN 3.2) ===")
    print(f"Safe zone (S): {lay.S}")
    print(f"Ambulance (A): {lay.A}")
    print(f"Cost S to A: {lay.cost(lay.S, lay.A)}")
    print()

    print("Victim routes:")
    print(f"{'Victim':<6} | {'Cell':<8} | {'Cost S to V':<11} | {'Injured trip':<12} | {'Non-injured trip':<16}")
    print("-" * 65)
    for i, v_pos in enumerate(lay.victim_cells, start=1):
        c_sv = lay.cost(lay.S, v_pos)
        c_va = lay.cost(v_pos, lay.A)
        c_vs = lay.cost(v_pos, lay.S)
        c_as = lay.cost(lay.A, lay.S)
        injured_trip = c_sv + c_va + c_as
        non_injured_trip = c_sv + c_vs
        print(f"V{i:<5} | {str(v_pos):<8} | {c_sv:<11} | {injured_trip:<12} | {non_injured_trip:<16}")
    print()

    print("Initial fire cells (cost from S):")
    for f_pos in lay.initial_fire_cells:
        print(f"{f_pos} = {lay.cost(lay.S, f_pos)}")
    print()

    print(f"d_max (largest Manhattan distance from S): {lay.d_max}")

    non_walls = [
        (r, c)
        for r in range(lay.rows)
        for c in range(lay.cols)
        if (r, c) not in lay.walls
    ]
    unreachable = sum(1 for cell in non_walls if (lay.S, cell) not in lay.cost_table)
    print(f"Unreachable non-wall cells from S: {unreachable}")
