"""Pygame replay viewer for Rescue-RL agents.
Implements DESIGN.md section 11: plain 2D visualization of grid cells, fire,
victims, agent, oxygen bar, carrying indicator, HUD metrics, and keyboard
playback controls. Replays any agent through `act(state, env=env, explore=False)`.
"""

from __future__ import annotations
import argparse
from typing import Any
import pygame
import config
from src.agents import GreedyAgent, RandomAgent
from src.env import ACTION_NAMES, RescueEnv

Color = tuple[int, int, int]

# Layout geometry constants
CELL_SIZE = 56
GRID_MARGIN = 16
HUD_WIDTH = 300
RENDER_FPS = 60
DEFAULT_STEP_FPS = 4
MIN_STEP_FPS = 1
MAX_STEP_FPS = 30

# Color palette (plain 2D shapes, no external assets)
BG_COLOR: Color = (24, 28, 36)
GRID_LINE_COLOR: Color = (180, 188, 198)
HUD_BG_COLOR: Color = (34, 39, 50)
HUD_BORDER_COLOR: Color = (68, 76, 94)
TEXT_COLOR: Color = (236, 240, 244)
MUTED_TEXT_COLOR: Color = (170, 178, 190)

FREE_COLOR: Color = (234, 238, 243)
WALL_COLOR: Color = (45, 52, 64)
RUBBLE_COLOR: Color = (166, 124, 82)
SAFE_COLOR: Color = (76, 175, 80)
AMBULANCE_COLOR: Color = (33, 150, 243)
FIRE_OUTER_COLOR: Color = (229, 57, 53)
FIRE_INNER_COLOR: Color = (255, 167, 38)

AGENT_COLOR: Color = (25, 65, 165)
AGENT_OUTLINE_COLOR: Color = (255, 255, 255)
INJURED_COLOR: Color = (216, 27, 96)
NON_INJURED_COLOR: Color = (245, 179, 1)

SUCCESS_COLOR: Color = (102, 187, 106)
FAIL_COLOR: Color = (239, 83, 80)
WARN_COLOR: Color = (255, 202, 40)


def build_agent(agent_name: str, seed: int) -> Any:
    """Instantiate a baseline agent by CLI name."""
    if agent_name == "random":
        return RandomAgent(seed=seed)
    if agent_name == "greedy":
        return GreedyAgent()
    raise ValueError(f"Unsupported agent type: {agent_name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the replay viewer."""
    parser = argparse.ArgumentParser(description="Rescue-RL pygame replay viewer.")
    parser.add_argument(
        "--agent",
        choices=("random", "greedy"),
        default="greedy",
        help="Baseline agent to replay (default: greedy).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Initial episode random seed (default: 0).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_STEP_FPS,
        help=f"Initial simulation steps per second ({MIN_STEP_FPS}-{MAX_STEP_FPS}).",
    )
    return parser.parse_args(argv)


class RescueViewer:
    """Interactive pygame replay window for a RescueEnv episode."""

    def __init__(
        self,
        agent: Any,
        agent_label: str,
        seed: int = 0,
        step_fps: int = DEFAULT_STEP_FPS,
    ) -> None:
        """Initialize environment, replay state, and pygame display."""
        self.agent: Any = agent
        self.agent_label: str = agent_label
        self.seed: int = seed
        self.step_fps: int = max(MIN_STEP_FPS, min(MAX_STEP_FPS, step_fps))

        self.env: RescueEnv = RescueEnv(seed=self.seed)
        self.state: tuple[int, ...] = ()
        self.info: dict[str, Any] = {}
        self.ep_return: float = 0.0
        self.last_action: str = "-"
        self.terminated: bool = False
        self.truncated: bool = False
        self.paused: bool = False
        self.step_timer: float = 0.0
        self.reset_episode(increment_seed=False)

        grid_w = self.env.layout.cols * CELL_SIZE
        grid_h = self.env.layout.rows * CELL_SIZE
        self.hud_x: int = GRID_MARGIN * 2 + grid_w
        self.win_w: int = self.hud_x + HUD_WIDTH + GRID_MARGIN
        self.win_h: int = grid_h + GRID_MARGIN * 2

        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption(f"Rescue-RL Viewer ({self.agent_label})")
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_body = pygame.font.SysFont("consolas", 15)
        self.font_small = pygame.font.SysFont("consolas", 13)
        self.font_cell = pygame.font.SysFont("consolas", 20, bold=True)

    @property
    def done(self) -> bool:
        """Return True when the current episode has ended."""
        return self.terminated or self.truncated

    def reset_episode(self, increment_seed: bool = False) -> None:
        """Reset environment for a new replay episode."""
        if increment_seed:
            self.seed += 1
        self.state, self.info = self.env.reset(seed=self.seed)
        self.ep_return = 0.0
        self.last_action = "-"
        self.terminated = False
        self.truncated = False
        self.step_timer = 0.0

    def step_once(self) -> None:
        """Query agent for one greedy action and advance the environment."""
        if self.done:
            return
        # DESIGN 11: replay policy greedily through the unified act interface
        action = self.agent.act(self.state, env=self.env, explore=False)
        self.last_action = ACTION_NAMES.get(action, str(action))
        self.state, reward, self.terminated, self.truncated, self.info = self.env.step(action)
        self.ep_return += reward

    def outcome_text(self) -> tuple[str, Color]:
        """Return human-readable episode status label and display color."""
        if not self.done:
            return ("PAUSED", WARN_COLOR) if self.paused else ("RUNNING", TEXT_COLOR)

        delivered = int(self.info.get("delivered", 0))
        death_cause = self.info.get("death_cause")
        if delivered == config.N_VICTIMS:
            return (f"SUCCESS ({delivered}/{config.N_VICTIMS})", SUCCESS_COLOR)
        if death_cause == "fire":
            return ("DEAD (Fire)", FAIL_COLOR)
        if death_cause == "oxygen":
            return ("DEAD (Oxygen)", FAIL_COLOR)
        if self.truncated:
            return (f"TRUNCATED ({delivered}/{config.N_VICTIMS})", WARN_COLOR)
        return (f"ENDED ({delivered}/{config.N_VICTIMS})", WARN_COLOR)

    def handle_key(self, key: int) -> bool:
        """Process keyboard controls; return False to quit."""
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.reset_episode(increment_seed=True)
        elif key == pygame.K_UP:
            self.step_fps = min(MAX_STEP_FPS, self.step_fps + 1)
        elif key == pygame.K_DOWN:
            self.step_fps = max(MIN_STEP_FPS, self.step_fps - 1)
        return True

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Compute pixel rectangle for grid cell (row, col)."""
        x = GRID_MARGIN + col * CELL_SIZE
        y = GRID_MARGIN + row * CELL_SIZE
        return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

    def _draw_grid(self) -> None:
        """Draw terrain cells, S/A labels, and active fire squares."""
        layout = self.env.layout
        for r in range(layout.rows):
            for c in range(layout.cols):
                pos = (r, c)
                rect = self._cell_rect(r, c)
                color = FREE_COLOR
                if pos in layout.walls:
                    color = WALL_COLOR
                elif pos in layout.rubble:
                    color = RUBBLE_COLOR
                elif pos == layout.S:
                    color = SAFE_COLOR
                elif pos == layout.A:
                    color = AMBULANCE_COLOR

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID_LINE_COLOR, rect, 1)

                if pos in self.env.fire:
                    pygame.draw.rect(self.screen, FIRE_OUTER_COLOR, rect.inflate(-6, -6))
                    pygame.draw.rect(self.screen, FIRE_INNER_COLOR, rect.inflate(-20, -20))
                elif pos == layout.S or pos == layout.A:
                    label = "S" if pos == layout.S else "A"
                    surf = self.font_cell.render(label, True, AGENT_OUTLINE_COLOR)
                    self.screen.blit(surf, surf.get_rect(center=rect.center))
                elif pos in layout.rubble:
                    surf = self.font_small.render("R", True, AGENT_OUTLINE_COLOR)
                    self.screen.blit(surf, (rect.x + 5, rect.y + 4))

    def _draw_victims_and_agent(self) -> None:
        """Draw waiting victim markers and the agent circle with carrying badge."""
        for v in self.env.victims:
            if v["status"] != "waiting":
                continue
            rect = self._cell_rect(*v["pos"])
            cx, cy = rect.center
            half = CELL_SIZE // 4
            color = INJURED_COLOR if v["injured"] else NON_INJURED_COLOR
            diamond = [(cx, cy - half), (cx + half, cy), (cx, cy + half), (cx - half, cy)]
            pygame.draw.polygon(self.screen, color, diamond)
            pygame.draw.polygon(self.screen, WALL_COLOR, diamond, 2)
            if v["injured"]:
                pygame.draw.line(self.screen, AGENT_OUTLINE_COLOR, (cx - 5, cy), (cx + 5, cy), 2)
                pygame.draw.line(self.screen, AGENT_OUTLINE_COLOR, (cx, cy - 5), (cx, cy + 5), 2)

        agent_rect = self._cell_rect(*self.env.agent_pos)
        radius = int(CELL_SIZE * 0.32)
        pygame.draw.circle(self.screen, AGENT_COLOR, agent_rect.center, radius)
        pygame.draw.circle(self.screen, AGENT_OUTLINE_COLOR, agent_rect.center, radius, 2)

        # Show carried victim badge directly inside the agent circle
        if self.env.carrying is not None:
            carried = self.env.victims[self.env.carrying]
            badge_color = INJURED_COLOR if carried["injured"] else NON_INJURED_COLOR
            pygame.draw.circle(self.screen, badge_color, agent_rect.center, radius // 2)

    def _draw_text(self, text: str, x: int, y: int, color: Color = TEXT_COLOR, font: Any = None) -> int:
        """Render a single text line and return the next y coordinate."""
        use_font = font if font is not None else self.font_body
        surf = use_font.render(text, True, color)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height() + 6

    def _draw_oxygen_bar(self, x: int, y: int, width: int) -> int:
        """Draw oxygen progress bar and return next y coordinate."""
        ratio = max(0.0, min(1.0, self.env.oxygen / float(config.OXYGEN_MAX)))
        bar_rect = pygame.Rect(x, y, width, 18)
        pygame.draw.rect(self.screen, BG_COLOR, bar_rect)

        if ratio > 0.5:
            fill_color = SUCCESS_COLOR
        elif ratio > 0.2:
            fill_color = WARN_COLOR
        else:
            fill_color = FAIL_COLOR

        fill_w = int(width * ratio)
        if fill_w > 0:
            pygame.draw.rect(self.screen, fill_color, pygame.Rect(x, y, fill_w, 18))
        pygame.draw.rect(self.screen, HUD_BORDER_COLOR, bar_rect, 1)
        return y + 26

    def _draw_legend(self, x: int, y: int) -> None:
        """Draw compact color legend at bottom of HUD."""
        y = self._draw_text("Legend:", x, y, MUTED_TEXT_COLOR, self.font_small)
        items = [
            (AGENT_COLOR, "Agent (circle)"),
            (INJURED_COLOR, "Injured V (+ -> A)"),
            (NON_INJURED_COLOR, "Non-injured V (-> S)"),
            (FIRE_OUTER_COLOR, "Fire (fatal)"),
            (RUBBLE_COLOR, "Rubble (cost 5 O2)"),
            (SAFE_COLOR, "Safe S (O2 refill)"),
            (AMBULANCE_COLOR, "Ambulance A"),
        ]
        for color, label in items:
            swatch = pygame.Rect(x, y + 2, 12, 12)
            pygame.draw.rect(self.screen, color, swatch)
            pygame.draw.rect(self.screen, GRID_LINE_COLOR, swatch, 1)
            surf = self.font_small.render(label, True, MUTED_TEXT_COLOR)
            self.screen.blit(surf, (x + 18, y))
            y += 18

    def _draw_hud(self) -> None:
        """Draw HUD panel with episode stats, oxygen bar, carrying status, and controls."""
        panel = pygame.Rect(self.hud_x, GRID_MARGIN, HUD_WIDTH, self.win_h - 2 * GRID_MARGIN)
        pygame.draw.rect(self.screen, HUD_BG_COLOR, panel)
        pygame.draw.rect(self.screen, HUD_BORDER_COLOR, panel, 2)

        x = self.hud_x + 14
        y = GRID_MARGIN + 14
        inner_w = HUD_WIDTH - 28

        y = self._draw_text(f"Rescue-RL ({self.agent_label})", x, y, TEXT_COLOR, self.font_title)
        y = self._draw_text(f"Seed: {self.seed} | Speed: {self.step_fps} Hz", x, y, MUTED_TEXT_COLOR, self.font_small)
        y += 6

        outcome, outcome_color = self.outcome_text()
        y = self._draw_text(f"Outcome: {outcome}", x, y, outcome_color)
        y = self._draw_text(f"Step: {self.env.steps} / {config.STEP_CAP}", x, y)
        y = self._draw_text(f"Return: {self.ep_return:+.1f} (Act: {self.last_action})", x, y)
        y = self._draw_text(f"Oxygen: {self.env.oxygen} / {config.OXYGEN_MAX}", x, y)
        y = self._draw_oxygen_bar(x, y, inner_w)

        if self.env.carrying is None:
            carry_text, carry_color = "Carrying: None", TEXT_COLOR
        else:
            v = self.env.victims[self.env.carrying]
            kind = "Injured -> A" if v["injured"] else "Non-injured -> S"
            carry_text = f"Carrying: V{self.env.carrying + 1} ({kind})"
            carry_color = INJURED_COLOR if v["injured"] else NON_INJURED_COLOR
        y = self._draw_text(carry_text, x, y, carry_color)

        delivered = self.info.get("delivered", 0)
        dead_v = self.info.get("dead_victims", 0)
        y = self._draw_text(
            f"Delivered: {delivered}/{config.N_VICTIMS} | Dead V: {dead_v}",
            x,
            y,
        )
        y += 8

        y = self._draw_text("Controls:", x, y, MUTED_TEXT_COLOR, self.font_small)
        y = self._draw_text("[Space] Pause/Resume  [R] New", x, y, MUTED_TEXT_COLOR, self.font_small)
        y = self._draw_text("[Up/Down] Speed  [Esc/Q] Quit", x, y, MUTED_TEXT_COLOR, self.font_small)
        y += 6
        self._draw_legend(x, y)

    def render(self) -> None:
        """Render the complete viewer frame."""
        self.screen.fill(BG_COLOR)
        self._draw_grid()
        self._draw_victims_and_agent()
        self._draw_hud()
        pygame.display.flip()

    def run(self) -> None:
        """Execute main event, stepping, and rendering loop until exit."""
        running = True
        while running:
            dt = self.clock.tick(RENDER_FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN and not self.handle_key(event.key):
                    running = False
                    break

            if not self.paused and not self.done:
                self.step_timer += dt
                step_interval = 1.0 / float(self.step_fps)
                while self.step_timer >= step_interval and not self.done:
                    self.step_timer -= step_interval
                    self.step_once()

            self.render()

        pygame.quit()


def main(argv: list[str] | None = None) -> None:
    """Entry point for CLI execution of the Rescue-RL viewer."""
    args = parse_args(argv)
    agent = build_agent(args.agent, seed=args.seed)
    viewer = RescueViewer(agent=agent, agent_label=args.agent, seed=args.seed, step_fps=args.fps)
    viewer.run()


if __name__ == "__main__":
    main()
