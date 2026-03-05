from typing import Generator

import pygame

from src.entity import Entity, Player
from src.pathfinding import State, astar
from src.world import World

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class Mob(Entity):
    # --- stats ---
    maxhealth: int = 100
    maxstagger: int = 60
    default_walk_speed: float = 1.0
    Defaultdmg: int = 8
    default_attack_speed: float = 1.5

    # --- physics ---
    bbox_size: pygame.Vector2 = pygame.Vector2(0.8, 1.8)

    # --- aggro ranges ---
    detect_range_x: float = 200.0
    detect_range_y: float = 120.0
    chase_range_x: float = 350.0
    chase_range_y: float = 200.0

    # --- pathfinding config ---
    max_jump: int = 2  # max tiles the mob can jump upward
    path_replan_dist: float = 3.0  # retrigger A* if goal moves this many tiles
    path_replan_delay: float = 0.5  # minimum seconds between replans

    def __init__(self, x: float, y: float, world: World) -> None:
        super().__init__(
            maxhealth=self.maxhealth,
            maxstagger=self.maxstagger,
            walkspeed=self.default_walk_speed,
            dmg=self.Defaultdmg,
            attackspeed=self.default_attack_speed,
            x=x,
            y=y,
            world=world,
            badEntity=True,
        )

        self.has_aggro: bool = False

        # pathfinding runtime state
        self._path: list[State] = []
        self._path_index: int = 0
        self._last_goal: State | None = None
        self._replan_timer: float = 0.0

    # -----------------------------------------------------------------------
    # Aggro
    # -----------------------------------------------------------------------

    def focus_player(self, player: Player) -> None:
        dx = abs(player.x - self.x)
        dy = abs(player.y - self.y)
        if not self.has_aggro:
            if dx < self.detect_range_x and dy < self.detect_range_y:
                self.has_aggro = True
        else:
            if dx > self.chase_range_x or dy > self.chase_range_y:
                self.has_aggro = False

    # -----------------------------------------------------------------------
    # Grid helpers
    # -----------------------------------------------------------------------

    def _tile(self, x: float, y: float) -> tuple[int, int]:
        """Convert world-space float position to tile coords."""
        return (int(x), int(y))

    def _is_solid(self, tx: int, ty: int) -> bool:
        return self.world.is_solid(tx, ty)

    def _mob_fits(self, tx: int, ty: int) -> bool:
        """True if the mob's bounding box fits without overlapping any solid tile."""
        w = max(1, round(self.bbox_size.x))
        h = max(1, round(self.bbox_size.y))
        for dy in range(h):
            for dx in range(w):
                if self._is_solid(tx + dx, ty + dy):
                    return False
        return True

    def _on_ground(self, tx: int, ty: int) -> bool:
        """True if at least one tile directly below the mob's feet is solid."""
        w = max(1, round(self.bbox_size.x))
        h = max(1, round(self.bbox_size.y))
        feet = ty + h
        return any(self._is_solid(tx + dx, feet) for dx in range(w))

    # -----------------------------------------------------------------------
    # Neighbour generation
    # -----------------------------------------------------------------------

    def _fall_to_ground(self, nx: int, start_y: int, max_fall: int = 64) -> int | None:
        """
        Drop from start_y downward until grounded.
        Returns landing y, or None if no ground found within max_fall tiles.
        """
        for ny in range(start_y, start_y + max_fall):
            if not self._mob_fits(nx, ny):
                return None  # hit a wall mid-fall, no valid landing
            if self._on_ground(nx, ny):
                return ny
        return None  # fell too far (void / no ground)

    def _get_neighbours(
        self, state: State
    ) -> Generator[tuple[State, float], None, None]:
        x, y = state.x, state.y
        on_ground = self._on_ground(x, y)

        # --- walk left / right (with implicit fall) ---
        for dx in [-1, 1]:
            nx = x + dx
            if not self._mob_fits(nx, y):
                continue

            ny = self._fall_to_ground(nx, y)
            if ny is None:
                continue  # no valid landing, skip this direction

            fall_cost = (ny - y) * 0.5
            yield State(nx, ny, 0), 1.0 + fall_cost

        # --- jump (only when grounded) ---
        if not on_ground:
            return

        for jump_h in range(1, self.max_jump + 1):
            # check vertical clearance for this jump height
            clear = all(self._mob_fits(x, y - dy) for dy in range(1, jump_h + 1))
            if not clear:
                break  # taller jumps will also be blocked

            apex_y = y - jump_h

            # jump diagonally left / right from apex
            for dx in [-1, 1]:
                nx = x + dx
                if not self._mob_fits(nx, apex_y):
                    continue

                ny = self._fall_to_ground(nx, apex_y)
                if ny is None:
                    continue

                fall_cost = (ny - apex_y) * 0.5
                yield State(nx, ny, 0), 1.0 + jump_h + fall_cost

            # jump straight up onto a platform directly above
            if self._mob_fits(x, apex_y) and self._on_ground(x, apex_y):
                yield (
                    State(x, apex_y, 0),
                    1.0 + jump_h,
                )

    def _heuristic(self, state: State, goal: State) -> float:
        return abs(state.x - goal.x) + abs(state.y - goal.y)

    def _should_replan(self, goal: State) -> bool:
        """True if we need a fresh A* run."""
        if self._last_goal is None:
            return True  # no path yet
        if self._replan_timer > 0.0:
            return False  # throttled
        dist = abs(goal.x - self._last_goal.x) + abs(goal.y - self._last_goal.y)
        return dist >= self.path_replan_dist  # goal moved far enough

    def _replan(self, goal: State) -> None:
        tx, ty = self._tile(self.x, self.y)
        start = State(tx, ty, 0)

        path = astar(
            start,
            goal,
            self._get_neighbours,
            lambda s: self._heuristic(s, goal),
        )

        self._path = path or []
        self._path_index = 1  # index 0 is the tile we're already on
        self._last_goal = goal
        self._replan_timer = self.path_replan_delay

    # -----------------------------------------------------------------------
    # Path following
    # -----------------------------------------------------------------------

    def _follow_path(self, dt: float) -> None:
        if not self._path or self._path_index >= len(self._path):
            print("Hi", self._path)
            self.vel_x = 0.0
            return

        target: State = self._path[self._path_index]
        tx, ty = self._tile(self.x, self.y)

        # advance waypoint when the mob reaches the current target tile
        if (tx, ty) == (target.x, target.y):
            self._path_index += 1
            if self._path_index >= len(self._path):
                self.vel_x = 0.0
                return
            target = self._path[self._path_index]

        dx: float = target.x - self.x
        dy: float = target.y - self.y  # negative = upward in world space

        # horizontal velocity toward the next waypoint
        self.vel_x = self.walkspeed * (1.0 if dx > 0 else -1.0 if dx < 0 else 0.0)

        # trigger a jump if the next waypoint is above us and we're grounded
        if dy < -0.5 and self.on_ground:
            self.jump()

    # -----------------------------------------------------------------------
    # Public update (same interface as before)
    # -----------------------------------------------------------------------

    def update_focus(self, player: Player, dt: float) -> None:
        self.focus_player(player)
        self._replan_timer = max(0.0, self._replan_timer - dt)

        if not self.has_aggro or self.is_staggered:
            self.vel_x = 0.0
            return

        ptx, pty = self._tile(player.x, player.y + 1)
        goal = State(ptx, pty, 0)

        if self._should_replan(goal):
            print("Replanning")
            self._replan(goal)

        self._follow_path(dt)
