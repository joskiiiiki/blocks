from __future__ import annotations

from dataclasses import dataclass

import pygame

from src.blocks import BLOCK_SPEED, Block
from src.collision import BoundingBox

# ---------------------------------------------------------------------------
# Stat blocks
# ---------------------------------------------------------------------------


@dataclass
class EntityStats:
    maxhealth: int
    maxstagger: int
    walk_speed: float
    dmg: int
    attack_speed: float
    bbox_size: tuple[float, float]
    jump_power: float = 6.0
    gravity: float = -9.81
    armor_value: float = 0.0
    passivregen: float = 0.0
    sprint_speed: float = 0.0  # 0 = no sprinting


PLAYER_STATS = EntityStats(
    maxhealth=200,
    maxstagger=100,
    walk_speed=5.0,
    dmg=10,
    attack_speed=1.0,
    bbox_size=(0.8, 1.8),
    jump_power=12.0,
    armor_value=0.0,
    passivregen=5.0,
    sprint_speed=8.0,
)

ENTITY_REGISTRY: dict[str, "EntityStats"] = {}  # populated after stats are defined

ZOMBIE_STATS = EntityStats(
    maxhealth=100,
    maxstagger=60,
    walk_speed=2.5,
    dmg=8,
    attack_speed=1.5,
    bbox_size=(0.8, 1.8),
    jump_power=6.0,
)

ENTITY_REGISTRY["Zombie"] = ZOMBIE_STATS


# ---------------------------------------------------------------------------
# Physics result — returned by the game loop, consumed by entity
# ---------------------------------------------------------------------------


@dataclass
class PhysicsResult:
    position: pygame.Vector2
    on_ground: bool
    hit_ceiling: bool
    x_collision: bool
    y_collision: bool


# ---------------------------------------------------------------------------
# Entity — no world dependency
# ---------------------------------------------------------------------------


class Entity:
    # physics
    auto_jump: bool = True
    bounding_box: BoundingBox
    velocity: pygame.Vector2
    on_ground: bool = False
    hit_ceiling: bool = False
    in_water: bool = False
    x_collision: bool = False
    y_collision: bool = False

    # combat state
    is_hit: bool = False
    is_parrying: bool = False
    is_blocking: bool = False
    is_attacking: bool = False
    is_staggered: bool = False
    is_dead: bool = False

    # timers
    attack_cooldown: float = 0.0
    hit_timer: float = 0.0
    stagger_timer: float = 0.0
    attack_frameindex: int = 0

    def __init__(self, stats: EntityStats, x: float, y: float) -> None:
        self.stats = stats

        self.maxhealth = stats.maxhealth
        self.health = float(stats.maxhealth)

        self.maxstagger = stats.maxstagger
        self.stagger = float(stats.maxstagger)

        self.walk_speed = stats.walk_speed
        self.walkspeed = stats.walk_speed
        self.dmg = stats.dmg
        self.attackspeed = stats.attack_speed
        self.default_attack_speed = stats.attack_speed
        self.armor_value = stats.armor_value
        self.passivregen = stats.passivregen
        self.jump_power = stats.jump_power
        self.gravity = stats.gravity
        self.sprint_speed = stats.sprint_speed

        self.bounding_box = BoundingBox(
            position=pygame.Vector2(x, y),
            size=pygame.Vector2(*stats.bbox_size),
        )
        self.velocity = pygame.Vector2(0, 0)

        self.on_ground = False
        self.hit_ceiling = False
        self.in_water = False
        self.x_collision = False
        self.y_collision = False

        self.is_hit = False
        self.is_parrying = False
        self.is_blocking = False
        self.is_attacking = False
        self.is_staggered = False
        self.is_dead = False

        self.attack_cooldown = 0.0
        self.hit_timer = 0.0
        self.stagger_timer = 0.0
        self.attack_frameindex = 0

    # --- position ---

    @property
    def position(self) -> pygame.Vector2:
        return self.bounding_box.position

    @position.setter
    def position(self, value: pygame.Vector2) -> None:
        self.bounding_box.position = value

    @property
    def x(self) -> float:
        return self.bounding_box.position.x

    @x.setter
    def x(self, value: float) -> None:
        self.bounding_box.position.x = value

    @property
    def y(self) -> float:
        return self.bounding_box.position.y

    @y.setter
    def y(self, value: float) -> None:
        self.bounding_box.position.y = value

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @xy.setter
    def xy(self, value: tuple[float, float]) -> None:
        self.x, self.y = value

    # --- velocity ---

    @property
    def vel_x(self) -> float:
        return self.velocity.x

    @vel_x.setter
    def vel_x(self, value: float) -> None:
        self.velocity.x = value

    @property
    def vel_y(self) -> float:
        return self.velocity.y

    @vel_y.setter
    def vel_y(self, value: float) -> None:
        self.velocity.y = value

    @property
    def vel_xy(self) -> tuple[float, float]:
        return (self.velocity.x, self.velocity.y)

    @vel_xy.setter
    def vel_xy(self, value: tuple[float, float]) -> None:
        self.velocity.x, self.velocity.y = value

    # --- physics (pure, no world) ---

    def apply_gravity(self, dt: float, multiplier: float = 1.0) -> None:
        self.vel_y += self.gravity * dt * multiplier

    def apply_physics_result(self, result: PhysicsResult) -> None:
        """Consume a PhysicsResult produced by the game loop's sweep_collision call."""
        self.position = result.position
        self.on_ground = result.on_ground
        self.hit_ceiling = result.hit_ceiling
        self.x_collision = result.x_collision
        self.y_collision = result.y_collision

        if self.on_ground and self.vel_y < 0:
            self.vel_y = 0.0
        elif self.hit_ceiling and self.vel_y > 0:
            self.vel_y = 0.0

    # --- movement ---

    def jump(self) -> None:
        if self.on_ground:
            self.vel_y += self.jump_power
            self.on_ground = False

    def swim_up(self) -> None:
        if self.in_water:
            self.vel_y += 1.0

    def swim_down(self) -> None:
        if self.in_water:
            self.vel_y -= 1.0

    # --- update (pure combat/timer logic only) ---

    def update_entity(self, dt: float, in_water: bool) -> None:
        """
        Update timers, stagger, gravity, water drag.
        Physics (sweep collision) is handled externally — call apply_physics_result()
        with the result afterward.
        """
        if self.is_dead:
            return

        if self.attack_cooldown > 0:
            self.attack_cooldown -= dt

        if self.hit_timer > 0:
            self.hit_timer -= dt
        else:
            self.is_hit = False

        self._update_stagger(dt)
        self._regen_stagger(dt)

        self.in_water = in_water
        self.apply_gravity(dt)

        if self.in_water:
            self.velocity *= BLOCK_SPEED[Block.WATER.value]

    # --- damage ---

    def take_damage(self, damage: float, stagger_damage: float) -> None:
        if self.is_dead:
            return

        self.is_hit = True
        self.hit_timer = 150.0

        self.health -= damage * (1.0 - self.armor_value / 100.0)
        self.stagger -= stagger_damage

        if self.health <= 0:
            self.die()

    # --- stagger ---

    def _update_stagger(self, dt: float) -> None:
        if self.stagger <= 0 and not self.is_staggered:
            self.is_staggered = True
            self.walkspeed = 0.0
            self.attackspeed = 0.0
            self.stagger_timer = 1000.0

        if self.is_staggered:
            self.stagger_timer -= dt
            if self.stagger_timer <= 0:
                self.is_staggered = False
                self.stagger = float(self.maxstagger)
                self.walkspeed = self.walk_speed
                self.attackspeed = self.default_attack_speed

    def _regen_stagger(self, dt: float) -> None:
        if not self.is_hit and not self.is_staggered:
            if self.stagger < self.maxstagger:
                self.stagger = min(self.stagger + 0.02 * dt, float(self.maxstagger))

    # --- combat ---

    def attack(self) -> None:
        if self.attack_cooldown <= 0 and not self.is_staggered:
            self.is_attacking = True
            self.attack_frameindex = 0
            self.attack_cooldown = self.attackspeed

    def die(self) -> None:
        self.is_dead = True
        self.walkspeed = 0.0
        self.vel_x = 0.0
        self.is_attacking = False

    def regeneration(self, dt: float) -> None:
        if not self.is_hit and self.health < self.maxhealth:
            self.health = min(
                self.health + self.passivregen * dt / 1000.0,
                float(self.maxhealth),
            )

    # --- serialization ---

    def to_json(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "vel_x": self.vel_x,
            "vel_y": self.vel_y,
            "health": self.health,
            "stagger": self.stagger,
            "on_ground": self.on_ground,
            "in_water": self.in_water,
            "is_hit": self.is_hit,
            "is_parrying": self.is_parrying,
            "is_blocking": self.is_blocking,
            "is_attacking": self.is_attacking,
            "is_staggered": self.is_staggered,
            "is_dead": self.is_dead,
            "attack_cooldown": self.attack_cooldown,
            "hit_timer": self.hit_timer,
            "stagger_timer": self.stagger_timer,
            "attack_frameindex": self.attack_frameindex,
        }

    def from_json(self, data: dict) -> None:
        self.x = data["x"]
        self.y = data["y"]
        self.vel_x = data["vel_x"]
        self.vel_y = data["vel_y"]
        self.health = data["health"]
        self.stagger = data["stagger"]
        self.on_ground = data["on_ground"]
        self.in_water = data["in_water"]
        self.is_hit = data["is_hit"]
        self.is_parrying = data["is_parrying"]
        self.is_blocking = data["is_blocking"]
        self.is_attacking = data["is_attacking"]
        self.is_staggered = data["is_staggered"]
        self.is_dead = data["is_dead"]
        self.attack_cooldown = data["attack_cooldown"]
        self.hit_timer = data["hit_timer"]
        self.stagger_timer = data["stagger_timer"]
        self.attack_frameindex = data["attack_frameindex"]


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------


class Player(Entity):
    def __init__(self, x: float, y: float) -> None:
        super().__init__(PLAYER_STATS, x, y)

    def parry(self) -> None:
        self.is_parrying = True
        self.walkspeed = self.walk_speed * 0.5

    def stop_parry(self) -> None:
        self.is_parrying = False
        if not self.is_staggered:
            self.walkspeed = self.walk_speed

    def block(self) -> None:
        self.is_blocking = True

    def stop_block(self) -> None:
        self.is_blocking = False

    def swim_up(self) -> None:
        if self.in_water:
            self.vel_y += 1.0 / BLOCK_SPEED[Block.WATER.value]

    def swim_down(self) -> None:
        if self.in_water:
            self.vel_y -= 1.0 / BLOCK_SPEED[Block.WATER.value]

    def update_entity(self, dt: float, in_water: bool) -> None:
        super().update_entity(dt, in_water)
        self.regeneration(dt)


# ---------------------------------------------------------------------------
# Mob
# ---------------------------------------------------------------------------


class Mob(Entity):
    detect_range_x: float = 200.0
    detect_range_y: float = 120.0
    chase_range_x: float = 350.0
    chase_range_y: float = 200.0

    def __init__(
        self,
        x: float,
        y: float,
        stats: EntityStats = ZOMBIE_STATS,
    ) -> None:
        super().__init__(stats, x, y)
        self.has_aggro = False

    @classmethod
    def from_data(cls, data: dict) -> "Mob":
        stats = ENTITY_REGISTRY.get(data["type"], ZOMBIE_STATS)
        mob = cls(x=data["x"], y=data["y"], stats=stats)
        mob.from_json(data)
        return mob

    def to_json(self) -> dict:
        return super().to_json() | {
            "type": "Zombie",  # subclasses override this
            "has_aggro": self.has_aggro,
        }

    def from_json(self, data: dict) -> None:
        super().from_json(data)
        self.has_aggro = data.get("has_aggro", False)

    def focus_player(self, player: Player) -> None:
        dx = abs(player.x - self.x)
        dy = abs(player.y - self.y)

        if not self.has_aggro:
            if dx < self.detect_range_x and dy < self.detect_range_y:
                self.has_aggro = True
        else:
            if dx > self.chase_range_x or dy > self.chase_range_y:
                self.has_aggro = False

    def move_towards_player(self, player: Player) -> None:
        if not self.has_aggro or self.is_staggered:
            self.vel_x = 0.0
            return

        if player.x > self.x:
            self.vel_x = self.walk_speed
        elif player.x < self.x:
            self.vel_x = -self.walk_speed
        else:
            self.vel_x = 0.0

    def update_focus(self, player: Player, dt: float, in_water: bool) -> None:
        self.focus_player(player)
        self.move_towards_player(player)
        super().update_entity(dt, in_water)

    def draw(
        self,
        surface: pygame.Surface,
        resolution: tuple[int, int],
        player_x: float,
        player_y: float,
    ):
        pass
