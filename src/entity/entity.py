from __future__ import annotations
import src.inventory

from dataclasses import dataclass
from typing import Protocol

import pygame

from src.bboxed import BoundingBoxed

from src.blocks import BLOCK_SPEED, Block, damage_of_item
from src.collision import BoundingBox
from src.entity.stats import EntityStats


class EntityConstructor(Protocol):
    def __call__(self, x: float, y: float) -> Entity: ...


ENTITY_REGISTRY: dict[str, EntityConstructor] = {}


@dataclass
class PhysicsResult:
    position: pygame.Vector2
    on_ground: bool
    hit_ceiling: bool
    x_collision: bool
    y_collision: bool


class Entity(BoundingBoxed):
    # physics
    auto_jump: bool = True
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

    _walk_speed_modifier: float = 1.0
    _attack_speed_modifier: float = 1.0

    attack_bbox_size: pygame.Vector2

    def __init__(self, stats: EntityStats, x: float, y: float) -> None:
        self.stats = stats

        self.health = float(stats.maxhealth)

        self.stagger = float(stats.maxstagger)

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

        self.attack_bbox_size = pygame.Vector2(stats.attack_range, stats.bbox_size[1])

        super().__init__(BoundingBox.new_from_tuples((x, y), stats.bbox_size))

    def held_stack(self) -> None | src.inventory.Stack:
        return None

    @property
    def damage(self) -> float:
        stack = self.held_stack()
        if stack:
            return damage_of_item(stack[0]) * self.stats.dmg
        return self.stats.dmg

    @property
    def maxhealth(self) -> int:
        return self.stats.maxhealth

    @property
    def maxstagger(self) -> int:
        return self.stats.maxstagger

    @property
    def jump_power(self) -> float:
        return self.stats.jump_power

    @property
    def gravity(self) -> float:
        return self.stats.gravity

    @property
    def armor_value(self) -> float:
        return self.stats.armor_value

    @property
    def passivregen(self) -> float:
        return self.stats.passivregen

    @property
    def attack_speed(self) -> float:
        return self.stats.attack_speed * self._attack_speed_modifier

    @property
    def walk_speed(self) -> float:
        return self.stats.walk_speed * self._walk_speed_modifier

    @property
    def knockback(self) -> float:
        return self.stats.knockback

    def attack_bbox(self) -> BoundingBox:
        size = self.attack_bbox_size
        offset = (
            pygame.Vector2(0, 0)
            if self.is_facing_right
            else self.bounding_box.size - self.attack_bbox_size
        )
        position = self.position + offset
        return BoundingBox(position, size)

    @property
    def is_facing_right(self) -> bool:
        return self.velocity.x >= 0

    # --- position ---

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
        self.vel_x, self.vel_y = value

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

    def take_damage(
        self, damage: float, stagger_damage: float, knockback: pygame.Vector2
    ) -> None:
        if self.is_dead:
            return

        self.is_hit = True
        self.hit_timer = 150.0

        self.velocity += knockback

        self.health -= damage * (1.0 - self.armor_value / 100.0)
        self.stagger -= stagger_damage

        if self.health <= 0:
            self.die()

    # --- stagger ---

    def _update_stagger(self, dt: float) -> None:
        if self.stagger <= 0 and not self.is_staggered:
            self.is_staggered = True
            self._walk_speed_modifier = 0.0
            self._attack_speed_modifier = 0.0
            self.stagger_timer = 1000.0

        if self.is_staggered:
            self.stagger_timer -= dt
            if self.stagger_timer <= 0:
                self.is_staggered = False
                self.stagger = float(self.maxstagger)
                self._walk_speed_modifier = 1.0
                self._attack_speed_modifier = 1.0

    def _regen_stagger(self, dt: float) -> None:
        if not self.is_hit and not self.is_staggered:
            if self.stagger < self.maxstagger:
                self.stagger = min(self.stagger + 0.02 * dt, float(self.maxstagger))

    # --- combat ---

    def attack(self) -> bool:
        if self.attack_cooldown <= 0 and not self.is_staggered:
            self.attack_frameindex = 0
            self.attack_cooldown = self.attack_speed
            return True
        return False

    def die(self) -> None:
        self.is_dead = True
        self._walk_speed_modifier = 0.0
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
