import math
from time import time
from typing import Optional

import pygame

from src.assets import TILE_SIZE
from src.blocks import BLOCK_ID_MASK, BLOCK_SPEED, Block, Item
from src.collision import BoundingBox, sweep_collision
from src.inventory import Hotbar, Inventory
from src.utils import screen_to_world, to_block, world_to_screen
from src.world import (
    World,
)

PLAYER_SPRITE = pygame.Surface((32, 64))
PLAYER_SPRITE.fill((255, 255, 0))
PX = pygame.Surface((4, 4))
PX.fill((0, 255, 255))

BLOCK_SELECTION = pygame.Surface((32, 32), flags=pygame.SRCALPHA)
BLOCK_SELECTION.fill(pygame.Color(255, 255, 255, 255 // 5))


class Player:
    velocity: pygame.Vector2 = pygame.Vector2(0, 0)
    bounding_box: BoundingBox = BoundingBox(
        position=pygame.Vector2(0, 0), size=pygame.Vector2(0.8, 1.8)
    )
    speed: float = 5
    sprint_speed: float = 8
    jump_power: float = 12
    gravity: float = -9.81
    on_ground: bool = False
    hit_ceiling: bool = False
    sliding: bool = False
    slide_timer: float = 0
    cursor_position: tuple[int, int] = (0, 0)
    cursor_position_world: tuple[float, float] = (0.0, 0.0)
    world: World
    inventory: Inventory = Inventory()
    hotbar: Hotbar
    in_water: bool = False
    break_progress: tuple[float, float, int, int] | None = None  # duration, start, x, y

    def __init__(
        self,
        x: float,
        y: float,
        world: World,
        delta_t: Optional[float],
    ) -> None:
        self.x = x
        self.y = y
        if delta_t is not None:
            self.delta_t = delta_t

        self.hotbar = Hotbar()
        self.world = world
        self.inventory.add_stack((Item.TORCH, 100))

    def handle_mousewheel(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEWHEEL:
            self.hotbar.handle_scroll(event.y)

    def handle_input(self, resolution: tuple[int, int], delta_t: float) -> None:
        keys = pygame.key.get_pressed()

        self.vel_x = 0

        speed = self.sprint_speed if keys[pygame.K_LSHIFT] else self.speed

        if keys[pygame.K_a]:
            self.vel_x = -speed
        if keys[pygame.K_d]:
            self.vel_x = speed

        # springen
        if keys[pygame.K_SPACE] and self.on_ground:
            self.vel_y += self.jump_power
            self.on_ground = False

        # basic schwimmen
        elif (keys[pygame.K_SPACE] or keys[pygame.K_w]) and self.in_water:
            self.vel_y += 1.0 / BLOCK_SPEED[Block.WATER.value]
        if keys[pygame.K_s] and self.in_water:
            self.vel_y -= 1.0 / BLOCK_SPEED[Block.WATER.value]

        # sliden (nur am Boden + mit Sprint)
        if (
            keys[pygame.K_s]
            and keys[pygame.K_LSHIFT]
            and self.on_ground
            and not self.sliding
        ):
            self.sliding = True
            self.slide_timer = 20
            self.vel_x *= 2

        self.hotbar.handle_keys(keys)

        mouse_left, mouse_middle, mouse_right = pygame.mouse.get_pressed()
        self.cursor_position = pygame.mouse.get_pos()

        self.cursor_position_world = screen_to_world(
            self.x,
            self.y,
            self.cursor_position[0],
            self.cursor_position[1],
            resolution[0],
            resolution[1],
            TILE_SIZE,
        )
        if mouse_right:
            self.handle_right_click()

        elif mouse_left:
            self.handle_left_click(delta_t)

    def apply_gravity(self, delta_t: float, multiplier: float = 1.0) -> None:
        # if self.on_ground:
        #     return
        self.vel_y += self.gravity * delta_t * multiplier

    def handle_left_click(self, delta_t: float) -> bool:
        x, y = to_block(*self.cursor_position_world)

        # not in break progress or block is different => new process
        if self.break_progress is None or self.break_progress[2:4] != (x, y):
            self.break_progress = (1.0, time(), x, y)
            return False

        duration = self.break_progress[0]
        start = self.break_progress[1]
        now = time()

        if now - start < duration:
            return False

        self.break_progress = None

        block = self.world.destroy_block(x, y)
        if not block:
            return False
        item = block.get_item()
        if item:
            self.inventory.add_stack((item, 1))

        self.break_start = None

        return True

    def handle_right_click(self) -> None:
        item = self.inventory.get_slot(self.hotbar.selected_slot)
        if not item:
            return
        block = item[0].get_block()

        if block:
            was_set = self.world.set_block(
                self.cursor_position_world[0],
                self.cursor_position_world[1],
                block,
            )
            if was_set:
                self.inventory.increment_slot_count(self.hotbar.selected_slot, -1)

    def update(self, delta_t: float, resolution: tuple[int, int]) -> None:
        touching_blocks = self.get_touching_blocks()
        self.in_water = Block.WATER.value in touching_blocks
        self.handle_input(resolution, delta_t)

        if self.sliding:
            self.slide_timer -= 1
            self.vel_x *= 0.95
            if self.slide_timer <= 0:
                self.sliding = False

        self.apply_gravity(delta_t)

        if self.in_water:
            self.velocity *= BLOCK_SPEED[Block.WATER.value]

        self.position, _, self.on_ground, self.hit_ceiling = sweep_collision(
            bounding_box=self.bounding_box,
            velocity=self.velocity * delta_t,
            is_solid=self.world.is_solid,
        )

        if self.on_ground and self.velocity.y < 0:
            self.velocity.y = 0
        elif self.hit_ceiling and self.velocity.y > 0:
            self.velocity.y = 0

    def get_touching_blocks(self, inset: float = 0.1) -> set[int]:
        touching_blocks = set()
        check_points = [
            (
                self.bounding_box.left + inset,
                self.bounding_box.bottom + inset,
            ),  # Bottom-left
            (
                self.bounding_box.right - inset,
                self.bounding_box.bottom + inset,
            ),  # Bottom-right
            (
                self.bounding_box.left + inset,
                self.bounding_box.top - inset,
            ),  # Top-left
            (
                self.bounding_box.right - inset,
                self.bounding_box.top - inset,
            ),  # Top-right
            (
                self.bounding_box.center.x,
                self.bounding_box.center.y,
            ),  # Center
        ]
        for point in check_points:
            block = self.world.chunk_manager.get_block(point[0], point[1])
            if block is not None:
                touching_blocks.add(BLOCK_ID_MASK & block)

        return touching_blocks

    def draw(self, surface: pygame.Surface, resolution: tuple[int, int]) -> None:
        x = resolution[0] // 2 - (1 - self.bounding_box.size.x) * TILE_SIZE / 2
        y = resolution[1] // 2 - 64
        surface.blit(PLAYER_SPRITE, (x, y))

        pos_world = (
            math.floor(self.cursor_position_world[0]),
            math.floor(self.cursor_position_world[1]),
        )

        pos_screen_x, pos_screen_y = world_to_screen(
            self.x,
            self.y,
            pos_world[0],
            pos_world[1],
            resolution[0],
            resolution[1],
            TILE_SIZE,
        )

        surface.blit(
            BLOCK_SELECTION,
            (pos_screen_x, pos_screen_y - TILE_SIZE),
            special_flags=pygame.BLEND_ALPHA_SDL2,
        )

        self.hotbar.draw(surface, self.inventory.get_hotbar_slots())

    @property
    def position(self) -> pygame.Vector2:
        return self.bounding_box.position

    @position.setter
    def position(self, value: pygame.Vector2) -> None:
        self.bounding_box.position = value

    @property
    def x(self) -> float:
        return self.position.x

    @property
    def y(self) -> float:
        return self.position.y

    @x.setter
    def x(self, value: float) -> None:
        self.position.x = value

    @y.setter
    def y(self, value: float) -> None:
        self.position.y = value

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
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @xy.setter
    def xy(self, value: tuple[float, float]) -> None:
        self.x, self.y = value

    @property
    def vel_xy(self) -> tuple[float, float]:
        return (self.vel_x, self.vel_y)

    @vel_xy.setter
    def vel_xy(self, value: tuple[float, float]) -> None:
        self.vel_x, self.vel_y = value
