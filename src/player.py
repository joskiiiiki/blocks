from __future__ import annotations

import math
from time import time
from typing import Optional

import pygame

from src.assets import TILE_SIZE
from src.blocks import BLOCK_ID_MASK, BLOCK_SPEED, Block, Item
from src.entity import Player as _PlayerBase
from src.inventory import Hotbar, Inventory
from src.utils import screen_to_world, to_block, world_to_screen
from src.world import World

# --- sprites ---

PLAYER_SPRITE = pygame.Surface((32, 64))
PLAYER_SPRITE.fill((255, 255, 0))

BLOCK_SELECTION = pygame.Surface((32, 32), flags=pygame.SRCALPHA)
BLOCK_SELECTION.fill(pygame.Color(255, 255, 255, 255 // 5))


class Player(_PlayerBase):
    """Full player: extends the combat/physics Player with world interaction,
    inventory, input handling, block breaking/placing, and rendering."""

    # --- world interaction ---
    break_progress: tuple[float, float, int, int] | None = None
    cursor_position: tuple[int, int] = (0, 0)
    cursor_position_world: tuple[float, float] = (0.0, 0.0)

    default_walk_speed = 5
    sprint_speed = 8

    # --- slide ---
    sliding: bool = False
    slide_timer: float = 0.0

    def __init__(
        self,
        x: float,
        y: float,
        world: World,
        delta_t: Optional[float] = None,
    ) -> None:
        super().__init__(x=x, y=y, world=world)

        if delta_t is not None:
            self.delta_t = delta_t

        self.inventory = Inventory()
        self.hotbar = Hotbar()

        self.inventory.add_stack((Item.TORCH, 100))
        self.inventory.add_stack((Item.COPPER_TORCH, 100))

        self.sliding = False
        self.slide_timer = 0.0
        self.break_progress = None
        self.cursor_position = (0, 0)
        self.cursor_position_world = (0.0, 0.0)

    # --- input ---

    def handle_mousewheel(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEWHEEL:
            self.hotbar.handle_scroll(event.y)

    def handle_input(self, resolution: tuple[int, int], delta_t: float) -> None:
        keys = pygame.key.get_pressed()

        self.vel_x = 0.0
        speed = self.sprint_speed if keys[pygame.K_LSHIFT] else self.default_walk_speed

        if keys[pygame.K_a]:
            self.vel_x = -speed
        if keys[pygame.K_d]:
            self.vel_x = speed

        # jump
        if keys[pygame.K_SPACE] and self.on_ground:
            self.jump()

        # swimming
        elif (keys[pygame.K_SPACE] or keys[pygame.K_w]) and self.in_water:
            self.swim_up()
        if keys[pygame.K_s] and self.in_water:
            self.swim_down()

        # slide: ground + sprinting only
        # if (
        #     keys[pygame.K_s]
        #     and keys[pygame.K_LSHIFT]
        #     and self.on_ground
        #     and not self.sliding
        # ):
        #     self.sliding = True
        #     self.slide_timer = 20.0
        #     self.vel_x *= 2.0

        self.hotbar.handle_keys(keys)

        mouse_left, _, mouse_right = pygame.mouse.get_pressed()
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

    # --- swim uses BLOCK_SPEED (needs game imports, so defined here) ---

    def swim_up(self) -> None:
        if self.in_water:
            self.vel_y += 1.0 / BLOCK_SPEED[Block.WATER.value]

    def swim_down(self) -> None:
        if self.in_water:
            self.vel_y -= 1.0 / BLOCK_SPEED[Block.WATER.value]

    # --- block breaking / placing ---

    def handle_left_click(self, delta_t: float) -> bool:
        x, y = to_block(*self.cursor_position_world)

        if self.break_progress is None or self.break_progress[2:4] != (x, y):
            self.break_progress = (0.05, time(), x, y)
            return False

        duration = self.break_progress[0]
        start = self.break_progress[1]

        if time() - start < duration:
            return False

        self.break_progress = None

        block = self.world.destroy_block(x, y)
        if not block:
            return False

        item = block.get_item()
        if item:
            self.inventory.add_stack((item, 1))

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

    # --- world queries ---

    # --- update ---

    def update(self, delta_t: float, resolution: tuple[int, int]) -> None:
        if self.is_dead:
            return

        self.handle_input(resolution, delta_t)

        # slide decay
        if self.sliding:
            self.slide_timer -= 1.0
            self.vel_x *= 0.95
            if self.slide_timer <= 0:
                self.sliding = False

        # entity system: timers, stagger, regen
        super().update_entity(delta_t)

    # --- draw ---

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
