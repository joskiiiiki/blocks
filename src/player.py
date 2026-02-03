import math
from typing import Optional

import numpy as np
import pygame

from src.render import TILE_SIZE
from src.utils import screen_to_world, world_to_screen
from src.world import (
    IntersectionContext,
    IntersectionDirection,
    IntersectionType,
    World,
)

PLAYER_SPRITE = pygame.Surface((32, 64))
PX = pygame.Surface((4, 4))
PX.fill((0, 255, 255))

BLOCK_SELECTION = pygame.Surface((32, 32), flags=pygame.SRCALPHA)
BLOCK_SELECTION.fill(pygame.Color(255, 255, 255, 255 // 5))

INTERSECTION_ORTHOGONAL_INDICATOR = pygame.Surface((4, 4))
INTERSECTION_ORTHOGONAL_INDICATOR.fill((255, 255, 0))

INTERSECTION_DIAGONAL_HOR_INDICATOR = pygame.Surface((4, 4))
INTERSECTION_DIAGONAL_HOR_INDICATOR.fill((255, 0, 255))

INTERSECTION_DIAGONAL_VER_INDICATOR = pygame.Surface((4, 4))
INTERSECTION_DIAGONAL_VER_INDICATOR.fill((0, 255, 255))


class Player:
    vel_x: float = 0
    vel_y: float = 0
    speed: float = 5
    sprint_speed: float = 8
    jump_power: float = 12
    gravity: float = -9.81
    on_ground: bool = False
    sliding: bool = False
    slide_timer: float = 0
    x: float
    y: float
    delta_t: float = 1 / 60
    screen: pygame.Surface
    cursor_position: tuple[int, int] = (0, 0)
    cursor_position_world: tuple[float, float] = (0.0, 0.0)
    world: World

    _intersections: list[IntersectionContext] = []

    def __init__(
        self,
        x: int,
        y: int,
        screen: pygame.Surface,
        world: World,
        delta_t: Optional[float],
    ) -> None:
        self.x = x
        self.y = y
        if delta_t is not None:
            self.delta_t = delta_t

        self.screen = screen
        self.world = world

    def handle_input(self) -> None:
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

        mouse_left, mouse_middle, mouse_right = pygame.mouse.get_pressed()
        self.cursor_position = pygame.mouse.get_pos()

        self.cursor_position_world = screen_to_world(
            self.x,
            self.y,
            self.cursor_position[0],
            self.cursor_position[1],
            self.screen.width,
            self.screen.height,
            TILE_SIZE,
        )
        if mouse_right:
            self.world.set_block(
                self.cursor_position_world[0],
                self.cursor_position_world[1],
                np.uint32(1),
            )

    def apply_gravity(self) -> None:
        # if self.on_ground:
        #     return
        self.vel_y += self.gravity * self.delta_t

    def update(self) -> tuple[float, float]:
        self.handle_input()

        if self.sliding:
            self.slide_timer -= 1
            self.vel_x *= 0.95
            if self.slide_timer <= 0:
                self.sliding = False

        self.apply_gravity()

        return self.x + self.vel_x * self.delta_t, self.y + self.vel_y * self.delta_t

    def set_position(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def handle_intersection(self, context: IntersectionContext):
        # self._intersections.append(context)
        _, direction, x, y, _ = context

        match direction:
            case IntersectionDirection.DOWN:
                self.on_ground = True
                self.vel_y = 0
                self.set_position(x, y)
            case IntersectionDirection.UP:
                self.vel_y = 0
                self.set_position(x, y)
            case IntersectionDirection.LEFT:
                self.vel_x = 0
                self.set_position(x, y)
            case IntersectionDirection.RIGHT:
                self.vel_x = 0
                self.set_position(x, y)

    def draw(self) -> None:
        x = self.screen.width // 2 - 16
        y = self.screen.height // 2 - 64
        self.screen.blit(PLAYER_SPRITE, (x, y))

        pos_world = (
            math.floor(self.cursor_position_world[0]),
            math.floor(self.cursor_position_world[1]),
        )

        pos_screen_x, pos_screen_y = world_to_screen(
            self.x,
            self.y,
            pos_world[0],
            pos_world[1],
            self.screen.width,
            self.screen.height,
            TILE_SIZE,
        )

        self.screen.blit(
            BLOCK_SELECTION,
            (pos_screen_x, pos_screen_y - TILE_SIZE),
            special_flags=pygame.BLEND_ALPHA_SDL2,
        )

        # for _, _, x, y, type in self._intersections:
        #     screen_x, screen_y = world_to_screen(
        #         self.x, self.y, x, y, self.screen.width, self.screen.height, TILE_SIZE
        #     )
        #     match type:
        #         case IntersectionType.ORTHOGONAL_X | IntersectionType.ORTHOGONAL_Y:
        #             pass
        #         case IntersectionType.DIAGONAL_HORIZONTAL:
        #             self.screen.blit(
        #                 INTERSECTION_DIAGONAL_HOR_INDICATOR, (screen_x, screen_y)
        #             )
        #         case IntersectionType.DIAGONAL_VERTICAL:
        #             self.screen.blit(
        #                 INTERSECTION_DIAGONAL_VER_INDICATOR, (screen_x, screen_y)
        #             )
