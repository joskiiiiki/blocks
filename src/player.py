import math
from typing import Optional

import pygame

from src.assets import TILE_SIZE
from src.inventory import Hotbar, Inventory
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
    inventory: Inventory = Inventory()
    hotbar: Hotbar

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
        self.hotbar = Hotbar(self.screen)
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

        if keys[pygame.K_c]:
            self._intersections.clear()

        self.hotbar.handle_keys(keys)

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
            self.handle_right_click()

        elif mouse_left:
            self.handle_left_click()

    def apply_gravity(self) -> None:
        # if self.on_ground:
        #     return
        self.vel_y += self.gravity * self.delta_t

    def handle_left_click(self) -> None:
        block = self.world.destroy_block(
            self.cursor_position_world[0],
            self.cursor_position_world[1],
        )
        if not block:
            return
        item = block.get_item()
        if item:
            self.inventory.add_stack((item, 1))

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

    def set_position_tuple(self, position: tuple[float, float]) -> None:
        self.x, self.y = position

    def handle_intersection(self, primary_ctx: IntersectionContext):
        self._intersections.append(primary_ctx)

        match primary_ctx.type:
            case IntersectionType.ORTHOGONAL_X:
                print("ort x")
                self.vel_x = 0
                xs, ys = primary_ctx.start
                xi, yi = primary_ctx.intersect
                dir_x = primary_ctx.direction.vector[0]

                self.set_position(xi - dir_x / 10000, yi)
            case IntersectionType.ORTHOGONAL_Y:
                self.vel_y = 0
                xs, ys = primary_ctx.start
                xi, yi = primary_ctx.intersect
                dir_y = primary_ctx.direction.vector[1]

                self.set_position(xi, yi - dir_y / 10000)
                if primary_ctx.direction == IntersectionDirection.DOWN:
                    self.on_ground = True

            case IntersectionType.DIAGONAL_HORIZONTAL:
                print("dig hor")
                self.vel_y = 0

                secondary_ctx = primary_ctx.next
                xs, ys = primary_ctx.start
                xi, yi = primary_ctx.intersect
                xe, ye = primary_ctx.end
                dir_y = primary_ctx.direction.vector[1]

                if secondary_ctx:
                    xis, yis = secondary_ctx.intersect
                    dir_x = secondary_ctx.direction.vector[0]
                    print(xis, yis)
                    self.set_position(xis - dir_x / 10000, yi - dir_y / 10000)
                    self.vel_x = 0
                else:
                    self.set_position(xe, yi - dir_y / 10000)

            case IntersectionType.DIAGONAL_VERTICAL:
                print("dig vert")
                self.vel_x = 0

                secondary_ctx = primary_ctx.next
                xs, ys = primary_ctx.start
                xi, yi = primary_ctx.intersect
                xe, ye = primary_ctx.end
                dir_x = primary_ctx.direction.vector[0]

                if secondary_ctx:
                    print(f"sec vert {secondary_ctx.intersect}")
                    xis, yis = secondary_ctx.intersect
                    dir_y = secondary_ctx.direction.vector[1]
                    self.set_position(xi - dir_x / 10000, yis - dir_y / 10000)
                    self.vel_y = 0
                else:
                    self.set_position(xi - dir_x / 10000, ye)

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

        self.hotbar.draw(self.screen, self.inventory.get_hotbar_slots())

        for index, intersection in enumerate(self._intersections):
            screen_start_x, screen_start_y = world_to_screen(
                self.x,
                self.y,
                intersection.start[0],
                intersection.start[1],
                self.screen.width,
                self.screen.height,
                TILE_SIZE,
            )

            pygame.draw.circle(
                self.screen, (255, 0, 0), (screen_start_x, screen_start_y), 1
            )

            screen_end_x, screen_end_y = world_to_screen(
                self.x,
                self.y,
                intersection.end[0],
                intersection.end[1],
                self.screen.width,
                self.screen.height,
                TILE_SIZE,
            )

            pygame.draw.line(
                self.screen,
                (255, 0, 0),
                (screen_start_x, screen_start_y),
                (screen_end_x, screen_end_y),
                2,
            )

            if intersection.next:
                screen_next_x, screen_next_y = world_to_screen(
                    self.x,
                    self.y,
                    intersection.next.end[0],
                    intersection.next.end[1],
                    self.screen.width,
                    self.screen.height,
                    TILE_SIZE,
                )

                pygame.draw.line(
                    self.screen,
                    (0, 255, 255),
                    (screen_end_x, screen_end_y),
                    (screen_next_x, screen_next_y),
                    2,
                )
