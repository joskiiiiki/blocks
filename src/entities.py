import math

import pygame

from src.assets import TILE_SIZE
from src.entity import Mob
from src.utils import world_to_screen
from src.world import World

ENEMY_SPRITE = pygame.Surface((32, 64))
ENEMY_SPRITE.fill((255, 0, 0))


class Enemy(Mob):
    def __init__(self, x: int, y: int, world: World):
        super().__init__(x, y, world)

    def draw(
        self,
        surface: pygame.Surface,
        player_x: float,
        player_y: float,
        resolution: tuple[int, int],
    ) -> None:
        if self.is_dead:
            return

        screen_x, screen_y = world_to_screen(
            player_x,
            player_y,
            self.x,
            self.y,
            resolution[0],
            resolution[1],
            TILE_SIZE,
        )

        # offset so feet sit on the ground (sprite height = bbox height in pixels)
        sprite_w = int(self.bounding_box.size.x * TILE_SIZE)
        sprite_h = int(self.bounding_box.size.y * TILE_SIZE)
        surface.blit(ENEMY_SPRITE, (screen_x, screen_y - sprite_h))

        # health bar
        bar_w = sprite_w
        bar_h = 4
        bar_x = screen_x
        bar_y = screen_y - sprite_h - 8

        pygame.draw.rect(surface, (80, 0, 0), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(
            surface,
            (200, 0, 0),
            (bar_x, bar_y, int(bar_w * self.health / self.maxhealth), bar_h),
        )

        # stagger bar
        bar_y += bar_h + 2
        pygame.draw.rect(surface, (30, 30, 80), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(
            surface,
            (80, 80, 220),
            (bar_x, bar_y, int(bar_w * self.stagger / self.maxstagger), bar_h),
        )
