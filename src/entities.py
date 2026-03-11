import pygame

from src.assets import TILE_SIZE
from src.entity import ZOMBIE_STATS
from src.mob import Mob
from src.utils import world_to_screen

ENEMY_SPRITE = pygame.Surface((32, 64))
ENEMY_SPRITE.fill((255, 0, 0))


class Zombie(Mob):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, stats=ZOMBIE_STATS)

    def draw(
        self,
        surface: pygame.Surface,
        resolution: tuple[int, int],
        player_x: float,
        player_y: float,
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
        sprite_w = ENEMY_SPRITE.width
        sprite_h = ENEMY_SPRITE.height
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
