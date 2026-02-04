import pathlib

import pygame

asset_dir = pathlib.Path(__file__).parent.resolve()

TILE_SIZE = 32

HOTBAR = pygame.image.load(asset_dir / "hotbar.png")
HOTBAR = pygame.transform.scale2x(HOTBAR)
HOTBAR_SELECTED = pygame.image.load(asset_dir / "hotbar_selected.png")
HOTBAR_SELECTED = pygame.transform.scale2x(HOTBAR_SELECTED)
HOTBAR_RIM = 2 * 2

COLOR_SKY = pygame.Color(118, 183, 194)

STONE_BLOCK = pygame.Surface((TILE_SIZE, TILE_SIZE))
STONE_BLOCK.fill(pygame.Color(100, 100, 100))

DIRT_BLOCK = pygame.Surface((TILE_SIZE, TILE_SIZE))
DIRT_BLOCK.fill(pygame.Color(64, 43, 26))

GRASS_BLOCK = pygame.Surface((TILE_SIZE, TILE_SIZE))
GRASS_BLOCK.fill(pygame.Color(31, 105, 55))

WATER_BLOCK = pygame.Surface((TILE_SIZE, TILE_SIZE))
WATER_BLOCK.fill(pygame.Color(43, 134, 204, 125))

UNKNOWN_BLOCK = pygame.Surface((TILE_SIZE, TILE_SIZE))
UNKNOWN_BLOCK.fill(pygame.Color(255, 0, 255))
