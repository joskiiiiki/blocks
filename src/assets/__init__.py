import pathlib

import pygame

asset_dir = pathlib.Path(__file__).parent.resolve()

TILE_SIZE = 32
UNKNOWN_BLOCK = pygame.Surface((TILE_SIZE, TILE_SIZE))
UNKNOWN_BLOCK.fill(pygame.Color(255, 0, 255))
HOTBAR = pygame.image.load(asset_dir / "hotbar.png")
HOTBAR = pygame.transform.scale2x(HOTBAR)
HOTBAR_SELECTED = pygame.image.load(asset_dir / "hotbar_selected.png")
HOTBAR_SELECTED = pygame.transform.scale2x(HOTBAR_SELECTED)
HOTBAR_RIM = 2 * 2

COLOR_SKY = pygame.Color(118, 183, 194)


class Textures:
    _textures: dict[str, pygame.Surface | None] = {
        "stone": None,
        "dirt": None,
        "grass": None,
        "water": None,
        "water_top": None,
        "log": None,
        "leaves": None,
        "unknown": UNKNOWN_BLOCK,
    }

    def __init__(self):
        pass

    def get_texture(self, name: str) -> pygame.Surface:
        texture = self._textures.get(name, UNKNOWN_BLOCK)
        if texture is None:
            return UNKNOWN_BLOCK
        return texture

    def load(self) -> None:
        for name in self._textures:
            # check if path exists
            if (
                not (asset_dir / f"{name}.png").exists()
                or not (asset_dir / f"{name}.png").is_file()
            ):
                self._textures[name] = None
            else:
                self._textures[name] = pygame.image.load(
                    asset_dir / f"{name}.png"
                ).convert_alpha()


TEXTURES = Textures()
