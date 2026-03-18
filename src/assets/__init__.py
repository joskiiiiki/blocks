import os
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
COLOR_SKY


class Texture:
    def __init__(self, path: str | os.PathLike) -> None:
        self.path = pathlib.Path(path)
        self.__surface: pygame.Surface | None = None

    def surface(self) -> pygame.Surface:
        if self.__surface is None:
            if self.path.exists():
                self.__surface = pygame.image.load(self.path).convert_alpha()
            else:
                self.__surface = UNKNOWN_BLOCK
        return self.__surface


class Animation:
    def __init__(self, path: str | os.PathLike, num_frames: int) -> None:
        self.path = pathlib.Path(path)
        self.num_frames = num_frames
        self._frames: list[pygame.Surface] | None = None
        self._flipped: list[pygame.Surface] | None = None

    def _load(self) -> None:
        if not self.path.exists():
            return
        sheet = pygame.image.load(self.path).convert_alpha()
        w = sheet.get_width() // self.num_frames
        h = sheet.get_height()
        self._frames = [
            sheet.subsurface(pygame.Rect(i * w, 0, w, h))
            for i in range(self.num_frames)
        ]

    def _flip(self) -> None:
        if self._flipped is None:
            if self._frames is None:
                self._load()
            if self._frames:
                self._flipped = [
                    pygame.transform.flip(frame, True, False) for frame in self._frames
                ]

    def frame(self, index: int, flipped: bool = False) -> pygame.Surface | None:
        if self._frames is None:
            self._load()

        if self._flipped is None and flipped:
            self._flip()

        if not self._frames or (flipped and not self._flipped):
            return None

        if flipped:
            if self._flipped is None:
                return None
            return self._flipped[index % len(self._flipped)]

        return self._frames[index % len(self._frames)]

    def by_progress(
        self, progress: float, flipped: bool = False
    ) -> pygame.Surface | None:
        frame_count = len(self._frames) if self._frames else 0
        index = int(progress * frame_count)
        return self.frame(index, flipped)


ANIMATIONS: dict[str, Animation] = {
    "attack_start": Animation(asset_dir / "player" / "attack_start.png", num_frames=2),
    "attack_loop": Animation(asset_dir / "player" / "attack_loop.png", num_frames=4),
    "idle": Animation(asset_dir / "player" / "idle.png", num_frames=4),
    "walk": Animation(asset_dir / "player" / "walk.png", num_frames=8),
    "jump": Animation(asset_dir / "player" / "jump.png", num_frames=6),
    "run": Animation(asset_dir / "player" / "run.png", num_frames=8),
}


def get_animation(name: str) -> Animation:
    return ANIMATIONS[name]


TEXTURES: dict[str, Texture] = {
    "player": Texture(asset_dir / "player" / "player.png"),
    "arrow": Texture(asset_dir / "arrow.png"),
    "axe": Texture(asset_dir / "axe.png"),
    "azurite": Texture(asset_dir / "azurite.png"),
    "azurite_ore": Texture(asset_dir / "azurite_ore.png"),
    "bow": Texture(asset_dir / "bow.png"),
    "chestplate": Texture(asset_dir / "chestplate.png"),
    "coal": Texture(asset_dir / "coal.png"),
    "coal_ore": Texture(asset_dir / "coal_ore.png"),
    "copper_torch": Texture(asset_dir / "copper_torch.png"),
    "diamond": Texture(asset_dir / "diamond.png"),
    "diamond_ore": Texture(asset_dir / "diamond_ore.png"),
    "dirt": Texture(asset_dir / "dirt.png"),
    "emerald": Texture(asset_dir / "emerald.png"),
    "emerald_ore": Texture(asset_dir / "emerald_ore.png"),
    "grass": Texture(asset_dir / "grass.png"),
    "helmet": Texture(asset_dir / "helmet.png"),
    "hotbar": Texture(asset_dir / "hotbar.png"),
    "hotbar_selected": Texture(asset_dir / "hotbar_selected.png"),
    "iron_ingot": Texture(asset_dir / "iron_ingot.png"),
    "iron_ore": Texture(asset_dir / "iron_ore.png"),
    "ironquartz_ingot": Texture(asset_dir / "ironquartz_ingot.png"),
    "ironquartz_ore": Texture(asset_dir / "ironquartz_ore.png"),
    "leaves": Texture(asset_dir / "leaves.png"),
    "log": Texture(asset_dir / "log.png"),
    "pants": Texture(asset_dir / "pants.png"),
    "pickaxe_icon": Texture(asset_dir / "pickaxe_icon.png"),
    "planks": Texture(asset_dir / "planks.png"),
    "shadow_gem": Texture(asset_dir / "shadow_gem.png"),
    "shadow_ore": Texture(asset_dir / "shadow_ore.png"),
    "shoes": Texture(asset_dir / "shoes.png"),
    "stick": Texture(asset_dir / "stick.png"),
    "stone": Texture(asset_dir / "stone.png"),
    "stone_background": Texture(asset_dir / "stone_background.png"),
    "sword": Texture(asset_dir / "sword.png"),
    "torch": Texture(asset_dir / "torch.png"),
    "unknown": Texture(asset_dir / "unknown.png"),
    "void_ore": Texture(asset_dir / "void_ore.png"),
    "void_shard": Texture(asset_dir / "void_shard.png"),
    "water": Texture(asset_dir / "water.png"),
    "water_top": Texture(asset_dir / "water_top.png"),
}


def get_texture(name: str) -> Texture:
    return TEXTURES.get(name, TEXTURES["unknown"])
