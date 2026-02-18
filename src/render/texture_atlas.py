import math

import moderngl
import pygame

from src.assets import TEXTURES
from src.blocks import BLOCK_TO_TEXTURE, Block


class TextureAtlas:
    ctx: moderngl.Context
    tile_size: int
    texture_name_to_uv: dict[str, tuple[float, float]]
    atlas_size: int  # we do a square texture atlas - why tho TODO: check if rectangular is better

    def __init__(self, ctx: moderngl.Context, tile_size: int):
        self.ctx = ctx
        self.tile_size = tile_size
        self.texture_name_to_uv = {}

    def build(self) -> moderngl.Texture | None:
        textures: list[pygame.Surface] = []
        texture_names: list[str] = []

        for key, texture in BLOCK_TO_TEXTURE.items():
            if texture is None:
                continue

            tex = TEXTURES.get_texture(texture)

            if tex is None:
                continue

            textures.append(tex)
            texture_names.append(texture)

        if len(textures) == 0:
            print("WARNING: No textures found")
            return None

        num_textures = len(textures)

        self.atlas_size = int(math.ceil(math.sqrt(num_textures)))
        atlas_width = self.atlas_size * self.tile_size
        atlas_height = self.atlas_size * self.tile_size

        print(
            f"Building atlas: {atlas_width}x{atlas_height} with {num_textures} textures"
        )

        atlas_surface = pygame.Surface((atlas_width, atlas_height), pygame.SRCALPHA)
        atlas_surface.fill((0, 0, 0, 0))

        for i, (texture, tex_name) in enumerate(zip(textures, texture_names)):
            col = i % self.atlas_size
            row = i // self.atlas_size

            x = col * self.tile_size
            y = row * self.tile_size

            texture_flipped = pygame.transform.flip(texture, False, True)

            atlas_surface.blit(texture_flipped, (x, y))

            # normalize coordinates to uv [0, 1]
            u = x / atlas_width
            v = y / atlas_height

            self.texture_name_to_uv[tex_name] = (u, v)

        pygame.image.save(atlas_surface, "atlas.png")
        # flip the atlas vertically
        texture_data = pygame.image.tobytes(atlas_surface, "RGBA", False)
        print(
            f"Texture data size: {len(texture_data)} (expected {atlas_width * atlas_height * 4})"
        )

        self.texture = self.ctx.texture(
            (atlas_width, atlas_height),
            4,
            data=texture_data,
        )
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.repeat_x = False
        self.texture.repeat_y = False

        return self.texture

    def get_unknown_uv(self) -> tuple[float, float]:
        unknown_texture = Block.UNKNOWN.get_texture_name()
        if unknown_texture is None:
            return (0.0, 0.0)
        return self.texture_name_to_uv.get(unknown_texture, (0.0, 0.0))

    def uv(self, block_data: int) -> tuple[float, float]:
        texture_name = BLOCK_TO_TEXTURE.get(block_data)
        if texture_name is None:
            return self.get_unknown_uv()

        uv = self.texture_name_to_uv.get(texture_name)
        if uv is None:
            return self.get_unknown_uv()

        return uv[0], uv[1]

    def tile_size_normalized(self) -> float | None:
        return 1.0 / self.atlas_size if self.atlas_size > 0 else None
