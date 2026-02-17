import math

import moderngl
import pygame

from src.blocks import Block


class TextureAtlas:
    ctx: moderngl.Context
    tile_size: int
    block_id_to_uv: dict[int, tuple[float, float]]
    atlas_size: int  # we do a square texture atlas - why tho TODO: check if rectangular is better

    def __init__(self, ctx: moderngl.Context, tile_size: int):
        self.ctx = ctx
        self.tile_size = tile_size
        self.block_id_to_uv = {}

    def build(self) -> moderngl.Texture | None:
        textures: list[pygame.Surface] = []
        block_ids: list[int] = []

        for block in Block:
            tex = block.get_texture()

            if tex is None:
                continue

            textures.append(tex)
            block_ids.append(block.id)

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

        for i, (texture, block_id) in enumerate(zip(textures, block_ids)):
            col = i % self.atlas_size
            row = i // self.atlas_size

            x = col * self.tile_size
            y = row * self.tile_size

            texture_flipped = pygame.transform.flip(texture, False, True)

            atlas_surface.blit(texture_flipped, (x, y))

            # normalize coordinates to uv [0, 1]
            u = x / atlas_width
            v = y / atlas_height

            self.block_id_to_uv[block_id] = (u, v)

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

    def uv(self, block_id) -> tuple[float, float]:
        uv = self.block_id_to_uv.get(block_id)
        if uv is None:
            return self.block_id_to_uv.get(Block.UNKNOWN.id, (0.0, 0.0))

        return uv[0], uv[1]

    def tile_size_normalized(self) -> float | None:
        return 1.0 / self.atlas_size if self.atlas_size > 0 else None
