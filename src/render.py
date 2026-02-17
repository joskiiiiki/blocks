from typing import cast

import numpy as np
import pygame

from src import assets, blocks
from src.assets import COLOR_SKY, TILE_SIZE
from src.blocks import BLOCK_ID_MASK, Block
from src.lighting import LightingManagerGL
from src.utils import camera_to_world, world_to_screen
from src.world import ChunkManager

PX = pygame.Surface((assets.TILE_SIZE // 16, assets.TILE_SIZE // 16))
PX.fill((255, 255, 0))


class ChunkRenderer:
    chunk_manager: ChunkManager
    tile_size: int = assets.TILE_SIZE  # FIXED: Removed duplicate declaration
    screen: pygame.Surface
    font: pygame.Font
    lit_tile_cache: dict[tuple[int, float, float, float], pygame.Surface] = {}
    lighting_dirty: bool = False
    last_lit_chunks: set[int] = set()
    lighting_manager: LightingManagerGL

    def __init__(
        self,
        chunk_manager: ChunkManager,
        tile_size: int,
        screen: pygame.Surface,
        lighting_manager: LightingManagerGL,
    ):
        self.chunk_manager = chunk_manager
        self.tile_size = tile_size
        self.screen = screen
        self.font = pygame.Font(None, 16)
        self.lighting_manager = lighting_manager

    def render(
        self,
        cam_x: float,
        cam_y: float,
    ):
        screen_width, screen_height = self.screen.get_size()

        # Calculate visible blocks
        blocks_x = int(screen_width / self.tile_size)
        blocks_y = int(screen_height / self.tile_size)

        # Calculate world coordinates for screen corners
        upper_right = camera_to_world(
            cam_x, cam_y, blocks_x // 2 + 1, blocks_y // 2 + 1
        )
        lower_left = camera_to_world(
            cam_x, cam_y, -blocks_x // 2 - 1, -blocks_y // 2 - 1
        )

        min_chunk_x = self.chunk_manager.get_chunk_x(lower_left[0])
        max_chunk_x = self.chunk_manager.get_chunk_x(upper_right[0])

        self.chunk_manager.load_chunks_only(range(min_chunk_x, max_chunk_x + 1))

        current_chunks = set(range(min_chunk_x, max_chunk_x + 1))

        if self._should_update_lighting(current_chunks):
            self.lighting_manager.calculate_lighting_region(min_chunk_x, max_chunk_x)
            self.last_lit_chunks = current_chunks
            self.lighting_dirty = False

        for chunk_x in range(min_chunk_x, max_chunk_x + 1):
            chunk = self.chunk_manager.get_chunk_from_cache(chunk_x)
            if chunk is None:
                continue

            lightmap = self.lighting_manager.get_lightmap(chunk_x)

            base_x = chunk_x * self.chunk_manager.width

            # Fixed: Use int() instead of np.floor/ceil for indices
            clip_x_min = int(lower_left[0]) - base_x
            clip_x_max = int(upper_right[0]) - base_x + 1
            clip_y_min = int(max(lower_left[1], 0))
            clip_y_max = int(min(upper_right[1], self.chunk_manager.height - 1)) + 1

            range_x = range(
                max(0, clip_x_min), min(self.chunk_manager.width, clip_x_max)
            )
            range_y = range(
                max(0, clip_y_min), min(self.chunk_manager.height, clip_y_max)
            )

            for x in range_x:
                for y in range_y:
                    tile_idx = chunk.blocks[x, y] & BLOCK_ID_MASK
                    tile = Block.get_texture_from_id(tile_idx)

                    if (
                        tile_idx == Block.WATER.value
                        and chunk.blocks[x, y + 1] == Block.AIR.value
                    ):
                        tile = assets.TEXTURES.get_texture("water_top")

                    if lightmap is None:
                        continue

                    world_x = base_x + x
                    world_y = y

                    pixel_x, pixel_y = world_to_screen(
                        cam_x,
                        cam_y,
                        world_x,
                        world_y,
                        screen_width,
                        screen_height,
                        assets.TILE_SIZE,
                    )

                    dest = (pixel_x, pixel_y - assets.TILE_SIZE)

                    light_r = lightmap[x, y, 0]
                    light_g = lightmap[x, y, 1]
                    light_b = lightmap[x, y, 2]

                    if tile is None:
                        if not min(light_r, light_b, light_g) == 1.0:
                            r = 255 * light_r
                            g = 255 * light_g
                            b = 255 * light_b
                            pygame.draw.rect(
                                self.screen,
                                (r, g, b),
                                (dest[0], dest[1], TILE_SIZE, TILE_SIZE),
                            )

                        continue

                    lit_tile = self._apply_lighting(
                        tile, tile_idx, light_r, light_g, light_b
                    )

                    self.screen.blit(lit_tile, dest)

                    # self.screen.blit(PX, (pixel_x, pixel_y - PX.height))

            #             yt = self.font.render(f"{y}", False, (255, 255, 255))
            #             xt = self.font.render(f"{x}", False, (255, 255, 255))

            #             self.screen.blit(yt, (pixel_x, pixel_y - 14))
            #             self.screen.blit(xt, (pixel_x, pixel_y - 14 - 16))

            # self.screen.blit(PX, (0, 0))

    def _should_update_lighting(self, current_chunks: set[int]) -> bool:
        if self.lighting_dirty:
            return True

        if self.last_lit_chunks != current_chunks:
            return True

        return False

    def _apply_lighting(
        self,
        tile: pygame.Surface,
        tile_idx: int,
        light_r: float,
        light_g: float,
        light_b: float,
    ) -> pygame.Surface:
        """Apply RGB lighting to a tile surface"""
        # Round light values for caching (reduces cache size)
        cache_key = (
            tile_idx,
            round(light_r, 2),
            round(light_g, 2),
            round(light_b, 2),
        )

        # Check cache first
        if cache_key in self.lit_tile_cache:
            return self.lit_tile_cache[cache_key]

        # Create a copy of the tile
        lit_tile = tile.copy()

        # Convert to pixel array for faster manipulation
        pixel_array = pygame.surfarray.pixels3d(lit_tile)

        # Apply RGB multipliers
        # Convert to float for calculation, then back to uint8
        lit_pixels = pixel_array.astype(np.float32)
        lit_pixels[:, :, 0] *= light_r  # Red channel
        lit_pixels[:, :, 1] *= light_g  # Green channel
        lit_pixels[:, :, 2] *= light_b  # Blue channel

        # Clip to valid range and convert back
        np.copyto(pixel_array, np.clip(lit_pixels, 0, 255).astype(np.uint8))

        del pixel_array  # Release the pixel array lock

        # Cache the result (limit cache size)
        if len(self.lit_tile_cache) > 1000:
            # Clear oldest entries (simple strategy)
            self.lit_tile_cache.clear()

        self.lit_tile_cache[cache_key] = lit_tile
        return lit_tile

    def mark_lighting_dirty(self):
        """Call this when blocks change to trigger lighting recalculation"""
        self.lighting_dirty = True
        self.lit_tile_cache.clear()  # Clear cache since tiles will look different
