from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from src.blocks import BLOCK_ID_MASK, Block, BlockData
from src.world import ChunkManager
from src.world.chunk import Chunk

LightChannel: TypeAlias = np.float32
Light: TypeAlias = tuple[float, float, float]
LightData: TypeAlias = tuple[LightChannel, LightChannel, LightChannel]
LightMask: TypeAlias = npt.NDArray[LightChannel]
TransparencyMask: TypeAlias = npt.NDArray[np.bool]

SUN_LIGHT: Light = (1.0, 1.0, 1.0)
TRANSPARENT_BLOCKS = {Block.AIR.value, Block.WATER.value}
BLOCK_LIGHTS: dict[int, Light] = {
    Block.TORCH.value: (0.8, 0.6, 0.4),
}

MAX_AIR_PROPAGATION: int = 16
MAX_BLOCK_PROPAGATION: int = 4
FALLOFF_AIR: float = 1 / MAX_AIR_PROPAGATION
FALLOFF_FACTOR_DIAGONAL = np.sqrt(2)
FALLOFF_BLOCK: float = 1 / MAX_BLOCK_PROPAGATION


# returns 3d nd_array of light source blocks light
def light_source_map(chunk: Chunk) -> LightMask:
    mask: LightMask = np.zeros((chunk.width, chunk.height, 3), dtype=LightChannel)
    for x in range(chunk.width):
        transparent_idx: int | None = None
        for y in range(chunk.height):
            block_id: BlockData = chunk.blocks[x, y] & BLOCK_ID_MASK

            if block_id in TRANSPARENT_BLOCKS:
                if transparent_idx is None:
                    transparent_idx = y
                continue
            else:
                transparent_idx = None

            if block_id in BLOCK_LIGHTS:
                red, green, blue = BLOCK_LIGHTS[block_id]
                mask[x, y, 0] = red
                mask[x, y, 1] = green
                mask[x, y, 2] = blue

        if transparent_idx is not None:
            mask[x, transparent_idx:, 0] = SUN_LIGHT[0]
            mask[x, transparent_idx:, 1] = SUN_LIGHT[1]
            mask[x, transparent_idx:, 2] = SUN_LIGHT[2]

    return mask


def smooth_light_map_step(
    light_map: LightMask,
    transparency_mask: TransparencyMask,
    light_sources: LightMask,  # ADD THIS - preserve source emissions
    is_block_light: bool = False,
) -> LightMask:
    """Single iteration of light propagation (vectorized)"""
    width, height, _ = light_map.shape
    falloff_air = FALLOFF_BLOCK if is_block_light else FALLOFF_AIR
    falloff_solid = FALLOFF_BLOCK  # Higher falloff for solid blocks

    # Start with current light map
    new_light_map = light_map.copy()

    # Create padded version for easy neighbor access
    padded = np.pad(
        light_map, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=0
    )

    # Extract all neighbors at once using slicing
    # Cardinal neighbors
    upper = padded[1:-1, :-2, :]  # [x, y-1]
    lower = padded[1:-1, 2:, :]  # [x, y+1]
    left = padded[:-2, 1:-1, :]  # [x-1, y]
    right = padded[2:, 1:-1, :]  # [x+1, y]

    # Diagonal neighbors
    upper_left = padded[:-2, :-2, :]  # [x-1, y-1]
    upper_right = padded[2:, :-2, :]  # [x+1, y-1]
    lower_left = padded[:-2, 2:, :]  # [x-1, y+1]
    lower_right = padded[2:, 2:, :]  # [x+1, y+1]

    # Expand transparency mask to 3D for broadcasting
    transparent_3d = transparency_mask[:, :, np.newaxis]

    # Calculate falloff based on whether block is transparent or solid
    falloff = np.where(transparent_3d, falloff_air, falloff_solid)

    # Stack cardinal neighbors and find max after applying falloff
    cardinal_neighbors = np.stack([upper, lower, left, right], axis=0)
    cardinal_max = np.max(cardinal_neighbors - falloff, axis=0)

    # Stack diagonal neighbors and find max with diagonal falloff
    diagonal_neighbors = np.stack(
        [upper_left, upper_right, lower_left, lower_right], axis=0
    )
    diagonal_max = np.max(
        diagonal_neighbors - (falloff * FALLOFF_FACTOR_DIAGONAL), axis=0
    )

    # Find overall max from both cardinal and diagonal
    propagated_light = np.maximum(cardinal_max, diagonal_max)
    propagated_light = np.maximum(propagated_light, 0.0)  # Clamp to 0

    # Update all blocks with propagated light
    new_light_map = np.maximum(new_light_map, propagated_light)

    # CRITICAL: Restore light source emissions (they should never dim)
    new_light_map = np.maximum(new_light_map, light_sources)

    return new_light_map


def calculate_light_map(chunk: Chunk, iterations: int = 10) -> LightMask:
    """Calculate full lighting for a chunk"""
    # Start with light sources
    light_sources = light_source_map(chunk)  # Keep this separate
    light_map = light_sources.copy()

    # Build transparency mask
    transparency_mask = np.isin(chunk.blocks & BLOCK_ID_MASK, list(TRANSPARENT_BLOCKS))

    # Propagate light multiple times
    for _ in range(iterations):
        light_map = smooth_light_map_step(light_map, transparency_mask, light_sources)

    return np.clip(light_map, 0.0, 1.0)


class LightingManager:
    def __init__(self, chunk_manager):
        self.chunk_manager = chunk_manager
        self.lightmaps: dict[int, LightMask] = {}

    def calculate_lighting_region(
        self, chunk_x_min: int, chunk_x_max: int, iterations: int = 10
    ):
        """Calculate lighting across multiple chunks"""
        # Stitch chunks together horizontally
        combined_blocks = []
        for chunk_x in range(chunk_x_min, chunk_x_max + 1):
            chunk = self.chunk_manager.get_chunk_from_cache(chunk_x)
            if chunk:
                combined_blocks.append(chunk.blocks)

        if not combined_blocks:
            return

        combined = np.concatenate(combined_blocks, axis=0)

        # Calculate lighting for combined region
        transparency_mask = np.isin(combined & BLOCK_ID_MASK, list(TRANSPARENT_BLOCKS))

        # Build light source map for the combined region
        light_sources = np.zeros((*combined.shape, 3), dtype=LightChannel)

        # Populate light sources
        chunk_width = self.chunk_manager.width
        for i, chunk_x in enumerate(range(chunk_x_min, chunk_x_max + 1)):
            chunk = self.chunk_manager.get_chunk_from_cache(chunk_x)
            if chunk:
                chunk_light_sources = light_source_map(chunk)
                start_x = i * chunk_width
                end_x = start_x + chunk_width
                light_sources[start_x:end_x] = chunk_light_sources

        light_map = light_sources.copy()

        # Propagate
        print("Propagating light")
        for _ in range(iterations):
            light_map = smooth_light_map_step(
                light_map, transparency_mask, light_sources
            )

        # Split back into chunks
        for i, chunk_x in enumerate(range(chunk_x_min, chunk_x_max + 1)):
            start_x = i * chunk_width
            end_x = start_x + chunk_width
            self.lightmaps[chunk_x] = light_map[start_x:end_x]

    def get_lightmap(self, chunk_x: int) -> LightMask | None:
        return self.lightmaps.get(chunk_x, None)

    def mark_chunks_dirty(self, chunk_x_list: list[int]):
        """Mark specific chunks as needing lighting recalculation"""
        for chunk_x in chunk_x_list:
            if chunk_x in self.lightmaps:
                del self.lightmaps[chunk_x]

                def on_block_changed(self, world_x: int, world_y: int):
                    chunk_x = self.chunk_manager.get_chunk_x(world_x)

                    # Mark affected chunks dirty (include neighbors for light propagation)
                    self.lighting_manager.mark_chunks_dirty(
                        [chunk_x - 1, chunk_x, chunk_x + 1]
                    )

                    # Mark renderer dirty
                    self.renderer.mark_lighting_dirty()
