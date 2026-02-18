from __future__ import annotations

from enum import Enum
from typing import TypedDict

import numpy as np
import numpy.typing as npt
from pyfastnoiselite.pyfastnoiselite import FractalType  # ty:ignore[unresolved-import]

from src.blocks import BLOCK_ID_MASK, Block, BlockData, is_solid
from src.world.gen_context import WorldGenContext

CHUNK_WIDTH = 32
CHUNK_HEIGHT = 512


class ChunkStatus(Enum):
    UNGENERATED = 0
    TERRAIN_GENERATED = 1
    DECORATED = 2


class ChunkData(TypedDict):
    status: int


class Chunk:
    status: ChunkStatus = ChunkStatus.UNGENERATED
    chunk_x: int
    width: int = CHUNK_WIDTH
    height: int = CHUNK_HEIGHT
    blocks: npt.NDArray[BlockData]

    def __init__(
        self,
        chunk_x: int,
        blocks: npt.NDArray[BlockData] | None = None,
        status: ChunkStatus = ChunkStatus.UNGENERATED,
    ):
        self.chunk_x = chunk_x
        self.status = status

        if blocks is None:
            self.blocks = np.zeros((self.width, self.height), dtype=BlockData)
        else:
            self.blocks = blocks

    @staticmethod
    def from_data(
        chunk_x: int, data: ChunkData, blocks: npt.NDArray[BlockData]
    ) -> "Chunk":
        chunk = Chunk(
            chunk_x=chunk_x,
            blocks=blocks,
        )
        chunk.set_data(data)
        return chunk

    def data(self) -> ChunkData:
        return {
            "status": self.status.value,
        }

    def set_data(self, data: ChunkData):
        self.status = ChunkStatus(data["status"])

    def generate(self, gen_ctx: WorldGenContext):
        # Generate terrain height using fractal noise
        x_coords = np.arange(self.width)
        height_map = 0.5 * gen_ctx.fractal_noise_1d(
            x_coords=x_coords,
            chunk_x=self.chunk_x,
            octaves=5,
            persistence=0.5,
            lacunarity=2.0,
            scale=0.015,
            width=self.width,
        )
        earth_layer = 5 + 0.1 * gen_ctx.fractal_noise_1d(
            x_coords=x_coords,
            chunk_x=self.chunk_x,
            octaves=5,
            persistence=0.5,
            lacunarity=2.0,
            scale=0.1,
            width=self.width,
        )

        # Normalize height map to reasonable terrain range
        base_height = gen_ctx.base_height
        terrain_variation = 40
        heights = (height_map * terrain_variation + base_height).astype(int)

        # Fill terrain based on height map
        for x in range(self.width):
            terrain_height = min(max(heights[x], 0), self.height - 1)

            # Stone layer (everything below terrain - 5)
            if terrain_height > 5:
                self.blocks[x, : terrain_height - 5] = 1

            earth = earth_layer[x]

            # Dirt layer (5 blocks below surface)
            dirt_start = int(max(0, terrain_height - earth))
            self.blocks[x, dirt_start:terrain_height] = 2

            # Grass on top
            if terrain_height < self.height:
                self.blocks[x, terrain_height] = 3

            if terrain_height < base_height:
                self.blocks[x, terrain_height:(base_height)] = 4

        self.generate_caves(gen_ctx)

        self.status = ChunkStatus.TERRAIN_GENERATED

    def decorate(
        self,
        left_status: ChunkStatus | None,
        right_status: ChunkStatus | None,
        ctx: WorldGenContext,
    ) -> bool:
        """
        Decorate this chunk using the Minecraft offset pattern.
        Returns True if decoration succeeded, False if neighbors aren't ready.
        """
        # Check if already decorated
        if self.status == ChunkStatus.DECORATED:
            return True

        # Need terrain generated first
        if self.status != ChunkStatus.TERRAIN_GENERATED:
            return False

        # Need both neighbors to exist and have terrain
        if left_status is None or right_status is None:
            return False

        if (
            left_status != ChunkStatus.TERRAIN_GENERATED
            and left_status != ChunkStatus.DECORATED
        ):
            return False

        if (
            right_status != ChunkStatus.TERRAIN_GENERATED
            and right_status != ChunkStatus.DECORATED
        ):
            return False

        # Decorate the offset region (second half of left + first half of this)
        # This creates the +8 offset pattern

        # Decorate left half of this chunk (positions 0-7)
        # Trees here can extend into the left chunk safely
        self._decorate_region(0, self.width // 2, ctx)

        # Also help decorate the right half that overlaps into the right chunk
        # (This is the second half of this chunk: positions 8-15)
        # Trees here can extend into the right chunk safely
        self._decorate_region(self.width // 2, self.width, ctx)

        self.status = ChunkStatus.DECORATED
        return True

    def generate_caves(
        self,
        ctx: WorldGenContext,
        noise_scale: float = 0.03,
        noise_scale_aspect: float = 4,
        noise_threshold: float = 0.15,
        worm_scale: float = 0.04,
        worm_scale_aspect: float = 3,
        worm_threshold: float = 0.1,
        min_cave_y: int = 20,
        max_cave_y: int = 250,
    ):
        # Clamp to actual chunk bounds
        min_y = max(0, min_cave_y)
        max_y = min(self.height - 1, max_cave_y)

        if min_y > max_y:
            return

        region_height = max_y - min_y + 1

        # Generate world coordinates
        x_world = np.arange(
            self.chunk_x * self.width, (self.chunk_x + 1) * self.width, dtype=np.float32
        )
        y_world = np.arange(min_y, max_y + 1, dtype=np.float32)

        # Create meshgrid for 2D coordinates
        X, Y = np.meshgrid(x_world, y_world, indexing="ij")

        # Flatten and stack into shape (2, N)
        x_flat = X.flatten()
        y_flat = Y.flatten()
        coords = np.stack([x_flat, y_flat * noise_scale_aspect], axis=0)
        coords_worm = np.stack([x_flat, y_flat * worm_scale_aspect], axis=0)

        # Generate cave noise
        ctx.noise.frequency = noise_scale  # Assuming equal x/y scaling
        ctx.noise.fractal_type = (
            FractalType.FractalType_None
        )  # Disable fractal for caves
        noise_flat = ctx.noise.gen_from_coords(coords)
        noise = noise_flat.reshape(self.width, region_height)

        # Generate worm noise
        ctx.noise.frequency = worm_scale
        worm_flat = ctx.noise.gen_from_coords(coords_worm)
        worm_noise = worm_flat.reshape(self.width, region_height)

        # Apply caves
        region = self.blocks[:, min_y : max_y + 1]
        stone_mask = region == Block.STONE.value

        if not np.any(stone_mask):
            return

        cave_mask = stone_mask & ((noise > noise_threshold) | (
            worm_noise > worm_threshold
        ))

        region[cave_mask] = Block.AIR.value

    def _decorate_region(self, x_min: int, x_max: int, ctx: WorldGenContext):
        """Decorate a specific region of this chunk."""
        if x_min < 0 or x_max > self.width:
            raise ValueError(f"Invalid x range: {x_min}-{x_max}")

        # Use chunk position to seed randomness (consistent across runs)

        # Randomly place some trees
        num_trees = ctx.generator.randint(0, 3)  # 0-1 trees per half-chunk

        for _ in range(num_trees):
            x = ctx.generator.randint(x_min, x_max)

            # Find the ground at this x position
            ground_y = self._find_tree_ground(x)
            if ground_y is None:
                continue  # No ground found

            # Check if there's enough space above for a tree
            space_above = self._count_space_above(x, ground_y)

            if space_above >= 6:  # Need at least 6 blocks for a small tree
                self._place_tree(x, ground_y + 1, ctx)

    def _find_tree_ground(self, x: int) -> int | None:
        """Find the topmost solid block at position x. Returns None if not found."""
        for y in range(self.height - 1, -1, -1):
            block: BlockData = self.blocks[x, y]
            block_id = block & BLOCK_ID_MASK
            # dont generate on non solids (air / water) or another tree
            if (
                is_solid(block)
                and block_id != Block.LEAVES.value
                and block_id != Block.LOG.value
            ):
                return y
        return None

    def _find_ground(self, x: int) -> int | None:
        """Find the topmost solid block at position x. Returns None if not found."""
        for y in range(self.height - 1, -1, -1):
            if self.blocks[x, y] != Block.AIR.value:
                return y
        return None

    def _count_space_above(self, x: int, start_y: int) -> int:
        """Count consecutive air blocks above start_y."""
        count = 0
        for y in range(start_y + 1, self.height):
            if self.blocks[x, y] == Block.AIR.value:
                count += 1
            else:
                break
        return count

    def _place_tree(self, x: int, base_y: int, ctx: WorldGenContext):
        """Place a simple tree at the given position."""
        tree_height = ctx.generator.randint(4, 7)

        # Place trunk (logs)
        for h in range(tree_height):
            y = base_y + h
            if 0 <= y < self.height:
                self.blocks[x, y] = Block.LOG.value

        # Place leaves (simple sphere at the top)
        leaf_y = base_y + tree_height
        leaf_radius = ctx.generator.randint(2, 4)

        for dy in range(-2, 3):
            for dx in range(-leaf_radius, leaf_radius + 1):
                # Simple circular pattern
                if dx * dx + dy * dy <= leaf_radius * leaf_radius:
                    leaf_x = x + dx
                    leaf_height = leaf_y + dy

                    # Check bounds
                    if 0 <= leaf_x < self.width and 0 <= leaf_height < self.height:
                        # Only place leaves in air
                        if self.blocks[leaf_x, leaf_height] == Block.AIR.value:
                            self.blocks[leaf_x, leaf_height] = Block.LEAVES.value

    def copy(self) -> Chunk:
        return Chunk(
            chunk_x=self.chunk_x,
            status=self.status,
            blocks=self.blocks.copy(),
        )
