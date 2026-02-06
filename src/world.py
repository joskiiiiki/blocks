from __future__ import annotations

import json
import math
import os
import pathlib
import queue
import threading
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy as np
import opensimplex
import pygame
from numpy.typing import NDArray
from opensimplex import OpenSimplex
from platformdirs import user_data_path

from src.blocks import Block, BlockData, is_solid


def world_path(name: str) -> Path:
    return user_data_path("blocks") / name


MARKER = pygame.Surface((8, 8))
MARKER.fill((255, 0, 0))

CHUNK_WIDTH = 32
CHUNK_HEIGHT = 512


class WorldGenContext:
    seed: int
    generator: np.random.RandomState
    noise: OpenSimplex
    base_height: int = 256

    def __init__(
        self,
        seed: int,
    ):
        self.seed = seed
        self.generator = np.random.RandomState(seed=seed)
        self.noise = OpenSimplex(seed=seed)
        opensimplex.seed(seed)

    def fractal_noise_1d(
        self,
        x_coords: NDArray[np.int64],
        chunk_x: int,
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
        scale=0.02,
        width: int = CHUNK_WIDTH,
    ):
        """Generate 1D fractal noise using numpy"""
        height_map = np.zeros(len(x_coords), dtype=np.float64)
        y = np.zeros(1, dtype=np.float64)

        for octave in range(octaves):
            frequency = lacunarity**octave
            amplitude = persistence**octave

            # Generate smooth noise for this octave
            x_scaled = (x_coords + chunk_x * width) * scale * frequency

            # Simple interpolated noise using sine waves
            octave_noise = self.noise.noise2array(
                x_scaled,
                y,
            )[0]
            height_map += octave_noise * amplitude

        return height_map


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
    blocks: NDArray[BlockData]

    def __init__(
        self,
        chunk_x: int,
        blocks: NDArray[BlockData] | None = None,
        status: ChunkStatus = ChunkStatus.UNGENERATED,
    ):
        self.chunk_x = chunk_x
        self.status = status

        if blocks is None:
            self.blocks = np.zeros((self.width, self.height), dtype=BlockData)
        else:
            self.blocks = blocks

    @staticmethod
    def from_data(chunk_x: int, data: ChunkData, blocks: NDArray[BlockData]) -> "Chunk":
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
        noise_scale_x: float = 0.03,
        noise_scale_y: float = 0.03,
        noise_threshold: float = 0.15,  # FIXED TYPO: "treshold" → "threshold"
        worm_scale_x: float = 0.08,
        worm_scale_y: float = 0.08 * 1.6,
        worm_threshold: float = 0.4,  # FIXED TYPO
        min_cave_y: int = 20,
        max_cave_y: int = 256,
    ):
        # 🔒 CRITICAL: Clamp to actual chunk bounds (prevents IndexError)
        min_y = max(0, min_cave_y)
        max_y = min(self.height - 1, max_cave_y)  # height-1 because indices are 0-based

        # ✅ EARLY EXIT: Nothing to process
        if min_y > max_y:
            return

        # 📏 Calculate region height ONCE (avoids off-by-one errors)
        region_height = max_y - min_y + 1

        # 🗺️ Generate WORLD coordinates (NOT chunk-local!) for noise coherence
        x_world = np.arange(
            self.chunk_x * self.width, (self.chunk_x + 1) * self.width, dtype=np.float32
        )
        y_world = np.arange(min_y, max_y + 1, dtype=np.float32)  # +1 for inclusive end

        # ⚠️ CRITICAL FIX: noise2array returns (y_samples, x_samples) - MUST transpose!
        noise = ctx.noise.noise2array(
            x_world * noise_scale_x, y_world * noise_scale_y
        ).T  # Shape: (width, region_height)

        worm_noise = ctx.noise.noise2array(
            x_world * worm_scale_x, y_world * worm_scale_y
        ).T  # Shape: (width, region_height)

        # ✅ CORRECT MASK APPLICATION (avoids "mask doesn't cover whole chunk" bug)
        # 1. Get VIEW of the vertical slice (modifies original array)
        region = self.blocks[:, min_y : max_y + 1]  # Shape: (width, region_height)

        # 2. Create mask ONLY for stone blocks in this region
        stone_mask = region == Block.STONE.value

        # 3. Skip noise eval if no stone exists (massive speedup in sky regions)
        if not np.any(stone_mask):
            return

        # 4. Combine noise conditions
        cave_mask = stone_mask & (
            (noise > noise_threshold) | (worm_noise > worm_threshold)
        )

        # 5. ✅ SAFE ASSIGNMENT: Modify the VIEW (updates original self.blocks)
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
            block = self.blocks[x, y]
            # dont generate on non solids (air / water) or another tree
            if is_solid(block) and block != Block.LEAVES and block != Block.LOG:
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


class ChunkManager:
    width: int = CHUNK_WIDTH
    height: int = CHUNK_HEIGHT
    region_size: int
    chunk_cache: dict[int, Chunk]
    world_dir: Path | None = None
    _running: bool = True
    save_queue: queue.Queue[tuple[int, Chunk]]
    save_thread: threading.Thread
    generation_queue: queue.Queue[int]
    generation_thread: threading.Thread
    _lock: threading.Lock
    gen_ctx: WorldGenContext

    def __init__(
        self,
        gen_ctx: WorldGenContext,
        path: Optional[os.PathLike] = None,
        width: int = 32,
        height: int = 512,
        region_size: int = 32,
    ):
        self.height = height
        self.width = width
        self.region_size = region_size
        self.chunk_cache = dict()
        # path is optional to allow running headless for testing
        if path:
            self.world_dir = pathlib.Path(path)
            self.world_dir.mkdir(exist_ok=True, parents=True)

        self.save_queue = queue.Queue()
        self.generation_queue = queue.Queue()
        self._lock = threading.Lock()

        self.save_thread = threading.Thread(target=self._save_worker, daemon=False)
        self.generation_thread = threading.Thread(
            target=self._generation_worker, daemon=False
        )

        self.gen_ctx = gen_ctx

    def start(self):
        self._running = True

        if self.world_dir:
            self.save_thread.start()
            self.generation_thread.start()

    def _save_worker(self):
        while self._running:
            try:
                chunk_x, chunk = self.save_queue.get(timeout=0.5)
                self._write_chunk_to_disk(chunk_x, chunk)
                self.save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error saving chunk {chunk_x}: {e}")
                self.save_queue.task_done()

        print("Processing remaining save requests")
        while not self.save_queue.empty():
            try:
                chunk_x, chunk = self.save_queue.get_nowait()
                self._write_chunk_to_disk(chunk_x, chunk.copy())
                self.save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error saving chunk {chunk_x}: {e}")
                self.save_queue.task_done()

        print("Save worker exiting")

    def _generation_worker(self):
        while self._running:
            try:
                chunk_x = self.generation_queue.get(timeout=0.5)
                self._generate_chunk(chunk_x)
                self.generation_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error generating chunk {chunk_x}: {e}")
                self.generation_queue.task_done()

    def queue_generation(self, x: int):
        self.generation_queue.put(x)

    def _generate_chunk(self, x: int):
        chunk = Chunk(chunk_x=x)
        chunk.generate(self.gen_ctx)

        left, right = self.get_neighbours(chunk.chunk_x)
        left_status = left.status if left is not None else None
        right_status = right.status if right is not None else None
        chunk.decorate(left_status, right_status, self.gen_ctx)

        with self._lock:
            self.chunk_cache[x] = chunk

    def decorate_chunk(self, chunk: Chunk) -> bool:
        left, right = self.get_neighbours(chunk.chunk_x)

        left_status = left.status if left is not None else None
        right_status = right.status if right is not None else None
        return chunk.decorate(left_status, right_status, self.gen_ctx)

    def get_neighbours(self, x: int) -> tuple[Chunk | None, Chunk | None]:
        with self._lock:
            left = self.get_chunk(x - 1)
            right = self.get_chunk(x + 1)

        return left, right

    def get_chunk_from_cache(self, x: int) -> Chunk | None:
        with self._lock:
            return self.chunk_cache.get(x, None)

    def load_chunks_only(self, chunks: Iterable[int]):
        # FIXED: Corrected logic to properly load desired chunks and unload others
        chunks_set = set(chunks)

        with self._lock:
            # Find chunks to unload (in cache but not in desired set)
            to_unload = [x for x in self.chunk_cache.keys() if x not in chunks_set]

        # Unload chunks not in the desired set
        for x in to_unload:
            self.unload_chunk(x)

        # Load chunks that should be loaded
        for chunk in chunks_set:
            self.load_chunk(chunk)

    def load_chunk(self, x: int, generate_if_not_exists: bool = True):
        with self._lock:
            if x in self.chunk_cache:
                return None

        chunk = self.get_chunk_from_disk(x)
        if chunk is not None:
            with self._lock:
                self.chunk_cache[x] = chunk
            return

        if generate_if_not_exists:
            self.queue_generation(x)

    def load_chunks(self, range: Iterable[int]):
        for x in range:
            self.load_chunk(x)

    def unload_chunk(self, x: int):
        with self._lock:
            chunk = self.chunk_cache.pop(x, None)
            if chunk is not None:
                self.write_chunk(x, chunk)

    def unload_chunks(self, range: Iterable[int]):
        for x in range:
            self.unload_chunk(x)

    def write_chunk(self, x: int, chunk: Optional[Chunk] = None):
        if chunk is None:
            chunk = self.get_chunk_from_cache(x)

        if chunk is not None:
            chunk = chunk.copy()
            self.save_queue.put((x, chunk))

    def _write_chunk_to_disk(self, x: int, chunk: Chunk):
        if chunk is None or not self.world_dir:
            return

        region = x // self.region_size
        region_path = self.world_dir / str(region)
        region_path.mkdir(parents=True, exist_ok=True)

        chunk_path = region_path / f"{x}.npy"
        data_path = chunk_path.with_suffix(".json")

        # atomic write to dont mess up the world accidentally not that i care but for good measure
        tmp = chunk_path.with_suffix(".tmp")
        with tmp.open("wb") as t:
            np.save(t, chunk.blocks, allow_pickle=False)
        tmp.replace(chunk_path)

        tmp = data_path.with_suffix(".tmp")
        with tmp.open("w") as t:
            json.dump(chunk.data(), t)
        tmp.replace(data_path)

    def get_chunk_from_disk(self, x: int) -> Chunk | None:
        if not self.world_dir:
            return None

        region = x // self.region_size
        chunk_path = self.world_dir / str(region) / f"{x}.npy"
        data_path = self.world_dir / str(region) / f"{x}.json"

        if (
            not chunk_path.exists()
            or not chunk_path.is_file()
            or not data_path.exists()
            or not data_path.is_file()
        ):
            return None

        blocks = np.load(chunk_path, allow_pickle=False)
        data = json.load(data_path.open())
        return Chunk.from_data(chunk_x=x, blocks=blocks, data=data)

    def get_chunk(self, x: int) -> Chunk | None:
        if not self.world_dir:
            return None

        chunk = self.chunk_cache.get(x)
        if chunk:
            return chunk

        chunk = self.get_chunk_from_disk(x)

        if chunk is None:
            return None

        self.chunk_cache[x] = chunk
        return chunk

    def save_world(self):
        with self._lock:
            chunks_to_save = list(self.chunk_cache.keys())  # copy for iteration issues

        print(f"Queueing {len(chunks_to_save)} chunks for saving")
        for chunk in chunks_to_save:
            self.write_chunk(chunk)

    def get_chunk_x(self, x: float) -> int:
        return int(x) // self.width

    def world_to_chunk(self, x: float, y: float) -> tuple[int, float, float] | None:
        """
        Converts world coordinates to chunk coordinates.

        Parameters
        ----------
        x : float
            The x-coordinate in world space.
        y : float
            The y-coordinate in world space.

        Returns
        -------
        chunk_x : int
            The x-coordinate of the chunk in chunk space.
        chunk_local_x : float
            The x-coordinate inside the chunk from the lower left corner.
        chunk_local_y : float
            The y-coordinate inside the chunk from the lower left corner.
        """
        chunk_x = math.floor(x) // self.width
        chunk_local_x = x % self.width

        if y < 0 or y >= self.height:
            return None

        return chunk_x, chunk_local_x, y

    def set_block(
        self, x: float, y: float, block: BlockData, allow_replace: bool = False
    ) -> bool:
        """
        Set the block at the given coordinates.

        Parameters
        ----------
        x : float
            The x-coordinate in world space.
        y : float
            The y-coordinate in world space.
        block : BlockData
            The block data to set.

        Returns
        -------
        bool
            True if the block was set successfully, False otherwise.
        """
        coords = self.world_to_chunk(x, y)
        if coords is None:
            return False

        chunk_x, chunk_local_x, chunk_local_y = coords

        chunk = self.get_chunk_from_cache(chunk_x)
        if chunk is None:
            return False

        x = int(math.floor(chunk_local_x))
        y = int(math.floor(y))

        with self._lock:
            if not allow_replace and chunk.blocks[x, y] != Block.AIR.value:
                return False

            chunk.blocks[x, y] = block

        return True

    def destroy_block(self, x: float, y: float) -> BlockData | None:
        """
        Destroy the block at the given coordinates. (replaces with AIR)

        Parameters
        ----------
        x : float
            The x-coordinate in world space.
        y : float
            The y-coordinate in world space.

        Returns
        -------
        Block | None
            The block that was destroyed, or None if the block was not destroyed.
        """
        coords = self.world_to_chunk(x, y)
        if coords is None:
            return None

        chunk_x, chunk_local_x, chunk_local_y = coords

        chunk = self.get_chunk_from_cache(chunk_x)
        if chunk is None:
            return None

        x = int(math.floor(chunk_local_x))
        y = int(math.floor(y))

        with self._lock:
            block = chunk.blocks[x, y]
            chunk.blocks[x, y] = Block.AIR.value

        return block

    def shutdown(self):
        print("Shutting down ChunkManager ...")
        self.save_world()
        self.save_queue.join()

        self._running = False

        self.save_thread.join(timeout=10)
        self.generation_thread.join(timeout=10)

        if self.save_thread.is_alive():
            print("Save thread did not terminate within timeout")
        else:
            print("Save thread terminated successfully")
        if self.generation_thread.is_alive():
            print("Generation thread did not terminate within timeout")
        else:
            print("Generation thread terminated successfully")

    def get_block(self, x: float, y: float) -> BlockData | None:
        coords = self.world_to_chunk(x, y)
        if not coords:
            return None
        chunk_x, x, y = coords

        xn = int(np.floor(x))
        yn = int(np.floor(y))

        chunk = self.get_chunk_from_cache(chunk_x)
        if chunk is None:
            return None
        block: BlockData = chunk.blocks[xn, yn]

        return block


class WorldData:
    seed: int
    height: int
    width: int
    region_size: int

    def __init__(self, seed: int, height: int, width: int, region_size: int):
        self.seed = seed
        self.height = height
        self.width = width
        self.region_size = region_size

    @staticmethod
    def from_file(
        file: int | str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None | WorldData:
        with open(file, "r") as f:
            s = f.read()
            if len(s) == 0:
                s = "{}"
            return WorldData.from_json(json.loads(s))

    @staticmethod
    def from_json(data: Any) -> None | WorldData:
        if not data:
            return None
        if not isinstance(data, dict):
            return None
        attributes = {"seed": int, "height": int, "width": int, "region_size": int}
        if not all(
            key in data and isinstance(data[key], attributes[key]) for key in attributes
        ):
            return None

        return WorldData(**data)

    def save(self, file: int | str | bytes | os.PathLike[str] | os.PathLike[bytes]):
        with open(file, "w") as f:
            json.dump(self.to_json(), f)

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "height": self.height,
            "width": self.width,
            "region_size": self.region_size,
        }


class World:
    chunk_manager: ChunkManager
    world_path: pathlib.Path
    player_pos: tuple[float, float] = (0, 0)
    world_data: WorldData

    def __init__(self, path: pathlib.Path):
        self.world_path = path

        exists = path.exists() and path.is_dir()

        if not exists:
            self.world_path.mkdir(parents=True, exist_ok=True)
            self.world_data = self.new_world_data()
            world_data_path = path / "world.json"
            world_data_path.touch(exist_ok=True)
            self.world_data.save(world_data_path)
        else:
            world_data_path = path / "world.json"
            world_data_path.touch(exist_ok=True)
            world_data = WorldData.from_file(world_data_path)
            if world_data is None:
                self.world_data = self.new_world_data()
                world_data_path.unlink()
                world_data_path.touch(exist_ok=True)
                self.world_data.save(world_data_path)
            else:
                self.world_data = world_data

        print(self.world_data.to_json())

        self.gen_ctx = WorldGenContext(self.world_data.seed)
        self.chunk_manager = ChunkManager(
            gen_ctx=self.gen_ctx,
            height=self.world_data.height,
            width=self.world_data.width,
            region_size=self.world_data.region_size,
            path=self.world_path,
        )

        self.chunk_manager.start()

    def new_world_data(self) -> WorldData:
        return WorldData(
            seed=np.random.randint(0, 2**32 - 1),
            height=CHUNK_HEIGHT,
            width=CHUNK_WIDTH,
            region_size=32,
        )

    def update_chunk_cache(self):
        min_chunk = int(self.player_pos[0]) // self.chunk_manager.width - 4
        max_chunk = (
            int(self.player_pos[0]) // self.chunk_manager.width + 4
        )  # FIXED: Changed from player_pos[1] to player_pos[0]
        self.chunk_manager.load_chunks_only(range(min_chunk, max_chunk + 1))

    def get_block(self, x: float, y: float) -> Block | None:
        block = self.chunk_manager.get_block(x, y)
        if block is None:
            return None
        return Block(block)

    def set_block(self, x: float, y: float, block: Block) -> bool:
        return self.chunk_manager.set_block(x, y, block.value)

    def destroy_block(self, x: float, y: float) -> Block | None:
        return Block(self.chunk_manager.destroy_block(x, y))

    def is_solid(self, x: float, y: float) -> bool:
        block = self.chunk_manager.get_block(x, y)
        if block is None:
            return False
        return is_solid(block)
