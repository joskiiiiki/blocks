import json
import math
import os
import pathlib
import queue
import threading
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, TypeAlias

import numpy as np
import pygame
from numpy.typing import NDArray
from platformdirs import user_data_path

from src.blocks import Block, BlockData, is_solid


class IntersectionDirection(Enum):
    """Enum representing the direction of an intersected surface"""

    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @property
    def vector(self) -> tuple[Literal[0, 1], Literal[0, 1]]:
        return self.value


class IntersectionType(Enum):
    ORTHOGONAL_X = 0
    ORTHOGONAL_Y = 1
    DIAGONAL_HORIZONTAL = 2
    DIAGONAL_VERTICAL = 3


class IntersectionContext:
    block: BlockData
    direction: IntersectionDirection
    start: tuple[float, float]
    intersect: tuple[float, float]
    end: tuple[float, float]
    type: IntersectionType
    next: Optional["IntersectionContext"] = None

    def __init__(
        self,
        block: BlockData,
        direction: IntersectionDirection,
        start: tuple[float, float],
        intersect: tuple[float, float],
        end: tuple[float, float],
        type: IntersectionType,
        next: Optional["IntersectionContext"] = None,
    ):
        self.block = block
        self.direction = direction
        self.start = start
        self.end = end
        self.intersect = intersect
        self.type = type
        self.next = next


Chunk: TypeAlias = NDArray[BlockData]


def world_path(name: str) -> Path:
    return user_data_path("blocks") / name


MARKER = pygame.Surface((8, 8))
MARKER.fill((255, 0, 0))


class ChunkManager:
    width: int = 32
    height: int = 512
    region_size: int
    chunk_cache: dict[int, Chunk]
    world_dir: Path | None = None
    _running: bool = True
    save_queue: queue.Queue[tuple[int, Chunk]]
    save_thread: threading.Thread
    _lock: threading.Lock

    def __init__(
        self,
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
        self._lock = threading.Lock()

        self.save_thread = threading.Thread(target=self._save_worker, daemon=False)

    def start(self):
        self._running = True

        if self.world_dir:
            self.save_thread.start()

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

    def _perlin_noise_1d(
        self, x_coords, chunk_x, octaves=4, persistence=0.5, scale=0.02
    ):
        """Generate 1D fractal noise using numpy"""
        height_map = np.zeros(len(x_coords))

        for octave in range(octaves):
            frequency = 2**octave
            amplitude = persistence**octave

            # Generate smooth noise for this octave
            x_scaled = (x_coords + chunk_x * self.width) * scale * frequency

            # Simple interpolated noise using sine waves
            noise = (
                np.sin(x_scaled * 0.5) * np.cos(x_scaled * 0.7) * np.sin(x_scaled * 1.3)
            )
            noise += np.sin(x_scaled * 1.1) * 0.5

            height_map += noise * amplitude

        return height_map

    def empty_chunk(self) -> Chunk:
        return np.zeros(shape=(self.width, self.height), dtype=BlockData)

    def create_chunk(self, chunk_x: int):
        chunk: Chunk = self.empty_chunk()

        # Generate terrain height using fractal noise
        x_coords = np.arange(self.width)
        height_map = 0.5 * self._perlin_noise_1d(
            x_coords, chunk_x, octaves=5, persistence=0.5, scale=0.015
        )
        earth_layer = 5 + 0.1 * self._perlin_noise_1d(
            x_coords, chunk_x, octaves=5, persistence=0.5, scale=0.1
        )

        # Normalize height map to reasonable terrain range
        base_height = 256
        terrain_variation = 40
        heights = (height_map * terrain_variation + base_height).astype(int)

        # Fill terrain based on height map
        for x in range(self.width):
            terrain_height = min(max(heights[x], 0), self.height - 1)

            # Stone layer (everything below terrain - 5)
            if terrain_height > 5:
                chunk[x, : terrain_height - 5] = 1

            earth = earth_layer[x]

            # Dirt layer (5 blocks below surface)
            dirt_start = int(max(0, terrain_height - earth))
            chunk[x, dirt_start:terrain_height] = 2

            # Grass on top
            if terrain_height < self.height:
                chunk[x, terrain_height] = 3

            if terrain_height < base_height:
                chunk[x, terrain_height:(base_height)] = 4

        return chunk

    def generate_chunk(self, x: int):
        with self._lock:
            self.chunk_cache[x] = self.create_chunk(x)

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

    def load_chunk(self, x: int):
        with self._lock:
            if x in self.chunk_cache:
                return

        chunk = self.get_chunk_from_disk(x)
        if chunk is not None:
            with self._lock:
                self.chunk_cache[x] = chunk
            return

        self.generate_chunk(x)

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
        tmp = chunk_path.with_suffix(".tmp")

        with tmp.open("wb") as t:
            np.save(t, chunk, allow_pickle=False)
        tmp.replace(chunk_path)

    def get_chunk_from_disk(self, x: int) -> Chunk | None:
        if not self.world_dir:
            return None

        region = x // self.region_size
        chunk_path = self.world_dir / str(region) / f"{x}.npy"

        if not chunk_path.exists() or not chunk_path.is_file():
            return None

        return np.load(chunk_path, allow_pickle=False)

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
        chunk_x = int(x) // self.width
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
            if not allow_replace and chunk[x, y] != Block.AIR.value:
                return False

            chunk[x, y] = block

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
            block = chunk[x, y]
            chunk[x, y] = Block.AIR.value

        return block

    def shutdown(self):
        print("Shutting down ChunkManager ...")
        self.save_world()
        self.save_queue.join()

        self._running = False

        self.save_thread.join(timeout=10)

        if self.save_thread.is_alive():
            print("Save thread did not terminate within timeout")
        else:
            print("Save thread terminated successfully")

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
        block: BlockData = chunk[xn, yn]

        return block


class World:
    chunk_manager: ChunkManager
    path: pathlib.Path
    player_pos: tuple[float, float] = (0, 0)
    world_data: dict[str, Any]

    def __init__(self, path: pathlib.Path):
        self.chunk_manager = ChunkManager(path)
        self.world_path = path

        world_data = path / "world.json"
        world_data.touch(exist_ok=True)
        with open(world_data, "r") as f:
            s = f.read()
            if len(s) == 0:
                s = "{}"
            self.world_data = json.loads(s)

        self.chunk_manager.start()

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
