from PIL.ImageOps import scale
from i18n.translator import t
import platformdirs
import json
import os
import pathlib
import threading
import queue
from collections.abc import Iterable
import pygame
from datetime import datetime
from pathlib import Path
from platformdirs import user_data_path
from typing import TypeAlias, Self, TypeVar, Literal, Optional, Any
import numpy as np
from numpy.typing import NDArray

BlockData: TypeAlias = np.uint32  # Changed from uint64 to match actual usage
Chunk: TypeAlias = NDArray[BlockData]

TILE_SIZE = 32
COLOR_SKY = pygame.Color(118, 183, 194)
STONE_TILE = pygame.Surface((TILE_SIZE, TILE_SIZE))
STONE_TILE.fill(pygame.Color(100, 100, 100))

DIRT_TILE = pygame.Surface((TILE_SIZE, TILE_SIZE))
DIRT_TILE.fill(pygame.Color(64, 43, 26))

GRASS_TILE = pygame.Surface((TILE_SIZE, TILE_SIZE))
GRASS_TILE.fill(pygame.Color(31, 105, 55))

WATER_TILE = pygame.Surface((TILE_SIZE, TILE_SIZE))
WATER_TILE.fill(pygame.Color(43, 134, 204, 125))

TILES = {
    0: None,
    1: STONE_TILE,
    2: DIRT_TILE,
    3: GRASS_TILE,
    4: WATER_TILE
}

class ChunkManager:
    width: int = 32
    height: int = 512
    region_size: int
    chunk_cache: dict[int, Chunk]
    world_dir: Path
    _running: bool = True
    save_queue: queue.Queue[tuple[int, Chunk]]
    save_thread: threading.Thread
    _lock: threading.Lock

    def __init__(self, path: os.PathLike, width: int = 32, height: int = 512, region_size: int = 32):
        self.height = height
        self.width = width
        self.region_size = region_size
        self.chunk_cache = dict()
        self.world_dir = pathlib.Path(path)
        self.world_dir.mkdir(exist_ok=True, parents=True)

        self.save_queue = queue.Queue()
        self._lock = threading.Lock()
        self._running = True

        self.save_thread = threading.Thread(target=self._save_worker, daemon=False)
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

    def _perlin_noise_1d(self, x_coords, chunk_x, octaves=4, persistence=0.5, scale=0.02):
        """Generate 1D fractal noise using numpy"""
        height_map = np.zeros(len(x_coords))

        for octave in range(octaves):
            frequency = 2 ** octave
            amplitude = persistence ** octave

            # Generate smooth noise for this octave
            x_scaled = (x_coords + chunk_x * self.width) * scale * frequency

            # Simple interpolated noise using sine waves
            noise = np.sin(x_scaled * 0.5) * np.cos(x_scaled * 0.7) * np.sin(x_scaled * 1.3)
            noise += np.sin(x_scaled * 1.1) * 0.5

            height_map += noise * amplitude

        return height_map

    def create_chunk(self, chunk_x: int):
        chunk: Chunk = np.zeros(shape=(self.width, self.height), dtype=BlockData)

        # Generate terrain height using fractal noise
        x_coords = np.arange(self.width)
        height_map = 0.5 * self._perlin_noise_1d(x_coords, chunk_x, octaves=5, persistence=0.5, scale=0.015)
        earth_layer = 5 + 0.1 * self._perlin_noise_1d(x_coords, chunk_x, octaves=5, persistence=0.5, scale=0.1)

        # Normalize height map to reasonable terrain range
        base_height = 256
        terrain_variation = 40
        heights = (height_map * terrain_variation + base_height - 20).astype(int)

        # Fill terrain based on height map
        for x in range(self.width):
            terrain_height = min(max(heights[x], 0), self.height - 1)

            # Stone layer (everything below terrain - 5)
            if terrain_height > 5:
                chunk[x, :terrain_height - 5] = 1

            earth = earth_layer[x]
            print(earth)

            # Dirt layer (5 blocks below surface)
            dirt_start = int(max(0, terrain_height - earth))
            chunk[x, dirt_start:terrain_height] = 2

            # Grass on top
            if terrain_height < self.height:
                chunk[x, terrain_height] = 3

            if terrain_height < base_height + 50:
                chunk[x, terrain_height:(base_height + 50)] = 4

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
        if chunk is None:
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
        region = x // self.region_size
        chunk_path = self.world_dir / str(region) / f"{x}.npy"

        if not chunk_path.exists() or not chunk_path.is_file():
            return None

        return np.load(chunk_path, allow_pickle=False)

    def save_world(self):
        with self._lock:
            chunks_to_save = list(self.chunk_cache.keys()) # copy for iteration issues

        print(f"Queueing {len(chunks_to_save)} chunks for saving")
        for chunk in chunks_to_save:
            self.write_chunk(chunk)


    def get_chunk_x(self, x: float) -> int:
        return int(x) // self.width

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


class Camera:
    x: float = 0
    y: float = 0

    # block coordinates with origin from camera
    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        return wx - self.x, wy - self.y

    # block coordinates with origin from 0 0
    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return sx + self.x, sy + self.y


class ChunkRenderer:
    chunk_manager: ChunkManager
    tile_size: int = TILE_SIZE  # FIXED: Removed duplicate declaration
    screen: pygame.Surface

    def __init__(self, chunk_manager: ChunkManager, tile_size: int, screen: pygame.Surface):
        self.chunk_manager = chunk_manager
        self.tile_size = tile_size
        self.screen = screen

    def render(self, camera: Camera):
        screen_width, screen_height = self.screen.get_size()

        # Calculate visible blocks
        blocks_x = int(screen_width / self.tile_size)
        blocks_y = int(screen_height / self.tile_size)

        # Calculate world coordinates for screen corners
        upper_right = camera.screen_to_world(blocks_x // 2 + 1, blocks_y // 2 + 1)
        lower_left = camera.screen_to_world(-blocks_x // 2 - 1, -blocks_y // 2 - 1)

        min_chunk_x = self.chunk_manager.get_chunk_x(lower_left[0])
        max_chunk_x = self.chunk_manager.get_chunk_x(upper_right[0])

        self.chunk_manager.load_chunks(range(min_chunk_x, max_chunk_x + 1))

        for chunk_x in range(min_chunk_x, max_chunk_x + 1):
            chunk = self.chunk_manager.get_chunk_from_cache(chunk_x)
            if chunk is None:
                continue

            base_x = chunk_x * self.chunk_manager.width

            # Fixed: Use int() instead of np.floor/ceil for indices
            clip_x_min = int(lower_left[0]) - base_x
            clip_x_max = int(upper_right[0]) - base_x + 1
            clip_y_min = int(max(lower_left[1], 0))
            clip_y_max = int(min(upper_right[1], self.chunk_manager.height - 1)) + 1

            range_x = range(max(0, clip_x_min), min(self.chunk_manager.width, clip_x_max))
            range_y = range(max(0, clip_y_min), min(self.chunk_manager.height, clip_y_max))

            for x in range_x:
                for y in range_y:
                    tile_idx = chunk[x, y]
                    tile = TILES.get(tile_idx)

                    if tile is not None:
                        world_x = base_x + x
                        world_y = y

                        screen_x, screen_y = camera.world_to_screen(world_x, world_y)

                        # Convert to pixel coordinates
                        pixel_x = screen_x * self.tile_size + screen_width // 2
                        pixel_y = screen_height // 2 - screen_y * self.tile_size
                        self.screen.blit(tile, (pixel_x, pixel_y))



def world_path(name: str) -> Path:
    return user_data_path("blocks") / name

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

    def update_chunk_cache(self):
        min_chunk = int(self.player_pos[0]) // self.chunk_manager.width  - 4
        max_chunk = int(self.player_pos[0]) // self.chunk_manager.width + 4  # FIXED: Changed from player_pos[1] to player_pos[0]
        self.chunk_manager.load_chunks_only(range(min_chunk, max_chunk+1))


if __name__ == "__main__":



    pygame.init()

    print("hi")
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True

    path = world_path("world-1")
    world = World(path)
    world.update_chunk_cache()

    camera = Camera()
    camera.y = 250  # Start near the ground layer

    renderer = ChunkRenderer(world.chunk_manager, TILE_SIZE, screen)

    camera_speed = 0.5

    while running:
        screen.fill(COLOR_SKY)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # Camera movement
        if keys[pygame.K_a]:
            camera.x -= camera_speed
        if keys[pygame.K_d]:
            camera.x += camera_speed
        if keys[pygame.K_w]:
            camera.y += camera_speed
        if keys[pygame.K_s]:
            camera.y -= camera_speed

        world.player_pos = (camera.x, camera.y)

        world.update_chunk_cache()


        renderer.render(camera)

        pygame.display.flip()
        clock.tick(60)

    # FIXED: Changed from chunk_manager.shutdown() to world.chunk_manager.shutdown()
    world.chunk_manager.shutdown()
    pygame.quit()
