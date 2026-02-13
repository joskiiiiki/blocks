from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pygame

from src.blocks import Block, is_solid
from src.world.chunk import CHUNK_HEIGHT, CHUNK_WIDTH
from src.world.chunk_manager import ChunkManager
from src.world.gen_context import WorldGenContext

MARKER = pygame.Surface((8, 8))
MARKER.fill((255, 0, 0))


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
    on_block_changed: Callable[[int, int], None] | None

    def __init__(
        self,
        path: pathlib.Path,
        on_block_changed: Callable[[int, int], None] | None = None,
    ):
        self.world_path = path

        exists = path.exists() and path.is_dir()

        self.on_block_changed = on_block_changed

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
            seed=np.random.randint(
                0, 2**16 - 1
            ),  # 16 bit - C-Integer size for our noise generator TODO: wrap that shit myself this is so annoying
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
        if self.on_block_changed is not None:
            self.on_block_changed(int(x), int(y))
        return self.chunk_manager.set_block(x, y, block.value)

    def destroy_block(self, x: float, y: float) -> Block | None:
        if self.on_block_changed is not None:
            self.on_block_changed(int(x), int(y))
        return Block(self.chunk_manager.destroy_block(x, y))

    def is_solid(self, x: float, y: float) -> bool:
        block = self.chunk_manager.get_block(x, y)
        if block is None:
            return False
        return is_solid(block)

    def world_to_chunk(self, x: float, y: float) -> tuple[int, float, float] | None:
        return self.chunk_manager._world_to_chunk(x, y)
