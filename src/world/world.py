from __future__ import annotations

import atexit
import json
import os
import pathlib
import random
import signal
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pygame

from src.blocks import Block, is_solid
from src.emitter import Emitter
from src.entity.mob import Mob
from src.entity.zombie import Zombie
from src.interfaces import IPlayer, IWorld
from src.physics import get_touching_blocks, physics_step
from src.player import Player
from src.sounds import SoundManager
from src.world.chunk import CHUNK_HEIGHT, CHUNK_WIDTH
from src.world.chunk_manager import ChunkManager
from src.world.gen_context import WorldGenContext

MARKER = pygame.Surface((8, 8))
MARKER.fill((255, 0, 0))

# tune these
MAX_MOBS = 20
SPAWN_RANGE_X = 20  # world units from player
SPAWN_RANGE_MIN = 8  # don't spawn too close
SPAWN_ATTEMPTS = 10
SPAWN_INTERVAL = 3.0  # seconds between spawn ticks


def _setup_mob_sounds(mob: "Mob", sound_manager: SoundManager, player: Player) -> None:
    mob.on(
        "damage", lambda *_: sound_manager.play_at("zombie_damage", mob.xy, player.xy)
    )


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


class World(IWorld, Emitter):
    chunk_manager: ChunkManager
    world_path: pathlib.Path
    lock_path: pathlib.Path
    player_pos: tuple[float, float] = (0, 0)
    world_data: WorldData
    on_block_changed: Callable[[int, int], None] | None
    _spawn_timer: float = 0.0

    def __init__(
        self,
        path: pathlib.Path,
        on_block_changed: Callable[[int, int], None] | None = None,
    ):
        self.world_path = path
        self.lock_path = self.world_path / ".lock"

        exists = path.exists() and path.is_dir()

        self.on_block_changed = on_block_changed

        if not exists:
            self.world_path.mkdir(parents=True, exist_ok=True)
            self.world_data = self.new_world_data()
            world_data_path = path / "world.json"
            world_data_path.touch(exist_ok=True)
            self.world_data.save(world_data_path)
        elif not self.acquire():
            raise Exception(f"Could not aquire lock on {self.world_path}")
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

        Emitter.__init__(self)

    def release(self):
        if not self.lock_path.exists():
            return

        try:
            self.lock_path.unlink(missing_ok=False)
            print(f"Released lock on {self.world_path}")
        except Exception as e:
            print(f"Failed to release lock on {self.world_path}: {e}")

    def acquire(self) -> bool:
        if self.lock_path.exists():
            print(f"World {self.lock_path} is locked by another process")
            return False

        try:
            self.lock_path.touch(exist_ok=False)
            print(f"Lock aquired for {self.world_path}")

            atexit.register(self.release)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            return True
        except Exception as e:
            print(f"Failed to aquire lock on {self.world_path}: {e}")
            return False

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, cleaning up...")
        self.release()
        sys.exit(0)

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
        was_set = self.chunk_manager.set_block(x, y, block.value)

        if was_set:
            self.emit("block_set", x, y, block)
        return was_set

    def destroy_block(self, x: int, y: int) -> Block | None:
        if self.on_block_changed is not None:
            self.on_block_changed(x, y)

        block = Block(self.chunk_manager.destroy_block(x, y))
        if block is not None:
            self.emit("block_destroyed", x, y, block)
        return block

    def is_solid(self, x: float, y: float) -> bool:
        block = self.chunk_manager.get_block(x, y)
        if block is None:
            return False
        return is_solid(block)

    def world_to_chunk(self, x: float, y: float) -> tuple[int, float, float] | None:
        return self.chunk_manager._world_to_chunk(x, y)

    def update_mobs(
        self, player: Player, sound_manager: SoundManager, dt: float
    ) -> None:
        self.chunk_manager.entities = [
            e for e in self.chunk_manager.entities if not e.is_dead
        ]  # filter out dead entities
        self._spawn_timer -= dt / 1000.0
        if self._spawn_timer <= 0:
            self._spawn_timer = SPAWN_INTERVAL
            self._try_spawn(player)

        for mob in self.chunk_manager.entities:
            in_water = Block.WATER.value in get_touching_blocks(mob, self)
            if isinstance(mob, Mob):
                mob.update_focus(player.xy, dt, in_water)
            physics_step(mob, self, dt)

            sound_manager.update_mob_walk(mob, player.xy)

    def _try_spawn(self, player) -> None:
        if len(self.chunk_manager.entities) >= MAX_MOBS:
            return

        for _ in range(SPAWN_ATTEMPTS):
            # pick x outside the minimum range but within spawn range
            side = random.choice([-1, 1])
            x = player.x + side * random.uniform(SPAWN_RANGE_MIN, SPAWN_RANGE_X)
            chunk_x = int(x) // self.chunk_manager.width

            # only spawn in loaded chunks
            chunk = self.chunk_manager.get_chunk_from_cache(chunk_x)
            if chunk is None:
                continue

            local_x = int(x) % self.chunk_manager.width

            # find surface y
            surface_y = None
            for y in range(self.chunk_manager.height - 2, 0, -1):
                if (
                    chunk.blocks[local_x, y] != Block.AIR.value
                    and chunk.blocks[local_x, y + 1] == Block.AIR.value
                ):
                    surface_y = y + 1
                    break

            if surface_y is None:
                continue

            mob = Zombie(x=float(int(x)) + 0.5, y=float(surface_y))
            self.chunk_manager.add_entity(mob)
            return  # one mob per tick
