import render
from player import Player
from render import ChunkRenderer
from world import World, world_path
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

class Game:
    framerate = 60
    world: World
    chunk_render: ChunkRenderer
    tile_size: int = 32
    screen: pygame.Surface
    player: Player
    clock: pygame.time.Clock
    running: bool = False

    def __init__(self, world_path: Path):
        pygame.init()
        self.world = World(world_path)
        self.screen = pygame.display.set_mode((1280, 720))
        self.chunk_render = ChunkRenderer(self.world.chunk_manager, self.tile_size, self.screen)
        self.player = Player(x=0, y=512)
        self.clock = pygame.time.Clock()

    def main(self):
        self.world.update_chunk_cache()
        self.running = True
        while self.running:
            self.screen.fill(render.COLOR_SKY)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False

            new_x, new_y = self.player.update()
            # self.world.chunk_manager.check_collision(new_x, new_y)



            print(f"x:{self.player.x} y:{self.player.y}")

            self.world.player_pos = (self.player.x, self.player.y)

            self.world.update_chunk_cache()

            self.chunk_render.render(self.player.x, self.player.y)

            self.player.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(self.framerate)

    def on_exit(self):
        self.world.chunk_manager.shutdown()
        pygame.quit()









if __name__ == "__main__":
    path = world_path("world-1")
    game = Game(path)

    game.main()

    game.on_exit()
