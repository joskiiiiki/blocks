from pathlib import Path

import pygame

from src import render
from src.player import Player
from src.render import ChunkRenderer
from src.world import World, world_path

FONT_SIZE = 24


class Game:
    framerate = 60
    world: World
    chunk_render: ChunkRenderer
    tile_size: int = 32
    screen: pygame.Surface
    player: Player
    clock: pygame.time.Clock
    running: bool = False
    font: pygame.Font

    def __init__(self, world_path: Path):
        pygame.init()
        self.world = World(world_path)
        self.screen = pygame.display.set_mode(
            (1280, 720), flags=pygame.SRCALPHA | pygame.RESIZABLE
        )
        self.chunk_render = ChunkRenderer(
            self.world.chunk_manager, self.tile_size, self.screen
        )
        self.player = Player(
            x=0, y=300, world=self.world, screen=self.screen, delta_t=1 / self.framerate
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.Font(None, FONT_SIZE)

    def main(self):
        self.world.update_chunk_cache()
        self.running = True
        while self.running:
            self.screen.fill(render.COLOR_SKY)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                self.running = False

            new_x, new_y = self.player.update()
            intersect = self.world.chunk_manager.intersect(
                self.player.x, self.player.y, new_x, new_y
            )
            if intersect:
                self.player.handle_intersection(intersect)
            else:
                self.player.set_position(new_x, new_y)

            self.world.player_pos = (self.player.x, self.player.y)

            self.world.update_chunk_cache()

            self.chunk_render.render(self.player.x, self.player.y)

            self.player.draw()

            fps = self.clock.get_fps()
            fps_text = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
            self.screen.blit(fps_text, (10, 10))
            coords_text = self.font.render(
                f"X={self.player.x:.1f} Y={self.player.y:.1f}", True, (255, 255, 255)
            )
            self.screen.blit(coords_text, (10, 10 + FONT_SIZE * 1.1))
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
