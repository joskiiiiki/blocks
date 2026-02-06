from pathlib import Path

import pygame

from src import assets
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
            x=2, y=260, world=self.world, screen=self.screen, delta_t=1 / self.framerate
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.Font(None, FONT_SIZE)

    def main(self):
        self.world.update_chunk_cache()
        self.running = True
        while self.running:
            self.screen.fill(assets.COLOR_SKY)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEWHEEL:
                    self.player.handle_mousewheel(event)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                self.running = False

            self.player.update()
            self.world.player_pos = self.player.xy

            self.world.update_chunk_cache()

            self.chunk_render.render(self.player.x, self.player.y)

            self.player.draw()

            fps = self.clock.get_fps()
            fps_text = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
            self.screen.blit(fps_text, (10, 10))
            coords_text = self.font.render(
                f"X={self.player.x:.10f} Y={self.player.y:.10f}", True, (255, 255, 255)
            )
            self.screen.blit(coords_text, (10, 10 + FONT_SIZE * 1.1))

            chunk_coords = self.world.chunk_manager.world_to_chunk(
                self.player.x, self.player.y
            )
            if chunk_coords is not None:
                chunk_text = self.font.render(
                    f"Chunk: {chunk_coords[0]}", True, (255, 255, 255)
                )
                self.screen.blit(chunk_text, (10, 10 + FONT_SIZE * 2.2))

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
