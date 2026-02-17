from pathlib import Path

import moderngl
import pygame

from src import assets
from src.player import Player
from src.render_gl import ChunkRendererGL
from src.world import World, world_path

FONT_SIZE = 24


class Game:
    framerate = 60
    world: World
    chunk_render: ChunkRendererGL
    tile_size: int = 32
    _surface: pygame.Surface  # actual screen surface under opengl - dont blit to this
    screen: pygame.Surface  # surface to blit to - should be used except you know better
    player: Player
    clock: pygame.time.Clock
    running: bool = False
    font: pygame.Font
    # lighting_manager: LightingManagerGL
    gl_ctx: moderngl.Context

    def __init__(self, world_path: Path):
        pygame.init()
        self.world = World(world_path, self.on_block_changed)
        self._surface = pygame.display.set_mode(
            (1280, 720), flags=pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.OPENGL
        )
        self.gl_ctx = moderngl.create_context()
        # self.lighting_manager = LightingManagerGL(self.world.chunk_manager, self.gl_ctx)
        self.chunk_render = ChunkRendererGL(
            ctx=self.gl_ctx,
            chunk_manager=self.world.chunk_manager,
            tile_size=self.tile_size,
            screen=self._surface,  # draw to opengl surface directly since we re using gl
            # self.lighting_manager,
        )
        self.player = Player(
            x=2,
            y=265,
            world=self.world,
            screen=self._surface,  # draw onto another surface since were using blit here for simplicity
            delta_t=1 / self.framerate,
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.Font(None, FONT_SIZE)

    def main(self):
        assets.TEXTURES.load()
        self.world.update_chunk_cache()
        self.running = True
        delta_t = 1 / self.framerate
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

            self.player.update(delta_t)

            self.world.player_pos = self.player.xy

            self.world.update_chunk_cache()

            self.chunk_render.render(self.player.xy)

            self.player.draw()

            fps = self.clock.get_fps()
            fps_text = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
            self.screen.blit(fps_text, (10, 10))
            coords_text = self.font.render(
                f"X={self.player.x:.10f} Y={self.player.y:.10f}", True, (255, 255, 255)
            )
            self.screen.blit(coords_text, (10, 10 + FONT_SIZE * 1.1))

            chunk_coords = self.world.world_to_chunk(self.player.x, self.player.y)
            if chunk_coords is not None:
                chunk_text = self.font.render(
                    f"Chunk: {chunk_coords[0]}", True, (255, 255, 255)
                )
                self.screen.blit(chunk_text, (10, 10 + FONT_SIZE * 2.2))

            pygame.display.flip()
            delta_t = self.clock.tick(self.framerate) / 1000  # convert to seconds

    def on_block_changed(self, world_x: int, world_y: int):
        chunk_x = self.world.chunk_manager.get_chunk_x(world_x)

        # Mark affected chunks dirty (include neighbors for light propagation)
        # self.lighting_manager.mark_chunks_dirty([chunk_x - 1, chunk_x, chunk_x + 1])

        # Mark renderer dirty
        # self.chunk_render.mark_lighting_dirty()

    def on_exit(self):
        self.world.chunk_manager.shutdown()
        pygame.quit()


if __name__ == "__main__":
    path = world_path("world-1")
    game = Game(path)

    game.main()

    game.on_exit()
