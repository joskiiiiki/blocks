from pathlib import Path

import moderngl
import pygame

from src import assets
from src.player import Player
from src.render import ChunkRendererGL, PygameOverlay
from src.render.lighting import LightingManagerGL
from src.world import World, world_path

FONT_SIZE = 24


class Game:
    framerate = 60
    world: World
    chunk_render: ChunkRendererGL
    lighting_manager: LightingManagerGL
    resolution: tuple[int, int] = (1280, 720)
    tile_size: int = 32
    _screen: pygame.Surface  # actual screen surface under opengl - dont blit to this
    overlay: PygameOverlay
    player: Player
    clock: pygame.time.Clock
    running: bool = False
    font: pygame.Font
    # lighting_manager: LightingManagerGL
    ctx: moderngl.Context

    def __init__(self, world_path: Path):
        pygame.init()
        self.world = World(world_path, self.on_block_changed)
        self._screen = pygame.display.set_mode(
            self.resolution, flags=pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.OPENGL
        )
        assets.TEXTURES.load()
        self.ctx = moderngl.create_context()
        # self.lighting_manager = LightingManagerGL(self.world.chunk_manager, self.gl_ctx)
        self.lighting_manager = LightingManagerGL(self.world.chunk_manager, self.ctx)
        self.chunk_render = ChunkRendererGL(
            ctx=self.ctx,
            chunk_manager=self.world.chunk_manager,
            tile_size=self.tile_size,
            screen=self._screen,  # draw to opengl surface directly since we re using gl
            lighting_manager=self.lighting_manager,
        )
        self.overlay = PygameOverlay(self.ctx, self.resolution)

        self.player = Player(
            x=2,
            y=265,
            world=self.world,
            delta_t=1 / self.framerate,
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.Font(None, FONT_SIZE)

    def main(self):
        self.world.update_chunk_cache()
        self.running = True
        delta_t = 1 / self.framerate
        while self.running:
            self._screen.fill(assets.COLOR_SKY)
            self.overlay.clear()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEWHEEL:
                    self.player.handle_mousewheel(event)
                if event.type == pygame.VIDEORESIZE:
                    self.on_resize(event.size)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                self.running = False

            self.player.update(delta_t, self.resolution)

            self.world.player_pos = self.player.xy

            self.world.update_chunk_cache()

            self.chunk_render.render(self.player.xy, self.resolution)

            self.player.draw(self.overlay.surface, self.resolution)

            fps = self.clock.get_fps()
            fps_text = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
            self.overlay.blit(fps_text, (10, 10))
            coords_text = self.font.render(
                f"X={self.player.x:.10f} Y={self.player.y:.10f}", True, (255, 255, 255)
            )
            self.overlay.blit(coords_text, (10, 10 + FONT_SIZE * 1.1))

            chunk_coords = self.world.world_to_chunk(self.player.x, self.player.y)
            if chunk_coords is not None:
                chunk_text = self.font.render(
                    f"Chunk: {chunk_coords[0]}", True, (255, 255, 255)
                )
                self.overlay.blit(chunk_text, (10, 10 + FONT_SIZE * 2.2))

            self.overlay.render()

            pygame.display.flip()
            delta_t = self.clock.tick(self.framerate) / 1000  # convert to seconds

    def on_resize(self, resolution: tuple[int, int]):
        self.overlay.on_resize(resolution)
        self.resolution = resolution
        print(f"Resized: {resolution[0]}x{resolution[1]}")

    def on_block_changed(self, world_x: int, world_y: int):
        chunk_x = self.world.chunk_manager.get_chunk_x(world_x)

        self.lighting_manager.mark_chunks_dirty([chunk_x - 1, chunk_x, chunk_x + 1])

        # Mark renderer dirty
        self.chunk_render.mark_lighting_dirty()

    def on_exit(self):
        self.world.chunk_manager.shutdown()
        self.overlay.on_destroy()  # release buffers
        pygame.quit()


if __name__ == "__main__":
    path = world_path("world-1")
    game = Game(path)

    game.main()

    game.on_exit()
