from pathlib import Path

import moderngl
import pygame

from src.blocks import Block
from src.combat import process_combat
from src.entity.mob import Mob
from src.entity.zombie import Zombie
from src.inventory_renderer import InventoryRenderer
from src.physics import get_touching_blocks, physics_step
from src.player import HIT_FLASH_DURATION, Player
from src.recipes import craft
from src.render import ChunkRendererGL, PygameOverlay
from src.render.damage import DamageOverlay
from src.render.lighting import LightingManagerGL
from src.sounds import SoundManager
from src.world import World, world_path
from src.world.world import _setup_mob_sounds

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
    font: pygame.font.Font
    # lighting_manager: LightingManagerGL
    ctx: moderngl.Context
    damage_overlay: DamageOverlay
    inventory_renderer: InventoryRenderer
    inventory_open = False

    @property
    def world_interactions_blocked(self):
        return self.inventory_open

    def __init__(self, world_path: Path):
        pygame.init()
        world = World(world_path, self.on_block_changed)
        if not world:
            return

        self.clock = pygame.Clock()
        self.world = world

        self._screen = pygame.display.set_mode(
            self.resolution, flags=pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.OPENGL
        )
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
            delta_t=1 / self.framerate,
        )
        self.sound_manager = SoundManager()
        self.player.on("damage", lambda *_: self.sound_manager.play("player_damage"))
        self.player.on("start_walking", lambda *_: self.sound_manager.play_walk(True))
        self.player.on("stop_walking", lambda *_: self.sound_manager.play_walk(False))
        self.world.on(
            "block_destroyed",
            lambda *_: self.sound_manager.play_non_overlapping("break_single"),
        )

        # after world and player init:
        self.world.load_player(self.player)

        self.font = pygame.Font(None, FONT_SIZE)

        self.damage_overlay = DamageOverlay(self.ctx, HIT_FLASH_DURATION)
        self.inventory_renderer = InventoryRenderer(recipe=craft)

        self.world.chunk_manager.on(
            "entity_added",
            lambda entity, *_: _setup_mob_sounds(
                entity, self.sound_manager, self.player
            ),
        )

    def main(self):
        self.world.update_chunk_cache()
        self.running = True
        dt = 1 / self.framerate

        # self.sound_manager.play_music()

        while self.running:
            # self._screen.fill(assets.COLOR_SKY)
            self.overlay.clear()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEWHEEL:
                    self.player.handle_mousewheel(event)
                if event.type == pygame.VIDEORESIZE:
                    self.on_resize(event.size)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                    self.inventory_open = not self.inventory_open
                    if not self.inventory_open:
                        self.inventory_renderer.close(self.player.inventory.slots)

            if self.inventory_open:
                self.inventory_renderer.handle_event(
                    event, self.player.inventory.slots, self.resolution
                )

            # replace the player update + enemy block with:
            in_water = Block.WATER.value in get_touching_blocks(self.player, self.world)
            self.player.update_player(
                dt,
                self.resolution,
                in_water,
                self.world,
                self.world_interactions_blocked,
            )
            physics_step(self.player, self.world, dt)

            self.world.update_mobs(self.player, self.sound_manager, dt)

            mobs = filter(
                lambda e: isinstance(e, Mob), self.world.chunk_manager.entities
            )
            combat_results = process_combat(
                self.player,
                mobs,
                dt,
            )

            self.world.player_pos = self.player.xy
            self.world.update_chunk_cache()

            self.chunk_render.render(self.player.xy, self.resolution)
            self.player.draw(self.overlay.surface, self.resolution)

            for mob in self.world.chunk_manager.entities:
                if isinstance(mob, Zombie):
                    mob.draw(
                        self.overlay.surface,
                        self.resolution,
                        *self.player.xy,
                    )
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

            if self.inventory_open:
                self.inventory_renderer.draw(
                    self.overlay.surface,
                    self.player.inventory.slots,
                    self.resolution,
                )
            self.overlay.render()

            self.damage_overlay.render(
                self.player.hit_flash_timer,
                self.resolution,
            )

            pygame.display.flip()
            dt = self.clock.tick(self.framerate) / 1000  # convert to seconds

    def on_resize(self, resolution: tuple[int, int]):
        self.overlay.on_resize(resolution)
        self.resolution = resolution
        print(f"Resized: {resolution[0]}x{resolution[1]}")

    def on_block_changed(self, world_x: int, world_y: int):
        chunk_x = self.world.chunk_manager.get_chunk_x(world_x)

        self.lighting_manager.mark_chunks_dirty([chunk_x - 1, chunk_x, chunk_x + 1])
        self.chunk_render.mark_chunk_dirty(chunk_x)

        # Mark renderer dirty
        self.chunk_render.mark_lighting_dirty()

    def on_exit(self):
        self.world.chunk_manager.shutdown()
        self.overlay.on_destroy()  # release buffers
        self.damage_overlay.destroy()

        self.world.save_player(self.player)
        pygame.quit()


if __name__ == "__main__":
    path = world_path("world-1")
    game = Game(path)

    game.main()

    game.on_exit()
