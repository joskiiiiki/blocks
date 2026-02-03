import pygame

from src.blocks import Block
from src.utils import camera_to_world, world_to_screen
from src.world import ChunkManager

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

PX = pygame.Surface((TILE_SIZE // 16, TILE_SIZE // 16))
PX.fill(pygame.Color(255, 0, 0))

TILES = {
    Block.AIR.value: None,
    Block.STONE.value: STONE_TILE,
    Block.DIRT.value: DIRT_TILE,
    Block.GRASS.value: GRASS_TILE,
    Block.WATER.value: WATER_TILE,
}


class ChunkRenderer:
    chunk_manager: ChunkManager
    tile_size: int = TILE_SIZE  # FIXED: Removed duplicate declaration
    screen: pygame.Surface
    font: pygame.Font

    def __init__(
        self, chunk_manager: ChunkManager, tile_size: int, screen: pygame.Surface
    ):
        self.chunk_manager = chunk_manager
        self.tile_size = tile_size
        self.screen = screen
        self.font = pygame.Font(None, 16)

    def render(self, cam_x: float, cam_y: float):
        screen_width, screen_height = self.screen.get_size()

        # Calculate visible blocks
        blocks_x = int(screen_width / self.tile_size)
        blocks_y = int(screen_height / self.tile_size)

        # Calculate world coordinates for screen corners
        upper_right = camera_to_world(
            cam_x, cam_y, blocks_x // 2 + 1, blocks_y // 2 + 1
        )
        lower_left = camera_to_world(
            cam_x, cam_y, -blocks_x // 2 - 1, -blocks_y // 2 - 1
        )

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

            range_x = range(
                max(0, clip_x_min), min(self.chunk_manager.width, clip_x_max)
            )
            range_y = range(
                max(0, clip_y_min), min(self.chunk_manager.height, clip_y_max)
            )

            for x in range_x:
                for y in range_y:
                    tile_idx = chunk[x, y]
                    tile = TILES.get(tile_idx)

                    if tile is not None:
                        world_x = base_x + x
                        world_y = y

                        pixel_x, pixel_y = world_to_screen(
                            cam_x,
                            cam_y,
                            world_x,
                            world_y,
                            screen_width,
                            screen_height,
                            TILE_SIZE,
                        )

                        self.screen.blit(tile, (pixel_x, pixel_y - TILE_SIZE))
                        self.screen.blit(PX, (pixel_x, pixel_y - TILE_SIZE // 16))

            #             yt = self.font.render(f"{y}", False, (255, 255, 255))
            #             xt = self.font.render(f"{x}", False, (255, 255, 255))

            #             self.screen.blit(yt, (pixel_x, pixel_y - 14))
            #             self.screen.blit(xt, (pixel_x, pixel_y - 14 - 16))

            # self.screen.blit(PX, (0, 0))
