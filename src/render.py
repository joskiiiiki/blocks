import pygame

from src import assets, blocks
from src.blocks import Block
from src.utils import camera_to_world, world_to_screen
from src.world import ChunkManager

PX = pygame.Surface((assets.TILE_SIZE // 16, assets.TILE_SIZE // 16))
PX.fill((255, 255, 0))


class ChunkRenderer:
    chunk_manager: ChunkManager
    tile_size: int = assets.TILE_SIZE  # FIXED: Removed duplicate declaration
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

        self.chunk_manager.load_chunks_only(range(min_chunk_x, max_chunk_x + 1))

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
                    tile_idx = chunk.blocks[x, y]
                    tile = Block.get_texture_from_id(tile_idx)

                    if (
                        tile_idx == Block.WATER.value
                        and chunk.blocks[x, y + 1] == Block.AIR.value
                    ):
                        tile = assets.TEXTURES.get_texture("water_top")

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
                            assets.TILE_SIZE,
                        )

                        self.screen.blit(tile, (pixel_x, pixel_y - assets.TILE_SIZE))

                        # self.screen.blit(PX, (pixel_x, pixel_y - PX.height))

            #             yt = self.font.render(f"{y}", False, (255, 255, 255))
            #             xt = self.font.render(f"{x}", False, (255, 255, 255))

            #             self.screen.blit(yt, (pixel_x, pixel_y - 14))
            #             self.screen.blit(xt, (pixel_x, pixel_y - 14 - 16))

            # self.screen.blit(PX, (0, 0))
