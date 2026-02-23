# render_gl.py - ModernGL GPU-Accelerated Renderer

"""
ModernGL-based chunk renderer with GPU lighting.
Performance improvements:
- 10-100x faster than CPU rendering
- Batched instanced rendering (thousands of tiles in 1 draw call)
- GPU-based lighting calculations
- Texture atlas for efficient texture switching
"""

import math
from typing import TypeAlias

import moderngl
import numpy as np
import numpy.typing as npt
import pygame

from src import assets, shaders
from src.blocks import BLOCK_ID_MASK, Block, BlockData
from src.render.lighting import LightingManagerGL
from src.render.texture_atlas import TextureAtlas

# from src.shaders import RENDER_FRAGMENT_SHADER, RENDER_VERTEX_SHADER
from src.utils import camera_to_world
from src.world import ChunkManager

Instances: TypeAlias = npt.NDArray[np.float32]


class ChunkRendererGL:
    """GPU-accelerated chunk renderer using ModernGL"""

    ctx: moderngl.Context
    chunk_manager: ChunkManager
    tile_size: int
    max_instances: int = 100_000
    program: moderngl.Program
    instance_vbo: moderngl.Buffer
    instance_vao: moderngl.VertexArray
    vao: moderngl.VertexArray

    atlas: TextureAtlas

    last_screen_size: tuple[int, int] | None = None
    cached_projection: bytes

    lighting_manager: LightingManagerGL
    lighting_dirty: bool = False
    last_lit_chunks: set[int] = set()

    def __init__(
        self,
        ctx: moderngl.Context,
        chunk_manager: ChunkManager,
        tile_size: int,
        screen: pygame.Surface,
        lighting_manager: LightingManagerGL,
    ):
        self.ctx = ctx
        self.chunk_manager = chunk_manager
        self.tile_size = tile_size
        self.screen = screen

        self.program = self.ctx.program(
            vertex_shader=shaders.RENDER_VERTEX_SHADER,
            fragment_shader=shaders.RENDER_FRAGMENT_SHADER,
        )

        print("shader loaded with attributes:")
        for name in self.program:
            print(f"\t{name}")

        self.atlas = TextureAtlas(self.ctx, self.tile_size)
        self.atlas.build()
        tex_size = self.atlas.tile_size_normalized()

        quad_vertices = np.array(
            [
                # x, y, u, v
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, tex_size, 0.0],
                [1.0, 1.0, tex_size, tex_size],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, tex_size, tex_size],
                [0.0, 1.0, 0.0, tex_size],
            ],
            dtype="f4",
        )
        quad_vbo = self.ctx.buffer(quad_vertices.tobytes())
        self.instance_vbo = self.ctx.buffer(
            reserve=self.max_instances
            * 7
            * 4  # vec2 for position + vec2 for atlas offset + vec3 for lighting
        )
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (quad_vbo, "2f 2f", "in_position", "in_uv"),
                (
                    self.instance_vbo,
                    "2f 2f 3f/i",
                    "in_world_pos",
                    "in_atlas_offset",
                    "in_light",
                ),
            ],
        )

        self.lighting_manager = lighting_manager
        self._init_air()

    
    def _init_air(self):
        """Initialize light debug rendering"""
        print("Initializing light debug renderer...")
        
        # Compile debug shaders
        self.air_program = self.ctx.program(
            vertex_shader=shaders.AIR_VERTEX_SHADER,
            fragment_shader=shaders.AIR_FRAGMENT_SHADER
        )
        
        # Create full-screen quad
        quad_vertices = np.array([
            # pos_x, pos_y, u, v
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ], dtype='f4')
        
        quad_vbo = self.ctx.buffer(quad_vertices.tobytes())
        
        self.air_vao = self.ctx.vertex_array(
            self.air_program,
            [(quad_vbo, '2f 2f', 'in_position', 'in_uv')],
        )
    def update_projection(self, screen_width: int, screen_height: int):
        if self.last_screen_size == (screen_width, screen_height):
            return

        projection = np.array(
            [
                [2.0 / screen_width, 0, 0, -1],
                [0, -2.0 / screen_height, 0, 1],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype="f4",
        )
        self.cached_projection = projection.T.tobytes()
        self.last_screen_size = (screen_width, screen_height)
        print(f"Updated projection for {screen_width}x{screen_height}")

    def build_instances(
        self,
        min_chunk_x: int,
        max_chunk_x: int,
        lower_left: tuple[float, float],
        upper_right: tuple[float, float],
    ) -> Instances:
        # Build instances for rendering chunks
        instances: list[
            tuple[
                float,  # world_x
                float,  # world_y
                float,  # atlas_u
                float,  # atlas_v
                float,  # light_r
                float,  # light_g
                float,  # light_b
            ]
        ] = []

        for chunk_x in range(min_chunk_x, max_chunk_x + 1):
            chunk = self.chunk_manager.get_chunk_from_cache(chunk_x)

            if chunk is None:
                continue

            lightmap = self.lighting_manager.get_lightmap(chunk_x)

            if lightmap is None:
                print(f"WARNING: No lightmap for chunk {chunk_x}!")
                continue

            # DEBUG: Check first light value
            base_x = chunk_x * self.chunk_manager.width

            clip_x_min = math.floor(lower_left[0]) - base_x
            clip_x_max = math.floor(upper_right[0]) - base_x + 1
            clip_y_min = math.floor(max(lower_left[1], 0))
            clip_y_max = (
                math.floor(min(upper_right[1], self.chunk_manager.height - 1)) + 1
            )

            range_x = range(
                max(0, clip_x_min), min(self.chunk_manager.width, clip_x_max)
            )
            range_y = range(
                max(0, clip_y_min), min(self.chunk_manager.height, clip_y_max)
            )

            for x in range_x:
                for y in range_y:
                    block: BlockData = chunk.blocks[x, y]
                    block_id = int(block) & BLOCK_ID_MASK
                    if block_id == Block.AIR.value:
                        continue

                    world_x = base_x + x
                    world_y = y

                    match block_id:
                        case Block.WATER.value:
                            if y == self.chunk_manager.height - 1:
                                atlas_u, atlas_v = self.atlas.uv(
                                    Block.WATER.with_data((0, 1))
                                )
                            elif (
                                chunk.blocks[x, y + 1] & BLOCK_ID_MASK
                                != Block.WATER.value
                            ):
                                atlas_u, atlas_v = self.atlas.uv(
                                    Block.WATER.with_data((0, 1))
                                )
                            else:
                                atlas_u, atlas_v = self.atlas.uv(Block.WATER.value)
                        case _:
                            atlas_u, atlas_v = self.atlas.uv(block)

                    light_r = lightmap[x, y, 0]
                    light_g = lightmap[x, y, 1]
                    light_b = lightmap[x, y, 2]

                    # Add block instance to instances list
                    instances.append(
                        (world_x, world_y, atlas_u, atlas_v, light_r, light_g, light_b)
                    )

        instance_arr = np.array(instances, dtype=np.float32)

        return instance_arr

    def render(self, camera_pos: tuple[float, float], resolution: tuple[int, int]):
        """Render chunks using GPU-accelerated chunk renderer"""
        cam_x, cam_y = camera_pos

        screen_width = resolution[0]
        screen_height = resolution[1]

        blocks_x = int(screen_width / self.tile_size)
        blocks_y = int(screen_height / self.tile_size)
        half_span_x = blocks_x // 2 + 1
        half_span_y = blocks_y // 2 + 1

        upper_right = camera_to_world(cam_x, cam_y, half_span_x, half_span_y)
        lower_left = camera_to_world(cam_x, cam_y, -half_span_x, -half_span_y)

        min_chunk_x = self.chunk_manager.get_chunk_x(lower_left[0])
        max_chunk_x = self.chunk_manager.get_chunk_x(upper_right[0])

        to_update = range(min_chunk_x, max_chunk_x + 1)
        self.chunk_manager.load_chunks_only(to_update)
        current_chunks = set(to_update)

        if self._should_update_lighting(current_chunks):
            self.lighting_manager.calculate_lighting_region(
                min_chunk_x, max_chunk_x, iterations=16
            )
            self.last_lit_chunks = current_chunks
            self.lighting_dirty = False

        instances = self.build_instances(
            min_chunk_x, max_chunk_x, lower_left, upper_right
        )

        if len(instances) == 0:
            self.ctx.clear(*assets.COLOR_SKY.normalized)
            print("No instances to render")
            return

        self.ctx.viewport = (0, 0, screen_width, screen_height)
        self.ctx.clear(*assets.COLOR_SKY.normalized)
        
        # self._render_air(screen_width, screen_height, camera_pos, min_chunk_x, max_chunk_x)
        self._render_gpu(instances, screen_width, screen_height, camera_pos)

    def _render_gpu(
        self,
        instances: npt.NDArray[np.float32],
        screen_width: int,
        screen_height: int,
        camera_pos: tuple[float, float],
    ):
        num_instances = len(instances)


        self.instance_vbo.write(instances.tobytes())

        self.update_projection(screen_width, screen_height)

        # Set uniforms
        self.program["projection"].write(self.cached_projection)
        self.program["screen_size"] = (float(screen_width), float(screen_height))
        self.program["camera_pos"] = camera_pos
        self.program["tile_size"] = float(self.tile_size)
        self.program["texture_atlas"] = 0  # bind to zero

        self.atlas.texture.use(0)

        # Clear and render
        self.ctx.enable(moderngl.BLEND)

        self.vao.render(moderngl.TRIANGLES, instances=num_instances)
        self.ctx.disable(moderngl.BLEND)

    def _render_air(
        self,
        screen_width: int,
        screen_height: int,
        camera_pos: tuple[float, float],
        min_chunk_x: int,
        max_chunk_x: int,
    ):
        """Render light map as a debug overlay"""
        # Get combined lightmap for visible chunks
        combined_lightmaps = []
        for chunk_x in range(min_chunk_x, max_chunk_x + 1):
            lightmap = self.lighting_manager.get_lightmap(chunk_x)
            if lightmap is not None:
                combined_lightmaps.append(lightmap)
        
        if not combined_lightmaps:
            return
        
        # Combine lightmaps horizontally
        combined: npt.NDArray = np.concatenate(combined_lightmaps, axis=0)
        print(combined)
        width, height = combined.shape[0], combined.shape[1]
        
        # Add alpha channel (set to 1.0)
        light_rgba = np.ones((width, height, 4), dtype=np.float32)
        # light_rgba[:, :, :3] = combined
        
        # # Flip for OpenGL
        # light_rgba = np.flip(light_rgba, axis=1).copy()
        # light_rgba = 
        light_rgba[:, 0:width // 4 * 4, :] = 0
        
        # Create temporary texture
        light_texture = self.ctx.texture(
            (width, height),
            4,
            data=light_rgba.tobytes(),
            dtype='f4'
        )
        light_texture.repeat_x = False
        light_texture.repeat_y = False
        light_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        
        # Bind texture
        light_texture.use(0)
        
        # Set uniforms
        cam_x, cam_y = camera_pos
        world_offset_x = min_chunk_x * self.chunk_manager.width
        
        self.air_program['light_texture'] = 0
        # self.air_program['screen_size'] = (float(screen_width), float(screen_height))
        # self.air_program['camera_pos'] = (cam_x, cam_y)
        # self.air_program['tile_size'] = float(self.tile_size)
        # self.air_program['world_offset'] = (float(world_offset_x), 0.0)
        self.air_program['light_map_size'] = (float(width), float(height))
        
        # Enable blending for overlay
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Render full-screen quad
        self.air_vao.render(moderngl.TRIANGLES)

        self.ctx.disable(moderngl.BLEND)
        
        # Cleanup
        light_texture.release()

    def _should_update_lighting(self, current_chunks: set[int]) -> bool:
        return self.lighting_dirty or self.last_lit_chunks != current_chunks

    def mark_lighting_dirty(self):
        """Call when blocks change"""
        self.lighting_dirty = True
