# lighting.py - GPU-Accelerated Lighting with Compute Shaders (FIXED)

"""
GPU-based lighting system using compute shaders.
Much faster than CPU implementation - can handle 100k+ blocks in <1ms.
Uses ping-pong buffers to properly propagate light across iterations.

FIXED: Proper coordinate system handling - transposes arrays to match OpenGL texture layout.
"""

import moderngl
import numpy as np
from time import time
import numpy.typing as npt

from src.blocks import BLOCK_ID_MASK, Block
from src.shaders import LIGHTING_COMPUTE_SHADER
from src.world import ChunkManager

# Constants
SUN_LIGHT = (1.0, 1.0, 1.0)
TRANSPARENT_BLOCKS = {Block.AIR.value, Block.WATER.value}
BLOCK_LIGHTS = {
    Block.TORCH.value: (1.0, 0.75, 0.5),
}



class LightingManagerGL:
    """GPU-accelerated lighting manager using compute shaders"""

    ctx: moderngl.Context
    chunk_manager: ChunkManager
    compute_program: moderngl.ComputeShader
    lightmaps: dict[int, npt.NDArray[np.float32]]

    def __init__(self, chunk_manager: ChunkManager, ctx: moderngl.Context):
        self.ctx = ctx
        self.chunk_manager = chunk_manager
        self.lightmaps = {}

        # Compile compute shader
        print("Compiling lighting compute shader...")
        self.compute_program = self.ctx.compute_shader(LIGHTING_COMPUTE_SHADER)
        print("✓ Lighting compute shader compiled")

    def _build_light_sources(
        self, blocks: npt.NDArray[np.uint16]
    ) -> npt.NDArray[np.float32]:
        width, height = blocks.shape
        lightmap = np.zeros((width, height, 4), dtype=np.float32)

        block_ids = (blocks & BLOCK_ID_MASK).astype(np.uint8)

        # --- Sunlight ---
        # For each column, find the first non-air block from the top
        is_air = block_ids == Block.AIR.value  # [width, height]

        # cumprod from the top: stays True (1) until a non-air block is hit
        sunlit = np.cumprod(is_air[:, ::-1], axis=1)[:, ::-1].astype(bool)  # [width, height]

        lightmap[sunlit, 0] = SUN_LIGHT[0]
        lightmap[sunlit, 1] = SUN_LIGHT[1]
        lightmap[sunlit, 2] = SUN_LIGHT[2]

        # --- Block lights (torches etc.) ---
        for block_val, (r, g, b) in BLOCK_LIGHTS.items():
            mask = block_ids == block_val
            lightmap[:, :, 0] = np.where(mask, np.maximum(lightmap[:, :, 0], r), lightmap[:, :, 0])
            lightmap[:, :, 1] = np.where(mask, np.maximum(lightmap[:, :, 1], g), lightmap[:, :, 1])
            lightmap[:, :, 2] = np.where(mask, np.maximum(lightmap[:, :, 2], b), lightmap[:, :, 2])

        return lightmap

    def calculate_lighting_region(
        self, chunk_x_min: int, chunk_x_max: int, iterations: int = 16
    ):
        """Calculate lighting across multiple chunks using GPU with ping-pong buffers

        Args:
            chunk_x_min: Minimum chunk X coordinate
            chunk_x_max: Maximum chunk X coordinate
            iterations: Number of propagation iterations (increase for larger areas)
        """
        tt0 = time()
        print(f"Calculating GPU lighting for chunks {chunk_x_min} to {chunk_x_max}")

        # Stitch chunks together horizontally
        combined_blocks = []
        for chunk_x in range(chunk_x_min, chunk_x_max + 1):
            chunk = self.chunk_manager.get_chunk_from_cache(chunk_x)
            if chunk:
                combined_blocks.append(chunk.blocks)

        if not combined_blocks:
            print("  No chunks to light")
            return

        # Concatenate along X axis (chunks side-by-side)
        combined = np.concatenate(combined_blocks, axis=0)  # [total_width, height]

        print(f"  Combined shape (numpy): {combined.shape} [width, height]")

        # Get dimensions
        np_width, np_height = combined.shape

        # Build light sources BEFORE transposing
        t0 = time()
        light_sources = self._build_light_sources(combined)
        t1  = time()
        print(f"Buildings Lightsources: {t1 - t0}")

        # Create block ID map
        t0 = time()
        block_ids = (combined & BLOCK_ID_MASK).astype(np.uint8)
        
        block_ids_transposed = block_ids.T  # Now [height, width]
        light_sources_transposed = np.transpose(
            light_sources, (1, 0, 2)
        )  # [height, width, 4]

        # OpenGL texture dimensions (width, height)
        tex_width = np_width  # World X dimension
        tex_height = np_height  # World Y dimension

        t1 = time()

        print("Transposing: ", t1 - t0)

        print(f"  GPU texture size: {tex_width}x{tex_height} (width x height)")

        t0 = time()

        # Upload to GPU
        # Block map: R8UI (single channel, 8-bit unsigned int)
        block_texture = self.ctx.texture(
            (tex_width, tex_height),
            1,  # Single component
            data=block_ids_transposed.tobytes(),
            dtype="u1",
        )

        # Light source map: RGBA32F (4 components, 32-bit float)
        light_source_texture = self.ctx.texture(
            (tex_width, tex_height),
            4,
            data=light_sources_transposed.tobytes(),
            dtype="f4",
        )

        # Create ping-pong buffers
        light_map_a = self.ctx.texture(
            (tex_width, tex_height),
            4,
            data=light_sources_transposed.tobytes(),  # Start with sources
            dtype="f4",
        )

        light_map_b = self.ctx.texture(
            (tex_width, tex_height),
            4,
            data=np.zeros((tex_height, tex_width, 4), dtype=np.float32).tobytes(),
            dtype="f4",
        )

        # Bind static textures (never change during propagation)
        block_texture.bind_to_image(1, read=True, write=False)
        light_source_texture.bind_to_image(2, read=True, write=False)

        # Set uniforms
        self.compute_program["width"] = tex_width
        self.compute_program["height"] = tex_height

        # Calculate work groups (16x16 threads per group)
        groups_x = (tex_width + 15) // 16
        groups_y = (tex_height + 15) // 16

        t1 = time()
        print("Uploading: ", t1 - t0)

        print(f"  Running {iterations} iterations [{groups_x}x{groups_y} work groups]")


        t0 = time()
        # PING-PONG propagation
        for i in range(iterations):
            if i % 2 == 0:
                # Even iteration: Read from A, write to B
                light_map_a.bind_to_image(0, read=True, write=False)
                light_map_b.bind_to_image(3, read=False, write=True)
            else:
                # Odd iteration: Read from B, write to A
                light_map_b.bind_to_image(0, read=True, write=False)
                light_map_a.bind_to_image(3, read=False, write=True)

            # Run compute shader
            self.compute_program.run(groups_x, groups_y)

            # Memory barrier ensures writes complete before next iteration
            self.ctx.memory_barrier()

        # Final sync
        self.ctx.finish()
        t1 = time()

        print("Compute", t1 - t0)

        t0 = time()

        # Read back final result
        final_texture = light_map_b if iterations % 2 == 0 else light_map_a
        light_data = final_texture.read()
        light_map = np.frombuffer(light_data, dtype=np.float32).reshape(
            tex_height, tex_width, 4
        )

        # Transpose back to numpy [width, height, channels] format
        light_map = np.transpose(light_map, (1, 0, 2))  # [width, height, 4]
        light_map = light_map[:, :, :3]  # Drop alpha channel, keep RGB

        print(f"  Result shape: {light_map.shape}")

        # Split back into individual chunks
        chunk_width = self.chunk_manager.width
        for i, chunk_x in enumerate(range(chunk_x_min, chunk_x_max + 1)):
            start_x = i * chunk_width
            end_x = start_x + chunk_width
            self.lightmaps[chunk_x] = light_map[start_x:end_x, :, :]

        t1 = time()

        print("Transposing 2", t1 - t0)

        # Cleanup GPU resources
        block_texture.release()
        light_source_texture.release()
        light_map_a.release()
        light_map_b.release()

        tt1 = time()

        print(f"  ✓ GPU lighting complete {tt1 - tt0}")
        

    def get_lightmap(self, chunk_x: int) -> npt.NDArray[np.float32] | None:
        """Get lightmap for a chunk (returns None if not calculated)"""
        return self.lightmaps.get(chunk_x)

    def mark_chunks_dirty(self, chunk_x_list: list[int]):
        """Mark chunks as needing recalculation"""
        for chunk_x in chunk_x_list:
            if chunk_x in self.lightmaps:
                del self.lightmaps[chunk_x]
