"""
Per-chunk texture renderer.

Render passes per frame:
  1. Air pass     — 1 fullscreen quad (background + lighting for air/sky)
  2. Block passes — 1 quad per visible chunk (~8-10), each bound to its
                    own block_id texture (chunk_width x chunk_height, u16)

CPU work per frame: zero (unless a chunk is dirty).
atlas_offsets uniform is set once at startup and never touched again.
"""

from __future__ import annotations

import moderngl
import numpy as np
import numpy.typing as npt
import pygame

from src import assets, shaders
from src.assets import TEXTURES
from src.blocks import BLOCK_ID_MASK
from src.render.lighting import LightingManagerGL
from src.render.texture_atlas import TextureAtlas
from src.world import ChunkManager


class ChunkRendererGL:
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
        self.lighting_manager = lighting_manager

        self._block_textures: dict[int, moderngl.Texture] = {}
        self._dirty_chunks: set[int] = set()
        self.last_lit_chunks: set[int] = set()
        self.lighting_dirty: bool = False

        # atlas
        self.atlas = TextureAtlas(self.ctx, self.tile_size)
        self.atlas.build()

        # shaders — block and air share the same vert shader
        self.block_program = ctx.program(
            vertex_shader=shaders.RENDER_VERTEX_SHADER,
            fragment_shader=shaders.RENDER_FRAGMENT_SHADER,
        )
        self.air_program = ctx.program(
            vertex_shader=shaders.AIR_VERTEX_SHADER,
            fragment_shader=shaders.AIR_FRAGMENT_SHADER,
        )

        quad = self._make_quad_vbo()
        self.block_vao = ctx.vertex_array(
            self.block_program, [(quad, "2f 2f", "in_position", "in_uv")]
        )
        self.air_vao = ctx.vertex_array(
            self.air_program, [(quad, "2f 2f", "in_position", "in_uv")]
        )

        # write atlas offsets once — indexed by block_id, never changes
        flat: list[float] = []
        for i in range(256):
            u, v = self.atlas.uv(i)
            flat.extend([u, v])
        self.block_program["atlas_offsets"].write(
            np.array(flat, dtype="f4").tobytes()
        )
        self.block_program["atlas_tile_size"] = float(
            self.atlas.tile_size_normalized() or 0
        )
        self.block_program["chunk_width"] = float(self.chunk_manager.width)
        self.block_program["chunk_height"] = float(self.chunk_manager.height)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_quad_vbo(self) -> moderngl.Buffer:
        quad = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ], dtype="f4")
        return self.ctx.buffer(quad.tobytes())

    # ── chunk texture management ──────────────────────────────────────────────

    def _upload_chunk_texture(self, chunk_x: int) -> None:
        chunk = self.chunk_manager.get_chunk_from_cache(chunk_x)
        if chunk is None:
            return

        w = self.chunk_manager.width
        h = self.chunk_manager.height

        # shape (w, h) -> transpose to (h, w) for OpenGL row-major, then flip Y
        block_ids = (chunk.blocks & BLOCK_ID_MASK).astype(np.uint16)
        block_ids = np.flip(np.transpose(block_ids, (1, 0)), axis=0).copy()

        existing = self._block_textures.get(chunk_x)
        if existing is not None:
            existing.write(block_ids.tobytes())
        else:
            tex = self.ctx.texture((w, h), 1, data=block_ids.tobytes(), dtype="u2")
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            tex.repeat_x = False
            tex.repeat_y = False
            self._block_textures[chunk_x] = tex

    def mark_chunk_dirty(self, chunk_x: int) -> None:
        self._dirty_chunks.add(chunk_x)

    def mark_lighting_dirty(self) -> None:
        self.lighting_dirty = True

    # ── main render ───────────────────────────────────────────────────────────

    def render(self, camera_pos: tuple[float, float], resolution: tuple[int, int]):
        cam_x, _ = camera_pos
        sw, sh = resolution

        half_x = sw // self.tile_size // 2 + 1
        min_chunk_x = self.chunk_manager.get_chunk_x(cam_x - half_x)
        max_chunk_x = self.chunk_manager.get_chunk_x(cam_x + half_x)
        visible = list(range(min_chunk_x, max_chunk_x + 1))
        current_set = set(visible)

        self.chunk_manager.load_chunks(visible)

        # flush dirty and upload new chunks
        for cx in list(self._dirty_chunks):
            self._upload_chunk_texture(cx)
            self._dirty_chunks.discard(cx)
        for cx in visible:
            if cx not in self._block_textures:
                self._upload_chunk_texture(cx)

        # release unloaded chunk textures
        for cx in list(self._block_textures):
            if cx not in current_set:
                self._block_textures[cx].release()
                del self._block_textures[cx]

        # lighting
        if self.lighting_dirty or self.last_lit_chunks != current_set:
            self.lighting_manager.calculate_lighting_region(
                min_chunk_x, max_chunk_x, iterations=16
            )
            self.last_lit_chunks = current_set
            self.lighting_dirty = False

        lm_result = self._build_lightmap_textures(min_chunk_x, max_chunk_x)
        if lm_result is None:
            self.ctx.clear(*assets.COLOR_SKY.normalized)
            return

        lm_tex, sm_tex, lm_w, lm_h = lm_result
        world_offset_x = float(min_chunk_x * self.chunk_manager.width)

        self.ctx.viewport = (0, 0, sw, sh)
        self.ctx.clear(*assets.COLOR_SKY.normalized)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # ── pass 1: air / background ──────────────────────────────────────────
        bg_surf = TEXTURES["stone_background"].surface()
        bg_tex = self.ctx.texture(
            (32, 32), 4,
            data=pygame.image.tobytes(bg_surf, "RGBA", False),
            dtype="f1",
        )
        bg_tex.repeat_x = True
        bg_tex.repeat_y = True
        bg_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        bg_tex.use(0)
        lm_tex.use(1)
        sm_tex.use(2)

        p = self.air_program
        p["background_tile"] = 0
        p["light_map"] = 1
        p["sky_map"] = 2
        p["screen_size"] = (float(sw), float(sh))
        p["camera_pos"] = camera_pos
        p["tile_size"] = float(self.tile_size)
        p["world_offset_x"] = world_offset_x
        p["light_map_size"] = (float(lm_w), float(lm_h))
        self.air_vao.render(moderngl.TRIANGLES)
        bg_tex.release()

        # ── pass 2: one quad per visible chunk ────────────────────────────────
        self.atlas.texture.use(0)
        lm_tex.use(1)

        self.block_program["texture_atlas"] = 0
        self.block_program["light_map"] = 1
        self.block_program["screen_size"] = (float(sw), float(sh))
        self.block_program["camera_pos"] = camera_pos
        self.block_program["tile_size"] = float(self.tile_size)
        self.block_program["world_offset_x"] = world_offset_x
        self.block_program["light_map_size"] = (float(lm_w), float(lm_h))

        for cx in visible:
            block_tex = self._block_textures.get(cx)
            if block_tex is None:
                continue
            block_tex.use(2)
            self.block_program["block_map"] = 2
            self.block_program["chunk_world_x"] = float(
                cx * self.chunk_manager.width
            )
            self.block_vao.render(moderngl.TRIANGLES)

        self.ctx.disable(moderngl.BLEND)
        lm_tex.release()
        sm_tex.release()

    # ── lightmap helpers ──────────────────────────────────────────────────────

    def _build_lightmap_textures(
        self, min_chunk_x: int, max_chunk_x: int
    ) -> tuple[moderngl.Texture, moderngl.Texture, int, int] | None:
        lms: list[npt.NDArray] = []
        sms: list[npt.NDArray] = []
        for cx in range(min_chunk_x, max_chunk_x + 1):
            lm = self.lighting_manager.get_lightmap(cx)
            sm = self.lighting_manager.get_skymap(cx)
            if lm is not None and sm is not None:
                lms.append(lm)
                sms.append(sm)
        if not lms:
            return None

        lm_combined = np.concatenate(lms, axis=0)   # (total_w, h, 3)
        sm_combined = np.concatenate(sms, axis=0)   # (total_w, h)
        w, h = lm_combined.shape[0], lm_combined.shape[1]

        rgba = np.ones((w, h, 4), dtype=np.float32)
        rgba[:, :, :3] = lm_combined
        rgba = np.flip(np.transpose(rgba, (1, 0, 2)), axis=0).copy()
        lm_tex = self.ctx.texture((w, h), 4, data=rgba.tobytes(), dtype="f4")
        lm_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        sm_u8 = np.flip(
            np.transpose(sm_combined.astype(np.uint8), (1, 0)), axis=0
        ).copy()
        sm_tex = self.ctx.texture(
            (sm_combined.shape[0], sm_combined.shape[1]), 1,
            data=sm_u8.tobytes(), dtype="u1",
        )
        sm_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        return lm_tex, sm_tex, w, h
