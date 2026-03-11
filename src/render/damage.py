from typing import cast

import moderngl
import numpy as np
from pygame.math import clamp

from src.shaders import DAMAGE_FRAGMENT_SHADER, DAMAGE_VERTEX_SHADER


class DamageOverlay:
    def __init__(self, ctx: moderngl.Context, duration: float) -> None:
        self.ctx = ctx
        self.duration = duration
        self.prog = ctx.program(
            vertex_shader=DAMAGE_VERTEX_SHADER, fragment_shader=DAMAGE_FRAGMENT_SHADER
        )

        vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype="f4",
        )

        self.vbo = ctx.buffer(vertices)
        self.vao = ctx.vertex_array(
            self.prog,
            [(self.vbo, "2f 2f", "in_position", "in_uv")],
        )

    def uniform(prog: moderngl.Program, name: str) -> moderngl.Uniform:
        return cast(moderngl.Uniform, prog[name])

    def render(self, hit_timer: float, resolution: tuple[int, int]) -> None:
        elapsed = self.duration - hit_timer
        intensity = hit_timer / self.duration
        if intensity <= 0.0:
            return

        u_intensity = cast(moderngl.Uniform, self.prog["u_intensity"])
        u_time = cast(moderngl.Uniform, self.prog["u_time"])
        u_resolution = cast(moderngl.Uniform, self.prog["u_resolution"])

        u_intensity.value = clamp(intensity, 0.0, 1.0)
        u_time.value = elapsed
        u_resolution.value = (
            float(resolution[0]),
            float(resolution[1]),
        )
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    def destroy(self) -> None:
        self.vao.release()
        self.vbo.release()
        self.prog.release()
