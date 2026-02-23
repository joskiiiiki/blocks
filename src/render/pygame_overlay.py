import moderngl
import numpy as np
import pygame

# most basic vertex shader imaginable, just assigns given position and uv coordinates
VERTEX_SHADER = """
    #version 330

    in vec2 in_position;
    in vec2 in_uv;
    out vec2 uv;

    void main() {
        uv = in_uv;
        gl_Position = vec4(in_position, 0.0, 1.0);
    }
"""

# takes in the texture and texture coordinates, outputs the color at that coordinate
FRAGMENT_SHADER = """
    #version 330

    uniform sampler2D surface;
    in vec2 uv;
    out vec4 color;

    void main() {
        color = texture(surface, uv);
    }
"""


class PygameOverlay:
    resolution: tuple[int, int]
    surface: pygame.Surface
    ctx: moderngl.Context
    texture: moderngl.Texture
    texture_program: moderngl.Program
    vbo: moderngl.Buffer
    vao: moderngl.VertexArray

    def __init__(
        self,
        ctx: moderngl.Context,
        resolution: tuple[int, int],
    ):
        self.resolution = resolution
        self.surface = pygame.Surface(
            resolution,
            flags=pygame.SRCALPHA,
        )
        self.ctx = ctx
        self.texture = self.ctx.texture(self.resolution, 4)
        self.texture.filter = moderngl.NEAREST, moderngl.NEAREST
        # compile the shader program
        self.texture_program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        quad_vertices = np.array(
            [
                # x, y, u, v - we flip the y axis
                [-1.0, -1.0, 0.0, 1.0],  # top left
                [1.0, -1.0, 1.0, 1.0],  # top right
                [-1.0, 1.0, 0.0, 0.0],  # bottom left
                [1.0, 1.0, 1.0, 0.0],  # bottom right
            ],
            dtype="f4",
        )

        self.vbo = self.ctx.buffer(quad_vertices)
        self.vao = self.ctx.vertex_array(
            # 2 floats for position, 2 floats for texture coordinates
            self.texture_program,
            [(self.vbo, "2f 2f", "in_position", "in_uv")],
        )

    def clear(self):
        self.surface.fill((0, 0, 0, 0))  # transparent

    def blit(self, *args, **kwargs):
        self.surface.blit(*args, **kwargs)

    def upload(self):
        # use a low cost view - doest copy texture buffer so its cheap
        self.texture.write(self.surface.get_view("1"))

    def render(self):
        self.upload()
        # enable alpha blending
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.texture.swizzle = "BGRA"

        # binds texture to texture unit 0
        self.texture.use()

        # render the quads
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    def on_resize(self, resolution: tuple[int, int]):
        self.resolution = resolution
        self.surface = pygame.Surface(resolution, pygame.SRCALPHA)
        self.texture.release()
        self.texture = self.ctx.texture(resolution, 4)
        self.texture.filter = moderngl.NEAREST, moderngl.NEAREST

    def on_destroy(self):
        self.texture.release()
        self.vbo.release()
        self.vao.release()
