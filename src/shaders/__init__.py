import pathlib

shader_dir = pathlib.Path(__file__).parent.resolve()

render_frag_shader_path = shader_dir / "render.frag"
render_vert_shader_path = shader_dir / "render.vert"

RENDER_FRAGMENT_SHADER = render_frag_shader_path.read_text()
RENDER_VERTEX_SHADER = render_vert_shader_path.read_text()
