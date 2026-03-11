import pathlib

shader_dir = pathlib.Path(__file__).parent.resolve()

render_frag_shader_path = shader_dir / "render.frag"
render_vert_shader_path = shader_dir / "render.vert"
lighting_comp_shader_path = shader_dir / "lighting.comp"
air_frag_shader_path = shader_dir / "air.frag"
air_vert_shader_path = shader_dir / "air.vert"
damage_frag_shader_path = shader_dir / "damage.frag"
damage_vert_shader_path = shader_dir / "damage.vert"

RENDER_FRAGMENT_SHADER = render_frag_shader_path.read_text()
RENDER_VERTEX_SHADER = render_vert_shader_path.read_text()
LIGHTING_COMPUTE_SHADER = lighting_comp_shader_path.read_text()
AIR_FRAGMENT_SHADER = air_frag_shader_path.read_text()
AIR_VERTEX_SHADER = air_vert_shader_path.read_text()
DAMAGE_FRAGMENT_SHADER = damage_frag_shader_path.read_text()
DAMAGE_VERTEX_SHADER = damage_vert_shader_path.read_text()
