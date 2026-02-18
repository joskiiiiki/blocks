// render.vert
#version 330

in vec2 in_position;
in vec2 in_uv;
in vec2 in_world_pos;
in vec2 in_atlas_offset;
in vec3 in_light; // ADD THIS

uniform mat4 projection;
uniform vec2 screen_size;
uniform float tile_size;
uniform vec2 camera_pos;

out vec2 v_uv;
out vec3 v_light;

void main() {
    vec2 world_tile = in_world_pos + in_position;

    vec2 screen_pos;
    screen_pos.x = (world_tile.x - camera_pos.x) * tile_size + screen_size.x * 0.5;
    screen_pos.y = screen_size.y * 0.5 - (world_tile.y - camera_pos.y) * tile_size;

    gl_Position = projection * vec4(screen_pos, 0.0, 1.0);

    v_uv = in_uv + in_atlas_offset; // pass uv coordinates
    v_light = in_light; // pass light data
}
