#version 330

in vec2 in_position;
in vec2 in_world_pos;

uniform mat4 projection;
uniform vec2 screen_size;
uniform float tile_size;
uniform vec2 camera_pos; // ADD THIS

void main() {
    // Transform: world tiles -> relative to camera -> pixels -> screen space
    vec2 world_tile = in_world_pos + in_position;

    // Your coordinate system: screen_x = (world_x - camera_x) * tile_size + screen_width/2
    vec2 screen_pos;
    screen_pos.x = (world_tile.x - camera_pos.x) * tile_size + screen_size.x * 0.5;
    screen_pos.y = screen_size.y * 0.5 - (world_tile.y - camera_pos.y) * tile_size;

    gl_Position = projection * vec4(screen_pos, 0.0, 1.0);
}
