#version 330

in vec2 v_uv;

uniform sampler2D background_tile;
uniform sampler2D light_map;
uniform usampler2D sky_map;
uniform vec2 screen_size;
uniform vec2 camera_pos;
uniform float tile_size;
uniform float world_offset_x; // min_chunk_x * chunk_width (world X of lightmap left edge)
uniform vec2 light_map_size; // (total_width_in_blocks, height_in_blocks)

out vec4 fragColor;

void main() {
    // Screen pixel position (0,0 = top-left)
    vec2 screen_pixel = v_uv * screen_size;

    // Convert to world block coordinate
    // screen center maps to camera_pos
    vec2 world_pos;
    world_pos.x = camera_pos.x + (screen_pixel.x - screen_size.x * 0.5) / tile_size;
    // Y is flipped: screen Y increases downward, world Y increases upward
    world_pos.y = camera_pos.y - (screen_size.y * 0.5 - screen_pixel.y) / tile_size;

    // Sample tiling background (world-aligned)
    vec4 background = texture(background_tile, world_pos / 2.0);

    // Convert world pos to lightmap UV
    vec2 light_uv;
    light_uv.x = (world_pos.x - world_offset_x) / light_map_size.x;
    // Lightmap Y=0 is bottom of world, OpenGL textures are bottom-up so no flip needed
    // (you already flip in python with np.flip axis=1)
    light_uv.y = 1 - world_pos.y / light_map_size.y;

    uint is_sky = texture(sky_map, light_uv).r;

    if (is_sky > 0u || world_pos.y > 256) discard;

    vec4 light = texture(light_map, light_uv);

    // Multiply background by light
    fragColor = vec4(background.rgb * light.rgb, 1.0);
}
