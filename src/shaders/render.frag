#version 330

in vec2 v_uv;

uniform usampler2D block_map;       // chunk_width x chunk_height, r16ui
uniform sampler2D  texture_atlas;
uniform sampler2D  light_map;

uniform vec2  screen_size;
uniform vec2  camera_pos;
uniform float tile_size;

uniform float chunk_world_x;        // chunk_x * chunk_width
uniform float chunk_width;
uniform float chunk_height;

uniform float atlas_tile_size;      // 1 / atlas_size normalised
uniform vec2  atlas_offsets[256];   // block_id -> atlas UV, set once at startup

uniform float world_offset_x;       // min_chunk_x * chunk_width
uniform vec2  light_map_size;

out vec4 fragColor;

void main() {
    // screen pixel -> world position
    vec2 screen_pixel = v_uv * screen_size;
    vec2 world_pos;
    world_pos.x = camera_pos.x + (screen_pixel.x - screen_size.x * 0.5) / tile_size;
    world_pos.y = camera_pos.y - (screen_size.y * 0.5 - screen_pixel.y) / tile_size;

    // clip to this chunk's x range
    float local_x = world_pos.x - chunk_world_x;
    if (local_x < 0.0 || local_x >= chunk_width) {
        fragColor = vec4(0.0);
        return;
    }
    if (world_pos.y < 0.0 || world_pos.y >= chunk_height) {
        fragColor = vec4(0.0);
        return;
    }

    // sample block ID
    vec2 block_uv = vec2(
        (floor(local_x) + 0.5) / chunk_width,
        1.0 - (floor(world_pos.y) + 0.5) / chunk_height
    );
    uint block_id = texture(block_map, block_uv).r;

    if (block_id == 0u) {
        fragColor = vec4(0.0);
        return;
    }

    // atlas lookup via uniform array
    vec2 atlas_offset = atlas_offsets[int(block_id)];
    vec2 tile_frac = vec2(fract(local_x), 1.0 - fract(world_pos.y));
    vec4 tex_color = texture(texture_atlas, atlas_offset + tile_frac * atlas_tile_size);

    if (tex_color.a < 0.01) {
        fragColor = vec4(0.0);
        return;
    }

    // lightmap (same UV math as air.frag)
    vec2 light_uv = vec2(
        (world_pos.x - world_offset_x) / light_map_size.x,
        1.0 - world_pos.y / light_map_size.y
    );
    vec3 light = texture(light_map, light_uv).rgb;

    fragColor = vec4(tex_color.rgb * light, tex_color.a);
}
