#version 330

in vec2 v_uv;

uniform sampler2D light_texture;
uniform vec2 screen_size;
uniform vec2 camera_pos;
uniform float tile_size;
uniform vec2 world_offset;  // Offset to chunk start in world coords
uniform vec2 light_map_size;  // Size of light texture in tiles

out vec4 fragColor;

void main() {
    // Convert screen pixel to world tile coordinate
    vec2 screen_pixel = v_uv * screen_size;
    
    // Convert to world space
    vec2 from_camera;
    from_camera.x = (screen_pixel.x - screen_size.x * 0.5) / tile_size;
    from_camera.y = (screen_size.y * 0.5 - screen_pixel.y) / tile_size;

    vec2 world_tile = from_camera + camera_pos;
    
    // Convert to lightmap space (relative to world_offset)
    vec2 light_coord = world_tile - world_offset;
    
    // Normalize to [0, 1] for texture sampling
    vec2 light_uv = light_coord / light_map_size;
    
    // Check bounds
    if (light_uv.x < 0.0 || light_uv.x > 1.0 || 
        light_uv.y < 0.0 || light_uv.y > 1.0) {
        discard;
    }
    
    // Sample light texture
    vec4 light = texture(light_texture, v_uv);
    
    // Output light as color (semi-transparent overlay)
    fragColor = vec4(light.rgb, 1.0);
}
