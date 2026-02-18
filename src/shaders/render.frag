// render.frag
#version 330

in vec2 v_uv;
in vec3 v_light; // ADD THIS

uniform sampler2D texture_atlas;

out vec4 fragColor;

void main() {
    vec4 tex_color = texture(texture_atlas, v_uv);

    // Apply lighting
    vec3 lit_color = tex_color.rgb * v_light;

    fragColor = vec4(v_light, tex_color.a);
}
