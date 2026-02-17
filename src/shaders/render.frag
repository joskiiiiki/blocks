// render.frag
#version 330

in vec2 v_texcoord;

uniform sampler2D texture_atlas;

out vec4 fragColor;

void main() {
    // Sample the texture atlas
    vec4 tex_color = texture(texture_atlas, v_texcoord);

    // Output the color
    fragColor = tex_color;
}
