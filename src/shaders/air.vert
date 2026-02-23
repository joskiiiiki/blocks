#version 330

in vec2 in_position;  // Screen quad corner (0-1)
in vec2 in_uv;        // UV coordinates (0-1)

out vec2 v_uv;

void main() {
    // Full screen quad in NDC
    gl_Position = vec4(in_position * 2.0 - 1.0, 0.0, 1.0);
    v_uv = in_uv;
}
