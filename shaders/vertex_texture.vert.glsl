#version 460
in vec3 in_vert;
in vec2 in_uv;

out vec3 frag_vert;

void main() {
    gl_Position = vec4(in_uv, 0, 1.0);
    frag_vert = in_vert;
}