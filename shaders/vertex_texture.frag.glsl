#version 460
in vec3 frag_vert;
out vec4 out_vec;

void main()
{
    out_vec = vec4(frag_vert, 1.0);
}
