"""GLSL shaders for instanced voxel cubes with per-material lighting."""

INSTANCED_VERT = """
#version 330 core

layout(location = 0) in vec3 in_vertex;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 i_pos;
layout(location = 3) in float i_mat_id;

uniform mat4 mvp;
uniform vec3 u_light_dir;

out vec3 v_normal;
out float v_mat_id;

void main() {
    vec3 pos = i_pos + in_vertex;
    gl_Position = mvp * vec4(pos, 1.0);
    v_normal = in_normal;
    v_mat_id = i_mat_id;
}
"""

INSTANCED_FRAG = """
#version 330 core

in vec3 v_normal;
in float v_mat_id;

uniform vec3 u_light_dir;
uniform vec3 u_colors[32];
uniform float u_roughness[32];
uniform float u_metallic[32];
uniform int u_color_count;

out vec4 fragColor;

void main() {
    int id = int(v_mat_id + 0.5);
    if (id < 0 || id >= u_color_count) {
        fragColor = vec4(0.55, 0.55, 0.6, 1.0);
        return;
    }
    vec3 albedo = u_colors[id];
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    float diff = max(dot(N, L), 0.0);
    float ambient = 0.42;
    float diffuse = diff * 0.58;
    vec3 col = albedo * (ambient + diffuse);
    fragColor = vec4(col, 1.0);
}
"""
