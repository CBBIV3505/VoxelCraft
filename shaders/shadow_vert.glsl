#version 450 core

// Shadow pass: render terrain depth from the light's point of view.
layout(push_constant) uniform Push {
    mat4 light_vp;
    vec4 params; // x = depth bias scale (per-vertex slope factor)
} push;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor; // unused, kept for the shared vertex layout
layout(location = 3) in vec2 inUV;    // unused, kept for the shared vertex layout

layout(location = 0) out float outSlope;

void main() {
    gl_Position = push.light_vp * vec4(inPos, 1.0);
    // Steep surfaces (normal facing away from the light's up axis) get a
    // larger depth bias so they don't shadow themselves — the classic
    // slope-scaled bias trick, done per-vertex since the world is static.
    outSlope = clamp(1.0 - abs(inNormal.y), 0.0, 1.0);
}
