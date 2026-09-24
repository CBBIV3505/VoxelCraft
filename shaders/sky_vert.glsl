#version 450 core

// Sun/moon disc vertex shader: world-space billboard quads projected by the
// camera VP. No lighting, no fog — discs sit crisply against the sky.

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 pad;
    vec4 sun_dir;
    vec4 sky_params;
} push;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal; // unused (shared vertex layout)
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;     // unused (points at white tile 0)

layout(location = 0) out vec3 outColor;

void main() {
    gl_Position = push.mvp * vec4(inPos, 1.0);
    outColor = inColor;
}
