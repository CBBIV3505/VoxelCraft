#version 450 core

// Sky dome vertex shader: a hemisphere around the camera carrying each
// vertex's WORLD position so the fragment stage can derive the gradient from
// height above the dome center (= the camera). No lighting, no fog, no
// shadow map — the dome is drawn first and paints behind everything.

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 cam_pos;    // xyz = camera position (the dome's center)
    vec4 sun_dir;    // xyz = direction TO the sun
    vec4 sky_params; // x = sun intensity, y = ambient
    vec4 fog_params; // x = fog start, y = fog end (= dome radius)
} push;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal; // unused (shared vertex layout)
layout(location = 2) in vec3 inColor;  // unused (white); palette lives below
layout(location = 3) in vec2 inUV;     // unused (shared vertex layout)

layout(location = 0) out vec3 outWorldPos;

void main() {
    gl_Position = push.mvp * vec4(inPos, 1.0);
    outWorldPos = inPos;
}
