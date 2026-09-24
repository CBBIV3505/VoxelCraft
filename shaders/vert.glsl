#version 450 core

// World pipeline vertex shader: transforms positions by the camera MVP and
// passes the world position through so the fragment stage can project into
// the shadow map (comparison-sampled for hardware PCF).

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 pad;
    vec4 sun_dir;    // xyz = normalized direction TO the sun
    vec4 sky_params; // x = sun intensity, y = ambient intensity
    vec4 fog_params; // x = fog start, y = fog end, z = shadows enabled
} push;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outColor;
layout(location = 2) out float outViewDist;
layout(location = 3) out vec3 outWorldPos;
layout(location = 4) out vec2 outUV;

void main() {
    gl_Position = push.mvp * vec4(inPos, 1.0);
    outNormal = inNormal;
    outColor = inColor;
    outViewDist = gl_Position.w;
    outWorldPos = inPos;
    outUV = inUV;
}
