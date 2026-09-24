#version 450 core

// Sky dome fragment: vertical gradient from the horizon color at the rim to
// the zenith color at the top, tinted warm at the horizon on the sun's side
// near dawn/dusk (a cheap forward-scattering glow). Push constants carry the
// camera position (pad.xyz), the dome radius (fog_params.y) and the same
// sky palette the day/night cycle already computes.

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 cam_pos;    // xyz = camera position (the dome's center)
    vec4 sun_dir;    // xyz = normalized direction TO the sun
    vec4 sky_params; // x = sun intensity, y = ambient intensity
    vec4 fog_params; // x = fog start, y = fog end (= dome radius)
} push;

layout(location = 0) in vec3 inWorldPos;

layout(location = 0) out vec4 outColor;

// Sky palette anchors (linear RGB) — these mirror sky.rs's constants so the
// gradient lands exactly on the fog color at the rim and the day/night
// cycle's palette stays consistent. Kept in sync by the sky-palette test.
const vec3 DAY_ZENITH = vec3(0.22, 0.46, 0.85);
const vec3 DAY_HORIZON = vec3(0.62, 0.78, 0.94);
const vec3 NIGHT_ZENITH = vec3(0.02, 0.03, 0.08);
const vec3 NIGHT_HORIZON = vec3(0.05, 0.06, 0.12);
const vec3 DUSK_HORIZON = vec3(0.82, 0.40, 0.28);
const vec3 DUSK_FOG = vec3(0.56, 0.36, 0.34);

vec3 lerp3(vec3 a, vec3 b, float t) {
    return mix(a, b, clamp(t, 0.0, 1.0));
}

void main() {
    // Normalized height above the horizon: 0 at the rim, 1 at the apex
    // (the dome radius equals fog end and the rim sits at camera height).
    float h = clamp((inWorldPos.y - push.cam_pos.y) / max(push.fog_params.y, 1.0),
                    0.0, 1.0);

    // Day↔night blend from the sun's elevation (matches sky.rs).
    float sun_h = push.sun_dir.y;
    float day = smoothstep(-0.12, 0.30, sun_h);
    float warmth = clamp(1.0 - abs(sun_h) / 0.22, 0.0, 1.0);

    vec3 zenith = lerp3(NIGHT_ZENITH, DAY_ZENITH, day);
    vec3 horizon = lerp3(NIGHT_HORIZON, DAY_HORIZON, day);
    horizon = lerp3(horizon, DUSK_HORIZON, warmth * 0.85);

    // Gradient: smoothstep rise from the rim, flattening toward the apex.
    float g = h * h * (3.0 - 2.0 * h);
    vec3 col = lerp3(horizon, zenith, g);

    // Warm forward-scatter glow around the low sun (dawn/dusk).
    if (warmth > 0.0) {
        vec3 view_dir = normalize(inWorldPos - push.cam_pos.xyz);
        float sun_amount = max(dot(view_dir, normalize(push.sun_dir.xyz)), 0.0);
        col = lerp3(col, DUSK_FOG, warmth * 0.35 * pow(sun_amount, 4.0));
    }

    outColor = vec4(col, 1.0);
}
