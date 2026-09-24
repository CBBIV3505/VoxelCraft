#version 450 core

// World pipeline fragment shader: Minecraft-style face shading + directional
// sun/moon light, with a real shadow map (hardware PCF via a comparison
// sampler), the procedural block texture atlas, and distance fog into the
// current sky color.

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 pad;
    vec4 sun_dir;    // xyz = normalized direction TO the sun (already
                     // elevation-clamped on the CPU for shadow stability)
    vec4 sky_params; // x = sun intensity, y = ambient intensity,
                     // z = shadow strength, w = shadow texel world size
    vec4 fog_params; // x = fog start, y = fog end, z = shadows enabled (1/0)
} push;

// The light's view-projection: a push constant won't fit alongside the rest
// (128-byte standard limit minus the 96 already used), so it rides in a
// descriptor set instead — a uniform buffer shared by all frames.
layout(set = 0, binding = 0) uniform LightBlock {
    mat4 light_vp;
} light;

layout(set = 0, binding = 1) uniform sampler2D shadow_map;

// Block texture atlas: 16×16 procedural noise tiles in one row. UV (0,0)
// points at tile 0 — solid white — so every non-terrain vertex (HUD,
// overlays, drops) samples a no-op multiplier.
layout(set = 0, binding = 2) uniform sampler2D block_atlas;

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;
layout(location = 2) in float inViewDist;
layout(location = 3) in vec3 inWorldPos;
layout(location = 4) in vec2 inUV;

layout(location = 0) out vec4 outColor;

// Sky palette anchors for the fog TARGET — identical to dome_frag.glsl and
// sky.rs, so distant terrain fades into exactly the color the sky gradient
// shows at the rim (no seam between fogged terrain and the dome).
const vec3 FOG_DAY_HORIZON = vec3(0.62, 0.78, 0.94);
const vec3 FOG_NIGHT_HORIZON = vec3(0.05, 0.06, 0.12);
const vec3 FOG_DUSK_HORIZON = vec3(0.82, 0.40, 0.28);

void main() {
    // --- Atlas sample + keyed leaf transparency ----------------------------
    vec4 tex = texture(block_atlas, inUV);
    // The texture is an sRGB format, so the sampler returns linear values.
    // The leaf-hole key was authored in sRGB (255, 0, 254 — a magenta no
    // real tile uses); compare in sRGB space by converting back with sqrt.
    vec3 tex_srgb = sqrt(max(tex.rgb, vec3(0.0)));
    bool is_leaf_hole = tex_srgb.g < 0.004 && tex_srgb.b > 0.99 && tex_srgb.r > 0.99;
    if (is_leaf_hole) {
        discard;
    }

    vec3 n = normalize(inNormal);
    vec3 lightDir = normalize(push.sun_dir.xyz);

    float sun = push.sky_params.x;   // 1.0 day → 0.06 night (moonlight)
    float ambient = push.sky_params.y;

    // --- Minecraft-style per-face brightness --------------------------------
    float faceLight;
    if (n.y > 0.5) {
        faceLight = 1.0;
    } else if (n.y < -0.5) {
        faceLight = 0.55;
    } else {
        float axis = abs(n.x) * 0.86 + abs(n.z) * 0.72;
        faceLight = 0.55 + 0.35 * axis;
    }

    // --- Shadow-map lookup (2×2 hardware PCF) --------------------------------
    // Only surfaces facing the light can receive it; everything else is in
    // shadow by definition, so skip the map there.
    float directional = max(dot(n, lightDir), 0.0);
    float shadow = 1.0;
    if (directional > 0.001 && push.fog_params.z > 0.5) {
        vec4 clip = light.light_vp * vec4(inWorldPos, 1.0);
        vec3 proj = clip.xyz / clip.w;
        // Light clip space is [-1, 1] in X/Y; convert it to texture UVs.
        // The old shader used clip-space values directly, so only a narrow
        // quarter of the shadow map was ever sampled and most shadows looked
        // missing or detached.
        vec2 shadow_uv = proj.xy * 0.5 + 0.5;
        float bias = 0.0014 + push.sky_params.w * (1.0 - directional) * 1.5;
        float ref_depth = clamp(proj.z - bias, 0.0, 1.0);
        // Outside the shadow-map frustum → unshadowed.
        if (shadow_uv.x > 0.0 && shadow_uv.x < 1.0 && shadow_uv.y > 0.0 && shadow_uv.y < 1.0) {
            // A small manual PCF kernel is softer and more stable than a
            // single hardware comparison tap, especially on the Steam Deck's
            // 2048² shadow map.
            vec2 texel = vec2(push.sky_params.w);
            float sum = 0.0;
            for (int oy = -1; oy <= 1; ++oy) {
                for (int ox = -1; ox <= 1; ++ox) {
                    float stored = texture(shadow_map, shadow_uv + vec2(ox, oy) * texel).r;
                    sum += stored + 0.00001 >= ref_depth ? 1.0 : 0.0;
                }
            }
            shadow = sum / 9.0;
        }
    }

    float dirTerm = 0.55 + 0.45 * directional;
    float strength = push.sky_params.z;
    // In shadow the direct-light term is scaled down to (1 - strength);
    // strength < 1 keeps a sliver of bounce light inside shadows.
    float light = ambient * (faceLight + 0.45 * sun * dirTerm * mix(1.0 - strength, 1.0, shadow));

    vec3 lit_color = inColor * tex.rgb * light;

    // --- Distance fog into the sky gradient's horizon color ------------------
    // Fog end tracks the render distance (fog start = 40% of it) so distant
    // terrain always fades into the sky just inside the world edge. The
    // target color is derived from the sun elevation exactly like the dome
    // shader's rim, so fogged terrain melts into the gradient seamlessly.
    float fogStart = push.fog_params.x;
    float fogEnd = push.fog_params.y;
    float fog = clamp((inViewDist - fogStart) / (fogEnd - fogStart), 0.0, 1.0);
    fog = fog * fog * (3.0 - 2.0 * fog);

    float fday = smoothstep(-0.12, 0.30, push.sun_dir.y);
    float fwarmth = clamp(1.0 - abs(push.sun_dir.y) / 0.22, 0.0, 1.0);
    vec3 fog_target = mix(FOG_NIGHT_HORIZON, FOG_DAY_HORIZON, fday);
    fog_target = mix(fog_target, FOG_DUSK_HORIZON, fwarmth * 0.85);

    // Alpha rides pad.w: the opaque/HUD pipelines push 1.0 (and their blend
    // is disabled or opaque anyway); the water pipeline reads it (0.55–0.62)
    // for the translucent surface.
    outColor = vec4(mix(lit_color, fog_target, fog), push.pad.w);
}
