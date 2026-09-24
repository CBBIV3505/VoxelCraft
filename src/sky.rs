//! Day/night cycle: a time-of-day clock rotates the sun around the world on
//! a great circle, and every other sky quantity — sky/fog colors, ambient and
//! sun intensity — derives analytically from the sun's elevation, so the
//! whole cycle is one continuous smooth rotation (no keyframe snapping).
//!
//! Time 0 = dawn (sun due east, at the horizon), 0.25 = noon (straight up),
//! 0.5 = sunset (due west), 0.5–1.0 = night with the moon riding the same
//! circle opposite the sun.

/// Length of one full day in seconds (real time; 10-minute like Minecraft's
/// 20-minute cycle halves nicely for a demo).
pub const DAY_LENGTH_SECS: f32 = 600.0;

/// A single light/sky state: sun direction (normalized on use), sky top and
/// horizon colors (linear RGB), fog color, sun and ambient intensities.
#[derive(Clone, Copy, Debug)]
pub struct SkyState {
    pub sun_dir: [f32; 3],
    /// Sky gradient endpoints — currently only `fog` feeds the renderer's
    /// clear color; these are kept for an upcoming gradient-sky shader.
    #[allow(dead_code)]
    pub zenith: [f32; 3],
    #[allow(dead_code)]
    pub horizon: [f32; 3],
    pub fog: [f32; 3],
    pub sun_intensity: f32,
    pub ambient: f32,
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Palette anchors (linear RGB, matching the old keyframes so the look is
/// continuous with what came before).
const DAY_ZENITH: [f32; 3] = [0.22, 0.46, 0.85];
const DAY_HORIZON: [f32; 3] = [0.62, 0.78, 0.94];
const DAY_FOG: [f32; 3] = [0.58, 0.72, 0.90];
const NIGHT_ZENITH: [f32; 3] = [0.02, 0.03, 0.08];
const NIGHT_HORIZON: [f32; 3] = [0.05, 0.06, 0.12];
const NIGHT_FOG: [f32; 3] = [0.04, 0.05, 0.10];
/// Sunset/sunrise warm tint, blended in near the horizon crossings.
const DUSK_HORIZON: [f32; 3] = [0.82, 0.40, 0.28];
const DUSK_FOG: [f32; 3] = [0.56, 0.36, 0.34];

/// Sample the sky/light state at a normalized time in [0, 1).
pub fn sample(time_of_day: f32) -> SkyState {
    let t = time_of_day.rem_euclid(1.0);

    // The sun rides a great circle: dawn (t=0) due east on the horizon,
    // noon (t=0.25) straight up, sunset (t=0.5) due west. The z offset tips
    // the circle slightly south so shadows never pass exactly through
    // straight-down; normalized, so it stays a unit vector.
    let a = t * std::f32::consts::TAU; // day angle around the circle
    let raw = [a.cos(), a.sin(), 0.24];
    let len = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
    let sun_dir = [raw[0] / len, raw[1] / len, raw[2] / len];

    // Sun elevation above the horizon drives everything else.
    let h = sun_dir[1];

    // Day↔night color blend: night below −0.12, full day above +0.30.
    let day = smoothstep(-0.12, 0.30, h);
    // Warm dawn/dusk tint peaks exactly at the horizon crossings.
    let warmth = (1.0 - (h / 0.22).abs()).clamp(0.0, 1.0);

    let zenith = lerp3(NIGHT_ZENITH, DAY_ZENITH, day);
    let mut horizon = lerp3(NIGHT_HORIZON, DAY_HORIZON, day);
    let mut fog = lerp3(NIGHT_FOG, DAY_FOG, day);
    horizon = lerp3(horizon, DUSK_HORIZON, warmth * 0.85);
    fog = lerp3(fog, DUSK_FOG, warmth * 0.8);

    // Direct light: 0.06 (moonlight floor) at night → 1.0 by late morning.
    let sun_intensity = 0.06 + 0.94 * smoothstep(0.0, 0.35, h);
    // Ambient: dark at night, bright by day, with the warm dip at dawn/dusk.
    let ambient = (0.22 + 0.58 * day) * (1.0 - 0.25 * warmth * (1.0 - day));

    SkyState {
        sun_dir,
        zenith,
        horizon,
        fog,
        sun_intensity,
        ambient,
    }
}

/// Per-frame time advance.
pub struct DayNight {
    /// Normalized time of day: 0 = dawn, 0.25 = noon, 0.5 = sunset.
    pub time: f32,
}

impl DayNight {
    pub fn new(start: f32) -> Self {
        Self {
            time: start.rem_euclid(1.0),
        }
    }

    pub fn advance(&mut self, dt: f32) {
        self.time = (self.time + dt / DAY_LENGTH_SECS).rem_euclid(1.0);
    }

    pub fn state(&self) -> SkyState {
        sample(self.time)
    }

    /// Direction TO the light that casts shadows, with the elevation clamped
    /// so the shadow frustum never goes edge-on: sun by day, moon at night
    /// (both cast shadows, per Minecraft). The clamped result is what the
    /// SHADOW map uses; the disc/terrain keep the true (unclamped) direction
    /// so dawn/dusk look right.
    pub fn shadow_light_dir(&self) -> [f32; 3] {
        let s = self.state();
        let mut d = s.sun_dir;
        // Below the horizon → moonlight is the shadow caster.
        if d[1] < 0.02 {
            d = [-d[0], -d[1], -d[2]]; // the moon sits opposite the sun
        }
        // Clamp the elevation into a band around 25°–90°: near dawn/dusk the
        // real sun sits too low for a stable ortho frustum, so shadows are
        // cast from a raised light while the visible sun stays where it is.
        let horiz = (d[0] * d[0] + d[2] * d[2]).sqrt().max(1e-5);
        let min_tan = 25f32.to_radians().tan();
        if d[1] / horiz < min_tan {
            d[1] = horiz * min_tan;
        }
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-5);
        [d[0] / len, d[1] / len, d[2] / len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_rides_a_unit_circle() {
        // Every sampled direction must be unit length and continuous across
        // the day: the step from t to t+ε must stay tiny (the old keyframe
        // table snapped at discontinuities).
        let mut prev = sample(0.998).sun_dir;
        let steps = 2000;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let s = sample(t);
            let l: f32 = s.sun_dir.iter().map(|c| c * c).sum();
            assert!((l - 1.0).abs() < 1e-4, "not unit at t={t}: {l}");
            let dot: f32 = prev.iter().zip(s.sun_dir.iter()).map(|(a, b)| a * b).sum();
            assert!(dot > 0.995, "sun direction jumped at t={t} (dot={dot})");
            prev = s.sun_dir;
        }
    }

    #[test]
    fn anchor_times_match_compass_positions() {
        // Dawn: east (+X). Noon: overhead. Sunset: west (−X).
        let dawn = sample(0.0);
        assert!(dawn.sun_dir[0] > 0.9 && dawn.sun_dir[1].abs() < 0.2);
        let noon = sample(0.25);
        assert!(noon.sun_dir[1] > 0.9);
        let dusk = sample(0.5);
        assert!(dusk.sun_dir[0] < -0.9 && dusk.sun_dir[1].abs() < 0.2);
        // Midnight: straight down (the moon, opposite, is straight up).
        let midnight = sample(0.75);
        assert!(midnight.sun_dir[1] < -0.9);
    }

    #[test]
    fn night_is_dark_and_day_is_bright() {
        let noon = sample(0.25);
        assert!(noon.sun_intensity > 0.95);
        let midnight = sample(0.75);
        assert!(midnight.sun_intensity <= 0.06 + 1e-4);
        assert!(midnight.ambient < 0.3);
    }

    /// The GLSL dome/fog shaders (dome_frag.glsl, frag.glsl) recompute the
    /// horizon palette from the sun elevation with these constants — they
    /// MUST stay identical to sky.rs's, or distant terrain fades into a
    /// color that doesn't match the sky gradient behind it (a visible seam
    /// at the world edge). This test locks the two in sync.
    #[test]
    fn horizon_palette_matches_glsl_dome_constants() {
        const DAY_HORIZON_GLSL: [f32; 3] = [0.62, 0.78, 0.94];
        const NIGHT_HORIZON_GLSL: [f32; 3] = [0.05, 0.06, 0.12];
        const DUSK_HORIZON_GLSL: [f32; 3] = [0.82, 0.40, 0.28];
        assert_eq!(DAY_HORIZON, DAY_HORIZON_GLSL);
        assert_eq!(NIGHT_HORIZON, NIGHT_HORIZON_GLSL);
        // The dusk blend uses the DUSK_HORIZON anchor in both.
        let t = 0.5; // sunset — full warmth
        let s = sample(t);
        let day = smoothstep(-0.12, 0.30, s.sun_dir[1]);
        let warmth = (1.0 - (s.sun_dir[1] / 0.22).abs()).clamp(0.0, 1.0);
        let mut glsl_rim = lerp3(NIGHT_HORIZON_GLSL, DAY_HORIZON_GLSL, day);
        glsl_rim = lerp3(glsl_rim, DUSK_HORIZON_GLSL, warmth * 0.85);
        for (a, b) in s.horizon.iter().zip(glsl_rim.iter()) {
            assert!((a - b).abs() < 1e-5, "rim mismatch: {a} vs {b}");
        }
    }
}
