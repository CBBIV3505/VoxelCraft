//! Keyboard/mouse input state: raw mouse look + sensitivity.
//! Movement physics live in [`crate::player`].

use std::collections::HashSet;

use winit::keyboard::KeyCode;

use crate::camera::Camera;

/// Default mouse look sensitivity (radians per pixel of mouse motion) —
/// adjustable at runtime with `[` and `]`.
pub const DEFAULT_MOUSE_SENSITIVITY: f32 = 0.0012;
pub const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// Per-frame input state accumulated from winit events, consumed by the camera
/// in RedrawRequested and cleared each frame.
pub struct InputState {
    pub keys_held: HashSet<KeyCode>,
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    pub cursor_grabbed: bool,
    /// Radians of rotation per pixel of mouse motion.
    pub mouse_sensitivity: f32,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            keys_held: Default::default(),
            mouse_dx: 0.0,
            mouse_dy: 0.0,
            cursor_grabbed: false,
            mouse_sensitivity: DEFAULT_MOUSE_SENSITIVITY,
        }
    }
}

impl InputState {
    /// Raw mouse motion — the ONLY look input while the cursor is grabbed.
    /// (Cursor-position deltas double-count on X11, where raw events AND
    /// position updates both arrive, and freeze under Wayland pointer-lock.)
    pub fn add_raw_motion(&mut self, dx: f64, dy: f64) {
        if self.cursor_grabbed {
            self.mouse_dx += dx as f32;
            self.mouse_dy += dy as f32;
        }
    }

    /// Adjust sensitivity (called from `[` / `]` handlers); returns the new value.
    pub fn scale_sensitivity(&mut self, factor: f32) -> f32 {
        self.mouse_sensitivity = if factor < 1.0 {
            (self.mouse_sensitivity * factor).max(0.0002)
        } else {
            (self.mouse_sensitivity * factor).min(0.01)
        };
        self.mouse_sensitivity
    }

    /// Apply accumulated mouse motion to the camera's yaw/pitch and clear it.
    pub fn update_look(&mut self, cam: &mut Camera) {
        let sens = self.mouse_sensitivity;
        cam.yaw -= self.mouse_dx * sens; // moving mouse right turns right
        cam.pitch -= self.mouse_dy * sens; // moving mouse up looks up
        cam.pitch = cam.pitch.clamp(-MAX_PITCH, MAX_PITCH);
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
    }
}
