//! First-person fly camera (Minecraft creative-flight style).

use glam::{Mat4, Vec3, Vec4};

/// Simple first-person fly camera. Yaw wraps; pitch is clamped so you can't
/// flip over.
pub struct Camera {
    pub pos: Vec3,
    pub yaw: f32,   // radians, 0 = looking toward -Z (OpenGL convention)
    pub pitch: f32, // radians, + = looking up
}

impl Camera {
    pub fn new() -> Self {
        Self {
            pos: Vec3::new(0.0, 0.4, 3.2),
            // yaw = π/2 looks toward -Z, i.e. straight at the cube at spawn.
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: std::f32::consts::FRAC_PI_8,
        }
    }

    /// Direction the camera looks (unit vector).
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.sin() * self.pitch.cos(),
        )
    }

    /// Level right vector (perpendicular to forward, parallel to the ground).
    pub fn right_of(&self) -> Vec3 {
        Vec3::new(
            (self.yaw - std::f32::consts::FRAC_PI_2).cos(),
            0.0,
            -(self.yaw - std::f32::consts::FRAC_PI_2).sin(),
        )
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.pos, self.pos + self.forward(), Vec3::Y)
    }

    /// `far` is the camera far plane in blocks — driven by the render
    /// distance so distant terrain never gets clipped by it.
    pub fn proj_matrix_far(&self, aspect: f32, far: f32) -> Mat4 {
        // Keep the HORIZONTAL fov constant as the window resizes (like most PC
        // games): with a fixed vertical fov, the horizontal fov would shrink on
        // narrow windows, making mouse look feel faster at the same sensitivity.
        // 1.2695 rad ≈ 72.7°, preserving the old 45° vfov feel at 16:9.
        const HFOV: f32 = 1.2695;
        let vfov = 2.0 * ((HFOV / 2.0).tan() / aspect).atan().clamp(0.05, 1.0472); // never exceed 120° vertical on tall windows
                                                                                   // glam's perspective_rh uses OpenGL conventions, but Vulkan's NDC has a
                                                                                   // downward-pointing Y axis — without flipping Y the whole world renders
                                                                                   // vertically mirrored (dirt above grass, Space appears to descend).
        let mut p = Mat4::perspective_rh(vfov, aspect, 0.1, far);
        p.y_axis *= -1.0;
        p
    }
}

/// View-projection for the current camera, aspect ratio, and far plane.
pub fn view_proj_far(cam: &Camera, aspect: f32, far: f32) -> Mat4 {
    cam.proj_matrix_far(aspect, far) * cam.view_matrix()
}

/// The six frustum planes of a view-projection matrix, each as
/// `[normal, d]` with the plane `dot(normal, p) + d >= 0` INSIDE.
/// Gribb-Hartmann extraction: left = row3 + row0, right = row3 - row0,
/// bottom = row3 + row1, top = row3 - row1, near = row3 + row2,
/// far = row3 - row2.
pub fn frustum_planes(vp: &Mat4) -> [[Vec3; 2]; 6] {
    let m = vp.to_cols_array_2d(); // m[col][row]
    let row = |i: usize| Vec4::new(m[0][i], m[1][i], m[2][i], m[3][i]);
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    let raw = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2];
    let mut out = [[Vec3::ZERO; 2]; 6];
    for (i, p) in raw.iter().enumerate() {
        let n = Vec3::new(p.x, p.y, p.z);
        let len = n.length().max(1e-6);
        out[i] = [n / len, Vec3::splat(p.w / len)];
    }
    out
}

/// Is an AABB (min, max) fully outside any frustum plane?
pub fn aabb_outside_frustum(planes: &[[Vec3; 2]; 6], min: Vec3, max: Vec3) -> bool {
    for [n, d] in planes {
        // Positive vertex: the corner furthest along the plane normal.
        let px = if n.x >= 0.0 { max.x } else { min.x };
        let py = if n.y >= 0.0 { max.y } else { min.y };
        let pz = if n.z >= 0.0 { max.z } else { min.z };
        if n.x * px + n.y * py + n.z * pz + d.x < 0.0 {
            return true; // outside this plane → outside the frustum
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera at the origin looking down -Z must keep the chunk in front
    /// of it and cull the chunk behind it (the classic culling failure
    /// mode: everything disappears or nothing does).
    #[test]
    fn frustum_culls_behind_and_keeps_ahead() {
        let mut cam = Camera::new();
        cam.pos = Vec3::ZERO;
        cam.yaw = 0.0;
        cam.pitch = 0.0;
        let vp = view_proj_far(&cam, 16.0 / 9.0, 300.0);
        let planes = frustum_planes(&vp);
        let chunk = |x: f32| (Vec3::new(x, 0.0, 0.0), Vec3::new(x + 16.0, 16.0, 16.0));
        // In front of the camera (viewing direction): visible.
        assert!(!aabb_outside_frustum(&planes, chunk(8.0).0, chunk(8.0).1));
        // Directly behind: must be culled.
        assert!(aabb_outside_frustum(
            &planes,
            chunk(-24.0).0,
            chunk(-24.0).1
        ));
        // Containing the camera: never culled.
        assert!(!aabb_outside_frustum(&planes, chunk(-8.0).0, chunk(-8.0).1));
    }

    /// The camera basis convention: forward() must point where the frustum
    /// opens — the far chunk along forward() survives, the opposite one
    /// dies. Guards against a silently flipped extraction.
    #[test]
    fn frustum_opens_along_forward() {
        let mut cam = Camera::new();
        cam.pos = Vec3::ZERO;
        cam.yaw = std::f32::consts::FRAC_PI_2;
        cam.pitch = 0.0;
        let vp = view_proj_far(&cam, 16.0 / 9.0, 300.0);
        let planes = frustum_planes(&vp);
        let fwd = cam.forward();
        let near = fwd * 24.0;
        let min = near - Vec3::splat(8.0);
        let max = near + Vec3::splat(8.0);
        assert!(!aabb_outside_frustum(&planes, min, max));
        let min = -near - Vec3::splat(8.0);
        let max = -near + Vec3::splat(8.0);
        assert!(aabb_outside_frustum(&planes, min, max));
    }
}
