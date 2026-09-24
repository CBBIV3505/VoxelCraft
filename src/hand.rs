//! First-person viewmodel: the player's arm and whatever it holds — a block
//! (real textured cube), a tool (extruded pixel sprite), or nothing (bare
//! skin-toned arm) — rendered Minecraft-style just in front of the camera.
//!
//! The whole group rides an anchor offset from the camera in WORLD space, so
//! the ordinary camera transform projects it. A swing (mining/placing)
//! rotates it through a chop arc, equipping (item change) dips it in from
//! below, and walking adds a gentle bob. Drawn through the world pipeline
//! (depth-tested against terrain so walls occlude the hand properly, lit by
//! the day/night sun like everything else).

use glam::Vec3;

use crate::geometry::Vertex;
use crate::items::ItemType;

/// How far the held item sits from the camera, in blocks (right, down,
/// forward from the eye).
const ANCHOR_RIGHT: f32 = 0.62;
const ANCHOR_DOWN: f32 = 0.52;
const ANCHOR_FWD: f32 = 0.95;
/// Held block edge length (a real block is 1.0 — the hand holds a mini one).
const BLOCK_SCALE: f32 = 0.42;
/// Bare-arm dimensions (a stretched box, like Minecraft's arm).
const ARM_HALF_W: f32 = 0.11;
const ARM_LEN: f32 = 0.85;

/// Lighting on the viewmodel: face tints like the terrain's (top 1.0, sides
/// mid) so it reads as 3D, then the fragment shader applies day/night light.
fn face_tint(n: Vec3) -> f32 {
    if n.y > 0.5 {
        1.0
    } else if n.y < -0.5 {
        0.55
    } else {
        0.72 + 0.14 * n.x.abs()
    }
}

/// One face of a transformed unit cube: returns 6 vertices (two tris).
#[allow(clippy::too_many_arguments)]
fn cube_face(
    verts: &mut Vec<Vertex>,
    center: Vec3,
    u: Vec3,
    v: Vec3,
    n: Vec3,
    half_u: f32,
    half_v: f32,
    color: [f32; 3],
    uv_rect: Option<([f32; 2], [f32; 2])>,
) {
    let nu = n.to_array();
    let k = face_tint(n);
    let col = [color[0] * k, color[1] * k, color[2] * k];
    let c = center - u * half_u - v * half_v;
    let du = u * (2.0 * half_u);
    let dv = v * (2.0 * half_v);
    // Corners: c, c+du, c+du+dv, c+dv — two CCW triangles.
    let corners = [c, c + du, c + du + dv, c + dv];
    let (lo, hi) = uv_rect.unwrap_or(([0.0, 0.0], [0.0, 0.0]));
    // Match terrain corner_uv convention: v flipped so upright content
    // (grass fringe) sits at the top of the face.
    let uvs = [
        [lo[0], hi[1]],
        [hi[0], hi[1]],
        [hi[0], lo[1]],
        [lo[0], lo[1]],
    ];
    for idx in [0usize, 1, 2, 0, 2, 3] {
        let p = corners[idx];
        let uv = if uv_rect.is_some() {
            uvs[idx]
        } else {
            [0.0, 0.0]
        };
        verts.push(Vertex {
            pos: p.to_array(),
            normal: nu,
            color: col,
            uv,
        });
    }
}

/// Solid-color cube (6 faces × 6 verts), used for the bare arm and tool
/// handles: `size` = (half x, half y, half z), centered at `center`.
fn push_box(verts: &mut Vec<Vertex>, center: Vec3, half: Vec3, color: [f32; 3]) {
    // (u, v, n) per face: u/v span the face, n is the outward normal.
    let faces: [(Vec3, Vec3, Vec3); 6] = [
        (Vec3::X, Vec3::Z, Vec3::Y),  // top
        (Vec3::X, Vec3::Z, -Vec3::Y), // bottom
        (Vec3::Z, Vec3::Y, Vec3::X),  // +X
        (Vec3::Z, Vec3::Y, -Vec3::X), // -X
        (Vec3::X, Vec3::Y, Vec3::Z),  // +Z
        (Vec3::X, Vec3::Y, -Vec3::Z), // -Z
    ];
    for (u, v, n) in faces {
        let hu = u.abs().dot(half);
        let hv = v.abs().dot(half);
        let hn = n.abs().dot(half);
        // `cube_face` receives the centre of the face, not the centre of the
        // whole box. Without this normal offset all six faces collapsed into
        // two overlapping squares at the box centre — the viewmodel looked
        // flat even though it had six face draw calls.
        cube_face(verts, center + n * hn, u, v, n, hu, hv, color, None);
    }
}

/// Texture-mapped mini-cube (held block): per-face atlas tiles exactly like
/// terrain's `face_tiles`, so grass shows its fringe and logs their rings.
/// Returns 36 vertices, cube centered at origin, side = 1 (scaled by caller).
pub fn held_block_vertices(t: crate::world::BlockType) -> Vec<Vertex> {
    use crate::terrain::tiles_helpers::face_tiles_for_held;
    let mut verts = Vec::with_capacity(36);
    let s = 0.5;
    // Same face order as terrain's FACES: [+Y, -Y, +X, -X, +Z, -Z], with
    // matching corner layouts (unit cube at origin, CCW from outside).
    let faces: [([Vec3; 4], Vec3); 6] = [
        // +Y top
        (
            [
                Vec3::new(0.0, s, 0.0),
                Vec3::new(0.0, s, 1.0),
                Vec3::new(1.0, s, 1.0),
                Vec3::new(1.0, s, 0.0),
            ],
            Vec3::Y,
        ),
        // -Y bottom
        (
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            -Vec3::Y,
        ),
        // +X
        (
            [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
            ],
            Vec3::X,
        ),
        // -X
        (
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            -Vec3::X,
        ),
        // +Z
        (
            [
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(0.0, 1.0, 1.0),
            ],
            Vec3::Z,
        ),
        // -Z
        (
            [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
            -Vec3::Z,
        ),
    ];
    for (face_idx, (corners, n)) in faces.iter().enumerate() {
        let tile = face_tiles_for_held(t, face_idx);
        let (lo, hi) = crate::terrain::tile_uv(tile);
        let k = face_tint(*n);
        // The hand block isn't in the world, so per-block jitter is 1.0.
        let (lo, hi) = {
            let jx = 0.0;
            ([lo[0] + jx, lo[1]], hi)
        };
        // Emit 6 vertices per face (TRIANGLE_LIST, no index buffer): the
        // shared-corner 4-vert form is only for indexed draws.
        let face_uv = |c: &Vec3| -> [f32; 2] {
            // Map the corner onto the tile with the same axis convention
            // as terrain's corner_uv (v flipped on side faces).
            let (ua, va) = crate::terrain::tangent_axes([n.x as i32, n.y as i32, n.z as i32]);
            let arr = c.to_array();
            let (u_axis, v_axis, v_flip) = match (ua, va) {
                (0, 2) => (0usize, 2usize, false),
                (1, 2) => (2, 1, true),
                _ => (0, 1, true),
            };
            let u = lo[0] + (hi[0] - lo[0]) * arr[u_axis];
            let v_raw = lo[1] + (hi[1] - lo[1]) * arr[v_axis];
            let v = if v_flip {
                hi[1] - (v_raw - lo[1])
            } else {
                v_raw
            };
            [u, v]
        };
        for idx in [0usize, 1, 2, 0, 2, 3] {
            let c = &corners[idx];
            verts.push(Vertex {
                pos: (c - Vec3::ONE * 0.5).to_array(), // centered at origin
                normal: n.to_array(),
                color: [k, k, k],
                uv: face_uv(c),
            });
        }
    }
    verts
}

/// The held tool as an extruded 8×8 pixel sprite: each lit cell becomes a
/// small colored box, so the pickaxe/axe reads correctly from any angle.
/// Sprite authored top→bottom (like the hotbar bitmaps, scaled up).
const PICK_SPRITE: [[u8; 8]; 8] = [
    [0, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
];
const AXE_SPRITE: [[u8; 8]; 8] = [
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
];

/// Tool colors: wood handle + slightly lighter head.
const TOOL_HANDLE: [f32; 3] = [0.42, 0.30, 0.16];
const TOOL_HEAD: [f32; 3] = [0.62, 0.45, 0.26];

fn push_sprite_box(
    verts: &mut Vec<Vertex>,
    sprite: &[[u8; 8]; 8],
    head_mask: &[[u8; 8]; 8],
    origin: Vec3,
    cell: f32,
    thickness: f32,
    head_color: [f32; 3],
) {
    for (ry, row) in sprite.iter().enumerate() {
        for (rx, &on) in row.iter().enumerate() {
            if on == 0 {
                continue;
            }
            let color = if head_mask[ry][rx] == 1 {
                head_color
            } else {
                TOOL_HANDLE
            };
            let center = origin + Vec3::X * ((rx as f32 + 0.5) * cell)
                - Vec3::Y * ((ry as f32 + 0.5) * cell)
                + Vec3::Z * (thickness * 0.5);
            push_box(
                verts,
                center,
                Vec3::new(cell * 0.5, cell * 0.5, thickness * 0.5),
                color,
            );
        }
    }
}

/// Cell highlight masks for the tool heads (1 = head pixel).
const PICK_HEAD: [[u8; 8]; 8] = [
    [0, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
];
const AXE_HEAD: [[u8; 8]; 8] = [
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
];

/// Frame state for the viewmodel animation.
#[derive(Clone, Copy)]
pub struct HandAnim {
    /// Swing progress 0..1 while mining/placing (0 = idle).
    pub swing: f32,
    /// Equip dip 0..1 (1 = fully dipped; eases back to 0 after an item change).
    pub equip: f32,
    /// Walk-cycle phase (accumulated distance / stride); 0 = standing.
    pub walk: f32,
    /// True when the player is moving on foot (drives the bob amplitude).
    pub walking: bool,
}

/// Build this frame's viewmodel geometry, positioned in WORLD space relative
/// to the camera. `cam` gives position + basis; `held` picks the content.
pub fn build_hand(
    cam: &crate::camera::Camera,
    held: Option<ItemType>,
    anim: HandAnim,
) -> Vec<Vertex> {
    let mut verts = Vec::with_capacity(64);

    // Camera basis: right/level-forward.
    let right = cam.right_of();
    let flat_fwd = Vec3::new(cam.yaw.cos(), 0.0, -cam.yaw.sin()).normalize();
    let up = Vec3::Y;

    // --- Swing rotation: a chop around a pivot below-right of the item ---
    // Idle: 0. Swing: arc forward-down and back (sin easing).
    let s = anim.swing;
    let swing_fwd = (s * std::f32::consts::PI).sin(); // 0→1→0 over the swing
    let swing_pitch = -swing_fwd * 1.15; // chop downward (radians)
    let swing_roll = swing_fwd * 0.35;

    // --- Walk bob ---
    let bob = if anim.walking {
        (anim.walk * std::f32::consts::TAU).sin() * 0.035
    } else {
        0.0
    };
    let bob_x = if anim.walking {
        (anim.walk * std::f32::consts::TAU + std::f32::consts::FRAC_PI_2).sin() * 0.02
    } else {
        0.0
    };

    // --- Equip dip ---
    let dip = anim.equip * 0.55;

    // Anchor: camera + offsets (bob included), rotated by the swing around
    // a shoulder pivot: the pivot sits down-right of the eye; the whole
    // group swings about it like an arm.
    let pivot =
        cam.pos + right * (ANCHOR_RIGHT + bob_x) - up * 0.35 + flat_fwd * (ANCHOR_FWD * 0.35);
    // The offsets below are camera-relative; `local_off` used to be part of
    // the old mixed-space placement and is intentionally gone.

    // Swing rotation, expressed in CAMERA space: the chop pitches around
    // the camera's right axis and rolls around its level-forward axis. The
    // old world-axis rotation (rot_xz) kept the swing fixed in world space,
    // so turning the camera made the arm chop sideways or twist visibly.
    let rot = glam::Mat3::from_axis_angle(right, swing_pitch)
        * glam::Mat3::from_axis_angle(flat_fwd, swing_roll);
    let world_p = |p: Vec3| -> Vec3 { pivot + rot * (p - pivot) };

    match held {
        Some(ItemType::Block(t)) => {
            // Keep the block close to the right side of the viewport and put
            // the wrist underneath it.  The previous calculation mixed a
            // camera-relative offset with a world-space pivot, which made the
            // arm swing sideways/upward and look detached at some angles.
            let center =
                cam.pos + right * (0.58 + bob_x) - up * (0.42 - dip + bob) + flat_fwd * 0.88;
            let center = world_p(center);
            let block = held_block_vertices(t);
            for v in block {
                let p = Vec3::from(v.pos) * BLOCK_SCALE + center;
                // Rotate the cube by the swing around its own center.
                let p = center + rot * (p - center);
                verts.push(Vertex {
                    pos: p.to_array(),
                    ..v
                });
            }
            let arm_base = cam.pos + right * (0.66 + bob_x) - up * (0.92 + dip) + flat_fwd * 0.72;
            let arm_dir = (center - arm_base).normalize();
            let arm_mid = arm_base + arm_dir * (ARM_LEN * 0.45);
            let arm_mid = world_p(arm_mid);
            push_box(
                &mut verts,
                arm_mid,
                Vec3::new(ARM_HALF_W, ARM_LEN * 0.5, ARM_HALF_W),
                SKIN,
            );
        }
        Some(other) => {
            // Tool: sprite plane held up-right, plus the arm below it.
            let sprite = match other {
                ItemType::WoodPickaxe => (&PICK_SPRITE, &PICK_HEAD, TOOL_HEAD),
                // Stone pick: same silhouette, gray head.
                ItemType::StonePickaxe => (&PICK_SPRITE, &PICK_HEAD, [0.55, 0.55, 0.58]),
                ItemType::WoodAxe => (&AXE_SPRITE, &AXE_HEAD, TOOL_HEAD),
                ItemType::Stick => {
                    // A simple stick: a thin diagonal box.
                    let center = pivot + right * (ANCHOR_RIGHT + bob_x - 0.05)
                        - up * (ANCHOR_DOWN - dip + bob)
                        + flat_fwd * ANCHOR_FWD;
                    let center = world_p(center);
                    let stick = center + rot * (Vec3::ZERO);
                    push_box(&mut verts, stick, Vec3::new(0.03, 0.34, 0.03), TOOL_HANDLE);
                    // Bare arm below the stick.
                    let arm_mid = pivot + right * (ANCHOR_RIGHT + bob_x) - up * (0.30 + dip)
                        + flat_fwd * (ANCHOR_FWD * 0.7);
                    let arm_mid = world_p(arm_mid);
                    push_box(
                        &mut verts,
                        arm_mid,
                        Vec3::new(ARM_HALF_W, ARM_LEN * 0.45, ARM_HALF_W),
                        SKIN,
                    );
                    return verts;
                }
                ItemType::Block(_) => unreachable!(),
            };
            let (sprite, head, head_color) = sprite;
            let cell = 0.045;
            let origin = pivot + right * (ANCHOR_RIGHT + bob_x - 4.0 * cell)
                - up * (ANCHOR_DOWN - dip + bob - 4.0 * cell)
                + flat_fwd * ANCHOR_FWD;
            // Extrude each lit cell; the whole sprite swings with the arm.
            let mut sprite_verts = Vec::new();
            push_sprite_box(
                &mut sprite_verts,
                sprite,
                head,
                Vec3::ZERO,
                cell,
                0.03,
                head_color,
            );
            for v in sprite_verts {
                let p = Vec3::from(v.pos);
                // Rotate the sprite into the camera's frame: right/up/fwd.
                let world = origin + right * p.x - up * p.y + flat_fwd * p.z;
                let world = pivot + rot * (world - pivot);
                verts.push(Vertex {
                    pos: world.to_array(),
                    ..v
                });
            }
            let arm_mid = cam.pos + right * (0.62 + bob_x) - up * (0.58 + dip) + flat_fwd * 0.78;
            let arm_mid = world_p(arm_mid);
            push_box(
                &mut verts,
                arm_mid,
                Vec3::new(ARM_HALF_W, ARM_LEN * 0.45, ARM_HALF_W),
                SKIN,
            );
        }
        None => {
            // Bare arm: a long box angled up from the bottom-right corner,
            // swinging with the hand like the held-item group.
            let base = pivot + right * (ANCHOR_RIGHT + bob_x + 0.06) - up * (0.62 + dip)
                + flat_fwd * (ANCHOR_FWD * 0.55);
            let tip = pivot + right * (ANCHOR_RIGHT + bob_x - 0.10) - up * (0.10 + dip - bob)
                + flat_fwd * ANCHOR_FWD;
            let mid = (base + tip) * 0.5;
            let mid = world_p(mid);
            let len = (tip - base).length();
            // Orient the box along base→tip: build it axis-aligned then
            // rotate is overkill — approximate with a tilted box by
            // aligning its long axis to the arm direction.
            let dir = (tip - base).normalize();
            let yaw = dir.z.atan2(dir.x);
            let pitch = dir.y.asin();
            push_oriented_box(
                &mut verts,
                mid,
                ARM_HALF_W,
                len * 0.5,
                ARM_HALF_W,
                yaw,
                pitch,
                SKIN,
            );
        }
    }
    verts
}

/// Skin tone (linear RGB, warm tan like Minecraft's Steve arm).
const SKIN: [f32; 3] = [0.72, 0.52, 0.36];

/// Axis-aligned box rotated by yaw (around Y) then pitch (around X),
/// used for the bare arm's tilt.
#[allow(clippy::too_many_arguments)]
fn push_oriented_box(
    verts: &mut Vec<Vertex>,
    center: Vec3,
    hx: f32,
    hy: f32,
    hz: f32,
    yaw: f32,
    pitch: f32,
    color: [f32; 3],
) {
    let ry = glam::Mat3::from_rotation_y(yaw);
    let rx = glam::Mat3::from_rotation_x(pitch);
    let rot = rx * ry;
    let half = Vec3::new(hx, hy, hz);
    let faces: [(Vec3, Vec3, Vec3); 6] = [
        (Vec3::X, Vec3::Z, Vec3::Y),
        (Vec3::X, Vec3::Z, -Vec3::Y),
        (Vec3::Z, Vec3::Y, Vec3::X),
        (Vec3::Z, Vec3::Y, -Vec3::X),
        (Vec3::X, Vec3::Y, Vec3::Z),
        (Vec3::X, Vec3::Y, -Vec3::Z),
    ];
    for (u, v, n) in faces {
        let hu = half.dot(u.abs());
        let hv = half.dot(v.abs());
        let hn = half.dot(n.abs());
        let nu = rot * n;
        let k = face_tint(nu);
        let col = [color[0] * k, color[1] * k, color[2] * k];
        let c = center + rot * (n * hn - u * hu - v * hv);
        let du = rot * (u * (2.0 * hu));
        let dv = rot * (v * (2.0 * hv));
        let corners = [c, c + du, c + du + dv, c + dv];
        for idx in [0usize, 1, 2, 0, 2, 3] {
            verts.push(Vertex::overlay(corners[idx].to_array(), nu.to_array(), col));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::BlockType;

    fn cam() -> crate::camera::Camera {
        let mut c = crate::camera::Camera::new();
        c.pos = Vec3::new(4.0, 12.0, 7.0);
        c.yaw = 1.1;
        c.pitch = -0.2;
        c
    }

    #[test]
    fn solid_boxes_have_real_depth() {
        let mut verts = Vec::new();
        push_box(&mut verts, Vec3::ZERO, Vec3::new(0.11, 0.42, 0.17), SKIN);
        assert_eq!(verts.len(), 36);
        // The first six vertices are the +Y face. They must sit at the top
        // surface, not all collapse onto the box centre.
        assert!(verts[..6].iter().all(|v| (v.pos[1] - 0.42).abs() < 1e-6));
        assert!(verts[6..12].iter().all(|v| (v.pos[1] + 0.42).abs() < 1e-6));
        let min_x = verts.iter().map(|v| v.pos[0]).fold(f32::INFINITY, f32::min);
        let max_x = verts
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let min_z = verts.iter().map(|v| v.pos[2]).fold(f32::INFINITY, f32::min);
        let max_z = verts
            .iter()
            .map(|v| v.pos[2])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((max_x - min_x - 0.22).abs() < 1e-6);
        assert!((max_z - min_z - 0.34).abs() < 1e-6);
    }

    #[test]
    fn hand_builds_for_every_held_state() {
        let c = cam();
        let anim = HandAnim {
            swing: 0.3,
            equip: 0.0,
            walk: 0.1,
            walking: true,
        };
        for held in [
            None,
            Some(ItemType::Block(BlockType::Grass)),
            Some(ItemType::Block(BlockType::Log)),
            Some(ItemType::WoodPickaxe),
            Some(ItemType::WoodAxe),
            Some(ItemType::Stick),
        ] {
            let v = build_hand(&c, held, anim);
            assert!(!v.is_empty(), "no verts for {held:?}");
            for vert in &v {
                // Geometry must sit within ~2 blocks of the camera (it's a
                // viewmodel, not scenery).
                let d = Vec3::from(vert.pos).distance(c.pos);
                assert!(d < 2.5, "hand vert {d} blocks from camera for {held:?}");
            }
        }
    }

    #[test]
    fn held_block_carries_atlas_uvs() {
        let v = held_block_vertices(BlockType::Grass);
        assert_eq!(v.len(), 36);
        // Top face should sample the grass-top tile, not tile 0 (white).
        let top_uv = v[2].uv;
        let (lo, hi) = crate::terrain::tile_uv(crate::textures::TILE_GRASS_TOP);
        assert!(top_uv[0] >= lo[0] && top_uv[0] <= hi[0]);
        assert!(top_uv[1] >= lo[1] && top_uv[1] <= hi[1]);
        // No vertex may sample tile 0's white.
        for vert in &v {
            assert!(vert.uv[0] > 0.02 || vert.uv[1] > 0.02, "white tile used");
        }
    }
}
