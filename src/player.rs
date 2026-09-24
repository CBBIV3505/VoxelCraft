//! Player movement: walking (gravity, ground collision, jumping) and flying.
//!
//! Collision is heightfield-based: the terrain is solid below
//! `terrain_height(x, z)`, the player's feet live at `eye - EYE_HEIGHT`, and
//! horizontal moves that would climb at all are rejected (walls) — even a
//! 1-block ledge needs a jump, per the user's preference (no auto-jump).

use std::collections::HashSet;

use glam::Vec3;
use winit::keyboard::KeyCode;

use crate::camera::Camera;
use crate::world::World;

/// Eye height above the feet, in blocks (Minecraft is 1.62).
pub const EYE_HEIGHT: f32 = 1.62;
/// Horizontal walk speed in blocks/s (Minecraft default is 4.317; we run a
/// bit brisker so terrain exploration doesn't drag).
pub const WALK_SPEED: f32 = 7.0;
/// Fly speed in blocks/s (slightly faster than walking to feel useful).
pub const FLY_SPEED: f32 = 12.0;
/// Gravity in blocks/s² (Minecraft is ~32).
pub const GRAVITY: f32 = 28.0;
/// Jump takeoff velocity in blocks/s — yields a ~1.25-block jump apex.
pub const JUMP_VELOCITY: f32 = 8.5;
/// Terminal fall speed (blocks/s) — keeps tunneling bounded.
pub const MAX_FALL_SPEED: f32 = 60.0;
/// How far below the terrain the player may fall before respawning.
pub const VOID_RESPAWN_DROP: f32 = 40.0;
/// Swim: horizontal speed in water (slower than land).
pub const SWIM_SPEED: f32 = 4.2;
/// Swim: upward impulse while holding Space (against gravity).
pub const SWIM_UP: f32 = 4.5;
/// Water drag applied per second (velocity multiplier — 1 = no drag).
pub const WATER_DRAG: f32 = 0.82;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoveMode {
    Walk,
    Fly,
}

/// Per-frame movement state: consumes input, applies physics, writes the
/// result into the [`Camera`] (whose position is the *eye* position).
pub struct Player {
    pub mode: MoveMode,
    /// Vertical velocity while walking (blocks/s; 0 when grounded).
    vy: f32,
    pub grounded: bool,
}

impl Player {
    pub fn new() -> Self {
        Self {
            mode: MoveMode::Walk,
            vy: 0.0,
            grounded: false,
        }
    }

    pub fn is_flying(&self) -> bool {
        self.mode == MoveMode::Fly
    }

    pub fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            MoveMode::Walk => MoveMode::Fly,
            MoveMode::Fly => MoveMode::Walk,
        };
        self.vy = 0.0;
    }

    /// Ground height (top solid surface y) under a world position, honoring
    /// block edits.
    fn ground_height(world: &World, x: f32, z: f32, eye_y: f32) -> f32 {
        world.ground_under(x.floor() as i32, z.floor() as i32, eye_y)
    }

    /// Move the player according to held keys, physics, and collisions.
    pub fn update(&mut self, cam: &mut Camera, keys: &HashSet<KeyCode>, dt: f32, world: &World) {
        // ---- Shared look math -------------------------------------------------
        let fwd_flat = Vec3::new(cam.yaw.cos(), 0.0, -cam.yaw.sin());
        let right_flat = Vec3::new(
            (cam.yaw - std::f32::consts::FRAC_PI_2).cos(),
            0.0,
            -(cam.yaw - std::f32::consts::FRAC_PI_2).sin(),
        );
        let mut wish = Vec3::ZERO;
        if keys.contains(&KeyCode::KeyW) {
            wish += fwd_flat;
        }
        if keys.contains(&KeyCode::KeyS) {
            wish -= fwd_flat;
        }
        if keys.contains(&KeyCode::KeyD) {
            wish += right_flat;
        }
        if keys.contains(&KeyCode::KeyA) {
            wish -= right_flat;
        }
        if wish != Vec3::ZERO {
            wish = wish.normalize();
        }

        match self.mode {
            MoveMode::Fly => self.update_fly(cam, wish, keys, dt),
            MoveMode::Walk => {
                // Swimming overrides land physics whenever the EYE is below
                // the water surface at the player's column (chest-deep feels
                // right: you swim when submerged, wade when not).
                let submerged = cam.pos.y
                    < world.water_surface(
                        cam.pos.x.floor() as i32,
                        cam.pos.z.floor() as i32,
                        cam.pos.y,
                    );
                if submerged {
                    self.update_swim(cam, wish, keys, dt, world);
                } else {
                    self.update_walk(cam, wish, keys, dt, world);
                }
            }
        }

        // Safety net: if we somehow fall far below the terrain, respawn on top.
        let procedural =
            world.procedural_top(cam.pos.x.floor() as i32, cam.pos.z.floor() as i32) as f32;
        if cam.pos.y < procedural - VOID_RESPAWN_DROP {
            cam.pos.y = procedural as f32 + EYE_HEIGHT + 0.5;
            self.vy = 0.0;
        }
    }

    /// Flight: full 3D movement, Space up / Shift down, no gravity.
    fn update_fly(&mut self, cam: &mut Camera, wish: Vec3, keys: &HashSet<KeyCode>, dt: f32) {
        let mut vel = wish * FLY_SPEED;
        if keys.contains(&KeyCode::Space) {
            vel.y += FLY_SPEED;
        }
        if keys.contains(&KeyCode::ShiftLeft) {
            vel.y -= FLY_SPEED;
        }
        cam.pos += vel * dt;
        self.vy = 0.0;
        self.grounded = false;
    }

    /// Swimming: gentle gravity, Space to rise, drag on everything. The
    /// ground (lake floor) still stops the fall — you can walk out of a
    /// shallow lake bottom.
    fn update_swim(
        &mut self,
        cam: &mut Camera,
        wish: Vec3,
        keys: &HashSet<KeyCode>,
        dt: f32,
        world: &World,
    ) {
        self.grounded = false;
        // Horizontal: slower, and PER-AXIS SOLID-CHECKED. Water borders solid
        // shore terrain; an unchecked move could embed the player inside a
        // bank — then the swim/walk transition misjudges the ground and the
        // player clips through or snaps. Each axis moves only if both body
        // cells (feet + eye) stay out of solid blocks (slide along shores).
        let body_free = |x: f32, z: f32| -> bool {
            let (ix, iz) = (x.floor() as i32, z.floor() as i32);
            let feet = (cam.pos.y - EYE_HEIGHT).floor() as i32;
            let eye = cam.pos.y.floor() as i32;
            !world.solid(ix, feet, iz) && !world.solid(ix, eye, iz)
        };
        let nx = cam.pos.x + wish.x * SWIM_SPEED * dt;
        if body_free(nx, cam.pos.z) {
            cam.pos.x = nx;
        }
        let nz = cam.pos.z + wish.z * SWIM_SPEED * dt;
        if body_free(cam.pos.x, nz) {
            cam.pos.z = nz;
        }

        // Vertical: buoyant sink + Space to swim up + full drag each frame.
        let mut vy = self.vy - GRAVITY * 0.22 * dt;
        if keys.contains(&KeyCode::Space) {
            vy += SWIM_UP * dt * 6.0;
        }
        vy *= WATER_DRAG;
        self.vy = vy;
        cam.pos.y += vy * dt;

        // Lake floor stops the descent.
        let ground = world.ground_under(
            cam.pos.x.floor() as i32,
            cam.pos.z.floor() as i32,
            cam.pos.y,
        );
        let feet = cam.pos.y - EYE_HEIGHT;
        if feet <= ground {
            cam.pos.y = ground + EYE_HEIGHT;
            self.vy = 0.0;
            self.grounded = true;
        }
        // Head above water: pop out with a small boost (climbing onto shore).
        let surface = world.water_surface(
            cam.pos.x.floor() as i32,
            cam.pos.z.floor() as i32,
            cam.pos.y,
        );
        if cam.pos.y > surface && self.vy > 0.0 {
            self.vy = 2.0; // enough to clear a 1-block lip with a jump
        }
    }

    /// Walking: horizontal collision + auto-step, gravity, ground contact,
    /// jumping. The ground follows the terrain even when walking off ledges.
    fn update_walk(
        &mut self,
        cam: &mut Camera,
        wish: Vec3,
        keys: &HashSet<KeyCode>,
        dt: f32,
        world: &World,
    ) {
        // --- Horizontal movement with wall collision (no auto-step) --------
        let feet_y = cam.pos.y - EYE_HEIGHT;
        let new_x = cam.pos.x + wish.x * WALK_SPEED * dt;
        let new_z = cam.pos.z + wish.z * WALK_SPEED * dt;

        // A destination is walkable iff the surface there is NOT above the
        // current feet (any rise at all is a wall now — no auto-jump) AND the
        // body actually FITS there: the two cells the 1.8-tall body occupies
        // above the surface must be free. The fit test is what keeps tall
        // walls solid instead of snapping the player on top of them.
        let walkable_ground = |x: f32, z: f32| -> Option<f32> {
            let g = Self::ground_height(world, x, z, feet_y);
            if g > feet_y + 0.001 {
                return None; // any rise blocks the move — jump instead
            }
            let (bx, bz) = (x.floor() as i32, z.floor() as i32);
            // Body spans [g, g + 1.8): for an integer surface that's exactly
            // 2 cells — g and g+1. Checking those (not the old f+1/f+2,
            // which demanded a phantom 3rd free cell) is what makes 2-block
            // gaps walkable.
            let gf = g.floor() as i32;
            if world.solid(bx, gf, bz) || world.solid(bx, gf + 1, bz) {
                return None; // body would intersect the blocks above
            }
            Some(g)
        };

        // Try the full move first; if blocked, try each axis alone so we
        // slide along walls instead of sticking.
        let (new_x, new_z, move_ground) = if let Some(g) = walkable_ground(new_x, new_z) {
            (new_x, new_z, g)
        } else if let Some(g) = walkable_ground(new_x, cam.pos.z) {
            (new_x, cam.pos.z, g)
        } else if let Some(g) = walkable_ground(cam.pos.x, new_z) {
            (cam.pos.x, new_z, g)
        } else {
            (
                cam.pos.x,
                cam.pos.z,
                Self::ground_height(world, cam.pos.x, cam.pos.z, feet_y),
            )
        };
        cam.pos.x = new_x;
        cam.pos.z = new_z;

        // --- Vertical: gravity, ground contact, jumping ------------------------
        // move_ground is the floor under wherever we actually ended up. The
        // walkable filter guarantees it never sits above our feet (no rise is
        // step-up-able), so snapping to it can never teleport the player up.
        let ground = move_ground;
        let feet = cam.pos.y - EYE_HEIGHT;

        if keys.contains(&KeyCode::Space) && self.grounded {
            self.vy = JUMP_VELOCITY;
            self.grounded = false;
        }

        if !self.grounded {
            self.vy = (self.vy - GRAVITY * dt).max(-MAX_FALL_SPEED);
            let mut new_feet = feet + self.vy * dt;

            if new_feet <= ground {
                // Land (or stay glued) on the ground.
                new_feet = ground;
                self.vy = 0.0;
                self.grounded = true;
            }
            cam.pos.y = new_feet + EYE_HEIGHT;
        } else {
            // Grounded: stick to the terrain, but check we're still supported.
            if feet <= ground {
                cam.pos.y = ground + EYE_HEIGHT;
                self.vy = 0.0;
            } else {
                // Walked off an edge (or ground dropped away) — start falling.
                self.grounded = false;
            }
        }
    }
}
