//! Floating item drops and break particles: lightweight physics state only —
//! rendering lives in overlay.rs, collection logic in app.rs.

use crate::items::ItemType;
use glam::Vec3;

/// A dropped item: a mini-cube that pops out of a broken block, falls,
/// settles, bobs/spins, and magnet-flies into the player when close.
pub struct Drop {
    pub item: ItemType,
    pub pos: Vec3,
    pub vel: Vec3,
    /// Seconds alive (drives bob/spin; INFINITY marks "collected this frame").
    pub age: f32,
}

/// One break particle: a tiny tinted billboard with gravity and a TTL.
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub ttl: f32,
    pub age: f32,
    pub color: [f32; 3],
    pub half: f32,
}
