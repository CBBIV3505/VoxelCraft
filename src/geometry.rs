//! Vertex format and mesh data. Terrain chunks carry UVs into the block
//! texture atlas; overlay geometry (HUD, drops, discs) uses uv (0, 0), which
//! points at the atlas's solid-white tile 0 — sampling is a no-op there.

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    /// Atlas UV. (0, 0) = solid-white tile: every overlay vertex uses it, so
    /// the texture multiplies overlay colours by white and changes nothing.
    pub uv: [f32; 2],
}

impl Vertex {
    /// Convenience constructor for overlay vertices (uv pinned to tile 0).
    pub fn overlay(pos: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> Self {
        Vertex {
            pos,
            normal,
            color,
            uv: [0.0, 0.0],
        }
    }
}

/// 24 vertices (4 per face), 36 indices, centered at origin, side length 1.
/// Per-face colors give it a Minecraft-ish nod: grass top, dirt bottom, stone sides.
#[allow(dead_code)] // kept as a minimal reference mesh / test geometry
pub fn cube_vertices() -> (Vec<Vertex>, Vec<u16>) {
    // Authored as sRGB (what the eye should see); converted to linear here
    // because the swapchain is an sRGB format — feeding raw sRGB values would
    // brighten/wash out every color.
    fn srgb_channel(c: u8) -> f32 {
        let c = c as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }
    fn srgb(r: u8, g: u8, b: u8) -> [f32; 3] {
        [srgb_channel(r), srgb_channel(g), srgb_channel(b)]
    }
    let grass = srgb(110, 168, 68); // grass green
    let dirt = srgb(134, 96, 67); // dirt brown
    let stone = srgb(125, 125, 125); // stone gray

    let mut verts: Vec<Vertex> = Vec::with_capacity(24);
    let mut indices: Vec<u16> = Vec::with_capacity(36);

    let mut add_face = |normal: [f32; 3], color: [f32; 3], corners: [[f32; 3]; 4]| {
        let base = verts.len() as u16;
        for c in corners {
            verts.push(Vertex {
                pos: c,
                normal,
                color,
                uv: [0.0, 0.0],
            });
        }
        // Two triangles: 0-1-2, 0-2-3
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    let h = 0.5;
    // +Y (top)
    add_face(
        [0.0, 1.0, 0.0],
        grass,
        [[-h, h, -h], [-h, h, h], [h, h, h], [h, h, -h]],
    );
    // -Y (bottom)
    add_face(
        [0.0, -1.0, 0.0],
        dirt,
        [[-h, -h, h], [-h, -h, -h], [h, -h, -h], [h, -h, h]],
    );
    // +X
    add_face(
        [1.0, 0.0, 0.0],
        stone,
        [[h, -h, -h], [h, h, -h], [h, h, h], [h, -h, h]],
    );
    // -X
    add_face(
        [-1.0, 0.0, 0.0],
        stone,
        [[-h, -h, h], [-h, h, h], [-h, h, -h], [-h, -h, -h]],
    );
    // +Z
    add_face(
        [0.0, 0.0, 1.0],
        stone,
        [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]],
    );
    // -Z
    add_face(
        [0.0, 0.0, -1.0],
        stone,
        [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]],
    );

    (verts, indices)
}
