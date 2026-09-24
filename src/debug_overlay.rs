//! Debug overlay (F3): the screen-space info panel and the in-world player
//! collision box + targeted-cell outline. All geometry, no text/texture
//! dependencies — the panel renders with the shared 3×5 pixel font.

use glam::Vec3;

use crate::geometry::Vertex;

/// Advance per character, in cell units (3 glyph columns + 1 gap column).
const ADVANCE: f32 = 4.0;

/// Pixel width of a string in vertical-NDC units.
fn text_width(s: &str, cell: f32) -> f32 {
    s.chars().count() as f32 * ADVANCE * cell - cell
}

/// Push one line of text, top-left at (x, y), y growing downward.
fn push_text_at(
    verts: &mut Vec<Vertex>,
    x: f32,
    y: f32,
    s: &str,
    cell: f32,
    inv: f32,
    color: [f32; 3],
) {
    let mut cx = x;
    for ch in s.chars() {
        let rows = crate::overlay::glyph_for(ch.to_ascii_uppercase());
        for (row, bits) in rows.iter().enumerate() {
            let py = y + row as f32 * cell;
            for col in 0..3 {
                if bits & (1 << (2 - col)) != 0 {
                    crate::overlay::push_quad_pub(
                        verts,
                        cx + col as f32 * cell,
                        py,
                        cx + (col + 1) as f32 * cell,
                        py + cell,
                        inv,
                        color,
                    );
                }
            }
        }
        cx += ADVANCE * cell;
    }
}

/// Builds the debug info panel as HUD fill geometry, anchored top-left.
///
/// Layout is authored in vertical-NDC units; x is squeezed by `1/aspect` at
/// emission (same scheme as the hotbar). The first `highlight` lines render
/// in the accent color (used for the performance lines).
pub fn build_debug_panel(lines: &[String], highlight: usize, aspect: f32) -> Vec<Vertex> {
    let inv = 1.0 / aspect.max(0.0001);
    const CELL: f32 = 0.0075; // glyph cell size (vertical NDC units)
                              // Line pitch, in CELL units. A glyph is 5 CELLS tall — the pitch must
                              // exceed that or consecutive lines overprint each other (1.4 produced a
                              // fully garbled panel).
    const LINE_H: f32 = 7.0;
    const MARGIN: f32 = 0.015; // panel distance from the screen edge
    const PAD: f32 = 0.011; // plate padding around the text block

    let mut fills = Vec::new();
    if lines.is_empty() {
        return fills;
    }

    let line_h = CELL * LINE_H;
    let panel_w = lines
        .iter()
        .map(|l| text_width(l, CELL))
        .fold(0.0f32, f32::max)
        + 2.0 * PAD;
    // Total text block height = (n-1) line pitches + one glyph (5 cells).
    let panel_h = (lines.len() as f32 - 1.0) * line_h + 5.0 * CELL + 2.0 * PAD;

    // Dark plate behind everything (top-left corner of the screen; Vulkan NDC
    // +y is down, so the top edge is y = -1). x coords are pre-squeezed by
    // 1/aspect here and pushed with inv = 1.0.
    crate::overlay::push_quad_pub(
        &mut fills,
        (-1.0 + MARGIN) * inv,
        -1.0 + MARGIN,
        (-1.0 + MARGIN + panel_w) * inv,
        -1.0 + MARGIN + panel_h,
        1.0,
        [0.06, 0.06, 0.08],
    );

    for (i, line) in lines.iter().enumerate() {
        let color = if i < highlight {
            [0.98, 0.93, 0.55] // accent for the first (performance) lines
        } else {
            [0.92, 0.92, 0.95]
        };
        push_text_at(
            &mut fills,
            (-1.0 + MARGIN + PAD) * inv,
            -1.0 + MARGIN + PAD + i as f32 * line_h,
            line,
            CELL,
            1.0, // x already carries the squeeze
            color,
        );
    }
    fills
}

/// Builds the in-world player collision box (a 0.6×1.8×0.6 pill-shaped AABB
/// around the eye position) as line geometry, plus the targeted cell outline
/// in a second color. Drawn by the world-space line pipeline (depth-tested,
/// so it correctly hides behind terrain).
pub fn build_collision_box(
    eye: Vec3,
    eye_height: f32,
    half_width: f32,
    targeted: Option<&[i32; 3]>,
) -> Vec<Vertex> {
    let mut verts = Vec::new();

    // Player AABB: feet = eye.y - eye_height.
    let min = Vec3::new(eye.x - half_width, eye.y - eye_height, eye.z - half_width);
    let max = Vec3::new(
        eye.x + half_width,
        eye.y - eye_height + 1.8,
        eye.z + half_width,
    );
    push_box(&mut verts, min, max, [0.95, 0.35, 0.35]); // red = player body

    // Targeted cell: bright cyan wireframe slightly inflated.
    if let Some([x, y, z]) = targeted {
        let min = Vec3::new(*x as f32 - 0.004, *y as f32 - 0.004, *z as f32 - 0.004);
        let max = Vec3::new(*x as f32 + 1.004, *y as f32 + 1.004, *z as f32 + 1.004);
        push_box(&mut verts, min, max, [0.35, 0.95, 0.95]);
    }

    verts
}

/// Push the 12 edges of an axis-aligned box.
fn push_box(verts: &mut Vec<Vertex>, min: Vec3, max: Vec3, color: [f32; 3]) {
    let c = [
        min,
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        max,
        Vec3::new(min.x, max.y, max.z),
    ];
    for (a, b) in [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0), // bottom
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4), // top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // pillars
    ] {
        for p in [c[a], c[b]] {
            verts.push(Vertex::overlay(p.to_array(), [0.0, 1.0, 0.0], color));
        }
    }
}
