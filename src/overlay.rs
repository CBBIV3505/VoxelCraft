//! Screen-space and world-space overlay geometry: the 3×5 pixel font,
//! crosshair + targeted-block outline, hotbar, E inventory panel, mining
//! cracks, item-drop cubes, break particles, and the sun/moon discs.
//! Everything is plain vertices — no textures, no text rendering deps.

use glam::Vec3;

use crate::geometry::Vertex;

/// Hotbar slots (one per placeable block type — the bar's width).
pub const NUM_SLOTS: usize = crate::world::BlockType::SOLID_ALL.len();
/// Backpack slots shown in the E inventory panel (3 rows × 6 columns).
pub const BACKPACK_SLOTS: usize = 18;
/// Total clickable slots in the panel: backpack + the mirrored hotbar row.
pub const PANEL_SLOTS: usize = NUM_SLOTS + BACKPACK_SLOTS;

// Shared hotbar/panel slot metrics (vertical NDC units).
const SLOT: f32 = 0.14; // slot size
const GAP: f32 = 0.02; // gap between slots
/// The UI's master scale: one hotbar slot. Everything that should scale with
/// the UI (crosshair arms, font cells) derives from this instead of a fixed
/// pixel size, so resizing the window keeps the whole HUD coherent.
pub const UI_SLOT: f32 = SLOT;

// ---------------------------------------------------------------------------
// 3×5 pixel font — glyphs are 3 columns wide, 5 rows tall; each row is 3 bits
// (bit 2 = left column). `glyph_for` returns the 5 rows for a character.
// ---------------------------------------------------------------------------

/// Rows for one glyph (5 rows × 3 bits). Unknown characters render blank.
pub fn glyph_for(ch: char) -> [u8; 5] {
    match ch {
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b011, 0b100, 0b100, 0b100, 0b011],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b010],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b111, 0b101, 0b101, 0b101, 0b101],
        'O' => [0b111, 0b101, 0b101, 0b101, 0b111],
        'P' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'Q' => [0b010, 0b101, 0b101, 0b010, 0b001],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b011, 0b100, 0b010, 0b001, 0b110],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '0' => [0b010, 0b101, 0b101, 0b101, 0b010],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b110, 0b001, 0b010, 0b100, 0b111],
        '3' => [0b110, 0b001, 0b010, 0b001, 0b110],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b110, 0b001, 0b110],
        '6' => [0b011, 0b100, 0b110, 0b101, 0b010],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b010, 0b101, 0b010, 0b101, 0b010],
        '9' => [0b010, 0b101, 0b011, 0b001, 0b110],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        ',' => [0b000, 0b000, 0b000, 0b010, 0b100],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        ';' => [0b000, 0b010, 0b000, 0b010, 0b100],
        '!' => [0b010, 0b010, 0b010, 0b000, 0b010],
        '?' => [0b110, 0b001, 0b010, 0b000, 0b010],
        '\'' => [0b010, 0b010, 0b000, 0b000, 0b000],
        '"' => [0b101, 0b101, 0b000, 0b000, 0b000],
        '(' => [0b001, 0b010, 0b010, 0b010, 0b001],
        ')' => [0b100, 0b010, 0b010, 0b010, 0b100],
        '[' => [0b011, 0b010, 0b010, 0b010, 0b011],
        ']' => [0b110, 0b010, 0b010, 0b010, 0b110],
        '/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        '\\' => [0b100, 0b100, 0b010, 0b001, 0b001],
        '|' => [0b010, 0b010, 0b010, 0b010, 0b010],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '+' => [0b000, 0b010, 0b111, 0b010, 0b000],
        '=' => [0b000, 0b111, 0b000, 0b111, 0b000],
        '*' => [0b101, 0b010, 0b101, 0b000, 0b000],
        '#' => [0b101, 0b111, 0b101, 0b111, 0b101],
        '%' => [0b101, 0b001, 0b010, 0b100, 0b101],
        '_' => [0b000, 0b000, 0b000, 0b000, 0b111],
        '<' => [0b001, 0b010, 0b100, 0b010, 0b001],
        '>' => [0b100, 0b010, 0b001, 0b010, 0b100],
        '~' => [0b000, 0b000, 0b010, 0b101, 0b000],
        '×' => [0b000, 0b101, 0b010, 0b101, 0b000],
        '°' => [0b010, 0b101, 0b000, 0b000, 0b000],
        _ => [0b000; 5],
    }
}

/// Push a filled screen-space quad. Coordinates are in vertical-NDC units
/// (x already aspect-scaled by the caller via `inv`), y grows downward.
pub fn push_quad_pub(
    verts: &mut Vec<Vertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    inv: f32,
    color: [f32; 3],
) {
    // Mirror the fill color like the terrain does (the swapchain is sRGB).
    let lin = [
        color[0] * color[0],
        color[1] * color[1],
        color[2] * color[2],
    ];
    let a = [x0 * inv, y0, 0.0];
    let b = [x1 * inv, y0, 0.0];
    let c = [x1 * inv, y1, 0.0];
    let d = [x0 * inv, y1, 0.0];
    let normal = [0.0, 0.0, 1.0];
    for p in [a, b, c, a, c, d] {
        verts.push(Vertex::overlay(p, normal, lin));
    }
}

/// World-space line as a screen-independent thin quad pair (two crossed
/// quads so it reads from all angles). Used by the crosshair/highlight.
fn push_line(verts: &mut Vec<Vertex>, a: Vec3, b: Vec3, color: [f32; 3]) {
    let lin = [
        color[0] * color[0],
        color[1] * color[1],
        color[2] * color[2],
    ];
    let dir = b - a;
    let len = dir.length().max(1e-5);
    let axis = dir / len;
    // Any perpendicular; pick the axis least aligned with `dir` for stability.
    let ref_axis = if axis.x.abs() < 0.8 {
        Vec3::X
    } else if axis.y.abs() < 0.8 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let perp1 = axis.cross(ref_axis).normalize();
    let perp2 = axis.cross(perp1).normalize();
    // Half-thickness of the crossed quads. 0.022 (4.4 cm total) so the
    // targeted-block outline reads as bold at typical view distances — the
    // old 0.012 vanished against grass/stone textures.
    const W: f32 = 0.022;
    let corners = [
        a - (perp1 + perp2) * W,
        a + (perp1 - perp2) * W,
        b + (perp1 + perp2) * W,
        b - (perp1 - perp2) * W,
    ];
    let normal = [0.0, 1.0, 0.0];
    for p in [
        corners[0], corners[1], corners[2], corners[0], corners[2], corners[3],
    ] {
        verts.push(Vertex::overlay(p.to_array(), normal, lin));
    }
}

// ---------------------------------------------------------------------------
// World overlay: crosshair + targeted-block outline
// ---------------------------------------------------------------------------

/// Screen-space crosshair: four thin quads (gap in the middle) pinned to
/// the exact screen center. Rendered through the depth-test-off HUD fill
/// pipeline — the old world-space crosshair was projected 0.5 along the
/// view ray, which drifted off the true aim point and wobbled on resize.
/// Sized relative to the hotbar slot so it scales with the rest of the UI.
pub fn build_crosshair(aspect: f32) -> Vec<Vertex> {
    let mut verts = Vec::new();
    let inv = 1.0 / aspect.max(0.0001);
    const ARM: f32 = UI_SLOT * 0.0443; // half-length of each arm
    const GAP: f32 = UI_SLOT * 0.0143; // empty space around the exact center
    const T: f32 = UI_SLOT * 0.0079; // arm thickness
    let col = [0.95, 0.95, 0.92];
    // Horizontal arms (left, then right of the gap).
    push_quad_s(&mut verts, -(GAP + ARM), -T, -GAP, T, inv, col);
    push_quad_s(&mut verts, GAP, -T, GAP + ARM, T, inv, col);
    // Vertical arms (above, then below the gap).
    push_quad_s(&mut verts, -T, -(GAP + ARM), T, -GAP, inv, col);
    push_quad_s(&mut verts, -T, GAP, T, GAP + ARM, inv, col);
    verts
}

/// Targeted-block outline, in world space (12 edges of the cell, slightly
/// inflated so the outline never z-fights with the block's faces).
pub fn build_overlay(hit_cell: Option<&crate::world::RayHit>) -> Vec<Vertex> {
    let mut verts = Vec::new();

    // Targeted block: the 12 edges of the cell, inflated ONLY along the hit
    // face's normal. (Inflating on every axis — the old box — pushed the
    // tangential edges up to a block and a half off the block, which read as
    // a badly off-center highlight.) The edges are crossed thin quads
    // (push_line) so they stay thick from any angle, unlike 1-px lines.
    if let Some(hit) = hit_cell {
        let [x, y, z] = hit.block;
        const E: f32 = 0.0035;
        let mn = Vec3::new(x as f32, y as f32, z as f32);
        let mx = mn + Vec3::ONE;
        // Face direction = the empty cell the ray came from minus the hit.
        let nrm = Vec3::new(
            (hit.adjacent[0] - hit.block[0]) as f32,
            (hit.adjacent[1] - hit.block[1]) as f32,
            (hit.adjacent[2] - hit.block[2]) as f32,
        );
        let (lo, hi) = if nrm == Vec3::ZERO {
            // Ray started inside a solid cell: inflate every way instead.
            (mn - Vec3::splat(E), mx + Vec3::splat(E))
        } else {
            (mn - nrm * E, mx + nrm * E)
        };
        let c = [0.02, 0.02, 0.025];
        let p = |i: usize, j: usize, k: usize| {
            Vec3::new(
                if i == 0 { lo.x } else { hi.x },
                if j == 0 { lo.y } else { hi.y },
                if k == 0 { lo.z } else { hi.z },
            )
        };
        // Bottom rectangle, top rectangle, vertical edges.
        for (a, b) in [
            (p(0, 0, 0), p(1, 0, 0)),
            (p(1, 0, 0), p(1, 0, 1)),
            (p(1, 0, 1), p(0, 0, 1)),
            (p(0, 0, 1), p(0, 0, 0)),
            (p(0, 1, 0), p(1, 1, 0)),
            (p(1, 1, 0), p(1, 1, 1)),
            (p(1, 1, 1), p(0, 1, 1)),
            (p(0, 1, 1), p(0, 1, 0)),
            (p(0, 0, 0), p(0, 1, 0)),
            (p(1, 0, 0), p(1, 1, 0)),
            (p(1, 0, 1), p(1, 1, 1)),
            (p(0, 0, 1), p(0, 1, 1)),
        ] {
            push_line(&mut verts, a, b, c);
        }
    }
    verts
}

// ---------------------------------------------------------------------------
// HUD helpers (screen-space, aspect-scaled)
// ---------------------------------------------------------------------------

/// push_quad_pub with the aspect inverse folded in (screen helpers all take
/// vertical-NDC coords and multiply x by `inv`).
fn push_quad_s(
    verts: &mut Vec<Vertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    inv: f32,
    color: [f32; 3],
) {
    push_quad_pub(verts, x0, y0, x1, y1, inv, color);
}

/// Outlined rectangle: 4 thin edge quads (thickness in vertical-NDC units).
fn push_quad_outline_s(
    verts: &mut Vec<Vertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    inv: f32,
    color: [f32; 3],
) {
    const T: f32 = 0.0028;
    push_quad_s(verts, x0, y0, x1, y0 + T, inv, color);
    push_quad_s(verts, x0, y1 - T, x1, y1, inv, color);
    push_quad_s(verts, x0, y0 + T, x0 + T, y1 - T, inv, color);
    push_quad_s(verts, x1 - T, y0 + T, x1, y1 - T, inv, color);
}

/// Digits 0..=4294967295 via the shared font.
fn push_digits(
    verts: &mut Vec<Vertex>,
    x: f32,
    y: f32,
    n: u32,
    cell: f32,
    inv: f32,
    color: [f32; 3],
) {
    push_text(verts, x, y, &n.to_string(), cell, inv, color);
}

/// One line of text, top-left at (x, y), y growing downward. 4 cells per
/// character (3 glyph columns + 1 gap).
fn push_text(
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
        let rows = glyph_for(ch.to_ascii_uppercase());
        for (ry, &bits) in rows.iter().enumerate() {
            for col in 0..3 {
                if bits & (1 << (2 - col)) != 0 {
                    push_quad_s(
                        verts,
                        cx + col as f32 * cell,
                        y + ry as f32 * cell,
                        cx + (col + 1) as f32 * cell,
                        y + (ry + 1) as f32 * cell,
                        inv,
                        color,
                    );
                }
            }
        }
        cx += 4.0 * cell;
    }
}

/// Pixel width of a string in vertical-NDC units (4 cells per char, minus
/// the trailing gap).
pub fn text_width(s: &str, cell: f32) -> f32 {
    (4.0 * s.chars().count() as f32 - 1.0) * cell
}

// ---------------------------------------------------------------------------
// Block icon: isometric 3-face cube (hotbar + inventory panels)
// ---------------------------------------------------------------------------

/// Isometric block icon: top, left, and right faces of a cube, each face
/// tinted for depth. Drawn with the HUD fill pipeline (4-corner quads).
fn push_block_icon(
    fills: &mut Vec<Vertex>,
    _lines: &mut Vec<Vertex>,
    cx: f32,
    cy: f32,
    s: f32,
    inv: f32,
    t: crate::world::BlockType,
) {
    let base = t.srgb();
    // Slight face shading so the icon reads as a cube.
    let top = [base[0], base[1], base[2]];
    let left = [base[0] * 0.72, base[1] * 0.72, base[2] * 0.72];
    let right = [base[0] * 0.55, base[1] * 0.55, base[2] * 0.55];
    // Isometric diamond: top face corners.
    let tw = s * 0.92; // half width of the top diamond
    let th = s * 0.46; // half height of the top diamond
    let north = (cx, cy - th);
    let east = (cx + tw, cy);
    let south = (cx, cy + th);
    let west = (cx - tw, cy);
    let h = s * 0.9; // side face height
                     // Top face (fan of two triangles).
    push_icon_quad(fills, [north, east, south, north], inv, top);
    push_icon_quad(fills, [north, south, west, north], inv, top);
    // Left face: west, south, south+h, west+h.
    push_icon_quad(
        fills,
        [west, south, (south.0, south.1 + h), (west.0, west.1 + h)],
        inv,
        left,
    );
    // Right face: south, east, east+h, south+h.
    push_icon_quad(
        fills,
        [south, east, (east.0, east.1 + h), (south.0, south.1 + h)],
        inv,
        right,
    );
}

/// Arbitrary quad from 4 points (degenerate pairs allowed for triangles).
fn push_icon_quad(verts: &mut Vec<Vertex>, pts: [(f32, f32); 4], inv: f32, color: [f32; 3]) {
    let lin = [
        color[0] * color[0],
        color[1] * color[1],
        color[2] * color[2],
    ];
    let normal = [0.0, 0.0, 1.0];
    let p = |pt: (f32, f32)| [pt.0 * inv, pt.1, 0.0];
    for q in [pts[0], pts[1], pts[2], pts[0], pts[2], pts[3]] {
        verts.push(Vertex::overlay(p(q), normal, lin));
    }
}

// ---------------------------------------------------------------------------
// Item bitmaps (tools, sticks): 5×5 cell bitmaps, one colour per item
// ---------------------------------------------------------------------------

/// One 5×5 bitmap: rows top→bottom, 1 = filled cell.
pub type Bitmap = [[u8; 5]; 5];

const PICKAXE: Bitmap = [
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
];
const AXE: Bitmap = [
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
];
const STICK: Bitmap = [
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
];

/// Draw a 5×5 item bitmap centred at (cx, cy), cell size derived from `s`.
fn push_item_bitmap(
    fills: &mut Vec<Vertex>,
    cx: f32,
    cy: f32,
    s: f32,
    inv: f32,
    item: crate::items::ItemType,
) {
    let (bitmap, color) = match item {
        crate::items::ItemType::WoodPickaxe => (PICKAXE, [0.62, 0.44, 0.26]),
        crate::items::ItemType::StonePickaxe => (PICKAXE, [0.55, 0.55, 0.58]), // gray head
        crate::items::ItemType::WoodAxe => (AXE, [0.62, 0.44, 0.26]),
        crate::items::ItemType::Stick => (STICK, [0.55, 0.40, 0.22]),
        crate::items::ItemType::Block(t) => {
            // Fallback: flat tinted square (shouldn't happen — blocks route
            // to push_block_icon).
            let b = t.srgb();
            push_quad_s(
                fills,
                cx - s * 0.6,
                cy - s * 0.6,
                cx + s * 0.6,
                cy + s * 0.6,
                inv,
                b,
            );
            return;
        }
    };
    let cell = s * 0.42;
    let x0 = cx - 2.5 * cell;
    let y0 = cy - 2.5 * cell;
    for (ry, row) in bitmap.iter().enumerate() {
        for (rx, &on) in row.iter().enumerate() {
            if on == 1 {
                push_quad_s(
                    fills,
                    x0 + rx as f32 * cell,
                    y0 + ry as f32 * cell,
                    x0 + (rx + 1) as f32 * cell,
                    y0 + (ry + 1) as f32 * cell,
                    inv,
                    color,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Hotbar
// ---------------------------------------------------------------------------

/// The bottom hotbar: one slot per placeable block type, showing item icons,
/// counts, selection outline, and the transient item-name label above.
/// Returns (fills, lines) for the two HUD pipelines.
pub fn build_hotbar(
    selected: usize,
    slots: &[(Option<crate::items::ItemType>, u32); NUM_SLOTS],
    aspect: f32,
    label: Option<&str>,
) -> (Vec<Vertex>, Vec<Vertex>) {
    let inv = 1.0 / aspect.max(0.0001);

    const SLOT_TOP: f32 = 0.82; // top edge y (bottom edge = 0.96)
    const INSET: f32 = 0.006; // backing plate inset from the slot border
    const ICON_S: f32 = 0.034; // cube half-extent (icon silhouette 2*s tall)
    const ICON_CY: f32 = SLOT_TOP + 0.048; // icon center y (above the digits)
    const CELL: f32 = 0.009; // pixel-font cell size
    const DIGITS_TOP: f32 = SLOT_TOP + 0.086; // top row of the count digits
    const LABEL_CELL: f32 = 0.011; // label font cell (slightly larger than counts)
    const LABEL_TOP: f32 = SLOT_TOP - 0.088; // label text top y

    let n = NUM_SLOTS as f32;
    let total_w = n * SLOT + (n - 1.0) * GAP;
    let mut x = -total_w / 2.0;

    let mut fills = Vec::new();
    let mut lines = Vec::new();

    // --- Selected-item label (above the bar, fades out after ~2 s) ----------
    if let Some(text) = label {
        if !text.is_empty() {
            let upper = text.to_uppercase();
            let w = text_width(&upper, LABEL_CELL);
            // Dark backing plate so the text reads on sky and terrain alike.
            push_quad_s(
                &mut fills,
                -w * 0.5 - 0.015,
                LABEL_TOP - 0.016,
                w * 0.5 + 0.015,
                LABEL_TOP + 5.0 * LABEL_CELL + 0.016,
                inv,
                [0.08, 0.08, 0.10],
            );
            push_text(
                &mut fills,
                -w * 0.5,
                LABEL_TOP,
                &upper,
                LABEL_CELL,
                inv,
                [0.98, 0.98, 0.9],
            );
        }
    }

    for (slot, &(slot_type, count)) in slots.iter().enumerate() {
        let (x0, x1, y0, y1) = (x, x + SLOT, SLOT_TOP, SLOT_TOP + SLOT);
        let cx = (x0 + x1) * 0.5;

        // Backing plate: dark, so icons and digits read on sky and terrain.
        push_quad_s(
            &mut fills,
            x0 + INSET,
            y0 + INSET,
            x1 - INSET,
            y1 - INSET,
            inv,
            [0.10, 0.10, 0.12],
        );

        if let (Some(item), 1..) = (slot_type, count as usize) {
            match item {
                crate::items::ItemType::Block(t) => {
                    push_block_icon(&mut fills, &mut lines, cx, ICON_CY, ICON_S, inv, t);
                }
                other => push_item_bitmap(&mut fills, cx, ICON_CY, ICON_S, inv, other),
            }

            // Exact count, centered under the icon.
            let nd = count.to_string().len() as f32;
            let w = (4.0 * nd - 1.0) * CELL;
            push_digits(
                &mut fills,
                cx - w * 0.5,
                DIGITS_TOP,
                count,
                CELL,
                inv,
                [0.95, 0.95, 0.95],
            );
        }

        // Slot border: light gray, visible against both the dark plate and
        // whatever the world behind it is doing (dark-on-dark was invisible).
        push_quad_outline_s(&mut lines, x0, y0, x1, y1, inv, [0.42, 0.42, 0.45]);

        // Selection indicator: bright double outline around the slot.
        if slot == selected {
            let sel = [0.98, 0.98, 0.9];
            push_quad_outline_s(
                &mut lines,
                x0 - 0.012,
                y0 - 0.012,
                x1 + 0.012,
                y1 + 0.012,
                inv,
                sel,
            );
            push_quad_outline_s(
                &mut lines,
                x0 - 0.006,
                y0 - 0.006,
                x1 + 0.006,
                y1 + 0.006,
                inv,
                sel,
            );
        }

        x += SLOT + GAP;
    }
    (fills, lines)
}

// ---------------------------------------------------------------------------
// E inventory panel: backpack grid + mirrored hotbar row
// ---------------------------------------------------------------------------

/// Slot screen rects for the E panel, in draw order (0..18 backpack
/// row-major, 18..24 hotbar) — pure geometry, shared by the panel builder
/// and the app's click hit-testing. Must match `build_inventory_panel`'s
/// centered layout.
pub fn panel_rects(_aspect: f32) -> [(f32, f32, f32, f32); PANEL_SLOTS] {
    let mut rects = [(-2.0f32, -2.0f32, -2.0f32, -2.0f32); PANEL_SLOTS];
    const ROW_PITCH: f32 = SLOT + GAP;
    let n = NUM_SLOTS as f32;
    let grid_w = n * SLOT + (n - 1.0) * GAP;
    let x_left = -grid_w * 0.5;
    let plate_top = -0.345;
    let grid_top = plate_top + 0.078;
    for s in 0..BACKPACK_SLOTS {
        let (col, row) = (s % 6, s / 6);
        let x0 = x_left + col as f32 * (SLOT + GAP);
        // Keep all three backpack rows above the separate hotbar row.
        let y0 = grid_top + row as f32 * ROW_PITCH;
        rects[s] = (x0, y0, x0 + SLOT, y0 + SLOT);
    }
    // Hotbar row below the grid, same x positions as the hotbar itself.
    let grid_bottom = grid_top + 3.0 * ROW_PITCH - GAP;
    let hb_y0 = grid_bottom + GAP * 2.0;
    for s in 0..NUM_SLOTS {
        let x0 = x_left + s as f32 * (SLOT + GAP);
        rects[BACKPACK_SLOTS + s] = (x0, hb_y0, x0 + SLOT, hb_y0 + SLOT);
    }
    rects
}

/// Crafting-grid slot rects, in row-major order. The grid is drawn in the
/// right-hand panel beside the inventory; inactive cells are left at a
/// sentinel rect so the app cannot click them.
pub fn crafting_rects(size: usize) -> [(f32, f32, f32, f32); 9] {
    let mut rects = [(-2.0f32, -2.0f32, -2.0f32, -2.0f32); 9];
    let size = size.clamp(2, 3);
    let n = NUM_SLOTS as f32;
    let grid_w = n * SLOT + (n - 1.0) * GAP;
    let x_left = -grid_w * 0.5;
    let plate_right = x_left + grid_w + 0.03;
    let rx = plate_right + 0.04;
    let cell = if size == 3 { 0.072 } else { 0.092 };
    let gap = 0.012;
    let top = -0.345 + 0.122;
    for row in 0..size {
        for col in 0..size {
            let i = row * 3 + col;
            let x0 = rx + 0.018 + col as f32 * (cell + gap);
            let y0 = top + row as f32 * (cell + gap);
            rects[i] = (x0, y0, x0 + cell, y0 + cell);
        }
    }
    rects
}

/// Which crafting cell (row-major, stride 3) contains (sx, y), if any. The
/// 2×2 grid occupies indices 0, 1, 3, 4 of the 9-slot backing array — the
/// old `.take(4)` hit-test scanned 0..4 and could never reach cell 4, which
/// is why the bottom-right 2×2 slot accepted no items. We must scan ALL 9
/// rects: the unused ones sit at the (-2,-2,-2,-2) sentinel, which no real
/// cursor position can match.
pub fn craft_cell_at(size: usize, sx: f32, y: f32) -> Option<usize> {
    let size = size.clamp(2, 3);
    crafting_rects(size)
        .iter()
        .enumerate()
        .find(|(_, &(x0, y0, x1, y1))| sx >= x0 && sx <= x1 && y >= y0 && y <= y1)
        .map(|(i, _)| i)
        .filter(|&i| i < size * size || (size == 2 && (i == 0 || i == 1 || i == 3 || i == 4)))
}

/// Output slot rect beside the current crafting grid.
pub fn crafting_output_rect(size: usize) -> (f32, f32, f32, f32) {
    let size = size.clamp(2, 3);
    let rects = crafting_rects(size);
    let first = rects[0];
    let last = rects[(size - 1) * 3 + (size - 1)];
    let cell = last.2 - last.0;
    let cy = (first.1 + last.3) * 0.5;
    (
        last.2 + 0.07,
        cy - cell * 0.5,
        last.2 + 0.07 + cell,
        cy + cell * 0.5,
    )
}

/// The centered E inventory panel: backpack grid + hotbar row on the left,
/// the active 2×2/3×3 crafting grid and output in the middle, and a compact
/// recipe menu below it. Returns (fills, lines, hover slot).
pub fn build_inventory_panel(
    hotbar: &[(Option<crate::items::ItemType>, u32); NUM_SLOTS],
    backpack: &[(Option<crate::items::ItemType>, u32); BACKPACK_SLOTS],
    held: Option<crate::items::ItemType>,
    held_count: u32,
    hover: Option<usize>,
    cursor: (f32, f32),
    aspect: f32,
    craft_grid: &[(Option<crate::items::ItemType>, u32); 9],
    craft_size: usize,
    craft_hover: Option<usize>,
    craft_output_hover: bool,
) -> (Vec<Vertex>, Vec<Vertex>, Option<usize>) {
    let inv = 1.0 / aspect.max(0.0001);
    let mut fills = Vec::new();
    let mut lines = Vec::new();

    const ROW_PITCH: f32 = SLOT + GAP;
    let n = NUM_SLOTS as f32;
    let grid_w = n * SLOT + (n - 1.0) * GAP;
    let x_left = -grid_w * 0.5;
    let plate_top = -0.345;
    let plate_bottom = plate_top + 0.078 + 3.0 * ROW_PITCH + 2.0 * GAP + SLOT + 0.05;
    let plate_right = x_left + grid_w + 0.03;

    // Inventory backing plate.
    push_quad_s(
        &mut fills,
        x_left - 0.03,
        plate_top,
        plate_right,
        plate_bottom,
        inv,
        [0.09, 0.09, 0.11],
    );
    push_quad_outline_s(
        &mut lines,
        x_left - 0.03,
        plate_top,
        plate_right,
        plate_bottom,
        inv,
        [0.35, 0.35, 0.4],
    );
    push_text(
        &mut fills,
        x_left,
        plate_top + 0.014,
        "INVENTORY",
        0.011,
        inv,
        [0.95, 0.95, 0.9],
    );

    let rects = panel_rects(aspect);
    for (s, r) in rects.iter().enumerate() {
        let (x0, y0, x1, y1) = *r;
        let stack = if s < BACKPACK_SLOTS {
            backpack[s]
        } else {
            hotbar[s - BACKPACK_SLOTS]
        };
        let hovered = hover == Some(s);
        let plate = if hovered {
            [0.24, 0.36, 0.20]
        } else {
            [0.13, 0.13, 0.16]
        };
        push_quad_s(
            &mut fills,
            x0 + 0.006,
            y0 + 0.006,
            x1 - 0.006,
            y1 - 0.006,
            inv,
            plate,
        );
        if let (Some(item), 1..) = (stack.0, stack.1 as usize) {
            match item {
                crate::items::ItemType::Block(t) => push_block_icon(
                    &mut fills,
                    &mut lines,
                    (x0 + x1) * 0.5,
                    y0 + SLOT * 0.36,
                    SLOT * 0.24,
                    inv,
                    t,
                ),
                other => push_item_bitmap(
                    &mut fills,
                    (x0 + x1) * 0.5,
                    y0 + SLOT * 0.36,
                    SLOT * 0.24,
                    inv,
                    other,
                ),
            }
            let nd = stack.1.to_string().len() as f32;
            let w = (4.0 * nd - 1.0) * 0.008;
            push_digits(
                &mut fills,
                (x0 + x1) * 0.5 - w * 0.5,
                y1 - 0.026,
                stack.1,
                0.008,
                inv,
                [0.95, 0.95, 0.95],
            );
        }
        let border = if hovered {
            [0.98, 0.98, 0.9]
        } else {
            [0.42, 0.42, 0.45]
        };
        push_quad_outline_s(&mut lines, x0, y0, x1, y1, inv, border);
    }

    // Right-hand crafting panel. Its active size is 2×2 in the ordinary
    // inventory and 3×3 when the player opened a placed crafting table.
    let craft_size = craft_size.clamp(2, 3);
    let rx = plate_right + 0.04;
    // Give the compact 2×2 layout enough breathing room and keep the table
    // layout inside the same panel.  The old fixed width clipped the heading
    // and made the grid look like it had only three cells at 16:9.
    let rw = if craft_size == 3 { 0.40 } else { 0.34 };
    push_quad_s(
        &mut fills,
        rx,
        plate_top,
        rx + rw,
        plate_bottom,
        inv,
        [0.09, 0.09, 0.11],
    );
    push_quad_outline_s(
        &mut lines,
        rx,
        plate_top,
        rx + rw,
        plate_bottom,
        inv,
        [0.35, 0.35, 0.4],
    );
    let heading = if craft_size == 3 {
        "CRAFTING TABLE"
    } else {
        "CRAFTING 2X2"
    };
    push_text(
        &mut fills,
        rx + 0.015,
        plate_top + 0.014,
        heading,
        if craft_size == 3 { 0.007 } else { 0.009 },
        inv,
        [0.95, 0.95, 0.9],
    );

    let grid_rects = crafting_rects(craft_size);
    for row in 0..craft_size {
        for col in 0..craft_size {
            let i = row * 3 + col;
            let (x0, y0, x1, y1) = grid_rects[i];
            let hovered = craft_hover == Some(i);
            push_quad_s(
                &mut fills,
                x0 + 0.004,
                y0 + 0.004,
                x1 - 0.004,
                y1 - 0.004,
                inv,
                if hovered {
                    [0.24, 0.36, 0.20]
                } else {
                    [0.13, 0.13, 0.16]
                },
            );
            if let (Some(item), 1..) = craft_grid[i] {
                match item {
                    crate::items::ItemType::Block(t) => push_block_icon(
                        &mut fills,
                        &mut lines,
                        (x0 + x1) * 0.5,
                        (y0 + y1) * 0.5,
                        (x1 - x0) * 0.22,
                        inv,
                        t,
                    ),
                    other => push_item_bitmap(
                        &mut fills,
                        (x0 + x1) * 0.5,
                        (y0 + y1) * 0.5,
                        (x1 - x0) * 0.22,
                        inv,
                        other,
                    ),
                }
                if craft_grid[i].1 > 1 {
                    push_digits(
                        &mut fills,
                        x1 - 0.028,
                        y1 - 0.027,
                        craft_grid[i].1,
                        0.007,
                        inv,
                        [0.95, 0.95, 0.95],
                    );
                }
            }
            push_quad_outline_s(
                &mut lines,
                x0,
                y0,
                x1,
                y1,
                inv,
                if hovered {
                    [0.98, 0.98, 0.9]
                } else {
                    [0.42, 0.42, 0.45]
                },
            );
        }
    }

    let out_rect = crafting_output_rect(craft_size);
    let output_hover_color = if craft_output_hover {
        [0.28, 0.40, 0.22]
    } else {
        [0.15, 0.15, 0.18]
    };
    push_quad_s(
        &mut fills,
        out_rect.0,
        out_rect.1,
        out_rect.2,
        out_rect.3,
        inv,
        output_hover_color,
    );
    push_quad_outline_s(
        &mut lines,
        out_rect.0,
        out_rect.1,
        out_rect.2,
        out_rect.3,
        inv,
        if craft_output_hover {
            [0.98, 0.98, 0.9]
        } else {
            [0.42, 0.42, 0.45]
        },
    );
    push_text(
        &mut fills,
        out_rect.0 - 0.048,
        (out_rect.1 + out_rect.3) * 0.5 - 0.01,
        ">",
        0.012,
        inv,
        [0.65, 0.75, 0.62],
    );
    if let Some(recipe) = crate::items::matching_recipe(craft_grid, craft_size, craft_size == 3) {
        let (item, count) = recipe.output;
        match item {
            crate::items::ItemType::Block(t) => push_block_icon(
                &mut fills,
                &mut lines,
                (out_rect.0 + out_rect.2) * 0.5,
                (out_rect.1 + out_rect.3) * 0.5,
                (out_rect.2 - out_rect.0) * 0.24,
                inv,
                t,
            ),
            other => push_item_bitmap(
                &mut fills,
                (out_rect.0 + out_rect.2) * 0.5,
                (out_rect.1 + out_rect.3) * 0.5,
                (out_rect.2 - out_rect.0) * 0.24,
                inv,
                other,
            ),
        }
        push_digits(
            &mut fills,
            out_rect.2 - 0.028,
            out_rect.3 - 0.027,
            count,
            0.007,
            inv,
            [0.95, 0.95, 0.95],
        );
    }

    // (The recipe discovery list was removed: crafting is grid-driven —
    // load cells by dragging or right-clicking — and the list advertised a
    // shortcut system that no longer exists.)

    // Held stack rides the cursor while moving items between inventory and grid.
    if let Some(item) = held {
        let (cx, cy) = cursor;
        match item {
            crate::items::ItemType::Block(t) => {
                push_block_icon(&mut fills, &mut lines, cx, cy, 0.02, inv, t)
            }
            other => push_item_bitmap(&mut fills, cx, cy, 0.02, inv, other),
        }
        if held_count > 1 {
            push_text(
                &mut fills,
                cx + 0.014,
                cy + 0.014,
                &held_count.to_string(),
                0.008,
                inv,
                [0.98, 0.98, 0.9],
            );
        }
    }

    (fills, lines, hover)
}

// ---------------------------------------------------------------------------
// Mining cracks: chunky filled pixel cells over the targeted block's faces
// ---------------------------------------------------------------------------// ---------------------------------------------------------------------------

/// Minecraft-style destroy overlay: a 4×4 grid of pixel cells per face,
/// activated from the centre outward as `stage` (0..=9) rises. Cell
/// activation is hashed from block coords + cell only (never the stage), so
/// the pattern grows instead of reshuffling each frame.
pub fn build_crack_overlay(x: i32, y: i32, z: i32, stage: u32) -> Vec<Vertex> {
    let mut verts = Vec::new();
    if stage == 0 {
        return verts;
    }
    let t = (stage.min(10) as f32) / 10.0;
    // Cells active at this stage: centre-out radius + noise, so ~1 cell at
    // stage 1 up to all 16 at stage 9-10.
    let max_cells = ((t * 17.0) as usize).clamp(1, 16);

    // Per-cell hash: stable across stages and frames.
    let cell_hash = |cx: i32, cy: i32| -> u32 {
        let h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ ((y as u64) << 16)
            ^ ((z as u64) << 32)
            ^ ((cx as u64) << 40)
            ^ ((cy as u64) << 48)
            ^ 0x5EED_C7AC;
        (h >> 32) as u32
    };
    // Distance from cell centre to face centre (0..1.5 in cell units).
    let cell_dist = |cx: i32, cy: i32| -> f32 {
        let dx = (cx as f32 - 1.5).abs();
        let dy = (cy as f32 - 1.5).abs();
        (dx * dx + dy * dy).sqrt()
    };

    // Rank cells by (distance, hash) — centre-out with per-cell jitter so the
    // blob is irregular.
    let mut order: Vec<(u32, i32, i32)> = Vec::new();
    for cy in 0..4 {
        for cx in 0..4 {
            let key = cell_hash(cx, cy);
            let d = (cell_dist(cx, cy) * 1000.0) as u32;
            order.push(((d / 4 << 16) | (key & 0xFFFF), cx, cy));
        }
    }
    order.sort_unstable_by_key(|(k, _, _)| *k);

    let active: std::collections::HashSet<(i32, i32)> = order
        .iter()
        .take(max_cells)
        .map(|&(_, cx, cy)| (cx, cy))
        .collect();

    // Crack colour: dark, slightly translucent-looking (opaque dark gray).
    let crack = [0.06, 0.05, 0.045];
    const PAD: f32 = 0.02; // hover off the surface
    const CELL_F: f32 = 0.25; // 4 cells per face

    // The 6 face quads, each with its own tangent frame (u, v, n) so the
    // pixel grid maps onto the face.
    let faces: [([f32; 3], [f32; 3], [f32; 3], [f32; 3]); 6] = [
        // (origin, u axis, v axis, normal) — origin at the face's (u=0, v=0)
        (
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ), // +Y
        (
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ), // -Y
        (
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ), // +X
        (
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ), // -X
        (
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ), // +Z
        (
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ), // -Z (origin at u=v=0! was [1,0,0] → cells floated one block east)
    ];
    for (o, u, v, n) in faces {
        let n_off = [n[0] as f32 * PAD, n[1] as f32 * PAD, n[2] as f32 * PAD];
        for cy in 0..4 {
            for cx in 0..4 {
                if !active.contains(&(cx, cy)) {
                    continue;
                }
                // NOTE: the face quads are built around the UNIT cube — the
                // block's world coords (x, y, z) are added at emission below.
                let u0 =
                    o[0] + u[0] * (cx as f32 * CELL_F) + v[0] * (cy as f32 * CELL_F) + n_off[0];
                let u1 =
                    o[1] + u[1] * (cx as f32 * CELL_F) + v[1] * (cy as f32 * CELL_F) + n_off[1];
                let u2 =
                    o[2] + u[2] * (cx as f32 * CELL_F) + v[2] * (cy as f32 * CELL_F) + n_off[2];
                let uu = [u[0] * CELL_F, u[1] * CELL_F, u[2] * CELL_F];
                let vv = [v[0] * CELL_F, v[1] * CELL_F, v[2] * CELL_F];
                let a = [u0, u1, u2];
                let b = [u0 + uu[0], u1 + uu[1], u2 + uu[2]];
                let c = [u0 + uu[0] + vv[0], u1 + uu[1] + vv[1], u2 + uu[2] + vv[2]];
                let d = [u0 + vv[0], u1 + vv[1], u2 + vv[2]];
                for p in [a, b, c, a, c, d] {
                    // Add the block's world position: the face frame above is
                    // unit-cube local, and without this offset every crack
                    // cube rendered at the world origin, underground.
                    verts.push(Vertex::overlay(
                        [p[0] + x as f32, p[1] + y as f32, p[2] + z as f32],
                        n,
                        crack,
                    ));
                }
            }
        }
    }
    verts
}

// ---------------------------------------------------------------------------
// Item drops + break particles
// ---------------------------------------------------------------------------

/// A spinning 3D cube for a dropped item: 8 corners, 12 triangles, per-face
/// shading from a fixed light so faces brighten/darken as it spins.
pub fn build_drop_cube(
    center: Vec3,
    size: f32,
    spin: f32,
    item: crate::items::ItemType,
) -> Vec<Vertex> {
    let mut verts = Vec::new();
    let base = match item {
        crate::items::ItemType::Block(t) => t.srgb(),
        // Tools/sticks render as brown-ish cubes (like a stick bundle).
        _ => [0.55, 0.42, 0.24],
    };
    let (s, c) = (spin.sin(), spin.cos());
    let light = Vec3::new(0.5, 0.85, 0.2).normalize();

    let h = size * 0.5;
    let corners = [
        Vec3::new(-h, -h, -h),
        Vec3::new(h, -h, -h),
        Vec3::new(h, -h, h),
        Vec3::new(-h, -h, h),
        Vec3::new(-h, h, -h),
        Vec3::new(h, h, -h),
        Vec3::new(h, h, h),
        Vec3::new(-h, h, h),
    ];
    // Rotate around Y.
    let rot = |p: Vec3| -> Vec3 { Vec3::new(p.x * c - p.z * s, p.y, p.x * s + p.z * c) + center };
    // Faces as (corner indices, outward normal pre-rotation).
    let faces: [([usize; 4], Vec3); 6] = [
        ([4, 5, 6, 7], Vec3::Y),
        ([0, 3, 2, 1], -Vec3::Y),
        ([1, 2, 6, 5], Vec3::X),
        ([0, 4, 7, 3], -Vec3::X),
        ([2, 3, 7, 6], Vec3::Z),
        ([0, 1, 5, 4], -Vec3::Z),
    ];
    for (idx, n) in faces {
        let rn = Vec3::new(n.x * c - n.z * s, n.y, n.x * s + n.z * c);
        let shade = 0.55 + 0.45 * rn.dot(light).max(0.0);
        let col = [base[0] * shade, base[1] * shade, base[2] * shade];
        let p0 = rot(corners[idx[0]]);
        let p1 = rot(corners[idx[1]]);
        let p2 = rot(corners[idx[2]]);
        let p3 = rot(corners[idx[3]]);
        for p in [p0, p1, p2, p0, p2, p3] {
            verts.push(Vertex::overlay(p.to_array(), rn.to_array(), col));
        }
    }
    verts
}

/// Billboards for break particles: small camera-facing quads. The app side
/// supplies the camera-facing basis (center/right/up/half) per particle.
pub fn build_particles(billboards: &[crate::app::ParticleBillboard]) -> Vec<Vertex> {
    let mut verts = Vec::new();
    for b in billboards {
        let lin = [
            b.color[0] * b.color[0],
            b.color[1] * b.color[1],
            b.color[2] * b.color[2],
        ];
        let c = b.center;
        let r = b.right * b.half;
        let u = b.up * b.half;
        let a = c - r - u;
        let q1 = c + r - u;
        let q2 = c + r + u;
        let q3 = c - r + u;
        let normal = [0.0, 1.0, 0.0];
        for p in [a, q1, q2, a, q2, q3] {
            verts.push(Vertex::overlay(p.to_array(), normal, lin));
        }
    }
    verts
}

// ---------------------------------------------------------------------------
// Sky gradient dome
// ---------------------------------------------------------------------------

/// Bands in the sky-gradient dome (rings of latitude). 16 rings is smooth at
/// any resolution while staying ~1.3k triangles.
pub const SKY_BANDS: usize = 16;
/// Full-circle tessellation (longitude segments per ring).
pub const SKY_SECTORS: usize = 32;

/// Triangle-list vertices for a **sky gradient dome**: a hemisphere of
/// radius `r` centered on the camera, its fragment color interpolated from
/// the horizon color at the rim to the zenith color at the top (in the
/// shader, from the vertex's normalized height). Drawn FIRST with depth test
/// off so every other element paints over it; the sun/moon discs, clouds,
/// and fog-faded terrain all land against the correct part of the gradient.
///
/// The dome is rebuilt every frame (it must follow the camera), which is a
/// 16×32 ring generation — cheap (~1.3k verts) compared to a typical frame.
pub fn build_sky_dome(camera_pos: Vec3, sky: &crate::sky::SkyState, view_dist: f32) -> Vec<Vertex> {
    let _ = sky; // colors are applied per-fragment from push constants
    let mut verts = Vec::with_capacity(SKY_BANDS * SKY_SECTORS * 6);
    // Vertex color stays white — the fragment shader derives the gradient
    // from each vertex's normalized height and the push-constant palette.
    let white = [1.0, 1.0, 1.0];
    let r = view_dist;
    for band in 0..SKY_BANDS {
        // Latitude: phi ∈ (0, π/2] — ring 0 is the horizon rim, the last
        // band closes at the apex. Slightly above 0 so the rim doesn't
        // z-fight the terrain horizon.
        let phi0 = (band as f32 / SKY_BANDS as f32) * std::f32::consts::FRAC_PI_2;
        let phi1 = ((band + 1) as f32 / SKY_BANDS as f32) * std::f32::consts::FRAC_PI_2;
        let (y0, y1) = (phi0.sin() * r, phi1.sin() * r);
        let (rad0, rad1) = (phi0.cos() * r, phi1.cos() * r);
        for s in 0..SKY_SECTORS {
            let a0 = (s as f32 / SKY_SECTORS as f32) * std::f32::consts::TAU;
            let a1 = ((s + 1) as f32 / SKY_SECTORS as f32) * std::f32::consts::TAU;
            let (c0, s0) = (a0.cos(), a0.sin());
            let (c1, s1) = (a1.cos(), a1.sin());
            // Four ring corners (radial XZ, up Y), centered on the camera.
            let p00 = [
                camera_pos.x + rad0 * c0,
                camera_pos.y + y0,
                camera_pos.z + rad0 * s0,
            ];
            let p01 = [
                camera_pos.x + rad0 * c1,
                camera_pos.y + y0,
                camera_pos.z + rad0 * s1,
            ];
            let p10 = [
                camera_pos.x + rad1 * c0,
                camera_pos.y + y1,
                camera_pos.z + rad1 * s0,
            ];
            let p11 = [
                camera_pos.x + rad1 * c1,
                camera_pos.y + y1,
                camera_pos.z + rad1 * s1,
            ];
            // Two triangles per quad (band 0's lower edge is the horizon ring;
            // the last band collapses to the apex point — degenerate tris are
            // harmless in a triangle list).
            for p in [p00, p01, p11, p00, p11, p10] {
                verts.push(Vertex::overlay(p, [0.0, 1.0, 0.0], white));
            }
        }
    }
    verts
}

// ---------------------------------------------------------------------------
// Sun/moon discs
// ---------------------------------------------------------------------------

/// Billboard quads for the sun and moon, hanging along their sky directions,
/// fading out as they approach the horizon (they *set* instead of orbiting).
pub fn build_celestial_discs(camera_pos: Vec3, sky: &crate::sky::SkyState) -> Vec<Vertex> {
    let mut verts = Vec::new();
    const DIST: f32 = 90.0;
    let entries = [
        (sky.sun_dir, sky.sun_intensity, [1.0, 0.92, 0.55], 5.5),
        (
            [-sky.sun_dir[0], -sky.sun_dir[1], -sky.sun_dir[2]],
            1.0 - sky.sun_intensity,
            [0.82, 0.86, 0.95],
            4.0,
        ),
    ];
    for (dir, intensity, color, radius) in entries {
        if intensity < 0.03 {
            continue;
        }
        let d = Vec3::from(dir);
        let elev = d.y;
        // Fade over the last ~7° of descent; gone below the horizon.
        let fade = (elev / 0.12).clamp(0.0, 1.0);
        if fade <= 0.0 {
            continue;
        }
        let col = [color[0] * fade, color[1] * fade, color[2] * fade];
        let center = camera_pos + d * DIST;
        // Billboard basis perpendicular to the view direction (the disc dir).
        let up = Vec3::Y;
        let right = up.cross(d).normalize_or_zero();
        let bill_up = d.cross(right).normalize_or_zero();
        let mut push = |p: Vec3| {
            verts.push(Vertex::overlay(p.to_array(), [0.0, 1.0, 0.0], col));
        };
        let r = right * radius;
        let u = bill_up * radius;
        let c0 = center - r - u;
        let c1 = center + r - u;
        let c2 = center + r + u;
        let c3 = center - r + u;
        push(c0);
        push(c1);
        push(c2);
        push(c0);
        push(c2);
        push(c3);
    }
    verts
}

// ---------------------------------------------------------------------------
// Clouds
// ---------------------------------------------------------------------------

/// Grid spacing between candidate cloud puffs (also the unit of the
/// CLOUD DISTANCE setting, which counts these cells around the player).
pub const CLOUD_CELL: f32 = 14.0;

/// Drifting cloud layer: scattered **3D** puffs built from hashed
/// per-cell randomness — position, footprint, altitude, and thickness all
/// vary, and bigger puffs get a second lumpy tier, so the layer reads as
/// random cumulus blocks rather than a flat grid. Shapes are stable over
/// time (hashed from cell coords, never time); the layer drifts east on
/// the wind and recenters on the camera. `radius_cells` is how many grid
/// cells around the camera the layer fills (the CLOUD DISTANCE setting).
pub fn build_clouds(
    camera_pos: Vec3,
    sky: &crate::sky::SkyState,
    time_secs: f32,
    radius_cells: i32,
) -> Vec<Vertex> {
    let mut verts = Vec::new();
    const CLOUD_Y: f32 = 42.0; // base altitude of the cloud layer
    const CELL: f32 = CLOUD_CELL; // grid spacing between candidate puffs
    const WIND: f32 = 0.5; // world units per second of eastward drift

    // Cloud brightness follows the day/night cycle.
    let bright = 0.25 + 0.75 * sky.sun_intensity.max(sky.ambient * 0.5);
    let tint = [
        0.98 * bright + 0.02,
        0.98 * bright + 0.02,
        1.0 * bright + 0.02,
    ];
    let side_col = [tint[0] * 0.88, tint[1] * 0.88, tint[2] * 0.9];
    let bot_col = [tint[0] * 0.72, tint[1] * 0.72, tint[2] * 0.75];

    // One 3D box (all 6 faces) — the puff primitive.
    #[allow(clippy::too_many_arguments)]
    fn puff(
        verts: &mut Vec<Vertex>,
        px: f32,
        pz: f32,
        y0: f32,
        y1: f32,
        hw: f32,
        hd: f32,
        top: [f32; 3],
        side: [f32; 3],
        bot: [f32; 3],
    ) {
        let (x0, x1, z0, z1) = (px - hw, px + hw, pz - hd, pz + hd);
        // +Y top / -Y bottom
        for p in [
            [x0, y1, z0],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z0],
            [x1, y1, z1],
            [x0, y1, z1],
        ] {
            verts.push(Vertex::overlay(p, [0.0, 1.0, 0.0], top));
        }
        for p in [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z1],
            [x0, y0, z0],
            [x1, y0, z1],
            [x0, y0, z1],
        ] {
            verts.push(Vertex::overlay(p, [0.0, -1.0, 0.0], bot));
        }
        // ±X sides
        for p in [
            [x1, y0, z0],
            [x1, y1, z0],
            [x1, y1, z1],
            [x1, y0, z0],
            [x1, y1, z1],
            [x1, y0, z1],
        ] {
            verts.push(Vertex::overlay(p, [1.0, 0.0, 0.0], side));
        }
        for p in [
            [x0, y0, z0],
            [x0, y1, z0],
            [x0, y1, z1],
            [x0, y0, z0],
            [x0, y1, z1],
            [x0, y0, z1],
        ] {
            verts.push(Vertex::overlay(p, [-1.0, 0.0, 0.0], side));
        }
        // ±Z sides
        for p in [
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ] {
            verts.push(Vertex::overlay(p, [0.0, 0.0, 1.0], side));
        }
        for p in [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
        ] {
            verts.push(Vertex::overlay(p, [0.0, 0.0, -1.0], side));
        }
    }

    let drift = time_secs * WIND;
    let cam_cell_x = ((camera_pos.x + drift) / CELL).floor() as i32;
    let cam_cell_z = (camera_pos.z / CELL).floor() as i32;

    for dz in -radius_cells..=radius_cells {
        for dx in -radius_cells..=radius_cells {
            let cx = cam_cell_x + dx;
            let cz = cam_cell_z + dz;
            // Hash decides whether this cell has a cloud at all (~30%).
            let h = (cx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ ((cz as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
                ^ 0xC10D_C10D;
            if (h >> 33) % 100 > 30 {
                continue;
            }
            // Every dimension is hashed from the cell coords, so no two
            // puffs share a shape, size, or altitude.
            let rnd = |k: u32, denom: u64| -> f32 {
                let v = (h.wrapping_mul(0x9E37_79B9_u64.wrapping_mul(k as u64 + 7)) >> 20) % denom;
                v as f32 / denom as f32
            };
            let px = cx as f32 * CELL - drift + rnd(1, 100) * CELL * 0.7 - CELL * 0.35;
            let pz = cz as f32 * CELL + rnd(2, 100) * CELL * 0.7 - CELL * 0.35;
            let hw = CELL * (0.28 + rnd(3, 100) * 0.30); // half-width
            let hd = CELL * (0.24 + rnd(4, 100) * 0.30); // half-depth
            let y0 = CLOUD_Y + rnd(6, 100) * 3.0; // random altitude
            let y1 = y0 + 2.5 + rnd(5, 100) * 4.0; // random thickness
            puff(&mut verts, px, pz, y0, y1, hw, hd, tint, side_col, bot_col);
            // A second stacked tier (random offset/size) lumps up big puffs.
            if rnd(7, 100) < 45.0 {
                let ox = (rnd(8, 100) * 2.0 - 1.0) * hw * 0.5;
                let oz = (rnd(9, 100) * 2.0 - 1.0) * hd * 0.5;
                puff(
                    &mut verts,
                    px + ox,
                    pz + oz,
                    y1 - 0.4,
                    y1 + 1.2 + rnd(10, 100) * 1.5,
                    hw * 0.6,
                    hd * 0.6,
                    tint,
                    side_col,
                    bot_col,
                );
            }
        }
    }
    verts
}

// ---------------------------------------------------------------------------
// Settings page (O)
// ---------------------------------------------------------------------------

/// Slider/toggle rows on the settings page, in draw order.
#[derive(Clone, Copy, PartialEq)]
pub enum SettingsRow {
    RenderDistance,
    CloudDistance,
    Shadows,
    Clouds,
    RtShadows,
    RtAo,
    GodRays,
}

pub const SETTINGS_ROW_LIST: [SettingsRow; 7] = [
    SettingsRow::RenderDistance,
    SettingsRow::CloudDistance,
    SettingsRow::Shadows,
    SettingsRow::Clouds,
    SettingsRow::RtShadows,
    SettingsRow::RtAo,
    SettingsRow::GodRays,
];

/// The O settings page: render-distance slider + toggles. Returns
/// (fills, lines, hovered row).
/// Row count as a plain usize (app.rs sizes its rect array with this).
pub const SETTINGS_ROWS: usize = SETTINGS_ROW_LIST.len();

/// Settings-page geometry constants (shared with app.rs hit-testing). The
/// panel is a wide centered dialog: labels on the left, controls (slider /
/// checkbox) in a middle column, values right-aligned — long labels can
/// never run into the controls again.
pub const SETTINGS_LEFT: f32 = -0.56;
pub const SETTINGS_TOP: f32 = -0.285;
pub const SETTINGS_W: f32 = 1.12;
pub const SETTINGS_ROW_H: f32 = 0.075;
/// Header strip above the first row (title + close hint).
pub const SETTINGS_HEADER: f32 = 0.09;
/// Control column x (slider track / checkbox left edge).
pub const SETTINGS_CTRL_X: f32 = SETTINGS_LEFT + 0.56;
/// Slider track width.
pub const SETTINGS_TRACK_W: f32 = 0.18;

/// Row rects for hit-testing, in SETTINGS_ROW_LIST order (full-width rows
/// below the header, matching the hover highlight exactly).
pub fn settings_rects() -> [(f32, f32, f32, f32); SETTINGS_ROWS] {
    let mut rects = [(-2.0f32, -2.0f32, -2.0f32, -2.0f32); SETTINGS_ROWS];
    for (i, r) in rects.iter_mut().enumerate() {
        let y = SETTINGS_TOP + SETTINGS_HEADER + i as f32 * SETTINGS_ROW_H;
        *r = (
            SETTINGS_LEFT,
            y,
            SETTINGS_LEFT + SETTINGS_W,
            y + SETTINGS_ROW_H,
        );
    }
    rects
}

/// If the cursor (panel space: x already aspect-scaled, y vertical-NDC down)
/// is over a SLIDER row's track + margin, return that row so the app can
/// highlight it and start a drag there. Rows 0 (render distance) and 1 (cloud
/// distance) are the sliders; their geometry must match build_settings_page.
pub fn slider_hover(hover: Option<usize>, sx: f32, y: f32) -> Option<usize> {
    let row = hover?;
    if row != 0 && row != 1 {
        return None;
    }
    let rect = settings_rects()[row];
    let (x0, y0, x1, y1) = rect;
    if sx < x0 || sx > x1 || y < y0 || y > y1 {
        return None;
    }
    // Inside the row, but only track±margin counts as grabbing the slider.
    const MARGIN: f32 = 0.05;
    let tx0 = SETTINGS_CTRL_X - MARGIN;
    let tx1 = SETTINGS_CTRL_X + SETTINGS_TRACK_W + MARGIN;
    (sx >= tx0 && sx <= tx1).then_some(row)
}

pub fn build_settings_page(
    settings: &crate::settings::Settings,
    hover: Option<usize>,
    aspect: f32,
) -> (Vec<Vertex>, Vec<Vertex>) {
    let inv = 1.0 / aspect.max(0.0001);
    let mut fills = Vec::new();
    let mut lines = Vec::new();

    const W: f32 = SETTINGS_W;
    const ROW_H: f32 = SETTINGS_ROW_H;
    const TOP: f32 = SETTINGS_TOP;
    const LEFT: f32 = SETTINGS_LEFT;

    // Plate: title strip + one strip per row + bottom padding.
    let rows = SETTINGS_ROW_LIST.len() as f32;
    let bottom = TOP + SETTINGS_HEADER + rows * ROW_H + 0.025;
    push_quad_s(
        &mut fills,
        LEFT,
        TOP,
        LEFT + W,
        bottom,
        inv,
        [0.09, 0.09, 0.11],
    );
    push_quad_outline_s(
        &mut lines,
        LEFT,
        TOP,
        LEFT + W,
        bottom,
        inv,
        [0.35, 0.35, 0.4],
    );
    // Title strip (separated from the rows by a divider line).
    push_quad_s(
        &mut fills,
        LEFT,
        TOP,
        LEFT + W,
        TOP + SETTINGS_HEADER,
        inv,
        [0.13, 0.13, 0.16],
    );
    push_quad_s(
        &mut fills,
        LEFT,
        TOP + SETTINGS_HEADER - 0.0022,
        LEFT + W,
        TOP + SETTINGS_HEADER,
        inv,
        [0.3, 0.3, 0.36],
    );
    push_text(
        &mut fills,
        LEFT + 0.02,
        TOP + 0.012,
        "SETTINGS",
        0.012,
        inv,
        [0.95, 0.95, 0.9],
    );
    let hint = "ESC TO CLOSE";
    let hint_w = hint.chars().count() as f32 * 4.0 * 0.0075;
    push_text(
        &mut fills,
        LEFT + W - hint_w - 0.02,
        TOP + 0.014,
        hint,
        0.0075,
        inv,
        [0.55, 0.55, 0.6],
    );

    for (i, row) in SETTINGS_ROW_LIST.iter().enumerate() {
        let y = TOP + SETTINGS_HEADER + i as f32 * ROW_H;
        let hovered = hover == Some(i);
        if hovered {
            push_quad_s(
                &mut fills,
                LEFT + 0.01,
                y,
                LEFT + W - 0.01,
                y + ROW_H,
                inv,
                [0.15, 0.15, 0.19],
            );
        }
        let enabled = match row {
            SettingsRow::RenderDistance => true,
            SettingsRow::CloudDistance => true,
            SettingsRow::Shadows => true,
            SettingsRow::Clouds => true,
            _ => false, // future shader-pack rows
        };
        let text_col = if enabled {
            [0.9, 0.9, 0.86]
        } else {
            [0.4, 0.4, 0.42]
        };

        // Left column: the label only (never extends past the controls).
        let label = match row {
            SettingsRow::RenderDistance => "RENDER DISTANCE".to_string(),
            SettingsRow::CloudDistance => "CLOUD DISTANCE".to_string(),
            SettingsRow::Shadows => "SHADOWS".to_string(),
            SettingsRow::Clouds => "CLOUDS".to_string(),
            SettingsRow::RtShadows => "RT SHADOWS".to_string(),
            SettingsRow::RtAo => "RT AMB. OCCLUSION".to_string(),
            SettingsRow::GodRays => "GOD RAYS".to_string(),
        };
        push_text(
            &mut fills,
            LEFT + 0.02,
            y + 0.014,
            &label,
            0.010,
            inv,
            text_col,
        );

        // Right column: value text, right-aligned at the panel edge.
        let value = match row {
            SettingsRow::RenderDistance => format!("{} CHUNKS", settings.render_distance),
            SettingsRow::CloudDistance => format!("{} CELLS", settings.cloud_distance),
            SettingsRow::Shadows => {
                if settings.shadows {
                    "ON".to_string()
                } else {
                    "OFF".to_string()
                }
            }
            SettingsRow::Clouds => {
                if settings.clouds {
                    "ON".to_string()
                } else {
                    "OFF".to_string()
                }
            }
            _ => "SOON".to_string(),
        };
        let vw = value.chars().count() as f32 * 4.0 * 0.008;
        push_text(
            &mut fills,
            LEFT + W - vw - 0.02,
            y + 0.016,
            &value,
            0.008,
            inv,
            text_col,
        );

        match row {
            SettingsRow::RenderDistance | SettingsRow::CloudDistance => {
                // Middle column: slider track + filled portion + knob.
                let tx = SETTINGS_CTRL_X;
                let tw = SETTINGS_TRACK_W;
                let ty = y + ROW_H * 0.5 - 0.004;
                push_quad_s(
                    &mut fills,
                    tx,
                    ty,
                    tx + tw,
                    ty + 0.008,
                    inv,
                    [0.25, 0.25, 0.3],
                );
                let (lo, hi, v) = match row {
                    SettingsRow::RenderDistance => {
                        (1.0f32, 32.0f32, settings.render_distance as f32)
                    }
                    _ => (2.0, 10.0, settings.cloud_distance as f32),
                };
                let frac = (v - lo) / (hi - lo);
                push_quad_s(
                    &mut fills,
                    tx,
                    ty,
                    tx + tw * frac,
                    ty + 0.008,
                    inv,
                    [0.55, 0.85, 0.45],
                );
                let kx = tx + tw * frac;
                push_quad_s(
                    &mut fills,
                    kx - 0.008,
                    ty - 0.006,
                    kx + 0.008,
                    ty + 0.014,
                    inv,
                    [0.95, 0.95, 0.9],
                );
            }
            SettingsRow::Shadows | SettingsRow::Clouds => {
                // Middle column: checkbox (box + check when on).
                let bx = SETTINGS_CTRL_X;
                let by = y + ROW_H * 0.5 - 0.013;
                const BS: f32 = 0.026;
                push_quad_s(
                    &mut fills,
                    bx,
                    by,
                    bx + BS,
                    by + BS,
                    inv,
                    [0.16, 0.16, 0.19],
                );
                let border = if hovered {
                    [0.98, 0.98, 0.9]
                } else {
                    [0.42, 0.42, 0.45]
                };
                push_quad_outline_s(&mut lines, bx, by, bx + BS, by + BS, inv, border);
                let on = match row {
                    SettingsRow::Shadows => settings.shadows,
                    _ => settings.clouds,
                };
                if on {
                    push_quad_s(
                        &mut fills,
                        bx + 0.006,
                        by + BS * 0.5,
                        bx + BS * 0.38,
                        by + BS * 0.5 + 0.005,
                        inv,
                        [0.55, 0.85, 0.45],
                    );
                    push_quad_s(
                        &mut fills,
                        bx + BS * 0.38,
                        by + BS * 0.5,
                        bx + BS - 0.006,
                        by + BS * 0.14,
                        inv,
                        [0.55, 0.85, 0.45],
                    );
                    push_quad_s(
                        &mut fills,
                        bx + BS * 0.38,
                        by + BS * 0.5,
                        bx + BS - 0.006,
                        by + BS * 0.5 - 0.005,
                        inv,
                        [0.55, 0.85, 0.45],
                    );
                    push_quad_s(
                        &mut fills,
                        bx + 0.006,
                        by + BS * 0.5 - 0.005,
                        bx + BS * 0.38,
                        by + BS * 0.5,
                        inv,
                        [0.55, 0.85, 0.45],
                    );
                }
            }
            _ => {}
        }
    }
    (fills, lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every crack-overlay vertex must sit within the block's ±PAD volume —
    /// a regression guard for the -Z face whose origin was once [1,0,0],
    /// floating its crack cells a full block east of the mined block.
    #[test]
    fn crack_overlay_stays_inside_the_block() {
        let (x, y, z) = (12i32, -3i32, 47i32);
        for stage in 1..=10u32 {
            for v in build_crack_overlay(x, y, z, stage) {
                let p = v.pos;
                assert!(
                    p[0] >= x as f32 - 0.03 && p[0] <= x as f32 + 1.03,
                    "x {p:?}"
                );
                assert!(
                    p[1] >= y as f32 - 0.03 && p[1] <= y as f32 + 1.03,
                    "y {p:?}"
                );
                assert!(
                    p[2] >= z as f32 - 0.03 && p[2] <= z as f32 + 1.03,
                    "z {p:?}"
                );
            }
        }
    }

    /// The cloud layer's extent follows the CLOUD DISTANCE setting: doubling
    /// the radius strictly grows the vertex count (more cells generated).
    #[test]
    fn cloud_distance_setting_grows_the_layer() {
        let sky = crate::sky::sample(0.3); // mid-morning
        let small = build_clouds(Vec3::ZERO, &sky, 0.0, 2);
        let big = build_clouds(Vec3::ZERO, &sky, 0.0, 6);
        assert!(!small.is_empty());
        assert!(
            big.len() > small.len(),
            "radius 6 ({}) ≤ radius 2 ({})",
            big.len(),
            small.len()
        );
    }
}
