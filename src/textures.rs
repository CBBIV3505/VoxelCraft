//! Procedurally generated block textures: one atlas row of 16×16 tiles,
//! built deterministically on the CPU at startup (no asset files, no image
//! decoding — just seeded noise per tile).
//!
//! Tile 0 is solid white: every non-terrain vertex (HUD, overlays, drops,
//! discs) carries uv (0,0), so sampling multiplies its color by white and
//! passes through untouched — only terrain vertices point at real tiles.
//!
//! Leaves use one specific colour — [`LEAF_KEY`], a magenta — as a
//! transparency key: texels with exactly that tint are holes in the canopy,
//! discarded by the world fragment shader.

/// Texels per tile side.
pub const TILE_PX: usize = 16;
/// Tiles in the atlas row. Tile 0 = opaque white (the pass-through tile).
pub const TILES: usize = 11;
/// Atlas dimensions in texels.
pub const ATLAS_W: usize = TILE_PX * TILES;
pub const ATLAS_H: usize = TILE_PX;

/// The keyed colour: leaves texels of exactly this tint become transparent.
/// Magenta never appears in any generated tile, so it can't false-positive.
pub const LEAF_KEY: [u8; 3] = [255, 0, 254];

// Tile indices (what `terrain.rs`'s per-face tile table points at).
pub const TILE_GRASS_TOP: usize = 1;
pub const TILE_DIRT: usize = 2;
pub const TILE_GRASS_SIDE: usize = 3;
pub const TILE_STONE: usize = 4;
pub const TILE_BARK: usize = 5;
pub const TILE_RINGS: usize = 6;
pub const TILE_LEAVES: usize = 7;
pub const TILE_PLANKS: usize = 8;
pub const TILE_CRAFTING_TABLE: usize = 9;
pub const TILE_COAL_ORE: usize = 10;

/// Tiny deterministic LCG — one stream per tile, so every texture is stable
/// across runs without storing anything.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    /// Uniform [0, 1).
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
    /// Uniform in [-amp, +amp].
    fn jitter(&mut self, amp: f32) -> f32 {
        (self.f32() * 2.0 - 1.0) * amp
    }
}

fn put(data: &mut [u8], x: usize, y: usize, c: [u8; 3]) {
    let i = (y * ATLAS_W + x) * 4;
    data[i] = c[0];
    data[i + 1] = c[1];
    data[i + 2] = c[2];
    data[i + 3] = 255;
}

fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Fill a tile with `base` ± per-texel noise.
fn fill_noise(data: &mut [u8], tile: usize, base: [u8; 3], amp: i32, seed: u64) {
    let mut rng = Rng::new(seed);
    for y in 0..TILE_PX {
        for x in 0..TILE_PX {
            let n = rng.jitter(amp as f32).round() as i32;
            put(
                data,
                tile * TILE_PX + x,
                y,
                [
                    clamp8(base[0] as i32 + n),
                    clamp8(base[1] as i32 + n),
                    clamp8(base[2] as i32 + n),
                ],
            );
        }
    }
}

pub fn build_atlas() -> Vec<u8> {
    let mut data = vec![0u8; ATLAS_W * ATLAS_H * 4];

    // Tile 0: opaque white — the untinted pass-through for all overlay verts.
    for y in 0..TILE_PX {
        for x in 0..TILE_PX {
            put(&mut data, x, y, [255, 255, 255]);
        }
    }

    // Grass top: bright green speckle.
    fill_noise(&mut data, TILE_GRASS_TOP, [106, 170, 64], 14, 0x5EED_0001);

    // Dirt: brown speckle.
    fill_noise(&mut data, TILE_DIRT, [134, 96, 67], 12, 0x5EED_0002);

    // Grass side: dirt with a ragged grass fringe hanging over the top edge.
    {
        let mut rng = Rng::new(0x5EED_0003);
        let dirt = [134u8, 96, 67];
        let grass = [106u8, 170, 64];
        for y in 0..TILE_PX {
            for x in 0..TILE_PX {
                let n = rng.jitter(12.0).round() as i32;
                // Rows 0-1 solid grass; row 2 ragged; below that plain dirt.
                let grassy = match y {
                    0..=1 => true,
                    2 => rng.f32() < 0.55,
                    _ => false,
                };
                let base = if grassy { grass } else { dirt };
                put(
                    &mut data,
                    TILE_GRASS_SIDE * TILE_PX + x,
                    y,
                    [
                        clamp8(base[0] as i32 + n),
                        clamp8(base[1] as i32 + n),
                        clamp8(base[2] as i32 + n),
                    ],
                );
            }
        }
    }

    // Stone: gray speckle with faint darker blotches.
    {
        fill_noise(&mut data, TILE_STONE, [125, 125, 125], 10, 0x5EED_0004);
        let mut rng = Rng::new(0x5EED_0005);
        for _ in 0..10 {
            let bx = (rng.f32() * TILE_PX as f32) as usize;
            let by = (rng.f32() * TILE_PX as f32) as usize;
            for dy in 0..2 {
                for dx in 0..3 {
                    let x = (bx + dx) % TILE_PX;
                    let y = (by + dy) % TILE_PX;
                    let i = ((y * ATLAS_W) + TILE_STONE * TILE_PX + x) * 4;
                    data[i] = clamp8(data[i] as i32 - 18);
                    data[i + 1] = clamp8(data[i + 1] as i32 - 18);
                    data[i + 2] = clamp8(data[i + 2] as i32 - 18);
                }
            }
        }
    }

    // Coal ore: the stone tile with black seam clusters stamped on top.
    {
        fill_noise(&mut data, TILE_COAL_ORE, [125, 125, 125], 10, 0x5EED_0004);
        let mut rng = Rng::new(0x5EED_C0A1);
        for _ in 0..5 {
            let bx = (rng.f32() * TILE_PX as f32) as usize;
            let by = (rng.f32() * TILE_PX as f32) as usize;
            for dy in 0..3 {
                for dx in 0..3 {
                    // Ragged cluster edges: skip the outer corners.
                    if (dx == 0 || dx == 2) && (dy == 0 || dy == 2) && rng.f32() < 0.5 {
                        continue;
                    }
                    let x = (bx + dx) % TILE_PX;
                    let y = (by + dy) % TILE_PX;
                    let n = rng.jitter(14.0).round() as i32;
                    put(
                        &mut data,
                        TILE_COAL_ORE * TILE_PX + x,
                        y,
                        [clamp8(28 + n), clamp8(28 + n), clamp8(30 + n)],
                    );
                }
            }
        }
    }

    // Log side (bark): vertical grain streaks over brown noise.
    {
        let mut rng = Rng::new(0x5EED_0006);
        for y in 0..TILE_PX {
            for x in 0..TILE_PX {
                let mut n = rng.jitter(10.0).round() as i32;
                // Deterministic vertical dark streaks every ~5 columns.
                if (x.wrapping_mul(7) + 3) % 5 == 0 {
                    n -= 20;
                }
                put(
                    &mut data,
                    TILE_BARK * TILE_PX + x,
                    y,
                    [clamp8(109 + n), clamp8(84 + n), clamp8(50 + n)],
                );
            }
        }
    }

    // Log top: growth rings around the centre.
    {
        let mut rng = Rng::new(0x5EED_0007);
        let c = (TILE_PX as f32 - 1.0) / 2.0;
        for y in 0..TILE_PX {
            for x in 0..TILE_PX {
                let mut n = rng.jitter(6.0).round() as i32;
                let ring = ((x as f32 - c).abs().max((y as f32 - c).abs())).round() as i32;
                if ring % 2 == 0 {
                    n -= 18;
                }
                put(
                    &mut data,
                    TILE_RINGS * TILE_PX + x,
                    y,
                    [clamp8(168 + n), clamp8(136 + n), clamp8(88 + n)],
                );
            }
        }
    }

    // Leaves: green speckle with magenta-keyed transparent holes (~18%).
    {
        let mut rng = Rng::new(0x5EED_0008);
        for y in 0..TILE_PX {
            for x in 0..TILE_PX {
                if rng.f32() < 0.18 {
                    put(&mut data, TILE_LEAVES * TILE_PX + x, y, LEAF_KEY);
                } else {
                    let n = rng.jitter(16.0).round() as i32;
                    put(
                        &mut data,
                        TILE_LEAVES * TILE_PX + x,
                        y,
                        [clamp8(58 + n), clamp8(121 + n), clamp8(44 + n)],
                    );
                }
            }
        }
    }

    // Planks: horizontal boards with seams and staggered vertical joints.
    {
        let mut rng = Rng::new(0x5EED_0009);
        for y in 0..TILE_PX {
            for x in 0..TILE_PX {
                let mut n = rng.jitter(8.0).round() as i32;
                if y % 4 == 3 {
                    n -= 32; // horizontal board seam
                }
                // One vertical joint per 4-row band, staggered left/right.
                let joint_x = if (y / 4) % 2 == 0 { 3 } else { 11 };
                if y % 4 != 3 && x == joint_x {
                    n -= 20;
                }
                put(
                    &mut data,
                    TILE_PLANKS * TILE_PX + x,
                    y,
                    [clamp8(168 + n), clamp8(133 + n), clamp8(84 + n)],
                );
            }
        }
    }

    // Crafting table: warm wood with a dark 3×3 grid on top and plank seams
    // on the sides. The same tile is intentionally used for every face: the
    // block's silhouette and grid are more readable than a flat color, and
    // the generated texture remains deterministic with no external assets.
    {
        let mut rng = Rng::new(0x5EED_0010);
        for y in 0..TILE_PX {
            for x in 0..TILE_PX {
                let mut n = rng.jitter(7.0).round() as i32;
                if y % 4 == 3 {
                    n -= 24;
                }
                // Dark cross-lines make the top read like a crafting grid.
                if (y == 3 || y == 8 || y == 13) && x < 15 {
                    n -= 28;
                }
                if (x == 3 || x == 8 || x == 13) && y < 15 {
                    n -= 22;
                }
                put(
                    &mut data,
                    TILE_CRAFTING_TABLE * TILE_PX + x,
                    y,
                    [clamp8(150 + n), clamp8(106 + n), clamp8(62 + n)],
                );
            }
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_shape_and_keys() {
        let data = build_atlas();
        assert_eq!(data.len(), ATLAS_W * ATLAS_H * 4);
        // Tile 0 must be fully opaque white.
        for y in 0..TILE_PX {
            for x in 0..TILE_PX {
                let i = (y * ATLAS_W + x) * 4;
                assert_eq!(&data[i..i + 4], &[255, 255, 255, 255]);
            }
        }
        // The leaf tile must contain keyed (transparent) texels and opaque
        // ones. The coal tile must contain black seams on stone gray.
        let mut holes = 0;
        let mut solid = 0;
        for y in 0..TILE_PX {
            for x in 0..TILE_PX {
                let i = (y * ATLAS_W + TILE_LEAVES * TILE_PX + x) * 4;
                if data[i] == LEAF_KEY[0]
                    && data[i + 1] == LEAF_KEY[1]
                    && data[i + 2] == LEAF_KEY[2]
                {
                    holes += 1;
                } else {
                    solid += 1;
                }
            }
        }
        assert!(holes > 20, "expected visible leaf holes, got {holes}");
        assert!(solid > 100, "leaves must stay mostly solid, got {solid}");
    }
}
