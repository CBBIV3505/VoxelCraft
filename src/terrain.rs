//! Procedural terrain meshing: hidden-face culling + per-vertex ambient
//! occlusion + per-block color noise, meshed per 16×16 chunk.
//!
//! Solidity and block types come from the [`World`] overlay: mined cells are
//! skipped, placed blocks are meshed even above the procedural heightmap.
//! Every solid block is colored by its type (grass/dirt/stone stratification)
//! Trees come from the analytic world definition — trunk/leaf blocks are
//! materialized per chunk-mesh with no storage.

/// Blocks per chunk side.
pub const CHUNK_SIZE: i32 = 16;
/// Default view radius in chunks around the camera (the settings page's
/// render-distance slider overrides this at runtime; range 1–32).
pub const VIEW_RADIUS: i32 = 4;
/// Highest block y ever meshed. The world is finite vertically.
pub const MAX_Y: i32 = 48;

use std::collections::HashMap;

use crate::geometry::Vertex;
use crate::textures;
use crate::world::{Block, BlockType, World};

// ---------------------------------------------------------------------------
// Atlas UV mapping
// ---------------------------------------------------------------------------

/// Padding fraction kept inside each tile's border. Nearest sampling snaps
/// to texel centres, so a hairline inset keeps neighbouring-tile bleed out.
const TILE_INSET: f32 = 0.5 / crate::textures::TILE_PX as f32;

/// UV rect of one atlas tile, inset by [`TILE_INSET`] on every side.
pub fn tile_uv(tile: usize) -> ([f32; 2], [f32; 2]) {
    let x0 = tile as f32 / textures::TILES as f32;
    let x1 = (tile + 1) as f32 / textures::TILES as f32;
    (
        [x0 + TILE_INSET / textures::TILES as f32, TILE_INSET],
        [x1 - TILE_INSET / textures::TILES as f32, 1.0 - TILE_INSET],
    )
}

/// Map a face corner (unit cube, values 0/1) to a UV inside `tile`'s rect.
/// Side faces run u along the horizontal tangent and v along Y (grain stays
/// vertical); top/bottom faces use x→u, z→v.
fn corner_uv(tile: usize, corner: [f32; 3], neighbor: [i32; 3], jitter: f32) -> [f32; 2] {
    let (lo, hi) = tile_uv(tile);
    let (ua, va) = tangent_axes(neighbor);
    // Side faces run V along +Y so upright tile content (the grass-side
    // fringe, bark grain, plank seams) sits at the TOP of the block face.
    // Corner Y is 0 at the bottom and 1 at the top, while the atlas image's
    // row 0 is its top row — so V must be flipped on side faces or every
    // upright texture renders upside down.
    let (u_axis, v_axis, v_flip) = match (ua, va) {
        (0, 2) => (0, 2, false), // top/bottom: u=x, v=z (no upright content)
        (1, 2) => (2, 1, true),  // x faces: u=z, v=y (flipped)
        _ => (0, 1, true),       // z faces: u=x, v=y (flipped)
    };
    let u = lo[0] + (hi[0] - lo[0]) * corner[u_axis];
    let v_raw = lo[1] + (hi[1] - lo[1]) * corner[v_axis];
    let v = if v_flip {
        hi[1] - (v_raw - lo[1])
    } else {
        v_raw
    };
    // Per-block jitter: a sub-texel UV slide (well inside the tile's inset),
    // giving each block a whisper of deterministic tint variation.
    let jx = (jitter - 1.0) / textures::TILES as f32 * 0.5;
    [u + jx, v]
}

/// Which atlas tile each face of a block uses: [top, bottom, +X, -X, +Z, -Z]
/// matching the FACES order (Y, -Y, X, -X, Z, -Z).
fn face_tiles(t: BlockType, face_idx: usize) -> usize {
    use crate::textures as tx;
    let (top, bottom, side) = match t {
        BlockType::Grass => (tx::TILE_GRASS_TOP, tx::TILE_DIRT, tx::TILE_GRASS_SIDE),
        BlockType::Dirt => (tx::TILE_DIRT, tx::TILE_DIRT, tx::TILE_DIRT),
        BlockType::Stone => (tx::TILE_STONE, tx::TILE_STONE, tx::TILE_STONE),
        BlockType::Log => (tx::TILE_RINGS, tx::TILE_RINGS, tx::TILE_BARK),
        BlockType::Leaves => (tx::TILE_LEAVES, tx::TILE_LEAVES, tx::TILE_LEAVES),
        BlockType::Planks => (tx::TILE_PLANKS, tx::TILE_PLANKS, tx::TILE_PLANKS),
        BlockType::CraftingTable => (
            tx::TILE_CRAFTING_TABLE,
            tx::TILE_CRAFTING_TABLE,
            tx::TILE_CRAFTING_TABLE,
        ),
        BlockType::CoalOre => (tx::TILE_COAL_ORE, tx::TILE_COAL_ORE, tx::TILE_COAL_ORE),
        // Water meshes its own geometry (translucent pass) and is never a
        // held item, so it just gets the pass-through tile.
        BlockType::Water => (0, 0, 0),
        BlockType::Air => (0, 0, 0),
    };
    match face_idx {
        0 => top,
        1 => bottom,
        _ => side,
    }
}

/// Public re-exports for the first-person hand (src/hand.rs), which builds
/// its own textured cube using the same tile/axis conventions as terrain.
pub mod tiles_helpers {
    /// Which atlas tile a held block's face uses (face order matches
    /// terrain's FACES: [+Y, -Y, +X, -X, +Z, -Z]).
    pub fn face_tiles_for_held(t: crate::world::BlockType, face_idx: usize) -> usize {
        super::face_tiles(t, face_idx)
    }
}

// ---------------------------------------------------------------------------
// Deterministic value noise + fBm (heightmap shape)
// ---------------------------------------------------------------------------

/// Splitmix64 finalizer — fast, well-distributed integer hash.
fn hash_u64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic pseudo-random in [0, 1) for an integer lattice point.
fn rand01(x: i32, z: i32, seed: u64) -> f32 {
    let h = hash_u64((x as u64) << 32 ^ (z as u64 & 0xFFFF_FFFF) ^ seed.wrapping_mul(0x1234_5678));
    ((h >> 40) as f32) / (1u64 << 24) as f32
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t) // classic smoothstep fade
}

/// 2D value noise on the integer lattice, output [0, 1].
fn value_noise(x: f32, z: f32, seed: u64) -> f32 {
    let xi = x.floor() as i32;
    let zi = z.floor() as i32;
    let tx = smooth(x - x.floor());
    let tz = smooth(z - z.floor());

    let a = rand01(xi, zi, seed);
    let b = rand01(xi + 1, zi, seed);
    let c = rand01(xi, zi + 1, seed);
    let d = rand01(xi + 1, zi + 1, seed);

    a + (b - a) * tx + (c - a) * tz + (a - b - c + d) * tx * tz
}

/// Fractal Brownian motion over value noise — the heightmap.
/// Returns the procedural column height (top solid block y + 1).
pub fn terrain_height(x: i32, z: i32, seed: u64) -> i32 {
    const BASE_HEIGHT: f32 = 8.0;
    const AMPLITUDE: f32 = 10.0;

    let fx = x as f32;
    let fz = z as f32;
    // 4 octaves: broad hills + medium bumps + fine detail
    let noise = value_noise(fx * 0.015, fz * 0.015, seed) * 0.55
        + value_noise(fx * 0.04, fz * 0.04, seed ^ 0x1111) * 0.28
        + value_noise(fx * 0.09, fz * 0.09, seed ^ 0x2222) * 0.12
        + value_noise(fx * 0.2, fz * 0.2, seed ^ 0x3333) * 0.05;

    (BASE_HEIGHT + noise * AMPLITUDE).round() as i32
}

// ---------------------------------------------------------------------------
// Per-block color noise
// ---------------------------------------------------------------------------

/// Deterministic per-block brightness jitter: same block, same tint, every
/// rebuild. Range ±JITTER around 1.0.
const COLOR_JITTER: f32 = 0.06;

fn block_jitter(x: i32, y: i32, z: i32, seed: u64) -> f32 {
    let h = hash_u64(
        (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ (z as u64).wrapping_mul(0x1656_67C1_9A37_D1F9)
            ^ seed,
    );
    let r = ((h >> 40) as f32) / (1u64 << 24) as f32; // [0, 1)
    1.0 + (r * 2.0 - 1.0) * COLOR_JITTER
}

// ---------------------------------------------------------------------------
// Face table
// ---------------------------------------------------------------------------

/// Brightness for AO levels 0 (fully occluded corner) .. 3 (open corner).
const AO_CURVE: [f32; 4] = [0.45, 0.68, 0.85, 1.0];

struct Face {
    /// 4 corner offsets (unit cube, block at origin), CCW seen from outside.
    corners: [[f32; 3]; 4],
    normal: [f32; 3],
    /// Integer normal — the neighbor direction this face is visible toward.
    neighbor: [i32; 3],
    /// Face tint (Minecraft-style axis shading): top 1.0, bottom 0.55,
    /// sides distinct per axis.
    tint: f32,
}

const FACES: [Face; 6] = [
    // +Y top
    Face {
        corners: [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        normal: [0.0, 1.0, 0.0],
        neighbor: [0, 1, 0],
        tint: 1.0,
    },
    // -Y bottom
    Face {
        corners: [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        normal: [0.0, -1.0, 0.0],
        neighbor: [0, -1, 0],
        tint: 0.55,
    },
    // +X
    Face {
        corners: [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        normal: [1.0, 0.0, 0.0],
        neighbor: [1, 0, 0],
        tint: 0.86,
    },
    // -X
    Face {
        corners: [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        normal: [-1.0, 0.0, 0.0],
        neighbor: [-1, 0, 0],
        tint: 0.86,
    },
    // +Z
    Face {
        corners: [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        normal: [0.0, 0.0, 1.0],
        neighbor: [0, 0, 1],
        tint: 0.72,
    },
    // -Z
    Face {
        corners: [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        normal: [0.0, 0.0, -1.0],
        neighbor: [0, 0, -1],
        tint: 0.72,
    },
];

/// The two axes perpendicular to a face normal (the face's tangent plane).
/// Exactly one neighbor component is ±1; the two zero components are the
/// tangent axes. Arm order matters: `[0, 0, _]` must be tested before
/// `[0, _, 0]`, or Z faces fall through to the Y arm — which made their V
/// come from the constant Z axis and stretched one texel row over the face.
pub fn tangent_axes(neighbor: [i32; 3]) -> (usize, usize) {
    match neighbor {
        [0, 0, _] => (0, 1), // normal along Z → tangents X and Y
        [0, _, 0] => (0, 2), // normal along Y → tangents X and Z
        _ => (1, 2),         // normal along X → tangents Y and Z
    }
}

// ---------------------------------------------------------------------------
// Chunk meshing
// ---------------------------------------------------------------------------

/// Mesh one CHUNK_SIZE × CHUNK_SIZE chunk whose corner (minimum-x, minimum-z
/// block) is at (chunk_x, chunk_z) in world block coordinates.
pub fn mesh_chunk(
    world: &World,
    chunk_x: i32,
    chunk_z: i32,
) -> (Vec<Vertex>, Vec<u16>, Vec<Vertex>, Vec<u16>) {
    let seed = world.seed;
    let mut verts: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();

    // Height ring covering the chunk ±2 (AO/cull queries reach 2 blocks out);
    // procedural solidity comes from this cache, edits from the world overlay
    // — so the expensive fBm is evaluated exactly once per column.
    let ring = (CHUNK_SIZE + 4) as usize;
    let mut heights = vec![0i32; ring * ring];
    for dx in 0..ring {
        for dz in 0..ring {
            heights[dx * ring + dz] =
                terrain_height(chunk_x + dx as i32 - 2, chunk_z + dz as i32 - 2, seed);
        }
    }
    let proc_solid = |wx: i32, y: i32, wz: i32| -> bool {
        let dx = (wx - chunk_x + 2).clamp(0, ring as i32 - 1) as usize;
        let dz = (wz - chunk_z + 2).clamp(0, ring as i32 - 1) as usize;
        y < heights[dx * ring + dz]
    };
    let proc_top = |wx: i32, wz: i32| -> i32 {
        let dx = (wx - chunk_x + 2).clamp(0, ring as i32 - 1) as usize;
        let dz = (wz - chunk_z + 2).clamp(0, ring as i32 - 1) as usize;
        heights[dx * ring + dz]
    };

    // Materialize the analytic trees that reach this chunk (one HashMap built
    // per mesh; all tree queries hit it instead of recomputing hashes).
    let mut trees: HashMap<[i32; 3], BlockType> = HashMap::new();
    world.collect_tree_blocks(
        chunk_x,
        chunk_x + CHUNK_SIZE - 1,
        chunk_z,
        chunk_z + CHUNK_SIZE - 1,
        &mut trees,
    );
    let tree_cell = |wx: i32, y: i32, wz: i32| -> Option<BlockType> {
        if trees.is_empty() {
            return None;
        }
        trees.get(&[wx, y, wz]).copied()
    };
    // How high the tallest tree block reaches (chunk-local scan ceiling).
    let tree_top = trees.keys().map(|k| k[1]).max().unwrap_or(0);
    let proc_max = *heights.iter().max().unwrap();

    for lx in 0..CHUNK_SIZE {
        for lz in 0..CHUNK_SIZE {
            let wx = chunk_x + lx;
            let wz = chunk_z + lz;

            // Column scan ceiling: procedural top (or tallest tree), raised to
            // MAX_Y when edits exist so placed towers mesh at any height.
            // Mined cells are skipped by their Air overrides below.
            let col_top = heights[((lx + 2) as usize) * ring + (lz + 2) as usize];
            let top = if world.has_edits() {
                MAX_Y
            } else {
                col_top.max(tree_top + 1).max(proc_max)
            };
            for y in 0..top {
                // Edit override first (it must win over everything — mining a
                // tree block writes Air here), else analytic tree, else the
                // procedural stratification.
                let block_type = match world.override_at(wx, y, wz) {
                    Some(BlockType::Air) => continue, // mined — nothing to mesh
                    Some(t) => t,
                    None => match tree_cell(wx, y, wz) {
                        Some(t) => t,
                        None => {
                            if !proc_solid(wx, y, wz) {
                                continue;
                            }
                            let h = proc_top(wx, wz);
                            if y == h - 1 {
                                BlockType::Grass
                            } else if y >= h - 4 {
                                BlockType::Dirt
                            } else {
                                BlockType::Stone
                            }
                        }
                    },
                };

                let base_color = block_type.srgb();
                let jitter = block_jitter(wx, y, wz, seed);

                for (face_idx, face) in FACES.iter().enumerate() {
                    let nx = wx + face.neighbor[0];
                    let ny = y + face.neighbor[1];
                    let nz = wz + face.neighbor[2];
                    // Cull faces buried inside solid terrain (edits first,
                    // then trees, then the procedural ground).
                    let neighbor_type = match world.override_at(nx, ny, nz) {
                        Some(t) => Some(t),
                        None => tree_cell(nx, ny, nz),
                    };
                    let neighbor_solid = match neighbor_type {
                        Some(t) => t != BlockType::Air,
                        None => proc_solid(nx, ny, nz),
                    };
                    if neighbor_solid
                        && !(block_type == BlockType::Leaves
                            && neighbor_type == Some(BlockType::Leaves))
                    {
                        continue;
                    }

                    // Which atlas tile this face uses.
                    let tile = face_tiles(block_type, face_idx);

                    // --- Per-vertex ambient occlusion ----------------------
                    let (ua, va) = tangent_axes(face.neighbor);
                    let mut ao_levels = [3u32; 4];
                    for (i, c) in face.corners.iter().enumerate() {
                        let su = (c[ua] as i32) * 2 - 1;
                        let sv = (c[va] as i32) * 2 - 1;
                        let mut s1_off = [0i32; 3];
                        s1_off[ua] = su;
                        let mut s2_off = [0i32; 3];
                        s2_off[va] = sv;

                        let solid_q = |qx: i32, qy: i32, qz: i32| -> u32 {
                            match world.override_at(qx, qy, qz) {
                                Some(t) => (t != BlockType::Air) as u32,
                                None => {
                                    if tree_cell(qx, qy, qz).is_some() {
                                        1
                                    } else {
                                        proc_solid(qx, qy, qz) as u32
                                    }
                                }
                            }
                        };
                        let s1 = solid_q(nx + s1_off[0], ny + s1_off[1], nz + s1_off[2]);
                        let s2 = solid_q(nx + s2_off[0], ny + s2_off[1], nz + s2_off[2]);
                        let corner = solid_q(
                            nx + s1_off[0] + s2_off[0],
                            ny + s1_off[1] + s2_off[1],
                            nz + s1_off[2] + s2_off[2],
                        );

                        ao_levels[i] = if s1 == 1 && s2 == 1 {
                            0
                        } else {
                            3 - (s1 + s2 + corner)
                        };
                    }

                    // --- Emit vertices (type color × face tint × jitter × AO)
                    let base = verts.len() as u16;
                    for (i, c) in face.corners.iter().enumerate() {
                        let k = face.tint * AO_CURVE[ao_levels[i] as usize];
                        verts.push(Vertex {
                            pos: [(wx as f32) + c[0], (y as f32) + c[1], (wz as f32) + c[2]],
                            normal: face.normal,
                            color: [
                                (base_color[0] * k).min(1.0),
                                (base_color[1] * k).min(1.0),
                                (base_color[2] * k).min(1.0),
                            ],
                            uv: corner_uv(tile, *c, face.neighbor, jitter),
                        });
                    }

                    // --- Quad flip toward the brighter diagonal (AO fix) ----
                    if ao_levels[0] + ao_levels[2] > ao_levels[1] + ao_levels[3] {
                        indices.extend_from_slice(&[
                            base + 1,
                            base + 2,
                            base + 3,
                            base + 1,
                            base + 3,
                            base,
                        ]);
                    } else {
                        indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }
                }
            }
        }
    }

    let (wverts, windices) = mesh_water_chunk(world, chunk_x, chunk_z);
    (verts, indices, wverts, windices)
}

/// Build the translucent water mesh independently of opaque terrain. Cached
/// terrain chunks can then restore their opaque buffers and regenerate the
/// small water surface without rerunning the expensive full chunk mesher.
pub fn mesh_water_chunk(world: &World, chunk_x: i32, chunk_z: i32) -> (Vec<Vertex>, Vec<u16>) {
    let mut wverts: Vec<Vertex> = Vec::new();
    let mut windices: Vec<u16> = Vec::new();
    // Only surface water needs faces: top faces at the top of each water
    // column and side faces against air. Solid and water neighbors are culled.
    for lx in 0..CHUNK_SIZE {
        for lz in 0..CHUNK_SIZE {
            let wx = chunk_x + lx;
            let wz = chunk_z + lz;
            if world.block(wx, crate::world::SEA_LEVEL - 1, wz) != Block::Solid(BlockType::Water) {
                continue;
            }
            for y in (0..crate::world::SEA_LEVEL).rev() {
                if world.block(wx, y, wz) != Block::Solid(BlockType::Water) {
                    break;
                }
                let above = world.block(wx, y + 1, wz);
                let is_top = !matches!(above, Block::Solid(BlockType::Water));
                let color = [WATER_TINT[0], WATER_TINT[1], WATER_TINT[2]];
                if is_top {
                    let h = 0.875;
                    let base = wverts.len() as u16;
                    for (px, py, pz, uu, vv) in [
                        (0.0, h, 0.0, 0.0, 0.0),
                        (0.0, h, 1.0, 0.0, 1.0),
                        (1.0, h, 1.0, 1.0, 1.0),
                        (1.0, h, 0.0, 1.0, 0.0),
                    ] {
                        wverts.push(Vertex {
                            pos: [wx as f32 + px, y as f32 + py, wz as f32 + pz],
                            normal: [0.0, 1.0, 0.0],
                            color,
                            uv: [uu / crate::textures::TILES as f32, vv],
                        });
                    }
                    windices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                    // Duplicate the surface with the opposite normal/winding
                    // so it remains visible from below without disabling culling.
                    let base = wverts.len() as u16;
                    for (px, py, pz, uu, vv) in [
                        (0.0, h, 0.0, 0.0, 0.0),
                        (1.0, h, 0.0, 1.0, 0.0),
                        (1.0, h, 1.0, 1.0, 1.0),
                        (0.0, h, 1.0, 0.0, 1.0),
                    ] {
                        wverts.push(Vertex {
                            pos: [wx as f32 + px, y as f32 + py, wz as f32 + pz],
                            normal: [0.0, -1.0, 0.0],
                            color,
                            uv: [uu / crate::textures::TILES as f32, vv],
                        });
                    }
                    windices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
                for ((dx, dy, dz), corners) in WATER_SIDES {
                    if matches!(world.block(wx + dx, y + dy, wz + dz), Block::Solid(_)) {
                        continue;
                    }
                    let base = wverts.len() as u16;
                    for (px, py, pz, uu, vv) in corners {
                        wverts.push(Vertex {
                            pos: [wx as f32 + px, y as f32 + py, wz as f32 + pz],
                            normal: [dx as f32, dy as f32, dz as f32],
                            color,
                            uv: [uu / crate::textures::TILES as f32, vv],
                        });
                    }
                    windices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            }
        }
    }
    (wverts, windices)
}

/// Water surface tint (lit by the shared world shader; alpha comes from the
/// water pipeline's constant blend factor).
const WATER_TINT: [f32; 3] = [0.42, 0.65, 0.95];

/// Water side faces: (neighbor offset, corner positions + uvs). Corners are
/// wound so the face normal points outward.
const WATER_SIDES: [((i32, i32, i32), [(f32, f32, f32, f32, f32); 4]); 4] = [
    (
        (1, 0, 0),
        [
            (1.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0, 1.0, 1.0),
            (1.0, 0.875, 1.0, 1.0, 0.0),
            (1.0, 0.875, 0.0, 0.0, 0.0),
        ],
    ),
    (
        (-1, 0, 0),
        [
            (0.0, 0.0, 1.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0),
            (0.0, 0.875, 0.0, 1.0, 0.0),
            (0.0, 0.875, 1.0, 0.0, 0.0),
        ],
    ),
    (
        (0, 0, 1),
        [
            (0.0, 0.0, 1.0, 0.0, 1.0),
            (1.0, 0.0, 1.0, 1.0, 1.0),
            (1.0, 0.875, 1.0, 1.0, 0.0),
            (0.0, 0.875, 1.0, 0.0, 0.0),
        ],
    ),
    (
        (0, 0, -1),
        [
            (1.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0),
            (0.0, 0.875, 0.0, 1.0, 0.0),
            (1.0, 0.875, 0.0, 0.0, 0.0),
        ],
    ),
];

/// Chunk identifier (minimum corner in block coords).
pub type ChunkCoord = (i32, i32);

/// All chunk coords within `radius` of the given chunk (square footprint,
/// like Minecraft's square render-distance option).
pub fn chunks_around(center: ChunkCoord, radius: i32) -> Vec<ChunkCoord> {
    let mut coords = Vec::new();
    for dx in -radius..=radius {
        for dz in -radius..=radius {
            coords.push((center.0 + dx, center.1 + dz));
        }
    }
    coords
}

/// Convert a world position to the chunk containing it.
pub fn chunk_of(world_x: f32, world_z: f32) -> ChunkCoord {
    (
        (world_x / CHUNK_SIZE as f32).floor() as i32,
        (world_z / CHUNK_SIZE as f32).floor() as i32,
    )
}

#[cfg(test)]
mod uv_tests {
    use super::*;

    /// Regression: Z faces must take V from corner Y (tangent_axes used to
    /// fall through to the wrong arm, stretching one texel row over the face).
    #[test]
    fn side_face_v_tracks_y() {
        let dirt = textures::TILE_DIRT;
        // Z faces: v must differ bottom vs top and map y=1 to the atlas TOP
        // row (v smaller) so upright content isn't upside down.
        let vb = corner_uv(dirt, [0.0, 0.0, 1.0], [0, 0, 1], 1.0)[1];
        let vt = corner_uv(dirt, [0.0, 1.0, 1.0], [0, 0, 1], 1.0)[1];
        assert!(
            vt < vb - 0.2,
            "z face v must track -y (atlas rows): vb={vb} vt={vt}"
        );
        // X faces: same contract.
        let vb = corner_uv(dirt, [1.0, 0.0, 0.0], [1, 0, 0], 1.0)[1];
        let vt = corner_uv(dirt, [1.0, 1.0, 0.0], [1, 0, 0], 1.0)[1];
        assert!(vt < vb - 0.2, "x face v must track -y: vb={vb} vt={vt}");
        // Full tile coverage: u must span the tile too (no stretched rows).
        let ub = corner_uv(dirt, [0.0, 0.0, 1.0], [0, 0, 1], 1.0)[0];
        let ut = corner_uv(dirt, [1.0, 0.0, 1.0], [0, 0, 1], 1.0)[0];
        assert!(ut > ub + 0.05, "z face u must span x: ub={ub} ut={ut}");
        // Top faces: u/v track x/z (no v flip on non-upright content).
        let (_, lo) = tile_uv(dirt);
        let _ = lo;
    }
}

#[cfg(test)]
mod water_tests {
    use super::*;
    use crate::world::World;

    /// A depression below sea level floods; a column above sea level doesn't.
    #[test]
    fn sea_fills_open_basins_only() {
        // Build a world, then dig a basin below sea level near spawn via an
        // edit and confirm the flooded cell reads as water while a hillside
        // bore stays air.
        let w = World::new(0xABCD);
        // Find a column whose top is well above sea level (guaranteed to
        // exist with a small scan) and one at/below it.
        let mut land: Option<(i32, i32, i32)> = None;
        let mut basin: Option<(i32, i32, i32)> = None;
        for x in 0..400 {
            let top = w.procedural_top(x, 0);
            if top > crate::world::SEA_LEVEL + 4 && land.is_none() {
                land = Some((x, crate::world::SEA_LEVEL - 2, 0));
            }
            if top <= crate::world::SEA_LEVEL && basin.is_none() {
                basin = Some((x, crate::world::SEA_LEVEL - 1, 0));
            }
        }
        let (bx, by, bz) = basin.expect("a below-sea column exists in 400");
        assert!(matches!(
            w.block(bx, by, bz),
            Block::Solid(BlockType::Water)
        ));
        if let Some((lx, ly, lz)) = land {
            // Under a tall column, a cell at sea level is INSIDE terrain
            // (solid stone), not water.
            assert!(!matches!(
                w.block(lx, ly, lz),
                Block::Solid(BlockType::Water)
            ));
        }
    }

    /// Water is not solid: you swim through it, rays pass through it, and it
    /// can't be mined.
    #[test]
    fn water_is_not_solid_and_not_mineable() {
        let mut w = World::new(0xABCD);
        let mut found = None;
        for x in 0..400 {
            if matches!(
                w.block(x, crate::world::SEA_LEVEL - 1, 0),
                Block::Solid(BlockType::Water)
            ) {
                found = Some(x);
                break;
            }
        }
        let x = found.expect("water exists in 400 columns");
        let y = crate::world::SEA_LEVEL - 1;
        assert!(!w.solid(x, y, 0), "water must not collide");
        assert!(w.mine(x, y, 0).is_none(), "water must not be mineable");
    }

    /// The water mesh pass emits surface quads for flooded columns and
    /// nothing for dry ones.
    #[test]
    fn water_mesh_emits_surface_quads() {
        let w = World::new(0xABCD);
        let wet_x = (0..400)
            .find(|&x| {
                matches!(
                    w.block(x, crate::world::SEA_LEVEL - 1, 0),
                    Block::Solid(BlockType::Water)
                )
            })
            .expect("a flooded column exists in 400");
        let chunk_x = wet_x.div_euclid(CHUNK_SIZE) * CHUNK_SIZE;
        let (wv, wi) = mesh_water_chunk(&w, chunk_x, 0);
        assert!(
            !wv.is_empty(),
            "water-only meshing must retain lake surfaces"
        );
        assert_eq!(wv.len() % 4, 0);
        assert_eq!(wi.len(), wv.len() / 4 * 6);

        // The extracted path used by cached chunks must produce exactly the
        // same water geometry as a normal full chunk mesh.
        let (_, _, full_wv, full_wi) = mesh_chunk(&w, chunk_x, 0);
        assert_eq!(wv.len(), full_wv.len());
        assert_eq!(wi, full_wi);
    }
}
