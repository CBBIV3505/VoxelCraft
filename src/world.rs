//! Voxel world state: typed block edits layered over the procedural
//! heightmap, deterministic tree generation, solidity/type queries, ground
//! scanning for physics, and DDA voxel raycasting.
//!
//! Storage model: a sparse map of block overrides keyed by (x, y, z).
//! - `None`     → governed by the heightmap (+ analytic trees)
//! - `Some(t)`  → explicit block type, or `Air` (mined)
//!
//! Trees are a pure function of the seed: one candidate tree per 8×8 world
//! cell, at a hash-jittered position, existing with ~40% probability. This
//! means they "generate" themselves into any chunk without storage, survive
//! save/load trivially, and mining one just writes an Air override that wins
//! over the analytic tree.

use std::collections::{HashMap, HashSet};

use glam::Vec3;

use crate::terrain::terrain_height;

/// Trilinearly-interpolated value noise in 3D (hash lattice + smooth fade).
/// Used by the cave carving; independent of the 2D heightmap noise.
fn value_noise_3(x: f32, y: f32, z: f32, seed: u64) -> f32 {
    fn h(ix: i32, iy: i32, iz: i32, seed: u64) -> f32 {
        let mut z2 = (ix as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (iy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ (iz as u64).wrapping_mul(0x1656_67A5_35A5_30C9)
            ^ seed;
        z2 = (z2 ^ (z2 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z2 = (z2 ^ (z2 >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z2 ^ (z2 >> 31)) >> 11) as f32 / (1u64 << 53) as f32
    }
    let fade = |t: f32| t * t * (3.0 - 2.0 * t);
    let (xi, yi, zi) = (x.floor() as i32, y.floor() as i32, z.floor() as i32);
    let (xf, yf, zf) = (
        fade(x - xi as f32),
        fade(y - yi as f32),
        fade(z - zi as f32),
    );
    let c = |dx: i32, dy: i32, dz: i32| h(xi + dx, yi + dy, zi + dz, seed);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(c(0, 0, 0), c(1, 0, 0), xf);
    let x10 = lerp(c(0, 1, 0), c(1, 1, 0), xf);
    let x01 = lerp(c(0, 0, 1), c(1, 0, 1), xf);
    let x11 = lerp(c(0, 1, 1), c(1, 1, 1), xf);
    lerp(lerp(x00, x10, yf), lerp(x01, x11, yf), zf)
}

/// How far the player can reach to mine/place (blocks).
pub const REACH: f32 = 5.0;
/// Depth scanned when looking for ground under the player's feet.
const GROUND_SCAN: i32 = 64;
/// Sea level: terrain below this height fills with water (procedurally —
/// never stored as edits). Heightmap range is ~8–18, so 11 carves real
/// lakes without flooding most of the world.
pub const SEA_LEVEL: i32 = 11;
/// World-cell size for tree placement (one candidate tree per cell).
const TREE_CELL: i32 = 8;
/// Probability (out of 100) that a tree cell actually spawns a tree.
const TREE_CHANCE: u64 = 40;

/// Distinct block types. Natural terrain is column-stratified; trees add
/// logs/leaves; crafting turns logs into planks.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum BlockType {
    Grass,
    Dirt,
    Stone,
    Log,
    Leaves,
    Planks,
    /// A placeable crafting station; right-clicking one opens the 3×3 table.
    CraftingTable,
    /// Stone with coal seams. Requires a pickaxe to drop anything; found in
    /// veins underground and drops itself (placeable fuel-less ore for now).
    CoalOre,
    /// Translucent water filling terrain below [`SEA_LEVEL`]. Swimmable,
    /// non-solid, not collectable/placement-targetable.
    Water,
    /// A mined cell (explicit hole). Not placeable, not collectable.
    Air,
}

impl BlockType {
    /// All placeable types (used by the hotbar and crafting logic).
    pub const SOLID_ALL: [BlockType; 8] = [
        BlockType::Grass,
        BlockType::Dirt,
        BlockType::Stone,
        BlockType::Log,
        BlockType::Leaves,
        BlockType::Planks,
        BlockType::CraftingTable,
        BlockType::CoalOre,
    ];

    /// Water is never in SOLID_ALL (it isn't an item), so this is a plain
    /// predicate here: is this type a fluid the player swims in?
    pub fn is_water(self) -> bool {
        self == BlockType::Water
    }

    /// sRGB color of the block's base color (before face shading/AO/jitter).
    pub fn srgb(self) -> [f32; 3] {
        fn chan(c: u8) -> f32 {
            let c = c as f32 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }
        match self {
            BlockType::Grass => [chan(110), chan(168), chan(68)],
            BlockType::Dirt => [chan(134), chan(96), chan(67)],
            BlockType::Stone => [chan(125), chan(125), chan(125)],
            BlockType::Log => [chan(109), chan(84), chan(50)],
            BlockType::Leaves => [chan(60), chan(124), chan(45)],
            BlockType::Planks => [chan(168), chan(133), chan(84)],
            BlockType::CraftingTable => [chan(150), chan(106), chan(62)],
            BlockType::CoalOre => [chan(58), chan(58), chan(62)],
            BlockType::Water => [chan(48), chan(96), chan(216)],
            BlockType::Air => [0.0; 3],
        }
    }

    /// What you get when you mine this block (grass drops dirt, like MC).
    pub fn drop(self) -> BlockType {
        match self {
            BlockType::Grass => BlockType::Dirt,
            other => other,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BlockType::Grass => "grass",
            BlockType::Dirt => "dirt",
            BlockType::Stone => "stone",
            BlockType::Log => "log",
            BlockType::Leaves => "leaves",
            BlockType::Planks => "planks",
            BlockType::CraftingTable => "crafting table",
            BlockType::CoalOre => "coal ore",
            BlockType::Water => "water",
            BlockType::Air => "air",
        }
    }
}

/// The block occupying a cell: solid type or air.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Block {
    Air,
    Solid(BlockType),
}

impl Block {
    /// Water is `Block::Solid(Water)` in the type system (it occupies a
    /// cell) but is NOT solid for collision/raycasting — you swim through
    /// it, and it can't be mined or targeted.
    pub fn is_solid(self) -> bool {
        matches!(self, Block::Solid(t) if !t.is_water())
    }
}

/// A deterministic tree instance (trunk column + shape parameters).
#[derive(Clone, Copy, Debug)]
struct Tree {
    tx: i32,
    tz: i32,
    /// y of the lowest trunk block (= procedural ground top at the trunk).
    base: i32,
    /// Trunk height in blocks (4–6).
    height: i32,
}

pub struct World {
    pub seed: u64,
    // (Send bound: meshing runs on a background thread with an owned clone
    // of the world state — HashMap + HashSet + u64 are all Send.)
    /// Sparse block overrides: explicit type or mined air. Everything else is
    /// implied by the heightmap + analytic trees.
    blocks: HashMap<[i32; 3], BlockType>,
    /// Chunks affected by edits since the last drain (re-mesh targets).
    dirty_chunks: HashSet<(i32, i32)>,
}

impl World {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            blocks: HashMap::new(),
            dirty_chunks: HashSet::new(),
        }
    }

    /// Deep copy for the background mesher: the worker gets a frozen snapshot
    /// (edits included) and never mutates the live world.
    pub fn snapshot(&self) -> Self {
        Self {
            seed: self.seed,
            blocks: self.blocks.clone(),
            dirty_chunks: HashSet::new(),
        }
    }

    // -- deterministic hashing ------------------------------------------------

    fn hash(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn hash2(a: i64, b: i64, salt: u64, seed: u64) -> u64 {
        Self::hash(
            (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                ^ salt
                ^ seed,
        )
    }

    // -- trees ----------------------------------------------------------------

    /// The tree whose trunk lives in world cell (cx, cz) (cell coords, not
    /// block coords), if any.
    fn tree_in_cell(&self, cx: i32, cz: i32) -> Option<Tree> {
        let h = Self::hash2(cx as i64, cz as i64, 0x7EE5, self.seed);
        if h % 100 >= TREE_CHANCE {
            return None;
        }
        // Jittered trunk position, keeping a 1-block margin inside the cell so
        // leaves (radius 2) don't spill into cells two over.
        let jx = (Self::hash2(cx as i64, cz as i64, 0x11, self.seed) % 5) as i32;
        let jz = (Self::hash2(cx as i64, cz as i64, 0x22, self.seed) % 5) as i32;
        let tx = cx * TREE_CELL + 1 + jx;
        let tz = cz * TREE_CELL + 1 + jz;
        let base = terrain_height(tx, tz, self.seed);
        if base < 5 {
            return None; // too low / underwater-ish; no trees there
        }
        let height = 4 + (Self::hash2(cx as i64, cz as i64, 0x33, self.seed) % 3) as i32;
        Some(Tree {
            tx,
            tz,
            base,
            height,
        })
    }

    /// Trees whose cells could reach the given block column area.
    fn nearby_trees(&self, x: i32, z: i32) -> [Option<Tree>; 9] {
        let cx = x.div_euclid(TREE_CELL);
        let cz = z.div_euclid(TREE_CELL);
        let mut out = [None; 9];
        let mut i = 0;
        for dx in -1..=1 {
            for dz in -1..=1 {
                out[i] = self.tree_in_cell(cx + dx, cz + dz);
                i += 1;
            }
        }
        out
    }

    /// Is block (x, y, z) part of the given tree? Returns the type if so.
    fn tree_block(tree: &Tree, x: i32, y: i32, z: i32, seed: u64) -> Option<BlockType> {
        // Trunk.
        if x == tree.tx && z == tree.tz && y >= tree.base && y < tree.base + tree.height {
            return Some(BlockType::Log);
        }
        // Leaf blob: two 5×5 layers, a 3×3 layer, then a plus-shaped cap.
        let top = tree.base + tree.height - 1;
        let dy = y - (top - 1);
        if !(0..=3).contains(&dy) {
            return None;
        }
        let dx = x - tree.tx;
        let dz = z - tree.tz;
        let (ax, az) = (dx.abs(), dz.abs());
        let radius = match dy {
            0 | 1 => 2,
            _ => 1,
        };
        if ax > radius || az > radius {
            return None;
        }
        // Cap layer (dy == 3) is plus-shaped: no corners.
        if dy == 3 && ax == 1 && az == 1 {
            return None;
        }
        // Big layers: skip corners ~2/3 of the time (deterministic per block).
        if dy <= 1 && ax == 2 && az == 2 {
            let h = Self::hash2(x as i64 * 7 + y as i64, z as i64, 0xA1EAF, seed);
            if h % 3 != 0 {
                return None;
            }
        }
        if dx == 0 && dz == 0 && y < top {
            return Some(BlockType::Log); // trunk continues through leaf layers
        }
        Some(BlockType::Leaves)
    }

    // -- queries ----------------------------------------------------------------

    /// Procedural (unedited) column height — top solid block y + 1.
    pub fn procedural_top(&self, x: i32, z: i32) -> i32 {
        terrain_height(x, z, self.seed)
    }

    /// Raw edit-override lookup (meshing fast path): `Some(t)` if this exact
    /// cell has an explicit edit (type or Air), `None` if heightmap-governed.
    pub fn override_at(&self, x: i32, y: i32, z: i32) -> Option<BlockType> {
        if self.blocks.is_empty() {
            return None; // fast path: no edits at all
        }
        self.blocks.get(&[x, y, z]).copied()
    }

    /// Does any edit exist? (Meshing uses this to decide scan ceilings.)
    pub fn has_edits(&self) -> bool {
        !self.blocks.is_empty()
    }

    /// All tree blocks intersecting the given block-coordinate box, with their
    /// types. Called once per chunk-mesh so trees are materialized from the
    /// analytic definition without storage.
    pub fn collect_tree_blocks(
        &self,
        min_x: i32,
        max_x: i32,
        min_z: i32,
        max_z: i32,
        out: &mut HashMap<[i32; 3], BlockType>,
    ) {
        let cx0 = (min_x - 2).div_euclid(TREE_CELL);
        let cx1 = (max_x + 2).div_euclid(TREE_CELL);
        let cz0 = (min_z - 2).div_euclid(TREE_CELL);
        let cz1 = (max_z + 2).div_euclid(TREE_CELL);
        for cx in cx0..=cx1 {
            for cz in cz0..=cz1 {
                if let Some(tree) = self.tree_in_cell(cx, cz) {
                    // Trunk + leaf envelope.
                    for y in tree.base..=(tree.base + tree.height + 2) {
                        for dx in -2..=2 {
                            for dz in -2..=2 {
                                let x = tree.tx + dx;
                                let z = tree.tz + dz;
                                if x < min_x - 2 || x > max_x + 2 || z < min_z - 2 || z > max_z + 2
                                {
                                    continue;
                                }
                                if let Some(t) = Self::tree_block(&tree, x, y, z, self.seed) {
                                    out.insert([x, y, z], t);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The block occupying cell (x, y, z), edits and trees included.
    pub fn block(&self, x: i32, y: i32, z: i32) -> Block {
        if let Some(t) = self.blocks.get(&[x, y, z]) {
            return if *t == BlockType::Air {
                Block::Air
            } else {
                Block::Solid(*t)
            };
        }
        // Trees (analytic).
        for tree in self.nearby_trees(x, z).into_iter().flatten() {
            if let Some(t) = Self::tree_block(&tree, x, y, z, self.seed) {
                return Block::Solid(t);
            }
        }
        // Ocean/lake fill: empty cells at or below sea level above solid
        // ground are water (procedural — never stored, so it costs no save
        // space and regenerates identically). Only OPEN basins flood: a
        // column whose terrain top is under the sea. Caves and mine shafts
        // under land stay air (their column top is above sea level).
        let top = self.procedural_top(x, z);
        if y < SEA_LEVEL && top <= SEA_LEVEL {
            return Block::Solid(BlockType::Water);
        }
        // Procedural stratification: 1 grass layer on top, dirt, stone below.
        if y < top {
            // Caves: a 3D-noise worm field carving the underground. Two
            //ridged noise bands multiplied — where both are near zero the
            //cell is air, which produces connected winding tunnels instead of
            //round blobs. Sealed by 3 blocks of crust so the surface stays
            //intact and caves are something you dig into.
            if y <= top - 3 && Self::is_cave(x, y, z, self.seed) {
                return Block::Air;
            }
            Block::Solid(if y == top - 1 {
                BlockType::Grass
            } else if y >= top - 4 {
                BlockType::Dirt
                // Ore veins: coal sprinkled through the stone band with a
                // hashed pocket check (no extra noise pass).
            } else if Self::is_coal_vein(x, y, z, self.seed) {
                BlockType::CoalOre
            } else {
                BlockType::Stone
            })
        } else {
            Block::Air
        }
    }

    /// Cave test: two independent value-noise bands; a cell is a cave when
    /// BOTH bands are within their tube half-width of zero. Deterministic in
    /// (x, y, z, seed).
    fn is_cave(x: i32, y: i32, z: i32, seed: u64) -> bool {
        const S: f32 = 0.09; // horizontal scale (bigger = smaller tunnels)
        const SY: f32 = 0.14; // vertical squash (flatter, more cave-like)
        let a = value_noise_3(x as f32 * S, y as f32 * SY, z as f32 * S, seed ^ 0xCAFE);
        let b = value_noise_3(x as f32 * S, y as f32 * SY, z as f32 * S, seed ^ 0xBEEF);
        // Near-zero crossings of both bands = worm intersection tube.
        const W: f32 = 0.055;
        (a - 0.5).abs() < W && (b - 0.5).abs() < W
    }

    /// Coal pocket test: hashed coarse cells (4³ blocks), ~28% of pockets
    /// hold a smaller coal cluster inside the stone band only.
    fn is_coal_vein(x: i32, y: i32, z: i32, seed: u64) -> bool {
        let px = x.div_euclid(4);
        let py = y.div_euclid(4);
        let pz = z.div_euclid(4);
        let h = Self::hash2(px as i64 * 3 + py as i64, pz as i64, 0xC0A1, seed);
        if h % 100 >= 28 {
            return false;
        }
        // Inside the pocket: sub-hash decides which blocks of the 4³ cell are
        // coal (~40%), so veins look clumpy instead of cubic.
        let s = Self::hash((x as u64) << 32 ^ (y as u64) << 16 ^ z as u64 ^ 0xC0A1_F00D);
        s % 10 < 4
    }

    /// Is the block at (x, y, z) solid?
    pub fn solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.block(x, y, z).is_solid()
    }

    /// Mine the block at (x, y, z). Returns the dropped type on success.
    /// Mining bedrock (y = 0) is refused.
    pub fn mine(&mut self, x: i32, y: i32, z: i32) -> Option<BlockType> {
        match self.block(x, y, z) {
            Block::Solid(t) if y > 0 && !t.is_water() => {
                self.blocks.insert([x, y, z], BlockType::Air);
                self.touch(x, z);
                Some(t.drop())
            }
            _ => None,
        }
    }

    /// Place a solid block of type `t` at (x, y, z). Fails inside solids.
    pub fn place(&mut self, x: i32, y: i32, z: i32, t: BlockType) -> bool {
        if self.solid(x, y, z) {
            return false;
        }
        self.blocks.insert([x, y, z], t);
        self.touch(x, z);
        true
    }

    /// Mark chunks needing a re-mesh after an edit at (x, z): the containing
    /// chunk plus neighbors, since face culling and AO reach one block out.
    fn touch(&mut self, x: i32, z: i32) {
        for dx in -1..=1 {
            for dz in -1..=1 {
                let cx = (x + dx).div_euclid(crate::terrain::CHUNK_SIZE);
                let cz = (z + dz).div_euclid(crate::terrain::CHUNK_SIZE);
                self.dirty_chunks.insert((cx, cz));
            }
        }
    }

    /// Take the set of chunks edited since the last call (re-mesh targets).
    /// Number of stored block edits (for the debug overlay).
    pub fn edit_count(&self) -> usize {
        self.blocks.len()
    }

    /// Read-only view of the edit overlay for saving. Order is arbitrary but
    /// stable per world state.
    pub fn edits(&self) -> impl Iterator<Item = (&[i32; 3], BlockType)> {
        self.blocks.iter().map(|(k, &t)| (&*k, t))
    }

    /// Bulk-insert saved edits at load time (no dirty marking — meshes are
    /// restored from the chunk cache instead of re-meshed).
    pub fn restore_edits(&mut self, edits: impl Iterator<Item = ([i32; 3], BlockType)>) {
        for (k, t) in edits {
            self.blocks.insert(k, t);
        }
    }

    pub fn drain_dirty(&mut self) -> Vec<(i32, i32)> {
        self.dirty_chunks.drain().collect()
    }

    /// Highest solid-block top at or below `from_y` — the surface the player's
    /// feet rest on. Scans at most [`GROUND_SCAN`] blocks down.
    pub fn ground_under(&self, x: i32, z: i32, from_y: f32) -> f32 {
        let mut y = from_y.floor() as i32;
        for _ in 0..GROUND_SCAN {
            if self.solid(x, y, z) {
                return (y + 1) as f32;
            }
            y -= 1;
        }
        -1.0e6
    }

    /// Y of the water surface at (x, z) — the top of the highest water cell
    /// in the column — as `top_y + 0.875` (matching the meshed surface
    /// height). Returns -∞ when the column has no water at/above `from_y`.
    pub fn water_surface(&self, x: i32, z: i32, from_y: f32) -> f32 {
        let mut y = (from_y.floor() as i32 + 1).min(crate::world::SEA_LEVEL - 1);
        for _ in 0..(SEA_LEVEL + 1) {
            if y < 0 {
                break;
            }
            if self.block(x, y, z) == Block::Solid(BlockType::Water) {
                return y as f32 + 0.875;
            }
            y -= 1;
        }
        f32::NEG_INFINITY
    }
}

// ---------------------------------------------------------------------------
// DDA voxel raycast (Amanatides & Woo)
// ---------------------------------------------------------------------------

pub struct RayHit {
    /// Solid block that was hit.
    pub block: [i32; 3],
    /// Empty cell adjacent to the hit face (placement target).
    pub adjacent: [i32; 3],
}

/// March a ray from `origin` along `dir` (normalized) through the voxel grid,
/// returning the first solid cell hit within `max_dist`, plus the empty cell
/// just before it.
pub fn raycast(world: &World, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RayHit> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }

    let mut x = origin.x.floor() as i32;
    let mut y = origin.y.floor() as i32;
    let mut z = origin.z.floor() as i32;

    // The cell the ray starts in: if we're inside a block (shouldn't happen),
    // return it immediately so mining still works.
    if world.solid(x, y, z) {
        return Some(RayHit {
            block: [x, y, z],
            adjacent: [x, y, z],
        });
    }

    let step_x = dir.x.signum() as i32;
    let step_y = dir.y.signum() as i32;
    let step_z = dir.z.signum() as i32;

    // Distance along the ray to the first x/y/z grid-plane crossing.
    let inv = |d: f32| {
        if d != 0.0 {
            1.0 / d.abs()
        } else {
            f32::INFINITY
        }
    };
    let t_delta_x = inv(dir.x);
    let t_delta_y = inv(dir.y);
    let t_delta_z = inv(dir.z);

    let boundary = |p: f32, s: i32| {
        if s > 0 {
            p.floor() + 1.0 - p
        } else {
            p - p.floor()
        }
    };
    let mut t_max_x = boundary(origin.x, step_x) * t_delta_x;
    let mut t_max_y = boundary(origin.y, step_y) * t_delta_y;
    let mut t_max_z = boundary(origin.z, step_z) * t_delta_z;

    let mut t = 0.0f32;
    let mut adjacent = [x, y, z];
    while t <= max_dist {
        // Step to the nearest grid-plane crossing.
        if t_max_x <= t_max_y && t_max_x <= t_max_z {
            x += step_x;
            t = t_max_x;
            t_max_x += t_delta_x;
        } else if t_max_y <= t_max_z {
            y += step_y;
            t = t_max_y;
            t_max_y += t_delta_y;
        } else {
            z += step_z;
            t = t_max_z;
            t_max_z += t_delta_z;
        }

        if t > max_dist {
            break;
        }

        if world.solid(x, y, z) {
            return Some(RayHit {
                block: [x, y, z],
                adjacent,
            });
        }
        adjacent = [x, y, z];
    }

    None
}
