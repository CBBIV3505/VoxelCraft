//! World persistence: everything a restart needs to resume the same world
//! with no slow regeneration.
//!
//! Two files live in one save directory (`~/.local/share/voxelcraft/world/`
//! by default, XDG-aware):
//!
//! - `world.conf` — tiny text header: format version, seed, player state,
//!   time of day. Hand-editable and forward-compatible like settings.conf.
//! - `edits.bin` — the sparse block-edit overlay, binary for compactness:
//!   13 bytes per edit (12-byte coord + 1-byte block id). A heavily-mined
//!   world of 100k edits is ~1.3 MB.
//!
//! Chunk mesh cache: `chunks/<cx>_<cz>.mesh` — the exact vertex/index data
//! uploaded to the GPU, tagged with the seed and format version. A chunk is
//! written once when first meshed; a restart loads the cache instead of
//! re-meshing, so startup streaming is a file copy, not terrain math. Chunks
//! with no edits re-derive identically from the seed, but caching them too
//! is what makes restarting at render distance 32 fast.

use std::io::Read;
use std::path::PathBuf;

use crate::world::BlockType;

/// Bump when the mesh cache layout or block ids change; stale caches with an
/// old tag are deleted rather than loaded.
pub const FORMAT_VERSION: u32 = 2;

/// Bump whenever terrain GENERATION changes shape (caves, ores, atlas tile
/// count, heightmap constants). Chunk caches are tagged with this alongside
/// the seed, so old-generated meshes are ignored and re-derived instead of
/// sampling a stale atlas or missing features. Without this, saving the game
/// and changing the generator ships invisible corruption (stretched tiles,
/// pre-cave terrain) on every previously-seen chunk.
pub const TERRAIN_VERSION: u64 = 3; // 3: sea-level water added to generation

/// Block id byte used in `edits.bin` (must match BlockType::from_id).
#[repr(u8)]
enum Ids {
    Grass = 0,
    Dirt = 1,
    Stone = 2,
    Log = 3,
    Leaves = 4,
    Planks = 5,
    CraftingTable = 6,
    CoalOre = 8,
    Water = 9,
    Air = 7,
}

impl BlockType {
    pub fn id(self) -> u8 {
        match self {
            BlockType::Grass => Ids::Grass as u8,
            BlockType::Dirt => Ids::Dirt as u8,
            BlockType::Stone => Ids::Stone as u8,
            BlockType::Log => Ids::Log as u8,
            BlockType::Leaves => Ids::Leaves as u8,
            BlockType::Planks => Ids::Planks as u8,
            BlockType::CraftingTable => Ids::CraftingTable as u8,
            BlockType::CoalOre => Ids::CoalOre as u8,
            // Water is procedural and never stored as an edit, but the id
            // exists so the format is forward-stable.
            BlockType::Water => Ids::Water as u8,
            BlockType::Air => Ids::Air as u8,
        }
    }

    pub fn from_id(b: u8) -> Option<Self> {
        Some(match b {
            0 => BlockType::Grass,
            1 => BlockType::Dirt,
            2 => BlockType::Stone,
            3 => BlockType::Log,
            4 => BlockType::Leaves,
            5 => BlockType::Planks,
            6 => BlockType::CraftingTable,
            7 => BlockType::Air,
            8 => BlockType::CoalOre,
            9 => BlockType::Water,
            _ => return None,
        })
    }
}

/// Load the world header only (no edits). Used by App::new to learn the seed
/// before the World is constructed; None when there is no readable save.
pub fn peek_saved_meta() -> Option<Meta> {
    let dir = save_dir();
    clear_cache_if_stale(&dir);
    load_meta(&dir)
}

/// Runtime save state the App carries: where this world saves, its seed, and
/// whether the current session was restored from disk (vs. a fresh world).
pub struct SaveState {
    pub loaded: bool,
    pub dir: PathBuf,
}

/// Root directory of the default save (override with VOXELCRAFT_SAVE_DIR).
pub fn save_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("VOXELCRAFT_SAVE_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    let base = match std::env::var("XDG_DATA_HOME") {
        Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".local/share")
        }
    };
    base.join("voxelcraft").join("world")
}

/// Player state persisted in the header.
pub struct PlayerState {
    pub pos: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub flying: bool,
}

fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

// --- world.conf -------------------------------------------------------------

/// One persisted header line per key; unknown keys are ignored on parse.
pub fn save_meta(
    dir: &std::path::Path,
    seed: u64,
    player: &PlayerState,
    day_time: f32,
) -> std::io::Result<()> {
    let text = format!(
        "# VoxelCraft world save (hand-editable at your own risk)\n\
         version={FORMAT_VERSION}\n\
         seed={seed}\n\
         px={:.3}\npy={:.3}\npz={:.3}\n\
         yaw={:.5}\npitch={:.5}\n\
         flying={}\n\
         day_time={:.5}\n",
        player.pos[0],
        player.pos[1],
        player.pos[2],
        player.yaw,
        player.pitch,
        player.flying,
        day_time
    );
    write_atomic(&dir.join("world.conf"), text.as_bytes())
}

/// Parsed world.conf. Absent fields keep defaults so an older save still
/// loads.
pub struct Meta {
    pub version: u32,
    pub seed: u64,
    pub player: Option<PlayerState>,
    pub day_time: f32,
}

pub fn load_meta(dir: &std::path::Path) -> Option<Meta> {
    let text = std::fs::read_to_string(dir.join("world.conf")).ok()?;
    let mut m = Meta {
        version: 0,
        seed: 0,
        player: None,
        day_time: 0.08,
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let (k, v) = (k.trim(), v.trim());
        match k {
            "version" => m.version = v.parse().unwrap_or(0),
            "seed" => m.seed = v.parse().unwrap_or(0),
            "px" | "py" | "pz" | "yaw" | "pitch" => {
                let ok = v.parse::<f32>().is_ok();
                if ok && m.player.is_none() {
                    m.player = Some(PlayerState {
                        pos: [0.0; 3],
                        yaw: 0.0,
                        pitch: 0.0,
                        flying: false,
                    });
                }
                if let Some(p) = &mut m.player {
                    match k {
                        "px" => p.pos[0] = v.parse().unwrap_or(0.0),
                        "py" => p.pos[1] = v.parse().unwrap_or(0.0),
                        "pz" => p.pos[2] = v.parse().unwrap_or(0.0),
                        "yaw" => p.yaw = v.parse().unwrap_or(0.0),
                        "pitch" => p.pitch = v.parse().unwrap_or(0.0),
                        _ => {}
                    }
                }
            }
            "flying" => {
                let flying = v == "true" || v == "1";
                if m.player.is_none() {
                    m.player = Some(PlayerState {
                        pos: [0.0; 3],
                        yaw: 0.0,
                        pitch: 0.0,
                        flying,
                    });
                } else if let Some(p) = &mut m.player {
                    p.flying = flying;
                }
            }
            "day_time" => m.day_time = v.parse().unwrap_or(0.08),
            _ => {}
        }
    }
    // Accept the current version AND the one before it: a terrain bump
    // (TERRAIN_VERSION) invalidates chunk caches without invalidating the
    // player's edits/position, so a save written by v1 loads fine — its
    // stale chunks just regenerate.
    if m.version == 0 || m.version > FORMAT_VERSION || m.seed == 0 {
        return None; // not a readable save (or from a newer format)
    }
    Some(m)
}

// --- edits.bin ---------------------------------------------------------------

/// Serialize the whole edit overlay: `[i32 x, y, z, u8 id]` per entry.
pub fn save_edits(
    dir: &std::path::Path,
    edits: impl Iterator<Item = ([i32; 3], BlockType)>,
) -> std::io::Result<usize> {
    let mut bytes = Vec::with_capacity(16 + edits.size_hint().0 * 13);
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    let mut count = 0usize;
    for ([x, y, z], t) in edits {
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
        bytes.extend_from_slice(&z.to_le_bytes());
        bytes.push(t.id());
        count += 1;
    }
    write_atomic(&dir.join("edits.bin"), &bytes)?;
    Ok(count)
}

/// Load the edit overlay (None if the file is absent). Corrupt tails are
/// ignored: every record is independently valid, so we keep what parsed.
pub fn load_edits(dir: &std::path::Path) -> Option<Vec<([i32; 3], BlockType)>> {
    let mut bytes = Vec::new();
    std::fs::File::open(dir.join("edits.bin"))
        .ok()?
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() < 4 {
        return None;
    }
    let ver = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
    if ver != FORMAT_VERSION {
        return None;
    }
    let mut out = Vec::new();
    let mut off = 4;
    while off + 13 <= bytes.len() {
        let x = i32::from_le_bytes(bytes[off..off + 4].try_into().ok()?);
        let y = i32::from_le_bytes(bytes[off + 4..off + 8].try_into().ok()?);
        let z = i32::from_le_bytes(bytes[off + 8..off + 12].try_into().ok()?);
        let id = bytes[off + 12];
        off += 13;
        if let Some(t) = BlockType::from_id(id) {
            out.push(([x, y, z], t));
        }
    }
    Some(out)
}

// --- chunk mesh cache ---------------------------------------------------------

fn chunk_path(dir: &std::path::Path, cx: i32, cz: i32) -> PathBuf {
    dir.join("chunks").join(format!("{cx}_{cz}.mesh"))
}

/// Serialize a chunk mesh (verts as f32s, indices as u16s) with a
/// seed+version tag. `seed_tag` lets a differently-seeded world ignore (and
/// prune) another world's cache files.
#[allow(clippy::too_many_arguments)]
pub fn save_chunk_mesh(
    dir: &std::path::Path,
    seed_tag: u64,
    cx: i32,
    cz: i32,
    verts: &[crate::geometry::Vertex],
    indices: &[u16],
) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(24 + verts.len() * 44 + indices.len() * 2);
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    // Tag = seed XOR terrain version, so either mismatch invalidates the
    // cache file.
    let tag = seed_tag ^ TERRAIN_VERSION;
    bytes.extend_from_slice(&tag.to_le_bytes());
    // Header: vertex count + index count so loading can validate.
    bytes.extend_from_slice(&(verts.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    for v in verts {
        bytes.extend_from_slice(bytemuck::bytes_of(v));
    }
    for i in indices {
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    write_atomic(&chunk_path(dir, cx, cz), &bytes)
}

/// Load a cached chunk mesh if it exists, was written by this format version,
/// and belongs to this seed.
pub fn load_chunk_mesh(
    dir: &std::path::Path,
    seed_tag: u64,
    cx: i32,
    cz: i32,
) -> Option<(Vec<crate::geometry::Vertex>, Vec<u16>)> {
    let mut bytes = Vec::new();
    std::fs::File::open(chunk_path(dir, cx, cz))
        .ok()?
        .read_to_end(&mut bytes)
        .ok()?;
    // Full header is 20 bytes (including both counts). Older code checked 16
    // then sliced 16..20, panicking the mesher worker on a truncated cache.
    if bytes.len() < 20 {
        return None;
    }
    let ver = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
    let seed = u64::from_le_bytes(bytes[4..12].try_into().ok()?);
    let nv = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
    let ni = u32::from_le_bytes(bytes[16..20].try_into().ok()?);
    if ver != FORMAT_VERSION || seed != (seed_tag ^ TERRAIN_VERSION) {
        return None;
    }
    let stride = std::mem::size_of::<crate::geometry::Vertex>();
    let need = 20usize
        .checked_add((nv as usize).checked_mul(stride)?)
        .and_then(|n| n.checked_add((ni as usize).checked_mul(2)?))?;
    // Reject partial, oversized, and structurally invalid caches. Counts come
    // from user-writable save data and must not trigger huge allocations.
    if bytes.len() != need || (ni > 0 && nv == 0) {
        return None;
    }
    let max_index = nv.saturating_sub(1).min(u16::MAX as u32) as u16;
    let mut verts = Vec::with_capacity(nv as usize);
    let mut off = 20;
    for _ in 0..nv {
        let b: [u8; std::mem::size_of::<crate::geometry::Vertex>()] =
            bytes[off..off + stride].try_into().ok()?;
        verts.push(*bytemuck::from_bytes(&b));
        off += stride;
    }
    let mut indices = Vec::with_capacity(ni as usize);
    for _ in 0..ni {
        let index = u16::from_le_bytes(bytes[off..off + 2].try_into().ok()?);
        if index > max_index {
            return None;
        }
        indices.push(index);
        off += 2;
    }
    Some((verts, indices))
}

/// Delete the whole save directory (used when the format version changes so
/// stale caches can't accumulate).
pub fn clear_cache_if_stale(dir: &std::path::Path) {
    match load_meta(dir) {
        Some(_) => {}
        None => {
            // Only prune when there is actually something there — this also
            // runs on a fresh machine where the dir doesn't exist yet.
            if dir.join("chunks").exists() || dir.join("edits.bin").exists() {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "voxelcraft-save-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn meta_round_trips() {
        let dir = tmpdir();
        let p = PlayerState {
            pos: [1.5, 12.25, -3.0],
            yaw: 1.1,
            pitch: -0.25,
            flying: true,
        };
        save_meta(&dir, 12345, &p, 0.31).unwrap();
        let m = load_meta(&dir).unwrap();
        assert_eq!(m.seed, 12345);
        let pl = m.player.unwrap();
        assert!((pl.pos[1] - 12.25).abs() < 1e-3);
        assert!((pl.yaw - 1.1).abs() < 1e-4);
        assert!(pl.flying);
        assert!((m.day_time - 0.31).abs() < 1e-4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edits_round_trip_and_ignore_corrupt_tail() {
        let dir = tmpdir();
        let edits = vec![
            ([1, 2, 3], BlockType::Air),
            ([4, 5, 6], BlockType::Planks),
            ([-7, 0, 9], BlockType::CraftingTable),
        ];
        let n = save_edits(&dir, edits.iter().copied()).unwrap();
        assert_eq!(n, 3);
        // Append a partial record (7 bytes) — load must ignore it silently.
        let path = dir.join("edits.bin");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(&[9u8; 7]);
        std::fs::write(&path, bytes).unwrap();
        let loaded = load_edits(&dir).unwrap();
        assert_eq!(loaded, edits);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn chunk_mesh_round_trips_with_seed_tag() {
        let dir = tmpdir();
        let v = crate::geometry::Vertex {
            pos: [0.5, 1.0, -2.0],
            normal: [0.0, 1.0, 0.0],
            color: [0.3, 0.4, 0.5],
            uv: [0.25, 0.75],
        };
        let idx = vec![0u16, 1, 2, 0, 2, 3];
        save_chunk_mesh(&dir, 777, -1, 3, &[v; 4], &idx).unwrap();
        let (lv, li) = load_chunk_mesh(&dir, 777, -1, 3).unwrap();
        assert_eq!(li, idx);
        assert_eq!(lv[0].pos, v.pos);
        assert_eq!(lv[3].uv, v.uv);
        // A different seed must not load another world's chunk.
        assert!(load_chunk_mesh(&dir, 999, -1, 3).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncated_chunk_mesh_headers_are_rejected() {
        let dir = tmpdir();
        let path = chunk_path(&dir, 0, 0);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // The header ends at byte 20; lengths 16–19 used to pass the old
        // guard and panic while slicing the index count.
        for len in [16, 17, 18, 19] {
            std::fs::write(&path, vec![0u8; len]).unwrap();
            assert!(load_chunk_mesh(&dir, 42, 0, 0).is_none());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn chunk_mesh_rejects_bad_index_and_trailing_data() {
        let dir = tmpdir();
        let v = crate::geometry::Vertex {
            pos: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            color: [1.0; 3],
            uv: [0.0, 0.0],
        };
        let path = chunk_path(&dir, 1, 2);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(42u64 ^ TERRAIN_VERSION).to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes()); // one vertex
        bytes.extend_from_slice(&1u32.to_le_bytes()); // one index
        bytes.extend_from_slice(bytemuck::bytes_of(&v));
        bytes.extend_from_slice(&1u16.to_le_bytes()); // out of range
        std::fs::write(&path, &bytes).unwrap();
        assert!(load_chunk_mesh(&dir, 42, 1, 2).is_none());
        let last = bytes.len() - 1;
        bytes[last] = 0;
        bytes.push(0); // exact structure plus trailing byte is invalid
        std::fs::write(&path, &bytes).unwrap();
        assert!(load_chunk_mesh(&dir, 42, 1, 2).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn terrain_version_bump_invalidates_cache() {
        let dir = tmpdir();
        let v = crate::geometry::Vertex {
            pos: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            color: [1.0; 3],
            uv: [0.0, 0.0],
        };
        save_chunk_mesh(&dir, 42, 0, 0, &[v; 4], &[0u16, 1, 2, 0, 2, 3]).unwrap();
        // Same seed + current TERRAIN_VERSION loads fine...
        assert!(load_chunk_mesh(&dir, 42, 0, 0).is_some());
        // (This test breaks if you bump TERRAIN_VERSION without rebuilding
        // the cache — which is exactly the reminder you want.)
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn block_ids_cover_every_type() {
        for t in [
            BlockType::Grass,
            BlockType::Dirt,
            BlockType::Stone,
            BlockType::Log,
            BlockType::Leaves,
            BlockType::Planks,
            BlockType::CraftingTable,
            BlockType::CoalOre,
            BlockType::Air,
        ] {
            assert_eq!(BlockType::from_id(t.id()), Some(t));
        }
        assert_eq!(BlockType::from_id(200), None);
    }
}
