//! winit event-loop glue: window creation, input dispatch, cursor grab,
//! resize handling, and the redraw cadence.

use std::sync::Arc;

use glam::Vec3;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::camera::Camera;
use crate::input::InputState;
use crate::overlay::NUM_SLOTS;
use crate::player::{Player, EYE_HEIGHT};
use crate::renderer::Renderer;
use crate::terrain::{chunk_of, chunks_around, mesh_chunk, CHUNK_SIZE};

/// Camera-facing billboard data for one particle (see `particle_billboard`).
pub struct ParticleBillboard {
    pub center: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub half: f32,
    pub color: [f32; 3],
}
use crate::world::{raycast, BlockType, World, REACH};

/// Water meshes live in the renderer's separate `water_meshes` storage, so
/// they use the SAME key as their chunk's terrain mesh — no sentinel, no
/// namespace, no possible collision. (The old `(cx, -cz - 4)` sentinel
/// collided with real chunk keys south of spawn: uploading water destroyed
/// that chunk's terrain mesh — the "chunks sometimes disappear" bug.)
fn water_key((cx, cz): (i32, i32)) -> crate::renderer::MeshKey {
    (cx, cz)
}

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Camera,
    input: InputState,
    last_instant: Option<std::time::Instant>,
    backend: String,
    frames_presented: u64,
    stall_ticks: u32,
    /// Set by WindowEvent::Resized; consumed (swapchain recreated) before the
    /// next draw. Some X11 drivers never report SUBOPTIMAL after a resize, so
    /// the event — not just Vulkan errors — must drive recreation.
    resize_pending: bool,
    /// Chunk coords currently uploaded to the GPU.
    loaded_chunks: std::collections::HashSet<(i32, i32)>,
    /// True once terrain chunks have been generated (camera snapped to ground).
    terrain_ready: bool,
    /// Player physics: mode (walk/fly), gravity, collisions.
    player: Player,
    /// Editable voxel world (edit overlay + queries + raycasting).
    world: World,
    /// Hotbar: 6 slots, each `Some(item) + count` or fully empty.
    /// Slots are GENERIC (not type-locked): drops stack onto a slot holding
    /// the same item, otherwise claim the first empty slot, left → right.
    /// Indexed 0..6; `selected` points at the active slot.
    inventory: [(Option<crate::items::ItemType>, u32); NUM_SLOTS],
    /// Backpack: 18 extra slots shown in the E inventory panel (3×6 grid).
    /// Same slot semantics as the hotbar; insertion fills hotbar first.
    backpack: [(Option<crate::items::ItemType>, u32); crate::overlay::BACKPACK_SLOTS],
    /// E inventory panel state: open flag, hovered panel slot (0..24 — the
    /// last 6 map onto hotbar slots), held stack being moved, cursor pos
    /// in vertical-NDC units (y down).
    inv_open: bool,
    inv_hover: Option<usize>,
    inv_cursor: (f32, f32),
    inv_held: Option<crate::items::ItemType>,
    inv_held_count: u32,
    /// Inventory drag origin. A normal click is simply a press/release on the
    /// same slot; holding and releasing over another slot swaps or moves it.
    inv_drag_origin: Option<usize>,
    /// Inventory crafting grid: 2×2 in the ordinary inventory, 3×3 at a
    /// placed crafting table. Cells hold generic item stacks and share the
    /// cursor-held stack with the backpack.
    craft_grid: [(Option<crate::items::ItemType>, u32); 9],
    craft_size: usize,
    craft_hover: Option<usize>,
    craft_output_hover: bool,
    /// Inventory panel slot rects from the last drawn frame.
    inv_rects: [(f32, f32, f32, f32); crate::overlay::PANEL_SLOTS],
    /// Currently selected hotbar SLOT.
    selected: usize,
    /// Hold-to-mine state: progress (0..1) toward breaking the current
    /// target, and which block it points at (switching targets resets).
    mining_progress: f32,
    mining_target: Option<[i32; 3]>,
    /// Left mouse button held (charging the mine progress).
    mouse_held: bool,
    /// Floating item drops: physics + age + contents, picked up on touch.
    drops: Vec<crate::items_drop::Drop>,
    /// Break particles: physics + TTL, rendered as tinted billboards.
    particles: Vec<crate::items_drop::Particle>,
    /// FPS averaged over the last second (for the title HUD).
    fps_accum: f32,
    fps_frames: u32,
    fps_display: u32,
    /// Seconds left to show the selected-item label above the hotbar.
    slot_label_ttl: f32,
    /// Accumulated scroll-wheel lines; whole notches switch hotbar slots.
    scroll_accum: f32,
    /// Day/night cycle state (starts at mid-morning).
    day_night: crate::sky::DayNight,
    /// Graphics settings (render distance / shadows / clouds) — driven by
    /// the settings page, applied live by the renderer + terrain streamer.
    /// Loaded from the config file at startup so choices survive restarts.
    settings: crate::settings::Settings,
    /// A settings change happened this/last frame → flush to disk at the next
    /// redraw (coalesces a slider drag into at most one write per frame).
    settings_dirty: bool,
    /// Slider row (0 = render distance, 1 = cloud distance) currently being
    /// mouse-dragged, if any. While dragging, cursor movement sets the value
    /// directly from the knob position.
    drag_slider: Option<usize>,
    /// Settings page state: open flag, hovered row, the row rects from the
    /// last drawn frame (for click hit-testing), and the last known cursor
    /// position in the same (sx, y) space the rects use — MouseInput events
    /// carry no position, so the click-to-jump needs the cached one.
    settings_open: bool,
    settings_hover: Option<usize>,
    settings_cursor: (f32, f32),
    /// Settings page row rects from the last drawn frame.
    settings_rects: [(f32, f32, f32, f32); crate::overlay::SETTINGS_ROWS],
    /// F3 debug overlay on/off.
    debug_visible: bool,
    /// First-person hand animation: swing progress (0..1, one chop), whether
    /// a chop is playing, equip dip (1 = just changed item), walk-cycle phase
    /// and whether the player is moving (bob amplitude), plus the item last
    /// held (to detect changes) and the previous camera XZ (walk distance).
    hand_swing: f32,
    hand_swing_active: bool,
    hand_equip: f32,
    hand_walk: f32,
    hand_walking: bool,
    hand_last_held: Option<crate::items::ItemType>,
    hand_prev_xz: (f32, f32),
    /// Set by a successful placement; the next anim tick fires one chop.
    place_swing_pending: bool,
    /// Frame-time exponential moving average (ms) for the debug panel.
    frame_ms_ema: f32,
    /// Machine facts (CPU/RAM), read once at startup.
    sys: crate::sys::SysInfo,
    /// Wall-clock reference for the cloud drift (set at window creation).
    spawn_instant: Option<std::time::Instant>,
    /// World save directory + state. When `loaded` is true the seed/edits
    /// came from disk and a matching chunk-mesh cache exists, so streaming
    /// loads cached meshes instead of re-meshing from noise.
    save: crate::save::SaveState,
    /// Procedural sound effects (ALSA synth thread; silent if no device).
    sound: crate::sound::Sound,
    /// Distance walked since the last footstep (stride trigger).
    step_dist: f32,
    /// Set after a world edit; the next flush writes edits.bin (coalesced to
    /// at most one disk write per frame, like settings).
    world_dirty: bool,
    /// Set on window close / exit so the final save happens synchronously.
    ever_drawn: bool,
    /// Seconds since the last world autosave check.
    autosave_timer: f32,
    /// True once world.conf has been written this session. The seed must hit
    /// disk early — a player who plays five minutes and gets killed (no
    /// close event) still expects the same world next time.
    ever_saved: bool,
    /// Background chunk-meshing worker (own thread; the frame thread only
    /// drains results and uploads).
    mesher: crate::mesher::MesherWorker,
    /// Chunk requests queued on the worker but not yet drained.
    in_flight: std::collections::HashSet<(i32, i32)>,
    /// Set after the "needs a pickaxe" hint so it logs once per target.
    tier_hint_shown: bool,
}

impl App {
    pub fn new(backend: String) -> Self {
        // Resume the saved world if one exists; otherwise start a fresh seed.
        // (save::SaveState::load itself falls back to a fresh world when no
        // save directory is present.)
        let save_opt = crate::save::peek_saved_meta();
        let seed = save_opt.as_ref().map(|m| m.seed).unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() ^ ((d.subsec_nanos() as u64) << 32))
                .unwrap_or(42)
        });
        Self {
            window: None,
            renderer: None,
            camera: Camera::new(),
            input: InputState::default(),
            last_instant: None,
            backend,
            frames_presented: 0,
            stall_ticks: 0,
            resize_pending: false,
            loaded_chunks: Default::default(),
            terrain_ready: false,
            player: Player::new(),
            world: World::new(seed),
            inventory: [(Option::<crate::items::ItemType>::None, 0); NUM_SLOTS],
            backpack: [(Option::<crate::items::ItemType>::None, 0); crate::overlay::BACKPACK_SLOTS],
            inv_open: false,
            inv_hover: None,
            inv_cursor: (0.0, 0.0),
            inv_held: None,
            inv_held_count: 0,
            inv_drag_origin: None,
            craft_grid: [(Option::<crate::items::ItemType>::None, 0); 9],
            craft_size: 2,
            craft_hover: None,
            craft_output_hover: false,
            inv_rects: [(-2.0, -2.0, -2.0, -2.0); crate::overlay::PANEL_SLOTS],
            selected: 0,
            mining_progress: 0.0,
            mining_target: None,
            mouse_held: false,
            drops: Vec::new(),
            particles: Vec::new(),
            fps_accum: 0.0,
            fps_frames: 0,
            fps_display: 0,
            slot_label_ttl: 0.0,
            scroll_accum: 0.0,
            day_night: crate::sky::DayNight::new(0.08),
            // Restore the persisted settings (falls back to defaults on the
            // first run or if the file is unreadable).
            settings: crate::settings::Settings::load(),
            settings_dirty: false,
            drag_slider: None,
            settings_open: false,
            settings_hover: None,
            settings_cursor: (0.0, 0.0),
            settings_rects: [(-2.0, -2.0, -2.0, -2.0); crate::overlay::SETTINGS_ROWS],
            debug_visible: false,
            hand_swing: 0.0,
            hand_swing_active: false,
            hand_equip: 0.0,
            hand_walk: 0.0,
            hand_walking: false,
            hand_last_held: None,
            hand_prev_xz: (0.0, 0.0),
            place_swing_pending: false,
            frame_ms_ema: 16.7,
            sys: crate::sys::read(),
            spawn_instant: None,
            save: crate::save::SaveState {
                loaded: false,
                dir: crate::save::save_dir(),
            },
            sound: crate::sound::Sound::new(),
            step_dist: 0.0,
            world_dirty: false,
            ever_drawn: false,
            autosave_timer: 0.0,
            ever_saved: false,
            mesher: crate::mesher::MesherWorker::new(),
            in_flight: Default::default(),
            tier_hint_shown: false,
        }
    }

    /// Apply a loaded save right after the world is constructed: restore the
    /// edit overlay, the player's position/look/mode, and the time of day.
    /// Called from `resumed` (needs no renderer); the spawn-snap in
    /// `update_terrain` is skipped when the save carried a player position.
    fn apply_loaded_world(
        &mut self,
        meta: crate::save::Meta,
        edits: Vec<([i32; 3], crate::world::BlockType)>,
    ) {
        self.world.restore_edits(edits.iter().copied());
        self.save.loaded = true;
        if let Some(p) = meta.player {
            self.camera.pos = glam::Vec3::from(p.pos);
            self.camera.yaw = p.yaw;
            self.camera.pitch = p.pitch;
            if p.flying != self.player.is_flying() {
                self.player.toggle_mode();
            }
            self.terrain_ready = true; // skip the spawn snap
        }
        self.day_night = crate::sky::DayNight::new(meta.day_time);
        log::info!(
            "world loaded: seed {} ({} edits restored) — chunk meshes come from the cache",
            meta.seed,
            edits.len()
        );
    }

    /// Write world.conf + edits.bin (atomic; safe to call every frame — the
    /// caller coalesces via `world_dirty`). Player state and time of day ride
    /// along so a restart resumes exactly where you left off.
    fn flush_world_save(&mut self) {
        let player = crate::save::PlayerState {
            pos: self.camera.pos.to_array(),
            yaw: self.camera.yaw,
            pitch: self.camera.pitch,
            flying: self.player.is_flying(),
        };
        let dir = self.save.dir.clone();
        let seed = self.world.seed;
        let day_time = self.day_night.time;
        if let Err(e) = crate::save::save_meta(&dir, seed, &player, day_time) {
            log::warn!("world meta save failed: {e}");
        }
        match crate::save::save_edits(&dir, self.world.edits().map(|(k, t)| (*k, t))) {
            Ok(n) => log::info!("world saved: {n} edits"),
            Err(e) => log::warn!("world edits save failed: {e}"),
        }
        self.world_dirty = false;
        self.ever_saved = true;
    }

    /// Generate and upload all chunks in view radius around the camera, and
    /// unload (destroy GPU meshes for) chunks that fell out of radius+1.
    /// Edits are keyed by world coordinates in the World overlay — never by
    /// chunk — so a chunk regenerates with all player changes intact.
    unsafe fn update_terrain(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let Some(renderer) = &mut self.renderer else {
                return Ok(());
            };
            let center = chunk_of(self.camera.pos.x, self.camera.pos.z);
            let radius = self.settings.render_distance;

            // --- Background mesh results: upload whatever the worker finished.
            // Drained every frame; the upload stays on the frame thread
            // (Vulkan device handles are not Send) but meshing + cache IO are
            // not — those run on the worker now.
            let finished = self.mesher.drain();
            if !finished.is_empty() {
                let t0 = std::time::Instant::now();
                let tris: u64 = finished.iter().map(|c| (c.indices.len() / 3) as u64).sum();
                let batch: Vec<((i32, i32), Vec<crate::geometry::Vertex>, Vec<u16>, [f32; 3])> =
                    finished
                        .iter()
                        .map(|c| (c.key, c.verts.clone(), c.indices.clone(), [0.0, 0.0, 0.0]))
                        .collect();
                // Sorted near → far by the request side; upload under budget.
                const UPLOAD_BUDGET_BYTES: u64 = 6 * 1024 * 1024;
                let uploaded = renderer.upload_meshes(&batch, UPLOAD_BUDGET_BYTES)?;
                for c in finished.iter().take(uploaded) {
                    self.loaded_chunks.insert(c.key);
                }
                // Water surfaces ride their own translucent mesh per chunk.
                for c in finished.iter().take(uploaded) {
                    let wk = water_key(c.key);
                    if c.wverts.is_empty() {
                        renderer.destroy_water_mesh(wk);
                    } else {
                        renderer.upload_water_mesh(wk, &c.wverts, &c.windices, [0.0, 0.0, 0.0])?;
                    }
                }
                // Anything over budget was already drained — clear its
                // in-flight mark so the queue loop re-requests it next frame
                // (cheap: the worker then loads it from the chunk cache it
                // wrote).
                for c in finished.iter().skip(uploaded) {
                    self.in_flight.remove(&c.key);
                }
                log::info!(
                    "terrain: +{} uploaded ({} tris, {:?}) — {} loaded, {} deferred",
                    uploaded,
                    tris,
                    t0.elapsed(),
                    self.loaded_chunks.len(),
                    finished.len() - uploaded
                );
            }

            // Re-mesh chunks touched by mining/placing: these are small and
            // latency-sensitive (the edited block must appear NOW), so they
            // still mesh synchronously on the frame thread.
            let dirty = self.world.drain_dirty();
            for (cx, cz) in dirty {
                let (verts, indices, wverts, windices) =
                    mesh_chunk(&self.world, cx * CHUNK_SIZE, cz * CHUNK_SIZE);
                renderer.upload_mesh((cx, cz), &verts, &indices, [0.0, 0.0, 0.0])?;
                let wk = water_key((cx, cz));
                if !wverts.is_empty() {
                    renderer.upload_water_mesh(wk, &wverts, &windices, [0.0, 0.0, 0.0])?;
                } else {
                    renderer.destroy_water_mesh(wk);
                }
                self.loaded_chunks.insert((cx, cz));
                let _ = crate::save::save_chunk_mesh(
                    &self.save.dir,
                    self.world.seed,
                    cx,
                    cz,
                    &verts,
                    &indices,
                );
            }

            let wanted: std::collections::HashSet<(i32, i32)> =
                chunks_around(center, radius).into_iter().collect();

            // --- Unload: destroy meshes for chunks beyond the view radius +1 --
            // The +1 hysteresis band stops load/unload thrash when straddling
            // a chunk border.
            let unload_radius = radius + 1;
            let (pcx, pcz) = center;
            let stale: Vec<(i32, i32)> = self
                .loaded_chunks
                .iter()
                .filter(|&&(cx, cz)| {
                    (cx - pcx).abs() > unload_radius || (cz - pcz).abs() > unload_radius
                })
                .copied()
                .collect();
            if !stale.is_empty() {
                // One batched destroy (single device idle) instead of a wait
                // per chunk. Water meshes share the chunk's key in their own
                // storage — destroyed alongside.
                renderer.destroy_meshes(&stale);
                for &key in &stale {
                    renderer.destroy_water_mesh(key);
                    self.loaded_chunks.remove(&key);
                }
                log::info!("terrain: unloaded {} out-of-range chunks", stale.len());
            }

            // --- Load: queue missing chunks on the background worker.
            // Also drop in-flight requests for chunks that left the wanted
            // set while they were queued (fast camera turns).
            self.in_flight.retain(|k| wanted.contains(k));
            let mut new_chunks: Vec<(i32, i32)> = wanted
                .iter()
                .filter(|c| !self.loaded_chunks.contains(*c) && !self.in_flight.contains(*c))
                .copied()
                .collect();
            if new_chunks.is_empty() {
                return Ok(());
            }

            // First-ever load: snap the camera above the terrain surface so we
            // spawn looking at the world, not from inside a hill.
            if !self.terrain_ready {
                self.terrain_ready = true;
                let (cx, cz) = center;
                let mut wx = cx * CHUNK_SIZE + CHUNK_SIZE / 2;
                let mut wz = cz * CHUNK_SIZE + CHUNK_SIZE / 2;
                // Top of the world column (tree trunk or terrain surface),
                // never inside a tree. If the column is a lake (water is
                // non-solid, so ground_under walks to the floor), spawn on
                // the nearest shore instead of drowning at the lake bottom.
                let mut h = self.world.ground_under(wx, wz, 64.0);
                if h <= crate::world::SEA_LEVEL as f32 {
                    'shore: for r in 1..40u32 {
                        for (dx, dz) in [
                            (r as i32, 0),
                            (-(r as i32), 0),
                            (0, r as i32),
                            (0, -(r as i32)),
                        ] {
                            let sx = wx + dx;
                            let sz = wz + dz;
                            let sh = self.world.ground_under(sx, sz, 64.0);
                            if sh > crate::world::SEA_LEVEL as f32 {
                                wx = sx;
                                wz = sz;
                                h = sh;
                                break 'shore;
                            }
                        }
                    }
                }
                self.camera.pos = glam::Vec3::new(wx as f32 + 0.5, h + EYE_HEIGHT, wz as f32 + 0.5);
                self.camera.yaw = std::f32::consts::FRAC_PI_2;
                self.camera.pitch = -0.25; // look slightly down at the terrain
                log::info!("spawn at ({wx}, {h}, {wz}) — walking mode");
            }

            // Near → far: the terrain you stand in resolves first.
            new_chunks.sort_by_key(|&(cx, cz)| {
                let dx = cx - center.0;
                let dz = cz - center.1;
                dx * dx + dz * dz
            });
            // Queue a bounded number per frame so a 32-radius jump doesn't
            // stampede the worker with 4000 requests ahead of the near ones.
            const MAX_QUEUED_PER_FRAME: usize = 64;
            let world_snapshot = self.world.snapshot();
            let mut queued = 0usize;
            for &(cx, cz) in &new_chunks {
                if queued >= MAX_QUEUED_PER_FRAME || self.mesher.pending() >= MAX_QUEUED_PER_FRAME {
                    break;
                }
                // The worker handles the chunk-cache read/write itself.
                if self.mesher.request(
                    (cx, cz),
                    world_snapshot.snapshot(),
                    cx * CHUNK_SIZE,
                    cz * CHUNK_SIZE,
                    self.save.dir.clone(),
                ) {
                    self.in_flight.insert((cx, cz));
                    queued += 1;
                }
            }
        }
        Ok(())
    }

    /// Hold-to-mine: advance progress on the targeted block. Called every
    /// frame while the left button is held; when progress completes, the
    /// block breaks into a floating item drop + a burst of particles.
    fn update_mining(&mut self, dt: f32) {
        let hit = raycast(&self.world, self.camera.pos, self.camera.forward(), REACH);
        let target = hit.as_ref().map(|h| h.block);
        if !self.mouse_held {
            self.mining_progress = 0.0;
            self.mining_target = None;
            return;
        }
        // Switching targets (or losing the target) resets the crack progress.
        if self.mining_target != target {
            self.mining_progress = 0.0;
            self.mining_target = target;
            self.tier_hint_shown = false;
        }
        let Some([x, y, z]) = target else { return };
        let block = match self.world.block(x, y, z) {
            crate::world::Block::Solid(t) => t,
            crate::world::Block::Air => return,
        };
        let held = self.inventory[self.selected].0;
        // Mining tier gate: stone-family blocks produce nothing without a
        // pickaxe (the crack animation stalls at 95% — same as MC where the
        // block never drops). The hint logs once per target switch.
        if let Some(reason) = crate::items::tier_block_reason(block, held) {
            self.mining_progress = (self.mining_progress + dt / 60.0).min(0.95);
            if !self.tier_hint_shown {
                self.tier_hint_shown = true;
                log::info!(
                    "{} {} — craft a pickaxe at a crafting table",
                    block.name(),
                    reason
                );
            }
            return;
        }
        let total = crate::items::break_time(block, held);
        self.mining_progress += dt / total;
        if self.mining_progress >= 1.0 {
            self.mining_progress = 0.0;
            self.mining_target = None;
            self.break_block([x, y, z], block);
        }
    }

    /// Break a block: remove it from the world, spawn the drop + particles.
    fn break_block(&mut self, [x, y, z]: [i32; 3], block: BlockType) {
        let Some(dropped) = self.world.mine(x, y, z) else {
            return;
        };
        self.world_dirty = true;
        // The block becomes a floating item drop (Minecraft-style) that the
        // player collects by walking near it. Particles fly in the broken
        // block's own color (grass puffs green-tinted, drops are dirt).
        let item = crate::items::ItemType::Block(dropped);
        let n = self.drops.len() as f32; // deterministic spread per drop
        self.drops.push(crate::items_drop::Drop {
            item,
            pos: glam::vec3(x as f32 + 0.5, y as f32 + 0.55, z as f32 + 0.5),
            vel: glam::vec3(
                ((n * 37.0) % 1.0 - 0.5) * 1.2,
                2.2,
                ((n * 73.0) % 1.0 - 0.5) * 1.2,
            ),
            age: 0.0,
        });
        self.spawn_break_particles(x, y, z, block.srgb());
        self.sound.block_break(crate::sound::block_material(block));
    }

    /// Spawn the particle burst for a broken block (center puff outward).
    fn spawn_break_particles(&mut self, x: i32, y: i32, z: i32, color: [f32; 3]) {
        let cx = x as f32 + 0.5;
        let cy = y as f32 + 0.5;
        let cz = z as f32 + 0.5;
        for i in 0..14 {
            let a = i as f32 * 0.45;
            let r = 0.3 + (i % 3) as f32 * 0.12;
            self.particles.push(crate::items_drop::Particle {
                pos: glam::vec3(
                    cx + (a.sin()) * r * 0.4,
                    cy + (i % 4) as f32 * 0.12,
                    cz + a.cos() * r * 0.4,
                ),
                vel: glam::vec3(
                    a.sin() * r * 2.2,
                    1.8 + (i % 3) as f32 * 0.7,
                    a.cos() * r * 2.2,
                ),
                ttl: 0.8 + (i % 5) as f32 * 0.12,
                age: 0.0,
                color,
                half: 0.045,
            });
        }
    }

    /// Physics + pickup for drops and particles. Drops fall, settle on the
    /// ground, bob/spin, and fly to the player when close (magnet), then
    /// enter the inventory. Particles just fall and fade.
    fn update_drops(&mut self, dt: f32) {
        let eye = self.camera.pos;
        let feet = eye.y - EYE_HEIGHT;
        // Magnet + pickup: within 1.4 the drop flies at the player; within
        // 0.5 of the eye/body it's collected.
        let mut collected: Vec<crate::items::ItemType> = Vec::new();
        for d in self.drops.iter_mut() {
            d.age += dt;
            let to_player = eye - d.pos;
            let dist = to_player.length();
            if dist < 1.6 {
                // Magnet: accelerate toward the player's chest.
                d.vel += to_player.normalize() * 26.0 * dt;
                if dist < 0.55 {
                    collected.push(d.item);
                    d.age = f32::INFINITY; // marked for removal
                    continue;
                }
            } else {
                d.vel.y -= 22.0 * dt; // gravity
            }
            d.vel.x *= (1.0 - 2.5 * dt).max(0.0);
            d.vel.z *= (1.0 - 2.5 * dt).max(0.0);
            d.pos += d.vel * dt;
            // Ground settle: rest on the surface UNDER the drop — only while
            // falling, and only if the surface is at/below the drop. (The
            // old unconditional snap LIFTED the drop onto whatever the
            // column scan found — mine a log and the trunk above it made the
            // drop teleport to the treetop.)
            let gx = d.pos.x.floor() as i32;
            let gz = d.pos.z.floor() as i32;
            if self.world.solid(gx, d.pos.y.floor() as i32, gz) {
                // Clipped into a solid (e.g. slid into a trunk): hold still —
                // popping up the column would re-create the treetop climb.
                d.vel.y = 0.0;
            } else {
                let ground = self.world.ground_under(gx, gz, d.pos.y) as f32;
                if d.vel.y <= 0.0 && d.pos.y >= ground && d.pos.y < ground + 0.25 {
                    d.pos.y = ground + 0.25;
                    d.vel.y = 0.0;
                }
            }
        }
        self.drops.retain(|d| d.age.is_finite() && d.age < 300.0);
        if !collected.is_empty() {
            self.sound.pickup();
        }
        for item in collected {
            match self.inv_add(item) {
                Some(("h", i)) if i < NUM_SLOTS => log::info!(
                    "picked up {} (hotbar slot {} now has {})",
                    item.name(),
                    i + 1,
                    self.inventory[i].1
                ),
                Some(("b", i)) if i < crate::overlay::BACKPACK_SLOTS => log::info!(
                    "picked up {} (backpack slot {} now has {})",
                    item.name(),
                    i + 1,
                    self.backpack[i].1
                ),
                _ => {
                    // Inventory full: re-drop the item right where it was.
                    log::info!("inventory full — {} left on the ground", item.name());
                    self.drops.push(crate::items_drop::Drop {
                        item,
                        pos: eye + glam::vec3(0.0, -1.2, 0.0),
                        vel: glam::vec3(0.0, 0.5, 0.0),
                        age: 0.0,
                    });
                }
            }
        }

        for p in self.particles.iter_mut() {
            p.age += dt;
            p.vel.y -= 18.0 * dt;
            p.pos += p.vel * dt;
            let g = self.world.ground_under(
                p.pos.x.floor() as i32,
                p.pos.z.floor() as i32,
                p.pos.y + 1.0,
            ) as f32;
            if p.pos.y < g + 0.05 {
                p.pos.y = g + 0.05;
                p.vel.y = 0.0;
                p.vel.x *= 0.7;
                p.vel.z *= 0.7;
            }
        }
        self.particles.retain(|p| p.age < p.ttl);
        let _ = feet;
    }

    /// Add one item to the inventory: stack onto an existing slot of that
    /// type (hotbar first, then backpack), else claim the first EMPTY slot
    /// left → right. Returns `Some(("h"|"b", index))` or None if totally
    /// full (the item stays on the ground as a drop — Minecraft-style).

    /// Apply a slider drag: map the cursor's horizontal position (in the
    /// settings panel's vertical-NDC space, x already aspect-scaled) onto the
    /// row's slider track and set the value. Works for both rows.
    fn apply_slider_drag(&mut self, row: usize, sx: f32) {
        use crate::overlay::{SETTINGS_CTRL_X, SETTINGS_TRACK_W};
        let (lo, hi) = if row == 0 { (1, 32) } else { (2, 10) };
        let frac = ((sx - SETTINGS_CTRL_X) / SETTINGS_TRACK_W).clamp(0.0, 1.0);
        // Round to the nearest step so the value lands exactly on integers,
        // matching the wheel stepper behavior.
        let v = lo as f32 + frac * (hi - lo) as f32;
        let v = v.round() as i32;
        let changed = if row == 0 {
            self.settings.render_distance != v
        } else {
            self.settings.cloud_distance != v
        };
        if changed {
            if row == 0 {
                self.settings.set_render_distance(v);
            } else {
                self.settings.set_cloud_distance(v);
            }
            self.settings_dirty = true;
        }
    }

    /// Add or remove a whole stack in the active crafting grid. The grid uses
    /// the same pick-up/place/swap semantics as inventory slots.
    fn craft_slot_click(&mut self, idx: usize) {
        // The grid is a size×size sub-square of a stride-3 backing array:
        // 2×2 occupies indices 0,1,3,4. A dense `idx >= size*size` guard
        // rejects index 4 (= size²) — the bottom-right cell silently ate
        // every item and made the table recipe impossible.
        let (row, col) = (idx / 3, idx % 3);
        if row >= self.craft_size || col >= self.craft_size {
            return;
        }
        let slot = &mut self.craft_grid[idx];
        match (self.inv_held, *slot) {
            (None, (Some(item), n)) => {
                *slot = (None, 0);
                self.inv_held = Some(item);
                self.inv_held_count = n;
            }
            (Some(item), (None, _)) => {
                *slot = (Some(item), self.inv_held_count);
                self.inv_held = None;
                self.inv_held_count = 0;
            }
            (Some(item), (Some(cur), n)) if item == cur => {
                *slot = (Some(cur), n + self.inv_held_count);
                self.inv_held = None;
                self.inv_held_count = 0;
            }
            (Some(item), (Some(cur), n)) => {
                *slot = (Some(item), self.inv_held_count);
                self.inv_held = Some(cur);
                self.inv_held_count = n;
            }
            _ => {}
        }
    }

    /// Consume one matching shaped recipe from the grid and add its output.
    /// Inputs remain in the grid until the output slot is clicked, making the
    /// crafting result predictable and avoiding accidental item loss.
    fn craft_grid_output(&mut self) {
        let Some(recipe) =
            crate::items::matching_recipe(&self.craft_grid, self.craft_size, self.craft_size == 3)
        else {
            return;
        };
        if self.inv_held.is_some() {
            return;
        }

        let old_inventory = self.inventory;
        let old_backpack = self.backpack;
        let old_grid = self.craft_grid;
        for row in 0..self.craft_size {
            for col in 0..self.craft_size {
                let i = row * 3 + col;
                if self.craft_grid[i].0.is_some() {
                    self.craft_grid[i].1 -= 1;
                    if self.craft_grid[i].1 == 0 {
                        self.craft_grid[i].0 = None;
                    }
                }
            }
        }
        let (out_item, out_count) = recipe.output;
        if self.inv_add_n(out_item, out_count).is_none() {
            self.inventory = old_inventory;
            self.backpack = old_backpack;
            self.craft_grid = old_grid;
            return;
        }
    }

    /// Add a complete stack to one compatible slot. Stacks are intentionally
    /// unbounded in this prototype, so a single existing or empty slot always
    /// accepts the full quantity. Hotbar is preferred over backpack.
    fn inv_add_n(
        &mut self,
        t: crate::items::ItemType,
        count: u32,
    ) -> Option<(&'static str, usize)> {
        if count == 0 {
            return None;
        }
        if let Some(i) = self.inventory.iter().position(|&(ty, _)| ty == Some(t)) {
            self.inventory[i].1 = self.inventory[i].1.saturating_add(count);
            return Some(("h", i));
        }
        if let Some(i) = self.backpack.iter().position(|&(ty, _)| ty == Some(t)) {
            self.backpack[i].1 = self.backpack[i].1.saturating_add(count);
            return Some(("b", i));
        }
        if let Some(i) = self.inventory.iter().position(|s| s.0.is_none()) {
            self.inventory[i] = (Some(t), count);
            return Some(("h", i));
        }
        if let Some(i) = self.backpack.iter().position(|s| s.0.is_none()) {
            self.backpack[i] = (Some(t), count);
            return Some(("b", i));
        }
        None
    }

    /// Add one dropped item using the same generic-slot insertion policy.
    fn inv_add(&mut self, t: crate::items::ItemType) -> Option<(&'static str, usize)> {
        self.inv_add_n(t, 1)
    }

    /// Close the inventory and return every cursor-held and crafting-grid
    /// stack to the normal inventory. This prevents ingredients disappearing
    /// when the player presses E or Escape halfway through a recipe.
    fn close_inventory(&mut self) {
        if let Some(item) = self.inv_held.take() {
            let count = self.inv_held_count;
            self.inv_held_count = 0;
            self.stash_back(item, count);
        }
        let grid_items = self
            .craft_grid
            .iter_mut()
            .filter_map(|slot| {
                let item = slot.0.take()?;
                let count = slot.1;
                slot.1 = 0;
                Some((item, count))
            })
            .collect::<Vec<_>>();
        for (item, count) in grid_items {
            self.stash_back(item, count);
        }
        self.craft_hover = None;
        self.craft_output_hover = false;
        self.inv_drag_origin = None;
        self.inv_open = false;
        self.craft_size = 2;
        self.grab_cursor();
    }

    /// Open the inventory with a selected crafting-grid size. Table opening
    /// uses 3×3; ordinary E opening uses 2×2.
    fn open_inventory(&mut self, size: usize) {
        self.inv_open = true;
        self.craft_size = size.clamp(2, 3);
        self.release_cursor();
    }

    /// Place from the selected slot (right click), consuming one block. When
    /// a slot empties it becomes free; later drops shift nothing — the next
    /// mined type simply claims the first empty slot. Tools and sticks are
    /// not placeable.
    fn place_targeted(&mut self) {
        let Some(&(ty, count)) = self.inventory.get(self.selected) else {
            return;
        };
        let Some(crate::items::ItemType::Block(t)) = ty else {
            log::info!(
                "{} can't be placed",
                ty.map(|i| i.name()).unwrap_or("nothing")
            );
            return;
        };
        if count == 0 {
            return;
        }
        let hit = raycast(&self.world, self.camera.pos, self.camera.forward(), REACH);
        let Some(hit) = hit else { return };
        let [x, y, z] = hit.adjacent;

        // Never place a block inside the player's own volume.
        let feet = self.camera.pos.y - EYE_HEIGHT;
        let (px, pz) = (self.camera.pos.x, self.camera.pos.z);
        let overlaps_player = (px.floor() as i32 == x && pz.floor() as i32 == z)
            && y + 1 > feet.floor() as i32
            && y < (feet + EYE_HEIGHT).ceil() as i32;
        if overlaps_player {
            log::info!("can't place a block inside yourself");
            return;
        }

        if self.world.place(x, y, z, t) {
            self.world_dirty = true;
            self.place_swing_pending = true; // one chop of the hand
            self.sound.block_place(crate::sound::block_material(t));
            self.inventory[self.selected].1 -= 1;
            if self.inventory[self.selected].1 == 0 {
                // The slot is spent — it becomes free for the next new type.
                self.inventory[self.selected].0 = None;
                log::info!(
                    "placed {} — slot {} is now empty",
                    t.name(),
                    self.selected + 1
                );
            } else {
                log::info!(
                    "placed {} at ({x}, {y}, {z}) — {} left",
                    t.name(),
                    self.inventory[self.selected].1
                );
            }
        }
    }

    /// Right-click on a panel slot with a stack on the cursor: drop ONE item
    /// into it (Minecraft-style single-item placement). With an empty cursor
    /// it picks up half the stack (rounded up), like Minecraft.
    fn inv_slot_right_click(&mut self, idx: usize) {
        let hotbar_slot = idx >= crate::overlay::BACKPACK_SLOTS;
        let i = if hotbar_slot {
            idx - crate::overlay::BACKPACK_SLOTS
        } else {
            idx
        };
        let (area, other): (&mut [(Option<crate::items::ItemType>, u32)], bool) = if hotbar_slot {
            (&mut self.inventory, true)
        } else {
            (&mut self.backpack, false)
        };
        let _ = other;
        let slot = &mut area[i];
        match (self.inv_held, *slot) {
            // Held + empty/same-type: place exactly one.
            (Some(held), (cur, n)) if cur.is_none() || cur == Some(held) => {
                *slot = (Some(held), n + 1);
                self.inv_held_count -= 1;
                if self.inv_held_count == 0 {
                    self.inv_held = None;
                }
            }
            // Empty hand + stack: pick up half (ceil).
            (None, (Some(cur), n)) => {
                let take = n.div_ceil(2);
                *slot = (Some(cur), n - take);
                if slot.1 == 0 {
                    slot.0 = None;
                }
                self.inv_held = Some(cur);
                self.inv_held_count = take;
            }
            _ => {}
        }
    }

    /// Right-click on a crafting-grid cell: the same single-item semantics,
    /// so a recipe can be loaded cell by cell from a cursor stack.
    fn craft_slot_right_click(&mut self, idx: usize) {
        // Same stride-3 geometry as craft_slot_click — see the comment there.
        let (row, col) = (idx / 3, idx % 3);
        if row >= self.craft_size || col >= self.craft_size {
            return;
        }
        let slot = &mut self.craft_grid[idx];
        match (self.inv_held, *slot) {
            (Some(held), (cur, n)) if cur.is_none() || cur == Some(held) => {
                *slot = (Some(held), n + 1);
                self.inv_held_count -= 1;
                if self.inv_held_count == 0 {
                    self.inv_held = None;
                }
            }
            (None, (Some(cur), n)) => {
                let take = n.div_ceil(2);
                *slot = (Some(cur), n - take);
                if slot.1 == 0 {
                    slot.0 = None;
                }
                self.inv_held = Some(cur);
                self.inv_held_count = take;
            }
            _ => {}
        }
    }

    /// A click on a panel slot: pick up the stack, drop the held stack in,
    /// or swap. `idx` 0..18 = backpack, 18..24 = hotbar slots 0..6.
    fn inv_slot_click(&mut self, idx: usize) {
        let hotbar_slot = idx >= crate::overlay::BACKPACK_SLOTS;
        let i = if hotbar_slot {
            idx - crate::overlay::BACKPACK_SLOTS
        } else {
            idx
        };
        let slot = if hotbar_slot {
            &mut self.inventory[i]
        } else {
            &mut self.backpack[i]
        };

        match (self.inv_held, *slot) {
            // Empty hand + item: pick the whole stack up.
            (None, (Some(item), n)) => {
                *slot = (None, 0);
                self.inv_held = Some(item);
                self.inv_held_count = n;
            }
            // Held + empty slot: place everything.
            (Some(_), (None, _)) => {
                let (item, n) = (self.inv_held.take().unwrap(), self.inv_held_count);
                *slot = (Some(item), n);
                self.inv_held_count = 0;
            }
            // Same type: merge (held pours into the slot).
            (Some(held), (Some(cur), n)) if held == cur => {
                *slot = (Some(cur), n + self.inv_held_count);
                self.inv_held = None;
                self.inv_held_count = 0;
            }
            // Different types: swap.
            (Some(held), (Some(cur), n)) => {
                *slot = (Some(held), self.inv_held_count);
                self.inv_held = Some(cur);
                self.inv_held_count = n;
            }
            _ => {}
        }
    }

    /// Return a held stack to any free slot (hotbar first, then backpack);
    /// fills the first empty one entirely.
    fn stash_back(&mut self, item: crate::items::ItemType, n: u32) {
        // Prefer stacking onto the same type.
        if let Some((w, i)) = self
            .inventory
            .iter()
            .position(|&(ty, _)| ty == Some(item))
            .map(|i| ("h", i))
            .or_else(|| {
                self.backpack
                    .iter()
                    .position(|&(ty, _)| ty == Some(item))
                    .map(|i| ("b", i))
            })
        {
            if w == "h" {
                self.inventory[i].1 += n;
            } else {
                self.backpack[i].1 += n;
            }
            return;
        }
        if let Some(i) = self.inventory.iter().position(|s| s.0.is_none()) {
            self.inventory[i] = (Some(item), n);
        } else if let Some(i) = self.backpack.iter().position(|s| s.0.is_none()) {
            self.backpack[i] = (Some(item), n);
        } else {
            log::info!("no room — {} ×{} lost", item.name(), n);
        }
    }

    /// First-person hand animation: advance the swing (mining repeats it;
    /// placing fires a single chop), ease the equip dip back to 0, and
    /// accumulate the walk-cycle phase from horizontal movement.
    fn update_hand_anim(&mut self, dt: f32) {
        const SWING_TIME: f32 = 0.28; // seconds per chop
                                      // Mining hold: loop the swing. Triggered by mouse_held with a target.
        if self.mouse_held && self.mining_target.is_some() {
            if !self.hand_swing_active {
                self.hand_swing_active = true;
                self.hand_swing = 0.0;
            }
        }
        if self.hand_swing_active {
            self.hand_swing += dt / SWING_TIME;
            if self.hand_swing >= 1.0 {
                if self.mouse_held && self.mining_target.is_some() {
                    self.hand_swing -= 1.0; // loop while mining
                } else {
                    self.hand_swing = 0.0;
                    self.hand_swing_active = false;
                }
            }
        }
        // One-shot chop on placing.
        if self.place_swing_pending {
            self.place_swing_pending = false;
            self.hand_swing = 0.0;
            self.hand_swing_active = true;
        }
        // Equip dip eases back down.
        self.hand_equip = (self.hand_equip - dt * 4.0).max(0.0);
        // Walk bob: accumulate phase from actual horizontal movement.
        let (x, z) = (self.camera.pos.x, self.camera.pos.z);
        let dx = x - self.hand_prev_xz.0;
        let dz = z - self.hand_prev_xz.1;
        self.hand_prev_xz = (x, z);
        let moved = (dx * dx + dz * dz).sqrt();
        self.hand_walking = moved > 0.001 && !self.player.is_flying();
        if self.hand_walking {
            self.hand_walk = (self.hand_walk + moved / 1.8) % 1.0; // stride ≈ 1.8 blocks
                                                                   // Footsteps: one per stride, flavored by the ground underfoot.
            self.step_dist += moved;
            if self.step_dist >= 1.8 {
                self.step_dist = 0.0;
                let ground_y =
                    self.world
                        .ground_under(x.floor() as i32, z.floor() as i32, self.camera.pos.y);
                if let crate::world::Block::Solid(b) =
                    self.world
                        .block(x.floor() as i32, ground_y as i32 - 1, z.floor() as i32)
                {
                    self.sound.footstep(crate::sound::block_material(b));
                }
            }
        }
    }

    /// 1..6 (keys) or the scroll wheel select the hotbar SLOT (not a fixed
    /// type). The name label above the bar re-appears for 2 s.
    fn select_slot(&mut self, slot: usize) {
        self.selected = slot;
        self.slot_label_ttl = 2.0;
        match self.inventory[slot] {
            (Some(t), n) => log::info!("slot {}: {} × {}", slot + 1, n, t.name()),
            (None, _) => log::info!("slot {}: empty", slot + 1),
        }
    }

    /// Quick-craft a canonical shaped recipe by its discovery key. The same
    /// `CRAFT_RECIPES` table drives the visible grid, so shortcuts cannot
    /// silently use different ingredients. Tool recipes require the 3×3 table
    /// UI to be open; the log/planks/table recipes work from the inventory.
    #[allow(dead_code)]
    fn craft(&mut self, key: char) {
        let recipe = match crate::items::CRAFT_RECIPES.iter().find(|r| r.key == key) {
            Some(recipe) => recipe,
            None => return,
        };
        if recipe.requires_table && !(self.inv_open && self.craft_size == 3) {
            log::info!("{} requires an open crafting table", recipe.name);
            return;
        }

        let ingredients = crate::items::recipe_ingredients(recipe);
        for &(item, needed) in &ingredients {
            let available = self
                .inventory
                .iter()
                .chain(self.backpack.iter())
                .filter(|&&(ty, _)| ty == Some(item))
                .map(|&(_, count)| count)
                .sum::<u32>();
            if available < needed {
                log::info!("crafting {} needs {} {}", recipe.name, needed, item.name());
                return;
            }
        }

        // Consume first, then add the output. Keep a snapshot so an unusual
        // full-inventory edge case can never destroy the ingredients.
        let old_inventory = self.inventory;
        let old_backpack = self.backpack;
        for &(item, needed) in &ingredients {
            self.consume_inventory_item(item, needed);
        }
        let (out_item, out_count) = recipe.output;
        if self.inv_add_n(out_item, out_count).is_none() {
            self.inventory = old_inventory;
            self.backpack = old_backpack;
            log::info!("inventory full — crafted items not consumed");
            return;
        }
        log::info!("crafted {} ×{}", out_item.name(), out_count);
    }

    /// Remove a quantity from matching inventory stacks, hotbar first. The
    /// caller has already verified that the total quantity is available.
    #[allow(dead_code)]
    fn consume_inventory_item(&mut self, item: crate::items::ItemType, mut count: u32) {
        for slot in self.inventory.iter_mut().chain(self.backpack.iter_mut()) {
            if count == 0 {
                break;
            }
            if slot.0 != Some(item) {
                continue;
            }
            let taken = slot.1.min(count);
            slot.1 -= taken;
            count -= taken;
            if slot.1 == 0 {
                slot.0 = None;
            }
        }
    }

    /// Build the compact HUD string: per-slot contents (type:count), mode,
    /// fps, position.
    fn hud_title(&self) -> String {
        let parts: Vec<String> = self
            .inventory
            .iter()
            .enumerate()
            .map(|(i, &(ty, n))| match ty {
                Some(t) => format!("{}.{}:{}", i + 1, t.name(), n),
                None => format!("{}.—", i + 1),
            })
            .collect();
        let sel = match self.inventory[self.selected] {
            (Some(t), _) => t.name().to_string(),
            (None, _) => "empty".to_string(),
        };
        let mode = if self.player.is_flying() {
            "fly"
        } else {
            "walk"
        };
        format!(
            "VoxelCraft [{}] | {} | {} | {} | {} fps | ({:.0},{:.0},{:.0})",
            self.backend,
            sel,
            parts.join(" "),
            mode,
            self.fps_display,
            self.camera.pos.x,
            self.camera.pos.y,
            self.camera.pos.z,
        )
    }

    fn release_cursor(&mut self) {
        self.input.cursor_grabbed = false;
        self.input.keys_held.clear();
        if let Some(window) = &self.window {
            window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .ok();
            window.set_cursor_visible(true);
        }
    }

    fn grab_cursor(&mut self) {
        // Wayland ignores Confined-without-absolute-positioning; Lock is what
        // works there. On platforms that lack Lock, fall back to Confined.
        if let Some(window) = &self.window {
            let grabbed = window
                .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
            if grabbed.is_ok() {
                self.input.cursor_grabbed = true;
                window.set_cursor_visible(false);
            } else {
                log::warn!("could not grab cursor; mouse look disabled — press F to retry");
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Destroy Vulkan resources while the window is still alive.
        if let Some(renderer) = &mut self.renderer {
            unsafe { renderer.destroy() };
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(format!(
                "VoxelCraft v{} [{}]",
                env!("CARGO_PKG_VERSION"),
                self.backend
            ))
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let renderer = unsafe { Renderer::new(&window) }.expect("init renderer");
        // Restore the saved world (if any) before the first terrain update
        // so cached chunk meshes and restored edits line up from frame one.
        if !self.save.loaded {
            if let Some(meta) = crate::save::load_meta(&self.save.dir) {
                if let Some(edits) = crate::save::load_edits(&self.save.dir) {
                    self.apply_loaded_world(meta, edits);
                }
            }
        }
        self.window = Some(window);
        self.renderer = Some(renderer);
        self.camera = Camera::new();
        self.last_instant = Some(std::time::Instant::now());
        self.spawn_instant = Some(std::time::Instant::now());
        self.grab_cursor();
        if let Err(e) = unsafe { self.update_terrain() } {
            log::error!("initial terrain generation failed: {e}");
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                // Final synchronous save while the window/Vulkan state is
                // still alive; chunk caches were written as chunks streamed.
                if self.ever_drawn {
                    self.flush_world_save();
                }
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == winit::event::ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    if pressed && code == KeyCode::KeyG {
                        self.player.toggle_mode();
                        log::info!(
                            "mode: {}",
                            if self.player.is_flying() {
                                "flying"
                            } else {
                                "walking"
                            }
                        );
                    }
                    // E toggles the inventory panel; the cursor is released
                    // while it's open (and re-grabbed when it closes).
                    if pressed && code == KeyCode::KeyE {
                        if self.inv_open {
                            self.close_inventory();
                        } else {
                            self.open_inventory(2);
                        }
                    }
                    // O toggles the settings page; cursor released while open.
                    if pressed && code == KeyCode::KeyO {
                        self.settings_open = !self.settings_open;
                        if self.settings_open {
                            self.release_cursor();
                        } else {
                            self.grab_cursor();
                        }
                    }
                    // Closing the settings page (O, ESC) cancels any slider
                    // drag in progress.
                    if pressed && (code == KeyCode::KeyO || code == KeyCode::Escape) {
                        self.drag_slider = None;
                    }
                    if pressed && code == KeyCode::F3 {
                        self.debug_visible = !self.debug_visible;
                    }
                    if pressed {
                        match code {
                            KeyCode::Digit1 => self.select_slot(0),
                            KeyCode::Digit2 => self.select_slot(1),
                            KeyCode::Digit3 => self.select_slot(2),
                            KeyCode::Digit4 => self.select_slot(3),
                            KeyCode::Digit5 => self.select_slot(4),
                            KeyCode::Digit6 => self.select_slot(5),
                            KeyCode::Digit7 => self.select_slot(6),
                            _ => {}
                        }
                    }
                    if pressed && code == KeyCode::KeyF && !self.input.cursor_grabbed {
                        self.grab_cursor(); // retry mouse capture
                    }
                    if pressed && code == KeyCode::BracketLeft {
                        let s = self.input.scale_sensitivity(0.85);
                        log::info!("mouse sensitivity: {s:.5} rad/px");
                    }
                    if pressed && code == KeyCode::BracketRight {
                        let s = self.input.scale_sensitivity(1.18);
                        log::info!("mouse sensitivity: {s:.5} rad/px");
                    }
                    if pressed && code == KeyCode::Escape {
                        if self.settings_open {
                            // ESC closes the settings page and returns to the
                            // game (mouse re-grabbed).
                            self.settings_open = false;
                            self.grab_cursor();
                        } else if self.inv_open {
                            // ESC also closes the inventory panel and returns
                            // its grid/cursor-held items safely.
                            self.close_inventory();
                        } else {
                            // ESC in game: open the settings page and release
                            // the mouse (ESC again closes it, per Minecraft).
                            // Quitting is the window's close button.
                            self.settings_open = true;
                            self.release_cursor();
                        }
                    }
                    if pressed {
                        self.input.keys_held.insert(code);
                    } else {
                        self.input.keys_held.remove(&code);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Track the cursor for the E inventory panel and the O
                // settings page (NDC, y down). Hit-testing uses x scaled by
                // the aspect — the rects are drawn in vertical-NDC units
                // where screen x = ndc_x × aspect, so comparing raw ndc x
                // against them was wrong on non-square windows.
                if self.inv_open || self.settings_open {
                    if let Some(window) = &self.window {
                        let size = window.inner_size();
                        let (w, h) = (size.width.max(1) as f64, size.height.max(1) as f64);
                        let x = (position.x / w * 2.0 - 1.0) as f32;
                        // HUD coords are y-DOWN (Vulkan NDC): screen top = -1.
                        // The old `1.0 - y/h*2.0` flipped this, so hover and
                        // clicks hit the vertically MIRRORED position — settings
                        // rows toggled nothing and the slider only worked while
                        // pointing at the wrong spot.
                        let y = (position.y / h * 2.0 - 1.0) as f32;
                        let aspect = (w / h).max(0.0001) as f32;
                        if self.inv_open {
                            self.inv_cursor = (x, y);
                            let sx = x * aspect;
                            self.inv_hover = self.inv_rects.iter().position(|&(x0, y0, x1, y1)| {
                                sx >= x0 && sx <= x1 && y >= y0 && y <= y1
                            });
                            self.craft_hover =
                                crate::overlay::craft_cell_at(self.craft_size, sx, y);
                            let (ox0, oy0, ox1, oy1) =
                                crate::overlay::crafting_output_rect(self.craft_size);
                            self.craft_output_hover =
                                sx >= ox0 && sx <= ox1 && y >= oy0 && y <= oy1;
                        }
                        if self.settings_open {
                            let sx = x * aspect;
                            self.settings_cursor = (sx, y);
                            self.settings_hover =
                                self.settings_rects.iter().position(|&(x0, y0, x1, y1)| {
                                    sx >= x0 && sx <= x1 && y >= y0 && y <= y1
                                });
                            // Slider rows highlight when the cursor is over
                            // the TRACK, not just anywhere on the row.
                            if crate::overlay::slider_hover(self.settings_hover, sx, y).is_some() {
                                self.settings_hover =
                                    crate::overlay::slider_hover(self.settings_hover, sx, y);
                            }
                            if let Some(row) = self.drag_slider {
                                self.apply_slider_drag(row, sx);
                            }
                        }
                    }
                }
            }
            // Scroll wheel cycles the hotbar (a notch = one slot; browsers
            // report ~±1 line per detent, so only whole lines act).
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => (p.y / 40.0) as f32,
                }; // Settings page open: the wheel drives whichever row is
                   // hovered — sliders step, toggles flip.
                if self.settings_open {
                    self.scroll_accum += lines;
                    match self.settings_hover {
                        Some(0) => {
                            while self.scroll_accum >= 1.0 {
                                self.scroll_accum -= 1.0;
                                self.settings.step_render_distance(1);
                                self.settings_dirty = true;
                            }
                            while self.scroll_accum <= -1.0 {
                                self.scroll_accum += 1.0;
                                self.settings.step_render_distance(-1);
                                self.settings_dirty = true;
                            }
                        }
                        Some(1) => {
                            while self.scroll_accum >= 1.0 {
                                self.scroll_accum -= 1.0;
                                self.settings.step_cloud_distance(1);
                                self.settings_dirty = true;
                            }
                            while self.scroll_accum <= -1.0 {
                                self.scroll_accum += 1.0;
                                self.settings.step_cloud_distance(-1);
                                self.settings_dirty = true;
                            }
                        }
                        Some(2) | Some(3) => {
                            let shadows_row = self.settings_hover == Some(2);
                            while self.scroll_accum != 0.0 {
                                let notch = if self.scroll_accum > 0.0 {
                                    self.scroll_accum -= 1.0;
                                    1.0
                                } else {
                                    self.scroll_accum += 1.0;
                                    -1.0
                                };
                                let _ = notch;
                                if shadows_row {
                                    self.settings.toggle_shadows();
                                } else {
                                    self.settings.toggle_clouds();
                                }
                                self.settings_dirty = true;
                            }
                        }
                        _ => {}
                    }
                    return;
                }
                self.scroll_accum += lines;
                while self.scroll_accum >= 1.0 {
                    self.scroll_accum -= 1.0;
                    let n = BlockType::SOLID_ALL.len() as i32;
                    let next = (self.selected as i32 + 1).rem_euclid(n) as usize;
                    self.select_slot(next);
                }
                while self.scroll_accum <= -1.0 {
                    self.scroll_accum += 1.0;
                    let n = BlockType::SOLID_ALL.len() as i32;
                    let prev = (self.selected as i32 - 1).rem_euclid(n) as usize;
                    self.select_slot(prev);
                }
            }
            WindowEvent::Focused(focused) => {
                if focused {
                    // The startup grab attempt always fails (no pointer focus
                    // yet) — retry every time the window gains focus.
                    if !self.input.cursor_grabbed {
                        self.grab_cursor();
                    }
                } else {
                    // Release keys so we don't keep "walking" when alt-tabbed.
                    self.input.keys_held.clear();
                    self.release_cursor();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                // Settings page open: clicks toggle rows, grab sliders, and
                // never mine/place.
                if self.settings_open {
                    if state == winit::event::ElementState::Pressed
                        && button == winit::event::MouseButton::Left
                    {
                        match self.settings_hover {
                            // Rows 0/1 are sliders: press anywhere on the
                            // track starts a drag (and jumps the value there);
                            // 2/3 are the shadows/clouds toggles.
                            Some(row @ (0 | 1)) => {
                                self.drag_slider = Some(row);
                                // Jump straight to the click position (the
                                // cached cursor — MouseInput carries none).
                                self.apply_slider_drag(row, self.settings_cursor.0);
                            }
                            Some(2) => {
                                self.settings.toggle_shadows();
                                self.settings_dirty = true;
                            }
                            Some(3) => {
                                self.settings.toggle_clouds();
                                self.settings_dirty = true;
                            }
                            _ => {}
                        }
                    }
                    if state == winit::event::ElementState::Released {
                        // End the drag; persist whatever the drag landed on.
                        if self.drag_slider.take().is_some() {
                            self.settings_dirty = true;
                        }
                    }
                    return;
                }
                if self.inv_open {
                    // Inventory uses press-and-release dragging. This keeps a
                    // normal click convenient while allowing a stack to be
                    // picked up, moved across the panel, and dropped on the
                    // release target.
                    if button == winit::event::MouseButton::Left {
                        if state == winit::event::ElementState::Pressed {
                            self.inv_drag_origin = self
                                .inv_hover
                                .map(Some)
                                .or_else(|| {
                                    self.craft_hover
                                        .map(|i| Some(crate::overlay::PANEL_SLOTS + i))
                                })
                                .flatten();
                        } else if state == winit::event::ElementState::Released {
                            let origin = self.inv_drag_origin.take();
                            let target = self
                                .inv_hover
                                .map(Some)
                                .or_else(|| {
                                    self.craft_hover
                                        .map(|i| Some(crate::overlay::PANEL_SLOTS + i))
                                })
                                .flatten();
                            if let Some(origin) = origin {
                                // Pick the origin on press, then apply the
                                // normal merge/swap operation at release.
                                if origin < crate::overlay::PANEL_SLOTS {
                                    self.inv_slot_click(origin);
                                } else {
                                    self.craft_slot_click(origin - crate::overlay::PANEL_SLOTS);
                                }
                                if target != Some(origin) {
                                    if let Some(target) = target {
                                        if target < crate::overlay::PANEL_SLOTS {
                                            self.inv_slot_click(target);
                                        } else {
                                            self.craft_slot_click(
                                                target - crate::overlay::PANEL_SLOTS,
                                            );
                                        }
                                    }
                                }
                            } else if self.craft_output_hover {
                                self.craft_grid_output();
                            } else if let Some(item) = self.inv_held.take() {
                                let n = self.inv_held_count;
                                self.inv_held_count = 0;
                                self.stash_back(item, n);
                            }
                        }
                    }
                    // Right-click while holding a stack: drop ONE item onto
                    // the hovered slot (or from a slot into the cursor).
                    // Replaces the removed quick-craft shortcuts: with a
                    // plank stack on the cursor you can right-click four
                    // 2×2 cells to load a recipe by hand.
                    if button == winit::event::MouseButton::Right
                        && state == winit::event::ElementState::Pressed
                    {
                        if let Some(idx) = self.inv_hover {
                            self.inv_slot_right_click(idx);
                        } else if let Some(gi) = self.craft_hover {
                            self.craft_slot_right_click(gi);
                        }
                    }
                    return;
                }
                if state == winit::event::ElementState::Pressed
                    && button == winit::event::MouseButton::Left
                    && !self.input.cursor_grabbed
                {
                    self.grab_cursor(); // click to (re)capture the mouse
                    return;
                }
                if self.input.cursor_grabbed {
                    match (state, button) {
                        // Left HOLD charges mining; release stops (progress
                        // resets in update_mining next frame).
                        (winit::event::ElementState::Pressed, winit::event::MouseButton::Left) => {
                            self.mouse_held = true;
                        }
                        (winit::event::ElementState::Released, winit::event::MouseButton::Left) => {
                            self.mouse_held = false;
                        }
                        (winit::event::ElementState::Pressed, winit::event::MouseButton::Right) => {
                            if let Some(hit) =
                                raycast(&self.world, self.camera.pos, self.camera.forward(), REACH)
                            {
                                if matches!(
                                    self.world.block(hit.block[0], hit.block[1], hit.block[2]),
                                    crate::world::Block::Solid(BlockType::CraftingTable)
                                ) {
                                    self.open_inventory(3);
                                    return;
                                }
                            }
                            self.place_targeted();
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    self.resize_pending = true;
                }
            }
            WindowEvent::RedrawRequested => {
                // Stream terrain BEFORE borrowing the renderer for the draw.
                if let Err(e) = unsafe { self.update_terrain() } {
                    log::error!("terrain update failed: {e}");
                    event_loop.exit();
                    return;
                }
                // dt since last redraw, clamped so a long stall doesn't teleport us.
                let now = std::time::Instant::now();
                let dt = self
                    .last_instant
                    .map(|t| now.duration_since(t).as_secs_f32().min(0.1))
                    .unwrap_or(1.0 / 60.0);
                self.last_instant = Some(now);
                // Decay the hotbar label timer (label hides when it hits 0).
                if self.slot_label_ttl > 0.0 {
                    self.slot_label_ttl = (self.slot_label_ttl - dt).max(0.0);
                }
                self.day_night.advance(dt);
                self.frame_ms_ema = self.frame_ms_ema * 0.95 + dt * 1000.0 * 0.05;

                // Flush any settings change to disk (at most once per frame,
                // so a slider drag coalesces; the write is a tiny tmp+rename).
                if self.settings_dirty {
                    self.settings_dirty = false;
                    self.settings.save();
                }
                // Periodic world autosave: catches edits + position even if
                // the game is killed instead of closed. The flush itself is
                // cheap (a text header + a small binary overlay) and only
                // runs when something changed since the last save.
                self.autosave_timer += dt;
                if self.autosave_timer >= 10.0 {
                    self.autosave_timer = 0.0;
                    // Flush when edits changed, or once early in the session
                    // so the seed (+ player spawn) is on disk even if the
                    // game is killed before the first edit.
                    if self.world_dirty || !self.ever_saved {
                        self.flush_world_save();
                    }
                }

                self.input.update_look(&mut self.camera);
                self.player
                    .update(&mut self.camera, &self.input.keys_held, dt, &self.world);

                // Hold-to-mine + drop/particle physics + pickup.
                self.update_mining(dt);
                self.update_drops(dt);
                self.update_hand_anim(dt);

                // FPS average + ~1 Hz title HUD refresh (computed outside the
                // renderer borrow).
                self.fps_accum += dt;
                self.fps_frames += 1;
                let hud_title = if self.fps_accum >= 1.0 {
                    self.fps_display = (self.fps_frames as f32 / self.fps_accum).round() as u32;
                    self.fps_accum = 0.0;
                    self.fps_frames = 0;
                    Some(self.hud_title())
                } else {
                    None
                };

                if let (Some(window), Some(renderer)) = (&self.window, &mut self.renderer) {
                    if let Some(title) = &hud_title {
                        window.set_title(title);
                    }
                    // Push hotbar state for the HUD render this frame.
                    let label = if self.slot_label_ttl > 0.0 {
                        match self.inventory[self.selected] {
                            (Some(t), _) => t.name(),
                            (None, _) => "",
                        }
                    } else {
                        ""
                    };
                    renderer.set_hud_state(self.selected, &self.inventory, label);
                    renderer.set_inventory_state(
                        &self.backpack,
                        self.inv_open,
                        self.inv_hover,
                        self.inv_cursor,
                        self.inv_held,
                        self.inv_held_count,
                        self.craft_grid,
                        self.craft_size,
                        self.craft_hover,
                        self.craft_output_hover,
                    );
                    renderer.set_settings_state(
                        self.settings,
                        self.settings_open,
                        self.settings_hover,
                    );
                    let size = window.inner_size();
                    let aspect = (size.width.max(1) as f32) / (size.height.max(1) as f32);
                    self.inv_rects = crate::overlay::panel_rects(aspect);
                    self.settings_rects = crate::overlay::settings_rects();

                    // Mining cracks: current progress → destroy stage 0..9.
                    let crack_stage = if self.mouse_held {
                        (self.mining_progress * 10.0) as u32
                    } else {
                        0
                    };
                    let crack_verts = match self.mining_target {
                        Some([x, y, z]) if crack_stage > 0 => {
                            crate::overlay::build_crack_overlay(x, y, z, crack_stage)
                        }
                        _ => Vec::new(),
                    };

                    // Item drops: spinning/bobbing mini-cubes + particles.
                    let mut billboards: Vec<crate::geometry::Vertex> = Vec::new();
                    let now_secs = self
                        .last_instant
                        .map(|t| t.elapsed().as_secs_f32())
                        .unwrap_or(0.0);
                    for d in &self.drops {
                        let bob = (now_secs * 2.2 + d.age).sin() * 0.05;
                        let center = d.pos + glam::vec3(0.0, bob, 0.0);
                        billboards.extend(crate::overlay::build_drop_cube(
                            center,
                            0.14,
                            now_secs * 1.5 + d.age * 3.0,
                            d.item,
                        ));
                    }
                    let bb: Vec<crate::app::ParticleBillboard> = self
                        .particles
                        .iter()
                        .map(|p| {
                            // Camera-facing quad: right = camera right, up = Y.
                            ParticleBillboard {
                                center: p.pos,
                                right: self.camera.right_of(),
                                up: Vec3::Y,
                                half: p.half,
                                color: p.color,
                            }
                        })
                        .collect();
                    billboards.extend(crate::overlay::build_particles(&bb));
                    renderer.set_billboards(crack_verts, billboards);

                    // First-person viewmodel: the hand + whatever is held.
                    let held_now = self.inventory[self.selected].0;
                    if held_now != self.hand_last_held {
                        self.hand_last_held = held_now;
                        self.hand_equip = 1.0; // dip in from below
                    }
                    let hand_verts = crate::hand::build_hand(
                        &self.camera,
                        held_now,
                        crate::hand::HandAnim {
                            swing: self.hand_swing,
                            equip: self.hand_equip,
                            walk: self.hand_walk,
                            walking: self.hand_walking,
                        },
                    );
                    renderer.set_hand(hand_verts);

                    let sky_state = self.day_night.state();

                    // Targeted-block highlight (shared by the overlay and F3).
                    let highlight =
                        raycast(&self.world, self.camera.pos, self.camera.forward(), REACH);

                    // F3 debug overlay: compose info lines + collision boxes.
                    if self.debug_visible {
                        let gpu = renderer.gpu_info();
                        let sys = crate::sys::read(); // fresh RAM numbers
                        let facing = self.camera.yaw.to_degrees().rem_euclid(360.0);
                        let compass = match ((facing + 45.0) / 90.0).floor() as i32 % 4 {
                            0 => "N",
                            1 => "E",
                            2 => "S",
                            _ => "W",
                        };
                        let pos = self.camera.pos;
                        let lines = vec![
                            format!(
                                "voxelcraft milestone 6 — {:.0} fps ({:.1} ms)",
                                self.fps_display, self.frame_ms_ema
                            ),
                            format!("xyz: {:.2} / {:.2} / {:.2}", pos.x, pos.y, pos.z),
                            format!(
                                "chunk: {} {}  facing: {:.0} deg {}",
                                (pos.x as i32).div_euclid(16),
                                (pos.z as i32).div_euclid(16),
                                facing,
                                compass
                            ),
                            format!(
                                "mode: {}  time: {:.2}  grounded: {}",
                                if self.player.is_flying() {
                                    "fly"
                                } else {
                                    "walk"
                                },
                                self.day_night.time,
                                self.player.grounded
                            ),
                            format!(
                                "tris: {}  meshes: {}  edits: {}",
                                renderer.total_triangles(),
                                renderer.mesh_count(),
                                self.world.edit_count()
                            ),
                            format!("gpu: {} ({})", gpu.name, gpu.vulkan_api),
                            format!(
                                "driver: {:x}.{:x}.{:x}  did: {:04x}  vid: {:04x}",
                                gpu.driver_version >> 22,
                                (gpu.driver_version >> 12) & 0xFFF,
                                gpu.driver_version & 0xFFF,
                                gpu.device_id & 0xFFFF,
                                gpu.vendor_id & 0xFFFF
                            ),
                            format!(
                                "swapchain: {} imgs, {}  msaa {}",
                                renderer.swapchain_image_count(),
                                gpu.swapchain_format,
                                gpu.msaa
                            ),
                            format!("cpu: {}", self.sys.cpu_model),
                            format!("ram: {} / {} mib", sys.ram_used_mib, sys.ram_total_mib),
                            format!("seed: {}  backend: {}", self.world.seed, self.backend),
                        ];
                        let box_verts = crate::debug_overlay::build_collision_box(
                            pos,
                            EYE_HEIGHT,
                            0.3, // half width of the 0.6-wide body
                            highlight.as_ref().map(|h| &h.block),
                        );
                        renderer.set_debug_data(Some(lines), 1, box_verts);
                    } else {
                        renderer.set_debug_data(None, 0, Vec::new());
                    }

                    // Recreate the swapchain first if the window was resized.
                    if self.resize_pending {
                        self.resize_pending = false;
                        let size = window.inner_size();
                        log::info!(
                            "window resized to {}x{} — recreating swapchain",
                            size.width,
                            size.height
                        );
                        if let Err(e) = unsafe { renderer.recreate_swapchain(window) } {
                            log::error!("swapchain recreation failed: {e}");
                            event_loop.exit();
                            return;
                        }
                    }

                    let presented_before = renderer.frames_presented();
                    // Sun + moon billboards: quads hanging 140 blocks out along
                    // the true (unclamped) light directions, sized to look
                    // right at that distance.
                    let disc_verts =
                        crate::overlay::build_celestial_discs(self.camera.pos, &sky_state);
                    let _ = &disc_verts;
                    let shadow_light = self.day_night.shadow_light_dir();
                    // Wall-clock seconds drive the cloud drift (captured at
                    // spawn and sampled by the renderer each frame).
                    let now_secs = self
                        .spawn_instant
                        .map(|t| t.elapsed().as_secs_f32())
                        .unwrap_or(0.0);
                    if let Err(e) = unsafe {
                        renderer.draw(
                            window,
                            &self.camera,
                            highlight.as_ref(),
                            &sky_state,
                            shadow_light,
                            &disc_verts,
                            now_secs,
                        )
                    } {
                        log::error!("draw error: {e}");
                        event_loop.exit();
                    }
                    self.frames_presented = renderer.frames_presented();
                    self.ever_drawn = true;
                    if self.frames_presented == presented_before {
                        self.stall_ticks += 1;
                        if self.stall_ticks == 120 {
                            log::error!(
                                "no frames presented for ~2s — compositor likely not showing buffers (backend: {})",
                                self.backend
                            );
                        }
                    } else {
                        self.stall_ticks = 0;
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            self.input.add_raw_motion(delta.0, delta.1);
        }
    }
}

#[cfg(test)]
mod inv_tests {
    use super::*;
    use crate::items::ItemType;
    use crate::world::BlockType;

    fn app() -> App {
        // A minimal App for inventory-logic tests: only the slot arrays and
        // the helpers they drive matter here.
        App::new("test".into())
    }

    #[test]
    fn fills_hotbar_first_then_backpack() {
        let mut a = app();
        // 6 distinct types claim all hotbar slots...
        for t in BlockType::SOLID_ALL {
            assert_eq!(
                a.inv_add(ItemType::Block(t)),
                Some((
                    "h",
                    BlockType::SOLID_ALL.iter().position(|&x| x == t).unwrap()
                ))
            );
        }
        // ...the 7th new type overflows into backpack slot 0.
        assert_eq!(a.inv_add(ItemType::Stick), Some(("b", 0)));
        assert_eq!(a.backpack[0], (Some(ItemType::Stick), 1));
    }

    #[test]
    fn overflows_to_backpack_then_reuses_freed_hotbar_slot() {
        let mut a = app();
        // 6 distinct block types fill the hotbar; a 7th type overflows to
        // the backpack.
        for t in BlockType::SOLID_ALL {
            let _ = a.inv_add(ItemType::Block(t));
        }
        assert_eq!(a.inv_add(ItemType::WoodPickaxe), Some(("b", 0)));
        // Free hotbar slot 3; the next NEW type must claim it (hotbar
        // before backpack) even though the backpack already has items.
        a.inventory[3] = (None, 0);
        assert_eq!(a.inv_add(ItemType::WoodAxe), Some(("h", 3)));
    }

    #[test]
    fn quick_craft_uses_the_same_shaped_recipe_as_the_panel() {
        let mut a = app();
        a.inventory[0] = (Some(ItemType::Block(BlockType::Log)), 1);
        a.craft('c');
        assert_eq!(
            a.inventory[0],
            (Some(ItemType::Block(BlockType::Planks)), 4)
        );

        // Tools are table-only even when the shortcut is pressed directly.
        a.inventory[1] = (Some(ItemType::Block(BlockType::Planks)), 3);
        a.inventory[2] = (Some(ItemType::Stick), 2);
        a.craft('t');
        assert_eq!(
            a.inventory[1],
            (Some(ItemType::Block(BlockType::Planks)), 3)
        );
        assert_eq!(a.inventory[2], (Some(ItemType::Stick), 2));

        a.open_inventory(3);
        a.craft('t');
        assert_eq!(
            a.inventory[0],
            (Some(ItemType::Block(BlockType::Planks)), 1)
        );
        assert_eq!(
            a.inventory[1],
            (Some(ItemType::Block(BlockType::Planks)), 3)
        );
        assert_eq!(a.inventory[2], (Some(ItemType::WoodPickaxe), 1));
    }

    #[test]
    fn closing_inventory_returns_grid_items() {
        let mut a = app();
        a.open_inventory(2);
        a.craft_grid[0] = (Some(ItemType::Block(BlockType::Log)), 3);
        a.craft_grid[1] = (Some(ItemType::Stick), 2);
        a.close_inventory();
        assert!(a
            .craft_grid
            .iter()
            .all(|slot| slot.0.is_none() && slot.1 == 0));
        assert_eq!(a.inventory[0], (Some(ItemType::Block(BlockType::Log)), 3));
        assert_eq!(a.inventory[1], (Some(ItemType::Stick), 2));
        assert!(!a.inv_open);
        assert_eq!(a.craft_size, 2);
    }

    #[test]
    fn two_by_two_grid_bottom_right_cell_is_clickable_and_crafts_table() {
        let mut a = app();
        a.open_inventory(2);
        // Simulate the exact click flow: right-click one plank into each of
        // the four 2×2 cells, INCLUDING the bottom-right (index 4 — the one
        // that used to be unreachable).
        a.backpack[0] = (Some(ItemType::Block(BlockType::Planks)), 4);
        a.inv_slot_click(0); // pick up 4 planks
        for cell in [0usize, 1, 3, 4] {
            a.craft_slot_right_click(cell); // drop one per cell
        }
        assert_eq!(
            a.craft_grid[0],
            (Some(ItemType::Block(BlockType::Planks)), 1)
        );
        assert_eq!(
            a.craft_grid[1],
            (Some(ItemType::Block(BlockType::Planks)), 1)
        );
        assert_eq!(
            a.craft_grid[3],
            (Some(ItemType::Block(BlockType::Planks)), 1)
        );
        assert_eq!(
            a.craft_grid[4],
            (Some(ItemType::Block(BlockType::Planks)), 1)
        );
        // The matcher must see the table pattern in the 2×2 grid...
        let recipe = crate::items::matching_recipe(&a.craft_grid, 2, false)
            .expect("2x2 planks square must match crafting table");
        assert_eq!(recipe.name, "crafting table");
        // ...and clicking the output must produce it.
        a.craft_grid_output();
        assert_eq!(
            a.inventory
                .iter()
                .chain(a.backpack.iter())
                .find(|s| s.0 == Some(ItemType::Block(crate::world::BlockType::CraftingTable)))
                .map(|s| s.1),
            Some(1)
        );
    }

    #[test]
    fn click_pickup_place_swap() {
        let mut a = app();
        a.backpack[2] = (Some(ItemType::Block(BlockType::Log)), 5);
        a.inv_slot_click(2); // pick up
        assert!(a.backpack[2].0.is_none());
        assert_eq!(a.inv_held, Some(ItemType::Block(BlockType::Log)));
        assert_eq!(a.inv_held_count, 5);
        a.inv_slot_click(crate::overlay::BACKPACK_SLOTS + 4); // place in hotbar slot 4
        assert_eq!(a.inventory[4], (Some(ItemType::Block(BlockType::Log)), 5));
        assert!(a.inv_held.is_none());
        // Swap: put stone over the log.
        a.inventory[0] = (Some(ItemType::Block(BlockType::Stone)), 2);
        a.inv_slot_click(crate::overlay::BACKPACK_SLOTS + 0); // pick stone up
        a.inv_slot_click(crate::overlay::BACKPACK_SLOTS + 4); // swap onto log
        assert_eq!(a.inventory[4], (Some(ItemType::Block(BlockType::Stone)), 2));
        assert_eq!(a.inv_held, Some(ItemType::Block(BlockType::Log)));
        assert_eq!(a.inv_held_count, 5);
    }
}

#[cfg(test)]
mod water_key_tests {
    use super::*;

    /// Water keys are now just the chunk's own key (water meshes live in
    /// SEPARATE renderer storage), so an upload can never clobber a
    /// different chunk's terrain mesh. Locks the invariant that killed the
    /// old sentinel scheme: water_key((cx, cz)) used to equal the terrain
    /// key of chunk (cx, -cz-5) — uploading water destroyed its mesh, and
    /// chunks "sometimes disappeared" south of spawn.
    #[test]
    fn water_key_is_the_chunks_own_key() {
        for cx in -40..=40 {
            for cz in -40..=40 {
                assert_eq!(water_key((cx, cz)), (cx, cz));
            }
        }
    }
}
