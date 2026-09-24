//! Background chunk-meshing worker.
//!
//! The frame thread sends chunk requests; one worker thread meshes them
//! (terrain math + the chunk mesh-cache file IO) and sends finished geometry
//! back. The frame thread never blocks on meshing — it only drains the
//! result channel each frame and uploads at most its GPU budget. This keeps
//! streaming hitches off the render loop entirely (the previous batch path
//! still meshed on the frame thread, just faster).
//!
//! Requests carry an owned `World` snapshot so the worker needs no locks.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::JoinHandle;

use crate::geometry::Vertex;
use crate::terrain::{mesh_chunk, ChunkCoord};
use crate::world::World;

/// One finished chunk: geometry + the key to upload it under. `w*` is the
/// translucent water surface mesh (uploaded to its own key, drawn with
/// alpha blending after all opaque geometry).
pub struct MeshedChunk {
    pub key: ChunkCoord,
    pub verts: Vec<Vertex>,
    pub indices: Vec<u16>,
    pub wverts: Vec<Vertex>,
    pub windices: Vec<u16>,
}

enum Msg {
    Mesh {
        key: ChunkCoord,
        world: World,
        chunk_x: i32,
        chunk_z: i32,
        save_dir: std::path::PathBuf,
    },
    Shutdown,
}

pub struct MesherWorker {
    tx: Sender<Msg>,
    rx: Receiver<MeshedChunk>,
    handle: Option<JoinHandle<()>>,
    /// Queue depth shared with the frame thread (request ++, drain --).
    pending_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl MesherWorker {
    /// Spawn the worker thread.
    pub fn new() -> Self {
        let (req_tx, req_rx) = mpsc::channel::<Msg>();
        let (out_tx, out_rx) = mpsc::channel::<MeshedChunk>();
        let pending_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handle = std::thread::Builder::new()
            .name("chunk-mesher".into())
            .spawn(move || {
                // The request side never blocks the worker: it processes one
                // chunk at a time, in arrival order.
                while let Ok(msg) = req_rx.recv() {
                    match msg {
                        Msg::Mesh {
                            key,
                            world,
                            chunk_x,
                            chunk_z,
                            save_dir,
                        } => {
                            // Chunk cache first: a previously-seen chunk is a
                            // file read, not terrain math. (The cache holds
                            // the opaque terrain mesh only; water surfaces are
                            // a handful of quads, cheap to rebuild live.)
                            let vi = if let Some((verts, indices)) =
                                crate::save::load_chunk_mesh(&save_dir, world.seed, key.0, key.1)
                            {
                                // The on-disk cache contains opaque terrain
                                // only. Regenerate the cheap water surface on
                                // cache hits too, or lakes vanish after restart.
                                let (wverts, windices) =
                                    crate::terrain::mesh_water_chunk(&world, chunk_x, chunk_z);
                                (verts, indices, wverts, windices)
                            } else {
                                let vi = mesh_chunk(&world, chunk_x, chunk_z);
                                // Cache opaque geometry; water remains cheap
                                // enough to rebuild from the current world.
                                let _ = crate::save::save_chunk_mesh(
                                    &save_dir, world.seed, key.0, key.1, &vi.0, &vi.1,
                                );
                                vi
                            };
                            // If the receiver is gone we're shutting down.
                            if out_tx
                                .send(MeshedChunk {
                                    key,
                                    verts: vi.0,
                                    indices: vi.1,
                                    wverts: vi.2,
                                    windices: vi.3,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Msg::Shutdown => break,
                    }
                }
            })
            .expect("spawn chunk-mesher thread");
        Self {
            tx: req_tx,
            rx: out_rx,
            handle: Some(handle),
            pending_count,
        }
    }

    /// Queue a chunk for background meshing (`world` is a snapshot the
    /// worker owns; `save_dir` is where the chunk cache lives). The worker
    /// consults the cache first and writes newly-meshed chunks back, so the
    /// frame thread does no mesh-related IO at all. Returns false if the
    /// worker is gone.
    pub fn request(
        &self,
        key: ChunkCoord,
        world: World,
        chunk_x: i32,
        chunk_z: i32,
        save_dir: std::path::PathBuf,
    ) -> bool {
        let sent = self
            .tx
            .send(Msg::Mesh {
                key,
                world,
                chunk_x,
                chunk_z,
                save_dir,
            })
            .is_ok();
        if sent {
            self.pending_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        sent
    }

    /// Drain every finished chunk currently available (never blocks).
    pub fn drain(&self) -> Vec<MeshedChunk> {
        let mut out = Vec::new();
        loop {
            match self.rx.try_recv() {
                Ok(c) => {
                    self.pending_count
                        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    out.push(c);
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
        out
    }

    /// Number of requests still queued or in flight, tracked by the frame
    /// thread itself (std mpsc has no len; the App keeps `in_flight` and
    /// passes its count here is unnecessary — this counts via the shared
    /// atomic instead).
    pub fn pending(&self) -> usize {
        self.pending_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for MesherWorker {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::CHUNK_SIZE;

    #[test]
    fn cached_chunk_rebuilds_water_mesh() {
        let world = World::new(0xABCD);
        let wet_x = (0..400)
            .find(|&x| {
                matches!(
                    world.block(x, crate::world::SEA_LEVEL - 1, 0),
                    crate::world::Block::Solid(crate::world::BlockType::Water)
                )
            })
            .expect("a flooded column exists in 400");
        let chunk_x = wet_x.div_euclid(CHUNK_SIZE) * CHUNK_SIZE;
        let key = (chunk_x.div_euclid(CHUNK_SIZE), 0);
        let save_dir = std::env::temp_dir().join(format!(
            "voxelcraft-mesher-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        // Store an empty opaque cache deliberately: the worker must take its
        // cache-hit branch while still rebuilding water from the world.
        crate::save::save_chunk_mesh(&save_dir, world.seed, key.0, key.1, &[], &[]).unwrap();

        let worker = MesherWorker::new();
        assert!(worker.request(key, world.snapshot(), chunk_x, 0, save_dir.clone(),));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let result = loop {
            if let Some(result) = worker.drain().into_iter().next() {
                break result;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "mesher worker timed out"
            );
            std::thread::yield_now();
        };
        assert_eq!(result.key, key);
        assert!(
            !result.wverts.is_empty(),
            "cached chunk lost its water mesh"
        );
        assert_eq!(result.windices.len(), result.wverts.len() / 4 * 6);

        drop(worker);
        let _ = std::fs::remove_dir_all(save_dir);
    }
}
