//! Procedural audio: tiny synthesized sound effects through ALSA directly
//! (no asset files, no audio framework — the whole mixer is ~150 lines).
//!
//! Every effect is generated as a short f32 sample buffer at the device rate
//! and handed to a dedicated playback thread through a channel; the game
//! thread never blocks on audio (play() is a lock-free enqueue). Sounds are
//! deterministic recipes — noise bursts with envelopes + a tonal thump —
//! tuned per effect.

use std::sync::mpsc::{self, Receiver, Sender};

pub const SAMPLE_RATE: u32 = 48000;

/// One queued clip: interleaved mono f32 samples + volume.
struct Clip {
    samples: std::sync::Arc<Vec<f32>>,
    volume: f32,
}

enum Msg {
    Play(Clip),
    Shutdown,
}

/// Handle to the audio thread. Cheap to clone-free — the App owns one.
pub struct Sound {
    tx: Option<Sender<Msg>>,
    /// Pedestrian footstep alternation (left/right emphasis).
    step_flip: bool,
}

impl Sound {
    /// Spawn the audio thread. On failure (no ALSA device, permissions) the
    /// game runs silent — sound is never allowed to break the game.
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Msg>();
        let opened = open_playback_gate(rx);
        let tx = if opened { Some(tx) } else { None };
        Self {
            tx,
            step_flip: false,
        }
    }

    /// Enqueue a clip (never blocks; drops the sound if the channel is
    /// full — a stuttering game is worse than a missed footstep). The
    /// samples are Arc-shared so the enqueue is a pointer bump.
    fn play(&mut self, samples: Vec<f32>, volume: f32) {
        let Some(tx) = &self.tx else { return };
        let clip = Clip {
            samples: std::sync::Arc::new(samples),
            volume,
        };
        if tx.send(Msg::Play(clip)).is_err() {
            // Audio thread gone — run silent.
            self.tx = None;
        }
    }

    /// Block broken: crunchy low noise burst + a thump. `material` picks
    /// the flavor (stone is brighter, dirt duller).
    pub fn block_break(&mut self, material: Material) {
        let (cut, tone, dur) = match material {
            Material::Stone => (3200.0, 180.0, 0.16),
            Material::Wood => (1800.0, 140.0, 0.18),
            Material::Soft => (900.0, 90.0, 0.20),
            Material::Leaf => (5000.0, 320.0, 0.10),
        };
        self.play(gen_break(cut, tone, dur), 0.55);
    }

    /// Block placed: a short knock.
    pub fn block_place(&mut self, material: Material) {
        let (cut, tone, dur) = match material {
            Material::Stone => (2400.0, 220.0, 0.09),
            Material::Wood => (1500.0, 170.0, 0.10),
            Material::Soft => (800.0, 110.0, 0.12),
            Material::Leaf => (3600.0, 300.0, 0.07),
        };
        self.play(gen_break(cut, tone, dur), 0.4);
    }

    /// Footstep: alternating soft shuffle, pitch varies slightly so a run
    /// doesn't sound like a machine gun.
    pub fn footstep(&mut self, material: Material) {
        self.step_flip = !self.step_flip;
        let cut = match material {
            Material::Stone => 1600.0,
            Material::Wood => 1200.0,
            Material::Soft => 650.0,
            Material::Leaf => 2600.0,
        };
        let seed = if self.step_flip { 0xBEEF } else { 0xF00D };
        self.play(gen_step(cut, seed), 0.25);
    }

    /// Pickup pop: quick rising blip (Minecraft's "plop").
    pub fn pickup(&mut self) {
        self.play(gen_pop(), 0.4);
    }
}

impl Default for Sound {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Sound {
    fn drop(&mut self) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Msg::Shutdown);
        }
    }
}

/// Try to spawn the audio thread; returns whether it started (the device
/// open happens on the thread itself — the game thread never waits on ALSA).
fn open_playback_gate(rx: Receiver<Msg>) -> bool {
    std::thread::Builder::new()
        .name("audio".into())
        .spawn(move || audio_thread(rx))
        .is_ok()
}
#[derive(Clone, Copy)]
pub enum Material {
    Stone,
    Wood,
    Soft,
    Leaf,
}

/// The sound material for a block type.
pub fn block_material(t: crate::world::BlockType) -> Material {
    use crate::world::BlockType::*;
    match t {
        Stone | CoalOre => Material::Stone,
        Log | Planks | CraftingTable => Material::Wood,
        Leaves => Material::Leaf,
        _ => Material::Soft,
    }
}

// ---------------------------------------------------------------------------
// Synthesis recipes
// ---------------------------------------------------------------------------

/// Tiny xorshift for deterministic-but-varying noise.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Filtered noise burst with exponential decay + a low sine thump.
fn gen_break(cut_hz: f32, tone_hz: f32, dur_s: f32) -> Vec<f32> {
    let n = (dur_s * SAMPLE_RATE as f32) as usize;
    let mut out = Vec::with_capacity(n);
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);
    // One-pole low-pass toward the cutoff (simple, cheap, good enough).
    let alpha = (cut_hz / SAMPLE_RATE as f32).clamp(0.01, 0.95);
    let mut lp = 0.0f32;
    for i in 0..n {
        let t = i as f32 / SAMPLE_RATE as f32;
        let env = (-t * 18.0).exp(); // fast attack-ish decay
        lp += alpha * (rng.next_f32() - lp);
        let thump = (tone_hz * t * std::f32::consts::TAU).sin() * (-t * 24.0).exp() * 0.8;
        out.push((lp * 0.9 + thump) * env);
    }
    out
}

/// Footstep: very short soft noise tick, slight lowpass, quick decay.
fn gen_step(cut_hz: f32, seed: u64) -> Vec<f32> {
    let n = (0.09 * SAMPLE_RATE as f32) as usize;
    let mut out = Vec::with_capacity(n);
    let mut rng = Rng(seed);
    let alpha = (cut_hz / SAMPLE_RATE as f32).clamp(0.01, 0.95);
    let mut lp = 0.0f32;
    for i in 0..n {
        let t = i as f32 / SAMPLE_RATE as f32;
        let env = (-t * 40.0).exp() * (t * 400.0).min(1.0); // soft attack
        lp += alpha * (rng.next_f32() - lp);
        out.push(lp * env);
    }
    out
}

/// Pickup: rising sine blip (400→900 Hz over 90 ms) with fast decay.
fn gen_pop() -> Vec<f32> {
    let n = (0.09 * SAMPLE_RATE as f32) as usize;
    let mut out = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / SAMPLE_RATE as f32;
        let f = 400.0 + (t / 0.09) * 500.0;
        phase += f * std::f32::consts::TAU / SAMPLE_RATE as f32;
        let env = (-t * 30.0).exp();
        out.push(phase.sin() * env);
    }
    out
}

// ---------------------------------------------------------------------------
// ALSA playback thread (direct FFI — no audio crate dependency; the game
// already requires a Linux box with libasound, which every SteamOS/desktop
// distro ships)
// ---------------------------------------------------------------------------

#[link(name = "asound")]
extern "C" {
    fn snd_pcm_open(
        pcm: *mut *mut SndPcm,
        name: *const std::os::raw::c_char,
        stream: std::os::raw::c_int,
        mode: std::os::raw::c_int,
    ) -> std::os::raw::c_int;
    fn snd_pcm_set_params(
        pcm: *mut SndPcm,
        format: std::os::raw::c_uint,
        access: std::os::raw::c_uint,
        channels: std::os::raw::c_uint,
        rate: std::os::raw::c_uint,
        soft_resample: std::os::raw::c_int,
        latency_us: std::os::raw::c_uint,
    ) -> std::os::raw::c_int;
    fn snd_pcm_writei(pcm: *mut SndPcm, buffer: *const std::os::raw::c_void, frames: u64) -> i64;
    fn snd_pcm_prepare(pcm: *mut SndPcm) -> std::os::raw::c_int;
    fn snd_pcm_close(pcm: *mut SndPcm) -> std::os::raw::c_int;
}

#[allow(non_camel_case_types)]
type SndPcm = std::os::raw::c_void;

const SND_PCM_STREAM_PLAYBACK: std::os::raw::c_int = 0;
const SND_PCM_ACCESS_RW_INTERLEAVED: std::os::raw::c_uint = 3;
const SND_PCM_FORMAT_FLOAT_LE: std::os::raw::c_uint = 14;

/// Open the default ALSA playback device: mono f32 interleaved at the fixed
/// rate, ~100 ms latency.
unsafe fn open_playback() -> Option<*mut SndPcm> {
    let mut pcm: *mut SndPcm = std::ptr::null_mut();
    let name = c"default".as_ptr();
    if snd_pcm_open(&mut pcm, name, SND_PCM_STREAM_PLAYBACK, 0) != 0 {
        return None;
    }
    let ok = snd_pcm_set_params(
        pcm,
        SND_PCM_FORMAT_FLOAT_LE,
        SND_PCM_ACCESS_RW_INTERLEAVED,
        1,
        SAMPLE_RATE,
        1, // allow the device to resample if the card wants something else
        100_000,
    ) == 0;
    if !ok {
        snd_pcm_close(pcm);
        return None;
    }
    unsafe { snd_pcm_prepare(pcm) };
    Some(pcm)
}

fn audio_thread(rx: Receiver<Msg>) {
    // The live mix: queued clips, each with its playhead.
    let mut active: Vec<(Clip, usize)> = Vec::new();
    let mut buf = vec![0.0f32; 512];

    let Some(pcm) = (unsafe { open_playback() }) else {
        // No usable audio device: run silent for the life of the process.
        return;
    };

    loop {
        // Drain any new clips (bounded so a spamming producer can't OOM us).
        while let Ok(msg) = rx.try_recv() {
            match msg {
                Msg::Play(clip) => {
                    if active.len() < 16 {
                        active.push((clip, 0));
                    }
                }
                Msg::Shutdown => {
                    unsafe { snd_pcm_close(pcm) };
                    return;
                }
            }
        }

        // Mix up to one buffer of audio.
        for s in buf.iter_mut() {
            *s = 0.0;
        }
        let mut mixed_any = false;
        for (clip, pos) in active.iter_mut() {
            let take = (clip.samples.len() - *pos).min(buf.len());
            for i in 0..take {
                buf[i] += clip.samples[*pos + i] * clip.volume;
            }
            *pos += take;
            if *pos < clip.samples.len() {
                mixed_any = true;
            }
        }
        active.retain(|(c, p)| *p < c.samples.len());

        // Idle with no clips: wait for work instead of spinning.
        if !mixed_any {
            match rx.recv_timeout(std::time::Duration::from_millis(200)) {
                Ok(Msg::Play(clip)) => {
                    if active.len() < 16 {
                        active.push((clip, 0));
                    }
                }
                Ok(Msg::Shutdown) => {
                    unsafe { snd_pcm_close(pcm) };
                    return;
                }
                Err(_) => {} // timeout — loop, re-check
            }
            continue;
        }

        // Clamp + write; on underrun (-EPIPE) re-prepare and retry once.
        for s in buf.iter_mut() {
            *s = s.clamp(-1.0, 1.0);
        }
        let written = unsafe {
            snd_pcm_writei(
                pcm,
                buf.as_ptr() as *const std::os::raw::c_void,
                buf.len() as u64,
            )
        };
        if written < 0 {
            unsafe { snd_pcm_prepare(pcm) };
        }
    }
}
