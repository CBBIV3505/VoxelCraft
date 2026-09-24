# VoxelCraft

A FULLY AI GENERATED voxel engine written in Rust on raw Vulkan (via `ash`) — no
game engine, no rendering framework. Procedural terrain, day/night cycle,
mining/building, crafting, swimming, and synthesized audio.

## Running

```bash
cargo run
```

Requirements:
- Rust 1.97+ (`cargo --version`)
- A Vulkan loader + driver (`vulkaninfo` works)
- `glslc` on PATH (Vulkan SDK) — `build.rs` compiles the shaders
- ALSA (`libasound.so.2`) for sound — present on virtually every Linux
  desktop; without it the game runs silent instead of failing

Steam Deck / SteamOS note: the game forces the X11 backend internally to
avoid a KWin/Wayland swapchain bug, so run it from desktop mode as-is.

## Controls

| Input | Action |
| --- | --- |
| WASD + mouse | Move / look |
| Space | Jump / swim up (fly: up) |
| Shift | Fly down |
| G | Toggle walk / fly |
| Left click (hold) | Mine (crack animation, tool-tier gated) |
| Right click | Place block |
| 1–8 / scroll | Select hotbar slot |
| E | Inventory (2×2 crafting grid) |
| Right-click a placed crafting table | 3×3 crafting grid |
| O / Escape | Settings (render distance, clouds, shadows, sensitivity) |
| F3 | Debug overlay (collision box, GPU/CPU/RAM stats) |
| Escape ×2 / window close | Quit (autosaves) |

## Gameplay

- **World**: procedural fBm heightmap with grass/dirt/stone strata, coal ore
  veins, winding 3D-noise caves, lakes below sea level, and analytic trees —
  deterministic per seed, streamed in 16×16 chunks around the player.
- **Progression**: punch logs → planks → crafting table → wooden pickaxe →
  stone (and coal) → stone pickaxe. Stone-tier blocks need a pickaxe.
- **Water**: translucent lakes you can swim in (slower movement, Space to
  surface, drag on everything).
- **Saving**: seed, block edits, player position, time of day, and per-chunk
  mesh caches persist under `~/.local/share/voxelcraft/world/` (override
  with `VOXELCRAFT_SAVE_DIR`). Delete that folder for a fresh world.

## Versioning

Versions follow `MAJOR.MINOR.PATCH`:

- **MAJOR** — breaking save-format or world-generation changes (bump
  `FORMAT_VERSION` / `TERRAIN_VERSION` in `src/save.rs` alongside).
- **MINOR** — new gameplay features (water, tools tiers, mobs, …).
- **PATCH** — bug fixes and polish.

The current version lives in `Cargo.toml` (`[package] version`) and is
compiled into the binary — the startup log line and window title show it.
`CHANGELOG.md` documents each release. Before tagging a release:
`cargo test && cargo build --release`, smoke-run, update the changelog.

## Project layout

`src/` — one module per concern: `renderer` (Vulkan), `terrain`/`world`
(generation + edits), `mesher` (background chunk-meshing thread), `player`
(physics), `app` (event loop + gameplay glue), `overlay` (HUD), `sound`
(ALSA synth), `save`/`settings` (persistence). `shaders/` — GLSL compiled by
`build.rs` at build time.

## License

VoxelCraft is licensed under the [MIT License](LICENSE) — free to use,
modify, and share, including for commercial purposes. Third-party software
and system-library notices are documented in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).


## Contributing

Issues and pull requests welcome. Keep the release checklist in mind:
`cargo test`, `cargo build --release`, a smoke run, and a CHANGELOG entry
before tagging a version.
