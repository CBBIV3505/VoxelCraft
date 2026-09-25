# Changelog

All notable changes to VoxelCraft. Versions follow MAJOR.MINOR.PATCH
(see README.md for the policy).

## [0.1.0] — First playable

### Added
- **Lakes**: terrain below sea level (y=11) fills with translucent water —
  procedural, so it regenerates identically and costs no save space.
- **Swimming**: eye-deep water switches to swim physics — slower movement,
  Space to surface, drag on everything, lake floor walk-out, shore pop-out.
- **Sound**: fully procedural effects through ALSA (no asset files, no audio
  crate — direct FFI): per-material block break/place, footsteps, and the
  pickup pop. No audio device → the game runs silent, never fails.
- **Version system**: `Cargo.toml` version compiled into the binary; shown
  in the startup log and window title. `CHANGELOG.md` (this file) and the
  README versioning policy added.

### Fixed
- Hand viewmodel crash: the held-pickaxe sprite geometry (one extruded box
  per sprite pixel) overflowed its 704-vertex overlay slice and tripped the
  debug assert. The slice is now 8192 vertices with a runtime clamp + warning
  as a release-mode safety net.

### Basically the game
- Raw-Vulkan renderer (ash): shadow mapping with PCF, procedural 16×16
  block-texture atlas with keyed leaf transparency, sky dome + sun/moon
  discs, day/night cycle, distance fog matched to the sky palette.
- Procedural world: fBm heightmap with grass/dirt/stone strata, coal ore
  veins, 3D-noise caves, analytic trees; chunk streaming meshed on a
  background worker thread with a per-frame GPU upload budget and disk mesh
  cache.
- Gameplay: walking/gravity/jumping + creative flight, hold-to-mine with
  crack animation and tool tiers (wood → stone), block placing, item drops
  with magnet pickup, break particles, hotbar + backpack with drag-and-drop,
  2×2 inventory crafting and 3×3 crafting-table grid with a canonical recipe
  table, first-person hand with swing/equip/bob.
- UI: HUD (hotbar with 3D block icons, crosshair, item label), settings page
  (render distance, cloud distance, shadows, sensitivity; persisted),
  F3 debug overlay, per-chunk frustum culling.
- Persistence: world saves (seed, edits, position, time) + settings, atomic
  writes, seed⊕version-tagged chunk caches.
