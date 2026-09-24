# VoxelCraft — run doc

Native Rust + raw-Vulkan (`ash`) voxel engine.

Release 0.2.0: **water, swimming, sound + version system** — terrain below
sea level (y=11, `world::SEA_LEVEL`) floods with translucent water
(procedural, never stored; only open basins — caves under land stay air).
Water is its own mesh per chunk in SEPARATE renderer storage (`water_meshes`,
keyed by the chunk's own coords — an earlier sentinel scheme `(cx, -cz-4)`
collided with real chunk keys south of spawn and destroyed terrain meshes on
water upload: the "chunks sometimes disappear" bug) drawn through a
dedicated pipeline: SRC_ALPHA blend (alpha rides push pad.w), depth-write
off, no cull, far→near order, culled in the shadow pass. Eye below the
water surface switches walk physics to swim (slower, Space to rise, drag,
lake-floor walk-out, shore pop-out boost). Sound: `src/sound.rs` — a
~250-line ALSA synth (direct FFI, no audio crate) on its own thread:
per-material break/place, footsteps (per-stride, ground-block flavored),
pickup pop; no device → runs silent. Version system: Cargo 0.2.0, banner +
window title from CARGO_PKG_VERSION, CHANGELOG.md, README versioning policy.
Fixed a use-after-free crash: copying `pipeline_info` into `water_info`
carried dangling `p_*` pointers (statement-local temporaries) — the water
pipeline now builds + creates in one statement. TERRAIN_VERSION bumped to 3
(one-time chunk-cache regeneration).

Milestone 21: **stale-cache invalidation + 2×2 grid fixes** — chunk caches
are now tagged seed XOR TERRAIN_VERSION, so changing terrain generation
(caves, ores, atlas layout) automatically regenerates stale chunks instead
of rendering them with the wrong atlas/generator (the "stretched textures"
and "sometimes nothing renders" screenshots were exactly this). world.conf
accepts v1 saves (edits survive; chunks regenerate). Root-caused the 2×2
crafting-grid dead cell to TWO stacked bugs: the hit-test scanned the wrong
backing-array range, AND the click guard `idx >= size*size` rejected index 4
(= size²) — the row/col decomposition now guards correctly. A regression
test loads a table recipe through all four 2×2 cells including bottom-right.

Milestone 20: **stone tier, coal ore + caves** — the underground is worth
digging into now. 3D-noise cave worms (two crossed value-noise bands, sealed
under 3 blocks of crust) wind through the stone band. Coal ore veins
(hashed 4³ pockets, ~28%) speckle the stone. Stone/coal REQUIRE a pickaxe to
break (the crack stalls — the game logs a hint); the wooden pickaxe is the
progression gate, and a STONE PICKAXE (3 stone + 2 sticks at the crafting
table) mines 1.6× faster than wood. New save-format block id for coal (v1
field added, ids stable).

Milestone 19: **threaded meshing + frustum culling + crafting UX fixes** —
chunk meshing (terrain math AND chunk-cache IO) moved to a background worker
thread (`src/mesher.rs`); the frame thread only drains finished geometry and
uploads it under a 6 MiB/frame budget, near→far, max 64 queued per frame.
Per-chunk frustum culling (Gribb-Hartmann plane extraction + AABB tests in
both the shadow and main passes) skips chunks outside the view. The inventory
recipe list was removed (crafting is grid-driven; right-click places single
items from the cursor, left-drag moves stacks). Fixed: the 2×2 grid's
bottom-right cell was unclickable (the hit-test scanned backing-array indices
0..4 but the 2×2 grid lives on 0,1,3,4).

Milestone 18: **world saving + chunk mesh cache** — the seed, all mined/placed
edits, the player position/look/mode, and the time of day persist to
`~/.local/share/voxelcraft/world/` (override with `VOXELCRAFT_SAVE_DIR`):
`world.conf` is a small text header, `edits.bin` a compact binary overlay
(13 bytes per edit), both written atomically. Edits flush after mining or
placing, autosave runs every 10 s (the seed is saved within the first 10 s
even with no edits), and a final save happens on window close. Every chunk
mesh is also cached to `chunks/<cx>_<cz>.mesh`, tagged with the seed +
format version, so a restart loads terrain from disk instead of re-meshing —
361 chunks loaded in ~0.3 s from cache vs ~1.8 s re-meshing in the smoke
test. A save from a different seed or format version is ignored (and pruned).

Milestone 17: **coherent crafting + viewmodel depth hardening** — the recipe
list, shaped-grid output, and keyboard quick-craft path now all use the same
`CRAFT_RECIPES` source of truth. E opens a 2×2 grid; right-clicking a placed
crafting table opens the 3×3 grid, where the table-only wooden pickaxe and axe
patterns can be placed and crafted. The output consumes one ingredient from
each occupied cell and returns the result to the hotbar/backpack; closing E
returns both cursor-held and grid stacks instead of losing them. Keyboard
quick-craft accepts ingredients across inventory stacks, requires the 3×3
table UI for tools, and rolls back safely if the inventory is full. Recipe
labels are generated from the same patterns, so the menu cannot advertise a
recipe with different inputs. The first-person arm/tool boxes now offset
faces along their normals, so their six faces have real depth rather than
collapsing into flat squares. Shadow light frustum snapping is applied to the
actual light camera center to reduce sub-texel shimmer.

Milestone 17: **coherent crafting + viewmodel depth hardening** — the recipe
list, shaped-grid output, and keyboard quick-craft path now all use the same
`CRAFT_RECIPES` source of truth. E opens a 2×2 grid; right-clicking a placed
crafting table opens the 3×3 grid, where the table-only wooden pickaxe and axe
patterns can be placed and crafted. The output consumes one ingredient from
each occupied cell and returns the result to the hotbar/backpack; closing E
returns both cursor-held and grid stacks instead of losing them. Keyboard
quick-craft accepts ingredients across inventory stacks, requires the 3×3
table UI for tools, and rolls back safely if the inventory is full. Recipe
labels are generated from the same patterns, so the menu cannot advertise a
recipe with different inputs. The first-person arm/tool boxes now offset
faces along their normals, so their six faces have real depth rather than
collapsing into flat squares. Shadow light frustum snapping is applied to the
actual light camera center to reduce sub-texel shimmer.

draggable sliders** — settings now live in
`$XDG_CONFIG_HOME/voxelcraft/settings.conf` (fallback `$HOME/.config/...`),
written atomically (tmp+rename) whenever anything changes and reloaded at
startup (`Settings::load`, parse ignores junk/out-of-range lines; round-trip
test). Sliders are mouse-draggable: press on the track jumps + drags the
value (wheel still steps). The **sky gradient dome** replaces the flat clear
color: a camera-centered hemisphere (16×32 rings) drawn first with depth
off through the new dome pipeline (`dome_vert/frag.glsl`, camera pos rides
the push-constant pad field), interpolating the zenith/horizon palette plus
a dusk forward-scatter glow on the sun's side; terrain fog now targets the
same horizon palette so fogged terrain melts seamlessly into the gradient
(seam covered by a Rust↔GLSL palette-sync test). **First-person hand**
(`src/hand.rs`, drawn through the world pipeline so it's textured + lit +
occluded): holds the selected block as a real mini textured cube
(per-face atlas tiles), tools as extruded 8×8 pixel sprites (wood handle +
lighter head), or a bare skin-toned arm; swing chop loops while mining and
fires once per placement, equip dip on item change, walk bob from actual
movement. Overlay buffer grew to 6 MiB (hand + dome slices added at the
tail, past the F3 slice).

Milestone 15: **settings interaction fixes + cloud settings + no auto-jump**
— the master bug behind "the shadows toggle does nothing" and the dead
slider: the settings cursor Y was computed **inverted** (`1.0 − y/h·2`
instead of `y/h·2 − 1`), so hover/clicks/wheel hit the vertically mirrored
screen position; all settings interaction now lands correctly. **ESC** opens
settings in game and closes it back into the game (mouse re-grabs).
New **CLOUD DISTANCE** slider (2–10 cells, one cell = 14 blocks; the layer
fills radius-cells around the player — worst case 31.7k verts, slice and
buffer re-budgeted to 4 MiB) and the wind slowed to 0.5 blocks/s.
Recipe rows are two-line (name on top, inputs below) so text never collides
with the hotkey chip (the garbled "PLAINIS" overlap). Mining cracks: the
−Z face's origin was [1,0,0], floating its cells one block east — fixed +
a bounds regression test. **Auto-jump removed**: any rise blocks horizontal
movement; 1-block ledges need Space (jump apex ≈ 1.29 blocks clears one).
Render-distance slider verified live end-to-end (stream radius, far plane,
fog all follow it every frame).

Milestone 14: **UI pipeline routing + overlay fixes** — the root cause of
several symptoms was pipelines being fed the wrong primitive type: HUD
"outline" borders are thin TRIANGLE quads but drew through a LINE_LIST
pipeline (broken 1-px traces), so the separate hud_pipeline is gone —
everything HUD draws through the fill pipeline; the block highlight is
crossed thin quads (bold from any angle) and now draws through the
sky (triangle) pipeline, inflated only along the hit-face normal (the old
all-axis inflate read as off-center). Cracks: `build_crack_overlay` was
rebuilt without the block's world offset — every crack cube rendered at the
origin, underground; offset re-added. Inventory block icons were passed
into the line pipeline arg (wireframe cubes) — arg order fixed. Crosshair
now scales with the UI (UI_SLOT-derived, like the rest of the HUD), and the
E-panel hover plate is visibly green-tinted.

Milestone 13: **texture-stretch fix, settings redesign, ESC menu, 3D
clouds** — `tangent_axes` matched Z faces into the X arm (pattern `[0, _, _]`
also catches `[0, 0, ±1]`), so side-face V came from the constant Z axis:
one texel row stretched over the whole face (and AO sampled the wrong
axes). Fixed + regression test. The settings page is now a wide centered
dialog — labels left, controls centered, values right-aligned, hover rows,
ESC-to-close hint — no more labels colliding with sliders. **ESC now opens
settings + frees the mouse** (ESC again closes and re-grabs; quit is the
window's close button). Clouds are **3D boxes** — all 6 faces, hashed
random position/footprint/altitude/thickness per puff, ~45% get a second
lumpy tier — drifting east and recentering on the camera.

Milestone 12: **true rotating sun + texture-orientation fixes** — the
sun/moon now ride a real great circle (`sky.rs` keyframe table replaced with
an analytic circular path: dawn east → noon up → sunset west, continuous
rotation all night long, no snapping), with sky/fog colors and light
intensities derived from the sun's elevation. Grass-side/bark/plank textures
no longer render upside down (side-face V flipped to match the atlas row
order), and the far plane never dips below 110 so the discs (at 90 blocks)
stay visible on low render distances. Also fixed a latent slice-stride bug:
the overlay buffer slice math still assumed the old 36-byte vertex stride
after UVs were added (44 B), which had been wiping out the HUD and the
sun/moon discs every frame.

Milestone 10: **clouds, centered inventory, settings page** — a drifting
cloud layer (40 blocks up, hashed blobby cells streaming east on the wind,
shaded by the day/night cycle, depth-tested so hills occlude them); the
E inventory is now **centered on screen** with a recipe column on the right
(icon rows for the current C/R/T/V recipes; new recipes plug in later);
and **O opens a settings page** with a render-distance **slider**
(scroll over it; 1–8 chunks — live: stream radius, camera far plane, and
fog end all follow it), **shadows** and **clouds** toggles, and grayed-out
rows for the planned shader-pack features (RT shadows, RT AO, god rays).
The crosshair is screen-space now — pinned to the exact screen center
(the old world-space projection drifted off the true aim point and wobbled
on resize). Panel hit-testing now scales the cursor by the aspect so
clicks land correctly on any window shape.

Milestone 8: **tools + mining animation** — hold left-click to mine:
progress cracks (10 destroy stages of chunky dark pixel cells that appear
near the face center and spread outward) cover the targeted block,
and when it breaks, particles in the block's color burst out and fall while
the item becomes a **floating mini-cube drop** (spins, bobs, magnet-flies
to you within ~1.5 blocks, then collects into the inventory). New item
types beyond blocks: **sticks** and **wooden pickaxe/axe** — recipes:
C 1 log → 4 planks, R 2 planks → 4 sticks, T 2 sticks → wood pickaxe,
V 2 sticks → wood axe. The pickaxe mines stone 7× faster (5 s → 0.71 s),
the axe chops wood 5× faster; tools/sticks render as pixel-bitmap icons in
the hotbar and can't be placed.

Milestone 7: **sun & moon discs + shadow mapping** — billboarded sun/moon
quads (drawn depth-tested, so terrain/hills occlude them; the moon rides
opposite the sun and is slightly smaller; both **fade out and vanish at
the horizon** instead of orbiting below it, since no terrain exists past
the view radius to occlude them — the sun's arc is symmetric:
dawn 0.0 → noon 0.25 → sunset 0.5 → night hold 0.62–0.92), plus a real
2048² shadow map:
a depth-only pre-pass renders terrain from an orthographic light frustum
(centered on the player, texel-snapped to avoid shimmer; elevation-clamped
sun by day, moon by night), and the world fragment shader samples it with
hardware 2×2 PCF. Trees, cliffs, and buildings cast proper shadows that
de-strengthen at night (SHADOW_STRENGTH keeps shadowed faces shaded, not
black) and fade out near the horizon like Minecraft's.

Milestone 6: **F3 debug overlay** — press F3 to toggle a stats panel
(top-left: FPS/frame time, XYZ, chunk + facing, mode/time/grounded,
triangle + mesh + edit counts, GPU name and Vulkan API version, driver/
device/vendor IDs, swapchain format and image count, CPU model, live RAM
usage, seed, backend) plus in-world wireframes: the player's 0.6×1.8
collision box (red) and the targeted cell (cyan). **Day/night cycle** — a
10-minute day drives the sun's arc,
sky/fog colors (dawn → day → dusk → deep night → dawn), and terrain
lighting: nights are genuinely dark under moonlight, dawns/dusks glow
orange. The sky color IS the clear color, so fog always fades into the
exact sky behind it. **Block textures** (Milestone 11): every terrain face
samples a procedurally generated 16×16 noise tile from a 9-tile atlas
(`src/textures.rs`, built on the CPU at startup — grass top/side fringe,
dirt, stone, bark grain, log rings, planks with seams, leaves). Tree leaves
have ~18% transparent texels keyed on a magenta sentinel (255, 0, 254) that
the world fragment shader discards, so canopies look airy; leaf-vs-leaf
faces now render (previously culled) so the holes reveal inner foliage.
All overlay/HUD geometry samples tile 0 (solid white) via uv (0,0), so the
UI is untouched by the texture path. Fixed in the same pass: the HUD push
constants carried fog 0/0 → NaN fog → the entire UI rendered white.
**Dynamic inventory**: hotbar slots are no longer
type-locked — drops stack onto a slot of the same type, or fill the first
empty slot left→right; a spent slot frees up for the next new type.
**E inventory panel** (Milestone 9): 18-slot backpack (3×6 grid) above a
mirrored hotbar row — 24 clickable slots. E opens/closes it (cursor
released while open); left-click picks up / places / merges / swaps
stacks; a held stack follows the cursor and returns to a free slot when
the panel closes or you click empty space. New items fill the hotbar
first, then overflow to the backpack; a full inventory leaves drops on
the ground instead of losing them.
**Collision fix**: movement now requires the whole body to FIT at the
destination (surface ≤ 1 block up AND both body cells free) — tall walls
no longer teleport-climb, and solid blocks can no longer be phased through
at surface level. The fit test checks exactly the two cells a 1.8-tall body
spans, so **2-block-tall gaps are walkable** (the earlier off-by-one demanded
a phantom third free cell).

Milestone 5: trees, hotbar & crafting seed — **trees** (deterministic from
the seed: trunk + leaf blob, no storage) grow across the terrain and can be
mined. **6 placeable block types**: grass, dirt, stone, log, leaves, planks.
**1–6 select the hotbar slot** (the on-screen hotbar at the bottom of the
screen shows a 3D isometric block icon with dark edges per slot plus the
exact count in a 3×5 pixel font, aspect-corrected so shapes stay square at
any window size, with a bright outline on the selected slot; the title HUD
shows the same counts, and the selected item's name appears above the bar
for ~2 s on every switch — by number key or scroll wheel). **C crafts
1 log → 4 planks** — the crafting
seed that tool recipes will build on. Left/right-click mining/placing and
the you-can-only-place-what-you've-mined inventory rule are unchanged.

Chunk streaming now **unloads** meshes beyond view radius +1 (hysteresis
band prevents border thrash): GPU memory stays bounded while roaming, and
re-entering an area regenerates it with all player edits intact (edits are
keyed by world coordinates, not by chunk).

Mining & placing — **left-click mines** the
targeted block (DDA voxel raycast, 5-block reach), **right-click places**
from the selected slot, adjacent to the aimed face. Mined grass drops dirt
(Minecraft-style). You can only build with what you've mined. A crosshair +
wireframe highlight shows the target; the window title
is a HUD (selection, per-slot inventory, mode, fps, position). Edited chunks
re-mesh automatically (including neighbors); edits persist for the session
and overlay the procedural terrain. Blocks render with per-block color
jitter (subtle natural variation).

Milestone 4: walking mode — gravity, heightmap collision (walls block,
≤1-block ledges auto-step), jumping. **G toggles walking / creative flight.**

Milestone 3: procedural terrain — deterministic value-noise fBm heightmap
meshed as 16×16 chunks (9×9 loaded around the camera, ~46k triangles) with
hidden-face culling. Chunks stream in as you fly across chunk borders and
unload when out of range. Each run's world is random (time-seeded).

## How to run the native build (primary artifact)

```bash
cargo run
```

Requirements (already present on this machine):
- Rust 1.97+ (`cargo --version`)
- Vulkan loader + RADV driver (`vulkaninfo` works)
- `glslc` on PATH (Vulkan SDK tooling) — used by `build.rs` to compile
  `shaders/vert.glsl` + `shaders/frag.glsl` to SPIR-V at build time

Behavior: opens a 1280×720 window showing blocky procedural terrain
(grass tops, stone sides, dirt underside) against a sky-blue background.
Spawn is on the terrain surface in **walking** mode; press G to fly.

Physics (walking): Minecraft-like — walk 7 blocks/s (brisker than MC default),
gravity 28 blocks/s², ~1.25-block jump height (clears one block), no
auto-jump — 1-block ledges need Space, walls and steep hills block movement.
Falling far below the terrain respawns on the surface.

Lighting: Minecraft-style fixed per-face shading (top brightest, sides
distinct per axis, bottom darkest) with a mild directional sun term, plus
per-vertex ambient occlusion ("smooth lighting": corners adjacent to more
solid blocks are darker, with the classic quad-flip to avoid crease
artifacts) and distance fog fading terrain into the sky beyond ~40 blocks.

Windowing backend (Linux):
- Defaults to **X11/XWayland**. On this SteamOS desktop-mode machine the
  Wayland path submits frames that KWin fails to composite (window shows
  magenta), so X11 is the supported default.
- Force Wayland with `VOXELCRAFT_BACKEND=wayland` if the compositor
  situation changes.
- The window title shows the active backend, and a stall watchdog logs an
  error if no frames present for ~2 seconds.

Controls:
- WASD — move (walk or fly)
- Space — jump (walking) / fly up (flying)
- Left Shift — fly down (flying)
- G — toggle walking ↔ flying
- Left click — mine the targeted block (grass drops dirt; +1 per-type)
- Right click — place the selected type (−1; refused when empty)
- 1…6 — select the hotbar slot (grass/dirt/stone/log/leaves/planks)
- Scroll wheel — cycle the hotbar slot (either direction, wraps around)
- C — craft 1 log → 4 planks
- Mouse — look (raw motion events only; cursor-position events are ignored)
- [ / ] — decrease / increase mouse sensitivity (logged to console)
- Left-click — capture the mouse (also auto-captured on window focus)
- ESC once — release the cursor; ESC again (or window close) — exit
- F — re-grab the cursor after releasing it (or if grab failed at startup)
- F3 — toggle the debug overlay (stats panel + collision box + target cell)
- E — inventory panel (backpack + hotbar, click to move stacks)
- O — settings page (render-distance + cloud-distance sliders, shadow/cloud
  toggles; scroll over a slider row to change it, click a toggle row)
- ESC — in game: open settings; in settings/inventory: close and re-grab

If cursor capture fails (e.g. some compositors refuse Lock), mouse look is
disabled and a warning is logged; keyboard movement still works.

## How to run the browser preview (this Preview tab)

No server, no port, no install — the preview is a single self-contained HTML
file that mirrors the native renderer's output with raw WebGL:

- File: `preview/index.html` (repo root, committed with the project)
- The Preview tab registers this file directly (`htmlPath` mode); any static
  file server pointed at the repo root also works if preferred.
- The page needs no network access and no dependencies.

The preview mirrors the native app's milestone-1 look (static cube from a
fixed angle, same face colors, directional light, and clear color) so shader
or color changes can be eyeballed here. The camera movement added in
milestone 2 is native-only — a browser mirror would need input plumbing that
would diverge from the real renderer.

## Reproducing artifacts in a fresh checkout

Nothing to copy — there are no env files, secrets, or generated assets.
`cargo build` regenerates SPIR-V via `build.rs` into `target/` automatically.
