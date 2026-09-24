//! VoxelCraft — a Minecraft-like voxel engine built on raw Vulkan (ash).
//!
//! Milestone 3: procedural terrain — noise heightmap meshed as 16×16 chunks
//! with hidden-face culling, rendered under the milestone-2 fly camera.
//!
//! Milestone 4: walking mode — gravity, terrain collision, jumping; G toggles
//! between walking and creative flight.
//!
//! Milestone 5: trees, hotbar & crafting seed — analytic trees (logs +
//! leaves), 6 placeable block types, per-type inventory gated by mining,
//! screen-space hotbar HUD, and C to craft logs into planks.
//!
//! Module layout:
//! - [`camera`]   — fly camera + view/projection math
//! - [`input`]    — raw mouse look + sensitivity
//! - [`player`]   — movement physics: walking (gravity, collision, jump) / flying
//! - [`geometry`] — vertex format shared by all meshes
//! - [`terrain`]  — value-noise heightmap + per-chunk meshing (culling + AO)
//! - [`world`]    — editable voxel state (edits, queries, raycasting)
//! - [`sky`]      — day/night cycle: sun path + sky/light palette
//! - [`renderer`] — raw Vulkan renderer (instance → device → swapchain → draw)
//! - [`app`]      — winit event-loop glue (input handling, resize, cursor grab)
//!
//! Controls: WASD move, mouse look, [ ] sensitivity, E inventory,
//! O/ESC settings, F3 debug
//! ESC opens/closes menus, F re-grab cursor, left-click mine,
//! right-click place, 1–7 / scroll select hotbar slot.

mod app;
mod camera;
mod debug_overlay;
mod geometry;
mod hand;
mod input;
mod items;
mod items_drop;
mod mesher;
mod overlay;
mod player;
mod renderer;
mod save;
mod settings;
mod sky;
mod sound;
mod sys;
mod terrain;
mod textures;
mod world;

use std::error::Error;

use winit::event_loop::EventLoop;

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Backend selection: on this machine (SteamOS desktop mode — KDE Wayland,
    // RADV VANGOGH), the Wayland path submits frames that KWin fails to
    // composite (window shows magenta) while the X11/XWayland path renders
    // correctly. Default to X11; force Wayland with VOXELCRAFT_BACKEND=wayland.
    let backend = match std::env::var("VOXELCRAFT_BACKEND") {
        Ok(b) if b.eq_ignore_ascii_case("wayland") => "wayland",
        _ => "x11",
    };

    log::info!(
        "VoxelCraft v{} starting (WASD + mouse, {backend} backend)",
        env!("CARGO_PKG_VERSION")
    );

    let event_loop = match backend {
        "wayland" => EventLoop::builder().build()?,
        _ => {
            use winit::platform::x11::EventLoopBuilderExtX11;
            EventLoop::builder().with_x11().build()?
        }
    };
    let mut app = app::App::new(backend.to_uppercase());
    event_loop.run_app(&mut app)?;
    Ok(())
}
