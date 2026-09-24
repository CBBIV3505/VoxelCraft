//! Raw Vulkan renderer (ash): instance → device → swapchain → pipeline →
//! frame loop. Milestone 2 draws a single static mesh with a per-frame
//! view-projection push constant.

use std::error::Error;
use std::ffi::CStr;

use ash::ext::debug_utils;
use ash::khr::{surface, swapchain};
use ash::util::read_spv;
use ash::vk;
use glam::{Mat4, Vec3};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::camera::{view_proj_far, Camera};
use crate::geometry::Vertex;
use crate::overlay::build_overlay;
use crate::overlay::NUM_SLOTS;
use crate::world::RayHit;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// GPU/driver facts captured at init for the F3 debug overlay.
#[derive(Clone, Debug, Default)]
pub struct GpuInfo {
    pub name: String,
    pub vulkan_api: String,
    pub driver_version: u32,
    pub device_id: u32,
    pub vendor_id: u32,
    pub swapchain_format: String,
    pub msaa: String,
}

/// Human-readable Vulkan format name (the ones this engine can hit).
fn format_name(f: vk::Format) -> String {
    match f {
        vk::Format::B8G8R8A8_SRGB => "B8G8R8A8_SRGB",
        vk::Format::R8G8B8A8_SRGB => "R8G8B8A8_SRGB",
        vk::Format::B8G8R8A8_UNORM => "B8G8R8A8_UNORM",
        vk::Format::R8G8B8A8_UNORM => "R8G8B8A8_UNORM",
        other => return format!("{:?}", other),
    }
    .to_string()
}

/// Per-frame push constants: MVP + day/night lighting (shared by all
/// pipelines; the HUD ignores the lighting fields).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    mvp: Mat4,
    /// Mostly padding — but the SKY DOME shaders read xyz as the camera
    /// position (the dome's center) to derive the gradient height, so the
    /// dome draw must fill it.
    pad: [f32; 4],
    sun_dir: [f32; 4],    // xyz = direction TO the sun
    sky_params: [f32; 4], // x = sun intensity, y = ambient
    fog_params: [f32; 4], // x = fog start, y = fog end, z = shadows enabled
}

/// Host-visible buffer size for per-frame dynamic overlay/HUD vertices.
/// Headroom rationale: the F3 panel renders ~500 characters ≈ 30k vertices;
/// the sky dome is 3072; the hand ≈ 72; everything else is small — 6 MiB
/// (≈143k verts) holds every slice with generous F3 headroom.
const OVERLAY_BUFFER_SIZE: u64 = 6291456;
/// World overlay (crosshair + block highlight) lives at the start of the
/// per-frame buffer; the HUD hotbar section is appended after this slice.
/// (Both used to write at offset 0 — the HUD silently clobbered the
/// crosshair/highlight data every frame it was drawn.)
const OVERLAY_WORLD_OFFSET: u64 = 0;
/// HUD section start, expressed in VERTICES so the byte offset stays an
/// exact multiple of the vertex stride — a byte offset that doesn't divide
/// evenly (e.g. 16384 % 36 ≠ 0) shifts every vertex by 4 bytes and the HUD
/// renders as garbage lines across the screen.
/// HUD slice: hotbar ≈ 2000 verts + the E inventory panel with its recipe
/// column ≈ 13k worst case + the settings page ≈ 10k worst case → ends at
/// 27136 verts (all slice starts are multiples of 64 so byte offsets stay
/// 256-aligned for map_memory).
const OVERLAY_HUD_VERTEX_START: u32 = 512;
/// HUD slice byte offset (512 verts × 36 B = 18432; also 256-aligned, which
/// satisfies the memory-map offset alignment requirement).
const OVERLAY_HUD_OFFSET: u64 = OVERLAY_HUD_VERTEX_START as u64 * VERTEX_STRIDE as u64;
/// Screen-space crosshair slice (24 verts; 27136 × 36 B = 976896,
/// 256-aligned).
const OVERLAY_CROSSHAIR_VERTEX_START: u32 = 27136;
const OVERLAY_CROSSHAIR_OFFSET: u64 = OVERLAY_CROSSHAIR_VERTEX_START as u64 * VERTEX_STRIDE as u64;
/// Sun/moon disc slice start, in vertices (after the crosshair; the
/// 2 discs only need 24 vertices).
const OVERLAY_DISC_VERTEX_START: u32 = 27200;
/// Drifting cloud slice (after the discs): worst case at CLOUD DISTANCE 10
/// = (2·10+1)² cells × 2 puffs × 36 verts = 31752; 31808 fits exactly.
const OVERLAY_CLOUD_VERTEX_START: u32 = 27264;
const OVERLAY_CLOUD_OFFSET: u64 = OVERLAY_CLOUD_VERTEX_START as u64 * VERTEX_STRIDE as u64;
/// Item drops + particles slice: drops ≈ 20 × 36 verts + particles ≈ 60 × 6
/// verts ≈ 1080; 1536 verts of headroom.
const OVERLAY_BILLBOARD_VERTEX_START: u32 = 59072;
const OVERLAY_BILLBOARD_OFFSET: u64 = OVERLAY_BILLBOARD_VERTEX_START as u64 * VERTEX_STRIDE as u64;
/// Mining-crack slice: filled pixel cells, worst case 6 faces × 16 cells ×
/// 6 verts = 576; 768 gives headroom.
const OVERLAY_CRACK_VERTEX_START: u32 = 60608;
const OVERLAY_CRACK_OFFSET: u64 = OVERLAY_CRACK_VERTEX_START as u64 * VERTEX_STRIDE as u64;
/// F3 debug panel slice start, in vertices and bytes (after the crack
/// slice; 61376 × 44 B = 2700544, 256-aligned for map_memory).
const OVERLAY_DEBUG_VERTEX_START: u32 = 61376;
const OVERLAY_DEBUG_OFFSET: u64 = OVERLAY_DEBUG_VERTEX_START as u64 * VERTEX_STRIDE as u64;
/// First-person hand slice (held block/tool): a block is 36 verts, a tool
/// sprite extrudes one box per lit pixel — worst case 8×8 = 64 cells × 36
/// = 2304 verts + arm ≈ 2340. 8192 gives >3× headroom so a new sprite
/// shape can never panic the slice again.
const OVERLAY_HAND_VERTEX_START: u32 = 95424;
const OVERLAY_HAND_VERTEX_COUNT: u32 = 8192;
const OVERLAY_HAND_OFFSET: u64 = OVERLAY_HAND_VERTEX_START as u64 * VERTEX_STRIDE as u64;
/// Sky gradient dome slice, at the tail of the buffer (after the hand
/// slice): SKY_BANDS × SKY_SECTORS × 6 = 3072 verts; 103616 × 44 B =
/// 4559104 (256-aligned for map_memory; end 106688 × 44 ≪ 6 MiB).
const OVERLAY_DOME_VERTEX_START: u32 = 103616;
const OVERLAY_DOME_OFFSET: u64 = OVERLAY_DOME_VERTEX_START as u64 * VERTEX_STRIDE as u64;
/// Size of one overlay/HUD vertex in bytes (pos + normal + color + uv).
/// Keep the slice layout tied to the actual Rust vertex type so adding an
/// attribute cannot silently corrupt every draw after the first slice.
const VERTEX_STRIDE: u32 = std::mem::size_of::<Vertex>() as u32;

// --- Shadow mapping ---------------------------------------------------------

/// Shadow map resolution (square): 2048 texels over a focused 144-block footprint.
const SHADOW_MAP_SIZE: u32 = 2048;
/// Half-extent of the square orthographic shadow frustum around the camera.
/// Keeping this local makes the same map much sharper than a render-distance-
/// sized shadow box while still covering the nearby terrain the player sees.
const SHADOW_FRUSTUM_HALF: f32 = 72.0;
/// The light camera is placed along the light direction so the camera/player
/// sits safely in front of the RH orthographic near plane.
const SHADOW_LIGHT_DISTANCE: f32 = 120.0;
/// Positive view-space distances for glam's right-handed [0,1] orthographic
/// projection (the old negative near value put most terrain outside the map).
const SHADOW_NEAR: f32 = 0.1;
const SHADOW_FAR: f32 = 280.0;
/// Fraction of direct light kept inside shadows (so shadowed ground still
/// shows its face shading instead of going pitch black).
const SHADOW_STRENGTH: f32 = 0.58;

/// Push constants for the shadow (depth-only) pass.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowPush {
    light_vp: Mat4,
    params: [f32; 4], // x = world size of one shadow texel (bias scale)
}

/// An uploaded triangle mesh: device-local buffers + index count.
pub struct Mesh {
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    index_count: u32,
    /// World-space translation applied when drawing (chunk origin).
    pub model_offset: [f32; 3],
}

impl Mesh {
    #[allow(dead_code)] // handy for terrain stats in upcoming milestones
    pub fn vertex_count(&self) -> u32 {
        self.index_count / 3
    }
}

/// Key for identifying an uploaded mesh (e.g. by chunk coordinate).
pub type MeshKey = (i32, i32);

// ---------------------------------------------------------------------------
// Debug callback
// ---------------------------------------------------------------------------

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = unsafe { CStr::from_ptr((*callback_data).p_message) };
    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("{msg:?}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::warn!("{msg:?}"),
        _ => log::debug!("{msg:?}"),
    }
    vk::FALSE
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

pub struct Renderer {
    #[allow(dead_code)] // kept for future milestones (multi-queue, memory reuse)
    entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: surface::Instance,
    surface_khr: vk::SurfaceKHR,
    pdevice: vk::PhysicalDevice,
    #[allow(dead_code)] // queue family index; needed when we split transfer/graphics queues
    graphics_family: u32,
    device: ash::Device,
    queue: vk::Queue,

    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,

    depth_format: vk::Format,
    /// Per-swapchain-image depth attachments avoid concurrent frame writes to
    /// one shared depth image when multiple frames are in flight.
    depth_images: Vec<vk::Image>,
    depth_memories: Vec<vk::DeviceMemory>,
    depth_views: Vec<vk::ImageView>,

    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    /// Line-list overlay pipeline (same shaders): block highlight + crosshair.
    line_pipeline: vk::Pipeline,
    /// Depth-test-off triangle pipeline (solid hotbar fills AND outlined
    /// borders — borders are thin triangle quads, not true lines).
    hud_fill_pipeline: vk::Pipeline,
    /// Translucent water pipeline: world shader + alpha blending, depth
    /// write off, no cull. Water meshes draw with this after all opaque
    /// geometry.
    water_pipeline: vk::Pipeline,
    /// Per-frame dynamic vertex buffer for the line overlay (host visible).
    overlay_buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    pipeline_layout: vk::PipelineLayout,
    framebuffers: Vec<vk::Framebuffer>,

    #[allow(dead_code)] // reset/trim per-frame allocations in later milestones
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>, // one per frame in flight

    /// Uploaded meshes keyed by chunk coordinate.
    meshes: Vec<(MeshKey, Mesh)>,
    /// Translucent water surface meshes, keyed by the same chunk coords as
    /// `meshes` but in SEPARATE storage: a shared key namespace with an
    /// offset sentinel once collided with real chunk keys south of spawn
    /// and destroyed their terrain meshes on water upload (disappearing
    /// chunks). Separate vectors make cross-namespace collision impossible
    /// by construction.
    water_meshes: Vec<(MeshKey, Mesh)>,
    /// Staging copies staged by create_device_buffer_from, flushed into the
    /// batch's single command buffer (cleared after each upload_meshes).
    pending_copies: Vec<(vk::Buffer, vk::Buffer, vk::DeviceSize, vk::DeviceSize)>,

    /// Hotbar slot currently selected by the player (for the HUD outline).
    selected_slot: usize,
    /// Per-slot contents (item + count) for the HUD icons and counts.
    slot_slots: [(Option<crate::items::ItemType>, u32); NUM_SLOTS],
    /// Selected-item label shown above the hotbar (empty = hidden).
    slot_label: String,
    /// Backpack contents + open/hover/cursor state for the E panel.
    backpack: [(Option<crate::items::ItemType>, u32); crate::overlay::BACKPACK_SLOTS],
    inv_open: bool,
    inv_hover: Option<usize>,
    inv_cursor: (f32, f32),
    inv_held: Option<crate::items::ItemType>,
    inv_held_count: u32,
    /// Active inventory crafting grid and hover state.
    craft_grid: [(Option<crate::items::ItemType>, u32); 9],
    craft_size: usize,
    craft_hover: Option<usize>,
    craft_output_hover: bool,
    /// F3 debug state: panel lines (None = hidden), accent line count, and
    /// in-world box line vertices.
    debug_panel: Option<Vec<String>>,
    debug_highlight: usize,
    debug_box: Vec<crate::geometry::Vertex>,
    /// Block texture atlas (16×16 noise tiles) + its view and sampler.
    atlas_image: vk::Image,
    atlas_memory: vk::DeviceMemory,
    atlas_view: vk::ImageView,
    atlas_sampler: vk::Sampler,
    /// Mining crack overlay: filled pixel cells (world-space quads,
    /// depth-tested, drawn through the billboard pipeline).
    crack_verts: Vec<crate::geometry::Vertex>,
    /// First-person viewmodel (hand + held item), world-space, drawn after
    /// terrain so depth handles occlusion. Empty = nothing held/not shown.
    hand_verts: Vec<crate::geometry::Vertex>,
    /// Item drops + break particles (world-space billboards, depth-tested,
    /// drawn through the sky-disc pipeline which is exactly that).
    billboard_verts: Vec<crate::geometry::Vertex>,
    /// Graphics settings + settings-page UI state (drives the shadow pass,
    /// camera far plane, fog, and the O settings panel).
    settings: crate::settings::Settings,
    settings_open: bool,
    settings_hover: Option<usize>,

    // Per FRAME in flight (not per image — image index is unknown at acquire time)
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    image_fences: Vec<vk::Fence>,
    current_frame: usize,
    frames_presented: u64,

    #[allow(dead_code)] // freed in destroy()
    debug_messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    /// GPU/driver facts for the F3 overlay.
    gpu_info: GpuInfo,

    // --- Shadows + celestial discs ------------------------------------------
    shadow_pass: vk::RenderPass,
    shadow_framebuffers: Vec<vk::Framebuffer>,
    shadow_images: Vec<vk::Image>,
    shadow_memories: Vec<vk::DeviceMemory>,
    shadow_views: Vec<vk::ImageView>,
    shadow_pipeline: vk::Pipeline,
    shadow_sampler: vk::Sampler,
    /// Per-frame host-visible UBOs carrying the light view-projection.
    light_vp_buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    descriptor_pool: vk::DescriptorPool,
    #[allow(dead_code)] // destroyed in destroy(); sets reference it
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
    /// Sun/moon billboard pipeline (depth-tested, no depth write).
    sky_pipeline: vk::Pipeline,
    /// Sky gradient dome: own shader pair (no descriptors, no fog), depth
    /// test + write OFF so it paints behind everything drawn after it.
    dome_pipeline: vk::Pipeline,
    /// Wall-clock seconds for the cloud drift, captured at each draw.
    now_secs: f32,
}

impl Renderer {
    pub unsafe fn new(window: &Window) -> Result<Self, Box<dyn Error>> {
        let entry = ash::Entry::linked();
        let display_handle = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();

        // --- Instance -------------------------------------------------------
        let app_info = vk::ApplicationInfo {
            p_application_name: c"VoxelCraft".as_ptr(),
            application_version: vk::make_api_version(0, 0, 1, 0),
            p_engine_name: c"VoxelCraft".as_ptr(),
            engine_version: vk::make_api_version(0, 0, 1, 0),
            api_version: vk::make_api_version(0, 1, 1, 0), // 1.1 is enough here
            ..Default::default()
        };

        let mut extensions: Vec<*const std::ffi::c_char> =
            ash_window::enumerate_required_extensions(display_handle)
                .map_err(|e| format!("surface extensions: {e}"))?
                .to_vec();

        let mut layers: Vec<*const std::ffi::c_char> = Vec::new();
        let mut debug_info = None;

        // Enable KHRONOS validation if installed (usually via the Vulkan SDK)
        let has_validation = entry
            .enumerate_instance_layer_properties()
            .unwrap_or_default()
            .iter()
            .any(|l| unsafe { CStr::from_ptr(l.layer_name.as_ptr()) } == c"VK_LAYER_KHRONOS_validation");
        if has_validation {
            extensions.push(c"VK_EXT_debug_utils".as_ptr());
            layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
            debug_info = Some(vk::DebugUtilsMessengerCreateInfoEXT {
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                pfn_user_callback: Some(debug_callback),
                ..Default::default()
            });
        } else {
            log::warn!("VK_LAYER_KHRONOS_validation not found, running without validation");
        }

        let instance_info = vk::InstanceCreateInfo {
            p_next: debug_info
                .as_ref()
                .map(|d| d as *const _ as *const std::ffi::c_void)
                .unwrap_or(std::ptr::null()),
            p_application_info: &app_info,
            enabled_layer_count: layers.len() as u32,
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };
        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        // Attach a live messenger too (captures everything after instance creation)
        let debug_messenger = if let Some(info) = &debug_info {
            let loader = debug_utils::Instance::new(&entry, &instance);
            let messenger = unsafe { loader.create_debug_utils_messenger(info, None)? };
            Some((loader, messenger))
        } else {
            None
        };

        // --- Surface --------------------------------------------------------
        let surface_loader = surface::Instance::new(&entry, &instance);
        let surface_khr = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)?
        };

        // --- Physical device -------------------------------------------------
        let pdevices = unsafe { instance.enumerate_physical_devices()? };
        let mut chosen: Option<(vk::PhysicalDevice, u32)> = None;
        for pd in pdevices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let has_swapchain = unsafe {
                instance
                    .enumerate_device_extension_properties(pd)
                    .unwrap_or_default()
                    .iter()
                    .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == c"VK_KHR_swapchain")
            };
            if !has_swapchain {
                continue;
            }
            let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let family = families
                .iter()
                .enumerate()
                .position(|(i, f)| {
                    f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && unsafe {
                            surface_loader
                                .get_physical_device_surface_support(pd, i as u32, surface_khr)
                                .unwrap_or(false)
                        }
                })
                .map(|i| i as u32);
            if let Some(family) = family {
                let prefer_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
                let is_better = chosen
                    .map(|_| prefer_discrete) // keep first unless this is discrete
                    .unwrap_or(true);
                if is_better {
                    chosen = Some((pd, family));
                }
                if prefer_discrete {
                    break;
                }
            }
        }
        let (pdevice, graphics_family) =
            chosen.ok_or("no GPU with graphics + present queue and swapchain support")?;
        let gpu_props = unsafe { instance.get_physical_device_properties(pdevice) };
        let gpu_name = unsafe { CStr::from_ptr(gpu_props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let v = gpu_props.api_version;
        let mut gpu_info = GpuInfo {
            name: gpu_name,
            vulkan_api: format!("{}.{}", (v >> 22) & 0x3FF, (v >> 12) & 0xFFF),
            driver_version: gpu_props.driver_version,
            device_id: gpu_props.device_id,
            vendor_id: gpu_props.vendor_id,
            swapchain_format: String::new(), // filled in below, after surface query
            msaa: "1x (off)".to_string(),
        };
        log::info!("GPU: {}", gpu_info.name);

        // --- Logical device --------------------------------------------------
        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: graphics_family,
            queue_count: 1,
            p_queue_priorities: priorities.as_ptr(),
            ..Default::default()
        };
        let device_extensions = [c"VK_KHR_swapchain".as_ptr()];
        let device_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            ..Default::default()
        };
        let device = unsafe { instance.create_device(pdevice, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(graphics_family, 0) };

        // --- Swapchain -------------------------------------------------------
        let swapchain_loader = swapchain::Device::new(&instance, &device);
        let swapchain = unsafe {
            Self::create_swapchain(
                &surface_loader,
                &swapchain_loader,
                pdevice,
                surface_khr,
                window,
                vk::SwapchainKHR::null(),
            )?
        };
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let swapchain_format = unsafe {
            surface_loader.get_physical_device_surface_formats(pdevice, surface_khr)?[0].format
        };
        gpu_info.swapchain_format = format_name(swapchain_format);
        // Recompute extent the same way create_swapchain did (current_extent can be
        // 0xFFFFFFFF on Wayland/gamescope before the window is mapped).
        let swapchain_extent = Self::compute_extent(
            &unsafe {
                surface_loader.get_physical_device_surface_capabilities(pdevice, surface_khr)?
            },
            window,
        );
        log::info!(
            "swapchain extent: {}x{}, images: {}",
            swapchain_extent.width,
            swapchain_extent.height,
            swapchain_images.len()
        );
        let image_views = unsafe {
            swapchain_images
                .iter()
                .map(|img| Self::create_image_view(&device, *img, swapchain_format))
                .collect::<Result<Vec<_>, _>>()?
        };

        // --- Depth buffer -----------------------------------------------------
        let depth_format = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
            vk::Format::D16_UNORM,
        ]
        .into_iter()
        .find(|f| {
            let props = unsafe { instance.get_physical_device_format_properties(pdevice, *f) };
            props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        })
        .ok_or("no supported depth format")?;
        let mut depth_images = Vec::with_capacity(swapchain_images.len());
        let mut depth_memories = Vec::with_capacity(swapchain_images.len());
        let mut depth_views = Vec::with_capacity(swapchain_images.len());
        for _ in &swapchain_images {
            let (image, memory, view) = unsafe {
                Self::create_depth_image(
                    &instance,
                    &device,
                    pdevice,
                    depth_format,
                    swapchain_extent,
                )
            }?;
            depth_images.push(image);
            depth_memories.push(memory);
            depth_views.push(view);
        }
        let depth_format_ = depth_format;

        // --- Render pass -------------------------------------------------------
        let color_attachment = vk::AttachmentDescription {
            format: swapchain_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };
        let depth_attachment = vk::AttachmentDescription {
            format: depth_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };
        let color_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };
        let depth_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_ref,
            p_depth_stencil_attachment: &depth_ref,
            ..Default::default()
        };
        let dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ..Default::default()
        };
        let render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo {
                    attachment_count: 2,
                    p_attachments: [color_attachment, depth_attachment].as_ptr(),
                    subpass_count: 1,
                    p_subpasses: &subpass,
                    dependency_count: 1,
                    p_dependencies: &dependency,
                    ..Default::default()
                },
                None,
            )?
        };

        // --- Shaders (built by build.rs via glslc into OUT_DIR) ----------------
        // OUT_DIR is a compile-time env var; bake it into the binary via env!().
        let spv_dir = std::path::PathBuf::from(env!("OUT_DIR"));
        let vert_spv = read_spv(&mut std::io::Cursor::new(std::fs::read(
            spv_dir.join("vert.spv"),
        )?))?;
        let frag_spv = read_spv(&mut std::io::Cursor::new(std::fs::read(
            spv_dir.join("frag.spv"),
        )?))?;

        let make_module = |code: Vec<u32>| -> Result<vk::ShaderModule, ash::vk::Result> {
            unsafe {
                device.create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        code_size: code.len() * 4,
                        p_code: code.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
            }
        };
        let vert_module = make_module(vert_spv)?;
        let frag_module = make_module(frag_spv)?;

        // --- Pipeline ----------------------------------------------------------
        // Shared descriptor layout: binding 0 = light view-projection UBO,
        // binding 1 = shadow map (comparison-sampled). Every pipeline carries
        // it so the set binds once per frame.
        let descriptor_layout_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    p_bindings: descriptor_layout_bindings.as_ptr(),
                    binding_count: descriptor_layout_bindings.len() as u32,
                    ..Default::default()
                },
                None,
            )?
        };
        let push_constant = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo {
                    push_constant_range_count: 1,
                    p_push_constant_ranges: &push_constant,
                    set_layout_count: 1,
                    p_set_layouts: &descriptor_layout,
                    ..Default::default()
                },
                None,
            )?
        };

        let vertex_stride = std::mem::size_of::<Vertex>() as u32;
        let vertex_attribs = [
            vk::VertexInputAttributeDescription {
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
                ..Default::default()
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
                ..Default::default()
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 24,
                ..Default::default()
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                format: vk::Format::R32G32_SFLOAT,
                offset: 36,
                ..Default::default()
            },
        ];
        let vertex_binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: vertex_stride,
            input_rate: vk::VertexInputRate::VERTEX,
        };
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_module,
                p_name: c"main".as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag_module,
                p_name: c"main".as_ptr(),
                ..Default::default()
            },
        ];
        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: 2,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vk::PipelineVertexInputStateCreateInfo {
                vertex_binding_description_count: 1,
                p_vertex_binding_descriptions: &vertex_binding,
                vertex_attribute_description_count: vertex_attribs.len() as u32,
                p_vertex_attribute_descriptions: vertex_attribs.as_ptr(),
                ..Default::default()
            },
            p_input_assembly_state: &vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            },
            p_viewport_state: &vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default() // actual values set dynamically
            },
            p_rasterization_state: &vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE, // depth test handles occlusion; keep simple
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            },
            p_multisample_state: &vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            },
            p_depth_stencil_state: &vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::TRUE,
                depth_write_enable: vk::TRUE,
                depth_compare_op: vk::CompareOp::LESS,
                ..Default::default()
            },
            p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
                attachment_count: 1,
                p_attachments: &vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::RGBA,
                    ..Default::default()
                },
                ..Default::default()
            },
            p_dynamic_state: &vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
                ..Default::default()
            },
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            ..Default::default()
        };
        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?[0]
        };

        // --- Water pipeline: the world shader with alpha blending ----------
        // Same vertex/fragment stages; differences from the opaque pipeline:
        // SRC_ALPHA blending (alpha rides push pad.w), depth WRITE off (water
        // never occludes — terrain behind it must stay visible), depth TEST
        // on (water hidden behind hills). Built and created in ONE statement:
        // pipeline_info's p_* fields point at statement-local temporaries, so
        // copying the struct and creating later would read freed stack.
        let water_pipeline = unsafe {
            let water_info = vk::GraphicsPipelineCreateInfo {
                stage_count: 2,
                p_stages: stages.as_ptr(),
                p_vertex_input_state: pipeline_info.p_vertex_input_state,
                p_input_assembly_state: pipeline_info.p_input_assembly_state,
                p_viewport_state: pipeline_info.p_viewport_state,
                p_rasterization_state: pipeline_info.p_rasterization_state,
                p_multisample_state: pipeline_info.p_multisample_state,
                p_depth_stencil_state: &vk::PipelineDepthStencilStateCreateInfo {
                    depth_test_enable: vk::TRUE,
                    depth_write_enable: vk::FALSE,
                    depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                    ..Default::default()
                },
                p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
                    attachment_count: 1,
                    p_attachments: &vk::PipelineColorBlendAttachmentState {
                        blend_enable: vk::TRUE,
                        src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        color_blend_op: vk::BlendOp::ADD,
                        src_alpha_blend_factor: vk::BlendFactor::ONE,
                        dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        alpha_blend_op: vk::BlendOp::ADD,
                        color_write_mask: vk::ColorComponentFlags::RGBA,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                p_dynamic_state: pipeline_info.p_dynamic_state,
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
                ..Default::default()
            };
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[water_info], None)
                .map_err(|(_, e)| e)?[0]
        };

        // --- Line overlay pipeline (same shaders, LINE_LIST, no depth write) --
        let line_pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: 2,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vk::PipelineVertexInputStateCreateInfo {
                vertex_binding_description_count: 1,
                p_vertex_binding_descriptions: &vertex_binding,
                vertex_attribute_description_count: vertex_attribs.len() as u32,
                p_vertex_attribute_descriptions: vertex_attribs.as_ptr(),
                ..Default::default()
            },
            p_input_assembly_state: &vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::LINE_LIST,
                ..Default::default()
            },
            p_viewport_state: &vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default() // dynamic
            },
            p_rasterization_state: &vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            },
            p_multisample_state: &vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            },
            p_depth_stencil_state: &vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::TRUE,
                depth_write_enable: vk::FALSE,
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                ..Default::default()
            },
            p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
                attachment_count: 1,
                p_attachments: &vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::RGBA,
                    ..Default::default()
                },
                ..Default::default()
            },
            p_dynamic_state: &vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
                ..Default::default()
            },
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            ..Default::default()
        };
        let line_pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[line_pipeline_info], None)
                .map_err(|(_, e)| e)?[0]
        };

        // --- HUD fill pipeline: TRIANGLE_LIST + depth-test off (solid hotbar
        // fills AND its outlined borders — borders are thin triangle quads;
        // the old separate LINE_LIST HUD pipeline rendered them as traces) ---
        let hud_depth_state: vk::PipelineDepthStencilStateCreateInfo =
            vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::FALSE,
                depth_write_enable: vk::FALSE,
                ..Default::default()
            };
        let mut hud_fill_pipeline_info = line_pipeline_info;
        let hud_fill_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };
        hud_fill_pipeline_info.p_input_assembly_state = &hud_fill_assembly;
        hud_fill_pipeline_info.p_depth_stencil_state = &hud_depth_state;
        let hud_fill_pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[hud_fill_pipeline_info],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };

        // --- Shadow map resources (one image per frame in flight: a frame's
        // shadow pass may still be executing while the next frame renders, so
        // sharing one image between frames in flight would be a hazard) ------
        let shadow_format = [vk::Format::D32_SFLOAT, vk::Format::D16_UNORM]
            .into_iter()
            .find(|f| {
                let props = instance.get_physical_device_format_properties(pdevice, *f);
                props.optimal_tiling_features.contains(
                    vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::FormatFeatureFlags::SAMPLED_IMAGE,
                )
            })
            .ok_or("no depth format usable for shadow mapping")?;
        let mut shadow_images = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut shadow_memories = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut shadow_views = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image = device.create_image(
                &vk::ImageCreateInfo {
                    image_type: vk::ImageType::TYPE_2D,
                    format: shadow_format,
                    extent: vk::Extent3D {
                        width: SHADOW_MAP_SIZE,
                        height: SHADOW_MAP_SIZE,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )?;
            let reqs = device.get_image_memory_requirements(image);
            let mem_type = find_memory_type(
                reqs,
                instance.get_physical_device_memory_properties(pdevice),
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let memory = device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: reqs.size,
                    memory_type_index: mem_type,
                    ..Default::default()
                },
                None,
            )?;
            device.bind_image_memory(image, memory, 0)?;
            let view = device.create_image_view(
                &vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: shadow_format,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )?;
            shadow_images.push(image);
            shadow_memories.push(memory);
            shadow_views.push(view);
        }

        // Shadow render pass: depth-only, ends in READ_ONLY so the main pass
        // can sample the map without an explicit barrier.
        let shadow_depth_attachment = vk::AttachmentDescription {
            format: shadow_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ..Default::default()
        };
        let shadow_depth_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let shadow_subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            p_depth_stencil_attachment: &shadow_depth_ref,
            ..Default::default()
        };
        let shadow_dependencies = [
            // External → subpass: depth writes must finish before depth tests.
            vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ..Default::default()
            },
            // Subpass → external: sampling (main pass fragment shader) waits
            // for the shadow depth writes.
            vk::SubpassDependency {
                src_subpass: 0,
                dst_subpass: vk::SUBPASS_EXTERNAL,
                src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            },
        ];
        let shadow_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo {
                    attachment_count: 1,
                    p_attachments: &shadow_depth_attachment,
                    subpass_count: 1,
                    p_subpasses: &shadow_subpass,
                    dependency_count: shadow_dependencies.len() as u32,
                    p_dependencies: shadow_dependencies.as_ptr(),
                    ..Default::default()
                },
                None,
            )?
        };
        let shadow_framebuffers = shadow_views
            .iter()
            .map(|view| {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo {
                        render_pass: shadow_pass,
                        attachment_count: 1,
                        p_attachments: [view.clone()].as_ptr(),
                        width: SHADOW_MAP_SIZE,
                        height: SHADOW_MAP_SIZE,
                        layers: 1,
                        ..Default::default()
                    },
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Shadow pipeline: the same vertex layout, depth-only, no color attach.
        let shadow_vert_spv = read_spv(&mut std::io::Cursor::new(std::fs::read(
            spv_dir.join("shadow_vert.spv"),
        )?))?;
        let shadow_frag_spv = read_spv(&mut std::io::Cursor::new(std::fs::read(
            spv_dir.join("shadow_frag.spv"),
        )?))?;
        let shadow_vert_module = make_module(shadow_vert_spv)?;
        let shadow_frag_module = make_module(shadow_frag_spv)?;
        let shadow_stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: shadow_vert_module,
                p_name: c"main".as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: shadow_frag_module,
                p_name: c"main".as_ptr(),
                ..Default::default()
            },
        ];
        let shadow_pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: shadow_stages.len() as u32,
            p_stages: shadow_stages.as_ptr(),
            p_vertex_input_state: &vk::PipelineVertexInputStateCreateInfo {
                vertex_binding_description_count: 1,
                p_vertex_binding_descriptions: &vertex_binding,
                vertex_attribute_description_count: vertex_attribs.len() as u32,
                p_vertex_attribute_descriptions: vertex_attribs.as_ptr(),
                ..Default::default()
            },
            p_input_assembly_state: &vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            },
            p_viewport_state: &vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            },
            p_rasterization_state: &vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            },
            p_multisample_state: &vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            },
            p_depth_stencil_state: &vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::TRUE,
                depth_write_enable: vk::TRUE,
                depth_compare_op: vk::CompareOp::LESS,
                ..Default::default()
            },
            // No color attachments in this pass.
            p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
                attachment_count: 0,
                ..Default::default()
            },
            p_dynamic_state: &vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
                ..Default::default()
            },
            layout: pipeline_layout,
            render_pass: shadow_pass,
            subpass: 0,
            ..Default::default()
        };
        let shadow_pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[shadow_pipeline_info], None)
                .map_err(|(_, e)| e)?[0]
        };

        // Comparison sampler: LINEAR + compare gives hardware 2×2 PCF.
        let shadow_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo {
                    mag_filter: vk::Filter::LINEAR,
                    min_filter: vk::Filter::LINEAR,
                    mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                    address_mode_u: vk::SamplerAddressMode::CLAMP_TO_BORDER,
                    address_mode_v: vk::SamplerAddressMode::CLAMP_TO_BORDER,
                    address_mode_w: vk::SamplerAddressMode::CLAMP_TO_BORDER,
                    border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE, // outside = lit
                    compare_enable: vk::TRUE,
                    compare_op: vk::CompareOp::LESS,
                    ..Default::default()
                },
                None,
            )?
        };

        // --- Command pool / buffers (created early: the atlas upload below
        // needs a one-off command buffer) -------------------------------
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index: graphics_family,
                    ..Default::default()
                },
                None,
            )?
        };
        let command_buffers = unsafe {
            device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: MAX_FRAMES_IN_FLIGHT as u32,
                ..Default::default()
            })?
        };

        // --- Block texture atlas: CPU-generated 16×16 noise tiles uploaded
        // through a staging buffer, then transitioned to shader-read-only.
        // Must exist before the descriptor sets below are written.
        let (atlas_image, atlas_memory, atlas_view, atlas_sampler) =
            unsafe { create_atlas_image(&device, &instance, pdevice, &command_pool, &queue)? };

        // Descriptor pool + per-frame sets: UBO (light VP) and the shadow
        // map. The world fragment shader samples them; every pipeline shares
        // the layout, so the set binds once per frame.
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                // 2 per frame: the shadow map + the block atlas.
                descriptor_count: 2 * MAX_FRAMES_IN_FLIGHT as u32,
            },
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: MAX_FRAMES_IN_FLIGHT as u32,
                    p_pool_sizes: pool_sizes.as_ptr(),
                    pool_size_count: pool_sizes.len() as u32,
                    ..Default::default()
                },
                None,
            )?
        };
        let set_layouts = [descriptor_layout; MAX_FRAMES_IN_FLIGHT];
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                p_set_layouts: set_layouts.as_ptr(),
                descriptor_set_count: set_layouts.len() as u32,
                ..Default::default()
            })?
        };
        let mut light_vp_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for (i, set) in descriptor_sets.iter().enumerate() {
            // 64-byte host-visible UBO for this frame's light VP.
            let buf = device.create_buffer(
                &vk::BufferCreateInfo {
                    size: 64,
                    usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )?;
            let reqs = device.get_buffer_memory_requirements(buf);
            let mem_type = find_memory_type(
                reqs,
                instance.get_physical_device_memory_properties(pdevice),
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let mem = device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: reqs.size,
                    memory_type_index: mem_type,
                    ..Default::default()
                },
                None,
            )?;
            device.bind_buffer_memory(buf, mem, 0)?;
            light_vp_buffers.push((buf, mem));
            unsafe {
                device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet {
                            dst_set: *set,
                            dst_binding: 0,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                            p_buffer_info: &vk::DescriptorBufferInfo {
                                buffer: buf,
                                offset: 0,
                                range: 64,
                            },
                            ..Default::default()
                        },
                        vk::WriteDescriptorSet {
                            dst_set: *set,
                            dst_binding: 1,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            p_image_info: &vk::DescriptorImageInfo {
                                sampler: shadow_sampler,
                                image_view: shadow_views[i],
                                image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                            },
                            ..Default::default()
                        },
                        vk::WriteDescriptorSet {
                            dst_set: *set,
                            dst_binding: 2,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            p_image_info: &vk::DescriptorImageInfo {
                                sampler: atlas_sampler,
                                image_view: atlas_view,
                                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            },
                            ..Default::default()
                        },
                    ],
                    &[],
                );
            }
        }

        // --- Sky-disc pipeline: sun/moon billboards, depth-tested but not
        // depth-written (terrain occludes them; nothing occludes later).
        // Own shader pair: no lighting, no fog (fog would erase a distant
        // disc — it must sit crisply against the sky).
        let sky_vert_spv = read_spv(&mut std::io::Cursor::new(std::fs::read(
            spv_dir.join("sky_vert.spv"),
        )?))?;
        let sky_frag_spv = read_spv(&mut std::io::Cursor::new(std::fs::read(
            spv_dir.join("sky_frag.spv"),
        )?))?;
        let sky_vert_module = make_module(sky_vert_spv)?;
        let sky_frag_module = make_module(sky_frag_spv)?;
        let sky_stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: sky_vert_module,
                p_name: c"main".as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: sky_frag_module,
                p_name: c"main".as_ptr(),
                ..Default::default()
            },
        ];
        let sky_pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: sky_stages.len() as u32,
            p_stages: sky_stages.as_ptr(),
            p_vertex_input_state: &vk::PipelineVertexInputStateCreateInfo {
                vertex_binding_description_count: 1,
                p_vertex_binding_descriptions: &vertex_binding,
                vertex_attribute_description_count: vertex_attribs.len() as u32,
                p_vertex_attribute_descriptions: vertex_attribs.as_ptr(),
                ..Default::default()
            },
            p_input_assembly_state: &vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            },
            p_viewport_state: &vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            },
            p_rasterization_state: &vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            },
            p_multisample_state: &vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            },
            p_depth_stencil_state: &vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::TRUE,
                depth_write_enable: vk::FALSE,
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                ..Default::default()
            },
            p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
                attachment_count: 1,
                p_attachments: &vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::RGBA,
                    ..Default::default()
                },
                ..Default::default()
            },
            p_dynamic_state: &vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
                ..Default::default()
            },
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            ..Default::default()
        };
        let sky_pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[sky_pipeline_info], None)
                .map_err(|(_, e)| e)?[0]
        };

        // --- Sky gradient dome pipeline: same vertex layout (shares the
        // per-frame overlay buffer), own shader pair. Depth test AND write
        // OFF so it paints behind everything; drawn FIRST each frame.
        let dome_vert_spv = read_spv(&mut std::io::Cursor::new(std::fs::read(
            spv_dir.join("dome_vert.spv"),
        )?))?;
        let dome_frag_spv = read_spv(&mut std::io::Cursor::new(std::fs::read(
            spv_dir.join("dome_frag.spv"),
        )?))?;
        let dome_vert_module = make_module(dome_vert_spv)?;
        let dome_frag_module = make_module(dome_frag_spv)?;
        let dome_stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: dome_vert_module,
                p_name: c"main".as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: dome_frag_module,
                p_name: c"main".as_ptr(),
                ..Default::default()
            },
        ];
        let dome_pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: dome_stages.len() as u32,
            p_stages: dome_stages.as_ptr(),
            p_vertex_input_state: &vk::PipelineVertexInputStateCreateInfo {
                vertex_binding_description_count: 1,
                p_vertex_binding_descriptions: &vertex_binding,
                vertex_attribute_description_count: vertex_attribs.len() as u32,
                p_vertex_attribute_descriptions: vertex_attribs.as_ptr(),
                ..Default::default()
            },
            p_input_assembly_state: &vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            },
            p_viewport_state: &vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            },
            p_rasterization_state: &vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            },
            p_multisample_state: &vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            },
            p_depth_stencil_state: &vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::FALSE,
                depth_write_enable: vk::FALSE,
                ..Default::default()
            },
            p_color_blend_state: &vk::PipelineColorBlendStateCreateInfo {
                attachment_count: 1,
                p_attachments: &vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::RGBA,
                    ..Default::default()
                },
                ..Default::default()
            },
            p_dynamic_state: &vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
                ..Default::default()
            },
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            ..Default::default()
        };
        let dome_pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[dome_pipeline_info], None)
                .map_err(|(_, e)| e)?[0]
        };
        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
            device.destroy_shader_module(shadow_vert_module, None);
            device.destroy_shader_module(shadow_frag_module, None);
            device.destroy_shader_module(sky_vert_module, None);
            device.destroy_shader_module(sky_frag_module, None);
            device.destroy_shader_module(dome_vert_module, None);
            device.destroy_shader_module(dome_frag_module, None);
        }

        // --- Framebuffers -------------------------------------------------------
        let framebuffers = unsafe {
            Self::create_framebuffers(
                &device,
                render_pass,
                &image_views,
                &depth_views,
                swapchain_extent,
            )?
        };

        // (Meshes are uploaded after construction via `upload_mesh`.)

        // --- Overlay dynamic vertex buffers (host visible, one per frame) ------
        let overlay_size = OVERLAY_BUFFER_SIZE; // crosshair + highlight + hotbar ≪ 64 KiB
        let mut overlay_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let buf = device.create_buffer(
                &vk::BufferCreateInfo {
                    size: overlay_size,
                    usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )?;
            let reqs = device.get_buffer_memory_requirements(buf);
            let mem_type = find_memory_type(
                reqs,
                instance.get_physical_device_memory_properties(pdevice),
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let mem = device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: reqs.size,
                    memory_type_index: mem_type,
                    ..Default::default()
                },
                None,
            )?;
            device.bind_buffer_memory(buf, mem, 0)?;
            overlay_buffers.push((buf, mem));
        }

        // --- Sync objects (one set per frame in flight) ---------------------------
        let make_semaphore = || unsafe { device.create_semaphore(&Default::default(), None) };
        let image_available: Vec<vk::Semaphore> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| make_semaphore())
            .collect::<Result<_, _>>()?;
        let render_finished: Vec<vk::Semaphore> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| make_semaphore())
            .collect::<Result<_, _>>()?;
        let image_fences: Vec<vk::Fence> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo {
                        flags: vk::FenceCreateFlags::SIGNALED,
                        ..Default::default()
                    },
                    None,
                )
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            entry,
            instance,
            surface_loader,
            surface_khr,
            pdevice,
            graphics_family,
            device,
            queue,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_format,
            swapchain_extent,
            image_views,
            depth_format: depth_format_,
            depth_images,
            depth_memories,
            depth_views,
            render_pass,
            pipeline,
            line_pipeline,
            hud_fill_pipeline,
            water_pipeline,
            overlay_buffers,
            pipeline_layout,
            framebuffers,
            command_pool,
            command_buffers,
            atlas_image,
            atlas_memory,
            atlas_view,
            atlas_sampler,
            meshes: Vec::new(),
            water_meshes: Vec::new(),
            pending_copies: Vec::new(),
            crack_verts: Vec::new(),
            hand_verts: Vec::new(),
            billboard_verts: Vec::new(),
            settings: crate::settings::Settings::default(),
            settings_open: false,
            settings_hover: None,
            shadow_pass,
            shadow_framebuffers,
            shadow_images,
            shadow_memories,
            shadow_views,
            shadow_pipeline,
            shadow_sampler,
            light_vp_buffers,
            descriptor_pool,
            descriptor_layout,
            descriptor_sets,
            sky_pipeline,
            dome_pipeline,
            now_secs: 0.0,
            selected_slot: 0,
            slot_slots: [(Option::<crate::items::ItemType>::None, 0); NUM_SLOTS],
            slot_label: String::new(),
            backpack: [(Option::<crate::items::ItemType>::None, 0); crate::overlay::BACKPACK_SLOTS],
            inv_open: false,
            inv_hover: None,
            inv_cursor: (0.0, 0.0),
            inv_held: None,
            inv_held_count: 0,
            craft_grid: [(Option::<crate::items::ItemType>::None, 0); 9],
            craft_size: 2,
            craft_hover: None,
            craft_output_hover: false,
            debug_panel: None,
            debug_highlight: 0,
            debug_box: Vec::new(),
            image_available,
            render_finished,
            image_fences,
            current_frame: 0,
            frames_presented: 0,
            debug_messenger,
            gpu_info,
        })
    }

    fn compute_extent(capabilities: &vk::SurfaceCapabilitiesKHR, window: &Window) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX
            && capabilities.current_extent.width > 0
            && capabilities.current_extent.height > 0
        {
            return capabilities.current_extent;
        }
        let size = window.inner_size();
        vk::Extent2D {
            width: size.width.clamp(
                capabilities.min_image_extent.width.max(1),
                capabilities.max_image_extent.width,
            ),
            height: size.height.clamp(
                capabilities.min_image_extent.height.max(1),
                capabilities.max_image_extent.height,
            ),
        }
    }

    unsafe fn create_image_view(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
    ) -> Result<vk::ImageView, Box<dyn Error>> {
        unsafe {
            Ok(device.create_image_view(
                &vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )?)
        }
    }

    unsafe fn create_depth_image(
        instance: &ash::Instance,
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), Box<dyn Error>> {
        unsafe {
            let image = device.create_image(
                &vk::ImageCreateInfo {
                    image_type: vk::ImageType::TYPE_2D,
                    format,
                    extent: vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )?;
            let reqs = device.get_image_memory_requirements(image);
            let mem_type = find_memory_type(
                reqs,
                instance.get_physical_device_memory_properties(pdevice),
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let memory = device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: reqs.size,
                    memory_type_index: mem_type,
                    ..Default::default()
                },
                None,
            )?;
            device.bind_image_memory(image, memory, 0)?;
            let view = device.create_image_view(
                &vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )?;
            Ok((image, memory, view))
        }
    }

    unsafe fn create_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        image_views: &[vk::ImageView],
        depth_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Result<Vec<vk::Framebuffer>, Box<dyn Error>> {
        if image_views.len() != depth_views.len() {
            return Err("swapchain color/depth attachment counts differ".into());
        }
        let mut framebuffers = Vec::with_capacity(image_views.len());
        for (&view, &depth_view) in image_views.iter().zip(depth_views) {
            let attachments = [view, depth_view];
            framebuffers.push(unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo {
                        render_pass,
                        attachment_count: attachments.len() as u32,
                        p_attachments: attachments.as_ptr(),
                        width: extent.width,
                        height: extent.height,
                        layers: 1,
                        ..Default::default()
                    },
                    None,
                )?
            });
        }
        Ok(framebuffers)
    }

    unsafe fn create_swapchain(
        surface_loader: &surface::Instance,
        swapchain_loader: &swapchain::Device,
        pdevice: vk::PhysicalDevice,
        surface_khr: vk::SurfaceKHR,
        window: &Window,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<vk::SwapchainKHR, Box<dyn Error>> {
        unsafe {
            let capabilities =
                surface_loader.get_physical_device_surface_capabilities(pdevice, surface_khr)?;
            let formats =
                surface_loader.get_physical_device_surface_formats(pdevice, surface_khr)?;
            let (format, color_space) = formats
                .iter()
                .find(|f| {
                    f.format == vk::Format::B8G8R8A8_SRGB || f.format == vk::Format::R8G8B8A8_SRGB
                })
                .map(|f| (f.format, f.color_space))
                .unwrap_or((formats[0].format, formats[0].color_space));

            let extent = Self::compute_extent(&capabilities, window);
            // max_image_count == 0 means NO LIMIT per spec — never clamp to 1!
            // Aim for triple buffering; a single-image swapchain deadlocks FIFO present.
            let mut image_count = capabilities.min_image_count.max(3);
            if capabilities.max_image_count != 0 {
                image_count = image_count.min(capabilities.max_image_count);
            }

            let sc_info = vk::SwapchainCreateInfoKHR {
                surface: surface_khr,
                min_image_count: image_count,
                image_format: format,
                image_color_space: color_space,
                image_extent: extent,
                image_array_layers: 1,
                image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform: capabilities.current_transform,
                composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
                present_mode: vk::PresentModeKHR::FIFO, // vsync, guaranteed support
                clipped: vk::TRUE,
                old_swapchain,
                ..Default::default()
            };
            Ok(swapchain_loader.create_swapchain(&sc_info, None)?)
        }
    }

    /// Recreate everything that depends on the swapchain size.
    pub unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
        unsafe {
            self.device.device_wait_idle()?;

            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            for &v in &self.image_views {
                self.device.destroy_image_view(v, None);
            }
            for view in self.depth_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            for (image, memory) in self
                .depth_images
                .drain(..)
                .zip(self.depth_memories.drain(..))
            {
                self.device.destroy_image(image, None);
                self.device.free_memory(memory, None);
            }
            let old = self.swapchain;

            let new_swapchain = Self::create_swapchain(
                &self.surface_loader,
                &self.swapchain_loader,
                self.pdevice,
                self.surface_khr,
                window,
                old,
            )?;
            self.swapchain = new_swapchain;
            self.swapchain_images = self.swapchain_loader.get_swapchain_images(new_swapchain)?;
            let formats = self
                .surface_loader
                .get_physical_device_surface_formats(self.pdevice, self.surface_khr)?;
            self.swapchain_format = formats[0].format;
            self.swapchain_extent = Self::compute_extent(
                &self
                    .surface_loader
                    .get_physical_device_surface_capabilities(self.pdevice, self.surface_khr)?,
                window,
            );
            self.image_views = self
                .swapchain_images
                .iter()
                .map(|img| Self::create_image_view(&self.device, *img, self.swapchain_format))
                .collect::<Result<Vec<_>, _>>()?;
            self.depth_images = Vec::with_capacity(self.swapchain_images.len());
            self.depth_memories = Vec::with_capacity(self.swapchain_images.len());
            self.depth_views = Vec::with_capacity(self.swapchain_images.len());
            for _ in &self.swapchain_images {
                let (image, memory, view) = Self::create_depth_image(
                    &self.instance,
                    &self.device,
                    self.pdevice,
                    self.depth_format,
                    self.swapchain_extent,
                )?;
                self.depth_images.push(image);
                self.depth_memories.push(memory);
                self.depth_views.push(view);
            }
            self.framebuffers = Self::create_framebuffers(
                &self.device,
                self.render_pass,
                &self.image_views,
                &self.depth_views,
                self.swapchain_extent,
            )?;

            // Frame-based sync objects and command buffers are independent of the
            // swapchain image count — nothing to recreate there.

            // Destroy the old swapchain last (it's only retired via old_swapchain)
            self.swapchain_loader.destroy_swapchain(old, None);
        }
        Ok(())
    }

    /// Upload a mesh (device-local via staging) registered under `key`.
    /// If a mesh with this key already exists, it is destroyed first — this is
    /// the chunk-replace path used when a chunk is edited.
    pub unsafe fn upload_mesh(
        &mut self,
        key: MeshKey,
        vertices: &[Vertex],
        indices: &[u16],
        model_offset: [f32; 3],
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            self.destroy_mesh(key);
            let (vertex_buffer, vertex_memory) = create_device_buffer(
                &self.device,
                &self.instance,
                self.pdevice,
                &self.command_pool,
                &self.queue,
                bytemuck::cast_slice(vertices),
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;
            let (index_buffer, index_memory) = create_device_buffer(
                &self.device,
                &self.instance,
                self.pdevice,
                &self.command_pool,
                &self.queue,
                bytemuck::cast_slice(indices),
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?;
            self.meshes.push((
                key,
                Mesh {
                    vertex_buffer,
                    vertex_memory,
                    index_buffer,
                    index_memory,
                    index_count: indices.len() as u32,
                    model_offset,
                },
            ));
        }
        Ok(())
    }

    /// Destroy a single mesh by key, if present.
    pub unsafe fn destroy_mesh(&mut self, key: MeshKey) {
        unsafe {
            if let Some(pos) = self.meshes.iter().position(|(k, _)| *k == key) {
                // Keep the mesh registered and alive if the queue cannot be
                // made idle; submitted command buffers may still reference it.
                if let Err(error) = self.device.device_wait_idle() {
                    log::error!("waiting for GPU before mesh destruction failed: {error}");
                    return;
                }
                let (_, m) = self.meshes.swap_remove(pos);
                self.device.destroy_buffer(m.vertex_buffer, None);
                self.device.free_memory(m.vertex_memory, None);
                self.device.destroy_buffer(m.index_buffer, None);
                self.device.free_memory(m.index_memory, None);
            }
        }
    }

    /// Upload/replace a chunk's translucent water mesh. Empty geometry
    /// destroys any existing water mesh for the key.
    pub unsafe fn upload_water_mesh(
        &mut self,
        key: MeshKey,
        vertices: &[Vertex],
        indices: &[u16],
        model_offset: [f32; 3],
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            if vertices.is_empty() || indices.is_empty() {
                self.destroy_water_mesh(key);
                return Ok(());
            }
            self.destroy_water_mesh(key);
            let (vertex_buffer, vertex_memory) = create_device_buffer(
                &self.device,
                &self.instance,
                self.pdevice,
                &self.command_pool,
                &self.queue,
                bytemuck::cast_slice(vertices),
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;
            let (index_buffer, index_memory) = create_device_buffer(
                &self.device,
                &self.instance,
                self.pdevice,
                &self.command_pool,
                &self.queue,
                bytemuck::cast_slice(indices),
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?;
            self.water_meshes.push((
                key,
                Mesh {
                    vertex_buffer,
                    vertex_memory,
                    index_buffer,
                    index_memory,
                    index_count: indices.len() as u32,
                    model_offset,
                },
            ));
            Ok(())
        }
    }

    /// Destroy a chunk's water mesh, if present.
    pub unsafe fn destroy_water_mesh(&mut self, key: MeshKey) {
        unsafe {
            if let Some(pos) = self.water_meshes.iter().position(|(k, _)| *k == key) {
                // A wait failure must not turn into freeing a buffer that may
                // still be referenced by an in-flight water draw.
                if let Err(error) = self.device.device_wait_idle() {
                    log::error!("waiting for GPU before water mesh destruction failed: {error}");
                    return;
                }
                let (_, m) = self.water_meshes.swap_remove(pos);
                self.device.destroy_buffer(m.vertex_buffer, None);
                self.device.free_memory(m.vertex_memory, None);
                self.device.destroy_buffer(m.index_buffer, None);
                self.device.free_memory(m.index_memory, None);
            }
        }
    }

    #[allow(dead_code)] // used by future milestones (chunk eviction stats)
    /// GPU facts for the debug overlay.
    pub fn gpu_info(&self) -> &GpuInfo {
        &self.gpu_info
    }

    /// Triangle count across all uploaded chunk meshes.
    pub fn total_triangles(&self) -> u64 {
        self.meshes
            .iter()
            .map(|(_, m)| m.index_count as u64 / 3)
            .sum()
    }

    /// Allocated swapchain image count.
    pub fn swapchain_image_count(&self) -> usize {
        self.swapchain_images.len()
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Destroy and forget all uploaded meshes (used when rebuilding terrain).
    pub unsafe fn clear_meshes(&mut self) {
        unsafe {
            for (_k, m) in self.meshes.drain(..) {
                self.device.destroy_buffer(m.vertex_buffer, None);
                self.device.free_memory(m.vertex_memory, None);
                self.device.destroy_buffer(m.index_buffer, None);
                self.device.free_memory(m.index_memory, None);
            }
        }
    }

    /// Number of frames that completed presentation successfully.
    pub fn frames_presented(&self) -> u64 {
        self.frames_presented
    }

    /// Update HUD state (selected hotbar slot + per-slot counts + the
    /// selected-item label shown above the bar, if any) for the next frame's
    /// hotbar rendering.
    pub fn set_hud_state(
        &mut self,
        selected: usize,
        slots: &[(Option<crate::items::ItemType>, u32); NUM_SLOTS],
        label: &str,
    ) {
        self.selected_slot = selected;
        self.slot_slots = *slots;
        self.slot_label.clear();
        self.slot_label.push_str(label);
    }

    /// Graphics settings + settings-page UI state for the next frame.
    pub fn set_settings_state(
        &mut self,
        settings: crate::settings::Settings,
        open: bool,
        hover: Option<usize>,
    ) {
        self.settings = settings;
        self.settings_open = open;
        self.settings_hover = hover;
    }

    /// E inventory panel state: backpack contents, open flag, hovered slot,
    /// and the cursor position (vertical-NDC units, y down).
    pub fn set_inventory_state(
        &mut self,
        backpack: &[(Option<crate::items::ItemType>, u32); crate::overlay::BACKPACK_SLOTS],
        open: bool,
        hover: Option<usize>,
        cursor: (f32, f32),
        held: Option<crate::items::ItemType>,
        held_count: u32,
        craft_grid: [(Option<crate::items::ItemType>, u32); 9],
        craft_size: usize,
        craft_hover: Option<usize>,
        craft_output_hover: bool,
    ) {
        self.backpack = *backpack;
        self.inv_open = open;
        self.inv_hover = hover;
        self.inv_cursor = cursor;
        self.inv_held = held;
        self.inv_held_count = held_count;
        self.craft_grid = craft_grid;
        self.craft_size = craft_size.clamp(2, 3);
        self.craft_hover = craft_hover;
        self.craft_output_hover = craft_output_hover;
    }

    /// Mining crack overlay + item drops + break particles for this frame.
    /// Empty vecs = nothing to draw (the drops/particles ride in the same
    /// billboard section, drawn after terrain with depth test, no write).
    pub fn set_billboards(
        &mut self,
        crack_lines: Vec<crate::geometry::Vertex>,
        billboard_verts: Vec<crate::geometry::Vertex>,
    ) {
        self.crack_verts = crack_lines;
        self.billboard_verts = billboard_verts;
    }

    /// First-person viewmodel geometry (hand + held item) for this frame.
    pub fn set_hand(&mut self, mut verts: Vec<crate::geometry::Vertex>) {
        // Hard clamp: the hand slice is 8192 verts (≈3× worst case); a
        // runaway sprite would otherwise corrupt the dome slice that follows.
        const MAX: usize = OVERLAY_HAND_VERTEX_COUNT as usize;
        if verts.len() > MAX {
            log::warn!(
                "hand geometry {} verts exceeded slice {} — truncating",
                verts.len(),
                MAX
            );
            verts.truncate(MAX);
        }
        self.hand_verts = verts;
    }

    /// Per-frame debug overlay geometry (F3): set by the app before draw.
    /// `None` = debug overlay hidden.
    pub fn set_debug_data(
        &mut self,
        panel_lines: Option<Vec<String>>,
        highlight_count: usize,
        box_lines: Vec<crate::geometry::Vertex>,
    ) {
        self.debug_panel = panel_lines;
        self.debug_highlight = highlight_count;
        self.debug_box = box_lines;
    }

    pub unsafe fn draw(
        &mut self,
        window: &Window,
        camera: &Camera,
        highlight: Option<&RayHit>,
        sky: &crate::sky::SkyState,
        shadow_light: [f32; 3],
        disc_verts: &[Vertex],
        now_secs: f32,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            let size = window.inner_size();
            if size.width == 0 || size.height == 0 {
                return Ok(()); // minimized
            }

            let frame = self.current_frame;
            self.now_secs = now_secs;

            // Wait until this frame's previous work finished (timeout so a problem
            // surfaces as a log line instead of a frozen window).
            match self
                .device
                .wait_for_fences(&[self.image_fences[frame]], true, 2_000_000_000)
            {
                Ok(_) => {}
                Err(vk::Result::TIMEOUT) => {
                    log::error!("frame {frame}: GPU fence timed out — likely GPU hang");
                    return Err("GPU fence timeout".into());
                }
                Err(e) => return Err(e.into()),
            }
            // Bounded acquire: on gamescope/Wayland the compositor can hold buffers
            // briefly; blocking forever here also starves winit's event dispatch.
            // On timeout we return to the event loop and retry on the next redraw.
            let (image_index, _suboptimal) = match self.swapchain_loader.acquire_next_image(
                self.swapchain,
                250_000_000, // 250 ms
                self.image_available[frame],
                vk::Fence::null(),
            ) {
                Ok(r) => r,
                Err(vk::Result::TIMEOUT) => {
                    return Ok(());
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    // Fence is still SIGNALLED here — reset it only after a
                    // successful acquire, otherwise this frame deadlocks forever.
                    log::info!("acquire OUT_OF_DATE — recreating swapchain");
                    self.recreate_swapchain(window)?;
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
            };
            let image_index = image_index as usize;

            // Reset AFTER a successful acquire (see comment above).
            self.device.reset_fences(&[self.image_fences[frame]])?;

            let aspect =
                self.swapchain_extent.width as f32 / self.swapchain_extent.height.max(1) as f32;
            // Far plane scales with the render distance so distant chunks are
            // never clipped by the projection.
            let far = crate::settings::far_plane(self.settings.render_distance);
            let vp = view_proj_far(camera, aspect, far);
            // Frustum planes for per-chunk culling: chunks fully outside the
            // view are skipped in BOTH the shadow pass and the main pass. At
            // 32 chunks the frustum covers roughly a quarter of the loaded
            // footprint (70° hfov over a 68-chunk square), so this drops
            // most draw calls AND most shadow-pass geometry.
            let planes = crate::camera::frustum_planes(&vp);
            // Chunk vertex positions are already in world space (mesh_chunk
            // emits wx/wy/wz directly), so model_offset is not the chunk
            // origin. Use the mesh key for frustum bounds; using model_offset
            // here treated every chunk as if it occupied the origin and made
            // terrain pop in/out as the camera moved.
            let visible = |key: MeshKey, _mesh: &Mesh| -> bool {
                let min = Vec3::new(
                    key.0 as f32 * crate::terrain::CHUNK_SIZE as f32,
                    0.0,
                    key.1 as f32 * crate::terrain::CHUNK_SIZE as f32,
                );
                let max = min
                    + Vec3::new(
                        crate::terrain::CHUNK_SIZE as f32,
                        crate::terrain::MAX_Y as f32,
                        crate::terrain::CHUNK_SIZE as f32,
                    );
                !crate::camera::aabb_outside_frustum(&planes, min, max)
            };

            // --- Light view-projection for the shadow map ------------------
            // Orthographic box centered on the camera's column, looking along
            // the (elevation-clamped) shadow light. glam's orthographic_rh
            // maps [near, far] to [0, 1] depth — exactly what the comparison
            // sampler wants. Centering the frustum on the camera and snapping
            // its X/Y origin to the shadow-texel grid keeps shimmering down
            // as the player moves.
            let ld = Vec3::from(shadow_light);
            // Basis of the light view (mirrors glam's look_to_rh): texels
            // tile along (s, u), so snapping those coordinates to the texel
            // grid removes sub-texel swimming as the player moves.
            let l_f = (-ld).normalize();
            let l_s = l_f.cross(Vec3::Y).normalize();
            let l_u = l_s.cross(l_f);
            let texel = SHADOW_FRUSTUM_HALF * 2.0 / SHADOW_MAP_SIZE as f32;
            let cam = camera.pos;
            let sc = Vec3::new(cam.dot(l_s), cam.dot(l_u), 0.0);
            let snapped = Vec3::new(
                (sc.x / texel).floor() * texel,
                (sc.y / texel).floor() * texel,
                0.0,
            );
            // Reconstruct the snapped world-space center rather than merely
            // calculating the rounded coordinates. The previous code computed
            // `snapped` but still built the light camera from `cam`, so the
            // shadow map continued to swim by sub-texel amounts as the player
            // walked.
            let snapped_center = cam + l_s * (snapped.x - sc.x) + l_u * (snapped.y - sc.y);
            // Put the light camera behind the snapped receiver center,
            // looking toward the world along the incoming light direction.
            let light_pos = snapped_center - ld * SHADOW_LIGHT_DISTANCE;
            let light_view_snapped = Mat4::look_to_rh(light_pos, ld, l_u);
            let light_proj = Mat4::orthographic_rh(
                -SHADOW_FRUSTUM_HALF,
                SHADOW_FRUSTUM_HALF,
                -SHADOW_FRUSTUM_HALF,
                SHADOW_FRUSTUM_HALF,
                SHADOW_NEAR,
                SHADOW_FAR,
            );
            let light_vp = light_proj * light_view_snapped;

            let cmd = self.command_buffers[frame];
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::RELEASE_RESOURCES)?;
            self.device.begin_command_buffer(cmd, &Default::default())?;

            // The sky IS the clear color (no sky dome yet): use the current
            // fog color so terrain fades into exactly the sky behind it.
            let clear_color = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [sky.fog[0], sky.fog[1], sky.fog[2], 1.0],
                },
            };
            let clear_depth = vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            };

            // --- Shadow pass: terrain depth from the light's point of view --
            // Runs before the main pass in the same command buffer; the render
            // pass dependency makes the main pass's shadow sampling wait for
            // these depth writes.
            let shadow_push = ShadowPush {
                light_vp,
                params: [
                    SHADOW_FRUSTUM_HALF * 2.0 / SHADOW_MAP_SIZE as f32,
                    0.0,
                    0.0,
                    0.0,
                ],
            };
            self.device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo {
                    render_pass: self.shadow_pass,
                    framebuffer: self.shadow_framebuffers[frame],
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: vk::Extent2D {
                            width: SHADOW_MAP_SIZE,
                            height: SHADOW_MAP_SIZE,
                        },
                    },
                    clear_value_count: 1,
                    p_clear_values: [vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    }]
                    .as_ptr(),
                    ..Default::default()
                },
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.shadow_pipeline,
            );
            let shadow_viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: SHADOW_MAP_SIZE as f32,
                height: SHADOW_MAP_SIZE as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device.cmd_set_viewport(cmd, 0, &[shadow_viewport]);
            self.device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D {
                        width: SHADOW_MAP_SIZE,
                        height: SHADOW_MAP_SIZE,
                    },
                }],
            );
            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&shadow_push),
            );
            // The pass itself always runs (cheap clear) so the shadow image's
            // layout transitions stay valid for the descriptor; when shadows
            // are toggled off we skip the expensive mesh draws — the cleared
            // map (all far) shadows nothing, and the world shader also skips
            // sampling entirely via its fog_params.z gate.
            if self.settings.shadows {
                for (key, mesh) in &self.meshes {
                    if !visible(*key, mesh) {
                        continue;
                    }
                    let model = Mat4::from_translation(Vec3::from(mesh.model_offset));
                    self.device
                        .cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertex_buffer], &[0]);
                    self.device.cmd_bind_index_buffer(
                        cmd,
                        mesh.index_buffer,
                        0,
                        vk::IndexType::UINT16,
                    );
                    let mesh_push = ShadowPush {
                        light_vp: light_vp * model,
                        params: shadow_push.params,
                    };
                    self.device.cmd_push_constants(
                        cmd,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&mesh_push),
                    );
                    self.device
                        .cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 1);
                }
            }
            self.device.cmd_end_render_pass(cmd);
            self.device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo {
                    render_pass: self.render_pass,
                    framebuffer: self.framebuffers[image_index as usize],
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: self.swapchain_extent,
                    },
                    clear_value_count: 2,
                    p_clear_values: [clear_color, clear_depth].as_ptr(),
                    ..Default::default()
                },
                vk::SubpassContents::INLINE,
            );

            // Pack the frame's lighting once. World geometry gets the real
            // sun+ambient; the HUD gets ambient=1, sun=0 so the shared
            // fragment shader passes its vertex colors through unlit (HUD
            // normals are all up → faceLight 1.0 → light = 1.0).
            let mut push_world = PushConstants {
                mvp: Mat4::IDENTITY,
                // pad.w is the fragment ALPHA: 1.0 everywhere except the
                // water pass, which overwrites it with the blend factor.
                pad: [0.0, 0.0, 0.0, 1.0],
                sun_dir: [sky.sun_dir[0], sky.sun_dir[1], sky.sun_dir[2], 0.0],
                sky_params: [
                    sky.sun_intensity,
                    sky.ambient,
                    SHADOW_STRENGTH,
                    SHADOW_FRUSTUM_HALF * 2.0 / SHADOW_MAP_SIZE as f32,
                ],
                fog_params: [
                    far * 0.4,
                    far * 0.95,
                    self.settings.shadows as u32 as f32,
                    0.0,
                ],
            };
            // Upload this frame's light VP for the shader's shadow projection.
            {
                let (_buf, mem) = self.light_vp_buffers[frame];
                let mapped = self
                    .device
                    .map_memory(mem, 0, 64, vk::MemoryMapFlags::empty())?;
                std::ptr::copy_nonoverlapping(
                    light_vp.as_ref() as *const f32 as *const u8,
                    mapped as *mut u8,
                    64,
                );
                self.device.unmap_memory(mem);
            }
            let push_hud = PushConstants {
                mvp: Mat4::IDENTITY,
                pad: [0.0, 0.0, 0.0, 1.0], // opaque
                sun_dir: [0.0, 1.0, 0.0, 0.0],
                sky_params: [0.0, 1.0, 0.0, 0.0],
                // Fog disabled with a sane denominator — fog_params = 0/0 made
                // the fog term NaN and the whole HUD rendered undefined-white.
                fog_params: [1.0, 0.0, 0.0, 0.0],
            }; // The clear color stays the fog color (dome rim, distant fog,
               // and sky all agree); the dome paints the gradient over it.

            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.swapchain_extent.width as f32,
                height: self.swapchain_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swapchain_extent,
            };
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);
            // Bind the shared descriptor set (light VP + shadow map) once for
            // every pipeline in this frame — they all carry the same layout.
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[frame]],
                &[],
            );
            // Per-frame overlay buffer (dome, crosshair, HUD, discs, clouds…).
            let (overlay_buf, overlay_mem) = self.overlay_buffers[frame];

            // --- Sky gradient dome FIRST: depth test/write off, so it fills
            // the clear color with the zenith→horizon gradient; terrain,
            // discs, clouds, and the hand all paint over it afterward.
            {
                let dome_verts = crate::overlay::build_sky_dome(
                    camera.pos, sky, far, // dome radius = fog end, so the rim fades into fog
                );
                let bytes: &[u8] = bytemuck::cast_slice(&dome_verts);
                debug_assert!(
                    OVERLAY_DOME_OFFSET + bytes.len() as u64 <= OVERLAY_BUFFER_SIZE,
                    "dome slice exceeded its buffer slice"
                );
                let mapped = self.device.map_memory(
                    overlay_mem,
                    OVERLAY_DOME_OFFSET,
                    bytes.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
                self.device.unmap_memory(overlay_mem);

                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.dome_pipeline,
                );
                self.device
                    .cmd_bind_vertex_buffers(cmd, 0, &[overlay_buf], &[0]);
                // The dome shaders read pad.xyz as the camera position.
                push_world.mvp = vp;
                push_world.pad = [camera.pos.x, camera.pos.y, camera.pos.z, 0.0];
                self.device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_world),
                );
                self.device.cmd_draw(
                    cmd,
                    dome_verts.len() as u32,
                    1,
                    OVERLAY_DOME_VERTEX_START,
                    0,
                );
                push_world.pad = [0.0, 0.0, 0.0, 1.0];
            }

            // The dome draw changes the bound pipeline. Rebind the world
            // pipeline before terrain; otherwise the terrain is interpreted by
            // the dome shader and disappears into the blue clear color.
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[frame]],
                &[],
            );

            for (key, mesh) in &self.meshes {
                if !visible(*key, mesh) {
                    continue;
                }
                let model = Mat4::from_translation(Vec3::from(mesh.model_offset));
                let mvp = vp * model;
                push_world.mvp = mvp;
                self.device
                    .cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertex_buffer], &[0]);
                self.device
                    .cmd_bind_index_buffer(cmd, mesh.index_buffer, 0, vk::IndexType::UINT16);
                self.device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_world),
                );
                self.device
                    .cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 1);
            }

            // --- Translucent water: after ALL opaque geometry (back-to-front
            // per-chunk far→near keeps blending right without a depth sort).
            // Alpha rides pad.w — the opaque draws leave it at 0.0, which is
            // invisible because their blend is disabled.
            let water_alpha = if camera.pos.y < crate::world::SEA_LEVEL as f32 {
                0.55 // underwater: stronger tint sells the submersion
            } else {
                0.62
            };
            push_world.pad = [0.0, 0.0, 0.0, water_alpha];
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.water_pipeline,
            );
            let mut water_meshes: Vec<(MeshKey, &Mesh)> = self
                .water_meshes
                .iter()
                .map(|(key, mesh)| (*key, mesh))
                .collect();
            // Far → near for correct alpha blending.
            water_meshes.sort_by(|(ka, _), (kb, _)| {
                let da = (camera.pos.x - (ka.0 * crate::terrain::CHUNK_SIZE) as f32).abs()
                    + (camera.pos.z - (ka.1 * crate::terrain::CHUNK_SIZE) as f32).abs();
                let db = (camera.pos.x - (kb.0 * crate::terrain::CHUNK_SIZE) as f32).abs()
                    + (camera.pos.z - (kb.1 * crate::terrain::CHUNK_SIZE) as f32).abs();
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (key, mesh) in water_meshes {
                if !visible(key, mesh) {
                    continue;
                }
                let model = Mat4::from_translation(Vec3::from(mesh.model_offset));
                push_world.mvp = vp * model;
                self.device
                    .cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertex_buffer], &[0]);
                self.device
                    .cmd_bind_index_buffer(cmd, mesh.index_buffer, 0, vk::IndexType::UINT16);
                self.device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_world),
                );
                self.device
                    .cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 1);
            }
            push_world.pad = [0.0, 0.0, 0.0, 1.0];
            // Back to the opaque world pipeline for the overlays that follow
            // (highlight, drops, hand).
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            // --- Screen-space crosshair: four thin quads pinned to the exact
            // screen center, through the depth-test-off HUD fill pipeline.
            {
                let crosshair = crate::overlay::build_crosshair(aspect);
                let bytes: &[u8] = bytemuck::cast_slice(&crosshair);
                let mapped = self.device.map_memory(
                    overlay_mem,
                    OVERLAY_CROSSHAIR_OFFSET,
                    bytes.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
                self.device.unmap_memory(overlay_mem);

                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.hud_fill_pipeline,
                );
                self.device
                    .cmd_bind_vertex_buffers(cmd, 0, &[overlay_buf], &[0]);
                self.device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_hud),
                );
                self.device.cmd_draw(
                    cmd,
                    crosshair.len() as u32,
                    1,
                    OVERLAY_CROSSHAIR_VERTEX_START,
                    0,
                );
            }

            // --- Sun/moon discs: world-space billboards, drawn after terrain
            // (depth-tested so hills hide them, no depth write so nothing
            // draws over them later except the HUD).
            if !disc_verts.is_empty() {
                let bytes: &[u8] = bytemuck::cast_slice(disc_verts);
                debug_assert!(
                    OVERLAY_DISC_VERTEX_START as u64 * VERTEX_STRIDE as u64 + bytes.len() as u64
                        <= OVERLAY_DEBUG_OFFSET,
                    "disc slice exceeded its buffer slice"
                );
                let mapped = self.device.map_memory(
                    overlay_mem,
                    OVERLAY_DISC_VERTEX_START as u64 * VERTEX_STRIDE as u64,
                    bytes.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
                self.device.unmap_memory(overlay_mem);

                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.sky_pipeline,
                );
                self.device
                    .cmd_bind_vertex_buffers(cmd, 0, &[overlay_buf], &[0]);
                push_world.mvp = vp; // world-space quad, camera transform
                self.device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_world),
                );
                self.device.cmd_draw(
                    cmd,
                    disc_verts.len() as u32,
                    1,
                    OVERLAY_DISC_VERTEX_START,
                    0,
                );
            }

            // --- Drifting clouds: static-filled world-space quads through
            // the same depth-tested sky pipeline (terrain hides them; they
            // hide nothing). Skipped when the cloud setting is off.
            if self.settings.clouds {
                let cloud_verts = crate::overlay::build_clouds(
                    camera.pos,
                    sky,
                    self.now_secs,
                    self.settings.cloud_distance,
                );
                if !cloud_verts.is_empty() {
                    let bytes: &[u8] = bytemuck::cast_slice(&cloud_verts);
                    debug_assert!(
                        OVERLAY_CLOUD_OFFSET + bytes.len() as u64 <= OVERLAY_BILLBOARD_OFFSET,
                        "cloud slice exceeded its buffer slice"
                    );
                    let mapped = self.device.map_memory(
                        overlay_mem,
                        OVERLAY_CLOUD_OFFSET,
                        bytes.len() as u64,
                        vk::MemoryMapFlags::empty(),
                    )?;
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
                    self.device.unmap_memory(overlay_mem);

                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.sky_pipeline,
                    );
                    self.device
                        .cmd_bind_vertex_buffers(cmd, 0, &[overlay_buf], &[0]);
                    push_world.mvp = vp;
                    self.device.cmd_push_constants(
                        cmd,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push_world),
                    );
                    self.device.cmd_draw(
                        cmd,
                        cloud_verts.len() as u32,
                        1,
                        OVERLAY_CLOUD_VERTEX_START,
                        0,
                    );
                }
            }

            // --- Billboards: item drops + break particles, via the sky
            // pipeline (depth-tested, no depth write) from their own slice.
            if !self.billboard_verts.is_empty() {
                let bytes: &[u8] = bytemuck::cast_slice(&self.billboard_verts);
                debug_assert!(
                    OVERLAY_BILLBOARD_OFFSET + bytes.len() as u64 <= OVERLAY_DEBUG_OFFSET,
                    "billboard slice exceeded its buffer slice"
                );
                let mapped = self.device.map_memory(
                    overlay_mem,
                    OVERLAY_BILLBOARD_OFFSET,
                    bytes.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
                self.device.unmap_memory(overlay_mem);

                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.sky_pipeline,
                );
                self.device
                    .cmd_bind_vertex_buffers(cmd, 0, &[overlay_buf], &[0]);
                push_world.mvp = vp;
                self.device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_world),
                );
                self.device.cmd_draw(
                    cmd,
                    self.billboard_verts.len() as u32,
                    1,
                    OVERLAY_BILLBOARD_VERTEX_START,
                    0,
                );
            }

            // --- Mining cracks: filled pixel cells, same depth-tested sky
            // pipeline as the billboards, from their own slice.
            if !self.crack_verts.is_empty() {
                let bytes: &[u8] = bytemuck::cast_slice(&self.crack_verts);
                debug_assert!(
                    OVERLAY_CRACK_OFFSET + bytes.len() as u64 <= OVERLAY_DEBUG_OFFSET,
                    "crack slice exceeded its buffer slice"
                );
                let mapped = self.device.map_memory(
                    overlay_mem,
                    OVERLAY_CRACK_OFFSET,
                    bytes.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
                self.device.unmap_memory(overlay_mem);

                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.sky_pipeline,
                );
                self.device
                    .cmd_bind_vertex_buffers(cmd, 0, &[overlay_buf], &[0]);
                push_world.mvp = vp;
                self.device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_world),
                );
                self.device.cmd_draw(
                    cmd,
                    self.crack_verts.len() as u32,
                    1,
                    OVERLAY_CRACK_VERTEX_START,
                    0,
                );
            }

            // --- First-person hand: held block/tool + arm, world-space
            // geometry riding the camera. Drawn through the WORLD pipeline so
            // it gets the atlas (held blocks are textured), real lighting,
            // and proper depth occlusion against terrain.
            if !self.hand_verts.is_empty() {
                let bytes: &[u8] = bytemuck::cast_slice(&self.hand_verts);
                debug_assert!(
                    OVERLAY_HAND_OFFSET + bytes.len() as u64
                        <= OVERLAY_HAND_OFFSET
                            + OVERLAY_HAND_VERTEX_COUNT as u64 * VERTEX_STRIDE as u64,
                    "hand slice exceeded its buffer slice"
                );
                let mapped = self.device.map_memory(
                    overlay_mem,
                    OVERLAY_HAND_OFFSET,
                    bytes.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
                self.device.unmap_memory(overlay_mem);

                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline, // the world pipeline (atlas + lighting)
                );
                self.device
                    .cmd_bind_vertex_buffers(cmd, 0, &[overlay_buf], &[0]);
                push_world.mvp = vp;
                self.device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_world),
                );
                self.device.cmd_draw(
                    cmd,
                    self.hand_verts.len() as u32,
                    1,
                    OVERLAY_HAND_VERTEX_START,
                    0,
                );
            }

            // --- World overlay: targeted-block highlight + F3 boxes ----------
            // The highlight is built from crossed thin QUADS (triangles) so it
            // reads bold from any angle — it draws through the sky pipeline
            // (TRIANGLE_LIST, depth-tested, no write). The F3 boxes are true
            // 2-vertex line SEGMENTS and draw through the line pipeline. Both
            // ride in one upload at the start of the per-frame buffer; each
            // draw starts at its own vertex offset.
            let highlight_verts = build_overlay(highlight);
            let line_verts: Vec<Vertex> = self.debug_box.iter().copied().collect();
            if !highlight_verts.is_empty() || !line_verts.is_empty() {
                let mut world_verts = highlight_verts.clone();
                let highlight_count = world_verts.len() as u32;
                world_verts.extend(line_verts.iter().copied());
                let bytes: &[u8] = bytemuck::cast_slice(&world_verts);
                debug_assert!(
                    OVERLAY_WORLD_OFFSET + bytes.len() as u64 <= OVERLAY_BUFFER_SIZE,
                    "world overlay exceeded its buffer slice"
                );
                let mapped = self.device.map_memory(
                    overlay_mem,
                    OVERLAY_WORLD_OFFSET,
                    bytes.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
                self.device.unmap_memory(overlay_mem);

                let world_start = OVERLAY_WORLD_OFFSET as u32 / VERTEX_STRIDE;
                self.device
                    .cmd_bind_vertex_buffers(cmd, 0, &[overlay_buf], &[0]);
                if !highlight_verts.is_empty() {
                    // Triangles: bold crossed quads, flat-colored, depth-tested.
                    push_world.mvp = vp;
                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.sky_pipeline,
                    );
                    self.device.cmd_push_constants(
                        cmd,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push_world),
                    );
                    self.device
                        .cmd_draw(cmd, highlight_count, 1, world_start, 0);
                }
                if !line_verts.is_empty() {
                    // True line segments (F3 collision/target boxes).
                    push_world.mvp = vp;
                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.line_pipeline,
                    );
                    self.device.cmd_push_constants(
                        cmd,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push_world),
                    );
                    self.device.cmd_draw(
                        cmd,
                        line_verts.len() as u32,
                        1,
                        world_start + highlight_count,
                        0,
                    );
                }
            }

            // --- HUD (screen-space, depth-test off): hotbar slots + F3 panel --
            // Appended AFTER the world-overlay section in the same per-frame
            // buffer (never at offset 0 — that used to clobber the crosshair).
            let (mut hud_fills, mut hud_lines) = crate::overlay::build_hotbar(
                self.selected_slot,
                &self.slot_slots,
                aspect,
                Some(&self.slot_label),
            );
            // Settings page rides in the same HUD slice when open.
            if self.settings_open {
                let (sf, sl) = crate::overlay::build_settings_page(
                    &self.settings,
                    self.settings_hover,
                    aspect,
                );
                hud_fills.extend(sf);
                hud_lines.extend(sl);
            }
            // E inventory panel rides in the same HUD slice when open.
            if self.inv_open {
                let (pf, pl, _) = crate::overlay::build_inventory_panel(
                    &self.slot_slots,
                    &self.backpack,
                    self.inv_held,
                    self.inv_held_count,
                    self.inv_hover,
                    self.inv_cursor,
                    aspect,
                    &self.craft_grid,
                    self.craft_size,
                    self.craft_hover,
                    self.craft_output_hover,
                );
                hud_fills.extend(pf);
                hud_lines.extend(pl);
            }
            if !hud_fills.is_empty() || !hud_lines.is_empty() {
                let (overlay_buf, overlay_mem) = self.overlay_buffers[frame];
                // Fills first in the section; lines follow at a firstVertex offset.
                let fill_bytes: &[u8] = bytemuck::cast_slice(&hud_fills);
                let line_bytes: &[u8] = bytemuck::cast_slice(&hud_lines);
                let mut bytes = Vec::with_capacity(fill_bytes.len() + line_bytes.len());
                bytes.extend_from_slice(fill_bytes);
                bytes.extend_from_slice(line_bytes);
                debug_assert!(
                    OVERLAY_HUD_OFFSET + bytes.len() as u64 <= OVERLAY_DEBUG_OFFSET,
                    "HUD overlay exceeded its buffer slice"
                );
                let mapped = self.device.map_memory(
                    overlay_mem,
                    OVERLAY_HUD_OFFSET,
                    bytes.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
                self.device.unmap_memory(overlay_mem);

                let first_line_vertex = OVERLAY_HUD_VERTEX_START + hud_fills.len() as u32;
                if !hud_fills.is_empty() {
                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.hud_fill_pipeline,
                    );
                    self.device
                        .cmd_bind_vertex_buffers(cmd, 0, &[overlay_buf], &[0]);
                    // Identity transform: HUD vertices are already in NDC.
                    self.device.cmd_push_constants(
                        cmd,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push_hud),
                    );
                    self.device.cmd_draw(
                        cmd,
                        hud_fills.len() as u32,
                        1,
                        OVERLAY_HUD_VERTEX_START,
                        0,
                    );
                }
                if !hud_lines.is_empty() {
                    // The "line" geometry (slot borders, panel outlines) is
                    // built from thin TRIANGLE quads, so it draws through the
                    // fill pipeline — the old LINE_LIST pipeline here rendered
                    // them as broken 1-px traces (diagonals + missing edges).
                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.hud_fill_pipeline,
                    );
                    self.device
                        .cmd_bind_vertex_buffers(cmd, 0, &[overlay_buf], &[0]);
                    self.device.cmd_push_constants(
                        cmd,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push_hud),
                    );
                    self.device
                        .cmd_draw(cmd, hud_lines.len() as u32, 1, first_line_vertex, 0);
                }
            }

            // --- F3 debug panel: own buffer slice (it can be large) ---------
            if let Some(lines) = &self.debug_panel {
                if !lines.is_empty() {
                    let debug_fills = crate::debug_overlay::build_debug_panel(
                        lines,
                        self.debug_highlight,
                        aspect,
                    );
                    let (overlay_buf, overlay_mem) = self.overlay_buffers[frame];
                    let bytes: &[u8] = bytemuck::cast_slice(&debug_fills);
                    debug_assert!(
                        OVERLAY_DEBUG_OFFSET + bytes.len() as u64 <= OVERLAY_BUFFER_SIZE,
                        "debug panel exceeded its buffer slice"
                    );
                    let mapped = self.device.map_memory(
                        overlay_mem,
                        OVERLAY_DEBUG_OFFSET,
                        bytes.len() as u64,
                        vk::MemoryMapFlags::empty(),
                    )?;
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
                    self.device.unmap_memory(overlay_mem);

                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.hud_fill_pipeline,
                    );
                    self.device
                        .cmd_bind_vertex_buffers(cmd, 0, &[overlay_buf], &[0]);
                    self.device.cmd_push_constants(
                        cmd,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push_hud),
                    );
                    self.device.cmd_draw(
                        cmd,
                        debug_fills.len() as u32,
                        1,
                        OVERLAY_DEBUG_VERTEX_START,
                        0,
                    );
                }
            }
            self.device.cmd_end_render_pass(cmd);
            self.device.end_command_buffer(cmd)?;

            let wait_stage = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: &self.image_available[frame],
                p_wait_dst_stage_mask: wait_stage.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                signal_semaphore_count: 1,
                p_signal_semaphores: &self.render_finished[frame],
                ..Default::default()
            };
            self.device
                .queue_submit(self.queue, &[submit_info], self.image_fences[frame])?;

            let present_result = self.swapchain_loader.queue_present(
                self.queue,
                &vk::PresentInfoKHR {
                    wait_semaphore_count: 1,
                    p_wait_semaphores: &self.render_finished[frame],
                    swapchain_count: 1,
                    p_swapchains: &self.swapchain,
                    p_image_indices: [image_index as u32].as_ptr(),
                    ..Default::default()
                },
            );
            match present_result {
                Ok(_) => self.frames_presented += 1,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    self.recreate_swapchain(window)?;
                }
                Err(e) => return Err(e.into()),
            }

            self.current_frame = (frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for &f in &self.image_fences {
                self.device.destroy_fence(f, None);
            }
            for &s in self
                .image_available
                .iter()
                .chain(self.render_finished.iter())
            {
                self.device.destroy_semaphore(s, None);
            }
            self.clear_meshes();
            self.device.destroy_command_pool(self.command_pool, None);
            // Shadow resources + celestial-disc pipeline.
            for fb in self.shadow_framebuffers.drain(..) {
                self.device.destroy_framebuffer(fb, None);
            }
            for (buf, mem) in self.light_vp_buffers.drain(..) {
                self.device.destroy_buffer(buf, None);
                self.device.free_memory(mem, None);
            }
            for v in self.shadow_views.drain(..) {
                self.device.destroy_image_view(v, None);
            }
            for (img, mem) in self
                .shadow_images
                .drain(..)
                .zip(self.shadow_memories.drain(..))
            {
                self.device.destroy_image(img, None);
                self.device.free_memory(mem, None);
            }
            self.device.destroy_pipeline(self.shadow_pipeline, None);
            self.device.destroy_render_pass(self.shadow_pass, None);
            self.device.destroy_sampler(self.shadow_sampler, None);
            self.device.destroy_sampler(self.atlas_sampler, None);
            self.device.destroy_image_view(self.atlas_view, None);
            self.device.destroy_image(self.atlas_image, None);
            self.device.free_memory(self.atlas_memory, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_pipeline(self.sky_pipeline, None);
            self.device.destroy_pipeline(self.dome_pipeline, None);
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline(self.line_pipeline, None);
            self.device.destroy_pipeline(self.hud_fill_pipeline, None);
            self.device.destroy_pipeline(self.water_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for view in self.depth_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            for (image, memory) in self
                .depth_images
                .drain(..)
                .zip(self.depth_memories.drain(..))
            {
                self.device.destroy_image(image, None);
                self.device.free_memory(memory, None);
            }
            for &v in &self.image_views {
                self.device.destroy_image_view(v, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface_khr, None);
            if let Some((loader, messenger)) = self.debug_messenger.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

fn find_memory_type(
    requirements: vk::MemoryRequirements,
    props: vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32, Box<dyn Error>> {
    for i in 0..props.memory_type_count {
        if requirements.memory_type_bits & (1 << i) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(flags)
        {
            return Ok(i);
        }
    }
    Err("no suitable memory type".into())
}

/// Create a device-local buffer and copy `data` into it via a staging buffer.
unsafe fn create_device_buffer(
    device: &ash::Device,
    instance: &ash::Instance,
    pdevice: vk::PhysicalDevice,
    command_pool: &vk::CommandPool,
    queue: &vk::Queue,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn Error>> {
    unsafe {
        let size = data.len() as vk::DeviceSize;

        // Staging (host visible + coherent)
        let staging = device.create_buffer(
            &vk::BufferCreateInfo {
                size,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            },
            None,
        )?;
        let reqs = device.get_buffer_memory_requirements(staging);
        let mem_type = find_memory_type(
            reqs,
            instance.get_physical_device_memory_properties(pdevice),
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let staging_memory = device.allocate_memory(
            &vk::MemoryAllocateInfo {
                allocation_size: reqs.size,
                memory_type_index: mem_type,
                ..Default::default()
            },
            None,
        )?;
        device.bind_buffer_memory(staging, staging_memory, 0)?;

        let mapped = device.map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut u8, data.len());
        device.unmap_memory(staging_memory);

        // Destination (device local)
        let buffer = device.create_buffer(
            &vk::BufferCreateInfo {
                size,
                usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            },
            None,
        )?;
        let reqs = device.get_buffer_memory_requirements(buffer);
        let mem_type = find_memory_type(
            reqs,
            instance.get_physical_device_memory_properties(pdevice),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let memory = device.allocate_memory(
            &vk::MemoryAllocateInfo {
                allocation_size: reqs.size,
                memory_type_index: mem_type,
                ..Default::default()
            },
            None,
        )?;
        device.bind_buffer_memory(buffer, memory, 0)?;

        // One-off copy command
        let cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
            command_pool: *command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        })?[0];
        device.begin_command_buffer(cmd, &Default::default())?;
        device.cmd_copy_buffer(
            cmd,
            staging,
            buffer,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }],
        );
        device.end_command_buffer(cmd)?;
        let fence = device.create_fence(&Default::default(), None)?;
        let copy_submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        device.queue_submit(*queue, &[copy_submit], fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
        device.destroy_fence(fence, None);
        device.free_command_buffers(*command_pool, &[cmd]);
        device.destroy_buffer(staging, None);
        device.free_memory(staging_memory, None);

        Ok((buffer, memory))
    }
}

impl Renderer {
    /// Destroy a batch of meshes with a SINGLE device_wait_idle instead of
    /// one per mesh. During heavy streaming (or a save-dir prune) the old
    /// per-destroy idle serialized the queue and each call also waited on
    /// in-flight frames; batched, the whole unload is one idle. (RAM growth
    /// was actually staging buffers: each create_device_buffer allocated a
    /// fresh staging allocation per chunk upload — see upload_meshes.)
    pub unsafe fn destroy_meshes(&mut self, keys: &[MeshKey]) {
        unsafe {
            // The command buffers may still reference any soon-to-be-removed
            // chunk buffers. Wait BEFORE freeing them; doing it after
            // swap_remove/destroy is a use-after-free on the GPU queue.
            if !keys
                .iter()
                .any(|key| self.meshes.iter().any(|(k, _)| k == key))
            {
                return;
            }
            if let Err(error) = self.device.device_wait_idle() {
                log::error!("waiting for GPU before chunk unload failed: {error}");
                return;
            }
            for key in keys {
                if let Some(pos) = self.meshes.iter().position(|(k, _)| k == key) {
                    let (_, m) = self.meshes.swap_remove(pos);
                    self.device.destroy_buffer(m.vertex_buffer, None);
                    self.device.free_memory(m.vertex_memory, None);
                    self.device.destroy_buffer(m.index_buffer, None);
                    self.device.free_memory(m.index_memory, None);
                }
            }
        }
    }

    /// Upload several meshes in one batch: ONE staging allocation sized for
    /// the whole batch (freed after), one command buffer, one submit, one
    /// fence. The old per-chunk path allocated + destroyed a staging buffer
    /// per chunk per frame; the allocator couldn't return those pages to the
    /// OS fast enough, so RSS crept up during streaming. Batched, streaming
    /// a 32-chunk radius is a single small transient allocation per frame.
    /// A per-frame budget (max_bytes) also bounds frame spikes.
    pub unsafe fn upload_meshes(
        &mut self,
        batch: &[(MeshKey, Vec<Vertex>, Vec<u16>, [f32; 3])],
        max_bytes: vk::DeviceSize,
    ) -> Result<usize, Box<dyn Error>> {
        unsafe {
            if batch.is_empty() {
                return Ok(0);
            }
            // Respect the budget: take whole meshes until the next one would
            // exceed it (always take at least one so progress is guaranteed).
            let mut taken: Vec<&(MeshKey, Vec<Vertex>, Vec<u16>, [f32; 3])> = Vec::new();
            let mut total: vk::DeviceSize = 0;
            for item in batch {
                let bytes = (item.1.len() * std::mem::size_of::<Vertex>()
                    + item.2.len() * std::mem::size_of::<u16>())
                    as vk::DeviceSize;
                if !taken.is_empty() && total + bytes > max_bytes {
                    break;
                }
                total += bytes;
                taken.push(item);
            }

            // Destroy the old meshes for these keys first (batched idle).
            let keys: Vec<MeshKey> = taken.iter().map(|(k, _, _, _)| *k).collect();
            self.destroy_meshes(&keys);

            // Clone the device/instance loaders (cheap Arc handles) so the
            // mutable `self` borrows below don't conflict.
            let device = self.device.clone();
            let instance = self.instance.clone();
            // One staging buffer for the whole batch.
            let staging = device.create_buffer(
                &vk::BufferCreateInfo {
                    size: total,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )?;
            let reqs = device.get_buffer_memory_requirements(staging);
            let mem_type = find_memory_type(
                reqs,
                instance.get_physical_device_memory_properties(self.pdevice),
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let staging_memory = device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: reqs.size,
                    memory_type_index: mem_type,
                    ..Default::default()
                },
                None,
            )?;
            device.bind_buffer_memory(staging, staging_memory, 0)?;
            let mapped =
                device.map_memory(staging_memory, 0, total, vk::MemoryMapFlags::empty())?;
            let mut base = mapped as *mut u8;
            // Record (dst buffer, offset, size) per mesh for the copies.
            let mut regions: Vec<(vk::Buffer, vk::DeviceSize, vk::DeviceSize)> = Vec::new();
            let mut offset: vk::DeviceSize = 0;
            for (_, verts, idx, _) in &taken {
                let vb = (verts.len() * std::mem::size_of::<Vertex>()) as vk::DeviceSize;
                std::ptr::copy_nonoverlapping(verts.as_ptr().cast::<u8>(), base, vb as usize);
                base = base.add(vb as usize);
                let ib = (idx.len() * std::mem::size_of::<u16>()) as vk::DeviceSize;
                std::ptr::copy_nonoverlapping(idx.as_ptr().cast::<u8>(), base, ib as usize);
                base = base.add(ib as usize);
                regions.push((vk::Buffer::null(), offset, vb));
                regions.last_mut().unwrap().0 = vk::Buffer::null(); // placeholder
                offset += vb;
                regions.push((vk::Buffer::null(), offset, ib));
                offset += ib;
            }
            device.unmap_memory(staging_memory);

            // Create the device-local buffers for every mesh in the batch.
            let mut created: Vec<(
                MeshKey,
                vk::Buffer,
                vk::DeviceMemory,
                vk::Buffer,
                vk::DeviceMemory,
                u32,
                [f32; 3],
            )> = Vec::with_capacity(taken.len());
            let mut region_i = 0usize;
            for (key, verts, idx, model_offset) in &taken {
                let vb = (verts.len() * std::mem::size_of::<Vertex>()) as vk::DeviceSize;
                let ib = (idx.len() * std::mem::size_of::<u16>()) as vk::DeviceSize;
                let (vertex_buffer, vertex_memory) = self.create_device_buffer_from(
                    staging,
                    regions[region_i].1,
                    vb,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                )?;
                let (index_buffer, index_memory) = self.create_device_buffer_from(
                    staging,
                    regions[region_i + 1].1,
                    ib,
                    vk::BufferUsageFlags::INDEX_BUFFER,
                )?;
                region_i += 2;
                created.push((
                    *key,
                    vertex_buffer,
                    vertex_memory,
                    index_buffer,
                    index_memory,
                    idx.len() as u32,
                    *model_offset,
                ));
            }

            // One one-off submit for all copies.
            let cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })?[0];
            device.begin_command_buffer(cmd, &Default::default())?;
            // Record every pending staging→device copy (one per created
            // buffer, registered by create_device_buffer_from).
            for (src, dst, src_offset, size) in &self.pending_copies {
                device.cmd_copy_buffer(
                    cmd,
                    *src,
                    *dst,
                    &[vk::BufferCopy {
                        src_offset: *src_offset,
                        dst_offset: 0,
                        size: *size,
                    }],
                );
            }
            device.end_command_buffer(cmd)?;
            let fence = device.create_fence(&Default::default(), None)?;
            let submit = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                ..Default::default()
            };
            device.queue_submit(self.queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.command_pool, &[cmd]);
            device.destroy_buffer(staging, None);
            device.free_memory(staging_memory, None);
            self.pending_copies.clear();

            for (key, vb, vm, ib, im, count, off) in created {
                self.meshes.push((
                    key,
                    Mesh {
                        vertex_buffer: vb,
                        vertex_memory: vm,
                        index_buffer: ib,
                        index_memory: im,
                        index_count: count,
                        model_offset: off,
                    },
                ));
            }
            Ok(taken.len())
        }
    }

    /// Create a device-local buffer whose initial contents are copied from
    /// [src_offset, src_offset + size) of an already-filled staging buffer.
    /// Shares the batch's single submit instead of allocating its own.
    unsafe fn create_device_buffer_from(
        &mut self,
        staging: vk::Buffer,
        src_offset: vk::DeviceSize,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn Error>> {
        unsafe {
            let device = &self.device;
            let buffer = device.create_buffer(
                &vk::BufferCreateInfo {
                    size,
                    usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )?;
            let reqs = device.get_buffer_memory_requirements(buffer);
            let mem_type = find_memory_type(
                reqs,
                self.instance
                    .get_physical_device_memory_properties(self.pdevice),
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let memory = device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: reqs.size,
                    memory_type_index: mem_type,
                    ..Default::default()
                },
                None,
            )?;
            device.bind_buffer_memory(buffer, memory, 0)?;
            self.pending_copies
                .push((staging, buffer, src_offset, size));
            Ok((buffer, memory))
        }
    }
}

/// Create the block texture atlas image, upload its texels through a
/// staging buffer, and transition it to SHADER_READ_ONLY_OPTIMAL.
unsafe fn create_atlas_image(
    device: &ash::Device,
    instance: &ash::Instance,
    pdevice: vk::PhysicalDevice,
    command_pool: &vk::CommandPool,
    queue: &vk::Queue,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler), Box<dyn Error>> {
    unsafe {
        let data = crate::textures::build_atlas();
        let (w, h) = (
            crate::textures::ATLAS_W as u32,
            crate::textures::ATLAS_H as u32,
        );

        let image = device.create_image(
            &vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_SRGB,
                extent: vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            },
            None,
        )?;
        let reqs = device.get_image_memory_requirements(image);
        let mem_type = find_memory_type(
            reqs,
            instance.get_physical_device_memory_properties(pdevice),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let memory = device.allocate_memory(
            &vk::MemoryAllocateInfo {
                allocation_size: reqs.size,
                memory_type_index: mem_type,
                ..Default::default()
            },
            None,
        )?;
        device.bind_image_memory(image, memory, 0)?;

        // Staging buffer with the texels.
        let (staging, staging_mem) = create_device_buffer(
            device,
            instance,
            pdevice,
            command_pool,
            queue,
            &data,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;

        // One-off command buffer: layout transitions + copy.
        let cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
            command_pool: *command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        })?[0];
        device.begin_command_buffer(cmd, &Default::default())?;
        let subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            level_count: 1,
            layer_count: 1,
            ..Default::default()
        };
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image,
                subresource_range: subresource,
                ..Default::default()
            }],
        );
        device.cmd_copy_buffer_to_image(
            cmd,
            staging,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                },
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
            }],
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image,
                subresource_range: subresource,
                ..Default::default()
            }],
        );
        device.end_command_buffer(cmd)?;
        let fence = device.create_fence(&Default::default(), None)?;
        device.queue_submit(
            *queue,
            &[vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                ..Default::default()
            }],
            fence,
        )?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
        device.destroy_fence(fence, None);
        device.free_command_buffers(*command_pool, &[cmd]);
        device.destroy_buffer(staging, None);
        device.free_memory(staging_mem, None);

        let view = device.create_image_view(
            &vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: vk::Format::R8G8B8A8_SRGB,
                subresource_range: subresource,
                ..Default::default()
            },
            None,
        )?;
        let sampler = device.create_sampler(
            &vk::SamplerCreateInfo {
                mag_filter: vk::Filter::NEAREST, // crisp Minecraft pixels
                min_filter: vk::Filter::NEAREST,
                mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                ..Default::default()
            },
            None,
        )?;

        Ok((image, memory, view, sampler))
    }
}
