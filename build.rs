use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for (src, dst, stage) in [
        ("shaders/vert.glsl", "vert.spv", "vertex"),
        ("shaders/frag.glsl", "frag.spv", "fragment"),
        ("shaders/shadow_vert.glsl", "shadow_vert.spv", "vertex"),
        ("shaders/shadow_frag.glsl", "shadow_frag.spv", "fragment"),
        ("shaders/sky_vert.glsl", "sky_vert.spv", "vertex"),
        ("shaders/sky_frag.glsl", "sky_frag.spv", "fragment"),
        ("shaders/dome_vert.glsl", "dome_vert.spv", "vertex"),
        ("shaders/dome_frag.glsl", "dome_frag.spv", "fragment"),
    ] {
        println!("cargo:rerun-if-changed={src}");
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let spv = format!("{out_dir}/{dst}");
        let status = Command::new("glslc")
            .args([
                "-O",
                "-x",
                "glsl",
                &format!("-fshader-stage={stage}"),
                "-o",
                &spv,
                src,
            ])
            .status()
            .unwrap_or_else(|e| {
                panic!(
                    "build failed: cannot run `glslc` ({e}) — the Vulkan SDK's shader\n\
                     compiler is required to build the shaders. Install it, e.g.:\n\
                     - Arch/SteamOS: pacman -S shaderc\n\
                     - Ubuntu/Debian: apt install glslc  (or shaderc package)\n\
                     - Or the full SDK: https://vulkan.lunarg.com/sdk/home\n\
                     Then run `cargo build` again."
                );
            });
        if !status.success() {
            panic!("glslc failed to compile {src}");
        }
    }
}
