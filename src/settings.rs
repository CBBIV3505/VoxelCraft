//! Graphics settings, driven by the in-game settings page (O to open).
//!
//! Every field maps directly onto a rendering knob: `render_distance` is the
//! chunk streaming radius (and therefore also the camera far plane and fog
//! end), `shadows` enables the shadow-map pass, `clouds` the drifting cloud
//! layer. Ray-traced shadows/AO and god rays are planned shader-pack features
//! — they get named (disabled) entries so the UI can grow into them without
//! restructuring.
//!
//! Settings persist to a tiny `key=value` file under the user's config dir
//! (`$XDG_CONFIG_HOME/voxelcraft/settings.conf`, falling back to
//! `$HOME/.config/...`) and are reloaded on the next launch.

/// Camera far plane in blocks, scaled with the render distance.
pub fn far_plane(render_distance: i32) -> f32 {
    // At least 110: the sun/moon discs billboard at 90 blocks from the
    // camera, so a tighter far plane would clip them out of the sky on low
    // render distances.
    ((render_distance as f32 + 2.0) * crate::terrain::CHUNK_SIZE as f32).max(110.0)
}

/// All settings with their live values, for the settings page.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Chunk streaming radius around the player (1–32).
    pub render_distance: i32,
    /// Cloud layer radius in grid cells around the player (2–10); one cell
    /// is [`crate::overlay::CLOUD_CELL`] blocks wide.
    pub cloud_distance: i32,
    /// Sun/moon shadow map on/off.
    pub shadows: bool,
    /// Drifting cloud layer on/off.
    pub clouds: bool,
    /// Planned shader-pack features — present in the UI as grayed-out rows
    /// but not yet implemented, so they're read-only for now.
    #[allow(dead_code)] // wired up when the shader-pack milestone lands
    pub rt_shadows: bool,
    #[allow(dead_code)]
    pub rt_ao: bool,
    #[allow(dead_code)]
    pub god_rays: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            render_distance: crate::terrain::VIEW_RADIUS, // the old constant's value
            cloud_distance: 7,
            shadows: true,
            clouds: true,
            rt_shadows: false,
            rt_ao: false,
            god_rays: false,
        }
    }
}

impl Settings {
    /// Wheel notch on the slider: one chunk step, clamped to the valid range.
    pub fn step_render_distance(&mut self, delta: i32) {
        self.render_distance = (self.render_distance + delta).clamp(1, 32);
    }

    /// Wheel notch on the cloud-distance slider: one cell step (2–10).
    pub fn step_cloud_distance(&mut self, delta: i32) {
        self.cloud_distance = (self.cloud_distance + delta).clamp(2, 10);
    }

    pub fn toggle_shadows(&mut self) {
        self.shadows = !self.shadows;
    }

    pub fn toggle_clouds(&mut self) {
        self.clouds = !self.clouds;
    }

    /// Direct slider assignment (mouse drag / track click), clamped.
    pub fn set_render_distance(&mut self, v: i32) {
        self.render_distance = v.clamp(1, 32);
    }

    /// Direct slider assignment (mouse drag / track click), clamped.
    pub fn set_cloud_distance(&mut self, v: i32) {
        self.cloud_distance = v.clamp(2, 10);
    }

    // --- Persistence ---------------------------------------------------------

    /// Config file location: `$XDG_CONFIG_HOME/voxelcraft/settings.conf`
    /// (falling back to `$HOME/.config/...`).
    fn config_path() -> std::path::PathBuf {
        let base = match std::env::var("XDG_CONFIG_HOME") {
            Ok(dir) if !dir.is_empty() => std::path::PathBuf::from(dir),
            _ => {
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
                std::path::PathBuf::from(home).join(".config")
            }
        };
        base.join("voxelcraft").join("settings.conf")
    }

    /// Load persisted settings; every absent/invalid line keeps its default.
    pub fn load() -> Self {
        let path = Self::config_path();
        match std::fs::read_to_string(&path) {
            Ok(text) => Self::parse(&text),
            Err(_) => Self::default(), // first run (or unreadable) → defaults
        }
    }

    /// Parse `key=value` lines; unknown keys and out-of-range values are
    /// ignored so an old or hand-edited file can never break the game.
    fn parse(text: &str) -> Self {
        let mut s = Self::default();
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
                "render_distance" => {
                    if let Ok(n) = v.parse::<i32>() {
                        s.set_render_distance(n);
                    }
                }
                "cloud_distance" => {
                    if let Ok(n) = v.parse::<i32>() {
                        s.set_cloud_distance(n);
                    }
                }
                "shadows" => s.shadows = v == "true" || v == "1",
                "clouds" => s.clouds = v == "true" || v == "1",
                _ => {} // future keys (rt_shadows, ...) parse when they exist
            }
        }
        s
    }

    /// Write the current settings to disk (atomic tmp-file + rename, so a
    /// crash mid-write can never truncate the real file).
    pub fn save(&self) {
        let path = Self::config_path();
        self.save_to(&path);
    }

    fn save_to(&self, path: &std::path::Path) {
        let text = format!(
            "# VoxelCraft settings (hand-editable; invalid lines are ignored)\n\
             render_distance={}\n\
             cloud_distance={}\n\
             shadows={}\n\
             clouds={}\n",
            self.render_distance, self.cloud_distance, self.shadows, self.clouds
        );
        if let Some(dir) = path.parent() {
            if std::fs::create_dir_all(dir).is_err() {
                return;
            }
        }
        let tmp = path.with_extension("conf.tmp");
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_disk() {
        let mut s = Settings::default();
        s.render_distance = 6;
        s.cloud_distance = 3;
        s.shadows = false;
        s.clouds = true;
        let path =
            std::env::temp_dir().join(format!("voxelcraft-test-{}.conf", std::process::id()));
        s.save_to(&path);
        let text = std::fs::read_to_string(&path).unwrap();
        let loaded = Settings::parse(&text);
        assert_eq!(loaded.render_distance, 6);
        assert_eq!(loaded.cloud_distance, 3);
        assert!(!loaded.shadows);
        assert!(loaded.clouds);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn garbage_and_gaps_fall_back_to_defaults() {
        let s = Settings::parse(
            "render_distance=99\ncloud_distance=nope\nshadows=yes\n\nbogus=1\nclouds=true\n",
        );
        // 99 clamps to the slider max; junk strings keep defaults; only
        // exact true/1 count as on.
        assert_eq!(s.render_distance, 32);
        assert_eq!(s.cloud_distance, Settings::default().cloud_distance);
        assert!(!s.shadows);
        assert!(s.clouds);
    }
}
