//! Lightweight system info: CPU model, memory totals/usage. Reads /proc on
//! Linux; other platforms get placeholders rather than pulling in a deps
//! crate (this is a debug overlay, not a monitoring tool).

/// What the debug overlay displays about the machine.
#[derive(Clone, Debug)]
pub struct SysInfo {
    pub cpu_model: String,
    pub ram_total_mib: u64,
    pub ram_used_mib: u64,
}

#[cfg(target_os = "linux")]
mod imp {
    use std::collections::HashMap;

    fn parse_meminfo() -> (u64, u64) {
        let mut map: HashMap<String, u64> = HashMap::new();
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    let kib: String = v.trim().chars().filter(|c| c.is_ascii_digit()).collect();
                    if let Ok(kib) = kib.parse::<u64>() {
                        map.insert(k.trim().to_string(), kib / 1024); // → MiB
                    }
                }
            }
        }
        let total = map.get("MemTotal").copied().unwrap_or(0);
        let available = map.get("MemAvailable").copied().unwrap_or(0);
        (total, total.saturating_sub(available))
    }

    pub fn read() -> super::SysInfo {
        let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split_once(':'))
                    .map(|(_, v)| v.trim().to_string())
            })
            .unwrap_or_else(|| "unknown".into());
        let (ram_total_mib, ram_used_mib) = parse_meminfo();
        super::SysInfo {
            cpu_model,
            ram_total_mib,
            ram_used_mib,
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    pub fn read() -> super::SysInfo {
        super::SysInfo {
            cpu_model: "unknown (unsupported platform)".into(),
            ram_total_mib: 0,
            ram_used_mib: 0,
        }
    }
}

pub fn read() -> SysInfo {
    imp::read()
}
