//! What the fake helpers say to `devices`, shared by both. `FAKE_DEVICES` lists the graphics devices as
//! `kind:backend:name:memory_mb` entries separated by `;` (kind is `gpu` or `igpu`), for example
//! `igpu:Vulkan:Radeon Graphics:512;gpu:Vulkan:RTX 5070 Ti:16303`; the processor is always added last, like ggml.
//! Unset: the processor only. The chosen index is the protocol's own `pick`.

use voice_engine_protocol::gpu::{Device, DeviceKind, DeviceList, pick};

pub fn list() -> DeviceList {
    let spec = std::env::var("FAKE_DEVICES").unwrap_or_default();
    let mut devices: Vec<Device> = spec
        .split(';')
        .filter(|e| !e.trim().is_empty())
        .enumerate()
        .map(|(index, entry)| {
            let mut f = entry.split(':');
            let kind = match f.next() {
                Some("gpu") => DeviceKind::Gpu,
                _ => DeviceKind::Igpu,
            };
            let backend = f.next().unwrap_or("Vulkan").to_string();
            Device {
                index,
                name: format!("{backend}{index}"),
                description: f.next().unwrap_or("").to_string(),
                backend,
                kind,
                memory_mb: f.next().and_then(|m| m.parse().ok()).unwrap_or(0),
            }
        })
        .collect();
    devices.push(Device {
        index: devices.len(),
        name: "CPU".into(),
        description: "Fake CPU".into(),
        backend: "CPU".into(),
        kind: DeviceKind::Cpu,
        memory_mb: 16000,
    });
    let chosen = pick(&devices).map(|d| d.index);
    DeviceList { devices, chosen }
}
