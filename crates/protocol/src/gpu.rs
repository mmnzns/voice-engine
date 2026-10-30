//! Which device a GPU helper uses. Pure, so it's tested here without a GPU; the helpers fill `Device` from ggml.
//!
//! The rule: the discrete GPU with the most memory, and only that one. Never an integrated GPU (Miguel's PC has an
//! AMD Radeon iGPU with 512 MB next to its RTX 5070 Ti, and Whisper would take the first GPU it finds), and never a
//! split across devices (llama.cpp's default spreads layers over every GPU). No discrete GPU: the processor.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Cpu,
    /// A discrete graphics card.
    Gpu,
    /// Integrated graphics, sharing the computer's memory.
    Igpu,
    Accel,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Device {
    /// ggml's device index (what llama.cpp's `with_devices` takes).
    pub index: usize,
    /// ggml's name ("Vulkan0").
    pub name: String,
    /// What the driver calls it ("NVIDIA GeForce RTX 5070 Ti").
    pub description: String,
    /// ggml backend ("Vulkan", "CPU").
    pub backend: String,
    pub kind: DeviceKind,
    pub memory_mb: u64,
}

/// The backend name of the adapters ONNX Runtime's DirectML can use (Parakeet, on Windows). Their `Device::index` is
/// DXGI's adapter number, the id DirectML takes, not a ggml index.
pub const DIRECTML: &str = "DirectML";

/// A helper's answer to `devices`: every device it sees, and the index of the one it uses (None: the processor).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DeviceList {
    pub devices: Vec<Device>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen: Option<usize>,
}

impl DeviceList {
    /// The discrete card the engine would use: the answer to "does this PC have one, and what is it called".
    pub fn discrete_card(&self) -> Option<&Device> {
        discrete_card(&self.devices)
    }
}

fn is_directml(d: &Device) -> bool {
    d.backend.eq_ignore_ascii_case(DIRECTML)
}

/// The discrete device with the most memory among the DirectML ones (`directml`) or the ggml ones (not).
fn best(devices: &[Device], directml: bool) -> Option<&Device> {
    devices
        .iter()
        .filter(|d| is_directml(d) == directml)
        .filter(|d| d.kind == DeviceKind::Gpu && !d.backend.eq_ignore_ascii_case("cpu"))
        // max_by_key keeps the last of equal maxima; reversed, the first one listed wins a tie.
        .rev()
        .max_by_key(|d| d.memory_mb)
}

/// The ggml device to use (Whisper, llama.cpp), or None for the processor. DirectML adapters are not ggml devices
/// and are never picked here.
pub fn pick(devices: &[Device]) -> Option<&Device> {
    best(devices, false)
}

/// Whether this PC has a discrete graphics card the engine would use, and which: the card `pick` chooses for ggml,
/// else (a build with DirectML only, which is Parakeet's) the DirectML adapter with the most memory. Apps turn the
/// card on by default on Windows only when this is Some.
pub fn discrete_card(devices: &[Device]) -> Option<&Device> {
    pick(devices).or_else(|| best(devices, true))
}

/// whisper.cpp's `gpu_device` for a chosen device: whisper.cpp counts only GPU and iGPU devices, in ggml's order,
/// and takes the n-th one. DirectML adapters are not in ggml's list.
pub fn whisper_gpu_index(devices: &[Device], chosen: &Device) -> Option<i32> {
    devices
        .iter()
        .filter(|d| !is_directml(d))
        .filter(|d| matches!(d.kind, DeviceKind::Gpu | DeviceKind::Igpu))
        .position(|d| d.index == chosen.index)
        .map(|n| n as i32)
}

pub fn bytes_to_mb(bytes: usize) -> u64 {
    (bytes as u64) / (1024 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(index: usize, description: &str, kind: DeviceKind, memory_mb: u64) -> Device {
        let backend = if kind == DeviceKind::Cpu {
            "CPU"
        } else {
            "Vulkan"
        };
        Device {
            index,
            name: format!("{backend}{index}"),
            description: description.into(),
            backend: backend.into(),
            kind,
            memory_mb,
        }
    }

    #[test]
    fn miguels_pc_picks_the_rtx_not_the_radeon_igpu() {
        // Vulkan lists the iGPU first on this PC; the CPU device comes last in ggml.
        let devices = vec![
            dev(0, "AMD Radeon(TM) Graphics", DeviceKind::Igpu, 512),
            dev(1, "NVIDIA GeForce RTX 5070 Ti", DeviceKind::Gpu, 16303),
            dev(2, "CPU", DeviceKind::Cpu, 64000),
        ];
        let chosen = pick(&devices).unwrap();
        assert_eq!(chosen.description, "NVIDIA GeForce RTX 5070 Ti");
        assert_eq!(chosen.index, 1);
        // whisper.cpp counts the iGPU too, so the RTX is its GPU number 1.
        assert_eq!(whisper_gpu_index(&devices, chosen), Some(1));
    }

    #[test]
    fn miguels_pc_has_a_discrete_card() {
        let devices = vec![
            dev(0, "AMD Radeon(TM) Graphics", DeviceKind::Igpu, 512),
            dev(1, "NVIDIA GeForce RTX 5070 Ti", DeviceKind::Gpu, 16303),
            dev(2, "CPU", DeviceKind::Cpu, 64000),
        ];
        let card = discrete_card(&devices).unwrap();
        assert_eq!(card.description, "NVIDIA GeForce RTX 5070 Ti");
        let list = DeviceList {
            devices: devices.clone(),
            chosen: Some(1),
        };
        assert_eq!(list.discrete_card(), Some(card));
    }

    #[test]
    fn a_pc_with_only_integrated_graphics_or_only_a_processor_has_no_discrete_card() {
        let igpu_only = [
            dev(0, "Intel(R) UHD Graphics", DeviceKind::Igpu, 8000),
            dev(1, "CPU", DeviceKind::Cpu, 32000),
        ];
        assert_eq!(discrete_card(&igpu_only), None);
        assert_eq!(
            discrete_card(&[dev(0, "CPU", DeviceKind::Cpu, 32000)]),
            None
        );
        assert_eq!(discrete_card(&[]), None);
        assert_eq!(DeviceList::default().discrete_card(), None);
    }

    fn directml(index: usize, description: &str, kind: DeviceKind, memory_mb: u64) -> Device {
        Device {
            index,
            name: format!("DirectML{index}"),
            description: description.into(),
            backend: DIRECTML.into(),
            kind,
            memory_mb,
        }
    }

    #[test]
    fn directml_adapters_never_change_the_ggml_choice_but_can_answer_alone() {
        let both = vec![
            dev(0, "AMD Radeon(TM) Graphics", DeviceKind::Igpu, 512),
            dev(1, "NVIDIA GeForce RTX 5070 Ti", DeviceKind::Gpu, 16303),
            dev(2, "CPU", DeviceKind::Cpu, 64000),
            directml(0, "NVIDIA GeForce RTX 5070 Ti", DeviceKind::Gpu, 16384),
            directml(1, "AMD Radeon(TM) Graphics", DeviceKind::Igpu, 512),
        ];
        // The ggml card wins even though DirectML counts more memory, and whisper's number ignores DirectML.
        let chosen = pick(&both).unwrap();
        assert_eq!(chosen.backend, "Vulkan");
        assert_eq!(whisper_gpu_index(&both, chosen), Some(1));
        assert_eq!(discrete_card(&both), Some(chosen));
        // A build with DirectML only (Parakeet in the app's own process).
        let dml_only = &both[3..];
        assert_eq!(pick(dml_only), None);
        let card = discrete_card(dml_only).unwrap();
        assert_eq!((card.backend.as_str(), card.index), ("DirectML", 0));
        assert_eq!(
            discrete_card(&[directml(0, "iGPU", DeviceKind::Igpu, 512)]),
            None
        );
    }

    #[test]
    fn the_largest_discrete_gpu_wins_and_a_tie_goes_to_the_first() {
        let devices = vec![
            dev(0, "small", DeviceKind::Gpu, 4096),
            dev(1, "big", DeviceKind::Gpu, 12288),
            dev(2, "big twin", DeviceKind::Gpu, 12288),
        ];
        assert_eq!(pick(&devices).unwrap().description, "big");
        assert_eq!(
            whisper_gpu_index(&devices, pick(&devices).unwrap()),
            Some(1)
        );
    }

    #[test]
    fn integrated_graphics_or_nothing_means_the_processor() {
        assert_eq!(pick(&[]), None);
        assert_eq!(pick(&[dev(0, "CPU", DeviceKind::Cpu, 32000)]), None);
        assert_eq!(
            pick(&[
                dev(0, "Intel(R) UHD Graphics", DeviceKind::Igpu, 8000),
                dev(1, "CPU", DeviceKind::Cpu, 32000)
            ]),
            None
        );
        assert_eq!(pick(&[dev(0, "Some NPU", DeviceKind::Accel, 8000)]), None);
        // A GPU-typed device on the CPU backend (never seen, but cheap to rule out).
        let odd = Device {
            backend: "CPU".into(),
            ..dev(0, "odd", DeviceKind::Gpu, 9000)
        };
        assert_eq!(pick(&[odd]), None);
    }

    #[test]
    fn a_device_whisper_cant_see_has_no_whisper_index() {
        let devices = vec![dev(0, "gpu", DeviceKind::Gpu, 8000)];
        assert_eq!(
            whisper_gpu_index(&devices, &dev(5, "elsewhere", DeviceKind::Gpu, 8000)),
            None
        );
    }

    #[test]
    fn memory_is_in_mebibytes() {
        assert_eq!(bytes_to_mb(17_095_983_104), 16304);
        assert_eq!(bytes_to_mb(0), 0);
    }
}
