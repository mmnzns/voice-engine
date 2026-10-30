//! Which graphics card DirectML runs Parakeet on (Windows, `directml` feature).
//!
//! ONNX Runtime's DirectML provider takes a device id, and with none it uses adapter 0: whatever Windows lists first,
//! which on Miguel's PC is the Ryzen's Radeon iGPU, not the RTX 5070 Ti. So the engine lists the adapters itself and
//! passes the id of the discrete one: the same card the Vulkan helper picks. The choice is a pure function
//! ([`discrete`]), tested on every platform; only the listing is Windows code.
//!
//! The id is DXGI's adapter number (ONNX Runtime's DirectML provider enumerates with DXGI). DXGI says whether an
//! adapter is software but not whether it is integrated, so that comes from DXCore, matched by the adapter's LUID.

use voice_engine_protocol::gpu::{DIRECTML, Device, DeviceKind};

/// One graphics adapter as Windows lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adapter {
    /// DXGI's adapter number: the device id DirectML takes.
    pub index: u32,
    /// What the driver calls it ("NVIDIA GeForce RTX 5070 Ti").
    pub name: String,
    /// Integrated graphics, sharing the computer's memory.
    pub integrated: bool,
    /// A software renderer ("Microsoft Basic Render Driver"), not a card.
    pub software: bool,
    /// Memory on the card itself; an integrated adapter has little or none.
    pub dedicated_mb: u64,
    /// The PCI vendor id (0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel, 0x1414 Microsoft).
    pub vendor_id: u32,
}

/// Microsoft's vendor id: "Microsoft Basic Display Adapter" (no driver installed) and Hyper-V or remote display
/// adapters are not cards, however they are flagged, and DirectML fails on them.
const MICROSOFT_VENDOR: u32 = 0x1414;

/// The adapter to run on: the discrete one with the most dedicated memory (the first listed wins a tie), never an
/// integrated, software or Microsoft one, and never one with no dedicated memory. None: DirectML is not tried and the
/// model runs on the processor.
pub fn discrete(adapters: &[Adapter]) -> Option<&Adapter> {
    adapters
        .iter()
        .filter(|a| {
            !a.software && !a.integrated && a.vendor_id != MICROSOFT_VENDOR && a.dedicated_mb > 0
        })
        // max_by_key keeps the last of equal maxima; reversed, the first one listed wins a tie.
        .rev()
        .max_by_key(|a| a.dedicated_mb)
}

/// The adapters as protocol devices (backend "DirectML", `index` the DXGI number), software and Microsoft ones left
/// out, for the speech helper's `devices` answer.
pub fn devices(adapters: &[Adapter]) -> Vec<Device> {
    adapters
        .iter()
        .filter(|a| !a.software && a.vendor_id != MICROSOFT_VENDOR)
        .map(|a| Device {
            index: a.index as usize,
            name: format!("{DIRECTML}{}", a.index),
            description: a.name.clone(),
            backend: DIRECTML.into(),
            kind: if a.integrated {
                DeviceKind::Igpu
            } else {
                DeviceKind::Gpu
            },
            memory_mb: a.dedicated_mb,
        })
        .collect()
}

/// Every adapter, or none when this build has no DirectML (not Windows, or without the `directml` feature).
pub fn adapters() -> Vec<Adapter> {
    #[cfg(all(windows, feature = "directml"))]
    {
        dxgi::adapters()
    }
    #[cfg(not(all(windows, feature = "directml")))]
    {
        Vec::new()
    }
}

/// The adapter a load with `gpu: true` runs DirectML on, if this build has DirectML and the PC has a discrete card.
pub fn card() -> Option<Adapter> {
    discrete(&adapters()).cloned()
}

#[cfg(all(windows, feature = "directml"))]
mod dxgi {
    use super::Adapter;
    use std::ffi::c_void;
    use voice_engine_protocol::gpu::bytes_to_mb;
    use windows::Win32::Foundation::LUID;
    use windows::Win32::Graphics::DXCore::{
        DXCoreCreateAdapterFactory, IDXCoreAdapter, IDXCoreAdapterFactory, IsIntegrated,
    };
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory1,
    };

    /// Every DXGI adapter, in DXGI's order. Empty when Windows will not list them.
    pub fn adapters() -> Vec<Adapter> {
        let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else {
            return Vec::new();
        };
        let core = unsafe { DXCoreCreateAdapterFactory::<IDXCoreAdapterFactory>() }.ok();
        let mut found = Vec::new();
        let mut index = 0;
        // Ends with DXGI_ERROR_NOT_FOUND after the last adapter.
        while let Ok(adapter) = unsafe { factory.EnumAdapters1(index) } {
            if let Ok(desc) = unsafe { adapter.GetDesc1() } {
                let len = desc
                    .Description
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(desc.Description.len());
                found.push(Adapter {
                    index,
                    name: String::from_utf16_lossy(&desc.Description[..len]),
                    // An adapter DXCore cannot vouch for is not offered as a card.
                    integrated: core
                        .as_ref()
                        .and_then(|f| integrated(f, desc.AdapterLuid))
                        .unwrap_or(true),
                    software: desc.Flags & (DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0,
                    dedicated_mb: bytes_to_mb(desc.DedicatedVideoMemory),
                    vendor_id: desc.VendorId,
                });
            }
            index += 1;
        }
        found
    }

    /// DXCore's `IsIntegrated` for the adapter with this LUID.
    fn integrated(factory: &IDXCoreAdapterFactory, luid: LUID) -> Option<bool> {
        let adapter: IDXCoreAdapter = unsafe { factory.GetAdapterByLuid(&luid) }.ok()?;
        if !unsafe { adapter.IsPropertySupported(IsIntegrated) } {
            return None;
        }
        let mut value = false;
        unsafe {
            adapter.GetProperty(
                IsIntegrated,
                std::mem::size_of::<bool>(),
                (&mut value as *mut bool).cast::<c_void>(),
            )
        }
        .ok()?;
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(index: u32, name: &str, integrated: bool, dedicated_mb: u64) -> Adapter {
        Adapter {
            index,
            name: name.into(),
            integrated,
            software: false,
            dedicated_mb,
            vendor_id: 0x10DE,
        }
    }

    fn basic_display_adapter(index: u32) -> Adapter {
        Adapter {
            vendor_id: MICROSOFT_VENDOR,
            ..adapter(index, "Microsoft Basic Display Adapter", false, 0)
        }
    }

    fn software(index: u32) -> Adapter {
        Adapter {
            software: true,
            ..adapter(index, "Microsoft Basic Render Driver", false, 0)
        }
    }

    #[test]
    fn miguels_pc_runs_directml_on_the_rtx_not_adapter_zero() {
        // DXGI lists the Radeon iGPU first, then the RTX, then the software renderer.
        let adapters = vec![
            adapter(0, "AMD Radeon(TM) Graphics", true, 512),
            adapter(1, "NVIDIA GeForce RTX 5070 Ti", false, 16303),
            software(2),
        ];
        let card = discrete(&adapters).unwrap();
        assert_eq!(
            (card.index, card.name.as_str()),
            (1, "NVIDIA GeForce RTX 5070 Ti")
        );
    }

    #[test]
    fn integrated_or_software_adapters_alone_mean_no_directml() {
        assert_eq!(discrete(&[]), None);
        assert_eq!(
            discrete(&[adapter(0, "Intel(R) UHD Graphics", true, 128), software(1)]),
            None
        );
        // An integrated adapter with a lot of dedicated memory is still integrated.
        assert_eq!(discrete(&[adapter(0, "Big APU", true, 8192)]), None);
    }

    #[test]
    fn a_basic_display_adapter_or_a_vm_with_only_microsoft_adapters_means_no_directml() {
        assert_eq!(discrete(&[basic_display_adapter(0)]), None);
        // Hyper-V: a Microsoft display adapter next to the software renderer.
        let vm = [
            Adapter {
                vendor_id: MICROSOFT_VENDOR,
                ..adapter(0, "Microsoft Hyper-V Video", false, 512)
            },
            software(1),
        ];
        assert_eq!(discrete(&vm), None);
    }

    #[test]
    fn an_adapter_with_no_dedicated_memory_is_not_a_card() {
        assert_eq!(discrete(&[adapter(0, "Some Vendor GPU", false, 0)]), None);
    }

    #[test]
    fn a_basic_display_adapter_does_not_hide_the_real_card() {
        let adapters = vec![
            basic_display_adapter(0),
            adapter(1, "NVIDIA GeForce RTX 5070 Ti", false, 16303),
        ];
        assert_eq!(discrete(&adapters).unwrap().index, 1);
    }

    #[test]
    fn the_largest_card_wins_and_a_tie_goes_to_the_first_listed() {
        let adapters = vec![
            adapter(0, "small", false, 4096),
            adapter(1, "big", false, 12288),
            adapter(2, "big twin", false, 12288),
        ];
        assert_eq!(discrete(&adapters).unwrap().index, 1);
    }

    #[test]
    fn the_helper_lists_each_real_adapter_as_a_directml_device() {
        let adapters = vec![
            adapter(0, "AMD Radeon(TM) Graphics", true, 512),
            adapter(1, "NVIDIA GeForce RTX 5070 Ti", false, 16303),
            software(2),
        ];
        let devices = devices(&adapters);
        assert_eq!(devices.len(), 2, "the software renderer is not listed");
        assert_eq!(
            super::devices(&[basic_display_adapter(0)]).len(),
            0,
            "a Microsoft adapter is not listed"
        );
        assert_eq!(
            (
                devices[0].kind,
                devices[0].index,
                devices[0].backend.as_str()
            ),
            (DeviceKind::Igpu, 0, "DirectML")
        );
        assert_eq!(
            (
                devices[1].kind,
                devices[1].description.as_str(),
                devices[1].memory_mb
            ),
            (DeviceKind::Gpu, "NVIDIA GeForce RTX 5070 Ti", 16303)
        );
        // The protocol's rule agrees with this one.
        let list = voice_engine_protocol::gpu::DeviceList {
            devices,
            chosen: None,
        };
        assert_eq!(list.discrete_card().unwrap().index, 1);
    }

    #[test]
    fn a_build_without_directml_lists_no_adapters() {
        if !cfg!(all(windows, feature = "directml")) {
            assert!(adapters().is_empty());
            assert_eq!(card(), None);
        }
    }
}
