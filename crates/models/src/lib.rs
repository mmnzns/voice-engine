//! What each model is (the catalog), the downloader (`download`), the shared model folder every app uses
//! (`shared`), where each app kept its models before it (`legacy`), the one-time move into it (`moving`) and ids
//! for GGUF files the person brought (`imports`).

pub mod catalog;
pub mod download;
pub mod imports;
pub mod legacy;
pub mod moving;
pub mod shared;

pub use catalog::{GpuSupport, Kind, Licence, LlmInfo, ModelFile, ModelInfo};
pub use download::{DownloadError, Progress};
pub use legacy::{App, LegacyEntry, LegacySource};
pub use moving::{ImportedGguf, MoveHooks, MoveReport};
pub use shared::{ModelHandle, Released, Removed, SharedStore, Status, StoreError};
