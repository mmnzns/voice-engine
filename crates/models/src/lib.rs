//! What each model is (the catalog), where each app kept its models before the shared folder (`legacy`), finding a
//! model on this computer (`store`, removed in step 4's last task), the shared model folder every app uses
//! (`shared`), and the one-time move into it (`moving`).

pub mod catalog;
pub mod download;
pub mod imports;
pub mod legacy;
pub mod moving;
pub mod shared;
pub mod store;

pub use catalog::{GpuSupport, Kind, Licence, LlmInfo, ModelFile, ModelInfo};
pub use download::{DownloadError, Progress};
pub use legacy::{App, Layout, LegacyEntry, LegacySource, OtherRoot};
pub use moving::{ImportedGguf, MoveHooks, MoveReport};
pub use shared::{ModelHandle, Released, SharedStore, Status, StoreError};
pub use store::{Location, ModelStore};
