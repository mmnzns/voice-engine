//! What each model is (the catalog), where each app keeps its models (`apps`), and finding a model on this computer
//! (`store`), and the shared model folder every app uses (`shared`).

pub mod apps;
pub mod catalog;
pub mod download;
pub mod shared;
pub mod store;

pub use apps::{App, Layout, OtherRoot};
pub use catalog::{GpuSupport, Kind, Licence, LlmInfo, ModelFile, ModelInfo};
pub use download::{DownloadError, Progress};
pub use shared::{ModelHandle, Released, SharedStore, Status, StoreError};
pub use store::{Location, ModelStore};
