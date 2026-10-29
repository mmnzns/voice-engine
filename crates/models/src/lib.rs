//! What each model is (the catalog), where each app keeps its models (`apps`), and finding a model on this computer
//! (`store`).

pub mod apps;
pub mod catalog;
pub mod store;

pub use apps::{App, Layout, OtherRoot};
pub use catalog::{GpuSupport, Kind, Licence, LlmInfo, ModelFile, ModelInfo};
pub use store::{Location, ModelStore};
