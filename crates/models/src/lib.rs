//! What each model is (the catalogs), where each app keeps its models (`apps`), and finding a model on this computer
//! (`store`).

pub mod apps;
pub mod catalog;
pub mod llm_catalog;

pub use apps::{App, Layout, OtherRoot};
pub use catalog::{EngineKind, ModelFile, ModelInfo};
