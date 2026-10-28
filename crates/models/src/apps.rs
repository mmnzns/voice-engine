//! Where each app keeps its models, the one place that says so. Paths were read from each app's code on 2026-10-28
//! (Talaan `transcriber/models.rs`, Ensayo `paths.rs` and `ai/local_models.rs`, Parlato's model managers, which use
//! Tauri's `app_local_data_dir`). Only paths were read from Parlato; no code.

use crate::catalog::{EngineKind, ModelInfo};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum App {
    Talaan,
    Ensayo,
    Parlato,
}

/// How an app lays out model files under its root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `<root>/<model id>/<files>` (Talaan, Ensayo).
    ById,
    /// `<root>/<each of ModelInfo::reuse_dirs>/<files>` (Parlato).
    ReuseDirs,
}

/// One app's models folder for one kind of model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtherRoot {
    pub app: App,
    pub root: PathBuf,
    pub layout: Layout,
}

impl OtherRoot {
    /// The folders under this root that may hold `m`'s files.
    pub fn candidates(&self, m: &ModelInfo) -> Vec<PathBuf> {
        match self.layout {
            Layout::ById => vec![self.root.join(m.id)],
            Layout::ReuseDirs => m.reuse_dirs.iter().map(|d| self.root.join(d)).collect(),
        }
    }
}

impl App {
    pub const ALL: [App; 3] = [App::Talaan, App::Ensayo, App::Parlato];

    pub fn bundle_id(self) -> &'static str {
        match self {
            App::Talaan => "com.craftconceptsdigital.talaan",
            App::Ensayo => "com.craftconceptsdigital.ensayo",
            App::Parlato => "com.craftconceptsdigital.parlato",
        }
    }

    /// Where this app keeps models of `engine`, given the local app data folder (`%LOCALAPPDATA%` on Windows,
    /// `~/Library/Application Support` on the Mac). Pure, so tests can check every platform.
    pub fn root_in(self, engine: EngineKind, local_data: &Path, windows: bool) -> OtherRoot {
        let (root, layout) = match self {
            App::Talaan if windows => (
                local_data.join(self.bundle_id()).join("models"),
                Layout::ById,
            ),
            App::Talaan => (local_data.join("Talaan").join("models"), Layout::ById),
            App::Ensayo => {
                let models = local_data.join(self.bundle_id()).join("models");
                (
                    if engine == EngineKind::Llm {
                        models.join("llm")
                    } else {
                        models
                    },
                    Layout::ById,
                )
            }
            App::Parlato => (local_data.join(self.bundle_id()), Layout::ReuseDirs),
        };
        OtherRoot {
            app: self,
            root,
            layout,
        }
    }

    /// This app's folder for `engine` on this computer. None when the system has no local data folder.
    pub fn root(self, engine: EngineKind) -> Option<OtherRoot> {
        dirs::data_local_dir().map(|d| self.root_in(engine, &d, cfg!(target_os = "windows")))
    }

    /// The other two apps' folders, given the local app data folder. Pure, for tests.
    pub fn others_in(
        me: App,
        engine: EngineKind,
        local_data: &Path,
        windows: bool,
    ) -> Vec<OtherRoot> {
        App::ALL
            .into_iter()
            .filter(|a| *a != me)
            .map(|a| a.root_in(engine, local_data, windows))
            .collect()
    }

    /// The other two apps' folders on this computer (Phase 4 turns this on for every app).
    pub fn others(me: App, engine: EngineKind) -> Vec<OtherRoot> {
        dirs::data_local_dir()
            .map(|d| App::others_in(me, engine, &d, cfg!(target_os = "windows")))
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{catalog, llm_catalog};
    use std::path::Path;

    const WIN: &str = r"C:\Users\a\AppData\Local";
    const MAC: &str = "/Users/a/Library/Application Support";

    fn whisper() -> &'static ModelInfo {
        catalog::find("whisper-turbo").unwrap()
    }
    fn qwen() -> &'static ModelInfo {
        llm_catalog::find("qwen3.5-4b").unwrap()
    }

    #[test]
    fn talaan_keeps_models_in_its_bundle_id_folder_on_windows_and_its_name_on_the_mac() {
        let w = App::Talaan.root_in(EngineKind::Whisper, Path::new(WIN), true);
        assert_eq!(
            w.root,
            Path::new(WIN)
                .join("com.craftconceptsdigital.talaan")
                .join("models")
        );
        assert_eq!(w.layout, Layout::ById);
        let m = App::Talaan.root_in(EngineKind::Whisper, Path::new(MAC), false);
        assert_eq!(m.root, Path::new(MAC).join("Talaan").join("models"));
        // Talaan keeps its AI models next to the speech ones.
        assert_eq!(
            App::Talaan
                .root_in(EngineKind::Llm, Path::new(WIN), true)
                .root,
            w.root
        );
    }

    #[test]
    fn ensayo_keeps_ai_models_in_an_llm_subfolder() {
        let s = App::Ensayo.root_in(EngineKind::Whisper, Path::new(WIN), true);
        assert_eq!(
            s.root,
            Path::new(WIN)
                .join("com.craftconceptsdigital.ensayo")
                .join("models")
        );
        let l = App::Ensayo.root_in(EngineKind::Llm, Path::new(MAC), false);
        assert_eq!(
            l.root,
            Path::new(MAC)
                .join("com.craftconceptsdigital.ensayo")
                .join("models")
                .join("llm")
        );
    }

    #[test]
    fn parlato_uses_its_reuse_dirs_under_local_app_data_not_roaming() {
        let p = App::Parlato.root_in(EngineKind::Whisper, Path::new(WIN), true);
        assert_eq!(
            p.root,
            Path::new(WIN).join("com.craftconceptsdigital.parlato")
        );
        assert_eq!(p.layout, Layout::ReuseDirs);
        assert_eq!(p.candidates(whisper()), vec![p.root.join("Models")]);
    }

    #[test]
    fn by_id_layout_looks_in_the_model_id_folder() {
        let t = App::Talaan.root_in(EngineKind::Llm, Path::new(WIN), true);
        assert_eq!(t.candidates(qwen()), vec![t.root.join("qwen3.5-4b")]);
    }

    #[test]
    fn others_never_includes_the_app_itself() {
        for me in App::ALL {
            let others = App::others_in(me, EngineKind::Whisper, Path::new(WIN), true);
            assert_eq!(others.len(), 2);
            assert!(others.iter().all(|o| o.app != me));
        }
    }
}
