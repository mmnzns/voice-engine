//! Where each app kept its models before the shared folder, the one place that says so: `LegacySource::all` lists an
//! app's old folders and what is in them (spec 5.4), and the one-time move (`moving`) reads it. Talaan's and
//! Ensayo's paths were read from their code on 2026-10-28 (Talaan `transcriber/models.rs`, Ensayo `paths.rs` and
//! `ai/local_models.rs`). Parlato's come from `docs/data/parlato-0.14.1-model-layout.md`; no Parlato code was read.

use crate::catalog::{ModelFile, ModelInfo};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum App {
    Talaan,
    Ensayo,
    Parlato,
}

/// One model an app kept in an old folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyEntry {
    /// The catalog id of the model.
    pub id: &'static str,
    /// The folder, relative to the source's root, that holds the files (always a folder, even for a single file).
    pub rel: &'static str,
    /// `(name on disk, catalog file name)`. Empty: the folder holds every catalog file under its catalog name.
    pub files: &'static [(&'static str, &'static str)],
    /// The old app never checked a hash (Parlato's Parakeet family), so every file's sha256 is checked before the
    /// model is adopted.
    pub verify_hashes: bool,
}

impl LegacyEntry {
    /// `(name on disk, catalog file)` for every file of `model`, in the catalog's order, or None when this entry
    /// does not account for every catalog file (or names one the model does not have).
    pub fn disk_files<'a>(&self, model: &'a ModelInfo) -> Option<Vec<(&'a str, &'a ModelFile)>> {
        if self.files.is_empty() {
            return Some(model.files.iter().map(|f| (f.name, f)).collect());
        }
        if self
            .files
            .iter()
            .any(|(_, name)| !model.files.iter().any(|f| f.name == *name))
        {
            return None;
        }
        model
            .files
            .iter()
            .map(|f| {
                self.files
                    .iter()
                    .find(|(_, name)| *name == f.name)
                    .map(|(disk, _)| (*disk, f))
            })
            .collect()
    }
}

/// One old folder of one app, with the models it may hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacySource {
    pub app: App,
    pub root: PathBuf,
    pub entries: Vec<LegacyEntry>,
    /// A folder under `root` whose `*.gguf` files that no entry claims are the person's own models (Parlato's
    /// dropped local AI models): the move registers them as imports and never touches them.
    pub imports_rel: Option<&'static str>,
}

impl LegacySource {
    /// Every old folder of `app`. `local_data` is `%LOCALAPPDATA%` on Windows and `~/Library/Application Support` on
    /// the Mac, `roaming` is `%APPDATA%` (only Talaan's oldest Windows folder is there).
    pub fn all(
        app: App,
        local_data: &Path,
        roaming: Option<&Path>,
        windows: bool,
    ) -> Vec<LegacySource> {
        let source = |root: PathBuf, entries: Vec<LegacyEntry>, imports_rel| LegacySource {
            app,
            root,
            entries,
            imports_rel,
        };
        match app {
            App::Talaan => {
                // Mac: `Talaan/models`. Windows: `<bundle id>/models` (1.1 and later) and, when it still exists,
                // the older Roaming `Talaan/models`; Talaan's own move can leave models in both.
                let mut out = Vec::new();
                if windows {
                    out.push(source(
                        local_data.join(app.bundle_id()).join("models"),
                        by_id(TALAAN_IDS),
                        None,
                    ));
                    if let Some(roaming) = roaming {
                        out.push(source(
                            roaming.join("Talaan").join("models"),
                            by_id(TALAAN_IDS),
                            None,
                        ));
                    }
                } else {
                    out.push(source(
                        local_data.join("Talaan").join("models"),
                        by_id(TALAAN_IDS),
                        None,
                    ));
                }
                out
            }
            App::Ensayo => {
                let models = local_data.join(app.bundle_id()).join("models");
                vec![
                    source(models.clone(), by_id(ENSAYO_SPEECH_IDS), None),
                    source(models.join("llm"), by_id(ENSAYO_AI_IDS), None),
                ]
            }
            App::Parlato => vec![source(
                local_data.join(app.bundle_id()),
                parlato_entries(),
                Some("LlmModels"),
            )],
        }
    }

    /// `all` for this computer's folders.
    pub fn on_this_computer(app: App) -> Vec<LegacySource> {
        match dirs::data_local_dir() {
            Some(local) => LegacySource::all(
                app,
                &local,
                dirs::data_dir().as_deref(),
                cfg!(target_os = "windows"),
            ),
            None => Vec::new(),
        }
    }
}

/// Models in an app's old folder as `<id>/<catalog file names>`: no renames, every hash checked at download.
fn by_id(ids: &'static [&'static str]) -> Vec<LegacyEntry> {
    ids.iter()
        .map(|id| LegacyEntry {
            id,
            rel: id,
            files: &[],
            verify_hashes: false,
        })
        .collect()
}

/// Ids Talaan kept in `models/<id>/` (its own catalog): speech, voice detection, speaker separation, search and
/// the two local AI models.
const TALAAN_IDS: &[&str] = &[
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large-v2",
    "whisper-large-v3",
    "whisper-turbo",
    "parakeet-v2",
    "parakeet-v3",
    "parakeet-ultra",
    "parakeet-unified-en",
    "nemotron-3.5",
    "vad",
    "diarization",
    "search",
    "qwen3.5-4b",
    "qwen3.5-9b",
];

/// Ids Ensayo kept in `models/<id>/` (same file names as Talaan).
const ENSAYO_SPEECH_IDS: &[&str] = &[
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large-v2",
    "whisper-large-v3",
    "whisper-turbo",
    "parakeet-v2",
    "parakeet-v3",
    "parakeet-ultra",
    "parakeet-unified-en",
    "nemotron-3.5",
    "vad",
];

/// Ids Ensayo kept in `models/llm/<id>/`.
const ENSAYO_AI_IDS: &[&str] = &["qwen3.5-4b", "qwen3.5-9b"];

/// A Parlato single-file model in `Models/` (the file name on disk is the catalog's).
macro_rules! whisper_file {
    ($id:expr, $file:expr) => {
        LegacyEntry {
            id: $id,
            rel: "Models",
            files: &[($file, $file)],
            verify_hashes: false,
        }
    };
}

/// A Parlato Parakeet-family folder: the files have the catalog's names, and Parlato never checked a hash.
fn parakeet_folder(id: &'static str, rel: &'static str) -> LegacyEntry {
    LegacyEntry {
        id,
        rel,
        files: &[],
        verify_hashes: true,
    }
}

/// Parlato 0.14.1's layout, from `docs/data/parlato-0.14.1-model-layout.md` (no Parlato code was read).
fn parlato_entries() -> Vec<LegacyEntry> {
    vec![
        whisper_file!("whisper-tiny", "ggml-tiny.bin"),
        whisper_file!("whisper-tiny-en", "ggml-tiny.en.bin"),
        whisper_file!("whisper-base", "ggml-base.bin"),
        whisper_file!("whisper-base-en", "ggml-base.en.bin"),
        whisper_file!("whisper-small", "ggml-small.bin"),
        whisper_file!("whisper-small-en", "ggml-small.en.bin"),
        whisper_file!("whisper-medium", "ggml-medium.bin"),
        whisper_file!("whisper-medium-en", "ggml-medium.en.bin"),
        whisper_file!("whisper-large-v2-full", "ggml-large-v2.bin"),
        whisper_file!("whisper-large-v3-full", "ggml-large-v3.bin"),
        whisper_file!("whisper-large-v3", "ggml-large-v3-q5_0.bin"),
        whisper_file!("whisper-turbo-full", "ggml-large-v3-turbo.bin"),
        whisper_file!("whisper-turbo", "ggml-large-v3-turbo-q5_0.bin"),
        whisper_file!("vad", "ggml-silero-v6.2.0.bin"),
        parakeet_folder("parakeet-v2-full", "ParakeetModels/parakeet-tdt-0.6b-v2"),
        parakeet_folder("parakeet-v2", "ParakeetModels/parakeet-tdt-0.6b-v2-int8"),
        parakeet_folder("parakeet-v3-full", "ParakeetModels/parakeet-tdt-0.6b-v3"),
        parakeet_folder("parakeet-v3", "ParakeetModels/parakeet-tdt-0.6b-v3-int8"),
        parakeet_folder("parakeet-ultra", "ParakeetModels/parakeet-ultra-0.6b-int8"),
        parakeet_folder(
            "parakeet-unified-en-full",
            "ParakeetModels/parakeet-unified-en-0.6b",
        ),
        parakeet_folder(
            "parakeet-unified-en",
            "ParakeetModels/parakeet-unified-en-0.6b-int8",
        ),
        parakeet_folder(
            "nemotron-3.5",
            "ParakeetModels/nemotron-3.5-asr-streaming-0.6b",
        ),
        // Parlato names local AI files after its own id, not the repository's file.
        LegacyEntry {
            id: "granite-4.2-3b",
            rel: "LlmModels",
            files: &[("granite-4.2-3b-q4.gguf", "granite-4.2-3b-Q4_K_M.gguf")],
            verify_hashes: false,
        },
    ]
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
}
