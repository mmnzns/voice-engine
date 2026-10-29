//! The old-folder table (`legacy`), checked against the real catalog so a typo in an id or a file name cannot pass.
//! Parlato's rows below are transcribed by hand from `docs/data/parlato-0.14.1-model-layout.md`, independently of
//! the table in `src/legacy.rs`: the two must agree, and every one must agree with the catalog.

use std::collections::BTreeSet;
use std::path::Path;
use voice_engine_models::{App, LegacyEntry, LegacySource, ModelInfo, catalog};

const WIN_LOCAL: &str = r"C:\Users\a\AppData\Local";
const WIN_ROAMING: &str = r"C:\Users\a\AppData\Roaming";
const MAC: &str = "/Users/a/Library/Application Support";

/// One line of Parlato's layout table.
struct Row {
    rel: &'static str,
    id: &'static str,
    /// `(name on disk, catalog file name)`.
    files: &'static [(&'static str, &'static str)],
    /// The exact total the document gives, when it gives one (not the rounded "≈" figures).
    size: Option<u64>,
    verify: bool,
}

const fn same(name: &'static str) -> (&'static str, &'static str) {
    (name, name)
}

const V3_FILES: &[(&str, &str)] = &[
    same("config.json"),
    same("vocab.txt"),
    same("nemo128.onnx"),
    same("encoder-model.onnx"),
    same("encoder-model.onnx.data"),
    same("decoder_joint-model.onnx"),
];
const V3_INT8_FILES: &[(&str, &str)] = &[
    same("config.json"),
    same("vocab.txt"),
    same("nemo128.onnx"),
    same("encoder-model.int8.onnx"),
    same("decoder_joint-model.int8.onnx"),
];
const UNIFIED_FILES: &[(&str, &str)] = &[
    same("tokenizer.model"),
    same("encoder.onnx"),
    same("encoder.onnx.data"),
    same("decoder_joint.onnx"),
];
const UNIFIED_INT8_FILES: &[(&str, &str)] = &[
    same("tokenizer.model"),
    same("encoder.int8.onnx"),
    same("encoder.int8.onnx.data"),
    same("decoder_joint.int8.onnx"),
];

// A `const fn` cannot build a one-element slice from a parameter, so the single-file rows use a macro.
macro_rules! single {
    ($id:expr, $file:expr, $size:expr) => {
        Row {
            rel: "Models",
            id: $id,
            files: &[($file, $file)],
            size: $size,
            verify: false,
        }
    };
}

fn parlato_rows() -> Vec<Row> {
    vec![
        single!("whisper-tiny", "ggml-tiny.bin", None),
        single!("whisper-tiny-en", "ggml-tiny.en.bin", None),
        single!("whisper-base", "ggml-base.bin", None),
        single!("whisper-base-en", "ggml-base.en.bin", None),
        single!("whisper-small", "ggml-small.bin", Some(487_601_967)),
        single!("whisper-small-en", "ggml-small.en.bin", Some(487_614_201)),
        single!("whisper-medium", "ggml-medium.bin", Some(1_533_763_059)),
        single!(
            "whisper-medium-en",
            "ggml-medium.en.bin",
            Some(1_533_774_781)
        ),
        single!("whisper-large-v2-full", "ggml-large-v2.bin", None),
        single!("whisper-large-v3-full", "ggml-large-v3.bin", None),
        single!(
            "whisper-large-v3",
            "ggml-large-v3-q5_0.bin",
            Some(1_081_140_203)
        ),
        single!("whisper-turbo-full", "ggml-large-v3-turbo.bin", None),
        single!("whisper-turbo", "ggml-large-v3-turbo-q5_0.bin", None),
        single!("vad", "ggml-silero-v6.2.0.bin", Some(885_098)),
        Row {
            rel: "ParakeetModels/parakeet-tdt-0.6b-v2",
            id: "parakeet-v2-full",
            files: V3_FILES,
            size: None,
            verify: true,
        },
        Row {
            rel: "ParakeetModels/parakeet-tdt-0.6b-v2-int8",
            id: "parakeet-v2",
            files: V3_INT8_FILES,
            size: None,
            verify: true,
        },
        Row {
            rel: "ParakeetModels/parakeet-tdt-0.6b-v3",
            id: "parakeet-v3-full",
            files: V3_FILES,
            size: None,
            verify: true,
        },
        Row {
            rel: "ParakeetModels/parakeet-tdt-0.6b-v3-int8",
            id: "parakeet-v3",
            files: V3_INT8_FILES,
            size: None,
            verify: true,
        },
        Row {
            rel: "ParakeetModels/parakeet-ultra-0.6b-int8",
            id: "parakeet-ultra",
            files: V3_INT8_FILES,
            size: Some(667_961_292),
            verify: true,
        },
        Row {
            rel: "ParakeetModels/parakeet-unified-en-0.6b",
            id: "parakeet-unified-en-full",
            files: UNIFIED_FILES,
            size: Some(2_515_023_099),
            verify: true,
        },
        Row {
            rel: "ParakeetModels/parakeet-unified-en-0.6b-int8",
            id: "parakeet-unified-en",
            files: UNIFIED_INT8_FILES,
            size: Some(663_344_373),
            verify: true,
        },
        Row {
            rel: "ParakeetModels/nemotron-3.5-asr-streaming-0.6b",
            id: "nemotron-3.5",
            files: UNIFIED_FILES,
            size: Some(2_594_566_700),
            verify: true,
        },
        Row {
            rel: "LlmModels",
            id: "granite-4.2-3b",
            files: &[("granite-4.2-3b-q4.gguf", "granite-4.2-3b-Q4_K_M.gguf")],
            size: Some(2_244_011_552),
            verify: false,
        },
    ]
}

const SHORT: &[(&str, &str)] = &[same("config.json"), same("vocab.txt"), same("nemo128.onnx")];

fn model(id: &str) -> &'static ModelInfo {
    catalog::find(id).unwrap_or_else(|| panic!("{id} is not a catalog id"))
}

/// What the entry says is on disk, as a sorted list of `(disk name, catalog name)`.
fn pairs(entry: &LegacyEntry) -> Vec<(String, String)> {
    let model = model(entry.id);
    let mut v: Vec<(String, String)> = entry
        .disk_files(model)
        .unwrap_or_else(|| panic!("{} does not account for every catalog file", entry.id))
        .into_iter()
        .map(|(disk, f)| (disk.to_string(), f.name.to_string()))
        .collect();
    v.sort();
    v
}

fn all_sources() -> Vec<LegacySource> {
    let mut out = Vec::new();
    for app in App::ALL {
        for windows in [false, true] {
            out.extend(LegacySource::all(
                app,
                Path::new(if windows { WIN_LOCAL } else { MAC }),
                Some(Path::new(WIN_ROAMING)),
                windows,
            ));
        }
    }
    out
}

#[test]
fn parlato_table_matches_the_documented_layout() {
    for windows in [false, true] {
        let local = Path::new(if windows { WIN_LOCAL } else { MAC });
        let sources = LegacySource::all(App::Parlato, local, Some(Path::new(WIN_ROAMING)), windows);
        assert_eq!(
            sources.len(),
            1,
            "Parlato has one folder (Local, not Roaming)"
        );
        let src = &sources[0];
        assert_eq!(src.app, App::Parlato);
        assert_eq!(src.root, local.join("com.craftconceptsdigital.parlato"));
        assert_eq!(src.imports_rel, Some("LlmModels"));

        let rows = parlato_rows();
        assert_eq!(
            src.entries.len(),
            rows.len(),
            "one entry for each row of the document"
        );
        for row in &rows {
            let entry = src
                .entries
                .iter()
                .find(|e| e.id == row.id)
                .unwrap_or_else(|| panic!("no entry for {}", row.id));
            assert_eq!(entry.rel, row.rel, "{}: folder", row.id);
            assert_eq!(entry.verify_hashes, row.verify, "{}: verify_hashes", row.id);

            // The document's file list, the entry and the catalog all name the same files.
            let mut documented: Vec<(String, String)> = row
                .files
                .iter()
                .map(|(d, c)| (d.to_string(), c.to_string()))
                .collect();
            documented.sort();
            assert_eq!(pairs(entry), documented, "{}: file names", row.id);
            let catalog_names: BTreeSet<&str> =
                model(row.id).files.iter().map(|f| f.name).collect();
            let listed: BTreeSet<&str> = row.files.iter().map(|(_, c)| *c).collect();
            assert_eq!(
                listed, catalog_names,
                "{}: every catalog file is on the document's list, and the other way round",
                row.id
            );
            if let Some(size) = row.size {
                assert_eq!(model(row.id).size(), size, "{}: total size", row.id);
            }
        }
    }
}

#[test]
fn parlato_ids_and_files_are_not_listed_twice() {
    let sources = LegacySource::all(App::Parlato, Path::new(MAC), None, false);
    let entries = &sources[0].entries;
    let ids: BTreeSet<&str> = entries.iter().map(|e| e.id).collect();
    assert_eq!(ids.len(), entries.len(), "an id appears twice");
    let mut disk_paths = BTreeSet::new();
    for e in entries {
        for (disk, _) in pairs(e) {
            assert!(
                disk_paths.insert(format!("{}/{disk}", e.rel)),
                "{}/{disk} belongs to two models",
                e.rel
            );
        }
    }
}

#[test]
fn only_the_parakeet_family_is_hash_checked() {
    for entry in &LegacySource::all(App::Parlato, Path::new(MAC), None, false)[0].entries {
        let is_family = entry.rel.starts_with("ParakeetModels/");
        assert_eq!(entry.verify_hashes, is_family, "{}", entry.id);
    }
    // Talaan and Ensayo checked every hash when they downloaded.
    for app in [App::Talaan, App::Ensayo] {
        for src in LegacySource::all(app, Path::new(MAC), None, false) {
            assert!(src.entries.iter().all(|e| !e.verify_hashes));
        }
    }
}

#[test]
fn every_entry_of_every_app_names_a_catalog_model_and_all_its_files() {
    let sources = all_sources();
    assert!(!sources.is_empty());
    for src in &sources {
        let mut seen = BTreeSet::new();
        for e in &src.entries {
            assert!(
                catalog::find(e.id).is_some(),
                "{} is not in the catalog",
                e.id
            );
            assert!(seen.insert(e.id), "{} twice in one folder", e.id);
            assert!(
                !e.rel.is_empty() && !e.rel.starts_with('/') && !e.rel.contains(".."),
                "{}: {} is not a folder inside the app's folder",
                e.id,
                e.rel
            );
            // Every catalog file is accounted for, and nothing else is named.
            let names: BTreeSet<String> = pairs(e).into_iter().map(|(_, c)| c).collect();
            let wanted: BTreeSet<String> = model(e.id)
                .files
                .iter()
                .map(|f| f.name.to_string())
                .collect();
            assert_eq!(names, wanted, "{}", e.id);
        }
    }
}

#[test]
fn an_entry_that_leaves_a_catalog_file_out_or_names_a_stranger_does_not_cover_the_model() {
    // parakeet-v3 has five files; this entry lists three.
    let short = LegacyEntry {
        id: "parakeet-v3",
        rel: "x",
        files: SHORT,
        verify_hashes: false,
    };
    assert!(short.disk_files(model("parakeet-v3")).is_none());
    let stranger = LegacyEntry {
        id: "vad",
        rel: "x",
        files: &[("a.bin", "not-a-catalog-name.bin")],
        verify_hashes: false,
    };
    assert!(stranger.disk_files(model("vad")).is_none());
    let folder = LegacyEntry {
        id: "vad",
        rel: "x",
        files: &[],
        verify_hashes: false,
    };
    assert_eq!(
        folder.disk_files(model("vad")).unwrap()[0].0,
        "ggml-silero-v6.2.0.bin"
    );
}

/// Ids Talaan used (its own catalog, read on 2026-10-28): speech, voice detection, speaker separation, search and
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

fn ids(src: &LegacySource) -> BTreeSet<&'static str> {
    src.entries.iter().map(|e| e.id).collect()
}

#[test]
fn talaan_on_the_mac_has_one_folder_named_after_the_app() {
    let s = LegacySource::all(
        App::Talaan,
        Path::new(MAC),
        Some(Path::new(WIN_ROAMING)),
        false,
    );
    assert_eq!(s.len(), 1, "the Mac has no Roaming folder");
    assert_eq!(s[0].root, Path::new(MAC).join("Talaan").join("models"));
    assert_eq!(ids(&s[0]), TALAAN_IDS.iter().copied().collect());
    assert!(
        s[0].entries
            .iter()
            .all(|e| e.rel == e.id && e.files.is_empty())
    );
    assert_eq!(s[0].imports_rel, None);
}

#[test]
fn talaan_on_windows_has_two_folders_and_the_roaming_one_only_when_there_is_one() {
    let both = LegacySource::all(
        App::Talaan,
        Path::new(WIN_LOCAL),
        Some(Path::new(WIN_ROAMING)),
        true,
    );
    assert_eq!(both.len(), 2);
    assert_eq!(
        both[0].root,
        Path::new(WIN_LOCAL)
            .join("com.craftconceptsdigital.talaan")
            .join("models")
    );
    assert_eq!(
        both[1].root,
        Path::new(WIN_ROAMING).join("Talaan").join("models")
    );
    for s in &both {
        assert_eq!(s.app, App::Talaan);
        assert_eq!(ids(s), TALAAN_IDS.iter().copied().collect());
    }
    let local_only = LegacySource::all(App::Talaan, Path::new(WIN_LOCAL), None, true);
    assert_eq!(local_only.len(), 1);
    assert_eq!(local_only[0].root, both[0].root);
}

#[test]
fn ensayo_keeps_speech_in_models_and_the_ai_models_in_models_llm() {
    for windows in [false, true] {
        let local = Path::new(if windows { WIN_LOCAL } else { MAC });
        let s = LegacySource::all(App::Ensayo, local, Some(Path::new(WIN_ROAMING)), windows);
        assert_eq!(s.len(), 2, "no Roaming folder for Ensayo");
        let models = local.join("com.craftconceptsdigital.ensayo").join("models");
        assert_eq!(s[0].root, models);
        assert_eq!(s[1].root, models.join("llm"));
        let speech: BTreeSet<&str> = [
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
        ]
        .into_iter()
        .collect();
        assert_eq!(ids(&s[0]), speech);
        assert_eq!(
            ids(&s[1]),
            ["qwen3.5-4b", "qwen3.5-9b"].into_iter().collect()
        );
        assert!(
            s.iter()
                .all(|x| x.app == App::Ensayo && x.imports_rel.is_none())
        );
    }
}
