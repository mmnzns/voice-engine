//! The one-time move (`SharedStore::move_in`) on temporary folders only. Each test builds the apps' old folders in a
//! temp "local app data" folder (zero-filled files of the catalog's exact sizes; sparse, so the big models cost
//! nothing) and a shared folder next to it. No test touches the real shared model folder, any app's real folder or
//! the network. Tests that need real hashes use a locally built `ModelInfo` through `move_in_with`.

use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use voice_engine_models::imports::import_id;
use voice_engine_models::moving::{ImportedGguf, MoveHooks, MoveReport};
use voice_engine_models::{
    App, LegacyEntry, LegacySource, ModelFile, ModelInfo, SharedStore, Status, catalog,
};

// ---- the temporary world ------------------------------------------------------------------------------------

/// A temporary "local app data", "roaming" and shared folder. Panics if any of them is inside a real folder.
struct World {
    _tmp: tempfile::TempDir,
    local: PathBuf,
    roaming: PathBuf,
    shared: PathBuf,
}

impl World {
    fn new() -> World {
        let tmp = tempfile::tempdir().unwrap();
        let world = World {
            local: tmp.path().join("local"),
            roaming: tmp.path().join("roaming"),
            shared: tmp.path().join("shared"),
            _tmp: tmp,
        };
        for dir in [&world.local, &world.roaming, &world.shared] {
            fs::create_dir_all(dir).unwrap();
            assert_not_real(dir);
        }
        world
    }

    /// `app`'s store on the Mac layout.
    fn store(&self, app: App) -> SharedStore {
        self.store_on(app, false, true)
    }

    fn store_on(&self, app: App, windows: bool, with_roaming: bool) -> SharedStore {
        assert_not_real(&self.shared);
        SharedStore::for_move_test(
            app,
            &self.shared,
            &self.local,
            with_roaming.then_some(self.roaming.as_path()),
            windows,
        )
        .unwrap()
    }

    // Old folders, Mac layout.
    fn talaan(&self) -> PathBuf {
        self.local.join("Talaan").join("models")
    }
    fn ensayo(&self) -> PathBuf {
        self.local
            .join("com.craftconceptsdigital.ensayo")
            .join("models")
    }
    fn parlato(&self) -> PathBuf {
        self.local.join("com.craftconceptsdigital.parlato")
    }
    // Windows layout.
    fn talaan_windows(&self) -> PathBuf {
        self.local
            .join("com.craftconceptsdigital.talaan")
            .join("models")
    }
    fn talaan_roaming(&self) -> PathBuf {
        self.roaming.join("Talaan").join("models")
    }

    fn moved_marker(&self, app: &str) -> PathBuf {
        self.shared.join(".engine/moved").join(app)
    }

    /// Every file name under `.engine/users/<id>/`, sorted.
    fn users(&self, id: &str) -> Vec<String> {
        sorted_names(&self.shared.join(".engine/users").join(id))
    }

    /// The model folders in the shared folder (everything but `.engine`), sorted.
    fn model_folders(&self) -> Vec<String> {
        sorted_names(&self.shared)
            .into_iter()
            .filter(|n| n != ".engine")
            .collect()
    }
}

fn assert_not_real(dir: &Path) {
    let real: Vec<PathBuf> = [SharedStore::default_dir(), dirs::data_local_dir()]
        .into_iter()
        .flatten()
        .collect();
    for real in real {
        assert!(
            !dir.starts_with(&real),
            "a test must never use a real folder: {}",
            dir.display()
        );
    }
}

fn sorted_names(dir: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn sorted(v: &[String]) -> Vec<String> {
    let mut v = v.to_vec();
    v.sort();
    v
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

// ---- files ---------------------------------------------------------------------------------------------------

/// A zero-filled (sparse) file of `size` bytes whose first bytes are `prefix`.
fn put_file(path: &Path, size: u64, prefix: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut file = File::create(path).unwrap();
    file.set_len(size).unwrap();
    if !prefix.is_empty() {
        file.write_all(prefix).unwrap();
    }
}

fn model(id: &str) -> &'static ModelInfo {
    catalog::find(id).unwrap_or_else(|| panic!("{id} is not a catalog id"))
}

/// Every file of catalog model `id`, under its catalog name, in `folder`.
fn put_model(folder: &Path, id: &str) {
    for f in model(id).files {
        put_file(&folder.join(f.name), f.size, b"");
    }
}

fn real_catalog() -> Vec<&'static ModelInfo> {
    catalog::all().iter().collect()
}

fn first_bytes(path: &Path, n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    File::open(path).unwrap().read_exact(&mut buf).unwrap();
    buf
}

/// Whether two paths are one file (a hard link). Unix: the same device and inode. Elsewhere: a byte written through
/// one path shows through the other.
#[cfg(unix)]
fn same_file(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    let (a, b) = (fs::metadata(a).unwrap(), fs::metadata(b).unwrap());
    a.dev() == b.dev() && a.ino() == b.ino()
}

#[cfg(not(unix))]
fn same_file(a: &Path, b: &Path) -> bool {
    use std::io::{Seek, SeekFrom};
    let mut through_a = fs::OpenOptions::new().write(true).open(a).unwrap();
    let before = first_bytes(b, 1)[0];
    let probe = before.wrapping_add(1);
    through_a.seek(SeekFrom::Start(0)).unwrap();
    through_a.write_all(&[probe]).unwrap();
    through_a.flush().unwrap();
    let seen = first_bytes(b, 1)[0];
    through_a.seek(SeekFrom::Start(0)).unwrap();
    through_a.write_all(&[before]).unwrap();
    seen == probe
}

/// The file's identity where the platform gives one cheaply (Unix); None elsewhere.
fn file_id(path: &Path) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let m = fs::metadata(path).unwrap();
        Some((m.dev(), m.ino()))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

/// `id` is installed in the shared folder: every catalog file with its exact size (what `status` checks).
fn assert_installed(store: &SharedStore, id: &str) -> PathBuf {
    match store.status(id) {
        Status::Installed { dir, .. } => dir,
        other => panic!("{id}: expected Installed, got {other:?}"),
    }
}

/// Every catalog file of `id` in `old_folder` is the same file as the one in the shared folder.
fn assert_linked(old_folder: &Path, shared: &Path, id: &str) {
    for f in model(id).files {
        let old = old_folder.join(f.name);
        let new = shared.join(id).join(f.name);
        assert!(
            old.is_file(),
            "{} is gone from the old folder",
            old.display()
        );
        assert_eq!(fs::metadata(&old).unwrap().len(), f.size);
        assert!(same_file(&old, &new), "{} is not linked", old.display());
    }
}

// ---- local models with real hashes ---------------------------------------------------------------------------

fn data(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// A model that is not in the catalog: its files have these bodies, and the hashes are computed from them.
fn local_model(id: &'static str, files: &[(&str, &[u8])]) -> &'static ModelInfo {
    let files: Vec<ModelFile> = files
        .iter()
        .map(|(name, body)| ModelFile {
            name: leak((*name).into()),
            url: "http://invalid.test/never-fetched",
            size: body.len() as u64,
            sha256: leak(format!("{:x}", Sha256::digest(body))),
        })
        .collect();
    let template = model("vad");
    Box::leak(Box::new(ModelInfo {
        id,
        name: "local test model",
        maker: template.maker,
        note: template.note,
        kind: template.kind,
        languages: template.languages,
        english_only: false,
        speed: 1,
        accuracy: 1,
        licence: template.licence,
        files: Box::leak(files.into_boxed_slice()),
        gpu: template.gpu,
        keeps_fillers: None,
        chunk_secs: None,
        llm: None,
    }))
}

fn entry(id: &'static str, rel: &'static str, verify_hashes: bool) -> LegacyEntry {
    LegacyEntry {
        id,
        rel,
        files: &[],
        verify_hashes,
    }
}

fn source(app: App, root: &Path, entries: Vec<LegacyEntry>) -> LegacySource {
    LegacySource {
        app,
        root: root.to_path_buf(),
        entries,
        imports_rel: None,
    }
}

/// Writes `model`'s files into `root/<rel>/` with these exact bodies.
fn write_bodies(root: &Path, rel: &str, bodies: &[(&str, &[u8])]) {
    let dir = root.join(rel);
    fs::create_dir_all(&dir).unwrap();
    for (name, body) in bodies {
        fs::write(dir.join(name), body).unwrap();
    }
}

fn with_one_byte_changed(body: &[u8]) -> Vec<u8> {
    let mut b = body.to_vec();
    b[10] ^= 0xff;
    b
}

// ---- own folders -----------------------------------------------------------------------------------------------

#[test]
fn talaan_on_the_mac_moves_its_models_in_and_links_them_back() {
    let w = World::new();
    put_file(
        &w.talaan().join("vad/ggml-silero-v6.2.0.bin"),
        885_098,
        b"TALAAN-VAD",
    );
    put_model(&w.talaan().join("whisper-tiny"), "whisper-tiny");
    let store = w.store(App::Talaan);

    let report = store.move_in(&[]).unwrap();

    assert_eq!(sorted(&report.moved), strings(&["vad", "whisper-tiny"]));
    assert!(report.linked.is_empty() && report.left_for_later.is_empty());
    assert!(report.imported.is_empty() && !report.already_done);
    for id in ["vad", "whisper-tiny"] {
        let dir = assert_installed(&store, id);
        assert_eq!(dir, w.shared.join(id));
        assert_linked(&w.talaan().join(id), &w.shared, id);
        assert_eq!(w.users(id), strings(&["talaan"]));
        assert_eq!(
            store.status(id),
            Status::Installed {
                dir: w.shared.join(id),
                users: vec![App::Talaan],
                used_by_me: true
            }
        );
    }
    // The content came along, and the old path still reads it.
    assert_eq!(
        &first_bytes(&w.shared.join("vad/ggml-silero-v6.2.0.bin"), 10),
        b"TALAAN-VAD"
    );
    assert_eq!(
        &first_bytes(&w.talaan().join("vad/ggml-silero-v6.2.0.bin"), 10),
        b"TALAAN-VAD"
    );
    assert!(w.moved_marker("talaan").is_file());
    assert_eq!(w.model_folders(), strings(&["vad", "whisper-tiny"]));
}

#[test]
fn a_second_run_is_already_done_and_looks_at_nothing() {
    let w = World::new();
    put_model(&w.talaan().join("vad"), "vad");
    let store = w.store(App::Talaan);
    assert_eq!(store.move_in(&[]).unwrap().moved, strings(&["vad"]));
    let before = file_id(&w.shared.join("vad/ggml-silero-v6.2.0.bin"));

    // A model that shows up in the old folder later is not touched: the move is done.
    put_model(&w.talaan().join("whisper-base"), "whisper-base");
    let report = store.move_in(&["whisper-base"]).unwrap();

    assert_eq!(report, MoveReport::skipped());
    assert!(report.already_done);
    assert_eq!(w.model_folders(), strings(&["vad"]));
    assert!(w.talaan().join("whisper-base/ggml-base.bin").is_file());
    assert!(
        w.users("whisper-base").is_empty(),
        "used is not looked at either"
    );
    assert_eq!(
        before,
        file_id(&w.shared.join("vad/ggml-silero-v6.2.0.bin"))
    );
    // Another store of the same app on the same folder sees the marker too.
    assert!(w.store(App::Talaan).move_in(&[]).unwrap().already_done);
}

#[test]
fn talaan_on_windows_moves_both_of_its_folders_and_a_model_in_both_is_moved_once() {
    let w = World::new();
    put_model(&w.talaan_windows().join("vad"), "vad");
    put_model(&w.talaan_windows().join("whisper-tiny"), "whisper-tiny");
    // The roaming folder holds a copy of whisper-tiny (Talaan's own move was half done) and a model of its own.
    put_model(&w.talaan_roaming().join("whisper-tiny"), "whisper-tiny");
    put_model(&w.talaan_roaming().join("whisper-base"), "whisper-base");
    let store = w.store_on(App::Talaan, true, true);

    let report = store.move_in(&[]).unwrap();

    assert_eq!(
        sorted(&report.moved),
        strings(&["vad", "whisper-base", "whisper-tiny"]),
        "whisper-tiny is listed once"
    );
    assert!(report.left_for_later.is_empty());
    for id in ["vad", "whisper-base", "whisper-tiny"] {
        assert_installed(&store, id);
    }
    assert_linked(&w.talaan_windows().join("vad"), &w.shared, "vad");
    assert_linked(
        &w.talaan_roaming().join("whisper-base"),
        &w.shared,
        "whisper-base",
    );
    // Both copies of whisper-tiny are now the one shared file.
    assert_linked(
        &w.talaan_windows().join("whisper-tiny"),
        &w.shared,
        "whisper-tiny",
    );
    assert_linked(
        &w.talaan_roaming().join("whisper-tiny"),
        &w.shared,
        "whisper-tiny",
    );
    assert!(w.moved_marker("talaan").is_file());
}

#[test]
fn talaan_on_windows_with_only_the_old_roaming_folder() {
    let w = World::new();
    put_model(&w.talaan_roaming().join("whisper-small"), "whisper-small");
    let store = w.store_on(App::Talaan, true, true);

    let report = store.move_in(&[]).unwrap();

    assert_eq!(report.moved, strings(&["whisper-small"]));
    assert_installed(&store, "whisper-small");
    assert_linked(
        &w.talaan_roaming().join("whisper-small"),
        &w.shared,
        "whisper-small",
    );
}

#[test]
fn a_roaming_folder_is_only_scanned_when_the_store_knows_one() {
    let w = World::new();
    put_model(&w.talaan_roaming().join("whisper-small"), "whisper-small");
    let store = w.store_on(App::Talaan, true, false);

    let report = store.move_in(&[]).unwrap();

    assert!(report.moved.is_empty());
    assert!(w.model_folders().is_empty());
    assert!(
        w.talaan_roaming()
            .join("whisper-small/ggml-small.bin")
            .is_file()
    );
}

#[test]
fn ensayo_moves_speech_models_and_its_llm_folder() {
    let w = World::new();
    put_model(&w.ensayo().join("whisper-tiny"), "whisper-tiny");
    put_model(&w.ensayo().join("llm/qwen3.5-4b"), "qwen3.5-4b");
    // The AI models are only looked for under `llm/`.
    put_model(&w.ensayo().join("qwen3.5-9b"), "qwen3.5-9b");
    let store = w.store(App::Ensayo);

    let report = store.move_in(&[]).unwrap();

    assert_eq!(
        sorted(&report.moved),
        strings(&["qwen3.5-4b", "whisper-tiny"])
    );
    assert_installed(&store, "whisper-tiny");
    assert_installed(&store, "qwen3.5-4b");
    assert_linked(&w.ensayo().join("whisper-tiny"), &w.shared, "whisper-tiny");
    assert_linked(&w.ensayo().join("llm/qwen3.5-4b"), &w.shared, "qwen3.5-4b");
    assert_eq!(w.users("qwen3.5-4b"), strings(&["ensayo"]));
    assert!(
        w.ensayo()
            .join("qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf")
            .is_file()
    );
    assert_eq!(w.model_folders(), strings(&["qwen3.5-4b", "whisper-tiny"]));
}

#[test]
fn parlato_moves_its_files_with_the_catalog_names_and_registers_a_dropped_gguf() {
    let w = World::new();
    let p = w.parlato();
    put_file(
        &p.join("Models/ggml-large-v3-turbo-q5_0.bin"),
        574_041_195,
        b"",
    );
    put_file(&p.join("Models/ggml-silero-v6.2.0.bin"), 885_098, b"");
    put_file(
        &p.join("LlmModels/granite-4.2-3b-q4.gguf"),
        2_244_011_552,
        b"GRANITE",
    );
    // Things the move must leave alone.
    let qwen25 = p.join("LlmModels/qwen2.5-3b-instruct-q4.gguf");
    let llama = p.join("LlmModels/llama-3.2-3b-instruct-q4.gguf");
    put_file(&qwen25, 4096, b"QWEN2.5 weights");
    put_file(&llama, 8192, b"LLAMA weights");
    let imported_gguf = p.join("LlmModels/imported/mine.gguf");
    let imported_bin = p.join("Models/imported/mine.bin");
    let part = p.join("LlmModels/granite-4.2-3b-q4.gguf.part");
    let wrong_size = p.join("Models/ggml-small.bin");
    put_file(&imported_gguf, 100, b"my own");
    put_file(&imported_bin, 100, b"my own");
    put_file(&part, 100, b"partial");
    put_file(&wrong_size, 487_601_966, b"");
    let store = w.store(App::Parlato);

    let report = store.move_in(&[]).unwrap();

    assert_eq!(
        sorted(&report.moved),
        strings(&["granite-4.2-3b", "vad", "whisper-turbo"])
    );
    assert!(report.left_for_later.is_empty() && report.linked.is_empty());
    // Granite is renamed on the way in and linked back under Parlato's own name.
    assert_installed(&store, "granite-4.2-3b");
    assert_eq!(
        sorted_names(&w.shared.join("granite-4.2-3b")),
        strings(&["granite-4.2-3b-Q4_K_M.gguf"])
    );
    let old_granite = p.join("LlmModels/granite-4.2-3b-q4.gguf");
    assert!(old_granite.is_file());
    assert!(same_file(
        &old_granite,
        &w.shared.join("granite-4.2-3b/granite-4.2-3b-Q4_K_M.gguf")
    ));
    assert_eq!(&first_bytes(&old_granite, 7), b"GRANITE");
    assert_installed(&store, "whisper-turbo");
    assert!(same_file(
        &p.join("Models/ggml-large-v3-turbo-q5_0.bin"),
        &w.shared.join("whisper-turbo/ggml-large-v3-turbo-q5_0.bin")
    ));
    assert_installed(&store, "vad");
    assert_eq!(
        w.model_folders(),
        strings(&["granite-4.2-3b", "vad", "whisper-turbo"])
    );

    // The two dropped models are reported as imports, where they are, and are not touched.
    let mut expected = vec![
        ImportedGguf {
            id: import_id(&llama).unwrap(),
            path: llama.clone(),
        },
        ImportedGguf {
            id: import_id(&qwen25).unwrap(),
            path: qwen25.clone(),
        },
    ];
    expected.sort_by(|a, b| a.path.cmp(&b.path));
    let mut got = report.imported.clone();
    got.sort_by(|a, b| a.path.cmp(&b.path));
    assert_eq!(got, expected);
    assert!(got.iter().all(|i| i.id.starts_with("import-")));
    assert_eq!(&first_bytes(&qwen25, 15), b"QWEN2.5 weights");
    assert_eq!(fs::metadata(&qwen25).unwrap().len(), 4096);
    assert_eq!(fs::metadata(&llama).unwrap().len(), 8192);
    // Imported copies, `.part` files and a file of the wrong size are neither moved nor registered.
    for untouched in [&imported_gguf, &imported_bin, &part] {
        assert!(untouched.is_file(), "{} was touched", untouched.display());
    }
    assert!(wrong_size.is_file());
    assert_eq!(fs::metadata(&wrong_size).unwrap().len(), 487_601_966);
    assert_eq!(w.users("granite-4.2-3b"), strings(&["parlato"]));
    assert!(w.moved_marker("parlato").is_file());
}

#[test]
fn only_parlato_registers_imports() {
    let w = World::new();
    put_file(
        &w.parlato().join("LlmModels/phi-3.5-mini-instruct-q4.gguf"),
        4096,
        b"PHI",
    );
    let report = w.store(App::Talaan).move_in(&[]).unwrap();
    assert!(report.imported.is_empty());
    assert!(report.moved.is_empty() && report.linked.is_empty());
}

// ---- hashes ------------------------------------------------------------------------------------------------------

#[test]
fn parakeet_style_folders_are_hash_checked_and_a_changed_byte_keeps_the_model_out() {
    let w = World::new();
    let a = data(1, 3000);
    let b = data(2, 100);
    let good = local_model("pk-good", &[("a.onnx", &a), ("b.txt", &b)]);
    let bad = local_model("pk-bad", &[("a.onnx", &a), ("b.txt", &b)]);
    let unchecked = local_model("pk-unchecked", &[("a.onnx", &a), ("b.txt", &b)]);
    let root = w.local.join("parlato-like");
    write_bodies(&root, "good", &[("a.onnx", &a), ("b.txt", &b)]);
    // Same sizes, one byte changed.
    let changed = with_one_byte_changed(&a);
    write_bodies(&root, "bad", &[("a.onnx", &changed), ("b.txt", &b)]);
    write_bodies(&root, "unchecked", &[("a.onnx", &changed), ("b.txt", &b)]);
    let legacy = [source(
        App::Parlato,
        &root,
        vec![
            entry("pk-good", "good", true),
            entry("pk-bad", "bad", true),
            // The same wrong bytes are adopted when the entry says the old app checked its hashes.
            entry("pk-unchecked", "unchecked", false),
        ],
    )];
    let store = w.store(App::Parlato);

    let report = store
        .move_in_with(&[good, bad, unchecked], &legacy, &[])
        .unwrap();

    assert_eq!(report.moved, strings(&["pk-good", "pk-unchecked"]));
    assert_eq!(
        report.left_for_later,
        vec![("pk-bad".to_string(), "hash mismatch".to_string())]
    );
    assert_installed_local(&store, good);
    assert_installed_local(&store, unchecked);
    // The model that did not match is exactly where it was and is not in the shared folder or a user of it.
    assert_eq!(fs::read(root.join("bad/a.onnx")).unwrap(), changed);
    assert!(!w.shared.join("pk-bad").exists());
    assert!(w.users("pk-bad").is_empty());
    assert_eq!(w.users("pk-good"), strings(&["parlato"]));
    assert_eq!(fs::read(w.shared.join("pk-good/a.onnx")).unwrap(), a);
    // A hash mismatch does not come back by itself, so it does not keep the move open.
    assert!(w.moved_marker("parlato").is_file());
}

fn assert_installed_local(store: &SharedStore, model: &ModelInfo) {
    match store.status_with(model) {
        Status::Installed { .. } => {}
        other => panic!("{}: expected Installed, got {other:?}", model.id),
    }
}

#[test]
fn another_apps_hash_checked_model_is_linked_only_when_it_matches() {
    let w = World::new();
    let a = data(3, 2000);
    let good = local_model("pk-good", &[("a.onnx", &a)]);
    let bad = local_model("pk-bad", &[("a.onnx", &a)]);
    let root = w.local.join("parlato-like");
    write_bodies(&root, "good", &[("a.onnx", &a)]);
    write_bodies(&root, "bad", &[("a.onnx", &with_one_byte_changed(&a))]);
    let legacy = [source(
        App::Parlato,
        &root,
        vec![entry("pk-good", "good", true), entry("pk-bad", "bad", true)],
    )];
    let store = w.store(App::Talaan);

    let report = store.move_in_with(&[good, bad], &legacy, &[]).unwrap();

    assert_eq!(report.linked, strings(&["pk-good"]));
    assert!(report.moved.is_empty());
    assert_eq!(
        report.left_for_later,
        vec![("pk-bad".to_string(), "hash mismatch".to_string())]
    );
    assert_installed_local(&store, good);
    assert!(same_file(
        &root.join("good/a.onnx"),
        &w.shared.join("pk-good/a.onnx")
    ));
    assert_eq!(w.users("pk-good"), strings(&["parlato"]));
    assert!(!w.shared.join("pk-bad").exists());
    assert!(w.users("pk-bad").is_empty());
}

// ---- other apps' folders -------------------------------------------------------------------------------------

/// Talaan has vad and whisper-tiny, Parlato has Turbo and Granite, in their own old folders.
fn talaan_and_parlato_have_models(w: &World) {
    put_file(
        &w.talaan().join("vad/ggml-silero-v6.2.0.bin"),
        885_098,
        b"TALAAN-VAD",
    );
    put_model(&w.talaan().join("whisper-tiny"), "whisper-tiny");
    put_file(
        &w.parlato().join("Models/ggml-large-v3-turbo-q5_0.bin"),
        574_041_195,
        b"",
    );
    put_file(
        &w.parlato().join("LlmModels/granite-4.2-3b-q4.gguf"),
        2_244_011_552,
        b"GRANITE",
    );
}

#[test]
fn ensayo_links_the_models_of_talaan_and_parlato_and_marks_them_as_users() {
    let w = World::new();
    talaan_and_parlato_have_models(&w);
    // A dropped GGUF of Parlato is Parlato's to register, not Ensayo's.
    put_file(
        &w.parlato().join("LlmModels/gemma-2-2b-it-q4.gguf"),
        4096,
        b"G",
    );
    let store = w.store(App::Ensayo);

    let report = store.move_in(&["whisper-tiny"]).unwrap();

    assert_eq!(
        sorted(&report.linked),
        strings(&["granite-4.2-3b", "vad", "whisper-tiny", "whisper-turbo"])
    );
    assert!(report.moved.is_empty() && report.imported.is_empty());
    assert!(report.left_for_later.is_empty());
    for id in ["granite-4.2-3b", "vad", "whisper-tiny", "whisper-turbo"] {
        assert_installed(&store, id);
    }
    // Hard links, not moves: the other apps' files are where they were, the same files as the shared ones.
    assert_linked(&w.talaan().join("vad"), &w.shared, "vad");
    assert_linked(&w.talaan().join("whisper-tiny"), &w.shared, "whisper-tiny");
    assert!(same_file(
        &w.parlato().join("LlmModels/granite-4.2-3b-q4.gguf"),
        &w.shared.join("granite-4.2-3b/granite-4.2-3b-Q4_K_M.gguf")
    ));
    assert!(same_file(
        &w.parlato().join("Models/ggml-large-v3-turbo-q5_0.bin"),
        &w.shared.join("whisper-turbo/ggml-large-v3-turbo-q5_0.bin")
    ));
    assert_eq!(
        &first_bytes(&w.talaan().join("vad/ggml-silero-v6.2.0.bin"), 10),
        b"TALAAN-VAD"
    );
    // The other apps are users of what they have, so `collect` must not erase it; Ensayo only of what it uses.
    assert_eq!(w.users("vad"), strings(&["talaan"]));
    assert_eq!(w.users("whisper-tiny"), strings(&["ensayo", "talaan"]));
    assert_eq!(w.users("whisper-turbo"), strings(&["parlato"]));
    assert_eq!(w.users("granite-4.2-3b"), strings(&["parlato"]));
    store.collect().unwrap();
    assert_eq!(
        w.model_folders(),
        strings(&["granite-4.2-3b", "vad", "whisper-tiny", "whisper-turbo"])
    );
    assert!(w.moved_marker("ensayo").is_file());
    assert!(!w.moved_marker("talaan").exists());
}

#[test]
fn talaan_moving_after_ensayo_finds_its_models_shared_and_replaces_its_copy_with_a_link() {
    let w = World::new();
    talaan_and_parlato_have_models(&w);
    w.store(App::Ensayo).move_in(&[]).unwrap();
    let shared_vad = w.shared.join("vad/ggml-silero-v6.2.0.bin");
    let id_before = file_id(&shared_vad);

    let report = w.store(App::Talaan).move_in(&[]).unwrap();

    // Talaan's own models: moved in (their copies replaced by links); the models of Parlato need nothing.
    assert_eq!(sorted(&report.moved), strings(&["vad", "whisper-tiny"]));
    assert!(report.linked.is_empty(), "nothing is linked twice");
    assert!(report.left_for_later.is_empty());
    assert_linked(&w.talaan().join("vad"), &w.shared, "vad");
    assert_linked(&w.talaan().join("whisper-tiny"), &w.shared, "whisper-tiny");
    assert_eq!(
        id_before,
        file_id(&shared_vad),
        "the shared file was not replaced"
    );
    assert_eq!(&first_bytes(&shared_vad, 10), b"TALAAN-VAD");
    assert_eq!(w.users("vad"), strings(&["talaan"]));
    assert_eq!(w.users("whisper-tiny"), strings(&["talaan"]));
    assert!(w.moved_marker("talaan").is_file());
}

#[test]
fn an_old_copy_of_a_model_the_shared_folder_already_has_is_replaced_by_a_link() {
    let w = World::new();
    let ensayo = w.store(App::Ensayo);
    // Someone downloaded vad into the shared folder: a different file with the same size.
    put_model(&w.shared.join("vad"), "vad");
    ensayo.use_model("vad").unwrap();
    put_file(
        &w.talaan().join("vad/ggml-silero-v6.2.0.bin"),
        885_098,
        b"OLD COPY",
    );
    let shared_file = w.shared.join("vad/ggml-silero-v6.2.0.bin");
    let old_file = w.talaan().join("vad/ggml-silero-v6.2.0.bin");
    if cfg!(unix) {
        assert!(!same_file(&old_file, &shared_file));
    }
    let shared_id = file_id(&shared_file);

    let report = w.store(App::Talaan).move_in(&[]).unwrap();

    assert_eq!(report.moved, strings(&["vad"]));
    assert!(same_file(&old_file, &shared_file));
    assert_eq!(
        shared_id,
        file_id(&shared_file),
        "the shared file stays as it was"
    );
    assert_eq!(
        first_bytes(&old_file, 8),
        vec![0u8; 8],
        "the old copy is gone, the shared bytes show"
    );
    assert_eq!(w.users("vad"), strings(&["ensayo", "talaan"]));
}

#[test]
fn files_of_the_wrong_size_are_ignored_in_every_folder() {
    let w = World::new();
    put_file(
        &w.talaan().join("whisper-tiny/ggml-tiny.bin"),
        77_691_712,
        b"short",
    );
    put_file(
        &w.ensayo().join("whisper-base/ggml-base.bin"),
        147_951_466,
        b"long",
    );
    let report = w.store(App::Talaan).move_in(&[]).unwrap();
    assert_eq!(report, MoveReport::default());
    assert!(w.model_folders().is_empty());
    assert_eq!(
        fs::metadata(w.talaan().join("whisper-tiny/ggml-tiny.bin"))
            .unwrap()
            .len(),
        77_691_712
    );
    // Nothing to do is done: the marker is written.
    assert!(w.moved_marker("talaan").is_file());
    let report = w.store(App::Ensayo).move_in(&[]).unwrap();
    assert!(report.moved.is_empty() && report.linked.is_empty());
    assert!(w.model_folders().is_empty());
}

#[test]
#[cfg(unix)]
fn a_link_that_cannot_be_made_is_skipped_for_other_apps_and_reported() {
    use std::os::unix::fs::PermissionsExt;
    struct Restore(PathBuf);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o755));
        }
    }
    let w = World::new();
    put_model(&w.talaan().join("whisper-tiny"), "whisper-tiny");
    put_model(&w.talaan().join("vad"), "vad");
    // A shared folder for whisper-tiny that nothing can be added to: the link fails. (vad links fine.)
    let blocked = w.shared.join("whisper-tiny");
    fs::create_dir_all(&blocked).unwrap();
    fs::set_permissions(&blocked, fs::Permissions::from_mode(0o555)).unwrap();
    let _restore = Restore(blocked.clone());
    let store = w.store(App::Ensayo);

    let report = store.move_in(&[]).unwrap();

    assert_eq!(report.linked, strings(&["vad"]));
    assert_eq!(
        report.left_for_later,
        vec![("whisper-tiny".to_string(), "link failed".to_string())]
    );
    assert!(sorted_names(&blocked).is_empty());
    assert!(
        w.users("whisper-tiny").is_empty(),
        "Talaan is not marked as using what is not there"
    );
    assert!(w.talaan().join("whisper-tiny/ggml-tiny.bin").is_file());
    assert_eq!(w.users("vad"), strings(&["talaan"]));
    // A link that fails does not keep the move open: it would fail again.
    assert!(w.moved_marker("ensayo").is_file());
}

#[test]
fn a_model_the_shared_folder_already_has_only_marks_the_other_app() {
    let w = World::new();
    // Someone downloaded whisper-tiny; Talaan also has its own copy in its old folder.
    put_model(&w.shared.join("whisper-tiny"), "whisper-tiny");
    w.store(App::Ensayo).use_model("whisper-tiny").unwrap();
    put_model(&w.talaan().join("whisper-tiny"), "whisper-tiny");
    let shared_file = w.shared.join("whisper-tiny/ggml-tiny.bin");
    let id_before = file_id(&shared_file);

    let report = w.store(App::Parlato).move_in(&[]).unwrap();

    // Parlato has nothing of its own; Talaan has the model, so it is a user, and nothing else happens.
    assert_eq!(report, MoveReport::default());
    assert_eq!(w.users("whisper-tiny"), strings(&["ensayo", "talaan"]));
    assert_eq!(id_before, file_id(&shared_file));
    if cfg!(unix) {
        assert!(
            !same_file(&w.talaan().join("whisper-tiny/ggml-tiny.bin"), &shared_file),
            "Talaan's own file is not touched"
        );
    }
}

#[test]
fn a_link_that_fails_half_way_leaves_no_half_model_and_no_user() {
    let w = World::new();
    let id = "parakeet-unified-en";
    put_model(&w.talaan().join(id), id);
    let names: Vec<&str> = model(id).files.iter().map(|f| f.name).collect();
    // The last file cannot be linked: something that is not a file is in its place.
    let dir = w.shared.join(id);
    fs::create_dir_all(dir.join(names[3])).unwrap();

    let report = w.store(App::Ensayo).move_in(&[]).unwrap();

    assert!(report.linked.is_empty());
    assert_eq!(
        report.left_for_later,
        vec![(id.to_string(), "link failed".to_string())]
    );
    assert_eq!(
        sorted_names(&dir),
        vec![names[3].to_string()],
        "the links made were taken back"
    );
    assert!(w.users(id).is_empty());
    assert!(w.talaan().join(id).join(names[0]).is_file());
}

// ---- used ---------------------------------------------------------------------------------------------------------

#[test]
fn the_models_the_app_passes_in_are_marked_as_used_and_nothing_else_is() {
    let w = World::new();
    put_model(&w.talaan().join("whisper-tiny"), "whisper-tiny");
    let store = w.store(App::Talaan);

    let report = store
        .move_in(&[
            "whisper-tiny",
            "whisper-small",
            "qwen3.5-4b",
            "import-0123456789ab",
            "../outside",
        ])
        .unwrap();

    assert_eq!(report.moved, strings(&["whisper-tiny"]));
    assert_eq!(w.users("whisper-tiny"), strings(&["talaan"]));
    assert_eq!(w.users("whisper-small"), strings(&["talaan"]));
    assert_eq!(w.users("qwen3.5-4b"), strings(&["talaan"]));
    // An imported GGUF lives where the person put it, and a path is not a model id.
    assert!(w.users("import-0123456789ab").is_empty());
    assert!(!w.shared.join(".engine/users/import-0123456789ab").exists());
    assert!(!w.shared.parent().unwrap().join("outside").exists());
    assert_eq!(
        sorted_names(&w.shared.join(".engine/users")),
        strings(&["qwen3.5-4b", "whisper-small", "whisper-tiny"])
    );
}

// ---- interrupted, locked -------------------------------------------------------------------------------------------

#[test]
fn a_move_that_was_interrupted_half_way_is_finished_by_running_it_again() {
    let w = World::new();
    let id = "parakeet-unified-en";
    let old = w.talaan().join(id);
    put_model(&old, id);
    let names: Vec<&str> = model(id).files.iter().map(|f| f.name).collect();
    assert_eq!(names.len(), 4);
    // The first run stopped after renaming two files (the first one's link back was made, the second's was not).
    let dir = w.shared.join(id);
    fs::create_dir_all(&dir).unwrap();
    fs::rename(old.join(names[0]), dir.join(names[0])).unwrap();
    fs::hard_link(dir.join(names[0]), old.join(names[0])).unwrap();
    fs::rename(old.join(names[1]), dir.join(names[1])).unwrap();
    let store = w.store(App::Talaan);
    assert!(matches!(store.status(id), Status::Partial { .. }));

    let report = store.move_in(&[]).unwrap();

    assert_eq!(report.moved, strings(&[id]));
    assert!(report.left_for_later.is_empty());
    assert_installed(&store, id);
    assert_linked(&old, &w.shared, id);
    assert_eq!(w.users(id), strings(&["talaan"]));
    assert!(w.moved_marker("talaan").is_file());

    // Running the whole thing again (the marker lost, say) changes nothing that is done.
    fs::remove_file(w.moved_marker("talaan")).unwrap();
    let ids_before: Vec<_> = names.iter().map(|n| file_id(&dir.join(n))).collect();
    let report = store.move_in(&[]).unwrap();
    assert!(report.left_for_later.is_empty());
    assert_installed(&store, id);
    assert_linked(&old, &w.shared, id);
    let ids_after: Vec<_> = names.iter().map(|n| file_id(&dir.join(n))).collect();
    assert_eq!(ids_before, ids_after, "the shared files were not replaced");
    assert_eq!(w.users(id), strings(&["talaan"]));
    assert_eq!(sorted_names(&dir).len(), 4);
    assert!(w.moved_marker("talaan").is_file());
}

#[test]
fn an_interrupted_link_in_from_another_app_is_finished_too() {
    let w = World::new();
    let id = "parakeet-unified-en";
    let talaan_old = w.talaan().join(id);
    put_model(&talaan_old, id);
    let names: Vec<&str> = model(id).files.iter().map(|f| f.name).collect();
    // Ensayo linked one file and stopped.
    let dir = w.shared.join(id);
    fs::create_dir_all(&dir).unwrap();
    fs::hard_link(talaan_old.join(names[2]), dir.join(names[2])).unwrap();
    let store = w.store(App::Ensayo);

    let report = store.move_in(&[]).unwrap();

    assert_eq!(report.linked, strings(&[id]));
    assert_installed(&store, id);
    assert_linked(&talaan_old, &w.shared, id);
    assert_eq!(w.users(id), strings(&["talaan"]));
}

fn is_the_vad_file(p: &Path) -> bool {
    p.file_name().is_some_and(|n| n == "ggml-silero-v6.2.0.bin")
}

#[test]
fn a_file_that_is_in_use_is_left_for_the_next_start_and_the_move_stays_open() {
    let w = World::new();
    put_file(
        &w.talaan().join("vad/ggml-silero-v6.2.0.bin"),
        885_098,
        b"BUSY",
    );
    put_model(&w.talaan().join("whisper-tiny"), "whisper-tiny");
    let store = w.store(App::Talaan);
    let hooks = MoveHooks {
        is_in_use: is_the_vad_file,
    };
    let legacy: Vec<LegacySource> = App::ALL
        .into_iter()
        .flat_map(|a| LegacySource::all(a, &w.local, Some(&w.roaming), false))
        .collect();

    let report = store
        .move_in_with_hooks(&real_catalog(), &legacy, &[], &hooks)
        .unwrap();

    assert_eq!(report.moved, strings(&["whisper-tiny"]));
    assert_eq!(
        report.left_for_later,
        vec![("vad".to_string(), "in use".to_string())]
    );
    assert!(
        !w.moved_marker("talaan").exists(),
        "the next start finishes the job"
    );
    // The busy file is exactly where it was, and the shared folder knows nothing of it.
    assert_eq!(
        &first_bytes(&w.talaan().join("vad/ggml-silero-v6.2.0.bin"), 4),
        b"BUSY"
    );
    assert!(!w.shared.join("vad").exists());
    assert!(w.users("vad").is_empty());
    assert_eq!(w.model_folders(), strings(&["whisper-tiny"]));

    // The next start: nothing is in use, the rest is moved and the marker is written.
    let report = store.move_in(&[]).unwrap();
    assert!(report.moved.contains(&"vad".to_string()));
    assert!(report.left_for_later.is_empty());
    assert_installed(&store, "vad");
    assert_linked(&w.talaan().join("vad"), &w.shared, "vad");
    assert_linked(&w.talaan().join("whisper-tiny"), &w.shared, "whisper-tiny");
    assert!(w.moved_marker("talaan").is_file());
    assert!(store.move_in(&[]).unwrap().already_done);
}

#[test]
fn a_file_in_use_in_another_apps_folder_is_not_linked_but_does_not_keep_this_move_open() {
    let w = World::new();
    put_file(&w.talaan().join("vad/ggml-silero-v6.2.0.bin"), 885_098, b"");
    let hooks = MoveHooks {
        is_in_use: is_the_vad_file,
    };
    let legacy = LegacySource::all(App::Talaan, &w.local, None, false);

    let report = w
        .store(App::Ensayo)
        .move_in_with_hooks(&real_catalog(), &legacy, &[], &hooks)
        .unwrap();

    // Linking reads the file and changes nothing in it, so another app having it open is no reason to wait.
    assert_eq!(report.linked, strings(&["vad"]));
    assert!(w.moved_marker("ensayo").is_file());
}

#[test]
#[cfg(unix)]
fn the_real_check_treats_a_file_that_cannot_be_opened_for_write_as_in_use() {
    use std::os::unix::fs::PermissionsExt;
    let w = World::new();
    let file = w.talaan().join("vad/ggml-silero-v6.2.0.bin");
    put_file(&file, 885_098, b"");
    fs::set_permissions(&file, fs::Permissions::from_mode(0o444)).unwrap();
    let store = w.store(App::Talaan);

    let report = store.move_in(&[]).unwrap();

    assert!(report.moved.is_empty());
    assert_eq!(
        report.left_for_later,
        vec![("vad".to_string(), "in use".to_string())]
    );
    assert!(!w.moved_marker("talaan").exists());
    assert!(file.is_file() && !w.shared.join("vad").exists());

    fs::set_permissions(&file, fs::Permissions::from_mode(0o644)).unwrap();
    let report = store.move_in(&[]).unwrap();
    assert_eq!(report.moved, strings(&["vad"]));
    assert!(w.moved_marker("talaan").is_file());
}

#[test]
#[cfg(windows)]
fn a_file_held_open_on_windows_is_left_for_the_next_start() {
    use std::os::windows::fs::OpenOptionsExt;
    let w = World::new();
    let file = w.talaan().join("vad/ggml-silero-v6.2.0.bin");
    put_file(&file, 885_098, b"");
    // No sharing at all: what an app with the model loaded looks like to the file system.
    let held = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .share_mode(0)
        .open(&file)
        .unwrap();
    let store = w.store_on(App::Talaan, true, false);
    let report = store.move_in(&[]).unwrap();
    assert!(report.moved.is_empty());
    assert_eq!(
        report.left_for_later,
        vec![("vad".to_string(), "in use".to_string())]
    );
    assert!(!w.moved_marker("talaan").exists());
    drop(held);
    assert_eq!(store.move_in(&[]).unwrap().moved, strings(&["vad"]));
}

// ---- stores that never move -----------------------------------------------------------------------------------------

#[test]
fn a_store_opened_with_open_at_never_moves_and_never_creates_anything_outside_its_folder() {
    let w = World::new();
    put_model(&w.talaan().join("vad"), "vad");
    let parent = w.shared.parent().unwrap().to_path_buf();
    let store = SharedStore::open_at(App::Talaan, w.shared.clone()).unwrap();
    let before = sorted_names(&parent);
    let engine_before = sorted_names(&w.shared.join(".engine"));

    let report = store.move_in(&["vad"]).unwrap();

    assert_eq!(report, MoveReport::skipped());
    // Not even through the explicit entry point, with the old folder handed to it.
    let legacy = LegacySource::all(App::Talaan, &w.local, None, false);
    let report = store
        .move_in_with(&real_catalog(), &legacy, &["vad"])
        .unwrap();
    assert_eq!(report, MoveReport::skipped());
    assert_eq!(sorted_names(&parent), before);
    assert_eq!(sorted_names(&w.shared.join(".engine")), engine_before);
    assert!(w.model_folders().is_empty());
    assert!(w.users("vad").is_empty());
    assert!(!w.moved_marker("talaan").exists());
    assert!(w.talaan().join("vad/ggml-silero-v6.2.0.bin").is_file());
    assert_ne!(Some(store.root().to_path_buf()), SharedStore::default_dir());
}
