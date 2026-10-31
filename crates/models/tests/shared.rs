//! `SharedStore` on temporary folders only. No test touches the real shared model folder, another app's folder or
//! the network (downloads run against the local test server in `tests/support`). Two stores ("apps") share one
//! root in one process: their locks must conflict, which is proved here on whatever platform runs the tests.

mod support;

use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use support::{Behaviour, TestServer};
use voice_engine_models::download::{DownloadError, Progress};
use voice_engine_models::{
    App, ModelFile, ModelInfo, Released, SharedStore, Status, StoreError, catalog,
};

const IDLE: Duration = Duration::from_secs(30);

fn vad() -> &'static ModelInfo {
    catalog::find("vad").expect("vad is in the catalog")
}

fn diarization() -> &'static ModelInfo {
    catalog::find("diarization").expect("diarization is in the catalog")
}

/// A store on `dir`. Panics if `dir` is inside a real folder.
fn store(app: App, dir: &Path) -> SharedStore {
    support::assert_not_real(dir);
    SharedStore::open_at(app, dir.to_path_buf()).unwrap()
}

/// A temporary root with a Talaan store and an Ensayo store on it.
fn two_stores() -> (tempfile::TempDir, SharedStore, SharedStore) {
    let tmp = tempfile::tempdir().unwrap();
    let talaan = store(App::Talaan, tmp.path());
    let ensayo = store(App::Ensayo, tmp.path());
    (tmp, talaan, ensayo)
}

/// Creates every file of `model` in its folder, zero-filled with the exact catalog size (sparse, so cheap).
fn install(s: &SharedStore, model: &ModelInfo) {
    let dir = s.model_dir(model.id);
    fs::create_dir_all(&dir).unwrap();
    for f in model.files {
        let file = File::create(dir.join(f.name)).unwrap();
        file.set_len(f.size).unwrap();
    }
}

fn set_age(path: &Path, age: Duration) {
    let file = OpenOptions::new().write(true).open(path).unwrap();
    file.set_modified(SystemTime::now() - age).unwrap();
}

fn not_cancelled() -> AtomicBool {
    AtomicBool::new(false)
}

fn ignore(_: Progress) {}

fn installed_users(status: Status) -> (Vec<App>, bool) {
    match status {
        Status::Installed {
            users, used_by_me, ..
        } => (users, used_by_me),
        other => panic!("expected Installed, got {other:?}"),
    }
}

// ---- a local model for the download tests -------------------------------------------------------------------

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

fn local_model(
    id: &'static str,
    server: &TestServer,
    files: &[(&str, &[u8])],
) -> &'static ModelInfo {
    let files: Vec<ModelFile> = files
        .iter()
        .map(|(name, body)| ModelFile {
            name: leak((*name).into()),
            url: leak(server.url(name)),
            size: body.len() as u64,
            sha256: leak(format!("{:x}", Sha256::digest(body))),
        })
        .collect();
    let template = vad();
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

/// Starts a download on its own thread. It stalls after `n` bytes (idle is long), so it holds the download lock
/// until the cancel flag is set. Returns once part of the file is on disk, which means the lock is held.
struct StalledDownload {
    cancel: &'static AtomicBool,
    thread: thread::JoinHandle<Result<PathBuf, StoreError>>,
}

fn start_stalled_download(
    s: &'static SharedStore,
    model: &'static ModelInfo,
    part: &Path,
    n: usize,
) -> StalledDownload {
    let cancel: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));
    let thread = thread::spawn(move || s.download_with(model, cancel, IDLE, &mut ignore));
    let deadline = Instant::now() + Duration::from_secs(10);
    // Looks at the part file only (no lock probe), so it cannot get in the downloader's way.
    while fs::metadata(part).map(|m| m.len()).unwrap_or(0) < n as u64 {
        assert!(Instant::now() < deadline, "the download never started");
        assert!(!thread.is_finished(), "the download ended too early");
        thread::sleep(Duration::from_millis(10));
    }
    StalledDownload { cancel, thread }
}

impl StalledDownload {
    fn stop(self) -> Result<PathBuf, StoreError> {
        self.cancel.store(true, Ordering::SeqCst);
        self.thread.join().unwrap()
    }
}

// ---- status, users ---------------------------------------------------------------------------------------------

#[test]
fn a_model_that_is_not_there_is_missing() {
    let (_tmp, talaan, _ensayo) = two_stores();
    assert_eq!(talaan.status("vad"), Status::Missing);
    assert_eq!(talaan.status("no-such-model"), Status::Missing);
    // Not a valid id and not a model: still just missing.
    assert_eq!(talaan.status("../elsewhere"), Status::Missing);
}

#[test]
fn installed_is_not_used_by_me_until_use_model() {
    let (_tmp, talaan, _ensayo) = two_stores();
    install(&talaan, vad());
    let status = talaan.status("vad");
    assert_eq!(
        status,
        Status::Installed {
            dir: talaan.model_dir("vad"),
            users: vec![],
            used_by_me: false
        }
    );
    talaan.use_model("vad").unwrap();
    assert_eq!(
        installed_users(talaan.status("vad")),
        (vec![App::Talaan], true)
    );
}

#[test]
fn a_file_of_the_wrong_size_is_not_installed() {
    let (_tmp, talaan, _ensayo) = two_stores();
    let dir = talaan.model_dir("vad");
    fs::create_dir_all(&dir).unwrap();
    let f = &vad().files[0];
    File::create(dir.join(f.name))
        .unwrap()
        .set_len(f.size - 1)
        .unwrap();
    // Not installed (and, with no .part file, nothing a resume could continue either).
    assert_eq!(talaan.status("vad"), Status::Missing);
    assert!(talaan.hold("vad").is_err());
}

#[test]
fn two_users_are_listed_and_each_sees_its_own_used_by_me() {
    let (_tmp, talaan, ensayo) = two_stores();
    install(&talaan, vad());
    ensayo.use_model("vad").unwrap();
    talaan.use_model("vad").unwrap();
    assert_eq!(
        installed_users(talaan.status("vad")),
        (vec![App::Talaan, App::Ensayo], true)
    );
    assert_eq!(
        installed_users(ensayo.status("vad")),
        (vec![App::Talaan, App::Ensayo], true)
    );
}

#[test]
fn use_model_refuses_an_unknown_id() {
    let (tmp, talaan, _ensayo) = two_stores();
    assert_eq!(
        talaan.use_model("../x"),
        Err(StoreError::UnknownModel("../x".into()))
    );
    assert!(!tmp.path().join(".engine/users/x").exists());
}

#[test]
fn list_has_every_catalog_model_in_catalog_order() {
    let (_tmp, talaan, _ensayo) = two_stores();
    install(&talaan, vad());
    let list = talaan.list();
    let listed: Vec<&str> = list.iter().map(|(m, _)| m.id).collect();
    let expected: Vec<&str> = catalog::all().iter().map(|m| m.id).collect();
    assert_eq!(listed, expected);
    for (model, status) in &list {
        if model.id == "vad" {
            assert!(matches!(status, Status::Installed { .. }));
        } else {
            assert_eq!(*status, Status::Missing, "{}", model.id);
        }
    }
}

// ---- release -----------------------------------------------------------------------------------------------------

#[test]
fn release_by_one_of_two_users_keeps_the_files() {
    let (_tmp, talaan, ensayo) = two_stores();
    install(&talaan, vad());
    talaan.use_model("vad").unwrap();
    ensayo.use_model("vad").unwrap();
    assert_eq!(
        talaan.release("vad").unwrap(),
        Released::Kept {
            users: vec![App::Ensayo]
        }
    );
    assert!(talaan.model_dir("vad").join(vad().files[0].name).exists());
    assert_eq!(
        installed_users(ensayo.status("vad")),
        (vec![App::Ensayo], true)
    );
}

#[test]
fn release_by_the_last_user_erases_the_model() {
    let (tmp, talaan, ensayo) = two_stores();
    install(&talaan, vad());
    talaan.use_model("vad").unwrap();
    ensayo.use_model("vad").unwrap();
    ensayo.release("vad").unwrap();
    assert_eq!(talaan.release("vad").unwrap(), Released::Erased);
    assert!(!talaan.model_dir("vad").exists());
    assert!(!tmp.path().join(".engine/users/vad").exists());
    assert_eq!(talaan.status("vad"), Status::Missing);
}

#[test]
fn release_without_being_a_user_is_not_using() {
    let (_tmp, talaan, ensayo) = two_stores();
    // Nothing there at all.
    assert_eq!(talaan.release("vad").unwrap(), Released::NotUsing);
    // Other apps use it and this one does not: nothing of Talaan's to remove, the model stays.
    install(&ensayo, vad());
    ensayo.use_model("vad").unwrap();
    assert_eq!(talaan.release("vad").unwrap(), Released::NotUsing);
    assert!(ensayo.model_dir("vad").exists());
    assert_eq!(
        installed_users(talaan.status("vad")),
        (vec![App::Ensayo], false)
    );
}

#[test]
fn release_of_a_leftover_with_no_users_erases_it() {
    let (_tmp, talaan, _ensayo) = two_stores();
    install(&talaan, vad());
    assert_eq!(talaan.release("vad").unwrap(), Released::Erased);
    assert!(!talaan.model_dir("vad").exists());
}

#[test]
fn release_refuses_ids_that_are_not_one_plain_folder_name() {
    let (tmp, talaan, _ensayo) = two_stores();
    install(&talaan, vad());
    talaan.use_model("vad").unwrap();
    for bad in ["", ".", "..", ".engine", "../x", "a/b", "a\\b", ".hidden"] {
        let err = talaan.release(bad).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{bad:?}");
    }
    assert!(tmp.path().join(".engine/users/vad/talaan").exists());
    assert!(talaan.model_dir("vad").exists());
}

#[test]
fn release_while_another_app_holds_the_model_keeps_the_folder_until_collect() {
    let (_tmp, talaan, ensayo) = two_stores();
    install(&talaan, vad());
    talaan.use_model("vad").unwrap();
    let handle = ensayo.hold("vad").unwrap();
    assert_eq!(handle.dir(), talaan.model_dir("vad"));

    // Store B (Ensayo) holds the model; store A (Talaan) is the last user and cannot erase it.
    assert_eq!(
        talaan.release("vad").unwrap(),
        Released::Kept { users: vec![] }
    );
    assert!(talaan.model_dir("vad").join(vad().files[0].name).exists());

    // Collect cannot erase it either while the handle lives.
    talaan.collect().unwrap();
    assert!(talaan.model_dir("vad").exists());

    drop(handle);
    talaan.collect().unwrap();
    assert!(!talaan.model_dir("vad").exists());
    assert_eq!(talaan.status("vad"), Status::Missing);
}

#[test]
fn the_use_lock_conflicts_between_two_stores_in_one_process() {
    // The platform check behind the tests above: a lock taken through one store is seen by a separately opened
    // file for the same path, even inside one process.
    let (tmp, talaan, ensayo) = two_stores();
    install(&talaan, vad());
    let lock_path = tmp.path().join(".engine/locks/vad.use");
    let probe = || {
        OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap()
    };
    let handle = ensayo.hold("vad").unwrap();
    assert!(
        matches!(probe().try_lock(), Err(std::fs::TryLockError::WouldBlock)),
        "an exclusive lock must be refused while a ModelHandle is alive"
    );
    // Shared locks do not conflict with each other: two apps can have the model loaded.
    let second = talaan.hold("vad").unwrap();
    drop(handle);
    assert!(matches!(
        probe().try_lock(),
        Err(std::fs::TryLockError::WouldBlock)
    ));
    drop(second);
    assert!(probe().try_lock().is_ok());
}

#[test]
fn a_dropped_handle_leaves_no_stuck_lock() {
    // A crash drops everything the app held; dropping the handle is that from this process's point of view.
    let (tmp, talaan, ensayo) = two_stores();
    install(&talaan, vad());
    ensayo.use_model("vad").unwrap();
    {
        let _handle = ensayo.hold("vad").unwrap();
        assert_eq!(
            ensayo.release("vad").unwrap(),
            Released::Kept { users: vec![] }
        );
    }
    // The handle is gone: the leftover goes, and so does a fresh release.
    talaan.collect().unwrap();
    assert!(!talaan.model_dir("vad").exists());
    install(&talaan, vad());
    ensayo.use_model("vad").unwrap();
    drop(ensayo.hold("vad").unwrap());
    assert_eq!(ensayo.release("vad").unwrap(), Released::Erased);
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .open(tmp.path().join(".engine/locks/vad.use"))
        .unwrap();
    assert!(lock.try_lock().is_ok());
}

#[test]
fn hold_needs_an_installed_model() {
    let (_tmp, talaan, _ensayo) = two_stores();
    assert_eq!(
        talaan.hold("vad").unwrap_err(),
        StoreError::NotInstalled("vad".into())
    );
    assert_eq!(
        talaan.hold("nope").unwrap_err(),
        StoreError::UnknownModel("nope".into())
    );
    // Part of a model is not enough.
    let dir = talaan.model_dir("vad");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join(format!("{}.part", vad().files[0].name)), b"abc").unwrap();
    assert!(talaan.hold("vad").is_err());
}

#[test]
fn release_all_releases_every_model_this_app_uses_and_only_those() {
    let (_tmp, talaan, ensayo) = two_stores();
    install(&talaan, vad());
    install(&talaan, diarization());
    let search = catalog::find("search").unwrap();
    install(&talaan, search);
    for id in ["vad", "diarization", "search"] {
        talaan.use_model(id).unwrap();
    }
    ensayo.use_model("vad").unwrap();
    talaan.release_all().unwrap();
    // Shared with Ensayo: stays. Used only by Talaan: gone.
    assert_eq!(
        installed_users(ensayo.status("vad")),
        (vec![App::Ensayo], true)
    );
    assert_eq!(talaan.status("diarization"), Status::Missing);
    assert_eq!(talaan.status("search"), Status::Missing);
    // Calling it again is fine.
    talaan.release_all().unwrap();
}

// ---- collect -------------------------------------------------------------------------------------------------------

#[test]
fn collect_erases_a_model_nobody_uses_and_keeps_the_rest() {
    let (_tmp, talaan, ensayo) = two_stores();
    install(&talaan, vad());
    install(&talaan, diarization());
    ensayo.use_model("diarization").unwrap();
    talaan.collect().unwrap();
    assert_eq!(talaan.status("vad"), Status::Missing);
    assert!(matches!(
        talaan.status("diarization"),
        Status::Installed { .. }
    ));
}

#[test]
fn collect_leaves_alone_what_is_not_a_model_folder() {
    let (tmp, talaan, _ensayo) = two_stores();
    fs::create_dir_all(tmp.path().join("my-own-folder")).unwrap();
    fs::write(tmp.path().join("my-own-folder/notes.txt"), b"x").unwrap();
    fs::write(tmp.path().join("readme.txt"), b"x").unwrap();
    talaan.collect().unwrap();
    assert!(tmp.path().join("my-own-folder/notes.txt").exists());
    assert!(tmp.path().join("readme.txt").exists());
    assert!(tmp.path().join(".engine/locks").is_dir());
}

#[test]
fn collect_removes_part_files_older_than_a_week_but_not_young_ones() {
    let (_tmp, talaan, _ensayo) = two_stores();
    // Both models are used, so they stay; only their stale part files go.
    talaan.use_model("vad").unwrap();
    talaan.use_model("diarization").unwrap();
    let old = talaan
        .model_dir("vad")
        .join(format!("{}.part", vad().files[0].name));
    let young = talaan
        .model_dir("diarization")
        .join(format!("{}.part", diarization().files[0].name));
    for p in [&old, &young] {
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, b"partial").unwrap();
    }
    set_age(&old, Duration::from_secs(8 * 24 * 3600));
    set_age(&young, Duration::from_secs(6 * 24 * 3600));
    talaan.collect().unwrap();
    assert!(!old.exists(), "an 8 day old part file is a leftover");
    assert!(young.exists(), "a 6 day old part file may still be resumed");
}

#[test]
fn collect_does_not_touch_a_part_file_while_a_download_holds_its_lock() {
    let server = TestServer::start(vec![]);
    let body = data(7, 300_000);
    server.set(
        "held.bin",
        Behaviour::StallsAfter {
            body: body.clone(),
            n: 100_000,
        },
    );
    let model = local_model("local-held", &server, &[("held.bin", &body)]);
    let (_tmp, talaan, ensayo) = two_stores();
    let talaan: &'static SharedStore = Box::leak(Box::new(talaan));
    let part = talaan.model_dir(model.id).join("held.bin.part");

    let running = start_stalled_download(talaan, model, &part, 100_000);
    set_age(&part, Duration::from_secs(30 * 24 * 3600));
    ensayo.collect().unwrap();
    assert!(part.exists(), "a part file that is being written stays");

    assert_eq!(
        running.stop().unwrap_err(),
        StoreError::Download(DownloadError::Cancelled)
    );
    // The lock is free now: the same old part file is a leftover.
    set_age(&part, Duration::from_secs(30 * 24 * 3600));
    ensayo.collect().unwrap();
    assert!(!part.exists());
}

// ---- download ------------------------------------------------------------------------------------------------------

#[test]
fn download_writes_the_files_and_the_user_marker_and_reports_progress_to_one() {
    let (a, b) = (data(1, 150_000), data(2, 90_000));
    let server = TestServer::start(vec![
        ("a.bin", Behaviour::Body(a.clone())),
        ("b.bin", Behaviour::Body(b.clone())),
    ]);
    let model = local_model("local-two-files", &server, &[("a.bin", &a), ("b.bin", &b)]);
    let (_tmp, talaan, ensayo) = two_stores();
    let mut reports = Vec::new();
    let dir = talaan
        .download_with(model, &not_cancelled(), IDLE, &mut |p| reports.push(p))
        .unwrap();
    assert_eq!(dir, talaan.model_dir(model.id));
    assert_eq!(fs::read(dir.join("a.bin")).unwrap(), a);
    assert_eq!(fs::read(dir.join("b.bin")).unwrap(), b);
    let last = reports.last().unwrap();
    assert_eq!(
        (last.fraction, last.bytes, last.total),
        (1.0, 240_000, 240_000)
    );
    assert_eq!(
        installed_users(talaan.status_with(model)),
        (vec![App::Talaan], true)
    );
    // The other app sees it as already on this computer, and one call adds it without a download.
    assert_eq!(
        installed_users(ensayo.status_with(model)),
        (vec![App::Talaan], false)
    );
    // Nothing is left in the folder but the model.
    assert!(!dir.join(".cancelled").exists());
}

#[test]
fn download_of_an_unknown_model_is_refused_before_anything_is_written() {
    let (tmp, talaan, _ensayo) = two_stores();
    let result = talaan.download("no-such-model", &not_cancelled(), &mut ignore);
    assert_eq!(
        result,
        Err(StoreError::UnknownModel("no-such-model".into()))
    );
    assert!(!tmp.path().join("no-such-model").exists());
    assert!(!tmp.path().join(".engine/users/no-such-model").exists());
}

#[test]
fn a_second_store_cannot_download_while_the_first_does() {
    let server = TestServer::start(vec![]);
    let body = data(3, 400_000);
    server.set(
        "big.bin",
        Behaviour::StallsAfter {
            body: body.clone(),
            n: 150_000,
        },
    );
    let model = local_model("local-busy", &server, &[("big.bin", &body)]);
    let (_tmp, talaan, ensayo) = two_stores();
    let talaan: &'static SharedStore = Box::leak(Box::new(talaan));
    let part = talaan.model_dir(model.id).join("big.bin.part");

    let running = start_stalled_download(talaan, model, &part, 150_000);

    // Store B: no second download, and the status shows A's progress, read from the part file sizes.
    let started = Instant::now();
    assert_eq!(
        ensayo
            .download_with(model, &not_cancelled(), IDLE, &mut ignore)
            .unwrap_err(),
        StoreError::DownloadingElsewhere
    );
    assert!(started.elapsed() < Duration::from_secs(2));
    match ensayo.status_with(model) {
        Status::Downloading {
            fraction,
            bytes,
            by_me,
        } => {
            assert!(!by_me);
            assert_eq!(bytes, 150_000);
            assert!(
                (fraction - 150_000.0 / 400_000.0).abs() < 0.001,
                "{fraction}"
            );
        }
        other => panic!("expected Downloading, got {other:?}"),
    }
    match talaan.status_with(model) {
        Status::Downloading { by_me, .. } => assert!(by_me),
        other => panic!("expected Downloading, got {other:?}"),
    }
    // B's attempt did not break A's, and B is a user too (its marker is written first).
    assert!(!running.thread.is_finished());
    let _ = running.stop();
    assert_eq!(
        ensayo.release(model.id).unwrap(),
        Released::Kept {
            users: vec![App::Talaan]
        }
    );
}

#[test]
fn a_cancelled_download_is_partial_and_cancelled_and_a_stalled_one_is_partial_and_not() {
    let server = TestServer::start(vec![]);
    let body = data(4, 500_000);
    server.set(
        "m.bin",
        Behaviour::Slow {
            body: body.clone(),
            per_chunk_delay: Duration::from_millis(5),
        },
    );
    let model = local_model("local-partial", &server, &[("m.bin", &body)]);
    let (_tmp, talaan, _ensayo) = two_stores();
    assert_eq!(talaan.status_with(model), Status::Missing);

    // Cancel part way.
    let cancel = not_cancelled();
    let result = talaan.download_with(model, &cancel, IDLE, &mut |p| {
        if p.bytes >= 100_000 {
            cancel.store(true, Ordering::SeqCst);
        }
    });
    assert_eq!(result, Err(StoreError::Download(DownloadError::Cancelled)));
    match talaan.status_with(model) {
        Status::Partial {
            cancelled,
            bytes,
            fraction,
        } => {
            assert!(cancelled);
            assert!((100_000..500_000).contains(&bytes), "{bytes}");
            assert!(fraction > 0.0 && fraction < 1.0);
        }
        other => panic!("expected Partial, got {other:?}"),
    }

    // The same model now stalls: not a failure, not cancelled.
    server.set(
        "m.bin",
        Behaviour::StallsAfter {
            body: body.clone(),
            n: 120_000,
        },
    );
    let result = talaan.download_with(
        model,
        &not_cancelled(),
        Duration::from_millis(300),
        &mut ignore,
    );
    assert_eq!(result, Err(StoreError::Download(DownloadError::Stalled)));
    assert!(matches!(
        talaan.status_with(model),
        Status::Partial {
            cancelled: false,
            ..
        }
    ));

    // The server recovers and the download finishes.
    server.set("m.bin", Behaviour::Body(body.clone()));
    talaan
        .download_with(model, &not_cancelled(), IDLE, &mut ignore)
        .unwrap();
    assert!(matches!(
        talaan.status_with(model),
        Status::Installed {
            used_by_me: true,
            ..
        }
    ));
}

// ---- where things are created -------------------------------------------------------------------------------------

fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn open_at_creates_and_uses_only_its_own_folder() {
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("models");
    assert!(listing(parent.path()).is_empty());

    let talaan = store(App::Talaan, &root);
    let ensayo = store(App::Ensayo, &root);
    assert_eq!(talaan.root(), root);
    assert_eq!(listing(parent.path()), vec!["models"]);

    install(&talaan, vad());
    talaan.use_model("vad").unwrap();
    ensayo.use_model("vad").unwrap();
    drop(ensayo.hold("vad").unwrap());
    let _ = talaan.list();
    talaan.release("vad").unwrap();
    ensayo.release("vad").unwrap();
    talaan.release_all().unwrap();
    talaan.collect().unwrap();
    let _ = talaan.download("no-such-model", &not_cancelled(), &mut ignore);

    assert_eq!(listing(parent.path()), vec!["models"]);
    assert_eq!(
        listing(&root),
        vec![".engine"],
        "after the last release the root holds only .engine"
    );
    assert_eq!(
        listing(&root.join(".engine")),
        vec!["locks", "moved", "users"]
    );
}

#[test]
fn default_dir_is_a_path_under_craft_concepts_digital_and_building_it_creates_nothing() {
    // Only builds a path. The real folder holds real models and no test touches it.
    let dir = SharedStore::default_dir().expect("a local app data folder");
    assert!(dir.ends_with(Path::new("Craft Concepts Digital").join("Models")));
    assert!(dir.is_absolute());
    if cfg!(target_os = "macos") {
        assert!(dir.ends_with("Library/Application Support/Craft Concepts Digital/Models"));
    }
}

#[test]
#[should_panic(expected = "a test must never use a real folder")]
fn the_test_helper_refuses_the_real_folder() {
    if let Some(real) = SharedStore::default_dir() {
        // `store` panics before it opens (or creates) anything.
        let _ = store(App::Talaan, &real.join("anything"));
    } else {
        panic!("a test must never use a real folder (none on this system)");
    }
}

#[test]
fn a_read_only_view_shows_what_is_there_and_creates_and_changes_nothing() {
    // A folder that does not exist: every model is missing and it is not created.
    let parent = tempfile::tempdir().unwrap();
    let absent = parent.path().join("models");
    let view = SharedStore::read_only(App::Talaan, absent.clone());
    assert!(
        view.list()
            .iter()
            .all(|(_, status)| *status == Status::Missing)
    );
    assert!(!absent.exists(), "looking must not create the folder");

    // A folder with a model in it: seen as installed, with its users, and nothing is added, removed or locked.
    let (tmp, talaan, ensayo) = two_stores();
    install(&talaan, vad());
    ensayo.use_model("vad").unwrap();
    let before = tree(tmp.path());
    let view = SharedStore::read_only(App::Talaan, tmp.path().to_path_buf());
    assert_eq!(
        installed_users(view.status("vad")),
        (vec![App::Ensayo], false)
    );
    assert_eq!(view.list().len(), voice_engine_models::catalog::all().len());
    assert_eq!(tree(tmp.path()), before);

    // Every call that would write is refused and leaves the folder as it was.
    for denied in [
        view.use_model("vad").map_err(|e| e.to_string()),
        view.release("vad")
            .map(|_| ())
            .map_err(|e| e.kind().to_string()),
        view.release_all().map_err(|e| e.kind().to_string()),
        view.collect().map_err(|e| e.kind().to_string()),
    ] {
        assert!(denied.is_err());
    }
    assert!(view.hold("vad").is_err());
    assert!(view.download("vad", &not_cancelled(), &mut ignore).is_err());
    assert!(
        view.move_in(&["vad"]).unwrap().already_done,
        "a view never moves"
    );
    assert_eq!(tree(tmp.path()), before);
}

/// Every path under `dir` with its size and modified time.
fn tree(dir: &Path) -> Vec<(PathBuf, u64, SystemTime)> {
    let mut out = Vec::new();
    let mut todo = vec![dir.to_path_buf()];
    while let Some(d) = todo.pop() {
        for e in fs::read_dir(&d).unwrap() {
            let path = e.unwrap().path();
            let meta = fs::metadata(&path).unwrap();
            out.push((path.clone(), meta.len(), meta.modified().unwrap()));
            if meta.is_dir() {
                todo.push(path);
            }
        }
    }
    out.sort();
    out
}
