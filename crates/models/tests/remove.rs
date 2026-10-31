//! `SharedStore::remove_everywhere` on temporary folders only. The apps' old folders, the "local app data" folder and
//! the shared folder are all under one temp dir; no test touches the real shared model folder or any app's real
//! folder. The old copies come from a real `move_in`, so the hard links are real.

mod support;

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use support::assert_not_real;
use voice_engine_models::{App, Released, Removed, SharedStore, Status, catalog};

const VAD_FILE: &str = "ggml-silero-v6.2.0.bin";
const VAD_SIZE: u64 = 885_098;

struct World {
    tmp: tempfile::TempDir,
    local: PathBuf,
    shared: PathBuf,
}

impl World {
    fn new() -> World {
        let tmp = tempfile::tempdir().unwrap();
        let world = World {
            local: tmp.path().join("local"),
            shared: tmp.path().join("shared"),
            tmp,
        };
        for dir in [&world.local, &world.shared] {
            fs::create_dir_all(dir).unwrap();
            assert_not_real(dir);
        }
        world
    }

    /// `app`'s store with old folders in the temp "local app data" (Mac layout).
    fn store(&self, app: App) -> SharedStore {
        SharedStore::for_move_test(app, &self.shared, &self.local, None, false).unwrap()
    }

    /// A store without old folders on the same shared folder.
    fn plain(&self, app: App) -> SharedStore {
        SharedStore::open_at(app, self.shared.clone()).unwrap()
    }

    fn talaan_vad(&self) -> PathBuf {
        self.local.join("Talaan/models/vad").join(VAD_FILE)
    }

    fn parlato_vad(&self) -> PathBuf {
        self.local
            .join("com.craftconceptsdigital.parlato/Models")
            .join(VAD_FILE)
    }

    fn markers(&self, id: &str) -> PathBuf {
        self.shared.join(".engine/users").join(id)
    }

    /// Talaan and Parlato both had the voice detection model; both moved in, so both old paths are links.
    fn moved_in(&self) {
        put_file(&self.talaan_vad(), VAD_SIZE);
        put_file(&self.parlato_vad(), VAD_SIZE);
        for app in [App::Talaan, App::Parlato, App::Ensayo] {
            let report = self.store(app).move_in(&[]).unwrap();
            assert!(report.left_for_later.is_empty(), "{report:?}");
        }
    }

    /// Every file under the temp dir whose length is `size`.
    fn files_of_size(&self, size: u64) -> Vec<PathBuf> {
        fn walk(dir: &Path, size: u64, out: &mut Vec<PathBuf>) {
            for e in fs::read_dir(dir).into_iter().flatten().flatten() {
                let path = e.path();
                if path.is_dir() {
                    walk(&path, size, out);
                } else if fs::metadata(&path).is_ok_and(|m| m.len() == size) {
                    out.push(path);
                }
            }
        }
        let mut out = Vec::new();
        walk(self.tmp.path(), size, &mut out);
        out
    }
}

fn put_file(path: &Path, size: u64) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    File::create(path).unwrap().set_len(size).unwrap();
}

fn install(store: &SharedStore, id: &str) {
    for f in catalog::find(id).unwrap().files {
        put_file(&store.model_dir(id).join(f.name), f.size);
    }
}

fn lock(path: &Path) -> File {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .unwrap()
}

#[test]
fn a_model_used_by_two_apps_is_erased_for_both() {
    let w = World::new();
    let (talaan, ensayo) = (w.plain(App::Talaan), w.plain(App::Ensayo));
    install(&talaan, "vad");
    talaan.use_model("vad").unwrap();
    ensayo.use_model("vad").unwrap();
    assert!(matches!(
        talaan.status("vad"),
        Status::Installed { users, .. } if users == vec![App::Talaan, App::Ensayo]
    ));

    assert_eq!(
        talaan.remove_everywhere("vad").unwrap(),
        Removed::Erased { old_copies_left: 0 }
    );

    assert!(!w.shared.join("vad").exists());
    assert!(!w.markers("vad").exists());
    assert_eq!(talaan.status("vad"), Status::Missing);
    assert_eq!(ensayo.status("vad"), Status::Missing);
    // Nothing else was touched.
    assert!(w.shared.join(".engine").is_dir());
}

#[test]
fn a_held_handle_or_a_download_lock_gives_in_use_and_touches_nothing() {
    let w = World::new();
    let (talaan, ensayo) = (w.plain(App::Talaan), w.plain(App::Ensayo));
    install(&talaan, "vad");
    talaan.use_model("vad").unwrap();
    ensayo.use_model("vad").unwrap();
    let intact = || {
        assert!(w.shared.join("vad").join(VAD_FILE).is_file());
        assert!(w.markers("vad").join("talaan").is_file());
        assert!(w.markers("vad").join("ensayo").is_file());
    };

    // Another app has it loaded.
    let handle = ensayo.hold("vad").unwrap();
    assert_eq!(talaan.remove_everywhere("vad").unwrap(), Removed::InUse);
    intact();
    drop(handle);

    // This app's own handle counts too.
    let own = talaan.hold("vad").unwrap();
    assert_eq!(talaan.remove_everywhere("vad").unwrap(), Removed::InUse);
    intact();
    drop(own);

    // A download holds its lock.
    let download = lock(&w.shared.join(".engine/locks/vad.download"));
    download.lock().unwrap();
    assert_eq!(talaan.remove_everywhere("vad").unwrap(), Removed::InUse);
    intact();
    drop(download);

    // With everything let go it works.
    assert_eq!(
        talaan.remove_everywhere("vad").unwrap(),
        Removed::Erased { old_copies_left: 0 }
    );
}

#[test]
fn old_copies_left_by_the_move_are_deleted_and_the_disk_is_freed() {
    let w = World::new();
    w.moved_in();
    // The shared file, Talaan's old link and Parlato's old link.
    assert_eq!(w.files_of_size(VAD_SIZE).len(), 3);
    let talaan_folder = w.talaan_vad().parent().unwrap().to_path_buf();
    let parlato = w.store(App::Parlato);

    assert_eq!(
        parlato.remove_everywhere("vad").unwrap(),
        Removed::Erased { old_copies_left: 0 }
    );

    assert!(!w.talaan_vad().exists());
    assert!(!w.parlato_vad().exists());
    assert!(!talaan_folder.exists(), "the emptied old folder is removed");
    assert!(!w.shared.join("vad").exists());
    assert!(
        w.files_of_size(VAD_SIZE).is_empty(),
        "no file keeps the bytes on disk"
    );
    // The emptied `Models` folder goes; Parlato's own folder above it stays.
    assert!(!w.parlato_vad().parent().unwrap().exists());
    assert!(w.parlato_vad().parent().unwrap().parent().unwrap().is_dir());
    assert_eq!(w.store(App::Ensayo).status("vad"), Status::Missing);
}

#[test]
fn an_old_file_of_another_size_is_left_alone_and_so_is_a_folder_with_other_files() {
    let w = World::new();
    w.moved_in();
    // The person put something else under the same name in another app's folder, and a note next to Talaan's file.
    let ensayo_vad = w
        .local
        .join("com.craftconceptsdigital.ensayo/models/vad")
        .join(VAD_FILE);
    put_file(&ensayo_vad, 1234);
    let note = w.talaan_vad().with_file_name("notes.txt");
    fs::write(&note, b"mine").unwrap();

    assert_eq!(
        w.store(App::Talaan).remove_everywhere("vad").unwrap(),
        Removed::Erased { old_copies_left: 0 }
    );

    assert_eq!(fs::metadata(&ensayo_vad).unwrap().len(), 1234);
    assert!(note.is_file(), "a folder with other files is never removed");
    assert!(!w.talaan_vad().exists());
}

#[test]
fn an_open_at_store_never_looks_for_old_folders() {
    let w = World::new();
    w.moved_in();
    let plain = w.plain(App::Talaan);

    assert_eq!(
        plain.remove_everywhere("vad").unwrap(),
        Removed::Erased { old_copies_left: 0 }
    );

    assert!(!w.shared.join("vad").exists());
    assert!(!w.markers("vad").exists());
    assert!(w.talaan_vad().is_file(), "the old copy stays");
    assert!(w.parlato_vad().is_file());
}

#[test]
fn an_unknown_or_never_used_id_is_not_there() {
    let w = World::new();
    let store = w.store(App::Talaan);
    assert_eq!(store.remove_everywhere("vad").unwrap(), Removed::NotThere);
    assert_eq!(
        store.remove_everywhere("no-such-model").unwrap(),
        Removed::NotThere
    );
    for bad in ["", "..", ".engine", "a/b"] {
        assert_eq!(
            store.remove_everywhere(bad).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput,
            "{bad:?}"
        );
    }
    assert!(w.shared.join(".engine").is_dir());
}

#[test]
fn a_read_only_store_is_refused_and_changes_nothing() {
    let w = World::new();
    let talaan = w.plain(App::Talaan);
    install(&talaan, "vad");
    talaan.use_model("vad").unwrap();
    let view = SharedStore::read_only(App::Talaan, w.shared.clone());

    let err = view.remove_everywhere("vad").unwrap_err();

    assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
    assert!(w.shared.join("vad").join(VAD_FILE).is_file());
    assert!(w.markers("vad").join("talaan").is_file());
}

#[test]
fn an_imported_model_outside_the_catalog_is_erased() {
    let w = World::new();
    let store = w.store(App::Parlato);
    let id = "imported-abc123";
    assert!(catalog::find(id).is_none());
    put_file(&store.model_dir(id).join("my.gguf"), 10);
    fs::create_dir_all(w.markers(id)).unwrap();
    let mut marker = File::create(w.markers(id).join("parlato")).unwrap();
    marker.write_all(b"").unwrap();

    assert_eq!(
        store.remove_everywhere(id).unwrap(),
        Removed::Erased { old_copies_left: 0 }
    );

    assert!(!store.model_dir(id).exists());
    assert!(!w.markers(id).exists());
    assert_eq!(store.remove_everywhere(id).unwrap(), Removed::NotThere);
}

#[test]
fn release_still_leaves_the_other_apps_alone() {
    // remove_everywhere is the only call that erases for everybody.
    let w = World::new();
    let (talaan, ensayo) = (w.plain(App::Talaan), w.plain(App::Ensayo));
    install(&talaan, "vad");
    talaan.use_model("vad").unwrap();
    ensayo.use_model("vad").unwrap();
    assert_eq!(
        talaan.release("vad").unwrap(),
        Released::Kept {
            users: vec![App::Ensayo]
        }
    );
}
