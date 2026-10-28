//! Finding a model on this computer: the app's own folder (and its legacy folder), then other apps' folders when
//! reuse is on. A copy in another app's folder is used in place and read only; this code never writes, moves or
//! deletes anything there.

use crate::apps::{App, OtherRoot};
use crate::catalog::ModelInfo;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    Missing,
    /// The app's own copy.
    Own(PathBuf),
    /// Another app's copy, read in place.
    OtherApp {
        app: App,
        dir: PathBuf,
    },
}

impl Location {
    pub fn dir(&self) -> Option<&Path> {
        match self {
            Location::Own(p) | Location::OtherApp { dir: p, .. } => Some(p),
            Location::Missing => None,
        }
    }
}

/// Where an app's models are and which other apps' folders it may read.
#[derive(Debug, Clone)]
pub struct ModelStore {
    /// `<root>/<model id>/<files>`: where the app downloads to.
    pub root: PathBuf,
    /// An older own folder still read as the app's own copy until the app has moved its models out of it.
    pub legacy_root: Option<PathBuf>,
    pub others: Vec<OtherRoot>,
    pub reuse: bool,
}

impl ModelStore {
    /// The app's own folder only; no legacy folder, no other apps.
    pub fn new(root: PathBuf) -> Self {
        ModelStore {
            root,
            legacy_root: None,
            others: Vec::new(),
            reuse: false,
        }
    }

    pub fn with_legacy(mut self, legacy_root: Option<PathBuf>) -> Self {
        self.legacy_root = legacy_root.filter(|l| *l != self.root);
        self
    }

    pub fn with_others(mut self, others: Vec<OtherRoot>, reuse: bool) -> Self {
        self.others = others;
        self.reuse = reuse;
        self
    }

    /// Where the app downloads `m` to.
    pub fn own_dir(&self, m: &ModelInfo) -> PathBuf {
        self.root.join(m.id)
    }

    /// Every own folder that may hold `m` (current, then legacy): what Delete removes.
    pub fn own_dirs(&self, m: &ModelInfo) -> Vec<PathBuf> {
        std::iter::once(self.own_dir(m))
            .chain(self.legacy_root.iter().map(|r| r.join(m.id)))
            .collect()
    }

    pub fn locate(&self, m: &ModelInfo) -> Location {
        if let Some(own) = self.own_dirs(m).into_iter().find(|d| complete(d, m)) {
            return Location::Own(own);
        }
        if self.reuse {
            for other in &self.others {
                if let Some(dir) = other.candidates(m).into_iter().find(|d| complete(d, m)) {
                    return Location::OtherApp {
                        app: other.app,
                        dir,
                    };
                }
            }
        }
        Location::Missing
    }
}

/// Every file present with its exact size. Hashes are checked at download; re-hashing gigabytes on every lookup would
/// be too slow, and a size match on files this large is a strong signal.
pub fn complete(dir: &Path, m: &ModelInfo) -> bool {
    m.files.iter().all(|f| {
        std::fs::metadata(dir.join(f.name))
            .map(|md| md.is_file() && md.len() == f.size)
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{EngineKind, ModelFile, ModelInfo};
    use crate::{App, Layout, OtherRoot};
    use std::fs;

    static FAKE: ModelInfo = ModelInfo {
        id: "fake",
        name: "Fake",
        maker: "Test",
        note: "",
        engine: EngineKind::Whisper,
        languages: "",
        speed: 1,
        accuracy: 1,
        files: &[ModelFile {
            name: "a.bin",
            url: "https://example.com/a.bin",
            size: 10,
            sha256: "",
        }],
        reuse_dirs: &["Models"],
    };

    /// A file of exactly `size` bytes (sparse, so tests stay fast).
    fn file(path: &Path, size: u64) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::File::create(path).unwrap().set_len(size).unwrap();
    }

    fn other(app: App, root: &Path, layout: Layout) -> OtherRoot {
        OtherRoot {
            app,
            root: root.to_path_buf(),
            layout,
        }
    }

    #[test]
    fn own_copy_wins_and_a_partial_file_does_not_count() {
        let own = tempfile::tempdir().unwrap();
        let store = ModelStore::new(own.path().to_path_buf());
        assert_eq!(store.locate(&FAKE), Location::Missing);
        file(&own.path().join("fake").join("a.bin"), 9);
        assert_eq!(store.locate(&FAKE), Location::Missing, "wrong size");
        file(&own.path().join("fake").join("a.bin"), 10);
        assert_eq!(store.locate(&FAKE), Location::Own(own.path().join("fake")));
    }

    #[test]
    fn the_legacy_folder_counts_as_own() {
        let own = tempfile::tempdir().unwrap();
        let old = tempfile::tempdir().unwrap();
        file(&old.path().join("fake").join("a.bin"), 10);
        let store =
            ModelStore::new(own.path().to_path_buf()).with_legacy(Some(old.path().to_path_buf()));
        assert_eq!(store.locate(&FAKE), Location::Own(old.path().join("fake")));
        assert_eq!(
            store.own_dirs(&FAKE),
            vec![own.path().join("fake"), old.path().join("fake")]
        );
    }

    #[test]
    fn another_apps_copy_is_found_only_when_reuse_is_on() {
        let own = tempfile::tempdir().unwrap();
        let parlato = tempfile::tempdir().unwrap();
        file(&parlato.path().join("Models").join("a.bin"), 10);
        let others = vec![other(App::Parlato, parlato.path(), Layout::ReuseDirs)];
        let on = ModelStore::new(own.path().to_path_buf()).with_others(others.clone(), true);
        assert_eq!(
            on.locate(&FAKE),
            Location::OtherApp {
                app: App::Parlato,
                dir: parlato.path().join("Models")
            }
        );
        let off = ModelStore::new(own.path().to_path_buf()).with_others(others, false);
        assert_eq!(off.locate(&FAKE), Location::Missing);
    }

    #[test]
    fn by_id_layout_is_found_in_the_model_id_folder() {
        let own = tempfile::tempdir().unwrap();
        let talaan = tempfile::tempdir().unwrap();
        file(&talaan.path().join("fake").join("a.bin"), 10);
        let store = ModelStore::new(own.path().to_path_buf())
            .with_others(vec![other(App::Talaan, talaan.path(), Layout::ById)], true);
        assert_eq!(
            store.locate(&FAKE),
            Location::OtherApp {
                app: App::Talaan,
                dir: talaan.path().join("fake")
            }
        );
    }

    #[test]
    fn own_copy_is_preferred_over_another_apps() {
        let own = tempfile::tempdir().unwrap();
        let talaan = tempfile::tempdir().unwrap();
        file(&own.path().join("fake").join("a.bin"), 10);
        file(&talaan.path().join("fake").join("a.bin"), 10);
        let store = ModelStore::new(own.path().to_path_buf())
            .with_others(vec![other(App::Talaan, talaan.path(), Layout::ById)], true);
        assert_eq!(store.locate(&FAKE), Location::Own(own.path().join("fake")));
    }

    #[test]
    fn a_shared_copy_that_disappears_reads_as_missing() {
        let own = tempfile::tempdir().unwrap();
        let talaan = tempfile::tempdir().unwrap();
        let f = talaan.path().join("fake").join("a.bin");
        file(&f, 10);
        let store = ModelStore::new(own.path().to_path_buf())
            .with_others(vec![other(App::Talaan, talaan.path(), Layout::ById)], true);
        assert!(matches!(store.locate(&FAKE), Location::OtherApp { .. }));
        fs::remove_file(&f).unwrap();
        assert_eq!(store.locate(&FAKE), Location::Missing);
    }

    #[test]
    fn location_dir() {
        let p = PathBuf::from("x");
        assert_eq!(Location::Own(p.clone()).dir(), Some(p.as_path()));
        assert_eq!(
            Location::OtherApp {
                app: App::Talaan,
                dir: p.clone()
            }
            .dir(),
            Some(p.as_path())
        );
        assert_eq!(Location::Missing.dir(), None);
    }
}
