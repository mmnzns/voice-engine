//! The one-time move of each app's old models into the shared folder (spec 5.4).
//!
//! `move_in` renames this app's own old files into `<root>/<id>/` and hard-links them back, hard-links the other
//! apps' complete models in (never moving them), marks the right apps as users, registers Parlato's dropped GGUF
//! files as imports and writes `moved/<app>` last. Every step is one file at a time and safe to repeat, so a run
//! that was interrupted is finished by running it again: a model counts as there when each of its files is either
//! in the old folder or already in the shared one with the catalog's exact size.

use crate::catalog::{self, ModelFile, ModelInfo};
use crate::imports::import_id;
use crate::legacy::{App, LegacyEntry, LegacySource};
use crate::shared::{SharedStore, app_name, valid_id};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// What `move_in` did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MoveReport {
    /// Ids adopted from this app's own old folders.
    pub moved: Vec<String>,
    /// Ids hard-linked in from other apps' folders.
    pub linked: Vec<String>,
    /// Parlato only: GGUF files in `LlmModels/` that are not in the catalog, left where they are.
    pub imported: Vec<ImportedGguf>,
    /// `(id, reason)`: "in use", "hash mismatch", "link failed" or "move failed".
    pub left_for_later: Vec<(String, String)>,
    /// This app's move was finished before; nothing was looked at.
    pub already_done: bool,
}

impl MoveReport {
    /// What a store that never moves (`open_at`) or an app that already moved reports.
    pub fn skipped() -> MoveReport {
        MoveReport {
            already_done: true,
            ..MoveReport::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedGguf {
    pub path: PathBuf,
    pub id: String,
}

/// Test hooks for the move.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct MoveHooks {
    /// Whether a file is locked or open elsewhere. The real check tries to open it for write and treats a failure
    /// as "in use".
    pub is_in_use: fn(&Path) -> bool,
}

impl Default for MoveHooks {
    fn default() -> Self {
        MoveHooks {
            is_in_use: real_is_in_use,
        }
    }
}

fn real_is_in_use(path: &Path) -> bool {
    OpenOptions::new().write(true).open(path).is_err()
}

/// A rename or delete the system refused because somebody has the file open (a sharing violation on Windows).
fn is_busy(e: &io::Error) -> bool {
    cfg!(windows)
        && (matches!(e.raw_os_error(), Some(5 | 32 | 33))
            || e.kind() == io::ErrorKind::PermissionDenied)
}

fn is_exact(path: &Path, size: u64) -> bool {
    fs::metadata(path)
        .map(|m| m.is_file() && m.len() == size)
        .unwrap_or(false)
}

fn sha256_of(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// One file of a model in an old folder and where it goes.
struct Part<'a> {
    old: PathBuf,
    new: PathBuf,
    file: &'a ModelFile,
}

/// What one model came to.
enum Outcome {
    /// Nothing to do: the old folder does not hold the model (or only part of it).
    Absent,
    Done,
    /// Left as it is; the reason is one of the strings in `MoveReport::left_for_later`.
    Left(&'static str),
}

impl SharedStore {
    /// Moves what the apps kept in their old folders into the shared folder. The app calls it first thing, before
    /// it opens a model or starts a helper. `used` are the model ids the app's settings point to.
    ///
    /// Does nothing (and says `already_done`) on a store opened with `open_at`, or when this app's move is done.
    /// A file that is locked or open is left alone and reported with the reason `"in use"`; the marker is then not
    /// written, so the next start finishes the job.
    pub fn move_in(&self, used: &[&str]) -> io::Result<MoveReport> {
        let Some(old) = &self.old_folders else {
            return Ok(MoveReport::skipped());
        };
        let legacy: Vec<LegacySource> = App::ALL
            .into_iter()
            .flat_map(|app| {
                LegacySource::all(app, &old.local_data, old.roaming.as_deref(), old.windows)
            })
            .collect();
        let catalog: Vec<&ModelInfo> = catalog::all().iter().collect();
        self.move_in_with(&catalog, &legacy, used)
    }

    /// `move_in` for any catalog and any table of old folders: the tests move locally built models whose hashes
    /// they know.
    #[doc(hidden)]
    pub fn move_in_with(
        &self,
        catalog: &[&ModelInfo],
        legacy: &[LegacySource],
        used: &[&str],
    ) -> io::Result<MoveReport> {
        self.move_in_with_hooks(catalog, legacy, used, &MoveHooks::default())
    }

    /// `move_in_with` with the "is this file in use" check replaced.
    #[doc(hidden)]
    pub fn move_in_with_hooks(
        &self,
        catalog: &[&ModelInfo],
        legacy: &[LegacySource],
        used: &[&str],
        hooks: &MoveHooks,
    ) -> io::Result<MoveReport> {
        if self.old_folders.is_none() || self.moved_marker(self.app()).exists() {
            return Ok(MoveReport::skipped());
        }
        let me = self.app();
        let mut report = MoveReport::default();
        let mut keep_open = false;
        let find = |id: &str| catalog.iter().find(|m| m.id == id).copied();

        // 1. This app's own old folders (Talaan's Windows pair included, in the order given).
        for src in legacy.iter().filter(|s| s.app == me) {
            for entry in &src.entries {
                let Some(model) = find(entry.id).filter(|m| valid_id(m.id)) else {
                    continue;
                };
                match self.adopt_own(src, entry, model, hooks)? {
                    Outcome::Absent => {}
                    Outcome::Done => push_once(&mut report.moved, model.id),
                    Outcome::Left(reason) => {
                        keep_open |= reason == "in use";
                        push_reason(&mut report.left_for_later, model.id, reason);
                    }
                }
            }
        }

        // 2. The other apps' old folders: hard links, and the other app becomes a user.
        for src in legacy.iter().filter(|s| s.app != me) {
            for entry in &src.entries {
                let Some(model) = find(entry.id).filter(|m| valid_id(m.id)) else {
                    continue;
                };
                match self.link_from_other(src, entry, model)? {
                    Outcome::Absent => {}
                    Outcome::Done => push_once(&mut report.linked, model.id),
                    Outcome::Left(reason) => {
                        push_reason(&mut report.left_for_later, model.id, reason)
                    }
                }
            }
        }

        // 3. The models the app's settings point to.
        for id in used {
            if find(id).is_some_and(|m| valid_id(m.id)) {
                self.add_user_as(id, me)?;
            }
        }

        // 4. Parlato's GGUF files that are not in the catalog stay where they are, as imports. Only Parlato's table
        // names a folder for them.
        for src in legacy.iter().filter(|s| s.app == me) {
            report.imported.extend(dropped_ggufs(src, catalog));
        }
        report.imported.sort_by(|a, b| a.path.cmp(&b.path));
        report.imported.dedup();

        // 5. Done, unless a file was in use: then the next start does the rest.
        if !keep_open {
            let dir = self.root().join(".engine").join("moved");
            fs::create_dir_all(&dir)?;
            File::create(dir.join(app_name(me)))?;
        }
        Ok(report)
    }

    fn moved_marker(&self, app: App) -> PathBuf {
        self.root()
            .join(".engine")
            .join("moved")
            .join(app_name(app))
    }

    /// The files of `model` as they lie in `src`'s folder and in the shared folder.
    fn parts<'a>(
        &self,
        src: &LegacySource,
        entry: &LegacyEntry,
        model: &'a ModelInfo,
    ) -> Option<Vec<Part<'a>>> {
        let folder = src.root.join(entry.rel);
        let dir = self.model_dir(model.id);
        Some(
            entry
                .disk_files(model)?
                .into_iter()
                .map(|(disk, file)| Part {
                    old: folder.join(disk),
                    new: dir.join(file.name),
                    file,
                })
                .collect(),
        )
    }

    /// Spec 5.4 item 1 for one model in one of this app's own old folders.
    fn adopt_own(
        &self,
        src: &LegacySource,
        entry: &LegacyEntry,
        model: &ModelInfo,
        hooks: &MoveHooks,
    ) -> io::Result<Outcome> {
        let Some(parts) = self.parts(src, entry, model) else {
            return Ok(Outcome::Absent);
        };
        // Each file is in the old folder or already in the shared one (an earlier run moved it).
        let mut in_old = Vec::with_capacity(parts.len());
        let mut in_shared = Vec::with_capacity(parts.len());
        for p in &parts {
            let (o, n) = (is_exact(&p.old, p.file.size), is_exact(&p.new, p.file.size));
            if !o && !n {
                return Ok(Outcome::Absent);
            }
            in_old.push(o);
            in_shared.push(n);
        }
        if !in_old.iter().any(|o| *o) {
            return Ok(Outcome::Absent);
        }
        if parts
            .iter()
            .zip(&in_old)
            .any(|(p, o)| *o && (hooks.is_in_use)(&p.old))
        {
            return Ok(Outcome::Left("in use"));
        }
        if entry.verify_hashes {
            for ((p, o), n) in parts.iter().zip(&in_old).zip(&in_shared) {
                // A file already in the shared folder was checked before it was moved.
                if *o && !*n {
                    match sha256_of(&p.old) {
                        Ok(hash) if hash.eq_ignore_ascii_case(p.file.sha256) => {}
                        Ok(_) => return Ok(Outcome::Left("hash mismatch")),
                        Err(_) => return Ok(Outcome::Left("in use")),
                    }
                }
            }
        }

        // The user marker first: from here on `collect` must not erase what is being moved in.
        self.add_user_as(model.id, self.app())?;
        fs::create_dir_all(self.model_dir(model.id))?;
        for ((p, o), n) in parts.iter().zip(&in_old).zip(&in_shared) {
            if *n {
                if *o {
                    // The shared folder has it: the old copy goes (the shared file is not touched).
                    match fs::remove_file(&p.old) {
                        Ok(()) => {}
                        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                        Err(e) if is_busy(&e) => return Ok(Outcome::Left("in use")),
                        Err(_) => return Ok(Outcome::Left("move failed")),
                    }
                }
            } else if let Err(e) = fs::rename(&p.old, &p.new) {
                return Ok(Outcome::Left(if is_busy(&e) {
                    "in use"
                } else {
                    "move failed"
                }));
            }
            // So an older version of the app that still reads its old folder keeps working. If the link cannot be
            // made the old path stays empty; the model is in the shared folder all the same.
            let _ = fs::hard_link(&p.new, &p.old);
        }
        Ok(Outcome::Done)
    }

    /// Spec 5.4 item 2 for one model in another app's old folder: hard links, never a move.
    fn link_from_other(
        &self,
        src: &LegacySource,
        entry: &LegacyEntry,
        model: &ModelInfo,
    ) -> io::Result<Outcome> {
        let Some(parts) = self.parts(src, entry, model) else {
            return Ok(Outcome::Absent);
        };
        if !parts.iter().all(|p| is_exact(&p.old, p.file.size)) {
            return Ok(Outcome::Absent);
        }
        let id = model.id;
        if parts.iter().all(|p| is_exact(&p.new, p.file.size)) {
            // Already shared: that app has the model, so it is a user, and that is all.
            self.add_user_as(id, src.app)?;
            return Ok(Outcome::Absent);
        }
        if entry.verify_hashes {
            for p in parts.iter().filter(|p| !is_exact(&p.new, p.file.size)) {
                match sha256_of(&p.old) {
                    Ok(hash) if hash.eq_ignore_ascii_case(p.file.sha256) => {}
                    Ok(_) => return Ok(Outcome::Left("hash mismatch")),
                    Err(_) => return Ok(Outcome::Left("link failed")),
                }
            }
        }

        let marker = self.marker(id, src.app);
        let had_marker = marker.exists();
        let dir = self.model_dir(id);
        let dir_existed = dir.exists();
        self.add_user_as(id, src.app)?;
        let mut made: Vec<&Path> = Vec::new();
        let linked = fs::create_dir_all(&dir).is_ok()
            && parts.iter().all(|p| {
                if is_exact(&p.new, p.file.size) {
                    return true; // an earlier, interrupted run linked this one
                }
                let ok = fs::hard_link(&p.old, &p.new).is_ok();
                if ok {
                    made.push(&p.new);
                }
                ok
            });
        if !linked {
            // Take back what this attempt did: no half model, no user for a model that is not there.
            for path in made {
                let _ = fs::remove_file(path);
            }
            if !had_marker {
                let _ = fs::remove_file(&marker);
                let _ = fs::remove_dir(self.users_dir(id));
            }
            if !dir_existed {
                let _ = fs::remove_dir(&dir);
            }
            return Ok(Outcome::Left("link failed"));
        }
        Ok(Outcome::Done)
    }
}

fn push_once(list: &mut Vec<String>, id: &str) {
    if !list.iter().any(|x| x == id) {
        list.push(id.to_string());
    }
}

fn push_reason(list: &mut Vec<(String, String)>, id: &str, reason: &str) {
    if !list.iter().any(|(i, r)| i == id && r == reason) {
        list.push((id.to_string(), reason.to_string()));
    }
}

/// The `*.gguf` files directly in `src`'s imports folder that no entry claims, with their import ids. Files that
/// cannot be read are left out.
fn dropped_ggufs(src: &LegacySource, catalog: &[&ModelInfo]) -> Vec<ImportedGguf> {
    let Some(rel) = src.imports_rel else {
        return Vec::new();
    };
    let dir = src.root.join(rel);
    let mut claimed: Vec<&str> = Vec::new();
    for entry in src.entries.iter().filter(|e| e.rel == rel) {
        claimed.extend(entry.files.iter().map(|(disk, _)| *disk));
        if entry.files.is_empty()
            && let Some(model) = catalog.iter().find(|m| m.id == entry.id)
        {
            claimed.extend(model.files.iter().map(|f| f.name));
        }
    }
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let is_gguf = name.to_ascii_lowercase().ends_with(".gguf");
            (is_gguf && !claimed.contains(&name.as_str())).then(|| e.path())
        })
        .filter_map(|path| {
            let id = import_id(&path).ok()?;
            Some(ImportedGguf { path, id })
        })
        .collect()
}
