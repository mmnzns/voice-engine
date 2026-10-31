//! The shared model folder: one folder for every app (spec 5.1 to 5.3). Layout under the store root:
//!
//! ```text
//! <root>/<model id>/<files>      <root>/<model id>/<file>.part      <root>/<model id>/.cancelled
//! <root>/.engine/users/<model id>/<app>      empty file: this app uses this model
//! <root>/.engine/locks/<model id>.use        shared lock while an app has the model loaded
//! <root>/.engine/locks/<model id>.download   exclusive lock while one app downloads the model
//! <root>/.engine/moved/<app>                 empty file: this app's one-time move is done
//! ```
//!
//! Locks are OS file locks (`File::lock_shared`, `File::try_lock`) on files this module opens itself, so they
//! conflict between two `SharedStore`s in one process just as they do between two apps, and the OS releases them
//! if an app crashes. Lock files are never deleted (deleting one while another process has it open would let a
//! third process lock a new file at the same path).

use crate::catalog::{self, ModelInfo};
use crate::download::{self, DownloadError, IDLE, Progress};
use crate::legacy::App;
use std::collections::HashSet;
use std::fmt;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::thread;
use std::time::{Duration, SystemTime};

/// Where a model is, as `SharedStore::status` sees it.
#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    /// Every catalog file is there with its exact size.
    Installed {
        dir: PathBuf,
        users: Vec<App>,
        used_by_me: bool,
    },
    /// Some app (maybe this one) holds the download lock.
    Downloading {
        fraction: f32,
        bytes: u64,
        by_me: bool,
    },
    /// Files or `.part` files are there, nobody is downloading. `cancelled`: the person stopped it.
    Partial {
        fraction: f32,
        bytes: u64,
        cancelled: bool,
    },
    Missing,
}

/// What `release` did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Released {
    /// This was the last user and nothing had the model open: the folder is gone.
    Erased,
    /// The folder stays. `users` are the apps that still use it; empty when another app has the model loaded (or
    /// is downloading it) right now and `collect` will erase it later.
    Kept { users: Vec<App> },
    /// This app was not a user of the model and there was nothing of its own to remove.
    NotUsing,
}

/// What `remove_everywhere` did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Removed {
    /// The model's folder and every app's marker are gone, and so are the old copies, except `old_copies_left`
    /// that could not be deleted (a file another program has open, on Windows).
    Erased { old_copies_left: usize },
    /// An app has the model loaded or is downloading it: nothing was touched.
    InUse,
    /// There was nothing of this model on this computer (no folder, no marker, no old copy).
    NotThere,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    UnknownModel(String),
    /// `hold` on a model that is not completely on this computer.
    NotInstalled(String),
    /// Another app (or another `SharedStore`) is downloading this model; read its progress from `status`.
    DownloadingElsewhere,
    Download(DownloadError),
    Io(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::UnknownModel(_) => write!(f, "not a model in the catalog"),
            StoreError::NotInstalled(_) => write!(f, "the model is not on this computer"),
            StoreError::DownloadingElsewhere => write!(f, "another app is downloading this model"),
            StoreError::Download(e) => write!(f, "{e}"),
            StoreError::Io(message) => write!(f, "model folder error: {message}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<io::Error> for StoreError {
    fn from(e: io::Error) -> Self {
        // Only the kind: OS messages can carry paths.
        StoreError::Io(e.kind().to_string())
    }
}

impl From<DownloadError> for StoreError {
    fn from(e: DownloadError) -> Self {
        StoreError::Download(e)
    }
}

/// A model that an app has loaded. Holds a shared lock on `locks/<id>.use` until dropped, so no other app can
/// erase the folder (and on Windows the files can't be deleted) while it lives.
#[derive(Debug)]
pub struct ModelHandle {
    dir: PathBuf,
    _lock: File,
}

impl ModelHandle {
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// One app's view of the shared folder.
#[derive(Debug)]
pub struct SharedStore {
    app: App,
    root: PathBuf,
    /// Where the apps' old folders are. None for a store opened with `open_at` (or the models-dir override): such a
    /// store never moves anything in.
    pub(crate) old_folders: Option<OldFolders>,
    /// Made by `read_only`: looks, never writes.
    read_only: bool,
    /// Models this instance is downloading right now (`Downloading.by_me`).
    downloading: Mutex<HashSet<String>>,
}

/// Where this computer keeps the apps' data, for the one-time move.
#[derive(Debug, Clone)]
pub(crate) struct OldFolders {
    pub(crate) local_data: PathBuf,
    pub(crate) roaming: Option<PathBuf>,
    pub(crate) windows: bool,
}

/// Part files this old with no download running are leftovers (`collect`).
const STALE_PART: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// A download that finds the lock taken tries a few more times before it says "elsewhere", because `status` takes
/// the lock for an instant to see whether anybody holds it.
const LOCK_TRIES: u32 = 5;
const LOCK_RETRY: Duration = Duration::from_millis(25);

pub(crate) fn app_name(app: App) -> &'static str {
    match app {
        App::Talaan => "talaan",
        App::Ensayo => "ensayo",
        App::Parlato => "parlato",
    }
}

/// A model id is one plain path component. Anything else (`..`, `.engine`, a separator) is refused before it can
/// reach a path that gets erased.
pub(crate) fn valid_id(id: &str) -> bool {
    let mut chars = id.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphanumeric())
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
}

fn invalid_id() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "not a valid model id")
}

/// `try_lock` as a plain answer: Ok(true) got it, Ok(false) someone else holds it.
fn try_exclusive(file: &File) -> io::Result<bool> {
    match file.try_lock() {
        Ok(()) => Ok(true),
        Err(TryLockError::WouldBlock) => Ok(false),
        Err(TryLockError::Error(e)) => Err(e),
    }
}

/// What `erase_if_unused` found.
enum Erase {
    Done,
    /// Another app has the model loaded or is downloading it.
    Busy,
    /// Somebody became a user in the meantime.
    Users(Vec<App>),
}

impl SharedStore {
    /// The shared folder (`default_dir`), or `VOICE_ENGINE_MODELS_DIR` when that is set, which is treated exactly
    /// like `open_at`.
    pub fn open(app: App) -> io::Result<SharedStore> {
        if let Some(dir) = std::env::var_os("VOICE_ENGINE_MODELS_DIR").filter(|d| !d.is_empty()) {
            return SharedStore::open_at(app, PathBuf::from(dir));
        }
        let dir = SharedStore::default_dir()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no local app data folder"))?;
        let mut store = SharedStore::open_at(app, dir)?;
        // Only the real folder moves anything in; `local_data` exists because `default_dir` found it.
        store.old_folders = dirs::data_local_dir().map(|local_data| OldFolders {
            local_data,
            roaming: dirs::data_dir(),
            windows: cfg!(target_os = "windows"),
        });
        Ok(store)
    }

    /// A store at `shared` that moves models in from old app folders under `local_data` (and `roaming`), laid out
    /// as on Windows or the Mac. For tests of `move_in`; apps use `open`.
    #[doc(hidden)]
    pub fn for_move_test(
        app: App,
        shared: &Path,
        local_data: &Path,
        roaming: Option<&Path>,
        windows: bool,
    ) -> io::Result<SharedStore> {
        let mut store = SharedStore::open_at(app, shared.to_path_buf())?;
        store.old_folders = Some(OldFolders {
            local_data: local_data.to_path_buf(),
            roaming: roaming.map(Path::to_path_buf),
            windows,
        });
        Ok(store)
    }

    /// Any folder the app passes in. Creates `dir/.engine`; nothing outside `dir` is read or written.
    pub fn open_at(app: App, dir: PathBuf) -> io::Result<SharedStore> {
        for sub in ["users", "locks", "moved"] {
            fs::create_dir_all(dir.join(".engine").join(sub))?;
        }
        Ok(SharedStore {
            app,
            root: dir,
            old_folders: None,
            read_only: false,
            downloading: Mutex::default(),
        })
    }

    /// A view of the folder at `dir` that creates nothing, not even `dir` or `.engine`, for showing "what is on this
    /// computer" (`status`, `list`, `root`, `model_dir`) without touching the folder. A folder that does not exist
    /// reads as every model missing. Every call that would write (`use_model`, `download`, `hold`, `release`,
    /// `release_all`, `collect`, `move_in`) is refused with `PermissionDenied`, or does nothing for `move_in`.
    pub fn read_only(app: App, dir: PathBuf) -> SharedStore {
        SharedStore {
            app,
            root: dir,
            old_folders: None,
            read_only: true,
            downloading: Mutex::default(),
        }
    }

    fn writable(&self) -> io::Result<()> {
        if self.read_only {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "this view of the model folder is read only",
            ))
        } else {
            Ok(())
        }
    }

    /// Mac: `<Application Support>/Craft Concepts Digital/Models`; Windows:
    /// `%LOCALAPPDATA%\Craft Concepts Digital\Models`. Only builds the path.
    pub fn default_dir() -> Option<PathBuf> {
        dirs::data_local_dir().map(|d| d.join("Craft Concepts Digital").join("Models"))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn model_dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }

    pub(crate) fn app(&self) -> App {
        self.app
    }

    fn engine(&self) -> PathBuf {
        self.root.join(".engine")
    }

    pub(crate) fn users_dir(&self, id: &str) -> PathBuf {
        self.engine().join("users").join(id)
    }

    pub(crate) fn marker(&self, id: &str, app: App) -> PathBuf {
        self.users_dir(id).join(app_name(app))
    }

    fn lock_path(&self, id: &str, kind: &str) -> PathBuf {
        self.engine().join("locks").join(format!("{id}.{kind}"))
    }

    /// Opens (creating it if needed) a lock file.
    fn lock_file(&self, id: &str, kind: &str) -> io::Result<File> {
        self.writable()?;
        let path = self.lock_path(id, kind);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
    }

    /// The apps that use `id`, in `App::ALL` order.
    fn users(&self, id: &str) -> Vec<App> {
        App::ALL
            .into_iter()
            .filter(|a| self.marker(id, *a).is_file())
            .collect()
    }

    /// Writes this app's user marker. Holds the `.use` lock shared for the instant, so an erase (which holds it
    /// exclusively while it re-checks the users) either sees this marker or finishes before it is written.
    fn add_user(&self, id: &str) -> io::Result<()> {
        self.add_user_as(id, self.app)
    }

    /// `add_user` for any app: the one-time move marks the apps whose old folders hold a model.
    pub(crate) fn add_user_as(&self, id: &str, app: App) -> io::Result<()> {
        self.writable()?;
        if !valid_id(id) {
            return Err(invalid_id());
        }
        let lock = self.lock_file(id, "use")?;
        lock.lock_shared()?;
        fs::create_dir_all(self.users_dir(id))?;
        File::create(self.marker(id, app))?;
        Ok(())
    }

    fn is_downloading_now(&self, id: &str) -> bool {
        // Read-only: no lock file means nobody ever downloaded this model.
        let Ok(file) = OpenOptions::new()
            .read(true)
            .write(true)
            .open(self.lock_path(id, "download"))
        else {
            return false;
        };
        match try_exclusive(&file) {
            Ok(true) => false, // dropping `file` releases the probe
            Ok(false) => true,
            Err(_) => false,
        }
    }

    pub fn status(&self, id: &str) -> Status {
        match catalog::find(id) {
            Some(model) => self.status_with(model),
            None => Status::Missing,
        }
    }

    /// `status` for a model that is not (or not as it is) in the catalog. Tests build a `ModelInfo` whose files
    /// point at a local server.
    #[doc(hidden)]
    pub fn status_with(&self, model: &ModelInfo) -> Status {
        if !valid_id(model.id) {
            return Status::Missing;
        }
        let dir = self.model_dir(model.id);
        let total = model.size();
        let complete = |name: &str, size: u64| {
            fs::metadata(dir.join(name))
                .map(|m| m.is_file() && m.len() == size)
                .unwrap_or(false)
        };
        if !model.files.is_empty() && model.files.iter().all(|f| complete(f.name, f.size)) {
            let users = self.users(model.id);
            let used_by_me = users.contains(&self.app);
            return Status::Installed {
                dir,
                users,
                used_by_me,
            };
        }
        let bytes = download::partial_bytes(model.files, &dir);
        let fraction = if total == 0 {
            0.0
        } else {
            (bytes as f64 / total as f64) as f32
        };
        if self.is_downloading_now(model.id) {
            let by_me = self.downloading.lock().unwrap().contains(model.id);
            return Status::Downloading {
                fraction,
                bytes,
                by_me,
            };
        }
        let has_part = model
            .files
            .iter()
            .any(|f| dir.join(format!("{}.part", f.name)).exists());
        if bytes > 0 || has_part {
            return Status::Partial {
                fraction,
                bytes,
                cancelled: download::was_cancelled(&dir),
            };
        }
        Status::Missing
    }

    /// Every catalog model, in catalog order, with where it is: "what is on this computer".
    pub fn list(&self) -> Vec<(&'static ModelInfo, Status)> {
        catalog::all()
            .iter()
            .map(|m| (m, self.status_with(m)))
            .collect()
    }

    /// This app uses the model: writes `users/<id>/<app>`. "Already on this computer" is one call of this.
    pub fn use_model(&self, id: &str) -> Result<(), StoreError> {
        if catalog::find(id).is_none() {
            return Err(StoreError::UnknownModel(id.into()));
        }
        Ok(self.add_user(id)?)
    }

    /// Downloads a catalog model into the shared folder and returns its folder. `Cancelled` and `Stalled` come back
    /// as `StoreError::Download(..)`; `DownloadError::is_failure` tells them from real failures.
    pub fn download(
        &self,
        id: &str,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<PathBuf, StoreError> {
        let model = catalog::find(id).ok_or_else(|| StoreError::UnknownModel(id.into()))?;
        self.download_with(model, cancel, IDLE, progress)
    }

    /// What `download` does, for any `ModelInfo` and with the stall time given: the tests download a locally built
    /// model from a local server. Writes the user marker first, then takes the download lock exclusively.
    #[doc(hidden)]
    pub fn download_with(
        &self,
        model: &ModelInfo,
        cancel: &AtomicBool,
        idle: Duration,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<PathBuf, StoreError> {
        let id = model.id;
        self.add_user(id)?;
        let lock = self.lock_file(id, "download")?;
        let mut tries = 0;
        while !try_exclusive(&lock)? {
            tries += 1;
            if tries >= LOCK_TRIES {
                return Err(StoreError::DownloadingElsewhere);
            }
            thread::sleep(LOCK_RETRY);
        }
        let _mine = InProcess::new(&self.downloading, id);
        let dir = self.model_dir(id);
        download::download_files(model.files, &dir, cancel, idle, progress)?;
        Ok(dir)
    }

    /// The model is loaded: holds a shared lock on `locks/<id>.use` until the handle is dropped. The app keeps it
    /// next to its loaded model. Fails when the model is not completely on this computer.
    pub fn hold(&self, id: &str) -> Result<ModelHandle, StoreError> {
        let model = catalog::find(id).ok_or_else(|| StoreError::UnknownModel(id.into()))?;
        let lock = self.lock_file(id, "use")?;
        lock.lock_shared()?;
        // Looked at after the lock is taken: an erase that got in first has already finished.
        match self.status_with(model) {
            Status::Installed { dir, .. } => Ok(ModelHandle { dir, _lock: lock }),
            _ => Err(StoreError::NotInstalled(id.into())),
        }
    }

    /// This app no longer uses the model. The last user to leave erases the folder, unless another app has the
    /// model loaded (or is downloading it): then it stays for `collect` (`Kept { users: [] }`).
    ///
    /// `NotUsing`: this app had no marker, and either other apps use the model or there is no folder at all.
    /// With no marker and no users but a folder (a leftover), the leftover is erased like any unused model.
    pub fn release(&self, id: &str) -> io::Result<Released> {
        self.writable()?;
        if !valid_id(id) {
            return Err(invalid_id());
        }
        let had_marker = match fs::remove_file(self.marker(id, self.app)) {
            Ok(()) => true,
            Err(e) if e.kind() == io::ErrorKind::NotFound => false,
            Err(e) => return Err(e),
        };
        let users = self.users(id);
        if !had_marker && (!users.is_empty() || !self.model_dir(id).exists()) {
            return Ok(Released::NotUsing);
        }
        if !users.is_empty() {
            return Ok(Released::Kept { users });
        }
        Ok(match self.erase_if_unused(id)? {
            Erase::Done => Released::Erased,
            Erase::Busy => Released::Kept { users: Vec::new() },
            Erase::Users(users) => Released::Kept { users },
        })
    }

    /// The "Remove from this computer" button: erases the model for every app at once, unlike `release`, which
    /// only lets go for this app. The model's folder, every app's marker and the old copies `move_in` left in the
    /// apps' old folders (hard links or full copies, which keep the bytes on disk) all go. The other apps then
    /// see the model as `Missing`, with a Download button. An old file is deleted only when its size is the
    /// catalog's; the old folder goes only when it is empty. Ids that are not in the catalog (imported GGUF files
    /// tracked by an app) have no old copies: their folder and markers go.
    ///
    /// Show a confirmation first, naming the other apps from `Status::Installed { users, .. }` ("Also used by ...").
    ///
    /// `InUse` when any app has the model loaded or is downloading it, nothing touched. This app's own
    /// `ModelHandle` counts too, so unload the model first. Refused with `PermissionDenied` on a `read_only` view
    /// and `InvalidInput` for an id that is not one plain folder name. A store from `open_at` never looks at old
    /// folders. An old copy that cannot be deleted (a file another program has open, on Windows) is counted in
    /// `old_copies_left`; the rest is still erased.
    pub fn remove_everywhere(&self, id: &str) -> io::Result<Removed> {
        self.writable()?;
        if !valid_id(id) {
            return Err(invalid_id());
        }
        let use_lock = self.lock_file(id, "use")?;
        if !try_exclusive(&use_lock)? {
            return Ok(Removed::InUse);
        }
        let download_lock = self.lock_file(id, "download")?;
        if !try_exclusive(&download_lock)? {
            return Ok(Removed::InUse);
        }
        let mut found = false;
        for dir in [self.users_dir(id), self.model_dir(id)] {
            match fs::remove_dir_all(dir) {
                Ok(()) => found = true,
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        let (old_found, old_copies_left) = self.remove_old_copies(id);
        if !found && old_found == 0 {
            return Ok(Removed::NotThere);
        }
        Ok(Removed::Erased { old_copies_left })
    }

    /// `release` for every model this app uses: "Delete all data" in an app calls this instead of removing a
    /// folder. Goes on after an error and returns the first one.
    pub fn release_all(&self) -> io::Result<()> {
        self.writable()?;
        let mut first_error = None;
        for id in self.tracked_ids() {
            if !self.marker(&id, self.app).is_file() {
                continue;
            }
            if let Err(e) = self.release(&id) {
                first_error.get_or_insert(e);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Erases models that no app uses and no app has loaded, and `.part` files older than 7 days that no download
    /// is writing. Only touches model folders (catalog ids and ids some app has used) and `.engine` inside the
    /// store root. Each app calls it at start. Goes on after an error and returns the first one.
    pub fn collect(&self) -> io::Result<()> {
        self.writable()?;
        let mut first_error = None;
        let mut note = |r: io::Result<()>| {
            if let Err(e) = r {
                first_error.get_or_insert(e);
            }
        };
        let mut candidates: Vec<String> = catalog::all().iter().map(|m| m.id.to_string()).collect();
        for id in self.tracked_ids() {
            if !candidates.contains(&id) {
                candidates.push(id);
            }
        }
        for id in candidates {
            if !self.users(&id).is_empty() {
                note(self.remove_stale_parts(&id));
            } else if self.model_dir(&id).exists() {
                note(self.erase_if_unused(&id).map(|_| ()));
            } else {
                // Nothing left but an empty users folder.
                let _ = fs::remove_dir(self.users_dir(&id));
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Ids that have a `users/<id>/` folder, i.e. that some app has used.
    fn tracked_ids(&self) -> Vec<String> {
        let Ok(entries) = fs::read_dir(self.engine().join("users")) else {
            return Vec::new();
        };
        let mut ids: Vec<String> = entries
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|id| valid_id(id))
            .collect();
        ids.sort();
        ids
    }

    /// Erases `<id>/` and the users folder if, under an exclusive `.use` lock (nobody has the model loaded) and
    /// the `.download` lock (nobody is downloading it), the users are still empty.
    fn erase_if_unused(&self, id: &str) -> io::Result<Erase> {
        let use_lock = self.lock_file(id, "use")?;
        if !try_exclusive(&use_lock)? {
            return Ok(Erase::Busy);
        }
        let download_lock = self.lock_file(id, "download")?;
        if !try_exclusive(&download_lock)? {
            return Ok(Erase::Busy);
        }
        let users = self.users(id);
        if !users.is_empty() {
            return Ok(Erase::Users(users));
        }
        match fs::remove_dir_all(self.model_dir(id)) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        match fs::remove_dir_all(self.users_dir(id)) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        Ok(Erase::Done)
    }

    /// Deletes `.part` files older than 7 days in `<id>/` unless a download holds the lock.
    fn remove_stale_parts(&self, id: &str) -> io::Result<()> {
        let Ok(entries) = fs::read_dir(self.model_dir(id)) else {
            return Ok(());
        };
        let now = SystemTime::now();
        let stale: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().ends_with(".part"))
            .filter(|e| {
                e.metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| now.duration_since(t).ok())
                    .is_some_and(|age| age > STALE_PART)
            })
            .map(|e| e.path())
            .collect();
        if stale.is_empty() {
            return Ok(());
        }
        let lock = self.lock_file(id, "download")?;
        if !try_exclusive(&lock)? {
            return Ok(());
        }
        for path in stale {
            match fs::remove_file(path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

/// Marks a model as downloading by this instance for as long as it lives.
struct InProcess<'a> {
    set: &'a Mutex<HashSet<String>>,
    id: String,
}

impl<'a> InProcess<'a> {
    fn new(set: &'a Mutex<HashSet<String>>, id: &str) -> Self {
        set.lock().unwrap().insert(id.to_string());
        InProcess {
            set,
            id: id.to_string(),
        }
    }
}

impl Drop for InProcess<'_> {
    fn drop(&mut self) {
        self.set.lock().unwrap().remove(&self.id);
    }
}
