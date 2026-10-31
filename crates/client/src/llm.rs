//! The AI helper from the app's side. One `LlmClient` per app: it starts the helper on the first request, keeps it
//! while the app runs, restarts it when it dies, and turns every failure into an [`EngineError`].
//!
//! Calls block and are safe from any worker thread; one request runs at a time (the helper runs one at a time too).
//! `cancel` and `shutdown` may be called from another thread while a request runs.

use crate::error::EngineError;
use crate::gpu_memory;
use crate::process::{Process, Recv, StartFailure, write_line};
use std::path::PathBuf;
use std::process::ChildStdin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use voice_engine_protocol::{
    Backend, Completion, Configure, Device, DeviceList, Embedding, Event, Fallback, Inspect,
    Inspection, Load, PROTOCOL_VERSION, Phase, Reply, Request, encode,
};

/// How long a helper that was asked to stop may keep going before it is killed.
pub(crate) const CANCEL_GRACE: Duration = Duration::from_secs(2);
/// A helper that dies at startup is started again this many times in all (Metal's shader cache can crash one when
/// several start at the same moment).
pub(crate) const START_TRIES: u32 = 3;
pub(crate) const START_RETRY_PAUSE: Duration = Duration::from_millis(300);
/// How long `devices` waits for the helper's answer (stall and total alike): an app asks at startup, so a helper that
/// never answers must not hold it for the general limits.
pub(crate) const DEVICES_LIMIT: Duration = Duration::from_secs(10);
/// How often a waiting request looks at its limits and at `cancel`.
pub(crate) const POLL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone)]
pub struct Limits {
    /// The helper must answer its first ping within this.
    pub start: Duration,
    /// No event at all for this long is a stall.
    pub stall: Duration,
    /// A whole request may take this long.
    pub total: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            start: Duration::from_secs(60),
            stall: Duration::from_secs(120),
            total: Duration::from_secs(30 * 60),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// The processor helper (every platform).
    pub cpu_helper: PathBuf,
    /// The Windows Vulkan helper; tried first when present, until it fails once.
    pub gpu_helper: Option<PathBuf>,
    /// Environment for both helpers (for example `TALAAN_GPU`).
    pub env: Vec<(String, String)>,
    /// Environment for the GPU helper only (for example `VK_LOADER_LAYERS_DISABLE=~implicit~`).
    pub gpu_env: Vec<(String, String)>,
    /// A file remembering that the graphics card failed, so it isn't tried again on every start.
    pub gpu_failures: Option<PathBuf>,
    /// What the failure is remembered for: the app version and the graphics driver, so an update or a new driver
    /// gets a fresh try.
    pub gpu_key: String,
    /// Seconds of idle before the helper exits; None: it runs while the app runs.
    pub idle_secs: Option<u64>,
    pub limits: Limits,
}

/// A successful call: the value, where it ran (what the helper said for this request when it said anything: a
/// request with `gpu: false` runs on the processor in a helper that started on Metal, and an older helper that says
/// nothing counts as what it said when it started), and why the card was not used.
///
/// `fallback` is `Some` only when the app asked for the card (`gpu: true`) and the work ran on the processor, and says
/// why. `CardFailed` and `CardOutOfMemory` are about this call: the app says "graphics card failed, used processor"
/// once and stops asking for the card. After that, and in a new run that reads the `gpu_failures` file, it is
/// `RememberedFailure`, which needs no message. `NoCardHelper` and `NoCardPath` are not failures at all. A request
/// with `gpu: false`, and any request that ran on the card, has `None`.
#[derive(Debug, Clone)]
pub struct Outcome<T> {
    pub value: T,
    pub backend: Backend,
    pub fallback: Option<Fallback>,
}

impl<T> Outcome<T> {
    /// The app asked for the card and the processor did the work (for any reason: see `fallback`).
    pub fn fell_back(&self) -> bool {
        self.fallback.is_some()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadInfo {
    /// Seconds the load took; 0 when the model was already loaded.
    pub seconds: f32,
    /// The longest context the model was trained for.
    pub n_ctx_train: u32,
}

struct Running {
    process: Process,
    backend: Backend,
    on_gpu: bool,
    /// This is the processor helper, running because the card helper failed (now or in an earlier run) or could not
    /// start. On the Mac the two helpers can be the same binary, which would load on Metal again for a request that
    /// still says `gpu: true`, so every request sent to it says `gpu: false`.
    processor_fallback: bool,
    /// This helper came from the card helper's path (`gpu_helper`), whatever it then said it runs on.
    card_helper: bool,
    /// The card helper just failed to start for this helper's `start`: the first call to use it reports that (later
    /// calls report the failure as remembered).
    fresh_failure: Option<Fallback>,
}

fn reply_of(ev: &Event) -> Option<Reply> {
    match ev {
        Event::Done(reply) => Some(reply.clone()),
        _ => None,
    }
}

/// What holds a request to a limit beyond the config's.
#[derive(Clone, Copy, Default)]
struct Limit {
    /// The stall and total limits are capped at this, from the send (`devices`: an app asks at startup).
    cap: Option<Duration>,
    /// The reply must be done this long after the model is loaded (`complete_within`).
    reply: Option<Duration>,
}

impl Limit {
    fn capped(cap: Duration) -> Self {
        Limit {
            cap: Some(cap),
            reply: None,
        }
    }
}

/// Why one attempt at a request ended without an answer.
enum Attempt {
    /// The helper process is gone. `answered`: it had sent at least one event for this request.
    Died { answered: bool },
    /// The helper said no, or a limit was hit. `kill`: the helper is in an unknown state and must be stopped.
    Failed { error: EngineError, kill: bool },
}

pub struct LlmClient {
    config: Mutex<LlmConfig>,
    /// Held for the whole of a request.
    running: Mutex<Option<Running>>,
    /// The running helper's stdin, so `cancel` can write to it while `running` is held by a request.
    live_stdin: Mutex<Option<Arc<Mutex<ChildStdin>>>>,
    /// When `cancel` was last asked for during the current request.
    cancel_asked: Mutex<Option<Instant>>,
    /// The graphics card failed in this process; never try it again until the app restarts.
    gpu_disabled: AtomicBool,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Self {
        LlmClient {
            config: Mutex::new(config),
            running: Mutex::new(None),
            live_stdin: Mutex::new(None),
            cancel_asked: Mutex::new(None),
            gpu_disabled: AtomicBool::new(false),
        }
    }

    /// Loads a model now, so the next request doesn't wait. A different path replaces the loaded model.
    pub fn load(
        &self,
        model_path: &str,
        n_ctx: u32,
        gpu: bool,
        progress: &mut dyn FnMut(&Event),
    ) -> Result<Outcome<LoadInfo>, EngineError> {
        let request = Request::Load(Load {
            model_path: model_path.into(),
            n_ctx,
            gpu,
        });
        self.exchange(&request, progress, &|ev| match ev {
            Event::Loaded {
                seconds,
                n_ctx_train,
                ..
            } => Some(LoadInfo {
                seconds: *seconds,
                n_ctx_train: *n_ctx_train,
            }),
            _ => None,
        })
    }

    pub fn complete(
        &self,
        completion: Completion,
        progress: &mut dyn FnMut(&Event),
    ) -> Result<Outcome<Reply>, EngineError> {
        self.exchange(&Request::Complete(completion), progress, &reply_of)
    }

    /// `complete` with its own limit on the reply: `limit` runs from the moment the model is loaded (from the send
    /// when it already is), so a short limit for a short reply does not also cut a slow model load, which keeps the
    /// config's limits. When the reply is not done within `limit` the call ends with `EngineError::TimedOut` and the
    /// helper is stopped, as for a stall: the model is gone, and the next call loads it again.
    pub fn complete_within(
        &self,
        limit: Duration,
        completion: Completion,
        progress: &mut dyn FnMut(&Event),
    ) -> Result<Outcome<Reply>, EngineError> {
        let limit = Limit {
            cap: None,
            reply: Some(limit),
        };
        self.exchange_with(limit, &Request::Complete(completion), progress, &reply_of)
    }

    pub fn embed(&self, embedding: Embedding) -> Result<Outcome<Vec<Vec<f32>>>, EngineError> {
        self.exchange(&Request::Embed(embedding), &mut |_| {}, &|ev| match ev {
            Event::Embedded { vectors, .. } => Some(vectors.clone()),
            _ => None,
        })
    }

    /// What a GGUF file says about itself (chat template, thinking, trained context), without loading it.
    pub fn inspect(&self, model_path: &str) -> Result<Inspection, EngineError> {
        let request = Request::Inspect(Inspect {
            model_path: model_path.into(),
        });
        self.exchange(&request, &mut |_| {}, &|ev| match ev {
            Event::Inspected(i) => Some(i.clone()),
            _ => None,
        })
        .map(|o| o.value)
    }

    /// Every device the helper sees and the one it uses. Asked of the graphics card helper when there is one and it
    /// hasn't failed (else of the processor helper, which lists the processor only). Starts the helper if none runs,
    /// so an app asks once at startup, before the first load.
    pub fn devices(&self) -> Result<DeviceList, EngineError> {
        self.exchange_with(
            Limit::capped(DEVICES_LIMIT),
            &Request::Devices,
            &mut |_| {},
            &|ev| match ev {
                Event::Devices { devices, chosen } => Some(DeviceList {
                    devices: devices.clone(),
                    chosen: *chosen,
                }),
                _ => None,
            },
        )
        .map(|o| o.value)
    }

    /// The discrete graphics card the engine would use on this PC, or None: what an app on Windows asks before the
    /// first load to decide whether the card is on by default (see [`DeviceList::discrete_card`]).
    pub fn discrete_card(&self) -> Result<Option<Device>, EngineError> {
        Ok(self.devices()?.discrete_card().cloned())
    }

    /// Frees the loaded model's memory now. With no helper running there is nothing to free, and none is started.
    pub fn unload(&self) -> Result<(), EngineError> {
        if self
            .running
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_none()
        {
            return Ok(());
        }
        self.exchange(&Request::Unload, &mut |_| {}, &|ev| match ev {
            Event::Unloaded => Some(()),
            _ => None,
        })
        .map(|_| ())
    }

    /// Changes how long the helper waits without a request before exiting (None: until the app exits), now and
    /// for every helper started later. Takes effect without a restart and without unloading the model.
    pub fn set_idle(&self, idle_secs: Option<u64>) -> Result<(), EngineError> {
        self.config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .idle_secs = idle_secs;
        if self
            .running
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_none()
        {
            return Ok(());
        }
        let request = Request::Configure(Configure { idle_secs });
        self.exchange(&request, &mut |_| {}, &|ev| match ev {
            Event::Configured { .. } => Some(()),
            _ => None,
        })
        .map(|_| ())
    }

    /// Stops the helper now (the app is closing, or wants the memory back).
    pub fn shutdown(&self) {
        // Kill through the stdin handle's owner: take the process out of `running` when it is free; when a request
        // holds it, closing happens when that request ends.
        if let Ok(mut running) = self.running.try_lock()
            && let Some(mut r) = running.take()
        {
            r.process.kill();
        }
    }

    /// The app's "Try again" for the graphics card on a live client: forgets that the card failed (in this client and
    /// in the `gpu_failures` file, when one is configured) so the next request tries the card again. It waits for
    /// a request in progress to finish. When the helper running now is the processor one that took over after the
    /// card failed, it is stopped, so the next request starts a new helper on the card (the model loads again there
    /// with that request); when the card helper is running, it is left alone. The card can fail again, and that is
    /// remembered again. Calling `clear_gpu_failure` alone does not reach a running client.
    pub fn retry_gpu(&self) {
        let mut running = self.running.lock().unwrap_or_else(|e| e.into_inner());
        {
            let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(path) = cfg.gpu_failures.as_deref() {
                gpu_memory::clear_gpu_failure(path);
            }
        }
        self.gpu_disabled.store(false, Ordering::Relaxed);
        if running.as_ref().is_some_and(|r| r.processor_fallback)
            && let Some(mut r) = running.take()
        {
            r.process.kill();
        }
    }

    // --- the core -----------------------------------------------------------------------------------------------

    /// Sends `request` and waits for the event `pick` accepts, passing every other event to `progress`.
    fn exchange<T>(
        &self,
        request: &Request,
        progress: &mut dyn FnMut(&Event),
        pick: &dyn Fn(&Event) -> Option<T>,
    ) -> Result<Outcome<T>, EngineError> {
        self.exchange_with(Limit::default(), request, progress, pick)
    }

    /// `exchange`, with the extra limits in `limit`.
    fn exchange_with<T>(
        &self,
        limit: Limit,
        request: &Request,
        progress: &mut dyn FnMut(&Event),
        pick: &dyn Fn(&Event) -> Option<T>,
    ) -> Result<Outcome<T>, EngineError> {
        let mut running = self.running.lock().unwrap_or_else(|e| e.into_inner());
        // Why the card was given up on during this call, when it was.
        let mut failed_now: Option<Fallback> = None;
        let mut retried = false;
        loop {
            if running.is_none() {
                *running = Some(self.start()?);
            }
            let r = running.as_mut().expect("started above");
            if let Some(why) = r.fresh_failure.take() {
                failed_now = failed_now.or(Some(why));
            }
            *self.live_stdin.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(r.process.stdin_handle());
            *self.cancel_asked.lock().unwrap_or_else(|e| e.into_inner()) = None;
            let sent = if r.processor_fallback {
                processor_only(request)
            } else {
                request.clone()
            };
            let attempt = self.attempt(r, &sent, limit, progress, pick);
            *self.live_stdin.lock().unwrap_or_else(|e| e.into_inner()) = None;
            match attempt {
                Ok((value, ran)) => {
                    let backend = ran.unwrap_or_else(|| r.backend.clone());
                    let fallback = fallback_reason(request, &backend, failed_now, r);
                    return Ok(Outcome {
                        value,
                        backend,
                        fallback,
                    });
                }
                Err(Attempt::Died { answered }) => {
                    let mut dead = running.take().expect("running above");
                    let death = dead.process.reap();
                    if dead.on_gpu {
                        // The graphics card helper died: remember it and do the work on the processor.
                        self.gpu_failed();
                        failed_now = Some(if death.out_of_memory() {
                            Fallback::CardOutOfMemory
                        } else {
                            Fallback::CardFailed
                        });
                        continue;
                    }
                    if !answered && !retried {
                        // An idle exit that raced the request: a fresh helper answers it.
                        retried = true;
                        continue;
                    }
                    return Err(EngineError::died(death));
                }
                Err(Attempt::Failed { error, kill }) => {
                    if matches!(error, EngineError::GpuFailed { .. }) && r.on_gpu {
                        let mut dead = running.take().expect("running above");
                        dead.process.kill();
                        self.gpu_failed();
                        failed_now = Some(Fallback::CardFailed);
                        continue;
                    }
                    if kill && let Some(mut dead) = running.take() {
                        dead.process.kill();
                    }
                    return Err(error);
                }
            }
        }
    }

    /// One try at a request on the running helper.
    fn attempt<T>(
        &self,
        r: &mut Running,
        request: &Request,
        extra: Limit,
        progress: &mut dyn FnMut(&Event),
        pick: &dyn Fn(&Event) -> Option<T>,
    ) -> Result<(T, Option<Backend>), Attempt> {
        if r.process.send(request).is_err() {
            return Err(Attempt::Died { answered: false });
        }
        let mut limits = self
            .config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .limits
            .clone();
        if let Some(cap) = extra.cap {
            limits.stall = limits.stall.min(cap);
            limits.total = limits.total.min(cap);
        }
        let began = Instant::now();
        let mut last_event = Instant::now();
        let mut answered = false;
        // The reply limit runs from the send, unless the helper says it is loading the model: then it runs from the
        // first event after that (the helper sends one as soon as the model is ready).
        let mut reply_by = extra.reply.map(|limit| began + limit);
        let mut loading = false;
        loop {
            // Checked on every pass, not only when the helper is quiet: a helper that keeps streaming progress while
            // ignoring `cancel` (or running past the total limit) must still be stopped.
            let asked = *self.cancel_asked.lock().unwrap_or_else(|e| e.into_inner());
            if asked.is_some_and(|t| t.elapsed() > CANCEL_GRACE) {
                // It was told to stop and didn't: stop it for good.
                return Err(Attempt::Failed {
                    error: EngineError::Cancelled,
                    kill: true,
                });
            }
            if last_event.elapsed() > limits.stall
                || began.elapsed() > limits.total
                || reply_by.is_some_and(|by| Instant::now() > by)
            {
                return Err(Attempt::Failed {
                    error: EngineError::TimedOut,
                    kill: true,
                });
            }
            match r.process.recv(POLL) {
                Recv::Event(Event::Error { message, error }) => {
                    return Err(Attempt::Failed {
                        error: EngineError::from_helper(error, message),
                        kill: false,
                    });
                }
                Recv::Event(ev) => {
                    answered = true;
                    last_event = Instant::now();
                    if let Some(limit) = extra.reply {
                        if matches!(
                            ev,
                            Event::Progress {
                                phase: Phase::Loading,
                                ..
                            }
                        ) {
                            loading = true;
                            reply_by = None;
                        } else if loading {
                            loading = false;
                            reply_by = Some(Instant::now() + limit);
                        }
                    }
                    match pick(&ev) {
                        Some(value) => return Ok((value, ran_on(&ev))),
                        None => progress(&ev),
                    }
                }
                Recv::Closed => return Err(Attempt::Died { answered }),
                Recv::Timeout => {}
            }
        }
    }

    /// Starts a helper: the graphics card one first when there is one and it hasn't failed, else the processor one.
    fn start(&self) -> Result<Running, EngineError> {
        let cfg = self
            .config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut fresh_failure = None;
        if let Some(gpu_path) = &cfg.gpu_helper
            && !self.gpu_disabled.load(Ordering::Relaxed)
            && !gpu_memory::failed(cfg.gpu_failures.as_deref(), &cfg.gpu_key)
        {
            let env: Vec<_> = cfg.env.iter().chain(&cfg.gpu_env).cloned().collect();
            match start_with_retries(gpu_path, &env, &cfg) {
                Ok((process, backend)) => {
                    let on_gpu = backend.on_gpu();
                    return Ok(Running {
                        process,
                        backend,
                        on_gpu,
                        processor_fallback: false,
                        card_helper: true,
                        fresh_failure: None,
                    });
                }
                // A version mismatch is the app's problem, not the card's.
                Err(e @ EngineError::VersionMismatch { .. }) => return Err(e),
                Err(_) => {
                    self.gpu_failed();
                    fresh_failure = Some(Fallback::CardFailed);
                }
            }
        }
        let (process, backend) = start_with_retries(&cfg.cpu_helper, &cfg.env, &cfg)?;
        Ok(Running {
            process,
            backend,
            on_gpu: false,
            // Without a card helper configured the one helper is the app's only choice (on the Mac it is the Metal
            // one): the app's `gpu` goes through as it is.
            processor_fallback: cfg.gpu_helper.is_some(),
            card_helper: false,
            fresh_failure,
        })
    }

    fn gpu_failed(&self) {
        self.gpu_disabled.store(true, Ordering::Relaxed);
        let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
        gpu_memory::record(cfg.gpu_failures.as_deref(), &cfg.gpu_key);
    }

    // --- cancel --------------------------------------------------------------------------------------------------

    /// Asks the running request to stop. The helper keeps its model; the request ends with
    /// `EngineError::Cancelled`. A helper that doesn't stop within a couple of seconds is killed. Does nothing when
    /// no request is running.
    pub fn cancel(&self) {
        let stdin = self
            .live_stdin
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let Some(stdin) = stdin else { return };
        *self.cancel_asked.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
        let _ = write_line(&stdin, &encode(&Request::Cancel));
    }
}

/// Whether the app asked for the card in `request`: `load`, `complete` and `embed` carry `gpu`; nothing else does.
fn asked_for_card(request: &Request) -> bool {
    match request {
        Request::Load(l) => l.gpu,
        Request::Complete(c) => c.gpu,
        Request::Embed(e) => e.gpu,
        _ => false,
    }
}

/// Why `request`, which ran on `ran`, was not on the card. `failed_now`: the card failed during this call. `r`: the
/// helper that answered.
fn fallback_reason(
    request: &Request,
    ran: &Backend,
    failed_now: Option<Fallback>,
    r: &Running,
) -> Option<Fallback> {
    if !asked_for_card(request) || ran.on_gpu() {
        return None;
    }
    Some(if let Some(why) = failed_now {
        why
    } else if r.processor_fallback {
        // The processor helper took over after the card failed, now or earlier: the request it got said `gpu: false`.
        Fallback::RememberedFailure
    } else if r.card_helper {
        // The card helper is running but its answer was the processor.
        Fallback::CardFailed
    } else if r.backend.on_gpu() {
        // The one helper is a card helper (the Mac's), and this request still ran on the processor.
        Fallback::NoCardPath
    } else {
        Fallback::NoCardHelper
    })
}

/// `request` as the processor helper must get it after the card helper failed: the same request with `gpu: false`.
fn processor_only(request: &Request) -> Request {
    let mut request = request.clone();
    match &mut request {
        Request::Load(l) => l.gpu = false,
        Request::Complete(c) => c.gpu = false,
        Request::Embed(e) => e.gpu = false,
        _ => {}
    }
    request
}

/// Where the helper says the request that `ev` answers ran: `loaded`, `done` and `embedded` carry it. None for any
/// other event, and for a helper from before replies said (an empty backend name).
fn ran_on(ev: &Event) -> Option<Backend> {
    let backend = match ev {
        Event::Loaded { backend, .. } | Event::Embedded { backend, .. } => backend,
        Event::Done(reply) => &reply.backend,
        _ => return None,
    };
    (!backend.backend.is_empty()).then(|| backend.clone())
}

/// Starts the helper at `path` and tells it the idle policy. A helper that dies at startup is started again.
fn start_with_retries(
    path: &std::path::Path,
    env: &[(String, String)],
    cfg: &LlmConfig,
) -> Result<(Process, Backend), EngineError> {
    let mut last = StartFailure::Died;
    for attempt in 0..START_TRIES {
        if attempt > 0 {
            std::thread::sleep(START_RETRY_PAUSE);
        }
        match Process::start(path, env, cfg.limits.start) {
            Ok((process, backend)) => {
                let configure = Request::Configure(Configure {
                    idle_secs: cfg.idle_secs,
                });
                if process.send(&configure).is_err() {
                    last = StartFailure::Died;
                    continue;
                }
                // `configured` is the next event; anything else, or silence, means the helper isn't usable.
                match process.recv(cfg.limits.start) {
                    Recv::Event(Event::Configured { .. }) => return Ok((process, backend)),
                    _ => {
                        last = StartFailure::NotReady;
                        continue;
                    }
                }
            }
            Err(StartFailure::VersionMismatch { helper }) => {
                return Err(EngineError::VersionMismatch {
                    helper,
                    app: PROTOCOL_VERSION,
                });
            }
            // Retrying a helper that couldn't be spawned or never answered would only wait longer.
            Err(e @ (StartFailure::CouldNotSpawn(_) | StartFailure::TimedOut)) => {
                return Err(start_error(e));
            }
            Err(e) => last = e,
        }
    }
    Err(start_error(last))
}

fn start_error(failure: StartFailure) -> EngineError {
    EngineError::CouldNotStart {
        message: match failure {
            StartFailure::CouldNotSpawn(m) => m,
            StartFailure::Died => "the AI helper exited while starting".into(),
            StartFailure::TimedOut => "the AI helper did not answer in time".into(),
            StartFailure::VersionMismatch { .. } => "the AI helper is from another version".into(),
            StartFailure::NotReady => "the AI helper did not say it was ready".into(),
        },
    }
}
