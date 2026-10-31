//! The speech helper from the app's side: the Windows Vulkan Whisper helper, and the process fallback on any system.
//! One `SttClient` per app, built like [`crate::LlmClient`]: it starts the helper on the first request, keeps it while
//! the app runs, restarts it when it dies, and turns every failure into an [`EngineError`].
//!
//! Calls block and are safe from any worker thread; one request runs at a time (the helper runs one at a time too).
//! `cancel` and `shutdown` may be called from another thread while a request runs.
//!
//! What differs from the AI client: a speech request carries no model path, so the client remembers the model the app
//! last loaded and puts it into every new helper (after a crash, a kill, a graphics card fallback, `shutdown`)
//! before the audio, without an error and without the app doing anything. `unload` is what makes it forget. The
//! helper has no idle timer (the app decides when to `unload` or `shutdown`), so a helper found dead before a request
//! is replaced and the request sent again; a helper that dies while working on one is a crash (`Crashed`, or
//! `OutOfMemory`), because trying a decode that crashed once again would only take as long again.
//!
//! The helper serves Whisper only (the ONNX models run in the app's own process, `voice_engine_stt::load`), so
//! `load` of another kind is `Unsupported`.
//!
//! Audio goes to the helper as base64 text, about 2.5 MB for 30 s; an hour is about 300 MB, so an app that sends long
//! recordings through the helper cuts them first.
//!
//! `cancel` ends the call with `Cancelled`. A Whisper decode cannot be interrupted, so the helper does not answer it:
//! the call ends 2 s after `cancel` by stopping the helper, and the next call loads the model again.

use crate::error::EngineError;
use crate::gpu_memory;
use crate::llm::{
    CANCEL_GRACE, DEVICES_LIMIT, Limits, Outcome, POLL, START_RETRY_PAUSE, START_TRIES,
};
use crate::process::{Recv, StartFailure, SttProcess, write_line};
use std::path::PathBuf;
use std::process::ChildStdin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};
use voice_engine_protocol::stt::{
    ChunkOptions, SpeechBackend, SttEvent, SttLoad, SttRequest, Transcript, encode_samples,
};
use voice_engine_protocol::{Backend, Device, DeviceList, Fallback, PROTOCOL_VERSION, encode};

#[derive(Debug, Clone)]
pub struct SttConfig {
    /// The processor helper (every platform; on the Mac it runs Whisper on Metal).
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
    /// `start`: the helper must answer its first ping within this. `stall`: no event at all for this long while a
    /// request runs (a Whisper decode sends nothing until it is done: set it above the longest decode). `total`: one
    /// call, with any restart and any loading of the model it needed, may take this long.
    pub limits: Limits,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SttLoadInfo {
    /// Seconds the load took.
    pub seconds: f32,
}

struct Running {
    process: SttProcess,
    /// What the helper said about itself in its `ready`.
    ready: Backend,
    /// The helper is the graphics card one and chose the card.
    on_gpu: bool,
    /// The helper holds the model the client remembers.
    loaded: bool,
    /// This is the processor helper, running because the card helper failed (now or in an earlier run) or could not
    /// start. On the Mac the two helpers can be the same binary, which would load on Metal again for a `load` that
    /// still says `gpu: true`, so every `load` sent to it says `gpu: false`. The model the client remembers keeps
    /// what the app asked for.
    processor_fallback: bool,
    /// This helper came from the card helper's path (`gpu_helper`), whatever it then said it runs on.
    card_helper: bool,
    /// The card helper just failed to start for this helper: the first call to use it reports that.
    fresh_failure: Option<Fallback>,
}

impl Running {
    /// The helper exited (it was killed, or crashed, while nobody was waiting on it). Events nobody asked for, which
    /// should not exist, are dropped.
    fn gone(&self) -> bool {
        loop {
            match self.process.recv(Duration::ZERO) {
                Recv::Closed => return true,
                Recv::Event(_) => {}
                Recv::Timeout => return false,
            }
        }
    }
}

/// Why one attempt at a request ended without an answer.
enum Attempt {
    /// The helper process is gone. `delivered`: the request had been written to it (so it may have been working on
    /// it); false: it was already gone, or the write failed.
    Died { delivered: bool },
    /// The helper said no, or a limit was hit. `kill`: the helper is in an unknown state and must be stopped.
    Failed { error: EngineError, kill: bool },
}

/// An answer, and where it was computed (when the event says: `unloaded` does not).
type Answer<T> = (T, Option<SpeechBackend>);

pub struct SttClient {
    config: Mutex<SttConfig>,
    /// Held for the whole of a request.
    running: Mutex<Option<Running>>,
    /// The model the app loaded last and has not unloaded: what a new helper is given.
    model: Mutex<Option<SttLoad>>,
    /// The running helper's stdin, so `cancel` can write to it while `running` is held by a request.
    live_stdin: Mutex<Option<Arc<Mutex<ChildStdin>>>>,
    /// When `cancel` was last asked for during the current request.
    cancel_asked: Mutex<Option<Instant>>,
    /// `shutdown` was called while a request held the helper: the request ends on its next pass.
    stop_asked: AtomicBool,
    /// The graphics card failed in this process; never try it again until the app restarts.
    gpu_disabled: AtomicBool,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl SttClient {
    pub fn new(config: SttConfig) -> Self {
        SttClient {
            config: Mutex::new(config),
            running: Mutex::new(None),
            model: Mutex::new(None),
            live_stdin: Mutex::new(None),
            cancel_asked: Mutex::new(None),
            stop_asked: AtomicBool::new(false),
            gpu_disabled: AtomicBool::new(false),
        }
    }

    /// Loads a Whisper model now, replacing any other, and remembers it for every helper started later. A load that
    /// fails leaves no model loaded (and none remembered).
    pub fn load(&self, model: SttLoad) -> Result<Outcome<SttLoadInfo>, EngineError> {
        let mut running = lock(&self.running);
        let request = SttRequest::Load(model);
        self.exchange(&mut running, &request, false, &|ev| match ev {
            SttEvent::Loaded { seconds, backend } => {
                Some((SttLoadInfo { seconds: *seconds }, Some(backend.clone())))
            }
            _ => None,
        })
    }

    /// Transcribes `audio` (16 kHz mono, any length) with the loaded model. With no model loaded (never loaded, or
    /// unloaded, or the last load failed) it is `BadRequest` and no helper is started.
    pub fn transcribe(
        &self,
        audio: &[f32],
        options: &ChunkOptions,
    ) -> Result<Outcome<Transcript>, EngineError> {
        let request = SttRequest::Transcribe {
            audio_b64: encode_samples(audio),
            options: options.clone(),
        };
        let mut running = lock(&self.running);
        self.exchange(&mut running, &request, true, &|ev| match ev {
            SttEvent::Transcribed {
                transcript,
                backend,
            } => Some((transcript.clone(), Some(backend.clone()))),
            _ => None,
        })
    }

    /// Every device the helper sees and the one it uses: ggml's (Vulkan, the processor). The helper runs Whisper
    /// only, so it never lists DirectML's adapters (`voice_engine_stt::discrete_card` is that, in the app). Asked of the graphics card helper when there
    /// is one and it hasn't failed (else of the processor helper, which lists the processor only). Starts the helper
    /// if none runs and loads no model, so an app asks once at startup, before the first load.
    pub fn devices(&self) -> Result<DeviceList, EngineError> {
        let mut running = lock(&self.running);
        self.exchange_within(
            Some(DEVICES_LIMIT),
            &mut running,
            &SttRequest::Devices,
            false,
            &|ev| match ev {
                SttEvent::Devices { devices, chosen } => Some((
                    DeviceList {
                        devices: devices.clone(),
                        chosen: *chosen,
                    },
                    None,
                )),
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

    /// Frees the model's memory now and forgets it (a helper started later gets none). With no helper running there
    /// is nothing to free, and none is started.
    pub fn unload(&self) -> Result<(), EngineError> {
        let mut running = lock(&self.running);
        *lock(&self.model) = None;
        // A helper that is already gone holds nothing: no need to start another one to tell it to let go.
        if running.as_ref().is_some_and(Running::gone) {
            *running = None;
        }
        if running.is_none() {
            return Ok(());
        }
        self.exchange(&mut running, &SttRequest::Unload, false, &|ev| match ev {
            SttEvent::Unloaded => Some(((), None)),
            _ => None,
        })
        .map(|_| ())
    }

    /// Stops the helper now (the app is closing, or wants the memory back). The model is remembered: the next call
    /// starts a helper and loads it again; `unload` forgets it. A request running on another thread ends with
    /// `Cancelled` within a moment.
    pub fn shutdown(&self) {
        match self.running.try_lock() {
            Ok(mut running) => {
                if let Some(mut r) = running.take() {
                    r.process.kill();
                }
            }
            Err(TryLockError::Poisoned(p)) => {
                if let Some(mut r) = p.into_inner().take() {
                    r.process.kill();
                }
            }
            // A request holds the helper: it sees this on its next pass, stops the helper and gives it up.
            Err(TryLockError::WouldBlock) => self.stop_asked.store(true, Ordering::SeqCst),
        }
    }

    /// Asks the running request to stop. It ends with `EngineError::Cancelled`: at once when the helper answers
    /// the cancel, else after 2 s, when the helper is stopped (a Whisper decode cannot be interrupted; the next call
    /// loads the model into a new helper). Does nothing when no request is running.
    pub fn cancel(&self) {
        let stdin = lock(&self.live_stdin).clone();
        let Some(stdin) = stdin else { return };
        *lock(&self.cancel_asked) = Some(Instant::now());
        let _ = write_line(&stdin, &encode(&SttRequest::Cancel));
    }

    /// The app's "Try again" for the graphics card on a live client: forgets that the card failed (in this client and
    /// in the `gpu_failures` file, when one is configured) so the next request tries the card again. It waits for
    /// a request in progress to finish. When the helper running now is the processor one that took over after the
    /// card failed, it is stopped, and the next request starts a new helper on the card and loads the remembered
    /// model into it (as after `shutdown`); when the card helper is running, it is left alone. The card can fail
    /// again, and that is remembered again. Calling `clear_gpu_failure` alone does not reach a running client.
    pub fn retry_gpu(&self) {
        let mut running = lock(&self.running);
        {
            let cfg = lock(&self.config);
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

    /// Sends `request` and waits for the event `pick` accepts. `needs_model`: the helper must hold the app's model
    /// first (it is loaded into a new helper before the request). The whole call, restarts included, is held to the
    /// total limit.
    fn exchange<T>(
        &self,
        running: &mut Option<Running>,
        request: &SttRequest,
        needs_model: bool,
        pick: &dyn Fn(&SttEvent) -> Option<Answer<T>>,
    ) -> Result<Outcome<T>, EngineError> {
        self.exchange_within(None, running, request, needs_model, pick)
    }

    /// `exchange`, with the stall and total limits capped at `cap` when there is one.
    fn exchange_within<T>(
        &self,
        cap: Option<Duration>,
        running: &mut Option<Running>,
        request: &SttRequest,
        needs_model: bool,
        pick: &dyn Fn(&SttEvent) -> Option<Answer<T>>,
    ) -> Result<Outcome<T>, EngineError> {
        self.stop_asked.store(false, Ordering::SeqCst);
        let began = Instant::now();
        // Why the card was given up on during this call, when it was.
        let mut failed_now: Option<Fallback> = None;
        let mut retried = false;
        loop {
            if needs_model && lock(&self.model).is_none() {
                return Err(EngineError::BadRequest);
            }
            if running.is_none() {
                *running = Some(self.start()?);
            }
            let r = running.as_mut().expect("started above");
            if let Some(why) = r.fresh_failure.take() {
                failed_now = failed_now.or(Some(why));
            }
            if let SttRequest::Load(_) = request {
                // The helper drops its model when it starts a load, so whatever happens next, none is loaded.
                r.loaded = false;
                *lock(&self.model) = None;
            }
            *lock(&self.live_stdin) = Some(r.process.stdin_handle());
            *lock(&self.cancel_asked) = None;
            let sent = match request {
                SttRequest::Load(spec) if r.processor_fallback => {
                    SttRequest::Load(processor_only(spec.clone()))
                }
                other => other.clone(),
            };
            let attempt = self.attempt(r, &sent, needs_model, began, cap, pick);
            *lock(&self.live_stdin) = None;
            match attempt {
                Ok((value, spoken)) => {
                    if let SttRequest::Load(spec) = request {
                        r.loaded = true;
                        *lock(&self.model) = Some(spec.clone());
                    }
                    let backend = match &spoken {
                        Some(s) => Backend {
                            backend: s.backend.clone(),
                            device: s.device.clone(),
                            vram_mb: if s.device.is_some() {
                                r.ready.vram_mb
                            } else {
                                0
                            },
                        },
                        None => r.ready.clone(),
                    };
                    let said = spoken.and_then(|s| s.fallback);
                    let fallback = self.fallback_reason(request, &backend, failed_now, said, r);
                    return Ok(Outcome {
                        value,
                        backend,
                        fallback,
                    });
                }
                Err(Attempt::Died { delivered }) => {
                    let mut dead = running.take().expect("running above");
                    let death = dead.process.reap();
                    if !delivered {
                        // It was already gone before it got the request (it was killed, or crashed, while nobody was
                        // waiting): a fresh helper answers it, once.
                        if !retried {
                            retried = true;
                            continue;
                        }
                        return Err(EngineError::died(death));
                    }
                    if dead.on_gpu {
                        // The graphics card helper died while working: remember it and do the work on the processor.
                        self.gpu_failed();
                        failed_now = Some(if death.out_of_memory() {
                            Fallback::CardOutOfMemory
                        } else {
                            Fallback::CardFailed
                        });
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

    /// One try at a request on the running helper, after putting the app's model into it when it holds none.
    fn attempt<T>(
        &self,
        r: &mut Running,
        request: &SttRequest,
        needs_model: bool,
        began: Instant,
        cap: Option<Duration>,
        pick: &dyn Fn(&SttEvent) -> Option<Answer<T>>,
    ) -> Result<Answer<T>, Attempt> {
        // A helper that exited while idle: say so before writing to it, so the call is not counted as a crash.
        if r.gone() {
            return Err(Attempt::Died { delivered: false });
        }
        if needs_model && !r.loaded {
            let Some(mut model) = lock(&self.model).clone() else {
                return Err(Attempt::Failed {
                    error: EngineError::BadRequest,
                    kill: false,
                });
            };
            if r.processor_fallback {
                // The processor helper has no use for the app's wish for the card.
                model = processor_only(model);
            }
            self.run(r, &SttRequest::Load(model), began, cap, &|ev| match ev {
                SttEvent::Loaded { backend, .. } => Some(((), Some(backend.clone()))),
                _ => None,
            })?;
            r.loaded = true;
        }
        self.run(r, request, began, cap, pick)
    }

    /// Writes one request and waits for its answer, holding it to the limits on every pass.
    fn run<T>(
        &self,
        r: &mut Running,
        request: &SttRequest,
        began: Instant,
        cap: Option<Duration>,
        pick: &dyn Fn(&SttEvent) -> Option<Answer<T>>,
    ) -> Result<Answer<T>, Attempt> {
        if r.process.send(request).is_err() {
            return Err(Attempt::Died { delivered: false });
        }
        let mut limits = lock(&self.config).limits.clone();
        if let Some(cap) = cap {
            limits.stall = limits.stall.min(cap);
            limits.total = limits.total.min(cap);
        }
        let mut last_event = Instant::now();
        loop {
            // Checked on every pass, not only when the helper is quiet: a helper that keeps writing events while
            // ignoring `cancel` (or running past the total limit) must still be stopped.
            if self.stop_asked.load(Ordering::SeqCst) {
                return Err(Attempt::Failed {
                    error: EngineError::Cancelled,
                    kill: true,
                });
            }
            let asked = *lock(&self.cancel_asked);
            if asked.is_some_and(|t| t.elapsed() > CANCEL_GRACE) {
                // It was told to stop and didn't: stop it for good.
                return Err(Attempt::Failed {
                    error: EngineError::Cancelled,
                    kill: true,
                });
            }
            if last_event.elapsed() > limits.stall || began.elapsed() > limits.total {
                return Err(Attempt::Failed {
                    error: EngineError::TimedOut,
                    kill: true,
                });
            }
            match r.process.recv(POLL) {
                Recv::Event(SttEvent::Error { message, error }) => {
                    return Err(Attempt::Failed {
                        error: EngineError::from_stt_helper(error, message),
                        kill: false,
                    });
                }
                Recv::Event(ev) => {
                    last_event = Instant::now();
                    if let Some(answer) = pick(&ev) {
                        return Ok(answer);
                    }
                }
                Recv::Closed => return Err(Attempt::Died { delivered: true }),
                Recv::Timeout => {}
            }
        }
    }

    /// Starts a helper: the graphics card one first when there is one and it hasn't failed, else the processor one.
    fn start(&self) -> Result<Running, EngineError> {
        let cfg = lock(&self.config).clone();
        let mut fresh_failure = None;
        if let Some(gpu_path) = &cfg.gpu_helper
            && !self.gpu_disabled.load(Ordering::Relaxed)
            && !gpu_memory::failed(cfg.gpu_failures.as_deref(), &cfg.gpu_key)
        {
            let env: Vec<_> = cfg.env.iter().chain(&cfg.gpu_env).cloned().collect();
            match start_with_retries(gpu_path, &env, &cfg) {
                Ok((process, ready)) => {
                    let on_gpu = ready.on_gpu();
                    return Ok(Running {
                        process,
                        ready,
                        on_gpu,
                        loaded: false,
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
        let (process, ready) = start_with_retries(&cfg.cpu_helper, &cfg.env, &cfg)?;
        Ok(Running {
            process,
            ready,
            on_gpu: false,
            loaded: false,
            // Without a card helper configured the one helper is the app's only choice (on the Mac it is the Metal
            // one): the app's `gpu` goes through as it is.
            processor_fallback: cfg.gpu_helper.is_some(),
            card_helper: false,
            fresh_failure,
        })
    }

    /// Why `request`, which ran on `ran`, was not on the card (see [`Outcome::fallback`]). The app asked for the
    /// card in a `load` that says `gpu: true`, and in a `transcribe` with a model loaded that way. `failed_now`: the
    /// card failed during this call. `said`: the reason the helper gave for running on the processor, when it gave one.
    fn fallback_reason(
        &self,
        request: &SttRequest,
        ran: &Backend,
        failed_now: Option<Fallback>,
        said: Option<Fallback>,
        r: &Running,
    ) -> Option<Fallback> {
        let asked = match request {
            SttRequest::Load(spec) => spec.gpu,
            SttRequest::Transcribe { .. } => lock(&self.model).as_ref().is_some_and(|m| m.gpu),
            _ => false,
        };
        if !asked || ran.on_gpu() {
            return None;
        }
        Some(if let Some(why) = failed_now {
            why
        } else if r.processor_fallback {
            // The processor helper took over after the card failed, now or earlier: its load said `gpu: false`.
            Fallback::RememberedFailure
        } else if !r.card_helper && !r.ready.on_gpu() {
            // Only the processor helper is here: it cannot tell the app that no card helper exists.
            Fallback::NoCardHelper
        } else if let Some(why) = said {
            // The helper that can use the card says why the model is not on it (a v0.2.x helper only says that it is
            // not, which reads as `CardFailed`).
            why
        } else if r.card_helper {
            // The helper that can use the card ran on the processor without saying why.
            Fallback::CardFailed
        } else {
            Fallback::NoCardPath
        })
    }

    fn gpu_failed(&self) {
        self.gpu_disabled.store(true, Ordering::Relaxed);
        let cfg = lock(&self.config);
        gpu_memory::record(cfg.gpu_failures.as_deref(), &cfg.gpu_key);
    }
}

/// `model` as the processor helper must get it after the card helper failed: the same load with `gpu: false`.
fn processor_only(mut model: SttLoad) -> SttLoad {
    model.gpu = false;
    model
}

/// Starts the helper at `path`. A helper that dies at startup is started again.
fn start_with_retries(
    path: &std::path::Path,
    env: &[(String, String)],
    cfg: &SttConfig,
) -> Result<(SttProcess, Backend), EngineError> {
    let mut last = StartFailure::Died;
    for attempt in 0..START_TRIES {
        if attempt > 0 {
            std::thread::sleep(START_RETRY_PAUSE);
        }
        match SttProcess::start(path, env, cfg.limits.start) {
            Ok(started) => return Ok(started),
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
            StartFailure::Died => "the speech helper exited while starting".into(),
            StartFailure::TimedOut => "the speech helper did not answer in time".into(),
            StartFailure::VersionMismatch { .. } => {
                "the speech helper is from another version".into()
            }
            StartFailure::NotReady => "the speech helper did not say it was ready".into(),
        },
    }
}
