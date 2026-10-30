//! Does asking whisper.cpp for token timestamps cost time? Whisper Base on the processor and Whisper Large Turbo on
//! Metal, over the same 29 s `say` clip as `bench`, in four ways, each with 6 threads and with the engine's default
//! threads (one load per model and way, one warm-up decode, then 3 timed decodes: median and range). Prints times,
//! sizes and counts only, never the transcript.
//!
//! ```text
//! cargo run --release -p voice-engine-stt --example bench_timing                 # both models
//! cargo run --release -p voice-engine-stt --example bench_timing -- base        # Whisper Base only
//! cargo run --release -p voice-engine-stt --example bench_timing -- turbo       # Whisper Large Turbo only
//! ```
//!
//! The ways:
//!
//! 1. `engine`: `WhisperEngine` with `Timing::Segments` as it is (token timestamps on, words in every segment).
//! 2. `talaan`: whisper-rs called directly with exactly the settings of `voice_engine_stt::transcribe`, Talaan's own
//!    chunk function that the real tests compare `Timing::Segments` against (no token timestamps).
//! 3. `tokens on`: whisper-rs called directly with the engine's settings (Talaan's plus `set_token_timestamps(true)`).
//! 4. `tokens off`: the same, with `set_token_timestamps(false)`: the engine's settings minus the one suspect.
//!
//! Ways 2 to 4 share one loaded model each (same context parameters as the engine: flash attention on Metal only), and
//! only one model is ever loaded at a time (Metal's shader cache can crash when several start at once). Models:
//! `whisper-turbo` from `$VOICE_ENGINE_TEST_MODELS` (default: Talaan's models folder), `whisper-base` from the shared
//! folder (`SharedStore::default_dir`). Only ever read.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use voice_engine_models::SharedStore;
use voice_engine_stt::threads::{ThreadKind, logical_cores, resolve_threads};
use voice_engine_stt::whisper_engine::{WhisperEngine, choose_gpu};
use voice_engine_stt::{ChunkOptions, LoadOptions, Timing, Transcriber};

const CLIP_SECS: f64 = 29.0;
const RUNS: usize = 3;

/// Sentences said in turn until the clip is long enough.
const SENTENCES: &[&str] = &[
    "Good morning everyone, thanks for joining the call today.",
    "The first thing on the agenda is the release schedule for the next quarter.",
    "We moved the review to Thursday afternoon because two people are travelling.",
    "Please send your notes to the team before the end of the day on Wednesday.",
    "The budget numbers look better than we expected, but the hiring plan is still open.",
    "Ottawa and Toronto will each get a small group to test the new version.",
    "If anything breaks during the trial, write down the steps and the time it happened.",
    "That is everything from my side, so let's open it up for questions.",
];

fn say(text: &str, dir: &Path) -> Vec<f32> {
    let aiff = dir.join("clip.aiff");
    let wav = dir.join("clip.wav");
    let ok = Command::new("say")
        .arg("-o")
        .arg(&aiff)
        .arg(text)
        .status()
        .expect("the macOS say command");
    assert!(ok.success(), "say failed");
    let ok = Command::new("afconvert")
        .args(["-f", "WAVE", "-d", "LEF32@16000", "-c", "1"])
        .arg(&aiff)
        .arg(&wav)
        .status()
        .expect("afconvert");
    assert!(ok.success(), "afconvert failed");
    read_wav_f32(&wav)
}

/// Samples of a 32-bit float mono WAV (what afconvert wrote), chunk by chunk.
fn read_wav_f32(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    assert_eq!(&b[0..4], b"RIFF");
    let mut i = 12;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = i + 8;
        if id == b"data" {
            let end = (body + len).min(b.len());
            return b[body..end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
        }
        i = body + len + (len & 1);
    }
    panic!("no data chunk");
}

fn clip() -> Vec<f32> {
    let dir = std::env::temp_dir().join(format!("voice-engine-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let want = (CLIP_SECS * 16_000.0) as usize;
    let mut out: Vec<f32> = vec![];
    for sentence in SENTENCES.iter().cycle() {
        if out.len() >= want {
            break;
        }
        out.extend(say(sentence, &dir));
        out.extend(std::iter::repeat_n(0.0, 4_800));
    }
    out.truncate(want);
    let _ = std::fs::remove_dir_all(&dir);
    out
}

fn median_and_range(times: &[f64]) -> (f64, f64, f64) {
    let mut t = times.to_vec();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (t[t.len() / 2], t[0], t[t.len() - 1])
}

/// Times `run` once as a warm-up and `RUNS` more times; the numbers of segments and words it returned (the last
/// run's) say the same speech was decoded.
fn time(mut run: impl FnMut() -> (usize, usize)) -> Timed {
    let began = Instant::now();
    run();
    let first_s = began.elapsed().as_secs_f64();
    let mut times = vec![];
    let mut counts = (0, 0);
    for _ in 0..RUNS {
        let began = Instant::now();
        counts = run();
        times.push(began.elapsed().as_secs_f64());
    }
    let (median_s, min_s, max_s) = median_and_range(&times);
    Timed {
        first_s,
        median_s,
        min_s,
        max_s,
        segments: counts.0,
        words: counts.1,
    }
}

struct Timed {
    first_s: f64,
    median_s: f64,
    min_s: f64,
    max_s: f64,
    segments: usize,
    words: usize,
}

fn print(model: &str, way: &str, threads: usize, audio_secs: f64, t: &Timed) {
    println!(
        "{model:<14} {way:<10} threads={threads:<2} first={:>5.2}s decode median={:>5.2}s range={:.2}-{:.2}s \
         rtf={:>5.1}x segments={} words={}",
        t.first_s,
        t.median_s,
        t.min_s,
        t.max_s,
        audio_secs / t.median_s,
        t.segments,
        t.words
    );
}

/// Way 1: the engine as it is.
fn engine(file: &Path, gpu: bool, threads: Option<usize>, audio: &[f32]) -> (usize, Timed) {
    let engine = WhisperEngine::load(file, LoadOptions { gpu, threads }).expect("load the model");
    let used = engine.threads();
    let mut engine: Box<dyn Transcriber> = Box::new(engine);
    let opts = ChunkOptions {
        timing: Timing::Segments,
        ..ChunkOptions::default()
    };
    let timed = time(|| {
        let t = engine.transcribe(audio, &opts).expect("decode");
        (
            t.segments.len(),
            t.segments.iter().map(|s| s.words.len()).sum(),
        )
    });
    (used, timed)
}

/// What a direct whisper-rs decode counts: segments and the words (whitespace-separated) of their text.
fn count(state: &whisper_rs::WhisperState) -> (usize, usize) {
    let mut words = 0;
    let n = state.full_n_segments();
    for i in 0..n {
        if let Some(seg) = state.get_segment(i) {
            words += seg
                .to_str_lossy()
                .map(|t| t.split_whitespace().count())
                .unwrap_or(0);
        }
    }
    (n as usize, words)
}

/// The engine's settings (`whisper_engine.rs`, `Timing::Segments`, no prompt), with token timestamps as asked.
fn decode_direct(
    state: &mut whisper_rs::WhisperState,
    audio: &[f32],
    threads: usize,
    token_timestamps: bool,
) -> (usize, usize) {
    let mut params =
        whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(threads as i32);
    params.set_language(Some("auto"));
    params.set_no_context(true);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_token_timestamps(token_timestamps);
    state.full(params, audio).expect("decode");
    count(state)
}

/// Ways 2 to 4 for one thread count, on one loaded model.
fn direct(
    model: &str,
    file: &Path,
    gpu: bool,
    thread_counts: &[(String, usize)],
    audio: &[f32],
    audio_secs: f64,
) {
    let mac = cfg!(target_os = "macos");
    let all = voice_engine_stt::devices();
    let number = if gpu {
        choose_gpu(&all, mac).map(|(_, n)| n)
    } else {
        None
    };
    let ctx = whisper_rs::WhisperContext::new_with_params(
        file,
        voice_engine_stt::context_params(number, number.is_some() && mac),
    )
    .expect("load the model");
    let mut state = ctx.create_state().expect("create the state");
    for (label, threads) in thread_counts {
        let talaan = time(|| {
            let (pieces, _) =
                voice_engine_stt::transcribe(&mut state, audio, None, *threads).expect("decode");
            (pieces.len(), count(&state).1)
        });
        print(model, "talaan", *threads, audio_secs, &talaan);
        for (way, on) in [("tokens on", true), ("tokens off", false)] {
            let t = time(|| decode_direct(&mut state, audio, *threads, on));
            print(model, way, *threads, audio_secs, &t);
        }
        println!("{model:<14} (the {label} thread count above)");
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_default();
    let talaan_dir = match std::env::var_os("VOICE_ENGINE_TEST_MODELS") {
        Some(d) => PathBuf::from(d),
        None => dirs::data_dir().unwrap().join("Talaan/models"),
    };
    let shared = SharedStore::default_dir().expect("the shared model folder");
    let audio = clip();
    let audio_secs = audio.len() as f64 / 16_000.0;
    let logical = logical_cores();
    println!(
        "clip: {audio_secs:.1} s, 16 kHz mono; {logical} logical cores; {RUNS} timed runs after 1 warm-up; language \
         auto, greedy, no context"
    );

    // (name, file, gpu, the engine's default thread count where it runs)
    let models = [
        (
            "whisper-base",
            shared.join("whisper-base/ggml-base.bin"),
            false,
            resolve_threads(ThreadKind::Processor, None, logical),
        ),
        (
            "whisper-turbo",
            talaan_dir.join("whisper-turbo/ggml-large-v3-turbo-q5_0.bin"),
            true,
            resolve_threads(ThreadKind::Gpu, None, logical),
        ),
    ];
    for (name, file, gpu, default) in models {
        if !which.is_empty() && !name.ends_with(&which) {
            continue;
        }
        if !file.is_file() {
            println!("{name}: not found at {}", file.display());
            continue;
        }
        let place = if gpu { "metal" } else { "cpu" };
        println!("--- {name} ({place}) ---");
        let counts = [("default".to_string(), default), ("6".to_string(), 6)];
        for (label, threads) in &counts {
            let requested = (label != "default").then_some(*threads);
            let (used, t) = engine(&file, gpu, requested, &audio);
            print(name, "engine", used, audio_secs, &t);
        }
        direct(name, &file, gpu, &counts, &audio, audio_secs);
    }
}
