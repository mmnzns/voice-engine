//! Speech speed on this computer: load time, decode time and real-time factor for a 29 s clip, next to the bars the
//! apps asked for. Prints times, sizes and counts only, never the transcript.
//!
//! ```text
//! cargo run --release -p voice-engine-stt --example bench            # the planned set
//! cargo run --release -p voice-engine-stt --example bench -- sweep   # also other thread counts for each model
//! ```
//!
//! Works on a Mac and on a Windows PC (see `support/mod.rs` for the details):
//!
//! * Models: each of `whisper-turbo`, `parakeet-ultra` and `whisper-base` is looked up on its own, in
//!   `$VOICE_ENGINE_TEST_MODELS` (a folder with a folder per model id) when that is set, else in the shared folder, then in
//!   Talaan's real folder for this platform. Only ever read. A model that is not found is named and skipped.
//! * The clip: `$VOICE_ENGINE_TEST_CLIP`, a 16 kHz mono WAV (16-bit PCM or 32-bit float), repeated 0.3 s apart and cut to
//!   29.0 s (`docs/data/clip-bench.wav` is 12 s of it; a clip of one short sentence repeated makes Whisper Turbo stop
//!   after the first, so the number would not mean much); only when that is not set, and only on a Mac, the macOS `say` command reading several sentences in turn the
//!   same way. Without either the bench says what to set.
//! * The Metal rows run on a Mac only. One model is loaded at a time. Each line is one load, one warm-up decode, then 3
//!   timed decodes (median and range); `Timing::Segments`, language auto.

use std::path::{Path, PathBuf};
use std::time::Instant;
use voice_engine_models::catalog::find;
use voice_engine_stt::parakeet_engine::ParakeetEngine;
use voice_engine_stt::whisper_engine::WhisperEngine;
use voice_engine_stt::{ChunkOptions, LoadOptions, Timing, Transcriber};

#[path = "../support/mod.rs"]
mod support;

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

fn clip() -> Vec<f32> {
    support::bench_clip(CLIP_SECS, SENTENCES).unwrap_or_else(|e| {
        eprintln!("bench: {e}");
        std::process::exit(2);
    })
}

fn median_and_range(times: &[f64]) -> (f64, f64, f64) {
    let mut t = times.to_vec();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (t[t.len() / 2], t[0], t[t.len() - 1])
}

struct Row {
    label: String,
    backend: String,
    threads: usize,
    load_s: f64,
    first_s: f64,
    median_s: f64,
    min_s: f64,
    max_s: f64,
    words: usize,
}

impl Row {
    fn rtf(&self, audio_secs: f64) -> f64 {
        audio_secs / self.median_s
    }
}

fn measure(
    label: &str,
    audio: &[f32],
    load: impl FnOnce() -> (Box<dyn Transcriber>, usize, f64),
) -> Row {
    let (mut engine, threads, load_s) = load();
    let backend = engine.backend();
    let opts = ChunkOptions {
        timing: Timing::Segments,
        ..ChunkOptions::default()
    };
    // The warm-up: the first decode allocates and compiles things the later ones reuse.
    let started = Instant::now();
    let warm = engine.transcribe(audio, &opts).expect("warm-up decode");
    let first_s = started.elapsed().as_secs_f64();
    let words = warm.segments.iter().map(|s| s.words.len()).sum::<usize>();
    let mut times = vec![];
    for _ in 0..RUNS {
        let started = Instant::now();
        engine.transcribe(audio, &opts).expect("decode");
        times.push(started.elapsed().as_secs_f64());
    }
    let (median_s, min_s, max_s) = median_and_range(&times);
    Row {
        label: label.to_string(),
        backend: format!(
            "{}{}",
            backend.backend,
            if backend.fell_back {
                " (fell back)"
            } else {
                ""
            }
        ),
        threads,
        load_s,
        first_s,
        median_s,
        min_s,
        max_s,
        words,
    }
}

fn whisper(label: &str, file: &Path, audio: &[f32], opts: LoadOptions) -> Row {
    measure(label, audio, || {
        let started = Instant::now();
        let engine = WhisperEngine::load(file, opts).expect("load the Whisper model");
        let load_s = started.elapsed().as_secs_f64();
        let threads = engine.threads();
        (Box::new(engine), threads, load_s)
    })
}

fn parakeet(label: &str, id: &str, dir: &Path, audio: &[f32], opts: LoadOptions) -> Row {
    measure(label, audio, || {
        let started = Instant::now();
        let engine = ParakeetEngine::load(find(id).unwrap(), dir, opts).expect("load Parakeet");
        let load_s = started.elapsed().as_secs_f64();
        let threads = engine.threads();
        (Box::new(engine), threads, load_s)
    })
}

fn print(row: &Row, audio_secs: f64, bar: &str) {
    println!(
        "{:<34} backend={:<5} threads={:<2} load={:>5.2}s first={:>5.2}s decode median={:>5.2}s range={:.2}-{:.2}s \
         rtf={:>5.1}x words={}  | {bar}",
        row.label,
        row.backend,
        row.threads,
        row.load_s,
        row.first_s,
        row.median_s,
        row.min_s,
        row.max_s,
        row.rtf(audio_secs),
        row.words,
    );
}

fn verdict(met: bool) -> &'static str {
    if met { "MET" } else { "MISSED" }
}

/// `file_name` inside the folder of model `id` (see `support::find_model_dir`), or a line saying why not.
fn model_file(id: &str, file_name: &str) -> Option<PathBuf> {
    match support::find_model_dir(id) {
        Ok(dir) => Some(dir.join(file_name)),
        Err(why) => {
            println!("{id}: skipped, {why}");
            None
        }
    }
}

fn main() {
    let sweep = std::env::args().any(|a| a == "sweep");
    let mac = cfg!(target_os = "macos");

    let audio = clip();
    let audio_secs = audio.len() as f64 / 16_000.0;
    println!(
        "clip: {audio_secs:.1} s, 16 kHz mono; {} logical cores; {RUNS} timed runs after 1 warm-up; \
         Timing::Segments, language auto",
        voice_engine_stt::threads::logical_cores()
    );

    // Each model on its own: one that is not found is named and skipped, the others still run.
    let turbo = model_file("whisper-turbo", "ggml-large-v3-turbo-q5_0.bin");
    let base = model_file("whisper-base", "ggml-base.bin");
    let ultra = support::find_model_dir("parakeet-ultra")
        .map_err(|why| println!("parakeet-ultra: skipped, {why}"))
        .ok();

    if let Some(turbo) = &turbo {
        if mac {
            let metal = whisper(
                "whisper-turbo (metal)",
                turbo,
                &audio,
                LoadOptions {
                    gpu: true,
                    threads: None,
                },
            );
            let bar = format!(
                "bar: 25x on Metal -> {}",
                verdict(metal.rtf(audio_secs) >= 25.0)
            );
            print(&metal, audio_secs, &bar);
            if sweep {
                for n in [4, 6, 12] {
                    let row = whisper(
                        &format!("whisper-turbo (metal, {n} threads)"),
                        turbo,
                        &audio,
                        LoadOptions {
                            gpu: true,
                            threads: Some(n),
                        },
                    );
                    print(&row, audio_secs, "sweep");
                }
            }
        }

        let cpu = whisper(
            "whisper-turbo (cpu)",
            turbo,
            &audio,
            LoadOptions {
                gpu: false,
                threads: None,
            },
        );
        print(&cpu, audio_secs, "no bar (the processor)");
    }

    if let Some(base) = base.as_ref().filter(|b| b.is_file()) {
        let rows = [
            whisper(
                "whisper-base (cpu, default)",
                base,
                &audio,
                LoadOptions {
                    gpu: false,
                    threads: None,
                },
            ),
            whisper(
                "whisper-base (cpu, 6 threads)",
                base,
                &audio,
                LoadOptions {
                    gpu: false,
                    threads: Some(6),
                },
            ),
        ];
        for row in &rows {
            let bar = format!(
                "bar: Parlato floor 0.50 s for 29 s on this Mac -> {}",
                verdict(row.median_s <= 0.50)
            );
            print(row, audio_secs, &bar);
        }
        if sweep {
            let mut rows = vec![];
            if mac {
                rows.push(whisper(
                    "whisper-base (metal, default)",
                    base,
                    &audio,
                    LoadOptions {
                        gpu: true,
                        threads: None,
                    },
                ));
            }
            for n in [4, 8, 12] {
                rows.push(whisper(
                    &format!("whisper-base (cpu, {n} threads)"),
                    base,
                    &audio,
                    LoadOptions {
                        gpu: false,
                        threads: Some(n),
                    },
                ));
            }
            for row in &rows {
                print(row, audio_secs, "sweep");
            }
        }
    }

    if let Some(ultra) = &ultra {
        let rows = [
            parakeet(
                "parakeet-ultra (cpu, default)",
                "parakeet-ultra",
                ultra,
                &audio,
                LoadOptions {
                    gpu: false,
                    threads: None,
                },
            ),
            parakeet(
                "parakeet-ultra (cpu, 12 threads)",
                "parakeet-ultra",
                ultra,
                &audio,
                LoadOptions {
                    gpu: false,
                    threads: Some(12),
                },
            ),
        ];
        for row in &rows {
            let bar = format!(
                "bar: Parlato floor 0.43 s for 29 s on this Mac -> {}; 50x is for a Windows processor, not comparable (here {:.1}x)",
                verdict(row.median_s <= 0.43),
                row.rtf(audio_secs)
            );
            print(row, audio_secs, &bar);
        }
        if sweep {
            for n in [6, 8, 10] {
                let row = parakeet(
                    &format!("parakeet-ultra (cpu, {n} threads)"),
                    "parakeet-ultra",
                    ultra,
                    &audio,
                    LoadOptions {
                        gpu: false,
                        threads: Some(n),
                    },
                );
                print(&row, audio_secs, "sweep");
            }
        }
    }
}
