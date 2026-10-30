//! Speech speed on this computer: load time, decode time and real-time factor for a 29 s clip, next to the bars the
//! apps asked for. Prints times, sizes and counts only, never the transcript.
//!
//! ```text
//! cargo run --release -p voice-engine-stt --example bench            # the planned set
//! cargo run --release -p voice-engine-stt --example bench -- sweep   # also other thread counts for each model
//! ```
//!
//! Models: `whisper-turbo` and `parakeet-ultra` from `$VOICE_ENGINE_TEST_MODELS` (default: Talaan's models folder),
//! `whisper-base` from the shared folder (`SharedStore::default_dir`). Only ever read. The clip is made with the macOS
//! `say` command (several sentences, 0.3 s apart, cut to 29.0 s). One model is loaded at a time. Each line is one
//! load, one warm-up decode, then 3 timed decodes (median and range); `Timing::Segments`, language auto.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use voice_engine_models::SharedStore;
use voice_engine_models::catalog::find;
use voice_engine_stt::parakeet_engine::ParakeetEngine;
use voice_engine_stt::whisper_engine::WhisperEngine;
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

fn main() {
    let sweep = std::env::args().any(|a| a == "sweep");
    let talaan = match std::env::var_os("VOICE_ENGINE_TEST_MODELS") {
        Some(d) => PathBuf::from(d),
        None => dirs::data_dir().unwrap().join("Talaan/models"),
    };
    let shared = SharedStore::default_dir().expect("the shared model folder");

    let audio = clip();
    let audio_secs = audio.len() as f64 / 16_000.0;
    println!(
        "clip: {audio_secs:.1} s, 16 kHz mono; {} logical cores; {RUNS} timed runs after 1 warm-up; \
         Timing::Segments, language auto",
        voice_engine_stt::threads::logical_cores()
    );

    let turbo = talaan.join("whisper-turbo/ggml-large-v3-turbo-q5_0.bin");
    let base = shared.join("whisper-base/ggml-base.bin");
    let ultra = talaan.join("parakeet-ultra");

    let metal = whisper(
        "whisper-turbo (metal)",
        &turbo,
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
                &turbo,
                &audio,
                LoadOptions {
                    gpu: true,
                    threads: Some(n),
                },
            );
            print(&row, audio_secs, "sweep");
        }
    }

    let cpu = whisper(
        "whisper-turbo (cpu)",
        &turbo,
        &audio,
        LoadOptions {
            gpu: false,
            threads: None,
        },
    );
    print(&cpu, audio_secs, "no bar (the processor)");

    if base.is_file() {
        let rows = [
            whisper(
                "whisper-base (cpu, default)",
                &base,
                &audio,
                LoadOptions {
                    gpu: false,
                    threads: None,
                },
            ),
            whisper(
                "whisper-base (cpu, 6 threads)",
                &base,
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
            let mut rows = vec![whisper(
                "whisper-base (metal, default)",
                &base,
                &audio,
                LoadOptions {
                    gpu: true,
                    threads: None,
                },
            )];
            for n in [4, 8, 12] {
                rows.push(whisper(
                    &format!("whisper-base (cpu, {n} threads)"),
                    &base,
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
    } else {
        println!(
            "whisper-base: not in the shared folder yet (run the download_proof example first)"
        );
    }

    if ultra.join("encoder-model.int8.onnx").is_file() {
        let rows = [
            parakeet(
                "parakeet-ultra (cpu, default)",
                "parakeet-ultra",
                &ultra,
                &audio,
                LoadOptions {
                    gpu: false,
                    threads: None,
                },
            ),
            parakeet(
                "parakeet-ultra (cpu, 12 threads)",
                "parakeet-ultra",
                &ultra,
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
                    &ultra,
                    &audio,
                    LoadOptions {
                        gpu: false,
                        threads: Some(n),
                    },
                );
                print(&row, audio_secs, "sweep");
            }
        }
    } else {
        println!("parakeet-ultra: not found in {}", ultra.display());
    }
}
