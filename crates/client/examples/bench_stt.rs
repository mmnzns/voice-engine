//! Speed of Whisper through the real speech helper, with `SttClient`: the graphics card next to `gpu: false`, for a 29 s
//! clip. Prints times, counts and backend names only, never the transcript.
//!
//! ```text
//! cargo build --release -p voice-engine-stt                       # the Mac: Whisper on Metal
//! cargo build --release -p voice-engine-stt --features vulkan     # Windows: needs the Vulkan SDK (VULKAN_SDK)
//! cargo run --release -p voice-engine-client --example bench_stt                  # whisper-turbo
//! cargo run --release -p voice-engine-client --example bench_stt -- whisper-base  # any Whisper catalog id
//! ```
//!
//! The helper is `$VOICE_ENGINE_TEST_HELPER` or `target/release/voice-engine-stt` (`.exe` on Windows). The clip is
//! `$VOICE_ENGINE_TEST_CLIP`, a 16 kHz mono WAV, repeated and cut to 29 s (the Mac's `say` command only when that is not
//! set); the models are found as the speech bench finds them (`$VOICE_ENGINE_TEST_MODELS`, the shared folder, Talaan's
//! folder; see `crates/stt/support/mod.rs`), only ever read. First the helper's device list is printed (`devices()`:
//! how many devices it sees and the one it chose). Then, for the card (`gpu: true`) and for `gpu: false`: a load, one
//! warm-up transcription, then 3 timed ones (median and range, `Timing::Segments`, language auto). Each line says
//! `backend`, `device` (the card's name, `-` on the processor) and `fell_back` (the card was asked for and the processor
//! ran: no card, a failed card, or a helper built without `vulkan`). Each client is stopped before the next starts.

use std::path::{Path, PathBuf};
use std::time::Instant;
use voice_engine_client::{Limits, SttClient, SttConfig};
use voice_engine_models::catalog::find;
use voice_engine_protocol::Backend;
use voice_engine_protocol::stt::{ChunkOptions, SttLoad, SttModelKind, Timing};

#[path = "../../stt/support/mod.rs"]
mod support;

const CLIP_SECS: f64 = 29.0;
const RUNS: usize = 3;

/// Sentences said in turn until the clip is long enough (only when no clip file is given, on a Mac).
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

fn client(helper: &Path) -> SttClient {
    SttClient::new(SttConfig {
        cpu_helper: helper.to_path_buf(),
        gpu_helper: None,
        env: vec![],
        gpu_env: vec![],
        gpu_failures: None,
        gpu_key: String::new(),
        limits: Limits::default(),
    })
}

fn median_and_range(times: &[f64]) -> (f64, f64, f64) {
    let mut t = times.to_vec();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (t[t.len() / 2], t[0], t[t.len() - 1])
}

fn device(b: &Backend) -> String {
    b.device.clone().unwrap_or_else(|| "-".into())
}

fn main() {
    let helper = PathBuf::from(
        std::env::var_os("VOICE_ENGINE_TEST_HELPER").unwrap_or_else(|| {
            format!(
                "target/release/voice-engine-stt{}",
                std::env::consts::EXE_SUFFIX
            )
            .into()
        }),
    );
    assert!(
        helper.is_file(),
        "no helper at {}: cargo build --release -p voice-engine-stt (add --features vulkan on Windows)",
        helper.display()
    );
    let id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "whisper-turbo".into());
    let model = find(&id).unwrap_or_else(|| panic!("{id} is not in the catalog"));
    let dir = support::find_model_dir(&id).unwrap_or_else(|why| {
        eprintln!("bench_stt: {why}");
        std::process::exit(2);
    });
    let audio = support::bench_clip(CLIP_SECS, SENTENCES).unwrap_or_else(|why| {
        eprintln!("bench_stt: {why}");
        std::process::exit(2);
    });
    let audio_secs = audio.len() as f64 / 16_000.0;
    println!(
        "helper {}; model {} ({}); clip {audio_secs:.1} s, 16 kHz mono; {RUNS} timed runs after 1 warm-up; \
         Timing::Segments, language auto",
        helper.display(),
        model.id,
        model.files[0].name
    );

    // What the helper sees and which one it chose (no model loaded yet).
    let probe = client(&helper);
    let list = probe.devices().expect("the helper's device list");
    probe.shutdown();
    let chosen = list
        .chosen
        .and_then(|i| list.devices.iter().find(|d| d.index == i));
    println!(
        "devices: {} listed; chosen {:?} ({})",
        list.devices.len(),
        chosen.map(|d| d.description.as_str()).unwrap_or("none"),
        chosen.map(|d| d.backend.as_str()).unwrap_or("-"),
    );

    let opts = ChunkOptions {
        timing: Timing::Segments,
        ..ChunkOptions::default()
    };
    for gpu in [true, false] {
        let c = client(&helper);
        let started = Instant::now();
        let loaded = c
            .load(SttLoad {
                model_id: id.clone(),
                model_dir: dir.display().to_string(),
                kind: SttModelKind::Whisper,
                gpu,
                threads: None,
            })
            .expect("load the model");
        let load_wall = started.elapsed().as_secs_f64();

        let started = Instant::now();
        let warm = c.transcribe(&audio, &opts).expect("warm-up transcription");
        let warm_s = started.elapsed().as_secs_f64();
        let words = warm
            .value
            .segments
            .iter()
            .map(|s| s.words.len())
            .sum::<usize>();
        let segments = warm.value.segments.len();

        let mut times = vec![];
        let mut last = warm.backend.clone();
        let mut fell_back = warm.fell_back;
        for _ in 0..RUNS {
            let started = Instant::now();
            let out = c.transcribe(&audio, &opts).expect("transcription");
            times.push(started.elapsed().as_secs_f64());
            last = out.backend.clone();
            fell_back |= out.fell_back;
        }
        let (median, lo, hi) = median_and_range(&times);
        println!(
            "{id:<14} gpu={gpu:<5} load backend={} device={:?} fell_back={} reported={:.2}s wall={load_wall:.2}s | \
             transcribe backend={} device={:?} fell_back={fell_back} warm-up={warm_s:.2}s median={median:.2}s \
             range={lo:.2}-{hi:.2}s rtf={:.1}x segments={segments} words={words}",
            loaded.backend.backend,
            device(&loaded.backend),
            loaded.fell_back,
            loaded.value.seconds,
            last.backend,
            device(&last),
            audio_secs / median,
        );
        c.shutdown();
    }
}
