//! Speed of the v0.4.0 speech models next to the ones they are compared with, on this computer: load time, decode time
//! and real-time factor for one 29 s track. Prints times, sizes and counts only, never the transcript.
//!
//! ```text
//! cargo run --release -p voice-engine-stt --example bench_v040
//! ```
//!
//! Rows: `whisper-turbo` (the processor), `parakeet-ultra`, `parakeet-unified-25`, `cohere-transcribe`, each through its
//! `Transcriber`, on the same audio; then `multitalker-tdt` on two voices that talk over each other for the whole
//! track (its own API: turns, not a transcript). Each line is one load, one warm-up decode, then 3 timed decodes
//! (median and range). `Timing::Segments`, no language (Cohere: English).
//!
//! The audio is made at run time by two voices that speak their own words (`say` on a Mac, the Windows voices on
//! Windows; `support::sapi_samples`): different sentences, so no model stops early on a repeat (a repeated clip makes
//! Cohere and Whisper Turbo stop after the first pass). A model that is not found is named and skipped.

use std::time::Instant;
use voice_engine_models::catalog::find;
use voice_engine_stt::cohere_engine::CohereEngine;
use voice_engine_stt::multitalker;
use voice_engine_stt::parakeet_engine::ParakeetEngine;
use voice_engine_stt::whisper_engine::WhisperEngine;
use voice_engine_stt::{ChunkOptions, LoadOptions, Timing, Transcriber};

#[path = "../support/mod.rs"]
mod support;

const SECS: f64 = 29.0;
const RUNS: usize = 3;

const FIRST: &str = "The first item on the list is the schedule for next week. We moved the review to Tuesday morning because \
    the room is free then. After that, the team will walk through the open questions one by one. Nobody has asked for \
    more time, so the plan stays as it is. The second item is about the budget, which is close to what we expected at \
    the start of the year. Travel costs are a little higher, and the printing costs are lower than before. If anything \
    changes, the finance group will send a short note on Friday. The last item is the visit from the partner company, \
    which is booked for the end of the month. Please check that your calendars are clear, and tell us early if they \
    are not.";
const SECOND: &str = "Thanks for that. From where I sit the schedule looks rather tight, so let us talk it through step by step. \
    I have a few questions about the budget as well, and about who is travelling to the partner site. Before we go on, \
    can somebody say who is writing the summary, and when it will be ready? I would also like to know whether the \
    printing order has been placed, because the deadline is closer than it looks. Finally, the visit needs a room for \
    twelve people, a projector, and coffee, and I will ask the front desk about all three.";

fn speak(voice_mac: &str, voice_win: &str, text: &str) -> Vec<f32> {
    if cfg!(target_os = "macos") {
        support::say_samples(Some(voice_mac), text)
    } else {
        support::sapi_samples(voice_win, text)
    }
    .unwrap_or_else(|e| {
        eprintln!("bench_v040: {e}");
        std::process::exit(2);
    })
}

fn median_and_range(times: &[f64]) -> (f64, f64, f64) {
    let mut t = times.to_vec();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (t[t.len() / 2], t[0], t[t.len() - 1])
}

fn rtf(audio_secs: f64, decode_secs: f64) -> f64 {
    audio_secs / decode_secs
}

fn row(
    label: &str,
    threads: usize,
    load_s: f64,
    first_s: f64,
    times: &[f64],
    count: &str,
    audio_secs: f64,
) {
    let (median, min, max) = median_and_range(times);
    println!(
        "{label:<26} threads={threads:<2} load={load_s:>5.2}s first={first_s:>5.2}s decode median={median:>5.2}s \
         range={min:.2}-{max:.2}s rtf={:>5.1}x {count}",
        rtf(audio_secs, median)
    );
}

fn measure(label: &str, audio: &[f32], load: impl FnOnce() -> (Box<dyn Transcriber>, usize, f64)) {
    let audio_secs = audio.len() as f64 / 16_000.0;
    let (mut engine, threads, load_s) = load();
    let opts = ChunkOptions {
        timing: Timing::Segments,
        ..ChunkOptions::default()
    };
    let started = Instant::now();
    let warm = engine.transcribe(audio, &opts).expect("warm-up decode");
    let first_s = started.elapsed().as_secs_f64();
    let words = warm.segments.iter().map(|s| s.words.len()).sum::<usize>();
    let chars = warm
        .segments
        .iter()
        .map(|s| s.text.chars().count())
        .sum::<usize>();
    let mut times = vec![];
    for _ in 0..RUNS {
        let started = Instant::now();
        engine.transcribe(audio, &opts).expect("decode");
        times.push(started.elapsed().as_secs_f64());
    }
    row(
        label,
        threads,
        load_s,
        first_s,
        &times,
        &format!(
            "segments={} words={words} chars={chars}",
            warm.segments.len()
        ),
        audio_secs,
    );
}

fn main() {
    let a = speak("Samantha", "Microsoft Zira Desktop", FIRST);
    let b = speak("Daniel", "Microsoft David Desktop", SECOND);
    let want = (SECS * 16_000.0) as usize;
    // One voice for the speed rows.
    let mut one = a.clone();
    one.truncate(want);
    let one_secs = one.len() as f64 / 16_000.0;
    // Two voices talking over each other for the Multitalker row: the second comes in 2 s later.
    let mut two = vec![0.0f32; want];
    for (i, s) in a.iter().take(want).enumerate() {
        two[i] += 0.7 * s;
    }
    for (i, s) in b.iter().enumerate().filter(|(i, _)| i + 32_000 < want) {
        two[i + 32_000] += 0.7 * s;
    }
    println!(
        "track: {one_secs:.1} s, 16 kHz mono; {} logical cores; {RUNS} timed runs after 1 warm-up; Timing::Segments",
        voice_engine_stt::threads::logical_cores()
    );
    let cpu = LoadOptions {
        gpu: false,
        threads: None,
    };

    match support::find_model_dir("whisper-turbo") {
        Ok(dir) => measure("whisper-turbo (cpu)", &one, || {
            let started = Instant::now();
            let e =
                WhisperEngine::load(&dir.join("ggml-large-v3-turbo-q5_0.bin"), cpu).expect("load");
            let load_s = started.elapsed().as_secs_f64();
            let threads = e.threads();
            (Box::new(e), threads, load_s)
        }),
        Err(why) => println!("whisper-turbo: skipped, {why}"),
    }
    for id in ["parakeet-ultra", "parakeet-unified-25"] {
        match support::find_model_dir(id) {
            Ok(dir) => measure(id, &one, || {
                let started = Instant::now();
                let e = ParakeetEngine::load(find(id).unwrap(), &dir, cpu).expect("load");
                let load_s = started.elapsed().as_secs_f64();
                let threads = e.threads();
                (Box::new(e), threads, load_s)
            }),
            Err(why) => println!("{id}: skipped, {why}"),
        }
    }
    match support::find_model_dir("cohere-transcribe") {
        Ok(dir) => measure("cohere-transcribe", &one, || {
            let started = Instant::now();
            let e =
                CohereEngine::load(find("cohere-transcribe").unwrap(), &dir, cpu).expect("load");
            let load_s = started.elapsed().as_secs_f64();
            let threads = e.threads();
            (Box::new(e), threads, load_s)
        }),
        Err(why) => println!("cohere-transcribe: skipped, {why}"),
    }
    match (
        support::find_model_dir("multitalker-tdt"),
        support::find_model_dir("diarization"),
    ) {
        (Ok(dir), Ok(diar)) => {
            let started = Instant::now();
            let mut m = multitalker::load(find("multitalker-tdt").unwrap(), &dir, &diar, None)
                .expect("load");
            let load_s = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let warm = m.run(&two).expect("warm-up");
            let first_s = started.elapsed().as_secs_f64();
            let mut times = vec![];
            for _ in 0..RUNS {
                let started = Instant::now();
                m.run(&two).expect("run");
                times.push(started.elapsed().as_secs_f64());
            }
            let mut speakers: Vec<usize> = warm.iter().map(|t| t.speaker).collect();
            speakers.sort_unstable();
            speakers.dedup();
            row(
                "multitalker-tdt (2 voices)",
                m.threads(),
                load_s,
                first_s,
                &times,
                &format!(
                    "turns={} speakers={} words={}",
                    warm.len(),
                    speakers.len(),
                    warm.iter().map(|t| t.words.len()).sum::<usize>()
                ),
                two.len() as f64 / 16_000.0,
            );
        }
        (m, d) => println!(
            "multitalker-tdt: skipped, {}",
            m.err().or(d.err()).unwrap_or_default()
        ),
    }
}
