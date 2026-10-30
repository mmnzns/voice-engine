//! Fillers, measured (spec 9.4): `cargo test -p voice-engine-stt --test real_fillers -- --ignored --test-threads=1
//! --nocapture`. For `whisper-turbo` and `parakeet-ultra` (from `$VOICE_ENGINE_TEST_MODELS`, default Talaan's models
//! folder, only ever read) it transcribes a recording with spoken "um", "uh", "hmm" and "like" in several voices, in
//! `Timing::Words` mode, with `keep_fillers` false and true, and counts the fillers that come back.
//!
//! The catalog's `keeps_fillers` for each model is the verdict of that count (the rule below), and this test fails
//! when the catalog says something else. Only filler tokens and counts are printed, never the transcript.
//!
//! Caveat: the speech is the macOS `say` command. A synthetic "um" is a clean, stressed syllable; a real one is
//! short, quiet and half swallowed. Models may keep text-to-speech fillers more readily than real ones, so a result
//! here is evidence, not proof (docs/bench-2026-10-29.md, "Fillers").

#![cfg(target_os = "macos")]

mod common;

use common::{guard, models_dir, ms, say_with, words_of};
use voice_engine_models::catalog::find;
use voice_engine_stt::{ChunkOptions, LoadOptions, Timing, Transcriber, load};

/// The words that count as a filler (whole word, case-insensitive).
const FILLERS: [&str; 6] = ["um", "umm", "uh", "uhh", "hmm", "like"];
/// Counted by kind in the report: `um` and `umm` together, `uh` and `uhh` together.
const KINDS: [&str; 4] = ["um", "uh", "hmm", "like"];

/// Seven sentences. `like` appears only as a filler (never as a verb or a comparison).
const SENTENCES: [&str; 7] = [
    "So, um, I think we should start on Monday.",
    "The report is, uh, almost finished, but the numbers need another look.",
    "We could, like, move the meeting to the afternoon.",
    "Hmm, that is a good point about the budget.",
    "I talked to the client yesterday, and they said, um, the deadline is flexible, uh, within reason.",
    "Well, the schedule is tight, so we need, like, a clear plan before Friday.",
    "Please send me the draft, hmm, no, wait, send it to the whole team, um.",
];

/// English macOS voices: US, UK and Australian.
const VOICES: [&str; 3] = ["Samantha", "Daniel", "Karen"];

/// The kind of filler a word is (`um`, `uh`, `hmm` or `like`), by whole-word, case-insensitive match after the
/// punctuation around it is trimmed.
fn kind_of(word: &str) -> Option<&'static str> {
    let w: String = word
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if !FILLERS.contains(&w.as_str()) {
        return None;
    }
    Some(match w.as_str() {
        "um" | "umm" => "um",
        "uh" | "uhh" => "uh",
        "hmm" => "hmm",
        _ => "like",
    })
}

/// Filler kinds of the words, in order.
fn fillers_in(words: &[String]) -> Vec<&'static str> {
    words.iter().filter_map(|w| kind_of(w)).collect()
}

fn per_kind(fillers: &[&str]) -> [usize; 4] {
    KINDS.map(|k| fillers.iter().filter(|f| **f == k).count())
}

/// Silence between the sentences (a speaker's pause), in samples.
fn pause() -> Vec<f32> {
    vec![0.0; 6_400]
}

fn recording(voice: &str) -> Vec<f32> {
    let mut clip = vec![];
    for s in SENTENCES {
        clip.extend(say_with(Some(voice), s));
        clip.extend(pause());
    }
    clip
}

/// What one transcription gave: the fillers (printed) and how many of the spoken ones they cover.
struct Run {
    voice: &'static str,
    keep: bool,
    fillers: Vec<&'static str>,
    /// Per kind, the count in the transcript.
    found: [usize; 4],
}

fn measure(id: &str, gpu: bool) -> (Vec<Run>, [usize; 4]) {
    let m = find(id).unwrap();
    let mut spoken_words: Vec<String> = SENTENCES
        .iter()
        .flat_map(|s| s.split_whitespace().map(str::to_string))
        .collect();
    spoken_words.retain(|w| kind_of(w).is_some());
    let spoken = per_kind(&fillers_in(&spoken_words));
    // Written down by hand from the sentences above: um 3, uh 2, hmm 2, like 2.
    assert_eq!(spoken, [3, 2, 2, 2], "the sentences changed");

    let clips: Vec<(&'static str, Vec<f32>)> = VOICES.iter().map(|v| (*v, recording(v))).collect();
    for (voice, clip) in &clips {
        eprintln!("{voice}: recording {} ms", ms(clip));
    }
    let mut engine: Box<dyn Transcriber> = load(
        m,
        &models_dir().join(id),
        LoadOptions { gpu, threads: None },
    )
    .unwrap();
    let mut runs = vec![];
    for (voice, clip) in &clips {
        for keep in [false, true] {
            let t = engine
                .transcribe(
                    clip,
                    &ChunkOptions {
                        timing: Timing::Words,
                        keep_fillers: keep,
                        ..ChunkOptions::default()
                    },
                )
                .unwrap();
            let words = words_of(&t);
            let fillers = fillers_in(&words);
            let found = per_kind(&fillers);
            // Only the filler tokens and counts: never the transcript.
            eprintln!(
                "{id} {voice} keep_fillers={keep}: {} words, fillers {:?} (um {}, uh {}, hmm {}, like {})",
                words.len(),
                fillers,
                found[0],
                found[1],
                found[2],
                found[3]
            );
            // The transcription worked: the recording's ordinary words came back (a count of 0 means "no
            // fillers", not "no decode").
            let heard: String = words.join(" ").to_lowercase();
            for must in ["monday", "report", "meeting", "budget", "friday"] {
                assert!(heard.contains(must), "{id} {voice}: {must:?} not heard");
            }
            runs.push(Run {
                voice,
                keep,
                fillers,
                found,
            });
        }
    }
    (runs, spoken)
}

/// How many filler tokens come back in all, with the option as given: per voice, at most as many as were spoken (a
/// made-up extra one is not a kept one). Counted over the filler words as a whole, not kind by kind: the speech
/// engine's "hmm" is often heard as "um", and a transcript that has a filler there has kept the filler.
/// The per-kind counts are printed next to this so a weak kind shows.
fn covered(runs: &[Run], keep: bool, spoken: [usize; 4]) -> (usize, usize) {
    let per_voice: usize = spoken.iter().sum();
    let mut hit = 0;
    let mut total = 0;
    for r in runs.iter().filter(|r| r.keep == keep) {
        hit += r.found.iter().sum::<usize>().min(per_voice);
        total += per_voice;
    }
    (hit, total)
}

/// The catalog rule (plan, Task 2): `Some(true)` when with the option at least 80% of the spoken fillers appear
/// (that is more than without the option, or the model keeps them either way); `Some(false)` when fewer do.
fn verdict(runs: &[Run], spoken: [usize; 4]) -> Option<bool> {
    let (without, total) = covered(runs, false, spoken);
    let (with, _) = covered(runs, true, spoken);
    eprintln!(
        "without the option {without}/{total} ({:.0}%), with it {with}/{total} ({:.0}%)",
        100.0 * without as f64 / total as f64,
        100.0 * with as f64 / total as f64
    );
    Some(with * 5 >= total * 4)
}

fn run(id: &str, gpu: bool) {
    let _g = guard();
    let (runs, spoken) = measure(id, gpu);
    eprintln!(
        "{id}: spoken per voice: um {}, uh {}, hmm {}, like {} ({} in all)",
        spoken[0],
        spoken[1],
        spoken[2],
        spoken[3],
        spoken.iter().sum::<usize>()
    );
    for r in &runs {
        eprintln!(
            "{id}: {} keep_fillers={}: {} of {} spoken",
            r.voice,
            r.keep,
            r.found.iter().sum::<usize>().min(spoken.iter().sum()),
            spoken.iter().sum::<usize>()
        );
        assert_eq!(r.fillers.len(), r.found.iter().sum::<usize>());
    }
    let measured = verdict(&runs, spoken);
    eprintln!("{id}: measured keeps_fillers = {measured:?}");
    assert_eq!(
        find(id).unwrap().keeps_fillers,
        measured,
        "{id}: the catalog's keeps_fillers does not match this measurement"
    );
}

#[test]
#[ignore]
fn whisper_turbo_fillers_match_the_catalog() {
    run("whisper-turbo", true);
}

#[test]
#[ignore]
fn parakeet_ultra_fillers_match_the_catalog() {
    run("parakeet-ultra", false);
}
