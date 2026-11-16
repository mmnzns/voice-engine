//! Multitalker Parakeet TDT on the real models, on a Mac or a Windows PC: `cargo test -p voice-engine-stt --test
//! real_multitalker -- --ignored --test-threads=1 --nocapture`.
//!
//! The models are `multitalker-tdt` and `diarization`, each looked up on its own: in `$VOICE_ENGINE_TEST_MODELS` when
//! that is set, else in the shared folder, then in Talaan's folder (`common`), only ever read. The speech is made at
//! test time with two different voices (Samantha and Daniel from `say` on a Mac, Zira and David from the Windows voices
//! on Windows; `common::voice_pair`): this file needs voices that speak their own words, which a single
//! `VOICE_ENGINE_TEST_CLIP` cannot give, so each test is skipped (it prints why and passes) anywhere else. The two
//! voices are mixed so that they talk over each other. Prints counts, sizes, times and the number of speakers, never
//! text. Every test goes through the same lock as the other real tests.

mod common;

use common::{guard, model_dir, ms, voice_pair};
use std::path::PathBuf;
use std::time::Instant;
use voice_engine_models::catalog::find;
use voice_engine_stt::multitalker::{Multitalker, SpeakerTurn, load};
use voice_engine_stt::threads::{ThreadKind, default_threads, logical_cores};

const ID: &str = "multitalker-tdt";

const FIRST: &str = "Good morning everyone, and thank you for joining the call today. I would like to start with the plan for the \
     week, and then hand over to the others. The schedule has changed a little since the last time we spoke.";
const SECOND: &str = "Thanks for that. From where I sit the schedule looks rather tight, so let us talk it through step by step. \
     I have a few questions about the budget as well, and about who is travelling to the partner site.";

fn dir(id: &str) -> PathBuf {
    let dir = model_dir(id);
    for f in find(id).unwrap().files {
        assert!(
            dir.join(f.name).is_file(),
            "the test model {id} is not complete at {}",
            dir.display()
        );
    }
    dir
}

fn model(threads: Option<usize>) -> Multitalker {
    load(find(ID).unwrap(), &dir(ID), &dir("diarization"), threads).unwrap()
}

/// `a` from the start and `b` from `offset` samples on, added together at 0.7 each, as long as the longer of the two.
fn mix(a: &[f32], b: &[f32], offset: usize) -> Vec<f32> {
    let len = a.len().max(offset + b.len());
    let mut out = vec![0.0f32; len];
    for (i, s) in a.iter().enumerate() {
        out[i] += 0.7 * s;
    }
    for (i, s) in b.iter().enumerate() {
        out[offset + i] += 0.7 * s;
    }
    out
}

/// One voice, a pause, the other.
fn one_after_the_other(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = a.to_vec();
    out.extend(std::iter::repeat_n(0.0, 8_000));
    out.extend_from_slice(b);
    out
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

fn speakers(turns: &[SpeakerTurn]) -> Vec<usize> {
    let mut s: Vec<usize> = turns.iter().map(|t| t.speaker).collect();
    s.sort_unstable();
    s.dedup();
    s
}

/// The longest stretch, in ms, in which two turns of different speakers are both going.
fn longest_overlap(turns: &[SpeakerTurn]) -> u64 {
    let mut best = 0;
    for (i, a) in turns.iter().enumerate() {
        for b in &turns[i + 1..] {
            if a.speaker != b.speaker {
                let from = a.start_ms.max(b.start_ms);
                let to = a.end_ms.min(b.end_ms);
                best = best.max(to.saturating_sub(from));
            }
        }
    }
    best
}

/// A line of counts, never text.
fn summary(turns: &[SpeakerTurn], audio_ms: u64) -> String {
    format!(
        "{} turns, {} speakers, {} words, longest overlap {} ms, audio {audio_ms} ms",
        turns.len(),
        speakers(turns).len(),
        turns.iter().map(|t| t.words.len()).sum::<usize>(),
        longest_overlap(turns)
    )
}

/// What every result must satisfy: times inside the audio and in order, words in order inside their turn, the turn's
/// text being its words.
fn check_turns(turns: &[SpeakerTurn], audio_ms: u64) {
    let mut last_start = 0;
    for t in turns {
        assert!(!t.words.is_empty() && !t.text.trim().is_empty());
        assert!(t.speaker < 8, "speaker {}", t.speaker);
        assert!(
            t.start_ms <= t.end_ms && t.end_ms <= audio_ms,
            "turn {}..{} ms in {audio_ms} ms",
            t.start_ms,
            t.end_ms
        );
        assert!(
            t.start_ms >= last_start,
            "turns are not in the order they start"
        );
        last_start = t.start_ms;
        let mut last_end = t.start_ms;
        for w in &t.words {
            assert!(w.start_ms <= w.end_ms && w.end_ms <= audio_ms);
            assert!(w.start_ms >= last_end || w.start_ms + 1 >= last_end);
            assert!(!w.text.trim().is_empty());
            last_end = w.end_ms;
        }
        assert_eq!(
            t.text.split_whitespace().count(),
            t.words.len(),
            "the turn's text has as many words as the turn has"
        );
    }
}

#[test]
#[ignore]
fn loads_with_the_thread_rule_and_with_the_apps_number() {
    need_voices!();
    let _g = guard();
    let started = Instant::now();
    let m = model(None);
    println!(
        "multitalker load: {:.1} s, {} threads",
        started.elapsed().as_secs_f64(),
        m.threads()
    );
    assert_eq!(
        m.threads(),
        default_threads(ThreadKind::Processor, logical_cores())
    );
    drop(m);
    assert_eq!(model(Some(3)).threads(), 3);
}

#[test]
#[ignore]
fn two_voices_one_after_the_other_are_two_speakers() {
    need_voices!();
    let _g = guard();
    let (a, b) = voice_pair(FIRST, SECOND);
    let clip = one_after_the_other(&a, &b);
    let audio_ms = ms(&clip);
    let mut m = model(None);
    let started = Instant::now();
    let turns = m.run(&clip).unwrap();
    let secs = started.elapsed().as_secs_f64();
    println!(
        "multitalker, one after the other: {} in {secs:.2} s (real-time factor {:.2}x)",
        summary(&turns, audio_ms),
        audio_ms as f64 / 1000.0 / secs
    );
    check_turns(&turns, audio_ms);
    assert!(speakers(&turns).len() >= 2, "{}", summary(&turns, audio_ms));
    // Almost every spoken word was heard, by count.
    let said = word_count(FIRST) + word_count(SECOND);
    let heard: usize = turns.iter().map(|t| t.words.len()).sum();
    assert!(heard * 10 >= said * 7, "{heard} words heard of {said} said");
}

#[test]
#[ignore]
fn two_voices_talking_over_each_other_give_overlapping_turns() {
    need_voices!();
    let _g = guard();
    let (a, b) = voice_pair(FIRST, SECOND);
    // The second voice starts 2 s in, while the first is still going.
    let clip = mix(&a, &b, 2 * 16_000);
    let audio_ms = ms(&clip);
    let mut m = model(None);
    let started = Instant::now();
    let turns = m.run(&clip).unwrap();
    let secs = started.elapsed().as_secs_f64();
    println!(
        "multitalker, overlapping: {} in {secs:.2} s (real-time factor {:.2}x)",
        summary(&turns, audio_ms),
        audio_ms as f64 / 1000.0 / secs
    );
    check_turns(&turns, audio_ms);
    assert!(speakers(&turns).len() >= 2, "{}", summary(&turns, audio_ms));
    assert!(
        longest_overlap(&turns) >= 1_000,
        "no two speakers overlap: {}",
        summary(&turns, audio_ms)
    );
    let said = word_count(FIRST) + word_count(SECOND);
    let heard: usize = turns.iter().map(|t| t.words.len()).sum();
    assert!(heard * 10 >= said * 6, "{heard} words heard of {said} said");
}

#[test]
#[ignore]
fn a_long_track_is_cut_inside_and_keeps_one_clock() {
    need_voices!();
    let _g = guard();
    let (a, b) = voice_pair(FIRST, SECOND);
    // About 70 s: the two voices apart, then overlapping, then apart again: more than three of the model's 20 s chunks.
    let mut clip = one_after_the_other(&a, &b);
    clip.extend(std::iter::repeat_n(0.0, 16_000));
    clip.extend(mix(&a, &b, 3 * 16_000));
    clip.extend(std::iter::repeat_n(0.0, 16_000));
    clip.extend(one_after_the_other(&b, &a));
    let audio_ms = ms(&clip);
    assert!(audio_ms > 60_000, "{audio_ms} ms of audio");
    let mut m = model(None);
    let started = Instant::now();
    let turns = m.run(&clip).unwrap();
    let secs = started.elapsed().as_secs_f64();
    println!(
        "multitalker, long track: {} in {secs:.2} s (real-time factor {:.2}x)",
        summary(&turns, audio_ms),
        audio_ms as f64 / 1000.0 / secs
    );
    check_turns(&turns, audio_ms);
    assert!(speakers(&turns).len() >= 2);
    // Turns exist late in the track too: the later chunks are on the same clock as the first.
    assert!(
        turns.iter().any(|t| t.end_ms > audio_ms * 3 / 4),
        "no turn in the last quarter: {}",
        summary(&turns, audio_ms)
    );
    assert!(turns.iter().any(|t| t.start_ms < audio_ms / 4));
}

#[test]
#[ignore]
fn empty_audio_and_silence_give_no_failure() {
    need_voices!();
    let _g = guard();
    let mut m = model(None);
    assert!(m.run(&[]).unwrap().is_empty());
    for samples in [160usize, 16_000, 5 * 16_000] {
        let started = Instant::now();
        let turns = m
            .run(&vec![0.0f32; samples])
            .unwrap_or_else(|e| panic!("{samples} samples of silence: {e}"));
        println!(
            "multitalker, {samples} samples of silence: {} turns in {:.2} s",
            turns.len(),
            started.elapsed().as_secs_f64()
        );
        check_turns(&turns, samples as u64 * 1000 / 16_000);
    }
}
