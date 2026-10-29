//! Speaker separation on the real model, on the Mac: `cargo test -p voice-engine-stt --test real_diarize -- --ignored
//! --test-threads=1 --nocapture`.
//!
//! The model is `diarization/nemotron3_diar_v3.onnx` from `$VOICE_ENGINE_TEST_MODELS` (default: Talaan's models
//! folder, `~/Library/Application Support/Talaan/models`), only ever read. The two voices are made with the macOS
//! `say` command (Samantha, then Daniel; see `common`). Every test goes through the same lock as the other real
//! tests.

#![cfg(target_os = "macos")]

mod common;

use common::{guard, models_dir, ms, say_with};
use std::path::PathBuf;
use std::time::Instant;
use voice_engine_models::catalog::find;
use voice_engine_stt::SpeechError;
use voice_engine_stt::diarize::{self, Diarizer, FRAME_MS, SPEAKERS, SpeakerFrame, load};
use voice_engine_stt::threads::{ThreadKind, default_threads, logical_cores};

fn model_dir() -> PathBuf {
    let dir = models_dir().join("diarization");
    let file = find("diarization").unwrap().files[0].name;
    assert!(
        dir.join(file).is_file(),
        "the test model is not at {}",
        dir.display()
    );
    dir
}

fn diarizer(threads: Option<usize>) -> Diarizer {
    load(&model_dir(), threads).unwrap()
}

const SAMANTHA: &str = "Good morning everyone, and thank you for joining the call today. I would like to start with the plan.";
const DANIEL: &str =
    "Thanks for that. From where I sit the schedule looks rather tight, so let us talk it through.";
const SILENCE_SAMPLES: usize = 8_000; // 0.5 s

/// Samantha, 0.5 s of silence, Daniel. Returns the clip and the two voices' lengths in samples.
fn two_voices() -> (Vec<f32>, usize, usize) {
    let a = say_with(Some("Samantha"), SAMANTHA);
    let b = say_with(Some("Daniel"), DANIEL);
    let mut clip = a.clone();
    clip.extend(std::iter::repeat_n(0.0, SILENCE_SAMPLES));
    clip.extend_from_slice(&b);
    (clip, a.len(), b.len())
}

/// The mean of each speaker column over the frames that start in `[from_ms, to_ms)`.
fn means(frames: &[SpeakerFrame], from_ms: u64, to_ms: u64) -> [f32; SPEAKERS] {
    let mut sum = [0.0f32; SPEAKERS];
    let mut n = 0;
    for f in frames
        .iter()
        .filter(|f| f.start_ms >= from_ms && f.start_ms < to_ms)
    {
        for (s, p) in sum.iter_mut().zip(f.speakers) {
            *s += p;
        }
        n += 1;
    }
    assert!(n > 0, "no frames in {from_ms}..{to_ms} ms");
    sum.map(|s| s / n as f32)
}

/// The column with the highest mean, its mean, and the next highest mean.
fn dominant(means: &[f32; SPEAKERS]) -> (usize, f32, f32) {
    let mut order: Vec<usize> = (0..SPEAKERS).collect();
    order.sort_by(|&a, &b| means[b].total_cmp(&means[a]));
    (order[0], means[order[0]], means[order[1]])
}

/// The share of the frames starting in `[from_ms, to_ms)` in which `column` has the highest probability.
fn top_share(frames: &[SpeakerFrame], from_ms: u64, to_ms: u64, column: usize) -> f32 {
    let inside: Vec<_> = frames
        .iter()
        .filter(|f| f.start_ms >= from_ms && f.start_ms < to_ms)
        .collect();
    assert!(!inside.is_empty());
    let on_top = inside
        .iter()
        .filter(|f| (0..SPEAKERS).all(|k| k == column || f.speakers[k] <= f.speakers[column]))
        .count();
    on_top as f32 / inside.len() as f32
}

fn show(m: &[f32; SPEAKERS]) -> String {
    m.iter()
        .map(|v| format!("{v:.2}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// KNOWN TO FAIL on this model, and kept `#[ignore]` on purpose: the assertions are the ones the plan asks for and are
/// not weakened. One 5 s turn per voice is too little for this streaming model to tell two `say` voices apart. Measured
/// on this Mac (macOS voices Samantha and Daniel; per-frame top column, mean probability of each half's top column):
///
/// * these two sentences (5.5 s and 5.6 s): Samantha on column 0 (mean 0.93), Daniel on column 0 as well (0.61, with
///   column 1 at 0.33): the same column for both, so the assertion `c1 != c2` fails;
/// * two shorter sentences (3.6 s and 3.8 s): column 0 then column 1, but only 0.54 against 0.36 for Daniel;
///   two longer ones (7.1 s and 7.5 s): column 0 then column 1, 0.59 against 0.31;
/// * Samantha with Fred, Rishi or Ralph, Moira with Daniel, Tessa with Alex: one column for both voices every time;
///   Karen then Daniel: separate on short sentences, the same column on long ones;
/// * the same voices four turns each (A B A B A B A B): one column for all of them.
///
/// The control is `three_voices_in_a_conversation_...` (twelve turns, Karen, Daniel and Rishi): three columns, 11 of 12
/// turns right, and Talaan's own `three-speakers` fixture (62 s, 12 turns) comes out 12 of 12 through this code. So the
/// port does what Talaan's does; two synthetic voices in one turn each are below what the model can separate.
#[test]
#[ignore]
fn two_different_voices_are_two_different_speaker_columns() {
    let _g = guard();
    let (clip, a_len, b_len) = two_voices();
    let (a_ms, b_ms) = (ms(&clip[..a_len]), ms(&clip[..b_len]));
    eprintln!(
        "clip: Samantha {a_ms} ms, 500 ms of silence, Daniel {b_ms} ms; {} ms in all",
        ms(&clip)
    );
    let mut d = diarizer(None);
    let started = Instant::now();
    let frames = d.run(&clip).unwrap();
    eprintln!(
        "{} frames in {:.1} s on {} threads",
        frames.len(),
        started.elapsed().as_secs_f32(),
        d.threads()
    );
    // The shape: one frame per 10 ms, covering the clip with no gap, the last one cut at the end of the audio.
    assert_eq!(frames.len() as u64, ms(&clip).div_ceil(FRAME_MS));
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.start_ms, i as u64 * FRAME_MS);
        assert!(f.end_ms > f.start_ms && f.end_ms <= ms(&clip), "{f:?}");
        assert!(f.speakers.iter().all(|p| (0.0..=1.0).contains(p)), "{f:?}");
    }
    assert_eq!(frames.last().unwrap().end_ms, ms(&clip));

    // First voice: the frames over its speech (a little in from the start and from the silence, where the model
    // is settling or has let go). Second voice: the same.
    // Per second: the column with the highest mean, to see where each voice is picked up.
    for from in (0..ms(&clip)).step_by(1_000) {
        let (c, m, _) = dominant(&means(&frames, from, (from + 1_000).min(ms(&clip))));
        eprintln!("  {from:>5} ms: column {c} at {m:.2}");
    }

    // First voice: the frames over its speech (a little in from the start and from the silence, where the model is
    // settling or has let go). Second voice: the same.
    let margin = 300;
    let (first_from, first_to) = (margin, a_ms - margin);
    let (second_from, second_to) = (a_ms + 500 + margin, ms(&clip) - margin);
    let first = means(&frames, first_from, first_to);
    let second = means(&frames, second_from, second_to);
    eprintln!("first voice  (Samantha) column means: {}", show(&first));
    eprintln!("second voice (Daniel)   column means: {}", show(&second));
    let (c1, _, _) = dominant(&first);
    let (c2, _, _) = dominant(&second);
    let (s1, s2) = (
        top_share(&frames, first_from, first_to, c1),
        top_share(&frames, second_from, second_to, c2),
    );
    eprintln!(
        "first half: column {c1} is the top column in {:.0}% of frames; second half: column {c2} in {:.0}%",
        s1 * 100.0,
        s2 * 100.0
    );

    // One column dominates each half: it is the model's top column in at least 80% of that half's frames, and it
    // is not the same column in both halves. (The model keeps part of the second voice on the first voice's
    // column, so the second voice's mean probability is well under the first's; what counts is who is on top.)
    assert!(s1 >= 0.8, "no column dominates the first half: {first:?}");
    assert!(s2 >= 0.8, "no column dominates the second half: {second:?}");
    assert_ne!(
        c1, c2,
        "the same column for both voices: {first:?} {second:?}"
    );
    // Each half's column is clearly above the other's in its own half.
    assert!(first[c1] > 2.0 * first[c2], "{first:?}");
    assert!(second[c2] > second[c1], "{second:?}");
}

/// Three voices in a conversation: four rounds of Karen, Daniel, Rishi, each turn a sentence or two, 0.6 s apart
/// (the shape of Talaan's own `three-speakers` fixture, which its port to this crate reproduces: 12 turns of 12
/// right). Returns the clip and each turn's `(voice index, start_ms, end_ms)`.
fn conversation() -> (Vec<f32>, Vec<(usize, u64, u64)>) {
    let voices = ["Karen", "Daniel", "Rishi"];
    let lines = [
        "Okay everyone, let us start with the launch timeline. Where are we on the landing page?",
        "The landing page is done. We still need the pricing table and the legal copy before we can publish.",
        "I can take the pricing table. I will have a draft by Thursday, assuming the numbers are final.",
        "Good. Legal says they need two days, so the copy has to go out tomorrow morning at the latest.",
        "Then I will write it tonight. Should the footer link to the old terms or to the new ones?",
        "The new ones, please. They went live on Monday and the old page is going away next week.",
        "Perfect. Can someone also check the mobile layout, because last time the buttons were cut off?",
        "I checked it this morning on two phones and it looks fine, but I will test one more.",
        "Great, thank you. Then the only open item is the price for the yearly plan.",
        "That depends on the discount we decide for the first year, which we should settle today.",
        "Let us keep it simple and take ten percent, and revisit it after the first month of sales.",
        "Agreed. Then I will send the summary to everyone and we can meet again on Friday.",
    ];
    let mut clip: Vec<f32> = vec![];
    let mut turns = vec![];
    for (i, line) in lines.iter().enumerate() {
        let voice = i % 3;
        let audio = say_with(Some(voices[voice]), line);
        let start = ms(&clip);
        clip.extend_from_slice(&audio);
        turns.push((voice, start, ms(&clip)));
        clip.extend(std::iter::repeat_n(0.0, 9_600));
    }
    (clip, turns)
}

#[test]
#[ignore]
fn three_voices_in_a_conversation_get_three_columns_and_most_turns_land_on_their_own() {
    // The positive control for the two-voice test below: the same model, real speech from `say`, twelve turns so the
    // model hears each voice several times. A voice's column is the one that is top in most of its turns; the three
    // voices must get three different columns, and at least 10 of the 12 turns must be on their voice's column. A
    // streaming model absorbs a new voice's first turn into a voice it already knows, so a miss on a first turn is the
    // model, not this code (the warm-up cuts those misses; Talaan measured 8 of 12 without it and 12 of 12 with it on
    // its own fixture, which this port also gets right).
    let _g = guard();
    let (clip, turns) = conversation();
    let mut d = diarizer(None);
    let started = Instant::now();
    let frames = d.run(&clip).unwrap();
    eprintln!(
        "{} ms, {} frames in {:.1} s",
        ms(&clip),
        frames.len(),
        started.elapsed().as_secs_f32()
    );
    let names = ["Karen", "Daniel", "Rishi"];
    let tops: Vec<(usize, usize)> = turns
        .iter()
        .map(|&(voice, from, to)| {
            let m = means(&frames, from, to);
            let (c, p, _) = dominant(&m);
            eprintln!(
                "  {:>6} {from:>6}-{to:<6} column {c} at {p:.2}  ({})",
                names[voice],
                show(&m)
            );
            (voice, c)
        })
        .collect();
    let column_of: Vec<usize> = (0..3)
        .map(|voice| {
            (0..SPEAKERS)
                .max_by_key(|&c| tops.iter().filter(|&&(v, t)| v == voice && t == c).count())
                .unwrap()
        })
        .collect();
    let right = tops.iter().filter(|&&(v, c)| column_of[v] == c).count();
    eprintln!(
        "columns of Karen, Daniel, Rishi: {column_of:?}; {right} of 12 turns on their voice's column"
    );
    assert!(
        column_of[0] != column_of[1]
            && column_of[0] != column_of[2]
            && column_of[1] != column_of[2],
        "{column_of:?}"
    );
    assert!(
        right >= 10,
        "{right} of 12 turns on their voice's column: {tops:?}"
    );
}

#[test]
#[ignore]
fn the_same_track_twice_gives_the_same_frames_because_each_run_starts_clean() {
    let _g = guard();
    let (clip, _, _) = two_voices();
    let mut d = diarizer(None);
    let one = d.run(&clip).unwrap();
    let two = d.run(&clip).unwrap();
    assert_eq!(one.len(), two.len());
    let worst = one
        .iter()
        .zip(&two)
        .flat_map(|(a, b)| {
            a.speakers
                .iter()
                .zip(b.speakers)
                .map(|(x, y)| (x - y).abs())
        })
        .fold(0.0f32, f32::max);
    eprintln!("largest difference between two runs: {worst}");
    assert!(worst < 1e-3, "{worst}");
}

#[test]
#[ignore]
fn empty_and_very_short_audio_on_the_real_model() {
    let _g = guard();
    let mut d = diarizer(None);
    assert!(d.run(&[]).unwrap().is_empty());
    assert!(d.run(&[0.0; 15]).unwrap().is_empty());
    // 100 samples are 6 ms: one frame, cut to the audio.
    let tiny = d.run(&[0.01; 100]).unwrap();
    assert_eq!(tiny.len(), 1);
    assert_eq!((tiny[0].start_ms, tiny[0].end_ms), (0, 6));
    // Half a second of silence: 50 frames, nobody talking much.
    let quiet = d.run(&vec![0.0; 8_000]).unwrap();
    assert_eq!(quiet.len(), 50);
    assert_eq!(quiet.last().unwrap().end_ms, 500);
    assert!(quiet.iter().all(|f| f.speakers.iter().all(|p| *p < 0.5)));
}

#[test]
#[ignore]
fn the_thread_rule_is_what_the_session_was_built_with() {
    let _g = guard();
    assert_eq!(
        diarizer(None).threads(),
        default_threads(ThreadKind::Processor, logical_cores())
    );
    assert_eq!(diarizer(Some(3)).threads(), 3);
    assert_eq!(diarizer(Some(0)).threads(), 1);
    // Whatever the count, the answer is the same kind of answer.
    let clip = say_with(Some("Samantha"), "One short sentence for the thread test.");
    assert!(!diarizer(Some(1)).run(&clip).unwrap().is_empty());
}

#[test]
#[ignore]
fn a_model_folder_without_the_file_is_model_missing() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        diarize::load(dir.path(), None).err().unwrap(),
        SpeechError::ModelMissing
    );
}

#[test]
#[ignore]
fn lengths_parakeet_rs_trips_on_run_all_the_same() {
    // parakeet-rs 0.3.8 fails ("non-contiguous layout") on a call whose mel frame count (samples / 160 + 1) is a
    // multiple of 8 and holds the whole call in one window. Tracks of 8, 16... frames, 12 s (1 192 frames, a clip
    // that tripped it) and just under 30 s must come back with every frame.
    let _g = guard();
    let mut d = diarizer(None);
    for frames in [8usize, 16, 40, 100 * 8, 1_192, 2_960, 3_040, 3_048] {
        let n = (frames - 1) * 160 + 11;
        let track: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin() * 0.1).collect();
        let got = d
            .run(&track)
            .unwrap_or_else(|e| panic!("{frames} frames: {e:?}"));
        assert_eq!(
            got.len() as u64,
            ms(&track).div_ceil(FRAME_MS),
            "{frames} frames"
        );
        assert_eq!(got.last().unwrap().end_ms, ms(&track));
    }
}
