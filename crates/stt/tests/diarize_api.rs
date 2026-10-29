//! Speaker separation tests that need no real model: milliseconds per frame, clamping to the real audio, the feeding
//! order and the warm-up (with a scripted model), empty and very short audio, errors, the thread rule, and the
//! errors of a load. The model itself runs in `tests/real_diarize.rs` (ignored).

use std::path::Path;
use voice_engine_models::catalog::{Kind, find};
use voice_engine_stt::SpeechError;
use voice_engine_stt::diarize::{
    CHUNK_SAMPLES, Diarizer, FRAME_MS, HOP_SAMPLES, SPEAKERS, SpeakerFrame, SpeakerModel,
    WARMUP_SAMPLES, audio_for_model, frames_from_rows, load, run_track, run_with, thread_count,
};

/// What a scripted model was asked, in order.
#[derive(Debug, Clone, PartialEq)]
enum Call {
    Reset,
    /// The length of the audio and its first sample.
    Feed {
        len: usize,
        first: f32,
    },
}

/// A model that answers like the real one (one row per 10 ms frame, plus the extra frame the real model's centred
/// window adds: `len / 160 + 1` rows) and writes each row's first sample into column 0 / 100 000 (the track is
/// a ramp, so a row says which part of the track it came from). It records what it was asked.
#[derive(Default)]
struct Scripted {
    calls: Vec<Call>,
    /// Fail the n-th feed (0 based).
    fail_feed: Option<usize>,
    feeds: usize,
    /// Rows to answer with instead of the real model's count.
    rows_override: Option<usize>,
}

impl SpeakerModel for Scripted {
    fn reset(&mut self) {
        self.calls.push(Call::Reset);
    }
    fn feed(&mut self, audio: &[f32]) -> Result<Vec<[f32; SPEAKERS]>, SpeechError> {
        self.calls.push(Call::Feed {
            len: audio.len(),
            first: audio.first().copied().unwrap_or(f32::NAN),
        });
        let n = self.feeds;
        self.feeds += 1;
        if self.fail_feed == Some(n) {
            return Err(SpeechError::Internal("onnx runtime".into()));
        }
        let rows = self.rows_override.unwrap_or(audio.len() / HOP_SAMPLES + 1);
        Ok((0..rows)
            .map(|r| {
                let mut row = [0.0; SPEAKERS];
                row[0] = audio.get(r * HOP_SAMPLES).copied().unwrap_or(0.0) / 100_000.0;
                row
            })
            .collect())
    }
}

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

fn row(v: f32) -> [f32; SPEAKERS] {
    let mut r = [0.0; SPEAKERS];
    r[0] = v;
    r
}

// ---- frames to milliseconds ----

#[test]
fn a_frame_is_ten_milliseconds() {
    assert_eq!(FRAME_MS, 10);
    assert_eq!(HOP_SAMPLES, 160);
    assert_eq!(SPEAKERS, 8);
}

#[test]
fn frame_n_covers_ten_n_to_ten_n_plus_ten_milliseconds() {
    let rows: Vec<_> = (0..5).map(|i| row(i as f32 / 10.0)).collect();
    let frames = frames_from_rows(rows, 1_000);
    let times: Vec<_> = frames.iter().map(|f| (f.start_ms, f.end_ms)).collect();
    assert_eq!(times, [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]);
    // The probabilities travel with their frame, per speaker column.
    assert_eq!(frames[3].speakers[0], 0.3);
}

#[test]
fn all_eight_columns_are_kept_in_order() {
    let r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    assert_eq!(frames_from_rows(vec![r], 1_000)[0].speakers, r);
}

#[test]
fn frames_stop_at_the_real_audio_length() {
    // The model's window is centred, so it answers one frame more than the audio holds: 100 ms of audio is
    // 10 frames, the model says 11.
    let rows = vec![row(0.5); 11];
    let frames = frames_from_rows(rows, 100);
    assert_eq!(frames.len(), 10);
    assert_eq!(frames.last().unwrap().end_ms, 100);
}

#[test]
fn the_last_frame_is_cut_where_the_audio_ends() {
    // 62 ms of audio: frames 0..=6, the last one 60 to 62, not 60 to 70.
    let frames = frames_from_rows(vec![row(0.5); 9], 62);
    assert_eq!(frames.len(), 7);
    assert_eq!((frames[6].start_ms, frames[6].end_ms), (60, 62));
}

#[test]
fn no_frame_starts_at_or_after_the_end_and_none_is_empty() {
    for f in frames_from_rows(vec![row(0.5); 50], 333) {
        assert!(f.start_ms < f.end_ms, "{f:?}");
        assert!(f.end_ms <= 333, "{f:?}");
    }
    assert!(frames_from_rows(vec![row(0.5); 5], 0).is_empty());
    assert!(frames_from_rows(vec![], 1_000).is_empty());
}

#[test]
fn a_probability_is_kept_inside_zero_to_one() {
    let mut r = [0.5; SPEAKERS];
    r[0] = f32::NAN;
    r[1] = -0.2;
    r[2] = 1.7;
    r[3] = f32::INFINITY;
    let f = frames_from_rows(vec![r], 100)[0];
    assert_eq!(f.speakers[0], 0.0);
    assert_eq!(f.speakers[1], 0.0);
    assert_eq!(f.speakers[2], 1.0);
    assert_eq!(f.speakers[3], 1.0);
    assert_eq!(f.speakers[4], 0.5);
}

// ---- the feeding order and the warm-up ----

#[test]
fn the_state_is_reset_then_warmed_up_on_the_start_then_the_track_is_fed_in_order() {
    // Warm-up 1 000 samples, chunks of 800, a 2 500 sample track: 800, 800, 800, 100.
    let track = ramp(2_500);
    let mut model = Scripted::default();
    run_with(&mut model, &track, 1_000, 800).unwrap();
    assert_eq!(
        model.calls,
        [
            Call::Reset,
            Call::Feed {
                len: 1_000,
                first: 0.0
            },
            Call::Feed {
                len: 800,
                first: 0.0
            },
            Call::Feed {
                len: 800,
                first: 800.0
            },
            Call::Feed {
                len: 800,
                first: 1_600.0
            },
            Call::Feed {
                len: 100,
                first: 2_400.0
            },
        ]
    );
}

#[test]
fn the_warm_up_answer_is_thrown_away_and_the_rest_comes_back_in_track_order() {
    let track = ramp(2_400);
    let mut model = Scripted::default();
    let frames = run_with(&mut model, &track, 1_000, 800).unwrap();
    // One frame per 160 samples, from the track and not from the warm-up, in order, with no gap or repeat: frame j
    // says it came from sample 160 j.
    assert_eq!(frames.len(), 2_400 / HOP_SAMPLES);
    for (j, f) in frames.iter().enumerate() {
        assert_eq!(f.start_ms, j as u64 * 10);
        let from = (f.speakers[0] * 100_000.0).round() as usize;
        assert_eq!(from, j * HOP_SAMPLES, "frame {j} came from sample {from}");
    }
}

#[test]
fn a_track_shorter_than_the_warm_up_warms_up_on_all_of_it() {
    let track = ramp(1_600);
    let mut model = Scripted::default();
    run_with(&mut model, &track, 5_000, 800).unwrap();
    assert_eq!(
        model.calls,
        [
            Call::Reset,
            Call::Feed {
                len: 1_600,
                first: 0.0
            },
            Call::Feed {
                len: 800,
                first: 0.0
            },
            Call::Feed {
                len: 800,
                first: 800.0
            },
        ]
    );
}

#[test]
fn the_warm_up_is_ten_minutes_and_a_call_is_five_minutes() {
    assert_eq!(WARMUP_SAMPLES, 10 * 60 * 16_000);
    assert_eq!(CHUNK_SAMPLES, 5 * 60 * 16_000);
    // A whole number of 10 ms frames, so chunk edges fall on frame edges.
    assert_eq!(CHUNK_SAMPLES % HOP_SAMPLES, 0);
    // 12 minutes: 10 minutes of warm-up, then 5 + 5 + 2 of track.
    let track = vec![0.0f32; 12 * 60 * 16_000];
    let mut model = Scripted::default();
    let frames = run_track(&mut model, &track).unwrap();
    let lens: Vec<_> = model
        .calls
        .iter()
        .map(|c| match c {
            Call::Reset => 0,
            Call::Feed { len, .. } => *len,
        })
        .collect();
    assert_eq!(
        lens,
        [
            0,
            10 * 60 * 16_000,
            5 * 60 * 16_000,
            5 * 60 * 16_000,
            2 * 60 * 16_000
        ]
    );
    assert_eq!(frames.len(), 12 * 60 * 100);
}

#[test]
fn chunks_do_not_push_later_frames_late() {
    // The real model answers one frame more than a chunk holds. Taken as they come, each chunk would shift the next
    // one by 10 ms; the frames of a long track still line up with its clock.
    let track = ramp(2_400);
    let mut model = Scripted::default();
    let frames = run_with(&mut model, &track, 800, 800).unwrap();
    assert_eq!(frames.len(), 15);
    assert_eq!(frames.last().unwrap().end_ms, 150);
    let froms: Vec<usize> = frames
        .iter()
        .map(|f| (f.speakers[0] * 100_000.0).round() as usize)
        .collect();
    assert_eq!(froms, (0..15).map(|j| j * 160).collect::<Vec<_>>());
}

#[test]
fn a_model_that_answers_short_leaves_the_missing_frames_out_rather_than_inventing_them() {
    let track = ramp(1_600);
    let mut model = Scripted {
        rows_override: Some(4),
        ..Scripted::default()
    };
    let frames = run_with(&mut model, &track, 1_600, 1_600).unwrap();
    assert_eq!(frames.len(), 4);
    assert_eq!(frames.last().unwrap().end_ms, 40);
}

// ---- empty and very short audio ----

#[test]
fn empty_audio_gives_no_frames_and_never_reaches_the_model() {
    let mut model = Scripted::default();
    assert!(run_track(&mut model, &[]).unwrap().is_empty());
    assert!(!model.calls.iter().any(|c| matches!(c, Call::Feed { .. })));
}

#[test]
fn audio_too_short_to_hold_a_millisecond_gives_no_frames() {
    let mut model = Scripted::default();
    assert!(run_track(&mut model, &[0.1; 15]).unwrap().is_empty());
    assert!(!model.calls.iter().any(|c| matches!(c, Call::Feed { .. })));
}

#[test]
fn very_short_audio_gives_one_frame_cut_to_the_audio() {
    // 100 samples are 6 ms: one frame, 0 to 6.
    let mut model = Scripted::default();
    let frames = run_track(&mut model, &ramp(100)).unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!((frames[0].start_ms, frames[0].end_ms), (0, 6));
    // A bit over one hop: 161 samples are 10 ms, one whole frame.
    let frames = run_track(&mut Scripted::default(), &ramp(161)).unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!((frames[0].start_ms, frames[0].end_ms), (0, 10));
    // Under a second: 0.5 s is 50 frames.
    let frames = run_track(&mut Scripted::default(), &ramp(8_000)).unwrap();
    assert_eq!(frames.len(), 50);
    assert_eq!(frames.last().unwrap().end_ms, 500);
}

// ---- a length parakeet-rs 0.3.8 cannot run ----

#[test]
fn a_length_that_makes_a_whole_number_of_eight_mel_frames_gets_ten_milliseconds_of_silence() {
    // parakeet-rs 0.3.8 fails ("non-contiguous layout") on a call whose mel frame count (samples / 160 + 1) is a
    // multiple of 8 and holds the whole call in one window; one frame more avoids it. Frame counts 8, 16, 24...
    for frames in [8usize, 16, 24, 40, 3_040, 3_048, 4_000] {
        let audio = ramp((frames - 1) * HOP_SAMPLES + 7);
        let fed = audio_for_model(&audio);
        assert_eq!(fed.len(), audio.len() + HOP_SAMPLES, "{frames} frames");
        assert_eq!(&fed[..audio.len()], &audio[..]);
        assert!(fed[audio.len()..].iter().all(|s| *s == 0.0));
    }
}

#[test]
fn every_other_length_is_fed_as_it_is() {
    for frames in [1usize, 2, 7, 9, 15, 17, 101, 3_039, 3_041] {
        let audio = ramp((frames - 1) * HOP_SAMPLES);
        assert_eq!(
            audio_for_model(&audio).len(),
            audio.len(),
            "{frames} frames"
        );
    }
    // The frame count is the whole number of hops plus one: 7 hops and 159 samples is still 8 frames.
    assert_eq!(
        audio_for_model(&ramp(7 * HOP_SAMPLES + 159)).len(),
        7 * HOP_SAMPLES + 159 + HOP_SAMPLES
    );
    assert_eq!(
        audio_for_model(&ramp(8 * HOP_SAMPLES)).len(),
        8 * HOP_SAMPLES
    );
    assert!(audio_for_model(&[]).is_empty());
}

// ---- errors from the model ----

#[test]
fn an_error_in_the_warm_up_stops_before_the_track_is_fed() {
    let mut model = Scripted {
        fail_feed: Some(0),
        ..Scripted::default()
    };
    let err = run_with(&mut model, &ramp(2_000), 1_000, 800).unwrap_err();
    assert_eq!(err, SpeechError::Internal("onnx runtime".into()));
    assert_eq!(model.feeds, 1);
}

#[test]
fn an_error_in_a_chunk_is_returned_whole_not_as_part_of_the_frames() {
    let mut model = Scripted {
        fail_feed: Some(2),
        ..Scripted::default()
    };
    let err = run_with(&mut model, &ramp(2_000), 1_000, 800).unwrap_err();
    assert_eq!(err, SpeechError::Internal("onnx runtime".into()));
    assert_eq!(model.feeds, 3);
}

// ---- the thread rule ----

#[test]
fn threads_follow_the_processor_rule_by_default() {
    // Three quarters of the logical cores, 2 to 12, as the Parakeet engine.
    assert_eq!(thread_count(None, 16), 12);
    assert_eq!(thread_count(None, 8), 6);
    assert_eq!(thread_count(None, 4), 3);
    assert_eq!(thread_count(None, 1), 2);
    assert_eq!(thread_count(None, 64), 12);
}

#[test]
fn the_apps_own_count_wins_and_is_at_least_one() {
    assert_eq!(thread_count(Some(3), 16), 3);
    assert_eq!(thread_count(Some(1), 16), 1);
    assert_eq!(thread_count(Some(0), 16), 1);
    // Not capped: it is the app's own measurement.
    assert_eq!(thread_count(Some(20), 4), 20);
}

// ---- the model file ----

/// The file name the catalog gives the speaker separation model.
fn model_file() -> &'static str {
    find("diarization").unwrap().files[0].name
}

#[test]
fn the_model_file_name_comes_from_the_catalog() {
    let info = find("diarization").unwrap();
    assert_eq!(info.kind, Kind::Diarization);
    assert_eq!(info.files.len(), 1);
    assert!(model_file().ends_with(".onnx"));
}

#[test]
fn a_folder_without_the_model_is_model_missing() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        load(dir.path(), None).err().unwrap(),
        SpeechError::ModelMissing
    );
    assert_eq!(
        load(&dir.path().join("not-there"), Some(2)).err().unwrap(),
        SpeechError::ModelMissing
    );
}

#[test]
fn a_directory_named_like_the_model_is_model_missing_not_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join(model_file())).unwrap();
    assert_eq!(
        load(dir.path(), None).err().unwrap(),
        SpeechError::ModelMissing
    );
}

#[test]
fn a_file_that_is_not_a_model_is_bad_model() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(model_file()), b"this is not an onnx model").unwrap();
    assert_eq!(load(dir.path(), None).err().unwrap(), SpeechError::BadModel);
    std::fs::write(dir.path().join(model_file()), b"").unwrap();
    assert_eq!(
        load(dir.path(), Some(1)).err().unwrap(),
        SpeechError::BadModel
    );
}

#[test]
fn a_load_error_carries_no_file_name() {
    let dir = tempfile::tempdir().unwrap();
    let secret = "private-meeting-folder";
    let folder = dir.path().join(secret);
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(folder.join(model_file()), b"junk").unwrap();
    let err = load(&folder, None).err().unwrap();
    assert!(!format!("{err} {err:?}").contains(secret));
}

#[test]
fn a_diarizer_can_move_to_a_worker_thread() {
    fn is_send<T: Send>() {}
    is_send::<Diarizer>();
    let _ = Path::new("");
    let _ = SpeakerFrame {
        start_ms: 0,
        end_ms: 10,
        speakers: [0.0; SPEAKERS],
    };
}
