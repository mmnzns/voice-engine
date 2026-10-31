//! What the speed benches (`examples/bench.rs`, `examples/bench_timing.rs`) and the real-model tests (`tests/common`)
//! share, so they run on a Windows PC as well as on a Mac: where the test clip comes from, how a WAV file is read, and
//! where the models are. Only ever reads the model folders. Included with `#[path]`, it is not part of the library.
//!
//! * The clip: a 16 kHz mono WAV file named by `VOICE_ENGINE_TEST_CLIP` (16-bit PCM or 32-bit float; another sample rate
//!   or more than one channel is refused with the command that converts it). The macOS `say` command makes speech only
//!   when that variable is not set, and only on a Mac. `docs/data/clip-sentence.wav` is a clip made with `say` that the
//!   real tests understand (it says `SENTENCE`); `docs/data/clip-bench.wav` is a 12 s one for the speed benches, which
//!   repeat a clip to 29 s (a repeated one-sentence clip makes Whisper Turbo stop after the first sentence).
//! * The models: `VOICE_ENGINE_TEST_MODELS`, a folder with one subfolder per model id, when it is set. Otherwise each
//!   model is looked for on its own, in the shared folder first (`SharedStore::default_dir`), then in Talaan's real
//!   folder for this platform (`LegacySource`: the Mac's `Application Support/Talaan/models`; Windows'
//!   `%LOCALAPPDATA%\com.craftconceptsdigital.talaan\models`, then `%APPDATA%\Talaan\models`).

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use voice_engine_models::catalog::find;
use voice_engine_models::{App, LegacySource, SharedStore};

pub const CLIP_ENV: &str = "VOICE_ENGINE_TEST_CLIP";
pub const MODELS_ENV: &str = "VOICE_ENGINE_TEST_MODELS";

/// What the real tests' `clip-sentence.wav` says (and what `VOICE_ENGINE_TEST_CLIP` must say for them).
pub const SENTENCE: &str = "Hello there. This is a short test of the speech engine.";

/// Set on a PC that has a card: the card tests must then run on it. A test that would skip, or accept the processor
/// with `fell_back`, because the card did not run fails instead (see the README, "Check the graphics card on a Windows
/// PC").
pub const EXPECT_CARD_ENV: &str = "VOICE_ENGINE_TEST_EXPECT_CARD";

/// Whether `VOICE_ENGINE_TEST_EXPECT_CARD=1` is set.
pub fn expect_card() -> bool {
    std::env::var(EXPECT_CARD_ENV).is_ok_and(|v| v == "1")
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

// ---- WAV ----------------------------------------------------------------------------------------------------

/// The samples of a 16 kHz mono WAV in memory: 16-bit PCM or 32-bit float (also inside WAVE_FORMAT_EXTENSIBLE).
/// Anything else is an Err that says what is wrong and how to convert it.
pub fn parse_wav(b: &[u8]) -> Result<Vec<f32>, String> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a WAV file (no RIFF/WAVE header)".into());
    }
    let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    // (format tag, bits per sample), once the "fmt " chunk has been read and accepted.
    let mut format: Option<(u16, u16)> = None;
    let mut i = 12;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32_at(i + 4) as usize;
        let body = i + 8;
        if id == b"fmt " {
            if len < 16 || body + 16 > b.len() {
                return Err("the fmt chunk is cut short".into());
            }
            let mut tag = u16_at(body);
            let channels = u16_at(body + 2);
            let rate = u32_at(body + 4);
            let bits = u16_at(body + 14);
            if tag == 0xFFFE && len >= 26 && body + 26 <= b.len() {
                tag = u16_at(body + 24); // the sub-format's first two bytes
            }
            if channels != 1 {
                return Err(format!(
                    "{channels} channels: the clip must be mono (ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav)"
                ));
            }
            if rate != 16_000 {
                return Err(format!(
                    "{rate} Hz: the clip must be 16 kHz (ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav)"
                ));
            }
            if !matches!((tag, bits), (1, 16) | (3, 32)) {
                return Err(format!(
                    "format tag {tag} with {bits} bits: only 16-bit PCM and 32-bit float are read \
                     (ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav)"
                ));
            }
            format = Some((tag, bits));
        } else if id == b"data" {
            let Some((tag, _)) = format else {
                return Err("the data chunk comes before the fmt chunk".into());
            };
            let end = body.saturating_add(len).min(b.len());
            let data = &b[body.min(end)..end];
            return Ok(if tag == 1 {
                data.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i16::from_le_bytes(*c) as f32 / 32768.0)
                    .collect()
            } else {
                data.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect()
            });
        }
        i = body.saturating_add(len).saturating_add(len & 1);
    }
    Err("no data chunk".into())
}

pub fn read_wav(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse_wav(&bytes).map_err(|e| format!("{} ({CLIP_ENV}): {e}", path.display()))
}

// ---- the clip -----------------------------------------------------------------------------------------------

/// The clip named by `VOICE_ENGINE_TEST_CLIP`, when it is set: Some(Err) when it cannot be read.
pub fn clip_from_env() -> Option<Result<Vec<f32>, String>> {
    env_path(CLIP_ENV).map(|p| read_wav(&p))
}

pub fn no_speech_message() -> String {
    format!(
        "no speech to test with: set {CLIP_ENV} to a 16 kHz mono WAV file (docs/data/clip-sentence.wav says \"{SENTENCE}\"); \
         the macOS `say` command is used only on a Mac when {CLIP_ENV} is not set"
    )
}

/// Can `speech` make a clip here: the variable is set, or this is a Mac.
pub fn speech_available() -> Result<(), String> {
    if env_path(CLIP_ENV).is_some() || cfg!(target_os = "macos") {
        Ok(())
    } else {
        Err(no_speech_message())
    }
}

/// Will `say` itself speak (so the wording and the voice asked for are the ones heard): a Mac, variable not set.
pub fn say_available() -> Result<(), String> {
    if env_path(CLIP_ENV).is_none() && cfg!(target_os = "macos") {
        Ok(())
    } else {
        Err(format!(
            "this test needs the macOS `say` command with its own wording or voices; it is skipped when {CLIP_ENV} is set \
             or the computer is not a Mac"
        ))
    }
}

/// 16 kHz mono samples of `text` spoken by the macOS `say` command (in `voice` when given). Mac only.
pub fn say_samples(voice: Option<&str>, text: &str) -> Result<Vec<f32>, String> {
    if !cfg!(target_os = "macos") {
        return Err(no_speech_message());
    }
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let aiff = dir.path().join("clip.aiff");
    let wav = dir.path().join("clip.wav");
    let mut cmd = Command::new("say");
    if let Some(voice) = voice {
        cmd.arg("-v").arg(voice);
    }
    let ok = cmd
        .arg("-o")
        .arg(&aiff)
        .arg(text)
        .status()
        .map_err(|e| format!("the macOS say command: {e}"))?;
    if !ok.success() {
        return Err("say failed".into());
    }
    let ok = Command::new("afconvert")
        .args(["-f", "WAVE", "-d", "LEF32@16000", "-c", "1"])
        .arg(&aiff)
        .arg(&wav)
        .status()
        .map_err(|e| format!("afconvert: {e}"))?;
    if !ok.success() {
        return Err("afconvert failed".into());
    }
    read_wav(&wav)
}

/// `secs` seconds of speech for a speed bench: the clip of `VOICE_ENGINE_TEST_CLIP` repeated (0.3 s apart) and cut to
/// the length, or, only when that is not set and on a Mac, `sentences` said by `say` in turn the same way.
pub fn bench_clip(secs: f64, sentences: &[&str]) -> Result<Vec<f32>, String> {
    let want = (secs * 16_000.0) as usize;
    let gap = std::iter::repeat_n(0.0f32, 4_800);
    let mut out: Vec<f32> = vec![];
    if let Some(clip) = clip_from_env() {
        let clip = clip?;
        if clip.is_empty() {
            return Err(format!("{CLIP_ENV}: the clip has no samples"));
        }
        while out.len() < want {
            out.extend_from_slice(&clip);
            out.extend(gap.clone());
        }
    } else if cfg!(target_os = "macos") {
        for sentence in sentences.iter().cycle() {
            if out.len() >= want {
                break;
            }
            out.extend(say_samples(None, sentence)?);
            out.extend(gap.clone());
        }
    } else {
        return Err(no_speech_message());
    }
    out.truncate(want);
    Ok(out)
}

// ---- the models ---------------------------------------------------------------------------------------------

/// Talaan's real model folders on this computer, in the order to look in them.
pub fn talaan_model_roots() -> Vec<PathBuf> {
    LegacySource::on_this_computer(App::Talaan)
        .into_iter()
        .map(|source| source.root)
        .collect()
}

/// The folder of model `id` (every catalog file present), as the tests and benches look for it: `models_override`
/// (`VOICE_ENGINE_TEST_MODELS`) when there is one, else the shared folder, then each of `talaan_roots`. Each model on
/// its own. Err says where it looked.
pub fn find_model_dir_in(
    id: &str,
    models_override: Option<&Path>,
    shared: Option<&Path>,
    talaan_roots: &[PathBuf],
) -> Result<PathBuf, String> {
    let model = find(id).ok_or_else(|| format!("{id} is not a catalog id"))?;
    let complete = |dir: &Path| model.files.iter().all(|f| dir.join(f.name).is_file());
    if let Some(root) = models_override {
        let dir = root.join(id);
        return if complete(&dir) {
            Ok(dir)
        } else {
            Err(format!(
                "{id} is not complete in {} ({MODELS_ENV} is set, so no other folder is looked in)",
                dir.display()
            ))
        };
    }
    let mut looked = vec![];
    if let Some(shared) = shared {
        looked.push(SharedStore::read_only(App::Talaan, shared.to_path_buf()).model_dir(id));
    }
    looked.extend(talaan_roots.iter().map(|root| root.join(id)));
    if let Some(dir) = looked.iter().find(|dir| complete(dir)) {
        return Ok(dir.clone());
    }
    Err(format!(
        "{id} was not found; looked in {}. Set {MODELS_ENV} to a folder that holds a folder named {id}",
        looked
            .iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// `find_model_dir_in` for this computer.
pub fn find_model_dir(id: &str) -> Result<PathBuf, String> {
    find_model_dir_in(
        id,
        env_path(MODELS_ENV).as_deref(),
        SharedStore::default_dir().as_deref(),
        &talaan_model_roots(),
    )
}
