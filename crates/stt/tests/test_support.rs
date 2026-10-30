//! The WAV reader and the model lookup the speed benches and the real-model tests share (`support/mod.rs`): pure
//! tests on bytes and on temporary folders, no model and no sound.

#[path = "../support/mod.rs"]
mod support;

use std::fs;
use std::path::{Path, PathBuf};
use support::{find_model_dir_in, parse_wav};
use voice_engine_models::catalog::find;

/// A WAV file in memory: `tag` is the format tag, `data` the sample bytes.
fn wav(tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
    let mut fmt = vec![];
    fmt.extend(tag.to_le_bytes());
    fmt.extend(channels.to_le_bytes());
    fmt.extend(rate.to_le_bytes());
    fmt.extend((rate * u32::from(channels) * u32::from(bits / 8)).to_le_bytes());
    fmt.extend((channels * (bits / 8)).to_le_bytes());
    fmt.extend(bits.to_le_bytes());
    riff(&[(b"fmt ", &fmt), (b"data", data)])
}

fn riff(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let mut body = b"WAVE".to_vec();
    for (id, data) in chunks {
        body.extend(*id);
        body.extend((data.len() as u32).to_le_bytes());
        body.extend(*data);
        if data.len() % 2 == 1 {
            body.push(0);
        }
    }
    let mut out = b"RIFF".to_vec();
    out.extend((body.len() as u32).to_le_bytes());
    out.extend(body);
    out
}

fn pcm16(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn float32(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

#[test]
fn reads_16_bit_pcm() {
    let b = wav(1, 1, 16_000, 16, &pcm16(&[0, 16_384, -32_768, 32_767]));
    let s = parse_wav(&b).unwrap();
    assert_eq!(&s[..3], &[0.0, 0.5, -1.0]);
    assert!((s[3] - 0.99997).abs() < 1e-4);
}

#[test]
fn reads_32_bit_float() {
    let b = wav(3, 1, 16_000, 32, &float32(&[0.25, -0.5, 1.0]));
    assert_eq!(parse_wav(&b).unwrap(), vec![0.25, -0.5, 1.0]);
}

#[test]
fn reads_float_inside_wave_format_extensible() {
    // fmt chunk of 40 bytes: the usual 16, cbSize 22, valid bits, channel mask, then the sub-format GUID (tag 3 first).
    let mut fmt = vec![];
    fmt.extend(0xFFFEu16.to_le_bytes());
    fmt.extend(1u16.to_le_bytes());
    fmt.extend(16_000u32.to_le_bytes());
    fmt.extend((16_000u32 * 4).to_le_bytes());
    fmt.extend(4u16.to_le_bytes());
    fmt.extend(32u16.to_le_bytes());
    fmt.extend(22u16.to_le_bytes());
    fmt.extend(32u16.to_le_bytes());
    fmt.extend(4u32.to_le_bytes());
    fmt.extend(3u16.to_le_bytes());
    fmt.extend([0u8; 14]);
    assert_eq!(fmt.len(), 40);
    let b = riff(&[(b"fmt ", &fmt), (b"data", &float32(&[0.5, -0.25]))]);
    assert_eq!(parse_wav(&b).unwrap(), vec![0.5, -0.25]);
}

#[test]
fn skips_chunks_it_does_not_know_including_odd_sized_ones() {
    let mut fmt = vec![];
    fmt.extend(1u16.to_le_bytes());
    fmt.extend(1u16.to_le_bytes());
    fmt.extend(16_000u32.to_le_bytes());
    fmt.extend(32_000u32.to_le_bytes());
    fmt.extend(2u16.to_le_bytes());
    fmt.extend(16u16.to_le_bytes());
    let b = riff(&[
        (b"fmt ", &fmt),
        (b"LIST", b"abc"),
        (b"data", &pcm16(&[8_192])),
    ]);
    assert_eq!(parse_wav(&b).unwrap(), vec![0.25]);
}

#[test]
fn a_data_chunk_longer_than_the_file_reads_what_is_there() {
    let mut b = wav(1, 1, 16_000, 16, &pcm16(&[16_384, 16_384]));
    let at = b.windows(4).position(|w| w == b"data").unwrap();
    b[at + 4..at + 8].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(parse_wav(&b).unwrap(), vec![0.5, 0.5]);
}

#[test]
fn another_sample_rate_is_refused_with_how_to_convert_it() {
    let e = parse_wav(&wav(1, 1, 44_100, 16, &pcm16(&[0]))).unwrap_err();
    assert!(
        e.contains("44100 Hz") && e.contains("16 kHz") && e.contains("ffmpeg"),
        "{e}"
    );
}

#[test]
fn stereo_and_other_depths_are_refused() {
    let e = parse_wav(&wav(1, 2, 16_000, 16, &pcm16(&[0, 0]))).unwrap_err();
    assert!(e.contains("2 channels") && e.contains("mono"), "{e}");
    let e = parse_wav(&wav(1, 1, 16_000, 24, &[0, 0, 0])).unwrap_err();
    assert!(e.contains("24 bits"), "{e}");
}

#[test]
fn not_a_wav_and_no_data_are_refused() {
    assert!(parse_wav(b"hello there, this is not a wav file").is_err());
    assert!(parse_wav(&[]).is_err());
    let e = parse_wav(&riff(&[(b"LIST", b"xx")])).unwrap_err();
    assert!(e.contains("no data"), "{e}");
}

// ---- the model lookup -----------------------------------------------------------------------------------------

/// Every catalog file of `id` (empty files), under `root/<id>`.
fn put(root: &Path, id: &str) -> PathBuf {
    let dir = root.join(id);
    fs::create_dir_all(&dir).unwrap();
    for f in find(id).unwrap().files {
        fs::write(dir.join(f.name), b"").unwrap();
    }
    dir
}

#[test]
fn the_shared_folder_comes_before_talaans_and_the_roots_come_in_order() {
    let shared = tempfile::tempdir().unwrap();
    let t1 = tempfile::tempdir().unwrap();
    let t2 = tempfile::tempdir().unwrap();
    let roots = [t1.path().to_path_buf(), t2.path().to_path_buf()];
    let in_t2 = put(t2.path(), "whisper-turbo");
    assert_eq!(
        find_model_dir_in("whisper-turbo", None, Some(shared.path()), &roots).unwrap(),
        in_t2,
        "only the second Talaan folder has it"
    );
    let in_t1 = put(t1.path(), "whisper-turbo");
    assert_eq!(
        find_model_dir_in("whisper-turbo", None, Some(shared.path()), &roots).unwrap(),
        in_t1
    );
    let in_shared = put(shared.path(), "whisper-turbo");
    assert_eq!(
        find_model_dir_in("whisper-turbo", None, Some(shared.path()), &roots).unwrap(),
        in_shared
    );
}

#[test]
fn each_model_is_found_on_its_own() {
    let shared = tempfile::tempdir().unwrap();
    let talaan = tempfile::tempdir().unwrap();
    let roots = [talaan.path().to_path_buf()];
    let base = put(shared.path(), "whisper-base");
    let ultra = put(talaan.path(), "parakeet-ultra");
    assert_eq!(
        find_model_dir_in("whisper-base", None, Some(shared.path()), &roots).unwrap(),
        base
    );
    assert_eq!(
        find_model_dir_in("parakeet-ultra", None, Some(shared.path()), &roots).unwrap(),
        ultra
    );
    assert!(find_model_dir_in("whisper-turbo", None, Some(shared.path()), &roots).is_err());
}

#[test]
fn a_model_with_a_file_missing_is_not_found() {
    let shared = tempfile::tempdir().unwrap();
    let dir = put(shared.path(), "parakeet-ultra");
    let first = find("parakeet-ultra").unwrap().files[0].name;
    fs::remove_file(dir.join(first)).unwrap();
    assert!(find_model_dir_in("parakeet-ultra", None, Some(shared.path()), &[]).is_err());
}

#[test]
fn the_override_is_the_only_place_looked_in() {
    let shared = tempfile::tempdir().unwrap();
    let over = tempfile::tempdir().unwrap();
    put(shared.path(), "whisper-base");
    let e =
        find_model_dir_in("whisper-base", Some(over.path()), Some(shared.path()), &[]).unwrap_err();
    assert!(e.contains("VOICE_ENGINE_TEST_MODELS"), "{e}");
    let in_over = put(over.path(), "whisper-base");
    assert_eq!(
        find_model_dir_in("whisper-base", Some(over.path()), Some(shared.path()), &[]).unwrap(),
        in_over
    );
}

#[test]
fn the_error_says_where_it_looked_and_what_to_set() {
    let shared = tempfile::tempdir().unwrap();
    let talaan = tempfile::tempdir().unwrap();
    let e = find_model_dir_in(
        "whisper-turbo",
        None,
        Some(shared.path()),
        &[talaan.path().to_path_buf()],
    )
    .unwrap_err();
    assert!(
        e.contains("whisper-turbo") && e.contains("VOICE_ENGINE_TEST_MODELS"),
        "{e}"
    );
    assert!(e.contains(&shared.path().display().to_string()), "{e}");
    assert!(e.contains(&talaan.path().display().to_string()), "{e}");
    assert!(find_model_dir_in("no-such-model", None, None, &[]).is_err());
}
