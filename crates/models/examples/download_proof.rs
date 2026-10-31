//! The downloader against the real Hugging Face. It adds `whisper-base` and `whisper-tiny` (and the folder's
//! `.engine`, with a `talaan` user marker for each) to the shared model folder, so it refuses to run unless you say
//! which folder: set `VOICE_ENGINE_MODELS_DIR` to a folder of your own, or pass `--real` on purpose to write into the
//! real shared model folder. It prints the folder before it writes. Prints sizes, times, fractions and statuses only.
//!
//! ```text
//! VOICE_ENGINE_MODELS_DIR=/some/folder cargo run --release -p voice-engine-models --example download_proof
//! cargo run --release -p voice-engine-models --example download_proof -- --real
//! ```
//!
//! 1. `whisper-base` (the whole file, from nothing): progress every 10%, speed, then `status`.
//! 2. `whisper-tiny`: cancelled at about half, `status` (`Partial` with `cancelled: true`), then downloaded again
//!    (it resumes from the `.part` file), then `status`.
//!
//! Afterwards compare `shasum -a 256` of the files with the catalog hashes (printed here).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use voice_engine_models::catalog::find;
use voice_engine_models::{App, DownloadError, Progress, SharedStore, StoreError};

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1_000_000.0
}

/// Downloads `id` and prints every 10%. With `cancel_at`, sets the cancel flag once the fraction passes it.
fn run(store: &SharedStore, id: &str, cancel_at: Option<f32>) -> Result<(), StoreError> {
    let cancel = AtomicBool::new(false);
    let started = Instant::now();
    let mut first: Option<u64> = None;
    let mut last = 0u64;
    let mut next_tenth = 1u32;
    let result = store.download(id, &cancel, &mut |p: Progress| {
        if first.is_none() {
            first = Some(p.bytes);
            println!(
                "  {id}: starting with {:.1} MB of {:.1} MB already on disk ({:.0}%)",
                mb(p.bytes),
                mb(p.total),
                p.fraction * 100.0
            );
            while (next_tenth as f32) / 10.0 <= p.fraction {
                next_tenth += 1;
            }
        }
        last = p.bytes;
        while next_tenth <= 10 && p.fraction >= next_tenth as f32 / 10.0 {
            println!(
                "  {id}: {:>3}%  {:.1} MB  after {:.1} s",
                next_tenth * 10,
                mb(p.bytes),
                started.elapsed().as_secs_f64()
            );
            next_tenth += 1;
        }
        if let Some(at) = cancel_at
            && p.fraction >= at
            && !cancel.swap(true, Ordering::Relaxed)
        {
            println!(
                "  {id}: cancel asked at {:.1}% ({:.1} MB)",
                p.fraction * 100.0,
                mb(p.bytes)
            );
        }
    });
    let secs = started.elapsed().as_secs_f64();
    let moved = last.saturating_sub(first.unwrap_or(0));
    println!(
        "  {id}: returned after {secs:.1} s, {:.1} MB moved this call, {:.2} MB/s",
        mb(moved),
        mb(moved) / secs.max(1e-9)
    );
    match result {
        Ok(dir) => {
            println!("  {id}: Ok, folder {}", dir.display());
            Ok(())
        }
        Err(e) => {
            println!("  {id}: Err({e:?})");
            Err(e)
        }
    }
}

fn main() {
    let real = std::env::args().any(|a| a == "--real");
    let from_env = std::env::var_os("VOICE_ENGINE_MODELS_DIR").is_some_and(|d| !d.is_empty());
    if !real && !from_env {
        eprintln!(
            "refusing to run: this downloads whisper-base and whisper-tiny and adds a Talaan user marker for each.\n\
             Set VOICE_ENGINE_MODELS_DIR to a folder of your own, or pass --real to write into the real shared folder."
        );
        std::process::exit(2);
    }
    let store = SharedStore::open(App::Talaan).expect("open the shared folder");
    println!(
        "writing into {} ({}): whisper-base and whisper-tiny, with a Talaan user marker for each",
        store.root().display(),
        if from_env {
            "VOICE_ENGINE_MODELS_DIR"
        } else {
            "the real shared folder, --real"
        }
    );
    for id in ["whisper-base", "whisper-tiny"] {
        let model = find(id).unwrap();
        println!(
            "catalog {id}: file {} size {} sha256 {}",
            model.files[0].name, model.files[0].size, model.files[0].sha256
        );
        println!("status before: {:?}", store.status(id));
    }

    println!("\n1. whisper-base, whole download");
    run(&store, "whisper-base", None).expect("whisper-base download");
    println!("status after: {:?}", store.status("whisper-base"));

    println!("\n2. whisper-tiny, cancelled at about half");
    match run(&store, "whisper-tiny", Some(0.5)) {
        Err(StoreError::Download(DownloadError::Cancelled)) => {}
        other => println!("  (not a cancel: {other:?})"),
    }
    println!("status after cancel: {:?}", store.status("whisper-tiny"));

    println!("\n3. whisper-tiny, resumed");
    run(&store, "whisper-tiny", None).expect("whisper-tiny resume");
    println!("status after resume: {:?}", store.status("whisper-tiny"));
}
