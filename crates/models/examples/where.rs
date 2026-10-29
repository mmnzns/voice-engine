//! Prints, without creating or changing anything: the shared model folder on this computer and whether it exists,
//! each catalog model's status for each of the three apps, and which of each app's old folders exist.
//! `cargo run -p voice-engine-models --example where`

use voice_engine_models::{App, LegacySource, SharedStore, Status};

fn main() {
    let Some(dir) = SharedStore::default_dir() else {
        println!("no local app data folder on this computer");
        return;
    };
    println!("Shared model folder: {}", dir.display());
    println!("  exists={}", dir.is_dir());

    // `read_only` creates nothing, not even the folder or `.engine`.
    let views: Vec<(App, SharedStore)> = App::ALL
        .into_iter()
        .map(|app| (app, SharedStore::read_only(app, dir.clone())))
        .collect();
    println!();
    println!("Models (status for talaan / ensayo / parlato, as each app would see it):");
    for model in voice_engine_models::catalog::all() {
        let shown: Vec<String> = views
            .iter()
            .map(|(_, view)| describe(&view.status(model.id)))
            .collect();
        println!("  {:<28} {}", model.id, shown.join(" / "));
    }

    println!();
    println!("Old folders (before the shared folder):");
    for app in App::ALL {
        for source in LegacySource::on_this_computer(app) {
            println!(
                "  {app:?}: {} exists={}",
                source.root.display(),
                source.root.is_dir()
            );
            for entry in &source.entries {
                // The folder of the model, or the first file when the entry names files (Parlato's single files).
                let mut path = source.root.join(entry.rel);
                if let Some((on_disk, _)) = entry.files.first() {
                    path.push(on_disk);
                }
                if path.exists() {
                    println!("    {} -> {}", entry.id, path.display());
                }
            }
        }
    }
}

fn describe(status: &Status) -> String {
    match status {
        Status::Installed {
            users, used_by_me, ..
        } => format!("installed(users={},mine={used_by_me})", users.len()),
        Status::Downloading { fraction, .. } => format!("downloading({:.0}%)", fraction * 100.0),
        Status::Partial {
            fraction,
            cancelled,
            ..
        } => format!("partial({:.0}%,cancelled={cancelled})", fraction * 100.0),
        Status::Missing => "missing".to_string(),
    }
}
