//! Prints the local AI servers `servers::detect()` finds on this computer: counts and model ids only.
//! Run it with `cargo run -p voice-engine-client --example servers`.

use voice_engine_client::servers::{ServerKind, ServerState, detect};

fn main() {
    let servers = detect();
    if servers.is_empty() {
        println!("no local AI servers found (Ollama on 11434, LM Studio on 1234)");
        return;
    }
    for server in servers {
        let name = match server.kind {
            ServerKind::Ollama => "Ollama",
            ServerKind::LmStudio => "LM Studio",
        };
        match server.state {
            ServerState::Running => println!(
                "{name} at {}: running, {} model(s), chat requests go to {}",
                server.base_url,
                server.models.len(),
                server.api_base
            ),
            ServerState::Unreadable => println!(
                "{name} at {}: something answers, but its model list could not be read",
                server.base_url
            ),
        }
        for model in &server.models {
            println!(
                "  {}  context {}",
                model.id,
                model
                    .context_length
                    .map_or("unknown".to_string(), |n| n.to_string())
            );
        }
    }
}
