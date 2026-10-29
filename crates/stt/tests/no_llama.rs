//! This crate never links llama.cpp. whisper.cpp and llama.cpp each carry their own copy of ggml, so a program that
//! linked both would get duplicate symbols; the speech engine has to be usable inside an app's own process next to
//! nothing from the local AI. This walks the resolved dependency graph (`cargo metadata`, every platform, every kind
//! of dependency, the build and test ones included) from `voice-engine-stt` and fails if `llama-cpp-2`,
//! `llama-cpp-sys-2` or `voice-engine-llm` is anywhere in it. No network and no model: cargo reads the lock file and
//! the crates it already has. (If a crate for another platform was never downloaded, cargo is allowed to fetch its
//! manifest, which is a few kilobytes and happens once.)

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::process::Command;

const FORBIDDEN: &[&str] = &["llama-cpp-2", "llama-cpp-sys-2", "voice-engine-llm"];
const CRATE: &str = "voice-engine-stt";

fn metadata() -> Value {
    let run = |offline: bool| {
        let mut cmd = Command::new(env!("CARGO"));
        cmd.args(["metadata", "--format-version", "1", "--locked"])
            .current_dir(env!("CARGO_MANIFEST_DIR"));
        if offline {
            cmd.arg("--offline");
        }
        cmd.output().expect("could not run cargo metadata")
    };
    let mut out = run(true);
    if !out.status.success() {
        out = run(false);
    }
    assert!(
        out.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("cargo metadata did not print JSON")
}

/// Package ids by name, and each package id's dependencies, from the resolved graph.
fn graph(meta: &Value) -> (HashMap<String, String>, HashMap<String, Vec<String>>) {
    let names: HashMap<String, String> = meta["packages"]
        .as_array()
        .expect("packages")
        .iter()
        .map(|p| {
            (
                p["id"].as_str().expect("id").to_string(),
                p["name"].as_str().expect("name").to_string(),
            )
        })
        .collect();
    let edges = meta["resolve"]["nodes"]
        .as_array()
        .expect("resolve.nodes")
        .iter()
        .map(|n| {
            let deps = n["deps"]
                .as_array()
                .expect("deps")
                .iter()
                .map(|d| d["pkg"].as_str().expect("pkg").to_string())
                .collect();
            (n["id"].as_str().expect("id").to_string(), deps)
        })
        .collect();
    (names, edges)
}

/// The names of every package reachable from `root_name` (the root itself included), and the path to each.
fn reachable(meta: &Value, root_name: &str) -> HashMap<String, Vec<String>> {
    let (names, edges) = graph(meta);
    let root = names
        .iter()
        .find(|(_, n)| n.as_str() == root_name)
        .map(|(id, _)| id.clone())
        .unwrap_or_else(|| panic!("{root_name} is not in the workspace"));
    let mut found: HashMap<String, Vec<String>> = HashMap::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut stack = vec![(root.clone(), vec![root_name.to_string()])];
    seen.insert(root);
    while let Some((id, path)) = stack.pop() {
        found.insert(names[&id].clone(), path.clone());
        for dep in edges.get(&id).into_iter().flatten() {
            if seen.insert(dep.clone()) {
                let mut next = path.clone();
                next.push(names[dep].clone());
                stack.push((dep.clone(), next));
            }
        }
    }
    found
}

#[test]
fn stt_has_no_path_to_llama_cpp_or_the_llm_crate() {
    let meta = metadata();
    let reach = reachable(&meta, CRATE);
    // The walk really went through the graph: whisper-rs and the shared crates are in it.
    for expected in ["whisper-rs", "parakeet-rs", "voice-engine-protocol"] {
        assert!(
            reach.contains_key(expected),
            "the walk did not reach {expected}: the check proves nothing"
        );
    }
    for name in FORBIDDEN {
        if let Some(path) = reach.get(*name) {
            panic!(
                "{CRATE} reaches {name} through {}: no llama.cpp may be linked next to whisper.cpp",
                path.join(" -> ")
            );
        }
    }
}

/// The walk finds a forbidden crate when there is one: from the local AI helper's crate the same names are reachable.
#[test]
fn the_walk_finds_llama_where_it_exists() {
    let meta = metadata();
    let reach = reachable(&meta, "voice-engine-llm");
    assert!(reach.contains_key("llama-cpp-2"));
    assert!(reach.contains_key("llama-cpp-sys-2"));
}
