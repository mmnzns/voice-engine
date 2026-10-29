//! The real helper binary, without a model: it answers a ping, exits when the app closes its end (the app quit
//! or crashed) and exits on its own after a while without requests, so no model memory is left behind.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The tests take turns. macOS compiles Metal shaders through a system cache file; several helpers starting at once
/// can crash one of them (SIGBUS inside Apple's MTLCompilerFSCache, found by running these tests together). An app
/// starting a helper at the same moment as another app's must expect that too, so the client restarts a helper that
/// dies at startup (voice-engine-client, step 2).
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

fn spawn(idle_secs: Option<&str>) -> std::process::Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_voice-engine-llm"));
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if let Some(s) = idle_secs {
        cmd.env("VOICE_ENGINE_LLM_IDLE_SECS", s);
    }
    cmd.spawn().unwrap()
}

fn wait_exit(
    child: &mut std::process::Child,
    within: Duration,
) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + within;
    while Instant::now() < deadline {
        if let Some(s) = child.try_wait().unwrap() {
            return Some(s);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    None
}

#[test]
fn it_answers_a_ping_and_exits_when_stdin_closes() {
    let _turn = one_at_a_time();
    let mut child = spawn(None);
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    writeln!(stdin, r#"{{"op":"ping"}}"#).unwrap();
    let mut line = String::new();
    out.read_line(&mut line).unwrap();
    assert!(line.contains(r#""event":"ready""#), "{line}");
    drop(stdin);
    let status = wait_exit(&mut child, Duration::from_secs(20));
    assert!(
        status.is_some_and(|s| s.success()),
        "the helper must exit when the app is gone"
    );
    let _ = child.kill();
}

#[test]
fn it_exits_by_itself_when_idle() {
    let _turn = one_at_a_time();
    let mut child = spawn(Some("1"));
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    writeln!(stdin, r#"{{"op":"ping"}}"#).unwrap();
    let mut line = String::new();
    out.read_line(&mut line).unwrap();
    assert!(line.contains("ready"), "{line}");
    // stdin stays open: only the idle timer can end it.
    let status = wait_exit(&mut child, Duration::from_secs(20));
    assert!(
        status.is_some_and(|s| s.success()),
        "the helper must free its memory by exiting when idle"
    );
    drop(stdin);
}

#[test]
fn a_bad_line_is_an_error_event_not_a_crash() {
    let _turn = one_at_a_time();
    let mut child = spawn(None);
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    writeln!(stdin, "this is not json").unwrap();
    let mut line = String::new();
    out.read_line(&mut line).unwrap();
    assert!(
        line.contains(r#""event":"error""#) && line.contains("bad request"),
        "{line}"
    );
    writeln!(stdin, r#"{{"op":"ping"}}"#).unwrap();
    line.clear();
    out.read_line(&mut line).unwrap();
    assert!(
        line.contains("ready"),
        "still alive after a bad line: {line}"
    );
    drop(stdin);
    let _ = wait_exit(&mut child, Duration::from_secs(20));
    let _ = child.kill();
}

fn ask(stdin: &mut impl Write, out: &mut impl BufRead, request: &str) -> String {
    writeln!(stdin, "{request}").unwrap();
    let mut line = String::new();
    out.read_line(&mut line).unwrap();
    line
}

#[test]
fn configure_never_keeps_it_running_past_the_idle_time() {
    let _turn = one_at_a_time();
    let mut child = spawn(Some("1"));
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let line = ask(
        &mut stdin,
        &mut out,
        r#"{"op":"configure","idle_secs":null}"#,
    );
    assert!(line.contains(r#""event":"configured""#), "{line}");
    assert!(
        wait_exit(&mut child, Duration::from_secs(3)).is_none(),
        "configured to never exit, it must outlive the 1 s idle time"
    );
    let line = ask(&mut stdin, &mut out, r#"{"op":"ping"}"#);
    assert!(line.contains("ready"), "{line}");
    drop(stdin);
    assert!(wait_exit(&mut child, Duration::from_secs(20)).is_some_and(|s| s.success()));
}

#[test]
fn configure_can_change_while_the_helper_runs() {
    let _turn = one_at_a_time();
    // An app's "keep models ready" setting flips the policy live; each change must take effect, no restart.
    let mut child = spawn(None);
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let line = ask(
        &mut stdin,
        &mut out,
        r#"{"op":"configure","idle_secs":null}"#,
    );
    assert!(
        line.contains(r#""event":"configured""#) && line.contains(r#""idle_secs":null"#),
        "{line}"
    );
    let line = ask(&mut stdin, &mut out, r#"{"op":"configure","idle_secs":1}"#);
    assert!(line.contains(r#""idle_secs":1"#), "{line}");
    assert!(
        wait_exit(&mut child, Duration::from_secs(20)).is_some_and(|s| s.success()),
        "the second configure (1 s) must win over the first (never)"
    );
    drop(stdin);
}

#[test]
fn configure_a_short_idle_time_and_it_exits() {
    let _turn = one_at_a_time();
    let mut child = spawn(None);
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let line = ask(&mut stdin, &mut out, r#"{"op":"configure","idle_secs":1}"#);
    assert!(line.contains(r#""idle_secs":1"#), "{line}");
    assert!(wait_exit(&mut child, Duration::from_secs(20)).is_some_and(|s| s.success()));
    drop(stdin);
}

#[test]
fn a_cancel_with_nothing_running_is_ignored() {
    let _turn = one_at_a_time();
    let mut child = spawn(None);
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    writeln!(stdin, r#"{{"op":"cancel"}}"#).unwrap();
    let line = ask(&mut stdin, &mut out, r#"{"op":"ping"}"#);
    assert!(
        line.contains("ready"),
        "the cancel must produce no event: {line}"
    );
    drop(stdin);
    let _ = wait_exit(&mut child, Duration::from_secs(20));
    let _ = child.kill();
}

#[test]
fn load_and_complete_with_a_missing_file_say_model_missing() {
    let _turn = one_at_a_time();
    let mut child = spawn(None);
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    for request in [
        r#"{"op":"load","model_path":"/nonexistent/m.gguf","n_ctx":512,"gpu":false}"#,
        r#"{"op":"complete","model_path":"/nonexistent/m.gguf","n_ctx":512,"system":"","user":"","max_tokens":1,"temperature":0.0,"gpu":false}"#,
    ] {
        let line = ask(&mut stdin, &mut out, request);
        assert!(
            line.contains(r#""kind":"model_missing""#) && line.contains("/nonexistent/m.gguf"),
            "{line}"
        );
    }
    drop(stdin);
    let _ = wait_exit(&mut child, Duration::from_secs(20));
    let _ = child.kill();
}

#[test]
fn unload_with_nothing_loaded_answers_unloaded_and_the_helper_keeps_going() {
    let _turn = one_at_a_time();
    let mut child = spawn(None);
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let line = ask(&mut stdin, &mut out, r#"{"op":"unload"}"#);
    assert_eq!(line.trim(), r#"{"event":"unloaded"}"#);
    let line = ask(&mut stdin, &mut out, r#"{"op":"ping"}"#);
    assert!(line.contains("ready"), "{line}");
    drop(stdin);
    let _ = wait_exit(&mut child, Duration::from_secs(20));
    let _ = child.kill();
}

#[test]
fn never_exit_still_exits_when_the_app_is_gone() {
    let _turn = one_at_a_time();
    // "Never exit" means never exit while the app runs: closing stdin (the app quit or crashed) must still end it,
    // so an update or a crash never leaves a helper behind.
    let mut child = spawn(None);
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let line = ask(
        &mut stdin,
        &mut out,
        r#"{"op":"configure","idle_secs":null}"#,
    );
    assert!(line.contains("configured"), "{line}");
    drop(stdin);
    assert!(
        wait_exit(&mut child, Duration::from_secs(20)).is_some_and(|s| s.success()),
        "the helper must exit when its stdin closes, even configured to never idle out"
    );
}
