//! The real helper binary, without a model: it answers a ping, exits when the app closes its end (the app quit
//! or crashed) and exits on its own after a while without requests, so no model memory is left behind.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

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
