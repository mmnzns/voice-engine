//! Scripted deaths, shared by the fake helpers (modes `die_noisy` and `sigkill`).

/// Writes the lines `FAKE_STDERR` holds (separated by `~`) to stderr, then a line of `FAKE_STDERR_LONG` letters when
/// that is set, and exits with `FAKE_EXIT` (default 1).
pub fn die_noisy() -> ! {
    if let Ok(text) = std::env::var("FAKE_STDERR") {
        for line in text.split('~') {
            eprintln!("{line}");
        }
    }
    if let Some(n) = std::env::var("FAKE_STDERR_LONG")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        eprintln!("{}", "x".repeat(n));
    }
    let code = std::env::var("FAKE_EXIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    std::process::exit(code)
}

/// Ends this process with SIGKILL, the way the system's out-of-memory killer does. Where there are no signals it
/// exits with code 137 instead.
pub fn sigkill() -> ! {
    #[cfg(unix)]
    {
        let _ = std::process::Command::new("kill")
            .arg("-KILL")
            .arg(std::process::id().to_string())
            .status();
        std::thread::sleep(std::time::Duration::from_secs(30));
    }
    std::process::exit(137)
}
