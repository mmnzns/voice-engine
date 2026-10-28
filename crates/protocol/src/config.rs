//! What differs between the apps' copies of a helper: only the names of its environment variables.

/// Passed by each app's helper `main.rs`, for example `HelperConfig { env_prefix: "TALAAN" }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HelperConfig {
    /// Upper-case app name; environment variables are `<prefix>_GPU` and `<prefix>_LLM_IDLE_SECS`.
    pub env_prefix: &'static str,
}

impl HelperConfig {
    /// `<prefix>_GPU=fail` makes a GPU build die on its first real request, to test the fallback on a real PC.
    pub fn gpu_env(&self) -> String {
        format!("{}_GPU", self.env_prefix)
    }

    /// `<prefix>_LLM_IDLE_SECS` shortens the AI helper's idle exit (tests only).
    pub fn idle_secs_env(&self) -> String {
        format!("{}_LLM_IDLE_SECS", self.env_prefix)
    }
}
