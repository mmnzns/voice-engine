//! Every way a request to a helper can fail, as one type the app matches on (never on message text). The AI helper
//! and the speech helper share it: a variant that one of them never produces (`PromptTooLong` for speech, `TooLong`
//! for the AI) is simply never returned by that client.

use crate::process::Death;
use voice_engine_protocol::ErrorKind;
use voice_engine_protocol::stt::SttErrorKind;

#[derive(Debug, Clone, PartialEq)]
pub enum EngineError {
    /// The model file isn't where it should be.
    ModelMissing {
        path: String,
    },
    /// Not enough memory for the model (the helper said so, or it died with an out-of-memory message).
    OutOfMemory,
    PromptTooLong {
        prompt_tokens: u32,
        n_ctx: u32,
        max_tokens: u32,
    },
    BadSchema {
        detail: String,
    },
    NoChatTemplate,
    /// The speech model's file is there but cannot be read as that model.
    BadModel,
    /// The audio is longer than the model takes in one call; cut it and call again.
    TooLong {
        max_secs: u32,
    },
    /// The helper cannot run that (a speech model the helper does not serve, or a file that is not a speech model).
    Unsupported {
        detail: String,
    },
    /// Stopped by the app (`LlmClient::cancel`, `SttClient::cancel`).
    Cancelled,
    /// The graphics card failed and the processor could not finish the work either, or the helper was a processor
    /// helper that reported it.
    GpuFailed {
        fell_back: bool,
    },
    BadRequest,
    Internal {
        message: String,
    },
    /// The helper process died and trying again did not help. What is known of how it died, for the app's log and
    /// its bug report; the app matches on the variant.
    ///
    /// `detail` is a short tail of the helper's stderr: at most 5 lines of at most 200 characters, empty when it wrote
    /// nothing. It holds no user text: the helpers write nothing a person said or typed to stderr (llama.cpp's and
    /// whisper.cpp's own logs are switched off, and what is left is the helper's one `GPU=fail` note and the
    /// library's abort and assert lines, which name a file, a line and a check, never a prompt or a transcript).
    Crashed {
        /// The code it exited with, when it exited by itself with one (none after a signal).
        exit_code: Option<i32>,
        /// The signal that ended it (Unix only).
        signal: Option<i32>,
        detail: String,
    },
    /// The helper and the app were built against different protocol versions.
    VersionMismatch {
        helper: u32,
        app: u32,
    },
    /// No answer within the limits (the helper was stopped).
    TimedOut,
    /// The helper could not be started at all.
    CouldNotStart {
        message: String,
    },
}

impl EngineError {
    /// How a helper that died ends the request: `OutOfMemory` when it looks like lack of memory, else `Crashed`.
    pub(crate) fn died(death: Death) -> Self {
        if death.out_of_memory() {
            return EngineError::OutOfMemory;
        }
        EngineError::Crashed {
            exit_code: death.exit_code,
            signal: death.signal,
            detail: death.detail,
        }
    }

    /// The error a helper reported, as the app sees it.
    pub fn from_helper(error: ErrorKind, message: String) -> Self {
        match error {
            ErrorKind::ModelMissing { path } => EngineError::ModelMissing { path },
            ErrorKind::OutOfMemory => EngineError::OutOfMemory,
            ErrorKind::PromptTooLong {
                prompt_tokens,
                n_ctx,
                max_tokens,
            } => EngineError::PromptTooLong {
                prompt_tokens,
                n_ctx,
                max_tokens,
            },
            ErrorKind::BadSchema { detail } => EngineError::BadSchema { detail },
            ErrorKind::NoChatTemplate => EngineError::NoChatTemplate,
            ErrorKind::Cancelled => EngineError::Cancelled,
            ErrorKind::GpuFailed { fell_back } => EngineError::GpuFailed { fell_back },
            ErrorKind::BadRequest => EngineError::BadRequest,
            ErrorKind::Internal => EngineError::Internal { message },
        }
    }
}

impl EngineError {
    /// The error the speech helper reported, as the app sees it.
    pub fn from_stt_helper(error: SttErrorKind, message: String) -> Self {
        match error {
            SttErrorKind::ModelMissing { path } => EngineError::ModelMissing { path },
            SttErrorKind::BadModel => EngineError::BadModel,
            SttErrorKind::OutOfMemory => EngineError::OutOfMemory,
            SttErrorKind::TooLong { max_secs } => EngineError::TooLong { max_secs },
            SttErrorKind::Unsupported { detail } => EngineError::Unsupported { detail },
            SttErrorKind::GpuFailed { fell_back } => EngineError::GpuFailed { fell_back },
            SttErrorKind::Cancelled => EngineError::Cancelled,
            SttErrorKind::BadRequest => EngineError::BadRequest,
            SttErrorKind::Internal => EngineError::Internal { message },
        }
    }
}

impl std::fmt::Display for EngineError {
    /// For logs only. Apps show their own wording and match on the variant.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Crashed {
                exit_code,
                signal,
                detail,
            } => {
                write!(f, "the helper crashed")?;
                match (exit_code, signal) {
                    (Some(code), _) => write!(f, " (exit code {code})")?,
                    (None, Some(signal)) => write!(f, " (signal {signal})")?,
                    (None, None) => {}
                }
                if !detail.is_empty() {
                    write!(f, ": {}", detail.replace('\n', " | "))?;
                }
                Ok(())
            }
            other => write!(f, "{other:?}"),
        }
    }
}

impl std::error::Error for EngineError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_helper_kind_maps_to_its_own_error() {
        let cases = [
            (
                ErrorKind::ModelMissing { path: "/m".into() },
                EngineError::ModelMissing { path: "/m".into() },
            ),
            (ErrorKind::OutOfMemory, EngineError::OutOfMemory),
            (ErrorKind::NoChatTemplate, EngineError::NoChatTemplate),
            (ErrorKind::Cancelled, EngineError::Cancelled),
            (
                ErrorKind::GpuFailed { fell_back: true },
                EngineError::GpuFailed { fell_back: true },
            ),
            (ErrorKind::BadRequest, EngineError::BadRequest),
            (
                ErrorKind::Internal,
                EngineError::Internal {
                    message: "boom".into(),
                },
            ),
        ];
        for (kind, want) in cases {
            assert_eq!(EngineError::from_helper(kind, "boom".into()), want);
        }
        assert_eq!(
            EngineError::from_helper(
                ErrorKind::PromptTooLong {
                    prompt_tokens: 9,
                    n_ctx: 8,
                    max_tokens: 1
                },
                String::new()
            ),
            EngineError::PromptTooLong {
                prompt_tokens: 9,
                n_ctx: 8,
                max_tokens: 1
            }
        );
        assert_eq!(
            EngineError::from_helper(ErrorKind::BadSchema { detail: "d".into() }, String::new()),
            EngineError::BadSchema { detail: "d".into() }
        );
    }

    #[test]
    fn a_crash_says_how_the_helper_died_in_the_log_line() {
        let died = |exit_code, signal, detail: &str| {
            EngineError::Crashed {
                exit_code,
                signal,
                detail: detail.into(),
            }
            .to_string()
        };
        assert_eq!(died(None, None, ""), "the helper crashed");
        assert_eq!(
            died(Some(139), None, "one\ntwo"),
            "the helper crashed (exit code 139): one | two"
        );
        assert_eq!(died(None, Some(11), ""), "the helper crashed (signal 11)");
        // The other variants keep showing as themselves.
        assert_eq!(EngineError::TimedOut.to_string(), "TimedOut");
    }

    #[test]
    fn every_speech_helper_kind_maps_to_its_own_error() {
        let cases = [
            (
                SttErrorKind::ModelMissing { path: "/m".into() },
                EngineError::ModelMissing { path: "/m".into() },
            ),
            (SttErrorKind::BadModel, EngineError::BadModel),
            (SttErrorKind::OutOfMemory, EngineError::OutOfMemory),
            (
                SttErrorKind::TooLong { max_secs: 40 },
                EngineError::TooLong { max_secs: 40 },
            ),
            (
                SttErrorKind::Unsupported {
                    detail: "whisper only".into(),
                },
                EngineError::Unsupported {
                    detail: "whisper only".into(),
                },
            ),
            (
                SttErrorKind::GpuFailed { fell_back: false },
                EngineError::GpuFailed { fell_back: false },
            ),
            (SttErrorKind::Cancelled, EngineError::Cancelled),
            (SttErrorKind::BadRequest, EngineError::BadRequest),
            (
                SttErrorKind::Internal,
                EngineError::Internal {
                    message: "boom".into(),
                },
            ),
        ];
        for (kind, want) in cases {
            assert_eq!(EngineError::from_stt_helper(kind, "boom".into()), want);
        }
    }
}
