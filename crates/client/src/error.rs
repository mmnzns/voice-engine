//! Every way a request to the helper can fail, as one type the app matches on (never on message text).

use voice_engine_protocol::ErrorKind;

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
    /// Stopped by the app (`LlmClient::cancel`).
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
    /// The helper process died and trying again did not help.
    Crashed,
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

impl std::fmt::Display for EngineError {
    /// For logs only. Apps show their own wording and match on the variant.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
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
}
