//! A failed request: what kind of failure (for the app) and a message (for logs).

use voice_engine_protocol::{ErrorKind, Event};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Failure {
    pub error: ErrorKind,
    pub message: String,
}

impl Failure {
    pub fn new(error: ErrorKind, message: impl Into<String>) -> Self {
        Failure {
            error,
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Failure::new(ErrorKind::Internal, message)
    }

    pub fn missing(path: &str) -> Self {
        Failure::new(
            ErrorKind::ModelMissing { path: path.into() },
            format!("couldn't load the AI model: {path} is missing"),
        )
    }

    pub fn out_of_memory(message: impl Into<String>) -> Self {
        Failure::new(ErrorKind::OutOfMemory, message)
    }

    // Used once a reply can be stopped between tokens (the inbox).
    #[allow(dead_code)]
    pub fn cancelled() -> Self {
        Failure::new(ErrorKind::Cancelled, "stopped by request")
    }

    pub fn event(self) -> Event {
        Event::Error {
            message: self.message,
            error: self.error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failure_becomes_an_error_event_with_its_kind() {
        let ev = Failure::missing("/m.gguf").event();
        assert_eq!(
            ev,
            Event::Error {
                message: "couldn't load the AI model: /m.gguf is missing".into(),
                error: ErrorKind::ModelMissing {
                    path: "/m.gguf".into()
                },
            }
        );
        assert_eq!(Failure::cancelled().error, ErrorKind::Cancelled);
        assert_eq!(Failure::internal("x").error, ErrorKind::Internal);
    }
}
