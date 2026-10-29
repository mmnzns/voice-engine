//! Which local AI servers are running on this computer, and which models each one offers.
//!
//! [`detect`] looks for Ollama (port 11434) and LM Studio (port 1234); [`detect_at`] takes the addresses to look at.
//! Both are OpenAI-compatible, and the apps keep making their chat requests to them themselves: this module only
//! finds the servers and lists their models, with each model's name and context size.
//!
//! - Ollama: `GET /api/tags` lists the models, then `POST /api/show` with `{"model": name}` gives each model's
//!   context length (`model_info["<architecture>.context_length"]`).
//! - LM Studio: `GET /api/v0/models` lists the models with `max_context_length` and `type`; only `llm` and `vlm`
//!   models are kept (the `embeddings` ones cannot chat).
//!
//! It is meant for a settings screen, so it is quick: connecting gives up after 300 ms, and one server gets 2 s in
//! all (its list and every context lookup together). The servers are asked at the same time, there are no retries,
//! nothing stays running afterwards, and nothing is logged. A person's model names are theirs, so no error carries
//! one: the only outcomes are "not there" (left out of the list), [`ServerState::Running`] and
//! [`ServerState::Unreadable`].
//!
//! The functions block. Call them from a worker thread, not from inside an async task.

use std::io::Read;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;

/// How long a connection may take to open before the server counts as not there.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(300);
/// How long one server gets in all: its model list and every context lookup together.
const TOTAL_TIMEOUT: Duration = Duration::from_secs(2);
/// More than this from a model listing is not a model listing.
const MAX_BODY_BYTES: u64 = 8 * 1024 * 1024;

/// Which kind of server it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerKind {
    Ollama,
    LmStudio,
}

/// What we could tell about a server that answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerState {
    /// It answered and its model list was read.
    Running,
    /// Something is listening there, but it did not answer with what we expect (an error status, something that is
    /// not JSON, a list of another shape, or no answer within the time allowed), so the list could not be read.
    Unreadable,
}

/// One model a server offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerModel {
    /// What the app puts in the request's `"model"` field.
    pub id: String,
    /// The model's context limit, `None` when the server does not say.
    pub context_length: Option<u32>,
    /// The model's size on disk, `None` when the server does not say (LM Studio's listing has no size).
    pub size_bytes: Option<u64>,
}

/// A local server that answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalServer {
    pub kind: ServerKind,
    /// The address that was probed, without a trailing slash and without `/v1`, such as `http://127.0.0.1:11434`.
    pub base_url: String,
    /// `base_url` plus `/v1`: the OpenAI-compatible root the apps send their chat requests to.
    pub api_base: String,
    pub state: ServerState,
    /// Empty when the state is [`ServerState::Unreadable`].
    pub models: Vec<ServerModel>,
}

/// One address to look at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    pub kind: ServerKind,
    pub base_url: String,
}

/// The two places [`detect`] looks: Ollama on 11434, then LM Studio on 1234.
pub fn default_probes() -> Vec<Probe> {
    vec![
        Probe {
            kind: ServerKind::Ollama,
            base_url: "http://127.0.0.1:11434".to_string(),
        },
        Probe {
            kind: ServerKind::LmStudio,
            base_url: "http://127.0.0.1:1234".to_string(),
        },
    ]
}

/// The local servers running on this computer, Ollama first. Servers that are not there are left out, so this is
/// empty when neither is running.
pub fn detect() -> Vec<LocalServer> {
    detect_at(&default_probes())
}

/// Like [`detect`], at the given addresses. The answer keeps the order of `probes`.
pub fn detect_at(probes: &[Probe]) -> Vec<LocalServer> {
    thread::scope(|scope| {
        let running: Vec<_> = probes
            .iter()
            .map(|probe| scope.spawn(move || check(probe)))
            .collect();
        running
            .into_iter()
            .filter_map(|handle| handle.join().ok().flatten())
            .collect()
    })
}

/// Why a request produced nothing to read.
enum Miss {
    /// Nothing accepted the connection.
    NotThere,
    /// Something is there, but the answer is no use.
    Unreadable,
}

/// The requests to one server, which share one deadline.
struct Session {
    client: reqwest::blocking::Client,
    deadline: Instant,
}

impl Session {
    fn new() -> Option<Session> {
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            // These are programs on this computer: never through a system proxy, never followed elsewhere.
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .ok()?;
        Some(Session {
            client,
            deadline: Instant::now() + TOTAL_TIMEOUT,
        })
    }

    fn left(&self) -> Option<Duration> {
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
    }

    fn get(&self, url: &str) -> Result<Value, Miss> {
        self.send(self.client.get(url))
    }

    fn post(&self, url: &str, body: &Value) -> Result<Value, Miss> {
        self.send(
            self.client
                .post(url)
                .header("Content-Type", "application/json")
                .body(body.to_string()),
        )
    }

    fn send(&self, request: reqwest::blocking::RequestBuilder) -> Result<Value, Miss> {
        let left = self.left().ok_or(Miss::Unreadable)?;
        let response = request.timeout(left).send().map_err(|e| {
            if e.is_connect() {
                Miss::NotThere
            } else {
                Miss::Unreadable
            }
        })?;
        if !response.status().is_success() {
            return Err(Miss::Unreadable);
        }
        let mut body = Vec::new();
        response
            .take(MAX_BODY_BYTES)
            .read_to_end(&mut body)
            .map_err(|_| Miss::Unreadable)?;
        serde_json::from_slice(&body).map_err(|_| Miss::Unreadable)
    }
}

/// Looks at one address: `None` when nothing is there.
fn check(probe: &Probe) -> Option<LocalServer> {
    let base_url = probe.base_url.trim().trim_end_matches('/').to_string();
    let session = Session::new()?;
    let listed = match probe.kind {
        ServerKind::Ollama => ollama_models(&session, &base_url),
        ServerKind::LmStudio => lm_studio_models(&session, &base_url),
    };
    let (state, models) = match listed {
        Ok(models) => (ServerState::Running, models),
        Err(Miss::NotThere) => return None,
        Err(Miss::Unreadable) => (ServerState::Unreadable, Vec::new()),
    };
    Some(LocalServer {
        kind: probe.kind,
        api_base: format!("{base_url}/v1"),
        base_url,
        state,
        models,
    })
}

fn ollama_models(session: &Session, base_url: &str) -> Result<Vec<ServerModel>, Miss> {
    let tags = session.get(&format!("{base_url}/api/tags"))?;
    let entries = tags
        .get("models")
        .and_then(Value::as_array)
        .ok_or(Miss::Unreadable)?;
    let mut models = Vec::new();
    for entry in entries {
        let Some(id) = entry
            .get("name")
            .or_else(|| entry.get("model"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        models.push(ServerModel {
            id: id.to_string(),
            context_length: ollama_context_length(session, base_url, id),
            size_bytes: entry.get("size").and_then(Value::as_u64),
        });
    }
    Ok(models)
}

/// The model's context limit from `/api/show`, `None` for any reason it cannot be had.
fn ollama_context_length(session: &Session, base_url: &str, id: &str) -> Option<u32> {
    let shown = session
        .post(
            &format!("{base_url}/api/show"),
            &serde_json::json!({ "model": id }),
        )
        .ok()?;
    let info = shown.get("model_info")?.as_object()?;
    let key = match info.get("general.architecture").and_then(Value::as_str) {
        Some(architecture) => format!("{architecture}.context_length"),
        // No architecture named: take the one context key there is, if there is exactly one.
        None => {
            let mut keys = info.keys().filter(|k| k.ends_with(".context_length"));
            let only = keys.next()?;
            if keys.next().is_some() {
                return None;
            }
            only.clone()
        }
    };
    u32::try_from(info.get(&key)?.as_u64()?).ok()
}

fn lm_studio_models(session: &Session, base_url: &str) -> Result<Vec<ServerModel>, Miss> {
    let listing = session.get(&format!("{base_url}/api/v0/models"))?;
    let entries = listing
        .get("data")
        .and_then(Value::as_array)
        .ok_or(Miss::Unreadable)?;
    Ok(entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.get("type").and_then(Value::as_str),
                Some("llm" | "vlm")
            )
        })
        .filter_map(|entry| {
            Some(ServerModel {
                id: entry.get("id")?.as_str()?.to_string(),
                context_length: entry
                    .get("max_context_length")
                    .and_then(Value::as_u64)
                    .and_then(|n| u32::try_from(n).ok()),
                size_bytes: None,
            })
        })
        .collect())
}
