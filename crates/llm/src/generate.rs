//! Writing a reply and computing embeddings with a loaded model.

use crate::failure::Failure;
use crate::inbox::Inbox;
use crate::model::{Loaded, Target, context_params, load};
use crate::plain;
use crate::send;
use crate::think;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaPoolingType;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::time::Instant;
use voice_engine_protocol::{Backend, Completion, Embedding, ErrorKind, Event, Phase, Reply};

const BATCH: usize = 512;

/// Longest text embedded, in tokens; longer ones are cut (passages are built well under this).
pub(crate) const EMBED_MAX_TOKENS: usize = 1024;

pub(crate) fn embed(
    llama: &LlamaBackend,
    loaded: &mut Option<Loaded>,
    t: Target,
    e: &Embedding,
    inbox: &mut Inbox,
) -> Result<(Vec<Vec<f32>>, Backend), Failure> {
    let (model, _) = load(llama, loaded, t)?;
    let ctx_params = context_params(EMBED_MAX_TOKENS as u32, t.place.on_card())
        .with_n_batch(EMBED_MAX_TOKENS as u32)
        .with_n_ubatch(EMBED_MAX_TOKENS as u32)
        .with_embeddings(true)
        // The model's own pooling (Qwen3-Embedding: last token) from its metadata.
        .with_pooling_type(LlamaPoolingType::Unspecified);
    let mut ctx = model.new_context(llama, ctx_params).map_err(|e| {
        Failure::out_of_memory(format!("not enough memory for the search model: {e}"))
    })?;
    let vocab = model.vocab();
    let mut out = Vec::with_capacity(e.texts.len());
    let mut batch = LlamaBatch::new(EMBED_MAX_TOKENS, 1);
    for text in &e.texts {
        if inbox.cancel_requested() {
            return Err(Failure::cancelled());
        }
        // add_special: the tokenizer adds what the model expects (Qwen3-Embedding pools on the closing token).
        let mut tokens = vocab.tokenize(text.as_bytes(), true, false);
        tokens.truncate(EMBED_MAX_TOKENS);
        ctx.clear_kv_cache();
        batch.clear();
        for (i, t) in tokens.iter().enumerate() {
            batch
                .add(*t, i as i32, &[0], true)
                .map_err(|e| Failure::internal(e.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| Failure::internal(format!("the search model failed: {e}")))?;
        let v = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| Failure::internal(format!("the search model gave no vector: {e}")))?;
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        out.push(v.iter().map(|x| x / norm).collect());
    }
    Ok((out, crate::backend(t.place)))
}

pub(crate) fn complete(
    llama: &LlamaBackend,
    loaded: &mut Option<Loaded>,
    t: Target,
    c: &Completion,
    inbox: &mut Inbox,
) -> Result<Reply, Failure> {
    let started = Instant::now();
    let (model, load_seconds) = load(llama, loaded, t)?;

    // A file with a chat template gets its own format; one without gets the plain format (plain.rs) and says so.
    let (prompt, thinks, plain_format) = match model.chat_template(None) {
        Ok(template) => {
            // Only a template with a thinking step gets a prompt ending, and only then can a reply have thinking.
            let thinks = think::template_thinks(&template.to_string().unwrap_or_default());
            let thinking = thinks && c.thinking;
            let messages = [
                LlamaChatMessage::new("system".into(), c.system.clone()).map_err(|_| {
                    Failure::internal("couldn't build the prompt: a null byte in the text")
                })?,
                LlamaChatMessage::new("user".into(), c.user.clone()).map_err(|_| {
                    Failure::internal("couldn't build the prompt: a null byte in the text")
                })?,
            ];
            let mut prompt = model
                .apply_chat_template(&template, &messages, true)
                .map_err(|e| Failure::internal(prompt_error(&e)))?;
            prompt.push_str(think::prompt_ending(thinks, thinking));
            (prompt, thinks, false)
        }
        Err(_) => (plain::prompt(&c.system, &c.user), false, true),
    };
    let thinking = thinks && c.thinking;
    let vocab = model.vocab();
    // A chat template writes the model's special tokens itself (its start-of-text one included), and they are parsed
    // as such. The plain prompt has none: the tokenizer adds the start-of-text token, and the text stays plain text.
    let tokens = if plain_format {
        vocab.tokenize(prompt.as_bytes(), true, false)
    } else {
        vocab.tokenize(prompt.as_bytes(), false, true)
    };
    let n_prompt = tokens.len() as u32;
    let budget = if thinking {
        c.thinking_budget.unwrap_or(0)
    } else {
        0
    };
    // The tokens the helper writes itself when thinking runs past its budget; they need room too.
    let force_end = if budget > 0 {
        vocab.tokenize(think::FORCE_END.as_bytes(), false, true)
    } else {
        vec![]
    };
    if n_prompt
        .saturating_add(c.max_tokens)
        .saturating_add(budget)
        .saturating_add(force_end.len() as u32)
        > c.n_ctx
    {
        return Err(Failure::new(
            ErrorKind::PromptTooLong {
                prompt_tokens: n_prompt,
                n_ctx: c.n_ctx,
                max_tokens: c.max_tokens,
            },
            format!(
                "the prompt is too long for the model ({n_prompt} tokens, context {})",
                c.n_ctx
            ),
        ));
    }

    // The answer's sampler first: a bad schema should fail before the model reads anything.
    let mut answer_sampler = Some(sampling_chain(model, c, true)?);
    let mut sampler = if thinking {
        sampling_chain(model, c, false)?
    } else {
        answer_sampler.take().expect("built above")
    };

    let ctx_params = context_params(c.n_ctx, t.place.on_card())
        .with_n_batch(BATCH as u32)
        .with_n_ubatch(BATCH as u32);
    let mut ctx = model
        .new_context(llama, ctx_params)
        .map_err(|e| Failure::out_of_memory(format!("not enough memory for the AI model: {e}")))?;

    // The model is loaded and the prompt is ready: say so before reading it, so a client can start a limit on the reply
    // here (`LlmClient::complete_within`), whether or not this request had to load the model.
    send(&Event::Progress {
        phase: Phase::Reading,
        done: 0,
        total: n_prompt,
    });
    // Read the prompt in batches; only the last token needs logits.
    let mut batch = LlamaBatch::new(BATCH, 1);
    for (start, chunk) in tokens
        .chunks(BATCH)
        .enumerate()
        .map(|(i, ch)| (i * BATCH, ch))
    {
        if inbox.cancel_requested() {
            return Err(Failure::cancelled());
        }
        batch.clear();
        for (j, t) in chunk.iter().enumerate() {
            let pos = start + j;
            batch
                .add(*t, pos as i32, &[0], pos + 1 == tokens.len())
                .map_err(|e| Failure::internal(e.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| Failure::internal(format!("the AI model failed reading: {e}")))?;
        send(&Event::Progress {
            phase: Phase::Reading,
            done: (start + chunk.len()) as u32,
            total: n_prompt,
        });
    }

    let mut out: Vec<u8> = vec![];
    // Where the answer starts in `out`: 0 without thinking, just after `</think>` with it.
    let mut answer_from: Option<usize> = if thinking { None } else { Some(0) };
    let mut n_cur = tokens.len() as i32;
    let mut written = 0u32;
    let mut thought = 0u32;
    let mut truncated = true;
    // Stops at max_tokens for the answer, or when the context is full (a long thinking without a budget).
    while written < c.max_tokens && (n_cur as u32) < c.n_ctx {
        if inbox.cancel_requested() {
            return Err(Failure::cancelled());
        }
        // Thinking past its budget: the helper writes the closing tag itself and the answer starts. The text
        // is rendered with special tokens so the closing tag is always found.
        if answer_from.is_none() && budget > 0 && thought >= budget {
            for token in &force_end {
                out.extend(vocab.token_to_piece(*token, true, None));
                feed(*token, &mut ctx, &mut batch, &mut n_cur)?;
            }
        }
        if answer_from.is_none()
            && let Some(end) = think::end_of_thinking(&out)
        {
            answer_from = Some(end);
            sampler = answer_sampler
                .take()
                .expect("the answer's sampler is used once");
        }
        // `sample` also accepts the token into the samplers (grammar state included); accepting it again
        // would push the grammar past it.
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        if vocab.is_eog(token) {
            truncated = false;
            break;
        }
        // While thinking, special tokens are rendered too, so a closing tag that is one is still seen.
        out.extend(vocab.token_to_piece(token, answer_from.is_none(), None));
        if answer_from.is_some() {
            written += 1;
            if written.is_multiple_of(32) {
                send(&Event::Progress {
                    phase: Phase::Writing,
                    done: written,
                    total: 0,
                });
            }
        } else {
            thought += 1;
        }
        feed(token, &mut ctx, &mut batch, &mut n_cur)?;
    }
    // The last token written may have been the closing tag, with the context full right after it.
    if answer_from.is_none() {
        answer_from = think::end_of_thinking(&out);
    }

    let (thinking_text, answer) = match answer_from {
        Some(from) if thinking => (
            Some(
                String::from_utf8_lossy(&out[..from])
                    .trim_end()
                    .trim_end_matches(think::CLOSE)
                    .trim()
                    .to_string(),
            ),
            String::from_utf8_lossy(&out[from..]).into_owned(),
        ),
        Some(_) => (None, String::from_utf8_lossy(&out).into_owned()),
        // Cut off while still thinking: no answer.
        None => (
            Some(String::from_utf8_lossy(&out).trim().to_string()),
            String::new(),
        ),
    };
    Ok(Reply {
        text: think::strip_think(&answer).trim_start().to_string(),
        prompt_tokens: n_prompt,
        output_tokens: written,
        seconds: started.elapsed().as_secs_f32(),
        truncated,
        thinking_tokens: thought,
        load_seconds,
        thinking_text,
        backend: crate::backend(t.place),
        plain_format,
    })
}

/// Puts one token in the context after the ones already there.
fn feed(
    token: LlamaToken,
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    n_cur: &mut i32,
) -> Result<(), Failure> {
    batch.clear();
    batch
        .add(token, *n_cur, &[0], true)
        .map_err(|e| Failure::internal(e.to_string()))?;
    *n_cur += 1;
    ctx.decode(batch)
        .map_err(|e| Failure::internal(format!("the AI model failed writing: {e}")))
}

/// The sampling chain for a request. With `answer`, the reply's schema (if any) constrains it; thinking is free
/// text. Temperature 0 is greedy; above 0 the seed makes it repeatable.
/// What went wrong applying the chat template, by kind only. The library's own text is not put in a message: it
/// wraps strings that were built from the prompt, and that can be what a person said.
fn prompt_error(e: &llama_cpp_2::ApplyChatTemplateError) -> String {
    use llama_cpp_2::ApplyChatTemplateError as E;
    let what = match e {
        E::NulError(_) => "a null byte in the text".to_string(),
        E::FromUtf8Error(_) => "the result was not valid UTF-8".to_string(),
        E::NullResult => "llama.cpp gave no result".to_string(),
        E::FfiError(code) => format!("llama.cpp error {code}"),
    };
    format!("couldn't build the prompt: {what}")
}

fn sampling_chain(
    model: &LlamaModel,
    c: &Completion,
    answer: bool,
) -> Result<LlamaSampler, Failure> {
    let mut samplers = vec![];
    if answer && let Some(schema) = &c.json_schema {
        let bad_schema = |e: &dyn std::fmt::Display| {
            Failure::new(
                ErrorKind::BadSchema {
                    detail: e.to_string(),
                },
                // The schema is the app's own, but the message follows the same rule as the prompt's: no library text.
                "bad output schema (the detail says what)",
            )
        };
        let grammar = llama_cpp_2::json_schema_to_grammar(schema).map_err(|e| bad_schema(&e))?;
        samplers.push(LlamaSampler::grammar(model, &grammar, "root").map_err(|e| bad_schema(&e))?);
    }
    if c.temperature <= 0.0 {
        samplers.push(LlamaSampler::greedy());
    } else {
        samplers.push(LlamaSampler::top_k(40));
        samplers.push(LlamaSampler::top_p(0.9, 1));
        samplers.push(LlamaSampler::temp(c.temperature));
        samplers.push(LlamaSampler::dist(c.seed.unwrap_or(1234)));
    }
    Ok(LlamaSampler::chain_simple(samplers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Place;

    #[test]
    fn ensayo_a_missing_model_file_is_a_plain_error_not_a_crash() {
        let _turn = crate::BACKEND_TURN
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let llama = LlamaBackend::init().unwrap();
        let mut loaded = None;
        let (_tx, rx) = std::sync::mpsc::channel();
        let mut inbox = Inbox::new(rx);
        let c = Completion {
            model_path: "/nonexistent/model.gguf".into(),
            n_ctx: 512,
            system: String::new(),
            user: String::new(),
            json_schema: None,
            max_tokens: 1,
            temperature: 0.0,
            gpu: false,
            thinking: false,
            thinking_budget: None,
            seed: None,
        };
        let target = Target {
            path: &c.model_path,
            place: Place::Processor,
            n_ctx: c.n_ctx,
        };
        let e = complete(&llama, &mut loaded, target, &c, &mut inbox).unwrap_err();
        assert_eq!(
            e.error,
            ErrorKind::ModelMissing {
                path: "/nonexistent/model.gguf".into()
            }
        );
        assert!(e.message.contains("missing"), "{}", e.message);
    }

    #[test]
    fn a_prompt_error_names_its_kind_and_never_the_text_it_wrapped() {
        let said = "remember to call my doctor about the rash on Tuesday";
        let nul = std::ffi::CString::new(format!("{said}\0")).unwrap_err();
        let utf8 = String::from_utf8(said.bytes().chain([0xff]).collect()).unwrap_err();
        for e in [
            llama_cpp_2::ApplyChatTemplateError::NulError(nul),
            llama_cpp_2::ApplyChatTemplateError::FromUtf8Error(utf8),
            llama_cpp_2::ApplyChatTemplateError::NullResult,
            llama_cpp_2::ApplyChatTemplateError::FfiError(-1),
        ] {
            let text = prompt_error(&e);
            assert!(text.starts_with("couldn't build the prompt: "), "{text}");
            assert!(!text.contains("doctor") && !text.contains("rash"), "{text}");
        }
        assert_eq!(
            prompt_error(&llama_cpp_2::ApplyChatTemplateError::FfiError(-3)),
            "couldn't build the prompt: llama.cpp error -3"
        );
    }
}
