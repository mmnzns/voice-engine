//! Writing a reply and computing embeddings with a loaded model.

use crate::model::{Loaded, Target, context_params, load};
use crate::send;
use llama_cpp_2::context::params::LlamaPoolingType;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaChatMessage;
use llama_cpp_2::sampling::LlamaSampler;
use std::time::Instant;
use voice_engine_protocol::{Completion, Embedding, Event, Phase, Reply};

const BATCH: usize = 512;

/// Longest text embedded, in tokens; longer ones are cut (passages are built well under this).
pub(crate) const EMBED_MAX_TOKENS: usize = 1024;

pub(crate) fn embed(
    llama: &LlamaBackend,
    loaded: &mut Option<Loaded>,
    t: Target,
    e: &Embedding,
) -> Result<Vec<Vec<f32>>, String> {
    let model = load(llama, loaded, t)?;
    let ctx_params = context_params(EMBED_MAX_TOKENS as u32, t.gpu.is_some())
        .with_n_batch(EMBED_MAX_TOKENS as u32)
        .with_n_ubatch(EMBED_MAX_TOKENS as u32)
        .with_embeddings(true)
        // The model's own pooling (Qwen3-Embedding: last token) from its metadata.
        .with_pooling_type(LlamaPoolingType::Unspecified);
    let mut ctx = model
        .new_context(llama, ctx_params)
        .map_err(|e| format!("not enough memory for the search model: {e}"))?;
    let vocab = model.vocab();
    let mut out = Vec::with_capacity(e.texts.len());
    let mut batch = LlamaBatch::new(EMBED_MAX_TOKENS, 1);
    for text in &e.texts {
        // add_special: the tokenizer adds what the model expects (Qwen3-Embedding pools on the closing token).
        let mut tokens = vocab.tokenize(text.as_bytes(), true, false);
        tokens.truncate(EMBED_MAX_TOKENS);
        ctx.clear_kv_cache();
        batch.clear();
        for (i, t) in tokens.iter().enumerate() {
            batch
                .add(*t, i as i32, &[0], true)
                .map_err(|e| e.to_string())?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| format!("the search model failed: {e}"))?;
        let v = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| format!("the search model gave no vector: {e}"))?;
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        out.push(v.iter().map(|x| x / norm).collect());
    }
    Ok(out)
}

pub(crate) fn complete(
    llama: &LlamaBackend,
    loaded: &mut Option<Loaded>,
    t: Target,
    c: &Completion,
) -> Result<Reply, String> {
    let started = Instant::now();
    let model = load(llama, loaded, t)?;

    let template = model
        .chat_template(None)
        .map_err(|e| format!("the model has no chat template: {e}"))?;
    let messages = [
        LlamaChatMessage::new("system".into(), c.system.clone()).map_err(|e| e.to_string())?,
        LlamaChatMessage::new("user".into(), c.user.clone()).map_err(|e| e.to_string())?,
    ];
    let prompt = model
        .apply_chat_template(&template, &messages, true)
        .map_err(|e| format!("couldn't build the prompt: {e}"))?;
    let vocab = model.vocab();
    let tokens = vocab.tokenize(prompt.as_bytes(), false, true);
    let n_prompt = tokens.len() as u32;
    if n_prompt + c.max_tokens > c.n_ctx {
        return Err(format!(
            "the transcript part is too long for the model ({n_prompt} tokens, context {})",
            c.n_ctx
        ));
    }

    // The sampler first: a bad schema should fail before the model reads anything.
    let mut samplers = vec![];
    if let Some(schema) = &c.json_schema {
        let grammar = llama_cpp_2::json_schema_to_grammar(schema)
            .map_err(|e| format!("bad output schema: {e}"))?;
        samplers.push(
            LlamaSampler::grammar(model, &grammar, "root")
                .map_err(|e| format!("bad output schema: {e}"))?,
        );
    }
    if c.temperature <= 0.0 {
        samplers.push(LlamaSampler::greedy());
    } else {
        samplers.push(LlamaSampler::top_k(40));
        samplers.push(LlamaSampler::top_p(0.9, 1));
        samplers.push(LlamaSampler::temp(c.temperature));
        samplers.push(LlamaSampler::dist(1234));
    }
    let mut sampler = LlamaSampler::chain_simple(samplers);

    let ctx_params = context_params(c.n_ctx, t.gpu.is_some())
        .with_n_batch(BATCH as u32)
        .with_n_ubatch(BATCH as u32);
    let mut ctx = model
        .new_context(llama, ctx_params)
        .map_err(|e| format!("not enough memory for the AI model: {e}"))?;

    // Read the prompt in batches; only the last token needs logits.
    let mut batch = LlamaBatch::new(BATCH, 1);
    for (start, chunk) in tokens
        .chunks(BATCH)
        .enumerate()
        .map(|(i, ch)| (i * BATCH, ch))
    {
        batch.clear();
        for (j, t) in chunk.iter().enumerate() {
            let pos = start + j;
            batch
                .add(*t, pos as i32, &[0], pos + 1 == tokens.len())
                .map_err(|e| e.to_string())?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| format!("the AI model failed reading: {e}"))?;
        send(&Event::Progress {
            phase: Phase::Reading,
            done: (start + chunk.len()) as u32,
            total: n_prompt,
        });
    }

    let mut out: Vec<u8> = vec![];
    let mut n_cur = tokens.len() as i32;
    let mut written = 0u32;
    let mut truncated = true;
    while written < c.max_tokens {
        // `sample` also accepts the token into the samplers (grammar state included); accepting it again
        // would push the grammar past it.
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        if vocab.is_eog(token) {
            truncated = false;
            break;
        }
        out.extend(vocab.token_to_piece(token, false, None));
        written += 1;
        if written.is_multiple_of(32) {
            send(&Event::Progress {
                phase: Phase::Writing,
                done: written,
                total: 0,
            });
        }
        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| e.to_string())?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|e| format!("the AI model failed writing: {e}"))?;
    }

    Ok(Reply {
        text: String::from_utf8_lossy(&out).into_owned(),
        prompt_tokens: n_prompt,
        output_tokens: written,
        seconds: started.elapsed().as_secs_f32(),
        truncated,
        thinking_tokens: 0,
        load_seconds: 0.0,
        thinking_text: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensayo_a_missing_model_file_is_a_plain_error_not_a_crash() {
        let _turn = crate::BACKEND_TURN
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let llama = LlamaBackend::init().unwrap();
        let mut loaded = None;
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
            gpu: None,
            n_ctx: c.n_ctx,
        };
        let e = complete(&llama, &mut loaded, target, &c).unwrap_err();
        assert!(e.contains("missing"), "{e}");
    }
}
