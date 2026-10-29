//! What a GGUF file says about itself, read from its header without loading the weights: whether it can chat, whether
//! it thinks, how much context it was trained for. Used before offering a file the person imported.

use crate::failure::Failure;
use crate::think;
use llama_cpp_2::gguf::GgufContext;
use std::path::Path;
use voice_engine_protocol::Inspection;

/// gguf_type values (ggml's gguf.h).
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_STRING: u32 = 8;

/// The stored type of a key's value. bindgen makes `gguf_type` a `u32` on macOS and Linux but an `i32` on Windows, so the
/// cast is needed there even though clippy sees it as a no-op here.
#[allow(clippy::unnecessary_cast)]
fn stored_type(gguf: &GgufContext, index: i64) -> u32 {
    gguf.kv_type(index) as u32
}

pub(crate) fn inspect(path: &str) -> Result<Inspection, Failure> {
    if !Path::new(path).is_file() {
        return Err(Failure::missing(path));
    }
    let gguf = GgufContext::from_file(Path::new(path))
        .ok_or_else(|| Failure::internal(format!("{path} is not a GGUF model file")))?;
    let string = |key: &str| {
        let i = gguf.find_key(key);
        // Reading a value of another type would abort inside llama.cpp, so check the stored type first.
        (i >= 0 && stored_type(&gguf, i) == GGUF_TYPE_STRING)
            .then(|| gguf.val_str(i))
            .flatten()
            .map(str::to_string)
    };
    let architecture = string("general.architecture").unwrap_or_default();
    let n_ctx_train = {
        let i = gguf.find_key(&format!("{architecture}.context_length"));
        if i < 0 {
            0
        } else {
            match stored_type(&gguf, i) {
                GGUF_TYPE_UINT32 => gguf.val_u32(i),
                GGUF_TYPE_UINT64 => gguf.val_u64(i).min(u32::MAX as u64) as u32,
                _ => 0,
            }
        }
    };
    let template = string("tokenizer.chat_template");
    Ok(Inspection {
        has_chat_template: template.is_some(),
        thinks: template.as_deref().is_some_and(think::template_thinks),
        n_ctx_train,
        architecture,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    /// A GGUF file with only metadata (no tensors): magic, version 3, 0 tensors, then each key and value.
    pub(crate) fn tiny_gguf(path: &Path, kv: &[(&str, Value)]) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&(kv.len() as u64).to_le_bytes()).unwrap();
        let s = |f: &mut std::fs::File, s: &str| {
            f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
            f.write_all(s.as_bytes()).unwrap();
        };
        for (key, value) in kv {
            s(&mut f, key);
            match value {
                Value::U32(v) => {
                    f.write_all(&GGUF_TYPE_UINT32.to_le_bytes()).unwrap();
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
                Value::Str(v) => {
                    f.write_all(&GGUF_TYPE_STRING.to_le_bytes()).unwrap();
                    s(&mut f, v);
                }
            }
        }
    }

    pub(crate) enum Value {
        U32(u32),
        Str(&'static str),
    }

    #[test]
    fn a_chat_model_that_thinks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        tiny_gguf(
            &path,
            &[
                ("general.architecture", Value::Str("qwen35")),
                ("qwen35.context_length", Value::U32(262_144)),
                (
                    "tokenizer.chat_template",
                    Value::Str("{% if enable_thinking %}<think>{% endif %}"),
                ),
            ],
        );
        assert_eq!(
            inspect(path.to_str().unwrap()).unwrap(),
            Inspection {
                has_chat_template: true,
                thinks: true,
                n_ctx_train: 262_144,
                architecture: "qwen35".into(),
            }
        );
    }

    #[test]
    fn a_model_without_a_chat_template_and_a_file_that_is_not_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("base.gguf");
        tiny_gguf(&path, &[("general.architecture", Value::Str("llama"))]);
        let i = inspect(path.to_str().unwrap()).unwrap();
        assert!(
            !i.has_chat_template && !i.thinks && i.n_ctx_train == 0,
            "{i:?}"
        );
        let junk = dir.path().join("junk.gguf");
        std::fs::write(&junk, b"not a model").unwrap();
        let e = inspect(junk.to_str().unwrap()).unwrap_err();
        assert_eq!(e.error, voice_engine_protocol::ErrorKind::Internal);
        let e = inspect("/nonexistent/x.gguf").unwrap_err();
        assert!(matches!(
            e.error,
            voice_engine_protocol::ErrorKind::ModelMissing { .. }
        ));
    }
}
