//! The local AI models (GGUF files run by the llm helper). What each model is and where it downloads from; which
//! one an app recommends, and its licence text, stay in the app.

use crate::catalog::{EngineKind, ModelFile, ModelInfo};

pub static CATALOG: &[ModelInfo] = &[
    ModelInfo {
        id: "qwen3.5-4b",
        name: "Qwen3.5 4B",
        maker: "Qwen",
        note: "Small and quick enough for a laptop.",
        engine: EngineKind::Llm,
        languages: "Many languages",
        speed: 3,
        accuracy: 2,
        files: &[ModelFile {
            name: "Qwen3.5-4B-Q4_K_M.gguf",
            url: "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
            size: 2_740_937_888,
            sha256: "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
        }],
        reuse_dirs: &[],
    },
    ModelInfo {
        id: "qwen3.5-9b",
        name: "Qwen3.5 9B",
        maker: "Qwen",
        note: "Larger and a little more thorough; needs 16 GB of memory and takes about twice as long.",
        engine: EngineKind::Llm,
        languages: "Many languages",
        speed: 2,
        accuracy: 3,
        files: &[ModelFile {
            name: "Qwen3.5-9B-Q4_K_M.gguf",
            url: "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf",
            size: 5_680_522_464,
            sha256: "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
        }],
        reuse_dirs: &[],
    },
];

pub fn find(id: &str) -> Option<&'static ModelInfo> {
    CATALOG.iter().find(|m| m.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineKind;

    #[test]
    fn local_ai_catalog_is_consistent() {
        assert!(!CATALOG.is_empty());
        for m in CATALOG {
            assert_eq!(m.engine, EngineKind::Llm, "{}", m.id);
            assert_eq!(m.files.len(), 1, "{}: one GGUF file", m.id);
            let f = &m.files[0];
            assert!(f.name.ends_with(".gguf"), "{}", f.name);
            assert!(f.url.ends_with(f.name), "{}: url names the file", m.id);
            assert!(
                f.url.contains("/resolve/") && !f.url.contains("/resolve/main/"),
                "{}: pinned to a commit",
                m.id
            );
            assert_eq!(f.sha256.len(), 64, "{}", m.id);
            assert!(f.size > 1_000_000_000, "{}", m.id);
        }
        assert_eq!(
            find("qwen3.5-4b").map(|m| m.files[0].name),
            Some("Qwen3.5-4B-Q4_K_M.gguf")
        );
        assert!(find("nope").is_none());
    }
}
