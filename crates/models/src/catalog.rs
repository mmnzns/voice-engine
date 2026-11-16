//! The model catalog: every model the engine can fetch (speech, voice detection, speaker separation, search, local
//! AI), with its files, sizes, hashes, licence and what runs it. The catalog never recommends; which model an app
//! offers first is the app's business.

/// `#[non_exhaustive]`: a kind can be added in a minor release, so an app that matches on it needs a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Kind {
    /// Whisper (whisper.cpp ggml files).
    Whisper,
    /// Parakeet TDT (v2, v3, Ultra).
    Parakeet,
    /// Parakeet Unified (English).
    ParakeetUnified,
    /// Nemotron ASR (multilingual).
    Nemotron,
    /// Cohere Transcribe: text only, one piece of text per chunk, no word times.
    Cohere,
    /// Multitalker Parakeet TDT: who said what, for overlapping speech. It needs the speaker separation model
    /// and has its own API (`voice_engine_stt::multitalker`), so it is not a plain speech model.
    Multitalker,
    /// Voice activity detection (Silero).
    Vad,
    /// Speaker separation for the Them track.
    Diarization,
    /// Text embeddings for search by meaning (run by the llm helper).
    Embedding,
    /// A GGUF model for the local AI (run by the llm helper).
    Llm,
}

impl Kind {
    /// A model that turns speech into text (the ones an app offers as speech models). `Multitalker` is not one:
    /// it needs the speaker separation model and has its own API.
    pub fn is_speech(self) -> bool {
        matches!(
            self,
            Kind::Whisper | Kind::Parakeet | Kind::ParakeetUnified | Kind::Nemotron | Kind::Cohere
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Licence {
    pub spdx: &'static str,
    pub url: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuSupport {
    pub mac_metal: bool,
    pub windows_vulkan: bool,
    pub windows_directml: bool,
}

/// What an app shows for a local AI model. Context sizes are in tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmInfo {
    /// The longest context the model was trained for (read from the GGUF header).
    pub context_max: u32,
    /// The longest context that fits on a computer with 8 GB of memory, weights included (measured, Task 4).
    pub context_8gb: u32,
    /// The same for 16 GB.
    pub context_16gb: u32,
    /// Its chat template has a thinking step.
    pub thinks: bool,
}

pub mod licences {
    use super::Licence;
    pub const MIT: Licence = Licence {
        spdx: "MIT",
        url: "https://opensource.org/license/mit",
    };
    pub const APACHE_2: Licence = Licence {
        spdx: "Apache-2.0",
        url: "https://www.apache.org/licenses/LICENSE-2.0",
    };
    pub const CC_BY_4: Licence = Licence {
        spdx: "CC-BY-4.0",
        url: "https://creativecommons.org/licenses/by/4.0/",
    };
    /// Gemma 4 is released under Apache 2.0; Google's own page for it is the link on the model card.
    pub const GEMMA_4: Licence = Licence {
        spdx: "Apache-2.0",
        url: "https://ai.google.dev/gemma/docs/gemma_4_license",
    };
    pub const NVIDIA_OPEN_MODEL: Licence = Licence {
        spdx: "NVIDIA-Open-Model-License",
        url: "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
    };
    pub const OPENMDW_1_1: Licence = Licence {
        spdx: "OpenMDW-1.1",
        url: "https://openmdw.ai/license/1-1/",
    };
}

pub mod gpu {
    use super::GpuSupport;
    pub const WHISPER_AND_LLM: GpuSupport = GpuSupport {
        mac_metal: true,
        windows_vulkan: true,
        windows_directml: false,
    };
    pub const PARAKEET_TDT: GpuSupport = GpuSupport {
        mac_metal: false,
        windows_vulkan: false,
        windows_directml: true,
    };
    pub const NONE: GpuSupport = GpuSupport {
        mac_metal: false,
        windows_vulkan: false,
        windows_directml: false,
    };
}

#[derive(Debug)]
pub struct ModelFile {
    pub name: &'static str,
    pub url: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

#[derive(Debug)]
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub maker: &'static str,
    /// Plain sentence, names no app, makes no recommendation.
    pub note: &'static str,
    pub kind: Kind,
    /// Shown by the app ("25 European languages").
    pub languages: &'static str,
    pub english_only: bool,
    /// 1..3, for the dots the app shows.
    pub speed: u8,
    pub accuracy: u8,
    pub licence: Licence,
    pub files: &'static [ModelFile],
    pub gpu: GpuSupport,
    /// Speech: whether it keeps "um", "uh" and "like" when asked to. `None` until measured (spec 9.4); measured
    /// for `whisper-turbo` and `parakeet-ultra` (docs/bench-2026-10-29.md, "Fillers").
    pub keeps_fillers: Option<bool>,
    /// Speech: the longest chunk to send (seconds). None: any length (Whisper).
    pub chunk_secs: Option<u32>,
    /// Speech: whether the transcript carries word times. False for Cohere: one segment per chunk, no words. True
    /// for every Whisper, Parakeet, Parakeet Unified and Nemotron entry, and for Multitalker. False for every entry
    /// that is not a speech model (voice detection, speaker separation, search, local AI).
    pub word_times: bool,
    /// Local AI only.
    pub llm: Option<LlmInfo>,
}

impl ModelInfo {
    pub fn size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    pub fn is_speech(&self) -> bool {
        self.kind.is_speech()
    }
}

pub fn all() -> &'static [ModelInfo] {
    CATALOG
}

pub fn speech() -> impl Iterator<Item = &'static ModelInfo> {
    CATALOG.iter().filter(|m| m.is_speech())
}

pub fn local_ai() -> impl Iterator<Item = &'static ModelInfo> {
    CATALOG.iter().filter(|m| m.kind == Kind::Llm)
}

pub fn find(id: &str) -> Option<&'static ModelInfo> {
    CATALOG.iter().find(|m| m.id == id)
}

/// Every model, speech first, in a stable order. Which one an app recommends is the app's business.
pub static CATALOG: &[ModelInfo] = &[
    // Whisper Turbo: on real calls it was more complete and accurate than Parakeet (which skipped stretches and
    // misheard more), covers Filipino, and at ~13x real time an hour-long call takes a few minutes.
    ModelInfo {
        id: "whisper-turbo",
        name: "Whisper Large Turbo (compressed)",
        maker: "OpenAI",
        note: "Compressed to a smaller download, 574 MB. 99 languages, including Filipino.",
        kind: Kind::Whisper,
        languages: "99 languages",
        english_only: false,
        speed: 2,
        accuracy: 3,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-large-v3-turbo-q5_0.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-large-v3-turbo-q5_0.bin"
            ),
            size: 574_041_195,
            sha256: "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: Some(true),
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-turbo-full",
        name: "Whisper Large Turbo (full size)",
        maker: "OpenAI",
        note: "The uncompressed Turbo model, a 1.6 GB download. 99 languages, including Filipino.",
        kind: Kind::Whisper,
        languages: "99 languages",
        english_only: false,
        speed: 2,
        accuracy: 3,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-large-v3-turbo.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-large-v3-turbo.bin"
            ),
            size: 1_624_555_275,
            sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "parakeet-v3",
        name: "Parakeet",
        maker: "NVIDIA",
        note: "Good for quick notes in 25 European languages.",
        kind: Kind::Parakeet,
        languages: "25 European languages",
        english_only: false,
        speed: 3,
        accuracy: 1,
        licence: licences::CC_BY_4,
        files: &[
            ModelFile {
                name: "encoder-model.int8.onnx",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/",
                    "encoder-model.int8.onnx"
                ),
                size: 652_183_999,
                sha256: "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09",
            },
            ModelFile {
                name: "decoder_joint-model.int8.onnx",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/",
                    "decoder_joint-model.int8.onnx"
                ),
                size: 18_202_004,
                sha256: "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
            },
            ModelFile {
                name: "nemo128.onnx",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/",
                    "nemo128.onnx"
                ),
                size: 139_764,
                sha256: "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f",
            },
            ModelFile {
                name: "vocab.txt",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/",
                    "vocab.txt"
                ),
                size: 93_939,
                sha256: "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
            },
            ModelFile {
                name: "config.json",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/",
                    "config.json"
                ),
                size: 97,
                sha256: "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
            },
        ],
        gpu: gpu::PARAKEET_TDT,
        keeps_fillers: None,
        chunk_secs: Some(40),
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "parakeet-v3-full",
        name: "Parakeet (full size)",
        maker: "NVIDIA",
        note: "The uncompressed Parakeet model, a 2.5 GB download. 25 European languages.",
        kind: Kind::Parakeet,
        languages: "25 European languages",
        english_only: false,
        speed: 3,
        accuracy: 1,
        licence: licences::CC_BY_4,
        files: &[
            ModelFile {
                name: "encoder-model.onnx",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/",
                    "encoder-model.onnx"
                ),
                size: 41_770_866,
                sha256: "98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1",
            },
            ModelFile {
                name: "encoder-model.onnx.data",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/",
                    "encoder-model.onnx.data"
                ),
                size: 2_435_420_160,
                sha256: "9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36",
            },
            ModelFile {
                name: "decoder_joint-model.onnx",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/",
                    "decoder_joint-model.onnx"
                ),
                size: 72_520_893,
                sha256: "e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1",
            },
            ModelFile {
                name: "nemo128.onnx",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/",
                    "nemo128.onnx"
                ),
                size: 139_764,
                sha256: "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f",
            },
            ModelFile {
                name: "vocab.txt",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/",
                    "vocab.txt"
                ),
                size: 93_939,
                sha256: "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
            },
            ModelFile {
                name: "config.json",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/",
                    "config.json"
                ),
                size: 97,
                sha256: "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
            },
        ],
        gpu: gpu::PARAKEET_TDT,
        keeps_fillers: None,
        chunk_secs: Some(40),
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-small",
        name: "Whisper Small",
        maker: "OpenAI",
        note: "A smaller download that works on older computers. Lower accuracy.",
        kind: Kind::Whisper,
        languages: "99 languages",
        english_only: false,
        speed: 3,
        accuracy: 1,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-small.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-small.bin"
            ),
            size: 487_601_967,
            sha256: "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-small-en",
        name: "Whisper Small (English)",
        maker: "OpenAI",
        note: "English only. A smaller download than the multilingual Small, with better English accuracy.",
        kind: Kind::Whisper,
        languages: "English",
        english_only: true,
        speed: 3,
        accuracy: 1,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-small.en.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-small.en.bin"
            ),
            size: 487_614_201,
            sha256: "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-large-v3",
        name: "Whisper Large v3 (compressed)",
        maker: "OpenAI",
        note: "Whisper’s largest model, compressed to a smaller download of 1.1 GB. Very accurate, slower than Turbo.",
        kind: Kind::Whisper,
        languages: "99 languages",
        english_only: false,
        speed: 1,
        accuracy: 3,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-large-v3-q5_0.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-large-v3-q5_0.bin"
            ),
            size: 1_081_140_203,
            sha256: "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-large-v3-full",
        name: "Whisper Large v3 (full size)",
        maker: "OpenAI",
        note: "The uncompressed Large v3 model, a 3.1 GB download. Very accurate, slower than Turbo.",
        kind: Kind::Whisper,
        languages: "99 languages",
        english_only: false,
        speed: 1,
        accuracy: 3,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-large-v3.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-large-v3.bin"
            ),
            size: 3_095_033_483,
            sha256: "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-large-v2",
        name: "Whisper Large v2 (compressed)",
        maker: "OpenAI",
        note: "The earlier large model, compressed to a smaller download of 1.1 GB. Some voices and accents come out better.",
        kind: Kind::Whisper,
        languages: "99 languages",
        english_only: false,
        speed: 1,
        accuracy: 3,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-large-v2-q5_0.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-large-v2-q5_0.bin"
            ),
            size: 1_080_732_091,
            sha256: "3a214837221e4530dbc1fe8d734f302af393eb30bd0ed046042ebf4baf70f6f2",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-large-v2-full",
        name: "Whisper Large v2 (full size)",
        maker: "OpenAI",
        note: "The uncompressed Large v2 model, a 3.1 GB download. Some voices and accents come out better.",
        kind: Kind::Whisper,
        languages: "99 languages",
        english_only: false,
        speed: 1,
        accuracy: 3,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-large-v2.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-large-v2.bin"
            ),
            size: 3_094_623_691,
            sha256: "9a423fe4d40c82774b6af34115b8b935f34152246eb19e80e376071d3f999487",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-medium",
        name: "Whisper Medium",
        maker: "OpenAI",
        note: "Between Small and Large: good accuracy at a moderate speed.",
        kind: Kind::Whisper,
        languages: "99 languages",
        english_only: false,
        speed: 1,
        accuracy: 2,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-medium.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-medium.bin"
            ),
            size: 1_533_763_059,
            sha256: "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-medium-en",
        name: "Whisper Medium (English)",
        maker: "OpenAI",
        note: "English only. Good accuracy at a moderate speed, a 1.5 GB download.",
        kind: Kind::Whisper,
        languages: "English",
        english_only: true,
        speed: 1,
        accuracy: 2,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-medium.en.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-medium.en.bin"
            ),
            size: 1_533_774_781,
            sha256: "cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-base",
        name: "Whisper Base",
        maker: "OpenAI",
        note: "Very small and fast. Rough on names and accents.",
        kind: Kind::Whisper,
        languages: "99 languages",
        english_only: false,
        speed: 3,
        accuracy: 1,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-base.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-base.bin"
            ),
            size: 147_951_465,
            sha256: "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-base-en",
        name: "Whisper Base (English)",
        maker: "OpenAI",
        note: "English only. Very small and fast.",
        kind: Kind::Whisper,
        languages: "English",
        english_only: true,
        speed: 3,
        accuracy: 1,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-base.en.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-base.en.bin"
            ),
            size: 147_964_211,
            sha256: "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-tiny",
        name: "Whisper Tiny",
        maker: "OpenAI",
        note: "The smallest. For trying the app out or very old computers.",
        kind: Kind::Whisper,
        languages: "99 languages",
        english_only: false,
        speed: 3,
        accuracy: 1,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-tiny.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-tiny.bin"
            ),
            size: 77_691_713,
            sha256: "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "whisper-tiny-en",
        name: "Whisper Tiny (English)",
        maker: "OpenAI",
        note: "English only. The smallest download, 78 MB.",
        kind: Kind::Whisper,
        languages: "English",
        english_only: true,
        speed: 3,
        accuracy: 1,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-tiny.en.bin",
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/",
                "ggml-tiny.en.bin"
            ),
            size: 77_704_715,
            sha256: "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "parakeet-ultra",
        name: "Parakeet Ultra",
        maker: "NVIDIA",
        note: "Parakeet v3 further trained by Moondream for better accuracy. 25 European languages.",
        kind: Kind::Parakeet,
        languages: "25 European languages",
        english_only: false,
        speed: 3,
        accuracy: 2,
        licence: licences::CC_BY_4,
        files: &[
            ModelFile {
                name: "encoder-model.int8.onnx",
                url: concat!(
                    "https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx/resolve/3646b2f1a516a2310810902bfaa67515f3c89d38/",
                    "int8/encoder-model.int8.onnx"
                ),
                size: 649_524_002,
                sha256: "fc7de298eae88ba8aae8ea62bde60cd401401bd294748b8d57a72e5ad6b9d04a",
            },
            ModelFile {
                name: "decoder_joint-model.int8.onnx",
                url: concat!(
                    "https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx/resolve/3646b2f1a516a2310810902bfaa67515f3c89d38/",
                    "int8/decoder_joint-model.int8.onnx"
                ),
                size: 18_203_490,
                sha256: "f7e2db395a3b738cb2893cfb853d25863cebcc5282583a2e86b5762559e0bd32",
            },
            ModelFile {
                name: "nemo128.onnx",
                url: concat!(
                    "https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx/resolve/3646b2f1a516a2310810902bfaa67515f3c89d38/",
                    "nemo128.onnx"
                ),
                size: 139_764,
                sha256: "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f",
            },
            ModelFile {
                name: "vocab.txt",
                url: concat!(
                    "https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx/resolve/3646b2f1a516a2310810902bfaa67515f3c89d38/",
                    "vocab.txt"
                ),
                size: 93_939,
                sha256: "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
            },
            ModelFile {
                name: "config.json",
                url: concat!(
                    "https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx/resolve/3646b2f1a516a2310810902bfaa67515f3c89d38/",
                    "config.json"
                ),
                size: 97,
                sha256: "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
            },
        ],
        gpu: gpu::PARAKEET_TDT,
        keeps_fillers: Some(true),
        chunk_secs: Some(40),
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "parakeet-v2",
        name: "Parakeet v2",
        maker: "NVIDIA",
        note: "English only. Fast and very accurate in English.",
        kind: Kind::Parakeet,
        languages: "English",
        english_only: true,
        speed: 3,
        accuracy: 2,
        licence: licences::CC_BY_4,
        files: &[
            ModelFile {
                name: "encoder-model.int8.onnx",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/0bbb45a3365852604aef28b538a8f066f4ccaa85/",
                    "encoder-model.int8.onnx"
                ),
                size: 652_184_014,
                sha256: "3e0581fda6ab843888b51e56d7ee78b6d5bc3237ec113af1f732d1d5286aa155",
            },
            ModelFile {
                name: "decoder_joint-model.int8.onnx",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/0bbb45a3365852604aef28b538a8f066f4ccaa85/",
                    "decoder_joint-model.int8.onnx"
                ),
                size: 8_998_286,
                sha256: "a449f49acd68979d418651dd2dcb737cc0f1bf0225e009e29ee326354edbf7d3",
            },
            ModelFile {
                name: "nemo128.onnx",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/0bbb45a3365852604aef28b538a8f066f4ccaa85/",
                    "nemo128.onnx"
                ),
                size: 139_764,
                sha256: "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f",
            },
            ModelFile {
                name: "vocab.txt",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/0bbb45a3365852604aef28b538a8f066f4ccaa85/",
                    "vocab.txt"
                ),
                size: 9_384,
                sha256: "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d",
            },
            ModelFile {
                name: "config.json",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/0bbb45a3365852604aef28b538a8f066f4ccaa85/",
                    "config.json"
                ),
                size: 97,
                sha256: "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
            },
        ],
        gpu: gpu::PARAKEET_TDT,
        keeps_fillers: None,
        chunk_secs: Some(40),
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "parakeet-v2-full",
        name: "Parakeet v2 (full size)",
        maker: "NVIDIA",
        note: "English only. The uncompressed Parakeet v2 model, a 2.5 GB download.",
        kind: Kind::Parakeet,
        languages: "English",
        english_only: true,
        speed: 3,
        accuracy: 2,
        licence: licences::CC_BY_4,
        files: &[
            ModelFile {
                name: "encoder-model.onnx",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/0bbb45a3365852604aef28b538a8f066f4ccaa85/",
                    "encoder-model.onnx"
                ),
                size: 41_770_866,
                sha256: "3987bcd28175d829d12888a996a84e8f62a0e374d9ffd640662c1515adc679d3",
            },
            ModelFile {
                name: "encoder-model.onnx.data",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/0bbb45a3365852604aef28b538a8f066f4ccaa85/",
                    "encoder-model.onnx.data"
                ),
                size: 2_435_420_160,
                sha256: "4dab7362d4874d85965045b1e41b2d61dd2cc0fb25671a7f6b3dc47bf120cc41",
            },
            ModelFile {
                name: "decoder_joint-model.onnx",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/0bbb45a3365852604aef28b538a8f066f4ccaa85/",
                    "decoder_joint-model.onnx"
                ),
                size: 35_792_059,
                sha256: "cbb52a07bd70ab5b67f8439d4b3cd8704b18467b4430bcacb5adabe154b8d191",
            },
            ModelFile {
                name: "nemo128.onnx",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/0bbb45a3365852604aef28b538a8f066f4ccaa85/",
                    "nemo128.onnx"
                ),
                size: 139_764,
                sha256: "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f",
            },
            ModelFile {
                name: "vocab.txt",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/0bbb45a3365852604aef28b538a8f066f4ccaa85/",
                    "vocab.txt"
                ),
                size: 9_384,
                sha256: "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d",
            },
            ModelFile {
                name: "config.json",
                url: concat!(
                    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/0bbb45a3365852604aef28b538a8f066f4ccaa85/",
                    "config.json"
                ),
                size: 97,
                sha256: "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
            },
        ],
        gpu: gpu::PARAKEET_TDT,
        keeps_fillers: None,
        chunk_secs: Some(40),
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "parakeet-unified-en",
        name: "Parakeet Unified",
        maker: "NVIDIA",
        note: "English only. NVIDIA’s newer English model, with punctuation.",
        kind: Kind::ParakeetUnified,
        languages: "English",
        english_only: true,
        speed: 3,
        accuracy: 2,
        licence: licences::NVIDIA_OPEN_MODEL,
        files: &[
            ModelFile {
                name: "encoder.int8.onnx",
                url: concat!(
                    "https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx/resolve/09e9060322d99c5f070010724786e6ee090fd51d/",
                    "encoder.int8.onnx"
                ),
                size: 42_606_669,
                sha256: "c81adfab77634e00c1668a221a14f244c5fb3409e7c14eeebaf6ac963425910f",
            },
            ModelFile {
                name: "encoder.int8.onnx.data",
                url: concat!(
                    "https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx/resolve/09e9060322d99c5f070010724786e6ee090fd51d/",
                    "encoder.int8.onnx.data"
                ),
                size: 611_491_584,
                sha256: "3d54dd04646c15677bd2844a84df3770b12cc1ce183481f7b6e0def31c92114a",
            },
            ModelFile {
                name: "decoder_joint.int8.onnx",
                url: concat!(
                    "https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx/resolve/09e9060322d99c5f070010724786e6ee090fd51d/",
                    "decoder_joint.int8.onnx"
                ),
                size: 8_995_064,
                sha256: "7f76ad5f35035f25630075699c6c942a2c0c05ff42cb398f966f3c256d148e1e",
            },
            ModelFile {
                name: "tokenizer.model",
                url: concat!(
                    "https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx/resolve/09e9060322d99c5f070010724786e6ee090fd51d/",
                    "tokenizer.model"
                ),
                size: 251_056,
                sha256: "07d4e5a63840a53ab2d4d106d2874768143fb3fbdd47938b3910d2da05bfb0a9",
            },
        ],
        gpu: gpu::NONE,
        keeps_fillers: None,
        chunk_secs: Some(40),
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "parakeet-unified-en-full",
        name: "Parakeet Unified (full size)",
        maker: "NVIDIA",
        note: "English only. The uncompressed Parakeet Unified model, a 2.5 GB download, with punctuation.",
        kind: Kind::ParakeetUnified,
        languages: "English",
        english_only: true,
        speed: 3,
        accuracy: 2,
        licence: licences::NVIDIA_OPEN_MODEL,
        files: &[
            ModelFile {
                name: "encoder.onnx",
                url: concat!(
                    "https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx/resolve/09e9060322d99c5f070010724786e6ee090fd51d/",
                    "encoder.onnx"
                ),
                size: 41_901_475,
                sha256: "50083dd0b2af503b87ea8fcf9fd7d7dc15dda3130e4113017c8b37ed00418ba4",
            },
            ModelFile {
                name: "encoder.onnx.data",
                url: concat!(
                    "https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx/resolve/09e9060322d99c5f070010724786e6ee090fd51d/",
                    "encoder.onnx.data"
                ),
                size: 2_437_091_328,
                sha256: "c054b2932ee13aa39ed84982a399822f3193311fb6fd8e1e204566722614bd87",
            },
            ModelFile {
                name: "decoder_joint.onnx",
                url: concat!(
                    "https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx/resolve/09e9060322d99c5f070010724786e6ee090fd51d/",
                    "decoder_joint.onnx"
                ),
                size: 35_779_240,
                sha256: "64648c91935ea4819e9c31ea8a20f011d3b1c960be3861cb8bc74df0259cd998",
            },
            ModelFile {
                name: "tokenizer.model",
                url: concat!(
                    "https://huggingface.co/bobNight/parakeet-unified-en-0.6b-onnx/resolve/09e9060322d99c5f070010724786e6ee090fd51d/",
                    "tokenizer.model"
                ),
                size: 251_056,
                sha256: "07d4e5a63840a53ab2d4d106d2874768143fb3fbdd47938b3910d2da05bfb0a9",
            },
        ],
        gpu: gpu::NONE,
        keeps_fillers: None,
        chunk_secs: Some(40),
        word_times: true,
        llm: None,
    },
    ModelInfo {
        id: "nemotron-3.5",
        name: "Nemotron 3.5",
        maker: "NVIDIA",
        note: "NVIDIA’s newest, 35 languages. A larger download.",
        kind: Kind::Nemotron,
        languages: "35 languages",
        english_only: false,
        speed: 2,
        accuracy: 2,
        licence: licences::OPENMDW_1_1,
        files: &[
            ModelFile {
                name: "encoder.onnx",
                url: concat!(
                    "https://huggingface.co/altunenes/parakeet-rs/resolve/4d2a8bc71f5c896ec40faa59732e6716295edaf2/nemotron-3.5-asr-streaming-0.6b-onnx/",
                    "encoder.onnx"
                ),
                size: 42_164_972,
                sha256: "d569fbe78b48fbb04e169d324f5d25463838ceed7b5fc3bfe209872441979bd9",
            },
            ModelFile {
                name: "encoder.onnx.data",
                url: concat!(
                    "https://huggingface.co/altunenes/parakeet-rs/resolve/4d2a8bc71f5c896ec40faa59732e6716295edaf2/nemotron-3.5-asr-streaming-0.6b-onnx/",
                    "encoder.onnx.data"
                ),
                size: 2_454_405_120,
                sha256: "7584f85df76bc9ae6fbdfa53aa8d97b07a842525d1c501d536d77fd9e4f57ac7",
            },
            ModelFile {
                name: "decoder_joint.onnx",
                url: concat!(
                    "https://huggingface.co/altunenes/parakeet-rs/resolve/4d2a8bc71f5c896ec40faa59732e6716295edaf2/nemotron-3.5-asr-streaming-0.6b-onnx/",
                    "decoder_joint.onnx"
                ),
                size: 97_590_054,
                sha256: "634dfadf24cb4f73c2fae170b36611d68db48186426882cbc8f7e02ed9f2bb29",
            },
            ModelFile {
                name: "tokenizer.model",
                url: concat!(
                    "https://huggingface.co/altunenes/parakeet-rs/resolve/4d2a8bc71f5c896ec40faa59732e6716295edaf2/nemotron-3.5-asr-streaming-0.6b-onnx/",
                    "tokenizer.model"
                ),
                size: 406_554,
                sha256: "ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291",
            },
        ],
        gpu: gpu::NONE,
        keeps_fillers: None,
        chunk_secs: Some(40),
        word_times: true,
        llm: None,
    },
    // Parakeet Unified for 25 European languages (NVIDIA's nemotron-labs-unified-asr-0.6b, the ONNX export in the
    // parakeet-rs repository). The same files and load path as the English Unified model; the encoder takes a language.
    ModelInfo {
        id: "parakeet-unified-25",
        name: "Parakeet Unified (25 languages)",
        maker: "NVIDIA",
        note: "25 European languages, with punctuation. Takes the language, or detects it. About 2.5 GB. On NVIDIA’s 25-language test its average word error rate is 9.49, Parakeet v3’s 10.55.",
        kind: Kind::ParakeetUnified,
        languages: "25 European languages",
        english_only: false,
        speed: 3,
        accuracy: 2,
        licence: licences::OPENMDW_1_1,
        files: &[
            ModelFile {
                name: "encoder.onnx",
                url: concat!(
                    "https://huggingface.co/altunenes/parakeet-rs/resolve/664cc742197053fa11afd12f5100749fb8000181/nemotron-labs-unified-asr-0.6b/",
                    "encoder.onnx"
                ),
                size: 4_176_628,
                sha256: "3e985eef88699f66f34ff67decaa0136e3915f58d79dcc209240a6a8f9e23cf9",
            },
            ModelFile {
                name: "encoder.onnx.data",
                url: concat!(
                    "https://huggingface.co/altunenes/parakeet-rs/resolve/664cc742197053fa11afd12f5100749fb8000181/nemotron-labs-unified-asr-0.6b/",
                    "encoder.onnx.data"
                ),
                size: 2_495_885_312,
                sha256: "ba6d0f46f2a4572c86503cfff437d935a3e0d9d759add4681f2f41984ae35510",
            },
            ModelFile {
                name: "decoder_joint.onnx",
                url: concat!(
                    "https://huggingface.co/altunenes/parakeet-rs/resolve/664cc742197053fa11afd12f5100749fb8000181/nemotron-labs-unified-asr-0.6b/",
                    "decoder_joint.onnx"
                ),
                size: 51_578_226,
                sha256: "da1e234ca1908dcce10e01f3cd5a4bab28583fe558d39a980a2f606240fd614c",
            },
            ModelFile {
                name: "tokenizer.model",
                url: concat!(
                    "https://huggingface.co/altunenes/parakeet-rs/resolve/664cc742197053fa11afd12f5100749fb8000181/nemotron-labs-unified-asr-0.6b/",
                    "tokenizer.model"
                ),
                size: 296_524,
                sha256: "f910aa78502daf04410654c3211180e6f83aafd6ef23a38739f25b6568421ba7",
            },
        ],
        gpu: gpu::NONE,
        keeps_fillers: None,
        chunk_secs: Some(40),
        word_times: true,
        llm: None,
    },
    // Cohere Transcribe 03-2026 (2B parameters, the int8 ONNX export by onnx-community). Text only: no word times. The
    // external-data files sit flat next to the .onnx files, which name them by relative path.
    ModelInfo {
        id: "cohere-transcribe",
        name: "Cohere Transcribe",
        maker: "Cohere",
        note: "14 languages. Gives the text of each chunk, with no word times. Trained on clips of up to 35 seconds. About 3.1 GB. It can write text for silence or noise, so give it audio that has speech in it.",
        kind: Kind::Cohere,
        languages: "14 languages",
        english_only: false,
        speed: 1,
        accuracy: 3,
        licence: licences::APACHE_2,
        files: &[
            ModelFile {
                name: "encoder_model_quantized.onnx",
                url: concat!(
                    "https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX/resolve/31b1c6211c9000d76b077ddd23b74c9090badeba/",
                    "onnx/encoder_model_quantized.onnx"
                ),
                size: 1_420_203,
                sha256: "9e7e59a14821e5a08c71b0d849a956b6375007a5f0be795c448499b0328d7923",
            },
            ModelFile {
                name: "encoder_model_quantized.onnx_data",
                url: concat!(
                    "https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX/resolve/31b1c6211c9000d76b077ddd23b74c9090badeba/",
                    "onnx/encoder_model_quantized.onnx_data"
                ),
                size: 2_096_573_952,
                sha256: "8733a14ed4e365cacaa8f1af8764e31b7f728fc4db79d8e8d4836e59d396ccea",
            },
            ModelFile {
                name: "encoder_model_quantized.onnx_data_1",
                url: concat!(
                    "https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX/resolve/31b1c6211c9000d76b077ddd23b74c9090badeba/",
                    "onnx/encoder_model_quantized.onnx_data_1"
                ),
                size: 774_430_720,
                sha256: "a86b005d5c77583389f0f0dcf558ebe4ed615fad96f3473303eaaf7d9990a7ae",
            },
            ModelFile {
                name: "decoder_model_merged_quantized.onnx",
                url: concat!(
                    "https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX/resolve/31b1c6211c9000d76b077ddd23b74c9090badeba/",
                    "onnx/decoder_model_merged_quantized.onnx"
                ),
                size: 196_039,
                sha256: "e9916f9a7ec732fd2181157e7899950d895ecf21867cea1aa0e8fff13a85fd53",
            },
            ModelFile {
                name: "decoder_model_merged_quantized.onnx_data",
                url: concat!(
                    "https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX/resolve/31b1c6211c9000d76b077ddd23b74c9090badeba/",
                    "onnx/decoder_model_merged_quantized.onnx_data"
                ),
                size: 195_903_488,
                sha256: "84d949b4b89d2b2b4e9ba201b28344f2c517fe03bd50a6b92be1756942f9d40b",
            },
            ModelFile {
                name: "tokenizer.json",
                url: concat!(
                    "https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX/resolve/31b1c6211c9000d76b077ddd23b74c9090badeba/",
                    "tokenizer.json"
                ),
                size: 1_152_395,
                sha256: "e263c0ba13be0f0803705b002756908f84efc6e75a4c273231a01c4371908a2b",
            },
        ],
        gpu: gpu::NONE,
        keeps_fillers: None,
        chunk_secs: Some(35),
        word_times: false,
        llm: None,
    },
    // Silero voice activity detection, used with every model. Downloaded with the first model.
    ModelInfo {
        id: "vad",
        name: "Silero VAD",
        maker: "Silero",
        note: "",
        kind: Kind::Vad,
        languages: "",
        english_only: false,
        speed: 3,
        accuracy: 3,
        licence: licences::MIT,
        files: &[ModelFile {
            name: "ggml-silero-v6.2.0.bin",
            url: "https://huggingface.co/ggml-org/whisper-vad/resolve/9ffd54a1e1ee413ddf265af9913beaf518d1639b/ggml-silero-v6.2.0.bin",
            size: 885_098,
            sha256: "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987",
        }],
        gpu: gpu::NONE,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: false,
        llm: None,
    },
    // Speaker separation (named speakers): NVIDIA Nemotron 3 Diarization, ONNX export for parakeet-rs.
    ModelInfo {
        id: "diarization",
        name: "Speaker separation",
        maker: "NVIDIA",
        note: "Tells the other people on a call apart: Speaker 1, 2, 3. Up to 8 people.",
        kind: Kind::Diarization,
        languages: "Any language",
        english_only: false,
        speed: 3,
        accuracy: 2,
        licence: licences::OPENMDW_1_1,
        files: &[ModelFile {
            name: "nemotron3_diar_v3.onnx",
            url: "https://huggingface.co/altunenes/parakeet-rs/resolve/4d2a8bc71f5c896ec40faa59732e6716295edaf2/nemotron-3-diarization/nemotron3_diar_v3.onnx",
            size: 400_506_656,
            sha256: "915e4fa23b0192ed9fadeb1cdd26847df986d50c92012d177be28d0343bbe03a",
        }],
        gpu: gpu::NONE,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: false,
        llm: None,
    },
    // Multitalker Parakeet TDT: a community fine-tune (altunenes, in the parakeet-rs repository) of NVIDIA's
    // parakeet-tdt-0.6b-v3. Pinned to the commit before the folder gained a `speaker_gate` input that parakeet-rs 0.4.4
    // cannot feed. It needs the `diarization` model too. It cuts the audio itself (about 20 s at a time), so it takes any length.
    ModelInfo {
        id: "multitalker-tdt",
        name: "Multitalker Parakeet TDT",
        maker: "altunenes",
        note: "Who said what when people talk over each other, in 25 European languages. A fine-tune of NVIDIA’s Parakeet TDT v3 by a community author, not an NVIDIA release. Needs the speaker separation model as well. About 2.6 GB.",
        kind: Kind::Multitalker,
        languages: "25 European languages",
        english_only: false,
        speed: 2,
        accuracy: 2,
        licence: licences::CC_BY_4,
        files: &[
            ModelFile {
                name: "encoder-model.onnx",
                url: concat!(
                    "https://huggingface.co/altunenes/parakeet-rs/resolve/7f997807e4dcfae115c260a4a537a1d3bfb2e738/multitalker-tdt-0.6b-v3/",
                    "encoder-model.onnx"
                ),
                size: 3_916_579,
                sha256: "12e70570b3c777271e3a981a5b858b0afebc9d20a2b847df065a1a6d2c12f869",
            },
            ModelFile {
                name: "encoder-model.onnx.data",
                url: concat!(
                    "https://huggingface.co/altunenes/parakeet-rs/resolve/7f997807e4dcfae115c260a4a537a1d3bfb2e738/multitalker-tdt-0.6b-v3/",
                    "encoder-model.onnx.data"
                ),
                size: 2_505_949_184,
                sha256: "0964d537682c69c219a40d8bd87b286a5173c43ed9c947e134633f57dff0b419",
            },
            ModelFile {
                name: "decoder_joint-model.onnx",
                url: concat!(
                    "https://huggingface.co/altunenes/parakeet-rs/resolve/7f997807e4dcfae115c260a4a537a1d3bfb2e738/multitalker-tdt-0.6b-v3/",
                    "decoder_joint-model.onnx"
                ),
                size: 72_584_074,
                sha256: "9ce45c12b84c88366e259d51701eb6061bb40a8f3caee486fb7c3fe52ac131ac",
            },
            ModelFile {
                name: "vocab.txt",
                url: concat!(
                    "https://huggingface.co/altunenes/parakeet-rs/resolve/7f997807e4dcfae115c260a4a537a1d3bfb2e738/multitalker-tdt-0.6b-v3/",
                    "vocab.txt"
                ),
                size: 93_939,
                sha256: "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
            },
        ],
        gpu: gpu::NONE,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: true,
        llm: None,
    },
    // Search by meaning: Qwen3-Embedding 0.6B, 100+ languages.
    ModelInfo {
        id: "search",
        name: "Search by meaning",
        maker: "Qwen",
        note: "Finds passages that mean the same as the question even when the words differ, in 100+ languages.",
        kind: Kind::Embedding,
        languages: "100+ languages",
        english_only: false,
        speed: 3,
        accuracy: 2,
        licence: licences::APACHE_2,
        files: &[ModelFile {
            name: "Qwen3-Embedding-0.6B-Q8_0.gguf",
            url: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-Q8_0.gguf",
            size: 639_150_592,
            sha256: "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
        }],
        gpu: gpu::NONE,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: false,
        llm: None,
    },
    ModelInfo {
        id: "qwen3.5-4b",
        name: "Qwen3.5 4B",
        maker: "Qwen",
        note: "Small and quick enough for a laptop.",
        kind: Kind::Llm,
        languages: "Many languages",
        english_only: false,
        speed: 3,
        accuracy: 2,
        licence: licences::APACHE_2,
        files: &[ModelFile {
            name: "Qwen3.5-4B-Q4_K_M.gguf",
            url: "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
            size: 2_740_937_888,
            sha256: "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: false,
        // Measured with the helper on the processor, 18 GB M-series Mac, 2026-10-29 (table in SOURCES.md).
        // Rule: context_8gb is the largest tested n_ctx with a peak size of at most 5 GiB, context_16gb the largest
        // with at most 11 GiB (4096 when even 8192 does not fit), both capped at context_max. The sweep went up to
        // 131072 tokens.
        llm: Some(LlmInfo {
            context_max: 262144,
            context_8gb: 65536,
            context_16gb: 131072,
            thinks: true,
        }),
    },
    ModelInfo {
        id: "qwen3.5-9b",
        name: "Qwen3.5 9B",
        maker: "Qwen",
        note: "Larger and a little more thorough; needs 16 GB of memory and takes about twice as long. On 8 GB of memory only a context of 4096 tokens fits.",
        kind: Kind::Llm,
        languages: "Many languages",
        english_only: false,
        speed: 2,
        accuracy: 3,
        licence: licences::APACHE_2,
        files: &[ModelFile {
            name: "Qwen3.5-9B-Q4_K_M.gguf",
            url: "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf",
            size: 5_680_522_464,
            sha256: "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: false,
        // Measured with the helper on the processor, 18 GB M-series Mac, 2026-10-29 (table in SOURCES.md).
        // Rule: context_8gb is the largest tested n_ctx with a peak size of at most 5 GiB, context_16gb the largest
        // with at most 11 GiB (4096 when even 8192 does not fit), both capped at context_max. The sweep went up to
        // 131072 tokens.
        llm: Some(LlmInfo {
            context_max: 262144,
            context_8gb: 4096,
            context_16gb: 131072,
            thinks: true,
        }),
    },
    ModelInfo {
        id: "granite-4.2-3b",
        name: "Granite 4.2 3B",
        maker: "IBM",
        note: "A small model of 3 billion parameters; the download is about 2.2 GB.",
        kind: Kind::Llm,
        languages: "Many languages",
        english_only: false,
        speed: 3,
        accuracy: 2,
        licence: licences::APACHE_2,
        files: &[ModelFile {
            name: "granite-4.2-3b-Q4_K_M.gguf",
            url: "https://huggingface.co/ibm-granite/granite-4.2-3b-GGUF/resolve/c40945d71cd90f249a56985e8155551a9188dc30/granite-4.2-3b-Q4_K_M.gguf",
            size: 2_244_011_552,
            sha256: "e0406663965846ae22a403456eb826ccce5f450840491f71952f18a7cb78e7d5",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: false,
        // Measured with the helper on the processor, 18 GB M-series Mac, 2026-10-29 (table in SOURCES.md).
        // Rule: context_8gb is the largest tested n_ctx with a peak size of at most 5 GiB, context_16gb the largest
        // with at most 11 GiB (4096 when even 8192 does not fit), both capped at context_max. The sweep went up to
        // 131072 tokens.
        llm: Some(LlmInfo {
            context_max: 131072,
            context_8gb: 32768,
            context_16gb: 65536,
            thinks: true,
        }),
    },
    ModelInfo {
        id: "gemma-4-e4b",
        name: "Gemma 4 E4B",
        maker: "Google",
        note: "Google’s small Gemma 4 model, 4.5 billion effective parameters, trained for 4-bit use; the download is about 5.2 GB and it needs 16 GB of memory for a long context: on 8 GB of memory only a context of 4096 tokens fits. Text only here.",
        kind: Kind::Llm,
        languages: "Many languages",
        english_only: false,
        speed: 3,
        accuracy: 2,
        licence: licences::GEMMA_4,
        files: &[ModelFile {
            name: "gemma-4-E4B_q4_0-it.gguf",
            url: concat!(
                "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/4b4a2c1d584be7264f87aac328a1bc739ce81b6c/",
                "gemma-4-E4B_q4_0-it.gguf"
            ),
            size: 5_154_941_280,
            sha256: "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee",
        }],
        gpu: gpu::WHISPER_AND_LLM,
        keeps_fillers: None,
        chunk_secs: None,
        word_times: false,
        // Measured with the helper on the processor, Windows 11 PC with 32 GB, 2026-11-16 (table in SOURCES.md), by the same
        // rule as the entries above, with the peak working set where the Mac used the peak resident size: context_8gb
        // is the largest tested n_ctx with a peak of at most 5 GiB (4096 when even 8192 does not fit: 5.27 GiB at 8192,
        // and 5.05 GiB at 4096, the weights alone are 4.8 GiB), context_16gb the largest with at most 11 GiB (8.34 GiB
        // at 65536, 11.84 GiB at 131072). context_max is read from the GGUF header. `thinks` is false: the template's
        // thinking is a channel it opens only when asked, and has no `<think>` block, so the helper never has Gemma think.
        llm: Some(LlmInfo {
            context_max: 131072,
            context_8gb: 4096,
            context_16gb: 65536,
            thinks: false,
        }),
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    const PARAKEET_V3: &str = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce/";
    const WHISPER: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/";

    #[test]
    fn catalog_is_consistent() {
        assert!(!all().is_empty());
        for m in all() {
            assert!(!m.files.is_empty(), "{}", m.id);
            for f in m.files {
                assert_eq!(f.sha256.len(), 64, "{}", f.name);
                assert!(
                    f.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                    "{}",
                    f.name
                );
                assert!(f.url.ends_with(f.name), "{}", f.url);
                assert!(f.url.starts_with("https://huggingface.co/"), "{}", f.url);
            }
        }
        let ids: std::collections::HashSet<_> = all().iter().map(|m| m.id).collect();
        assert_eq!(ids.len(), all().len(), "duplicate model id");
        assert!(find("whisper-turbo").is_some());
        assert!(find("qwen3.5-4b").is_some());
        assert!(find("nope").is_none());
        let v3 = find("parakeet-v3").unwrap();
        assert!(v3.files.iter().all(|f| f.url.starts_with(PARAKEET_V3)));
        for m in all().iter().filter(|m| m.kind == Kind::Whisper) {
            assert!(
                m.files.iter().all(|f| f.url.starts_with(WHISPER)),
                "{}",
                m.id
            );
        }
        for m in local_ai() {
            assert_eq!(m.files.len(), 1, "{}: one GGUF file", m.id);
            let f = &m.files[0];
            assert!(f.name.ends_with(".gguf"), "{}", f.name);
            assert!(
                f.url.contains("/resolve/") && !f.url.contains("/resolve/main/"),
                "{}: pinned to a commit",
                m.id
            );
            assert!(f.size > 1_000_000_000, "{}", m.id);
        }
        assert_eq!(
            find("qwen3.5-4b").map(|m| m.files[0].name),
            Some("Qwen3.5-4B-Q4_K_M.gguf")
        );
    }

    #[test]
    fn parlatos_variants_are_in_the_catalog() {
        for id in [
            "whisper-tiny-en",
            "whisper-base-en",
            "whisper-small-en",
            "whisper-medium-en",
            "whisper-large-v2-full",
            "whisper-large-v3-full",
            "whisper-turbo-full",
            "parakeet-v2-full",
            "parakeet-v3-full",
            "parakeet-unified-en-full",
        ] {
            assert!(find(id).is_some(), "{id}");
        }
        for id in [
            "whisper-tiny-en",
            "whisper-base-en",
            "whisper-small-en",
            "whisper-medium-en",
            "parakeet-v2-full",
            "parakeet-unified-en-full",
        ] {
            assert!(find(id).unwrap().english_only, "{id}");
        }
        assert!(!find("whisper-large-v3-full").unwrap().english_only);
        assert_eq!(find("whisper-turbo-full").unwrap().size(), 1_624_555_275);
        assert_eq!(find("parakeet-v3-full").unwrap().files.len(), 6);
        assert_eq!(find("parakeet-unified-en-full").unwrap().files.len(), 4);
    }

    /// The licence is the one on the model's own card (NVIDIA's), checked against Hugging Face on 2026-10-31: the
    /// community ONNX repository we download from says CC-BY-4.0, but `nvidia/parakeet-unified-en-0.6b` says the NVIDIA
    /// Open Model License.
    #[test]
    fn parakeet_unified_is_under_the_nvidia_open_model_license() {
        for id in ["parakeet-unified-en", "parakeet-unified-en-full"] {
            assert_eq!(
                find(id).unwrap().licence.spdx,
                "NVIDIA-Open-Model-License",
                "{id}"
            );
        }
        assert_eq!(find("parakeet-v3").unwrap().licence.spdx, "CC-BY-4.0");
    }

    #[test]
    fn every_entry_has_the_fields_the_apps_show() {
        for m in all() {
            assert!(!m.name.is_empty(), "{}: name", m.id);
            assert!(!m.maker.is_empty(), "{}: maker", m.id);
            // Silero has no note and no languages to show; every other entry has both.
            if m.kind != Kind::Vad {
                assert!(!m.note.is_empty(), "{}: note", m.id);
                assert!(!m.languages.is_empty(), "{}: languages", m.id);
            }
            assert!(
                !m.licence.spdx.is_empty() && m.licence.url.starts_with("https://"),
                "{}: licence",
                m.id
            );
            assert!(
                (1..=3).contains(&m.speed) && (1..=3).contains(&m.accuracy),
                "{}: speed and accuracy are 1 to 3",
                m.id
            );
            assert!(!m.files.is_empty(), "{}", m.id);
            for f in m.files {
                assert_eq!(f.sha256.len(), 64, "{}", f.name);
                assert!(
                    f.sha256
                        .chars()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                    "{}: sha256 is lower-case hex",
                    f.name
                );
                assert!(f.size > 0, "{}", f.name);
                assert!(
                    f.url.ends_with(&format!("/{}", f.name)),
                    "{}: url ends with the file name",
                    f.url
                );
                let (_, rest) = f
                    .url
                    .split_once("/resolve/")
                    .unwrap_or_else(|| panic!("{} is not a Hugging Face resolve url", f.url));
                let rev = rest.split('/').next().unwrap();
                assert!(
                    rev.len() == 40 && rev.chars().all(|c| c.is_ascii_hexdigit()),
                    "{} is not pinned to a commit",
                    f.url
                );
            }
        }
    }

    #[test]
    fn ids_are_unique_and_kinds_agree_with_fields() {
        let mut seen = std::collections::HashSet::new();
        for m in all() {
            assert!(seen.insert(m.id), "duplicate id {}", m.id);
            assert_eq!(m.kind == Kind::Llm, m.llm.is_some(), "{}", m.id);
            match m.kind {
                Kind::Parakeet | Kind::ParakeetUnified | Kind::Nemotron => {
                    assert_eq!(m.chunk_secs, Some(40), "{}", m.id)
                }
                // Trained on clips of up to 35 s.
                Kind::Cohere => assert_eq!(m.chunk_secs, Some(35), "{}", m.id),
                // Whisper takes any length, and Multitalker cuts the audio itself.
                Kind::Whisper | Kind::Multitalker => assert_eq!(m.chunk_secs, None, "{}", m.id),
                _ => {}
            }
        }
    }

    /// `word_times` is a speech field: true for every entry that gives words with times, false for Cohere (one piece
    /// of text per chunk) and false for everything that is not a speech model.
    #[test]
    fn word_times_follow_the_kind() {
        for m in all() {
            let want = matches!(
                m.kind,
                Kind::Whisper
                    | Kind::Parakeet
                    | Kind::ParakeetUnified
                    | Kind::Nemotron
                    | Kind::Multitalker
            );
            assert_eq!(m.word_times, want, "{}", m.id);
        }
        assert!(!find("cohere-transcribe").unwrap().word_times);
        assert!(find("multitalker-tdt").unwrap().word_times);
        assert!(find("whisper-turbo").unwrap().word_times);
        // Every speech entry that is not Cohere has word times.
        for m in speech().filter(|m| m.kind != Kind::Cohere) {
            assert!(m.word_times, "{}", m.id);
        }
    }

    #[test]
    fn multitalker_is_not_a_plain_speech_model() {
        assert!(Kind::Cohere.is_speech());
        assert!(!Kind::Multitalker.is_speech());
        assert!(speech().all(|m| m.id != "multitalker-tdt"));
        assert!(speech().any(|m| m.id == "cohere-transcribe"));
    }

    const U25: &str = "https://huggingface.co/altunenes/parakeet-rs/resolve/664cc742197053fa11afd12f5100749fb8000181/nemotron-labs-unified-asr-0.6b/";
    const MULTITALKER: &str = "https://huggingface.co/altunenes/parakeet-rs/resolve/7f997807e4dcfae115c260a4a537a1d3bfb2e738/multitalker-tdt-0.6b-v3/";
    const COHERE: &str = "https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX/resolve/31b1c6211c9000d76b077ddd23b74c9090badeba/";
    const GEMMA: &str = "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/4b4a2c1d584be7264f87aac328a1bc739ce81b6c/";

    /// The v0.4.0 entries: each pinned to the commit it was checked against (`tools/verify-catalog.py` checks the sizes and
    /// hashes against Hugging Face), with the file lists and sizes the engine loads.
    #[test]
    fn the_v0_4_0_entries_are_pinned_and_complete() {
        let u25 = find("parakeet-unified-25").unwrap();
        assert_eq!(u25.kind, Kind::ParakeetUnified);
        assert!(!u25.english_only);
        assert_eq!(u25.licence.spdx, "OpenMDW-1.1");
        assert_eq!(u25.chunk_secs, Some(40));
        assert_eq!(u25.size(), 2_551_936_690);
        assert!(u25.files.iter().all(|f| f.url.starts_with(U25)));
        let names: Vec<_> = u25.files.iter().map(|f| f.name).collect();
        assert_eq!(
            names,
            [
                "encoder.onnx",
                "encoder.onnx.data",
                "decoder_joint.onnx",
                "tokenizer.model"
            ]
        );

        let cohere = find("cohere-transcribe").unwrap();
        assert_eq!(cohere.kind, Kind::Cohere);
        assert_eq!(cohere.licence.spdx, "Apache-2.0");
        assert!(cohere.files.iter().all(|f| f.url.starts_with(COHERE)));
        assert_eq!(cohere.files.len(), 6);
        assert_eq!(cohere.size(), 3_069_676_797);
        // The store is flat: a name is the bare file name, the url keeps the folder.
        assert!(cohere.files.iter().all(|f| !f.name.contains('/')));
        assert!(cohere.files.iter().any(|f| f.name == "tokenizer.json"));

        let multi = find("multitalker-tdt").unwrap();
        assert_eq!(multi.kind, Kind::Multitalker);
        assert_eq!(multi.licence.spdx, "CC-BY-4.0");
        assert!(multi.files.iter().all(|f| f.url.starts_with(MULTITALKER)));
        assert_eq!(multi.files.len(), 4);
        assert_eq!(multi.size(), 2_582_543_776);
        // The commit before the folder gained a `speaker_gate` input.
        assert!(!MULTITALKER.contains("/resolve/main/"));
        // The speaker separation model it needs is in the catalog too.
        assert!(find("diarization").is_some());

        let gemma = find("gemma-4-e4b").unwrap();
        assert_eq!(gemma.kind, Kind::Llm);
        assert_eq!(gemma.licence.spdx, "Apache-2.0");
        assert_eq!(
            gemma.licence.url,
            "https://ai.google.dev/gemma/docs/gemma_4_license"
        );
        assert_eq!(
            gemma.files.len(),
            1,
            "no mmproj: the model is text only here"
        );
        assert!(gemma.files[0].url.starts_with(GEMMA));
        assert_eq!(gemma.files[0].size, 5_154_941_280);
        assert!(gemma.llm.is_some());
        assert!(gemma.gpu.mac_metal && gemma.gpu.windows_vulkan);
    }

    /// A catalog note names no app and recommends nothing.
    #[test]
    fn notes_name_no_app() {
        for m in all() {
            for app in ["Talaan", "Ensayo", "Parlato"] {
                assert!(!m.note.contains(app), "{}", m.id);
            }
        }
    }

    #[test]
    fn only_measured_models_have_a_fillers_value() {
        // Measured on a recording with fillers (crates/stt/tests/real_fillers.rs, docs/bench-2026-10-29.md); the
        // others stay None until someone measures them.
        let measured = ["whisper-turbo", "parakeet-ultra"];
        for m in all() {
            if measured.contains(&m.id) {
                assert_eq!(m.keeps_fillers, Some(true), "{}", m.id);
            } else {
                assert_eq!(m.keeps_fillers, None, "{} was not measured", m.id);
            }
        }
    }

    #[test]
    fn local_ai_numbers_are_measured_not_placeholders() {
        let ids: Vec<_> = local_ai().map(|m| m.id).collect();
        assert_eq!(
            ids,
            ["qwen3.5-4b", "qwen3.5-9b", "granite-4.2-3b", "gemma-4-e4b"]
        );
        for m in local_ai() {
            let l = m.llm.unwrap();
            assert!(l.context_8gb >= 4096, "{}", m.id);
            assert!(
                l.context_8gb <= l.context_16gb && l.context_16gb <= l.context_max,
                "{}",
                m.id
            );
        }
    }

    #[test]
    fn the_four_dropped_local_models_are_not_offered() {
        for id in [
            "qwen2.5-3b-instruct-q4",
            "llama-3.2-3b-instruct-q4",
            "gemma-2-2b-it-q4",
            "phi-3.5-mini-instruct-q4",
        ] {
            assert!(find(id).is_none(), "{id}");
        }
        for m in all() {
            for f in m.files {
                for dropped in ["Llama-3.2", "gemma-2", "Phi-3.5", "Qwen2.5"] {
                    assert!(
                        !f.url.contains(dropped),
                        "{}: {} is a dropped model",
                        m.id,
                        dropped
                    );
                }
            }
        }
    }

    #[test]
    fn every_speech_entry_says_where_it_runs() {
        for m in speech().filter(|m| m.kind == Kind::Whisper) {
            assert!(
                m.gpu.mac_metal && m.gpu.windows_vulkan && !m.gpu.windows_directml,
                "{}",
                m.id
            );
        }
        assert!(find("vad").unwrap().gpu == gpu::NONE);
        assert!(find("parakeet-ultra").unwrap().gpu.windows_directml);
    }
}
