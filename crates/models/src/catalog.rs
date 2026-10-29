//! The model catalog: every model the engine can fetch (speech, voice detection, speaker separation, search, local
//! AI), with its files, sizes, hashes, licence and what runs it. The catalog never recommends; which model an app
//! offers first is the app's business.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Whisper (whisper.cpp ggml files).
    Whisper,
    /// Parakeet TDT (v2, v3, Ultra).
    Parakeet,
    /// Parakeet Unified (English).
    ParakeetUnified,
    /// Nemotron ASR (multilingual).
    Nemotron,
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
    /// A model that turns speech into text (the ones an app offers as speech models).
    pub fn is_speech(self) -> bool {
        matches!(
            self,
            Kind::Whisper | Kind::Parakeet | Kind::ParakeetUnified | Kind::Nemotron
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
    /// Speech: whether it keeps "um" and "uh" when asked to. None until measured (spec 9.4).
    pub keeps_fillers: Option<bool>,
    /// Speech: the longest chunk to send (seconds). None: any length (Whisper).
    pub chunk_secs: Option<u32>,
    /// Local AI only.
    pub llm: Option<LlmInfo>,
    /// Folders, relative to another app's data folder, that may already hold these exact files (step 4 replaces
    /// this with the shared folder).
    pub reuse_dirs: &'static [&'static str],
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
        keeps_fillers: None,
        chunk_secs: None,
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["ParakeetModels/parakeet-tdt-0.6b-v3-int8"],
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
        llm: None,
        reuse_dirs: &["ParakeetModels/parakeet-tdt-0.6b-v3"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        keeps_fillers: None,
        chunk_secs: Some(40),
        llm: None,
        reuse_dirs: &["ParakeetModels/parakeet-ultra-0.6b-int8"],
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
        llm: None,
        reuse_dirs: &["ParakeetModels/parakeet-tdt-0.6b-v2-int8"],
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
        llm: None,
        reuse_dirs: &["ParakeetModels/parakeet-tdt-0.6b-v2"],
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
        licence: licences::CC_BY_4,
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
        llm: None,
        reuse_dirs: &["ParakeetModels/parakeet-unified-en-0.6b-int8"],
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
        licence: licences::CC_BY_4,
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
        llm: None,
        reuse_dirs: &["ParakeetModels/parakeet-unified-en-0.6b"],
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
        llm: None,
        reuse_dirs: &["ParakeetModels/nemotron-3.5-asr-streaming-0.6b"],
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
        llm: None,
        reuse_dirs: &["Models"],
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
        llm: None,
        reuse_dirs: &[],
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
        llm: None,
        reuse_dirs: &[],
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
        reuse_dirs: &[],
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
        reuse_dirs: &[],
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
        reuse_dirs: &[],
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
}
