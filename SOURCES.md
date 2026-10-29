# Where the code came from

| Crate | Source |
| --- | --- |
| protocol | Talaan `src-tauri/llm-protocol` @ 69f0e9101a5737627594f945f15000686c398422, merged with Ensayo `src-tauri/llm-protocol` @ f294fc48cf460b04b055698507297db2c8b1f7c9 |
| llm | Talaan `src-tauri/llm` @ 69f0e9101a5737627594f945f15000686c398422, merged with Ensayo `src-tauri/llm` @ f294fc48cf460b04b055698507297db2c8b1f7c9 |
| stt | Talaan `src-tauri/stt` @ 69f0e9101a5737627594f945f15000686c398422 |
| Whisper settings, `Timing::Segments` (`whisper_engine.rs`) | Talaan `src-tauri/src/transcriber/engine.rs`, `worker.rs` and `src-tauri/stt/src/lib.rs` @ 69f0e9101a5737627594f945f15000686c398422 (origin/main; the local checkout is on the branch feat/parlato-batch, so the hash is origin/main's). MIT |
| Whisper settings, `Timing::Words`, short chunk padding, early-stop resume (`whisper_engine.rs`) | Ensayo `src-tauri/src/speech/whisper.rs`, `transcribe.rs`, `loaded.rs` and `engine.rs` @ 07f797dc4643fd03f311fd6c1f591d75527d2a9d (the checkout's HEAD on main, clean; 14 commits ahead of origin/main f294fc4, and of these files only `loaded.rs` differs from origin/main). MIT |
| `words::clean` (`words.rs`) | Ensayo `src-tauri/src/speech/words.rs` @ 07f797dc4643fd03f311fd6c1f591d75527d2a9d. MIT |
| Parakeet, Parakeet Unified and Nemotron engine (`parakeet_engine.rs`) | Talaan `src-tauri/src/transcriber/engine.rs` @ 69f0e9101a5737627594f945f15000686c398422 (origin/main, as above); Ensayo `src-tauri/src/speech/engine.rs` and `transcribe.rs` @ 07f797dc4643fd03f311fd6c1f591d75527d2a9d. MIT |
| Voice detection (`vad.rs`) | Talaan `src-tauri/src/transcriber/vad.rs` @ 69f0e9101a5737627594f945f15000686c398422 (origin/main, as above). MIT |
| Speaker separation (`diarize.rs`) | Talaan `src-tauri/src/transcriber/diarize.rs` @ 69f0e9101a5737627594f945f15000686c398422 (origin/main, as above). MIT. The warm-up and chunk sizes are Talaan's; the workaround for parakeet-rs 0.3.8's non-contiguous layout failure is new |
| Speech dependency pins (`crates/stt/Cargo.toml`) | whisper-rs 0.16.0, parakeet-rs 0.3.8 (feature `sortformer`), ort =2.0.0-rc.13 (the pair both apps link; parakeet-rs 0.3.8 does not build against rc.14), all from crates.io. Upstream docs followed: whisper.cpp (initial prompt and filler prompt, VAD), parakeet-rs (Parakeet, Nemotron prompts, Sortformer), ONNX Runtime (DirectML provider) |
| Nothing from Parlato | No Parlato code was read or used for the speech engines. Parlato's needs came from the spec and the Notion page, as requirements (thread counts it measured are quoted as numbers in `threads.rs`) |
| models | Talaan `src-tauri/src/transcriber/models.rs` and `summarizer/models.rs` @ 69f0e9101a5737627594f945f15000686c398422, Ensayo `src-tauri/src/ai/local_models.rs` @ f294fc48cf460b04b055698507297db2c8b1f7c9 |
| downloader (`download.rs`) | Talaan `src-tauri/src/transcriber/download.rs` @ 69f0e9101a5737627594f945f15000686c398422 (MIT); blocking client, error kinds and whole-model progress are new |
| legacy folders (`legacy.rs`) | Talaan `src-tauri/src/transcriber/models.rs` @ 69f0e9101a5737627594f945f15000686c398422; Ensayo `src-tauri/src/paths.rs` and `src-tauri/src/ai/local_models.rs` @ f294fc48cf460b04b055698507297db2c8b1f7c9; Parlato's layout from the Notion table the Parlato chat wrote on 2026-10-29, kept in `docs/data/parlato-0.14.1-model-layout.md` (no Parlato code) |
| llm tests/schemas | Ensayo `prompts/schemas` @ 07f797d and Talaan `src-tauri/src/summarizer/prompt.rs` SCHEMA @ 69f0e91 (origin/main) |
| catalog | Hugging Face: ggerganov/whisper.cpp @ 5359861c739e955e79d9a303bcbc70fb988958b1, istupakov/parakeet-tdt-0.6b-v2-onnx @ 0bbb45a3365852604aef28b538a8f066f4ccaa85, istupakov/parakeet-tdt-0.6b-v3-onnx @ 8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce, bobNight/parakeet-unified-en-0.6b-onnx @ 09e9060322d99c5f070010724786e6ee090fd51d, Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx @ 3646b2f1a516a2310810902bfaa67515f3c89d38, altunenes/parakeet-rs @ 4d2a8bc71f5c896ec40faa59732e6716295edaf2, ggml-org/whisper-vad @ 9ffd54a1e1ee413ddf265af9913beaf518d1639b, Qwen/Qwen3-Embedding-0.6B-GGUF @ 370f27d7550e0def9b39c1f16d3fbaa13aa67728, unsloth/Qwen3.5-4B-GGUF @ e87f176479d0855a907a41277aca2f8ee7a09523, unsloth/Qwen3.5-9B-GGUF @ 3885219b6810b007914f3a7950a8d1b469d598a5, ibm-granite/granite-4.2-3b-GGUF @ c40945d71cd90f249a56985e8155551a9188dc30; verified on 2026-10-29 with `tools/verify-catalog.py` |

The history before the import is in those repositories.

## Local AI context numbers (measured on a 18 GB M-series Mac, 2026-10-29, helper on the processor)

Method: the release helper (`cargo build --release -p voice-engine-llm`) with `gpu: false`, one process per context
size, loading the model at `n_ctx` and generating 8 tokens from a short prompt so the cache is touched. Peak size is
`maximum resident set size` from `/usr/bin/time -l`, in GiB (bytes / 1073741824). The sweep is 8192, 16384, 32768,
65536 and 131072 (stopping at the model's `context_max` or at a peak above 14 GiB; none reached 14 GiB). The GGUF
header says `context_max`: 262144 for Qwen3.5 4B and 9B, 131072 for Granite 4.2 3B (read with the helper's `inspect`).

| Model | n_ctx | Peak GiB |
| --- | --- | --- |
| qwen3.5-4b | 8192 | 2.97 |
| qwen3.5-4b | 16384 | 3.26 |
| qwen3.5-4b | 32768 | 3.76 |
| qwen3.5-4b | 65536 | 4.76 |
| qwen3.5-4b | 131072 | 6.76 |
| qwen3.5-9b | 8192 | 5.67 |
| qwen3.5-9b | 16384 | 6.00 |
| qwen3.5-9b | 32768 | 6.50 |
| qwen3.5-9b | 65536 | 7.50 |
| qwen3.5-9b | 131072 | 9.37 |
| granite-4.2-3b | 8192 | 2.75 |
| granite-4.2-3b | 16384 | 3.44 |
| granite-4.2-3b | 32768 | 4.65 |
| granite-4.2-3b | 65536 | 7.14 |
| granite-4.2-3b | 131072 | 12.10 |

Rule: `context_8gb` is the largest tested `n_ctx` whose peak is at most 5 GiB (the rest is for the system and the
app); `context_16gb` is the largest at most 11 GiB; if even 8192 does not fit, `context_8gb` is 4096; both are capped
at `context_max`.

| Model | `context_max` | `context_8gb` | `context_16gb` | Why |
| --- | --- | --- | --- | --- |
| qwen3.5-4b | 262144 | 65536 | 131072 | 4.76 GiB at 65536; 6.76 GiB at 131072, the largest size tested |
| qwen3.5-9b | 262144 | 4096 | 131072 | 5.67 GiB at 8192 is over 5 GiB, so 4096 (not measured; the note says so); 9.37 GiB at 131072, the largest size tested |
| granite-4.2-3b | 131072 | 32768 | 65536 | 4.65 GiB at 32768; 7.14 GiB at 65536; 12.10 GiB at 131072 is over 11 GiB |
