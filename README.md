# qts

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/yet-another-ai/qts/actions/workflows/test.yml/badge.svg)](https://github.com/yet-another-ai/qts/actions/workflows/test.yml)
[![Build](https://github.com/yet-another-ai/qts/actions/workflows/build.yml/badge.svg)](https://github.com/yet-another-ai/qts/actions/workflows/build.yml)

[![crates.io: qts](https://img.shields.io/crates/v/qts.svg?label=qts&logo=rust)](https://crates.io/crates/qts)
[![crates.io: qts_cli](https://img.shields.io/crates/v/qts_cli.svg?label=qts_cli&logo=rust)](https://crates.io/crates/qts_cli)
[![crates.io: qts_ggml](https://img.shields.io/crates/v/qts_ggml.svg?label=qts_ggml&logo=rust)](https://crates.io/crates/qts_ggml)
[![crates.io: qts_ggml_sys](https://img.shields.io/crates/v/qts_ggml_sys.svg?label=qts_ggml_sys&logo=rust)](https://crates.io/crates/qts_ggml_sys)

**On-device [Qwen3 TTS](https://github.com/QwenLM/Qwen3-TTS)** in Rust: the speech model runs in **[ggml](https://github.com/ggml-org/ggml)** (GGUF weights), and the vocoder runs in **ONNX Runtime**. No server required—everything stays on your machine.

| If you want to… | Start here |
|-----------------|------------|
| Turn text into a WAV file | [Quick start](#quick-start) → [Synthesize](#synthesize-text-to-wav) |
| Match a reference voice (speaker / style) | [Voice clone prompts](#voice-clone-prompts) |
| Try it interactively in the terminal | [Interactive TUI](#interactive-tui) |
| Embed the engine in your own Rust app | Use the **`qts`** library crate (see [Crates](#crates)) |
| Tune GPU / CPU backends | [Runtime configuration](#runtime-configuration) |

---

## Quick start

**You need:** [Rust](https://rustup.rs/), **CMake** on your `PATH`, and Git (for the ggml submodule).
On Windows, use the MSVC Rust toolchain plus Visual Studio 2022 Build Tools;
Vulkan builds also need the Vulkan SDK (`glslc` on `PATH`).

1. **Clone and fetch ggml**

   ```bash
   git clone https://github.com/yet-another-ai/qts.git
   cd qts
   git submodule update --init --recursive
   ```

2. **Build the CLI** (first build compiles vendored ggml; it can take a few minutes)

   ```bash
   cargo build --release -p qts_cli
   ```

3. **Download model files** — this repo does not ship weights. Grab a main GGUF plus the shared vocoder ONNX from [Hugging Face](https://huggingface.co/dsh0416/Qwen3-TTS-12Hz-0.6B-Base-QTS) (or export your own; see [docs/models.md](docs/models.md)) and put them in one folder, for example:

   ```
   models/
     qwen3-tts-0.6b-f16.gguf    # or another supported q4_k / q5_k / q6_k / q8_0 variant
     qwen3-tts-vocoder.onnx
   ```

   Those names match the default lookup used by `--model-dir` (see [`ModelPaths`](crates/qts/src/model/paths.rs)).
   VoiceDesign and CustomVoice model folders also need their exported
   `config.json`; see [Model type config](#model-type-config).

4. **Synthesize**

   ```bash
   cargo run --release -p qts_cli -- synthesize \
     --model-dir models \
     --text "Hello from local TTS." \
     --out hello.wav
   ```

On **Apple Silicon**, default features include Metal and CoreML where applicable. On **Linux / Windows**, the default build also enables the NVIDIA-oriented vocoder EPs `cuda`, `nvrtx`, and `tensorrt`; DirectML remains available via an extra feature flag on Windows (see [Build options](#build-options)).

---

## Repository layout

| Path | What it is |
|------|------------|
| [`crates/`](crates/) | Rust: GGML bindings, TTS engine (`qts`), CLI/TUI (`qts_cli`) |
| [`scripts/`](scripts/) | Python (`uv`): export GGUF/ONNX and voice-clone protobuf prompts |
| [`docs/`](docs/) | Models, testing, releases, Hugging Face card template |
| [`testdata/`](testdata/) | Small fixtures only; keep large checkpoints outside the repo |

## Crates

| Crate | Role |
|-------|------|
| `qts_ggml_sys` | CMake + bindgen FFI to vendored ggml ([submodule](crates/qts_ggml_sys/vendor/ggml)) |
| `qts_ggml` | Thin wrappers + `sys` re-export |
| `qts` | Library: GGUF load, tokenizer, transformer inference, speaker encoding, vocoder bridge, protobuf voice-clone types |
| `qts_cli` | `synthesize`, `profile`, and interactive `tui` |

## Build options

**CLI** (same engine as the library):

```bash
cargo build -p qts_cli
cargo build -p qts_cli --features metal    # Apple GPU (GGML)
cargo build -p qts_cli --features vulkan   # Vulkan (GGML); needs SDK + `glslc` where applicable
cargo build -p qts_cli --features tensorrt # NVIDIA TensorRT vocoder
cargo build -p qts_cli --features directml # Windows vocoder (ONNX DirectML)
cargo build -p qts_cli --features cuda     # NVIDIA vocoder (ONNX CUDA)
```

**Library-only** examples:

```bash
cargo build -p qts --features metal
cargo build -p qts --features vulkan
```

GPU features are declared on `qts_ggml_sys` / `qts`; details and version pins live in [VERSIONS.md](VERSIONS.md). For the vocoder, `qts` and `qts_cli` forward the native ONNX Runtime EP feature set directly, including `acl`, `armnn`, `azure`, `cann`, `coreml`, `cuda`, `directml`, `migraphx`, `nnapi`, `nvrtx`, `onednn`, `openvino`, `qnn`, `rknpu`, `tensorrt`, `tvm`, `vitis`, `webgpu`, and `xnnpack`. The default feature set now includes `cuda`, `nvrtx`, and `tensorrt` in addition to the existing GGML defaults.

**ONNX Runtime build note:** ort does **not** ship prebuilt binaries for every EP combination. Its documented prebuilt bundles cover platform-native EPs like `directml`, `xnnpack`, and `coreml`, plus separate bundles for `cuda` + `tensorrt`, `webgpu`, and `nvrtx`. If you enable a mixed combination outside those bundles, ort may fall back to downloading a CPU-only runtime unless you compile ONNX Runtime from source. In practice, if you want a single build with `cuda`, `nvrtx`, and `tensorrt` all available together, plan on a source-built ORT.

**Runtime behavior:** with GPU features enabled, `auto` prefers **Metal** on Apple and **Vulkan** on other platforms, then falls back to **CPU** if init fails. Builds without those features use **CPU** only for GGML.

### Package the CLI runtime

Use `xtask package-cli` to build `qts_cli` and gather the executable plus
runtime DLLs into one directory:

```bash
cargo xtask package-cli
```

Defaults are equivalent to:

```bash
cargo build --release -p qts_cli --no-default-features --features vulkan
```

The package is written to `target/qts-cli-package/`. On Windows it contains
`qts_cli.exe`, `onnxruntime.dll`, the bundled MSVC-built `soxr.dll`, and any
available ONNX Runtime provider DLLs found next to the build output or in
`.venv/Lib/site-packages/onnxruntime/capi`.

Useful variants:

```bash
cargo xtask package-cli --out-dir target/qts-cli-win
cargo xtask package-cli --features "vulkan,directml"
cargo xtask package-cli --no-features
cargo xtask package-cli --profile debug --skip-build
```

`soxr.dll` is built automatically by `qts` using CMake. On Windows the build
script forces the Visual Studio 2022 generator (`-A x64` for x86_64 targets)
instead of MinGW. Set `QWEN3_TTS_SKIP_BUNDLED_SOXR=1` to skip that build, or
`QWEN3_TTS_SOXR_SRC=/path/to/soxr` to use an existing checkout.

Full workspace:

```bash
cargo build --workspace
cargo test --workspace
```

### Python helpers (`uv`)

Export and prompt tooling live under `scripts/`:

```bash
uv sync
uv run export-model-artifacts --help
uv run export-voice-clone-prompt --help
```

`qts` ships its protobuf schema in [`crates/qts/proto/`](crates/qts/proto/). Regenerate the checked-in Python stub with `uv run generate-voice-clone-prompt-pb2` after schema changes.

---

## Models

Where to download, how to export, and layout options: **[docs/models.md](docs/models.md)**.

**Default files in one directory** (used by `--model-dir`):

- `qwen3-tts-vocoder.onnx`
- One of: `qwen3-tts-0.6b-f16.gguf`, `qwen3-tts-0.6b-q8_0.gguf`, … (see `ModelPaths` for the full preference order)

### Model type config

VoiceDesign and CustomVoice exports should keep their source `config.json` in
the same `--model-dir` as the GGUF and vocoder. `qts` reads this file to
determine the model type, especially `tts_model_type` values such as
`voice_design` and `custom_voice`. Without it, a fixed server mode like
`--mode design` or `--mode custom` can reject requests or use the wrong
conditioning path.

Recommended layouts:

```text
models/Qwen3-TTS-12Hz-1.7B-VoiceDesign/
  qwen3-tts-1.7b-voicedesign-q8_0.gguf
  qwen3-tts-vocoder.onnx
  config.json
  qwen3-tts-tokenizer-encoder.onnx  # recommended for long-form voice stability

models/Qwen3-TTS-12Hz-1.7B-CustomVoice/
  qwen3-tts-1.7b-customvoice-q8_0.gguf
  qwen3-tts-vocoder.onnx
  config.json

models/Qwen3-TTS-12Hz-0.6B-Base/
  qwen3-tts-0.6b-f16.gguf
  qwen3-tts-vocoder.onnx
  qwen3-tts-tokenizer-encoder.onnx  # required for WAV voice clone prompts
```

For long-form VoiceDesign jobs, `qts_server` uses
`qwen3-tts-tokenizer-encoder.onnx` when it is present to build an ICL prompt
from the first generated segment, so later segments keep the same voice. If the
encoder is missing, it falls back to speaker-embedding reuse.

### Maintainers: two repos, one workflow

| Repo | Role |
|------|------|
| GitHub [`yet-another-ai/qts`](https://github.com/yet-another-ai/qts) | Source of truth for code, export scripts, tests, docs |
| Hugging Face [`dsh0416/Qwen3-TTS-12Hz-0.6B-Base-QTS`](https://huggingface.co/dsh0416/Qwen3-TTS-12Hz-0.6B-Base-QTS) | Published GGUF + ONNX artifacts |

Typical flow: change and export from a pinned commit here → upload only binaries to Hugging Face → keep the HF model card in sync with this repo’s docs (template: [`docs/huggingface-model-card.md`](docs/huggingface-model-card.md)).

Release packaging helper:

```bash
cargo xtask hf-release --model Qwen/qts-12Hz-0.6B-Base
```

Add `--hf-repo-dir /path/to/cloned-hf-repo` to sync into an existing clone. CI (`.github/workflows/`) builds release binaries and can publish tagged releases; see workflow comments for `HF_TOKEN` and related setup.

CLI runtime packaging helper:

```bash
cargo xtask package-cli
```

---

## Using the CLI

### HTTP server

`qts_server` is a separate executable. The conditioning mode is fixed at
startup, so requests cannot switch a running server between `none`, `custom`,
`design`, and `clone`.
For long inputs, the server splits text into sequential synthesis segments,
concatenates the audio into one WAV response, and reports aggregate async job
progress.

```bash
cargo run --release -p qts_cli --bin qts_server -- \
  --mode design \
  --model-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --backend vulkan \
  --vocoder-ep cpu \
  --language-id 2055 \
  --instruct "温柔年轻女声"
```

Health and async job flow:

```bash
curl http://127.0.0.1:8080/health

curl -X POST http://127.0.0.1:8080/v1/qts/audio/jobs \
  -H "content-type: application/json" \
  -d '{"input":"你好，测试一下进度查询。","instructions":"温柔年轻女声","frames":64,"response_format":"wav"}'

curl http://127.0.0.1:8080/v1/qts/audio/jobs/1
curl -o out.wav http://127.0.0.1:8080/v1/qts/audio/jobs/1/audio
```

OpenAI-compatible synchronous speech endpoint:

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{"model":"qwen3-tts","input":"你好","voice":"default","instructions":"温柔年轻女声","response_format":"wav"}' \
  -o out.wav
```

Mode-specific startup flags:

```bash
qts_server --mode custom --speaker serena --model-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
qts_server --mode clone --voice-clone-wav ref.wav --voice-clone-ref-text "reference text" --model-dir models/Qwen3-TTS-12Hz-0.6B-Base
```

### Synthesize text to WAV

```bash
cargo run --release -p qts_cli -- synthesize \
  --model-dir models \
  --text "Your line here." \
  --out out.wav
```

Useful knobs include `--threads`, `--frames` (max audio frames), `--temperature`, `--top-p`, `--top-k`, `--language-id`, and `--chunk-size` (see `--help` on the binary). Backend overrides: `--backend`, `--vocoder-ep`, plus fallback chains. `--vocoder-ep` accepts `auto` or any enabled native ORT EP token such as `coreml`, `directml`, `cuda`, `openvino`, `tensorrt`, or `xnnpack`.

### Voice clone from WAV / prompts

The CLI can consume a reference WAV directly without Python. `--voice-clone-wav`
alone uses the lighter x-vector-only path. Add `--voice-clone-ref-text` to use
the native upstream-style ICL clone path. Native ICL requires
`qwen3-tts-tokenizer-encoder.onnx`; the bundled MSVC-built `soxr.dll` handles
Python-free audio-code extraction at runtime. You can still override the
resampler with `QWEN3_TTS_SOXR_DLL`, or place `libsoxr.dll` / `soxr.dll` next to
the executable.

Reusable protobuf prompts generated by `export-voice-clone-prompt` are also
accepted as a compatibility/cache format.

**Direct WAV example**

```bash
cargo run --release -p qts_cli -- synthesize \
  --model-dir models \
  --text "hello" \
  --voice-clone-wav testdata/hello.wav \
  --out target/hello-from-xvector.wav
```

**Reusable ICL prompt example**

Reusable protobuf prompts are still accepted as a compatibility/cache format.
They are useful while the tokenizer encoder artifact is being produced.

```bash
uv run export-voice-clone-prompt \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio testdata/hello.wav \
  --ref-text "hello" \
  --out target/hello.voice-clone-prompt.pb

cargo run --release -p qts_cli -- synthesize \
  --model-dir models \
  --text "hello" \
  --voice-clone-prompt target/hello.voice-clone-prompt.pb \
  --out target/hello-from-icl.wav
```

**Native ICL WAV example**

```bash
cargo run --release -p qts_cli -- synthesize \
  --model-dir models \
  --text "hello" \
  --voice-clone-wav testdata/hello.wav \
  --voice-clone-ref-text "hello" \
  --out target/hello-from-icl.wav
```

`--voice-clone-prompt` files contain fields such as `ref_spk_embedding`,
`ref_code`, `ref_text`, and the `icl_mode` / `x_vector_only_mode` flags. Export
helper:

```bash
uv run export-voice-clone-prompt --help
```

### Interactive TUI

Loads once, then you type lines and hear audio via **cpal**.

```bash
cargo run --release -p qts_cli -- tui \
  --model-dir models \
  --voice-clone-prompt target/hello.xvector.voice-clone-prompt.pb \
  --language en \
  --chunk-size 4
```

| Key / input | Action |
|-------------|--------|
| `Enter` | Synthesize current line |
| `F2` | Cycle English / Chinese / Japanese |
| `Esc`, `Ctrl-C`, or `:q` | Quit |

The header shows the active **transformer** backend and **vocoder** execution provider. `--language en|zh|ja` is a friendly alias; `--language-id` still sets the raw codec id. `--chunk-size` trades startup latency vs scheduling overhead (codec frames per playback chunk).

**Apple (CoreML vocoder example)**

```bash
cargo run --release -p qts_cli -- tui \
  --model-dir models \
  --backend auto \
  --backend-fallback metal,vulkan,cpu \
  --vocoder-ep coreml \
  --chunk-size 4
```

**Windows (DirectML vocoder example)**

```bash
cargo run --release -p qts_cli --no-default-features --features vulkan,directml -- tui \
  --model-dir models \
  --backend auto \
  --backend-fallback vulkan,cpu \
  --vocoder-ep directml \
  --chunk-size 4
```

**Default `auto` chains**

| Platform | Transformer | Vocoder |
|----------|-------------|---------|
| Apple | `metal,vulkan,cpu` | `coreml,cpu` |
| Windows | `vulkan,cpu` | `cuda,nvrtx,tensorrt,directml,cpu` |
| Linux / Other | `vulkan,cpu` | `cuda,nvrtx,tensorrt,cpu` |

---

## Runtime configuration

| Concern | CLI flags | Environment variables |
|---------|-----------|-------------------------|
| GGML backend | `--backend`, `--backend-fallback` | `QWEN3_TTS_BACKEND`, `QWEN3_TTS_BACKEND_FALLBACK` |
| ONNX vocoder EP | `--vocoder-ep`, `--vocoder-ep-fallback` | `QWEN3_TTS_VOCODER_EP`, `QWEN3_TTS_VOCODER_EP_FALLBACK` |
| Experimental talker KV cache | `--talker-kv-mode f16|turboquant` | `QWEN3_TTS_TALKER_KV_MODE` |
| Multi-GPU adapter index | — | `QWEN3_TTS_GPU_DEVICE` (default `0`; e.g. `Vulkan0`, `MTL0`) |

When using `cargo run -p qts_cli` directly, **Cargo features** (e.g. `--features vulkan` or `--features cuda`) must include the backend / execution provider you select with `QWEN3_TTS_BACKEND` or `QWEN3_TTS_VOCODER_EP`, or init will fail. The vocoder accepts the native ORT EP tokens `cpu`, `acl`, `armnn`, `azure`, `cann`, `coreml`, `cuda`, `directml`, `migraphx`, `nnapi`, `nvrtx`, `onednn`, `openvino`, `qnn`, `rknpu`, `tensorrt`, `tvm`, `vitis`, `webgpu`, and `xnnpack` when the matching feature is enabled.

**Profiling:** `cargo xtask profile` runs the CLI with matching features and sets `QWEN3_TTS_BACKEND` for you (important for Vulkan on macOS). Example:

```bash
cargo xtask profile cpu --model-dir models --text "hello" --frames 64 --runs 3
cargo xtask profile metal --model-dir models --text "hello" --frames 64
```

Manual equivalent:

```bash
QWEN3_TTS_BACKEND=vulkan cargo run --release -p qts_cli --features vulkan -- profile \
  --text "hello" --model-dir models --frames 64
```

`profile` prints per-stage timings; `--out run1.wav` keeps audio from the first run.

Experimental note: `--talker-kv-mode turboquant` switches the talker KV cache to a quantized GGML-backed storage path. The cache itself now lives on the selected backend, while host-side quantization and upload are still part of the write-back path. `profile` reports talker KV allocation plus `kv_download`, `kv_quantize`, and `kv_upload` timing buckets.

---

## Tests and benchmarks

- **Fast tests:** `cargo test --workspace` (no large downloads).  
- **Optional integration tests** (real checkpoints): set `QWEN3_TTS_MODEL_DIR` — see **[docs/testing.md](docs/testing.md)**.

**Benchmarks** (needs `QWEN3_TTS_BENCH_MODEL_DIR`, etc.):

```bash
cargo xtask bench cpu
cargo xtask bench metal
cargo xtask bench vulkan
```

Set `QWEN3_TTS_BENCH_TALKER_KV_MODE=turboquant` to compare the experimental talker KV cache against the default `f16` path.

Alias definition: [`.cargo/config.toml`](.cargo/config.toml).

---

## License

**Apache License 2.0** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Godot / gdext

`qts` is a normal Rust `rlib`. A Godot extension can depend on it from a `gdext` crate without a separate C ABI, unless you choose to add one.

## Acknowledgments

- [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) for architecture and tensor naming.
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for the model and naming conventions.
