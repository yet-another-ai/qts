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

4. **Synthesize**

   ```bash
   cargo run --release -p qts_cli -- synthesize \
     --model-dir models \
     --text "Hello from local TTS." \
     --out hello.wav
   ```

On **Apple Silicon**, default features include Metal and CoreML where applicable. On **Linux / Windows**, Vulkan and DirectML are available via feature flags (see [Build options](#build-options)).

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
cargo build -p qts_cli --features directml # Windows vocoder (ONNX DirectML)
```

**Library-only** examples:

```bash
cargo build -p qts --features metal
cargo build -p qts --features vulkan
```

GPU features are declared on `qts_ggml_sys` / `qts`; details and version pins live in [VERSIONS.md](VERSIONS.md).

**Runtime behavior:** with GPU features enabled, `auto` prefers **Metal** on Apple and **Vulkan** on other platforms, then falls back to **CPU** if init fails. Builds without those features use **CPU** only for GGML.

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

---

## Using the CLI

### Synthesize text to WAV

```bash
cargo run --release -p qts_cli -- synthesize \
  --model-dir models \
  --text "Your line here." \
  --out out.wav
```

Useful knobs include `--threads`, `--frames` (max audio frames), `--temperature`, `--top-p`, `--top-k`, `--language-id`, and `--chunk-size` (see `--help` on the binary). Backend overrides: `--backend`, `--vocoder-ep`, plus fallback chains.

### Voice clone prompts

To stay aligned with upstream **Qwen3 TTS**, conditioning uses **protobuf prompts** (exported from Python), not raw reference audio at synthesis time.

**Modes:**

- **xvector-only** — speaker identity from the reference clip.
- **ICL** — identity plus reference text and codec prompt (closer to upstream `create_voice_clone_prompt`).

**xvector-only example**

```bash
uv sync

uv run export-voice-clone-prompt \
  --model Qwen/qts-12Hz-0.6B-Base \
  --ref-audio testdata/hello.wav \
  --x-vector-only-mode \
  --out target/hello.xvector.voice-clone-prompt.pb

cargo run --release -p qts_cli -- synthesize \
  --model-dir models \
  --text "hello" \
  --voice-clone-prompt target/hello.xvector.voice-clone-prompt.pb \
  --out target/hello-from-xvector.wav
```

**ICL example**

```bash
uv run export-voice-clone-prompt \
  --model Qwen/qts-12Hz-0.6B-Base \
  --ref-audio testdata/hello.wav \
  --ref-text "hello" \
  --out target/hello.voice-clone-prompt.pb

cargo run --release -p qts_cli -- synthesize \
  --model-dir models \
  --text "hello" \
  --voice-clone-prompt target/hello.voice-clone-prompt.pb \
  --out target/hello-from-icl.wav
```

The engine reads fields such as `ref_spk_embedding`, `ref_code`, `ref_text`, and the `icl_mode` / `x_vector_only_mode` flags. Legacy wrapper:

```bash
uv run python scripts/export_voice_clone_prompt.py --help
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
| Windows (with `directml` feature) | `vulkan,cpu` | `directml,cpu` |
| Other | `vulkan,cpu` | `cpu` |

---

## Runtime configuration

| Concern | CLI flags | Environment variables |
|---------|-----------|-------------------------|
| GGML backend | `--backend`, `--backend-fallback` | `QWEN3_TTS_BACKEND`, `QWEN3_TTS_BACKEND_FALLBACK` |
| ONNX vocoder EP | `--vocoder-ep`, `--vocoder-ep-fallback` | `QWEN3_TTS_VOCODER_EP`, `QWEN3_TTS_VOCODER_EP_FALLBACK` |
| Multi-GPU adapter index | — | `QWEN3_TTS_GPU_DEVICE` (default `0`; e.g. `Vulkan0`, `MTL0`) |

When using `cargo run -p qts_cli` directly, **Cargo features** (e.g. `--features vulkan`) must include the backend you select with `QWEN3_TTS_BACKEND`, or init will fail.

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

Alias definition: [`.cargo/config.toml`](.cargo/config.toml).

---

## License

**Apache License 2.0** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Godot / gdext

`qts` is a normal Rust `rlib`. A Godot extension can depend on it from a `gdext` crate without a separate C ABI, unless you choose to add one.

## Acknowledgments

- [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) for architecture and tensor naming.
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for the model and naming conventions.
