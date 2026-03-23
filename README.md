# qts

Rust workspace for on-device **Qwen3 TTS** using [ggml-org/ggml](https://github.com/ggml-org/ggml) for the main transformer and **ONNX Runtime** for the exported vocoder.

## Repository Layout

- `crates/`: Rust crates for GGML bindings, the TTS engine, and the CLI/TUI.
- `scripts/`: canonical Python helper entrypoints for model export and prompt tooling.
- `docs/`: model, testing, and release-oriented documentation.
- `testdata/`: small checked-in fixtures only; large local models and scratch audio stay outside the repo root.

## Crates

| Crate | Role |
|-------|------|
| `qts_ggml_sys` | CMake + bindgen FFI to `crates/qts_ggml_sys/vendor/ggml` ([ggml](https://github.com/ggml-org/ggml) Git submodule) |
| `qts_ggml` | Thin wrappers + `sys` re-export |
| `qts` | Pure Rust `rlib` for GGUF loading, speaker encoding, synthesis, and crate-local protobuf codegen |
| `qts_cli` | Command-line interface for synthesis, profiling, and interactive TUI playback |

## Prerequisites

- **CMake** on the PATH (for `qts_ggml_sys` building its vendored `ggml` submodule).

## Build

Fetch ggml first (submodule under `crates/qts_ggml_sys/vendor/ggml`):

```bash
git submodule update --init --recursive
```

Then:

```bash
cargo build --workspace
cargo test --workspace
```

Python helper scripts are managed with `uv` from the workspace root. Public entrypoints and shared Python support code now live under `scripts/`:

```bash
uv sync
uv run export-model-artifacts --help
uv run export-voice-clone-prompt --help
```

CLI builds the same engine as the library and supports the same backend features:

```bash
cargo build -p qts_cli
cargo build -p qts_cli --features metal
cargo build -p qts_cli --features vulkan
cargo build -p qts_cli --features directml
```

GPU / accelerator backends are Cargo features on `qts_ggml_sys` (see [VERSIONS.md](VERSIONS.md)).
`qts` keeps its protobuf schema under `crates/qts/proto/` so `cargo package` / `cargo publish` can verify the crate from a self-contained tarball.
The checked-in Python binding at `scripts/voice_clone_prompt_pb2.py` is generated from that same canonical schema; regenerate it with `uv run generate-voice-clone-prompt-pb2` after schema changes.

Common backend builds:

```bash
# Apple GPU path
cargo build -p qts --features metal

# Cross-platform GPU path (requires Vulkan SDK / loader + glslc)
cargo build -p qts --features vulkan
```

Runtime backend preference is automatic:

- Apple builds with `metal` prefer Metal and fall back to CPU if initialization fails.
- Non-Apple builds with `vulkan` prefer Vulkan and fall back to CPU if initialization fails.
- Builds without GPU features use CPU only.

## Models

Documented GGUF links and directory layout: [docs/models.md](docs/models.md).

### Repository Relationship

`qts` uses two repositories with different responsibilities:

- GitHub [`yet-another-ai/qts`](https://github.com/yet-another-ai/qts) is the source-of-truth repository for code, export scripts, tests, and developer documentation.
- Hugging Face [`dsh0416/qts-12Hz-0.6B-Base-QTS`](https://huggingface.co/dsh0416/Qwen3-TTS-12Hz-0.6B-Base-QTS) is the distribution repository for built model artifacts.

In practice:

- Make code, export, and format changes in this GitHub repository first.
- Export stable artifacts from a known Git commit.
- Upload only the resulting model files to the Hugging Face repository.
- Keep the Hugging Face model card aligned with this repository's docs, but do not treat it as a second source repository.

The intended artifact layout is one shared `qts-vocoder.onnx` plus one or more GGUF variants such as `qts-0.6b-f16.gguf` and `qts-0.6b-q8_0.gguf` in the same Hugging Face repository root. That layout matches the default `ModelPaths::from_model_dir(...)` resolution used by the Rust runtime.

A copy-ready Hugging Face model card template lives at [`docs/huggingface-model-card.md`](docs/huggingface-model-card.md).
To prepare a release directory with `README.md`, `SHA256SUMS`, and Hugging Face Xet
tracking for `*.gguf` / `*.onnx`, run `cargo xtask hf-release --model Qwen/qts-12Hz-0.6B-Base`. If you already have the Hugging Face repository cloned locally, add `--hf-repo-dir /path/to/cloned-hf-repo` to sync the managed release files into that existing git checkout.

Official Hugging Face publication is handled by GitHub Actions:

- `.github/workflows/build.yml` builds accelerated `qts_cli` archives for Linux, macOS, and Windows on pull requests and `main` (Vulkan on Linux, `metal+coreml+blas` on macOS via Accelerate, and `vulkan+directml` on Windows), and uploads tagged `v*` builds to GitHub Releases
- `.github/workflows/hf-release.yml` publishes tagged `v*` releases to `dsh0416/qts-12Hz-0.6B-Base-QTS`
- `.github/workflows/model-integration.yml` provides a manual release-preview run that exports and uploads the staged bundle without pushing
- repository secret `HF_TOKEN` must be configured with write access to the Hugging Face model repository

When `cargo xtask hf-release` receives `--hf-repo-dir` and you do not override `--artifacts-dir` or `--out-dir`, it now exports and packages directly in that cloned Hugging Face repository root to avoid extra copies. The local flow remains useful for previewing the exact files that the tagged release workflow will publish.

## Voice Clone Prompts

For better alignment with upstream `QwenLM/qts`, the native path consumes protobuf voice-clone prompts rather than direct reference-audio conditioning at synthesis time.

### Protobuf prompt export and native consumption

The voice-clone runtime now supports two protobuf prompt modes:

- **xvector-only mode**: use the reference speaker identity only.
- **ICL mode**: use the reference speaker identity plus the upstream in-context reference text and codec prompt.

Use the repository's `uv`-managed Python environment to export either form from `create_voice_clone_prompt(...)`.

### Xvector-only mode

This is the closest replacement for the old `speaker.bin` workflow, but the runtime now consumes the protobuf prompt directly instead of a raw embedding file.

```bash
uv sync

uv run export-voice-clone-prompt \
  --model Qwen/qts-12Hz-0.6B-Base \
  --ref-audio testdata/hello.wav \
  --x-vector-only-mode \
  --out target/hello.xvector.voice-clone-prompt.pb
```

Then synthesize with the prompt:

```bash
cargo run -p qts_cli -- synthesize \
  --model-dir models/qts-bundle \
  --text "hello" \
  --voice-clone-prompt target/hello.xvector.voice-clone-prompt.pb \
  --out target/hello-from-xvector-prompt.wav
```

### ICL mode

ICL mode mirrors upstream `create_voice_clone_prompt(...)` behavior: the prompt includes `ref_spk_embedding`, `ref_code`, and `ref_text`, and the native runtime uses those fields together.

```bash
uv sync

uv run export-voice-clone-prompt \
  --model Qwen/qts-12Hz-0.6B-Base \
  --ref-audio testdata/hello.wav \
  --ref-text "hello" \
  --out target/hello.voice-clone-prompt.pb
```

Then consume that prompt from `qts_cli`:

```bash
cargo run -p qts_cli -- synthesize \
  --model-dir models/qts-bundle \
  --text "hello" \
  --voice-clone-prompt target/hello.voice-clone-prompt.pb \
  --out target/hello-from-prompt.wav
```

The native engine reads the protobuf and uses:

- `ref_spk_embedding`
- `ref_code`
- `ref_text`
- `icl_mode` / `x_vector_only_mode`

Compatibility wrapper path still works too:

```bash
uv run python scripts/export_voice_clone_prompt.py --help
```

## Interactive TUI

For interactive latency demos, the CLI also has a `tui` mode that loads the model once, lets you type successive utterances, and streams audio directly to the default output device through `cpal`.

```bash
cargo run -p qts_cli -- tui \
  --model-dir models/qts-bundle \
  --voice-clone-prompt target/hello.xvector.voice-clone-prompt.pb \
  --language en \
  --chunk-size 4
```

Notes:

- Press `Enter` to synthesize the current line.
- Press `F2` to cycle the synthesis language between English, Chinese, and Japanese.
- Press `Esc`, `Ctrl-C`, or type `:q` to quit.
- The TUI header shows both the transformer backend (`qts_ggml`) and the active vocoder EP (`ORT/CPU`, `ORT/CoreML`, or `ORT/DirectML`).
- `--backend` and `--vocoder-ep` let you choose the transformer backend and vocoder EP directly from the command line.
- `--backend-fallback` and `--vocoder-ep-fallback` accept comma-separated fallback chains used when the corresponding selector is `auto`.
- `--language en|zh|ja` is the friendly startup flag; `--language-id` still works if you want to pass a raw codec language id.
- `--chunk-size` controls how many codec frames are vocoded per playback chunk. Smaller values reduce startup latency, while larger values reduce scheduling overhead.
- `--voice-clone-prompt` is loaded once up front and then reused for each prompt.

On Apple platforms, the ONNX vocoder can use CoreML:

```bash
cargo run -p qts_cli -- tui \
  --model-dir models/qts-f16-onnx \
  --backend auto \
  --backend-fallback metal,vulkan,cpu \
  --vocoder-ep coreml \
  --chunk-size 4
```

On Windows, the ONNX vocoder can use DirectML:

```bash
cargo run -p qts_cli --no-default-features --features vulkan,directml -- tui \
  --model-dir models/qts-f16-onnx \
  --backend auto \
  --backend-fallback vulkan,cpu \
  --vocoder-ep directml \
  --chunk-size 4
```

Defaults:

- Apple transformer `auto`: `metal,vulkan,cpu`
- Other transformer `auto`: `vulkan,cpu`
- Apple vocoder `auto`: `coreml,cpu`
- Windows vocoder `auto`: `directml,cpu`
- Other vocoder `auto`: `cpu`

## Tests

Fast tests run in CI; model-backed tests are opt-in: [docs/testing.md](docs/testing.md).

Criterion benchmarks and synthesis profiling are driven through the `xtask` Cargo alias (see [`.cargo/config.toml`](.cargo/config.toml)):

```bash
cargo xtask bench cpu
cargo xtask bench metal
cargo xtask bench vulkan
```

Stage timings (tokenizer, prefill build, codec rollout / transformer, vocoder, etc.) for a real synthesis pass:

```bash
cargo xtask profile cpu --model-dir models/qts-bundle --text "hello" --frames 64 --runs 3
cargo xtask profile metal --model-dir models/qts-bundle --text "hello" --frames 64
cargo xtask profile vulkan --model-dir models/qts-bundle --text "hello" --frames 64
```

`cargo xtask profile` sets **`QWEN3_TTS_BACKEND`** for the child to match the first token (`cpu` / `metal` / `vulkan`), so **Cargo features and the actual GGML primary backend stay aligned** (including Vulkan on macOS when you choose the `vulkan` profile).

For `cargo run -p qts_cli` directly, set the backend explicitly, for example:

```bash
QWEN3_TTS_BACKEND=vulkan cargo run -p qts_cli --features vulkan -- profile --text "hello" --model-dir models/qts-bundle --frames 64
```

This runs `qts_cli profile`, which prints per-stage milliseconds and percentage of total wall time. Use `--out run1.wav` to keep audio from the first run while profiling.

**Transformer backend selection:** use `--backend` / `--backend-fallback` on the CLI, or `QWEN3_TTS_BACKEND` / `QWEN3_TTS_BACKEND_FALLBACK` if you prefer environment variables. The default `auto` chain is `metal,vulkan,cpu` on Apple and `vulkan,cpu` elsewhere.

GGML picks GPU devices by **registry** name plus index: use **`QWEN3_TTS_GPU_DEVICE`** (default `0`) to choose among multiple Vulkan or Metal adapters (`Vulkan0` / `MTL0`, …).

**Vocoder EP selection:** use `--vocoder-ep` / `--vocoder-ep-fallback` on the CLI, or `QWEN3_TTS_VOCODER_EP` / `QWEN3_TTS_VOCODER_EP_FALLBACK` from the environment. The default `auto` chain is `coreml,cpu` on Apple, `directml,cpu` on Windows builds with the `directml` feature, and `cpu` elsewhere.

## License

This repository is licensed under **Apache License 2.0**. See [`LICENSE`](LICENSE) for the full text and [`NOTICE`](NOTICE) for repository-level attribution notes.

## Godot / gdext

The `qts` crate is a normal Rust library (`rlib`). A future Godot project can depend on it directly from a `gdext` crate without a separate `cdylib` ABI layer.

## Acknowledgments

- [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) for architecture and tensor naming.
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for the model architecture and tensor naming.
