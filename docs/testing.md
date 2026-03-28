# Testing

Default `cargo test --workspace` covers:

- **`qts_ggml_sys`**: minimal GGML graph smoke test (`smoke_add_graph`).
- **`qts_ggml`**: wrapper smoke test (`smoke_add_via_sys`).
- **`qts`**: library unit tests (e.g. request defaults, `ModelPaths`, tokenizer, speaker encoder, voice-clone protobuf, synthesis profiling helpers, and internal pipeline tests).

No network, no large files.

## Layer B — integration (ignored; real checkpoints)

Point `QWEN3_TTS_MODEL_DIR` at a directory that contains a main GGUF (e.g. `qwen3-tts-0.6b-f16.gguf`) and `qwen3-tts-vocoder.onnx` (see [`ModelPaths`](../crates/qts/src/model/paths.rs)).

```bash
export QWEN3_TTS_MODEL_DIR=/path/to/models
cargo test -p qts integration_ -- --ignored --nocapture
```

Tests in `crates/qts/tests/integration_model_dir.rs`:

- `integration_loads_models` — engine load and tokenizer sanity.
- `integration_synthesize_direct_path_audio` — short synthesis without a voice-clone prompt.
- `integration_voice_clone_prompt_xvector_mode` — loads `testdata/sample1.xvector.voice-clone-prompt.pb`.
- `integration_voice_clone_prompt_icl_mode` — loads `testdata/sample1.icl.voice-clone-prompt.pb`.

## Backend-specific compile checks

```bash
# Apple
cargo check -p qts --no-default-features --features metal

# Linux / Windows with Vulkan SDK + glslc available
cargo check -p qts --no-default-features --features vulkan

# Windows DirectML vocoder path (features pass through to `qts`)
cargo check -p qts_cli --no-default-features --features directml

# Linux CUDA vocoder path
cargo check -p qts_cli --no-default-features --features cuda

# x86_64 Linux / Windows OpenVINO vocoder path
cargo check -p qts_cli --no-default-features --features openvino

# CLI crate (same feature flags pass through to `qts`)
cargo check -p qts_cli
```

For ONNX Runtime EPs, keep in mind that ort prebuilt binaries only cover specific feature bundles. Platform-native EPs like `directml`, `xnnpack`, and `coreml` are broadly available where supported, and ort also publishes prebuilt bundles for `cuda` + `tensorrt`, `webgpu`, and `nvrtx`. Mixed combinations outside those bundles may resolve to a CPU-only ORT download unless you build ONNX Runtime from source. That matters for this repo because the default Linux / Windows vocoder feature set now includes `cuda`, `nvrtx`, and `tensorrt` together.

## Benchmarks

The Criterion target lives in the **`qts`** package (`synthesize` bench). Set `QWEN3_TTS_BENCH_MODEL_DIR` and optional tuning via `QWEN3_TTS_BENCH_*` env vars (see `crates/qts/benches/synthesize.rs`).

Convenience wrapper (see `.cargo/config.toml`):

```bash
cargo xtask bench cpu
cargo xtask bench metal
cargo xtask bench vulkan
```

The Vulkan path requires a working Vulkan SDK / loader and `glslc` on the machine performing the build.

Runtime backend is controlled by **`QWEN3_TTS_BACKEND`** (`auto`, `cpu`, `metal`, `vulkan`). When profiling with Vulkan on macOS (MoltenVK), build with `--features vulkan` and set `QWEN3_TTS_BACKEND=vulkan` for the CLI process.

The ONNX vocoder execution provider is controlled separately by **`QWEN3_TTS_VOCODER_EP`**. Supported tokens are `cpu`, `acl`, `armnn`, `azure`, `cann`, `coreml`, `cuda`, `directml`, `migraphx`, `nnapi`, `nvrtx`, `onednn`, `openvino`, `qnn`, `rknpu`, `tensorrt`, `tvm`, `vitis`, `webgpu`, and `xnnpack`, subject to the corresponding Cargo feature being enabled in the binary. On Apple platforms, `auto` prefers CoreML when available. On Windows, `auto` now tries `cuda,nvrtx,tensorrt,directml,cpu`. On Linux and other non-Apple targets, `auto` tries `cuda,nvrtx,tensorrt,cpu`.

If you expect all of those EPs to be usable in one binary, verify that your ONNX Runtime build actually includes the combination you enabled. With ort’s stock prebuilt downloads, some mixed EP sets require building ORT from source first.

The CLI also accepts `--backend`, `--backend-fallback`, `--vocoder-ep`, and `--vocoder-ep-fallback`, which override the environment variables for that invocation.

The experimental talker KV cache can be enabled with **`QWEN3_TTS_TALKER_KV_MODE=turboquant`** or `--talker-kv-mode turboquant`. The cache stays on the selected backend, but quantization still happens on the host before the quantized bytes are uploaded back into the KV tensor.

### Synthesis stage profile (wall clock)

End-to-end stage timings use the **`qts_cli`** `profile` subcommand. **`cargo xtask profile`** runs it with the right `-p qts_cli` / `--features` / `QWEN3_TTS_BACKEND` combination for the chosen backend token (`cpu`, `metal`, `vulkan`, `all`, `both`).

Direct invocation (pick GPU features on the `cargo run` line when needed, e.g. `--features metal` or `--features vulkan`):

```bash
cargo run --release -p qts_cli -- profile --model-dir "$QWEN3_TTS_MODEL_DIR" --text "hello" --frames 32 --runs 5
```

Experimental TurboQuant profile run:

```bash
QWEN3_TTS_BACKEND=vulkan \
QWEN3_TTS_TALKER_KV_MODE=turboquant \
cargo run --release -p qts_cli -- profile \
  --model-dir "$QWEN3_TTS_MODEL_DIR" \
  --text "hello" \
  --frames 32 \
  --runs 5
```

Equivalent convenience:

```bash
cargo xtask profile cpu --model-dir "$QWEN3_TTS_MODEL_DIR" --text "hello" --frames 32 --runs 5
```

Optional: `--out target/profile-run1.wav` writes WAV from the first run only. The table is printed to stderr; use `--runs N` to average stage times over multiple iterations (useful after a warmup run with `--runs 1` first if you care about steady state).

## CLI smoke checks

Voice-clone prompt import can be smoke-tested with protobuf prompts exported by the upstream helper.

Xvector-only mode:

```bash
uv sync
uv run export-voice-clone-prompt \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio testdata/hello.wav \
  --x-vector-only-mode \
  --out target/hello.xvector.voice-clone-prompt.pb

cargo run -p qts_cli -- synthesize \
  --model-dir "$QWEN3_TTS_MODEL_DIR" \
  --text "hello" \
  --voice-clone-prompt target/hello.xvector.voice-clone-prompt.pb \
  --out target/hello.from-xvector-prompt.wav
```

ICL mode:

```bash
uv sync
uv run export-voice-clone-prompt \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio testdata/hello.wav \
  --ref-text "hello" \
  --out target/hello.voice-clone-prompt.pb

cargo run -p qts_cli -- synthesize \
  --model-dir "$QWEN3_TTS_MODEL_DIR" \
  --text "hello" \
  --voice-clone-prompt target/hello.voice-clone-prompt.pb \
  --out target/hello.from-prompt.wav
```

## `testdata/`

See [testdata/README.md](../testdata/README.md). Most large or generated files under `testdata/` are ignored; only the small checked-in prompt fixtures and `testdata/README.md` are intended to be versioned.
