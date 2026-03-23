# Testing

- `ggml-sys`: minimal GGML graph smoke test.
- `ggml`: wrapper smoke test.
- `qwen3-tts`: request defaults, `ModelPaths` helpers.

No network, no large files.

Then:

```bash
export QWEN3_TTS_MODEL_DIR=/path/to/models
cargo test -p qwen3-tts -- --ignored
```

- `integration_voice_clone_prompt_xvector_mode` loads `testdata/sample1.xvector.voice-clone-prompt.pb`.
- `integration_voice_clone_prompt_icl_mode` loads `testdata/sample1.icl.voice-clone-prompt.pb`.

Backend-specific compile checks:

```bash
# Apple
cargo check -p qwen3-tts --no-default-features --features metal

# Linux / Windows with Vulkan SDK + glslc available
cargo check -p qwen3-tts --no-default-features --features vulkan

# Windows DirectML vocoder path
cargo check -p qwen3-tts-cli --no-default-features --features directml

# CLI crate (same feature flags pass through to qwen3-tts)
cargo check -p qwen3-tts-cli
```

## Layer C — golden numerics (planned)

Copy `reference/*.bin` from [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) and add component tests with explicit tolerances (tokenizer, encoder, transformer, vocoder).

## Layer D — heavy E2E (manual)

Python or upstream `compare_e2e.py`-style checks should stay out of default PR CI.

## Benchmarks

Use the `cargo xtask` alias (see `.cargo/config.toml`) to run Criterion with the intended backend feature enabled:

```bash
cargo xtask bench cpu
cargo xtask bench metal
cargo xtask bench vulkan
```

The Vulkan path requires a working Vulkan SDK / loader and `glslc` on the machine performing the build.

Runtime backend is controlled by **`QWEN3_TTS_BACKEND`** (`auto`, `cpu`, `metal`, `vulkan`). `cargo xtask profile vulkan` sets `QWEN3_TTS_BACKEND=vulkan` so macOS can use MoltenVK when the binary is built with `--features vulkan`.

The ONNX vocoder execution provider is controlled separately by **`QWEN3_TTS_VOCODER_EP`** (`auto`, `cpu`, `coreml`, `directml`). On Apple platforms, `auto` prefers CoreML when available. On Windows builds with the `directml` feature, `auto` prefers DirectML before CPU.

The CLI also accepts `--backend`, `--backend-fallback`, `--vocoder-ep`, and `--vocoder-ep-fallback`, which override the environment variables for that invocation.

### Synthesis stage profile (wall clock)

To see approximate time spent in tokenizer vs transformer (codec rollout) vs vocoder for one or more end-to-end runs:

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

cargo run -p qwen3-tts-cli -- synthesize \
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

cargo run -p qwen3-tts-cli -- synthesize \
  --model-dir "$QWEN3_TTS_MODEL_DIR" \
  --text "hello" \
  --voice-clone-prompt target/hello.voice-clone-prompt.pb \
  --out target/hello.from-prompt.wav
```

## `testdata/`

See [testdata/README.md](../testdata/README.md). Most large or generated files under `testdata/` are ignored; only the small checked-in prompt fixtures and `testdata/README.md` are intended to be versioned.
