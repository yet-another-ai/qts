# Versions & build matrix

## ggml (Git submodule)

- **Path**: `crates/qts_ggml_sys/vendor/ggml` → [ggml-org/ggml](https://github.com/ggml-org/ggml).
- **Pinned release**: **v0.9.8** (gitlink `2fb6431f67dd505584a9fefe94fd2866b944c85f`).
- **Clone**: `git submodule update --init --recursive` (or clone with `git clone --recurse-submodules …`).
- **Bump**: `cd crates/qts_ggml_sys/vendor/ggml && git fetch origin && git checkout <tag> && cd ../../../.. && git add crates/qts_ggml_sys/vendor/ggml && git commit`.
- **Reported version** (upstream `CMakeLists.txt` at this pin): **0.9.8**.

## `ggml-sys` Cargo features → CMake

| Feature | CMake option |
|---------|----------------|
| `native` | `GGML_NATIVE=ON` |
| `metal` | `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON` |
| `blas` | `GGML_BLAS=ON` (on Apple, `GGML_ACCELERATE` follows ggml defaults when BLAS on) |
| `cuda` | `GGML_CUDA=ON` |
| `vulkan` | `GGML_VULKAN=ON` |
| `hip` | `GGML_HIP=ON` |
| `musa` | `GGML_MUSA=ON` |
| `opencl` | `GGML_OPENCL=ON` |
| `rpc` | `GGML_RPC=ON` |
| `sycl` | `GGML_SYCL=ON` |
| `webgpu` | `GGML_WEBGPU=ON` |
| `openvino` | `GGML_OPENVINO=ON` |
| `hexagon` | `GGML_HEXAGON=ON` |
| `cann` | `GGML_CANN=ON` |
| `zendnn` | `GGML_ZENDNN=ON` |
| `zdnn` | `GGML_ZDNN=ON` |
| `virtgpu` | `GGML_VIRTGPU=ON` |

**Platform guards**

- `metal` is **Apple targets only** (build script will `panic!` elsewhere).
- `cuda` is **rejected on Apple** targets.
- `vulkan` requires the Vulkan toolchain to be discoverable by CMake, including `glslc`.

Default crate features enable **Metal** and **Vulkan** where applicable; **`blas` is opt-in** (OpenBLAS on Linux/Windows makes CI and local builds much slower). On Apple, enabling `blas` uses Accelerate.

## Override ggml path

Set `GGML_SRC` to point at another checkout when experimenting:

```bash
export GGML_SRC=/path/to/ggml
cargo build -p ggml-sys
```

## `qwen3-tts`

- Direct Rust implementation on top of `ggml` / `ggml-sys`.
- Current implemented pieces:
  - GGUF validation and metadata access
  - direct tokenizer loading from GGUF metadata
  - `encode_for_tts()` request formatting
- Reference only:
  - [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) for module boundaries, metadata keys, and tensor naming

| Feature | Effect |
|---------|--------|
| `hf` | Hugging Face download helpers. |
| `metal` | Enables Metal in GGML. Runtime: with `QWEN3_TTS_BACKEND=auto` (default), Apple builds prefer Metal then CPU; set `QWEN3_TTS_BACKEND=metal` to require Metal. |
| `vulkan` | Enables Vulkan in GGML on all targets. Runtime: `auto` uses Vulkan only on **non-Apple** (then CPU fallback). On **Apple**, set **`QWEN3_TTS_BACKEND=vulkan`** to select Vulkan (MoltenVK); otherwise `auto` will not pick Vulkan. |
| `coreml` | Enables the ONNX Runtime CoreML execution provider for the vocoder. Runtime: `QWEN3_TTS_VOCODER_EP=auto` prefers CoreML on Apple platforms, otherwise CPU. |
| `directml` | Enables the ONNX Runtime DirectML execution provider for the vocoder. Runtime: `QWEN3_TTS_VOCODER_EP=auto` prefers DirectML on Windows builds with this feature, otherwise CPU. |
| `native` | Enable host-CPU-specific ggml kernels for less portable but faster CPU binaries. |

## Vulkan prerequisites

- Linux: install a Vulkan loader / headers plus `glslc` before building `--features vulkan`. If you enable `blas`, install a BLAS implementation (e.g. OpenBLAS) and set `GGML_BLAS_VENDOR` / `BLA_VENDOR` as needed for CMake.
- Windows: for `blas`, install OpenBLAS (for example via vcpkg), point CMake at it, and set `GGML_BLAS_VENDOR` / `BLA_VENDOR` if required.
- Apple: use **`QWEN3_TTS_BACKEND`** to pick the GGML primary backend (`auto` \| `cpu` \| `metal` \| `vulkan`). `cargo xtask profile <cpu|metal|vulkan>` sets this env for the CLI child so profiling matches the intended backend.
