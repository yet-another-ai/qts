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

| Feature | CPU | GPU | NPU | GGML | Vocoder | Effect |
|---------|-----|-----|-----|------|---------|--------|
| `hf` | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | Hugging Face download helpers. |
| `native` | ✅ | ⬜ | ⬜ | ✅ | ⬜ | Enable host-CPU-specific ggml kernels for less portable but faster CPU binaries. |
| `metal` | ⬜ | ✅ | ⬜ | ✅ | ⬜ | Enables Metal in GGML. Runtime: with `QWEN3_TTS_BACKEND=auto` (default), Apple builds prefer Metal then CPU; set `QWEN3_TTS_BACKEND=metal` to require Metal. |
| `vulkan` | ⬜ | ✅ | ⬜ | ✅ | ⬜ | Enables Vulkan in GGML on all targets. Runtime: `auto` uses Vulkan only on **non-Apple** (then CPU fallback). On **Apple**, set **`QWEN3_TTS_BACKEND=vulkan`** to select Vulkan (MoltenVK); otherwise `auto` will not pick Vulkan. |
| `acl` | ✅ | ✅ | ⬜ | ⬜ | ✅ | Enables the ONNX Runtime ACL execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=acl` or include `acl` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `armnn` | ✅ | ✅ | ✅ | ⬜ | ✅ | Enables the ONNX Runtime ArmNN execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=armnn` or include `armnn` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `azure` | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | Enables the ONNX Runtime Azure execution provider for the vocoder. This is a remote / service-backed EP, so it does not map cleanly to local CPU / GPU / NPU buckets. Runtime: select with `QWEN3_TTS_VOCODER_EP=azure` or include `azure` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `cann` | ⬜ | ⬜ | ✅ | ⬜ | ✅ | Enables the ONNX Runtime CANN execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=cann` or include `cann` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `coreml` | ⬜ | ✅ | ✅ | ⬜ | ✅ | Enables the ONNX Runtime CoreML execution provider for the vocoder. Runtime: `QWEN3_TTS_VOCODER_EP=auto` prefers CoreML on Apple platforms; you can also require it with `QWEN3_TTS_VOCODER_EP=coreml`. |
| `cuda` | ⬜ | ✅ | ⬜ | ⬜ | ✅ | Enables the ONNX Runtime CUDA execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=cuda` or include `cuda` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `directml` | ⬜ | ✅ | ✅ | ⬜ | ✅ | Enables the ONNX Runtime DirectML execution provider for the vocoder. Runtime: `QWEN3_TTS_VOCODER_EP=auto` prefers DirectML on Windows builds with this feature; you can also require it with `QWEN3_TTS_VOCODER_EP=directml`. |
| `migraphx` | ⬜ | ✅ | ⬜ | ⬜ | ✅ | Enables the ONNX Runtime MIGraphX execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=migraphx` or include `migraphx` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `nnapi` | ⬜ | ✅ | ✅ | ⬜ | ✅ | Enables the ONNX Runtime NNAPI execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=nnapi` or include `nnapi` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `nvrtx` | ⬜ | ✅ | ⬜ | ⬜ | ✅ | Enables the ONNX Runtime NVRTX execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=nvrtx` or include `nvrtx` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `onednn` | ✅ | ✅ | ⬜ | ⬜ | ✅ | Enables the ONNX Runtime oneDNN execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=onednn` or include `onednn` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `openvino` | ✅ | ✅ | ✅ | ⬜ | ✅ | Enables the ONNX Runtime OpenVINO execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=openvino` or include `openvino` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `qnn` | ⬜ | ✅ | ✅ | ⬜ | ✅ | Enables the ONNX Runtime QNN execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=qnn` or include `qnn` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `rknpu` | ⬜ | ⬜ | ✅ | ⬜ | ✅ | Enables the ONNX Runtime RKNPU execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=rknpu` or include `rknpu` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `tensorrt` | ⬜ | ✅ | ⬜ | ⬜ | ✅ | Enables the ONNX Runtime TensorRT execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=tensorrt` or include `tensorrt` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `tvm` | ✅ | ✅ | ✅ | ⬜ | ✅ | Enables the ONNX Runtime TVM execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=tvm` or include `tvm` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `vitis` | ⬜ | ⬜ | ✅ | ⬜ | ✅ | Enables the ONNX Runtime Vitis execution provider for the vocoder. It is often associated with FPGA / accelerator flows; the checkbox marks the closest fit here as NPU-style offload. Runtime: select with `QWEN3_TTS_VOCODER_EP=vitis` or include `vitis` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `webgpu` | ⬜ | ✅ | ⬜ | ⬜ | ✅ | Enables the ONNX Runtime WebGPU execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=webgpu` or include `webgpu` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |
| `xnnpack` | ✅ | ⬜ | ⬜ | ⬜ | ✅ | Enables the ONNX Runtime XNNPACK execution provider for the vocoder. Runtime: select with `QWEN3_TTS_VOCODER_EP=xnnpack` or include `xnnpack` in `QWEN3_TTS_VOCODER_EP_FALLBACK`. |

The table is intentionally **feature-first** so a future backend can list multiple consumers in the same row once it is used by both GGML and the vocoder.

For ONNX Runtime features, remember that Cargo features alone do not guarantee a matching prebuilt ORT binary exists. ort documents prebuilt bundles for platform-native EPs like `directml`, `xnnpack`, and `coreml`, plus separate bundles for `cuda` + `tensorrt`, `webgpu`, and `nvrtx`. If you combine EP families outside those bundles, you may need to compile ONNX Runtime from source to avoid falling back to a CPU-only runtime.

Experimental runtime knobs:
- `QWEN3_TTS_TALKER_KV_MODE=f16|turboquant` switches the talker KV cache path at runtime.
- `turboquant` uses quantized GGML storage for the talker KV cache on the selected backend, with host-side quantization plus backend upload during KV write-back.

## Vulkan prerequisites

- Linux: install a Vulkan loader / headers plus `glslc` before building `--features vulkan`. If you enable `blas`, install a BLAS implementation (e.g. OpenBLAS) and set `GGML_BLAS_VENDOR` / `BLA_VENDOR` as needed for CMake.
- Windows: for `blas`, install OpenBLAS (for example via vcpkg), point CMake at it, and set `GGML_BLAS_VENDOR` / `BLA_VENDOR` if required.
- Apple: use **`QWEN3_TTS_BACKEND`** to pick the GGML primary backend (`auto` \| `cpu` \| `metal` \| `vulkan`). `cargo xtask profile <cpu|metal|vulkan>` sets this env for the CLI child so profiling matches the intended backend.
