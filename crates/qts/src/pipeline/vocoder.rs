//! ONNX Runtime vocoder wrapper for exported Qwen3-TTS speech tokenizer decoder.

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::{Mutex, OnceLock};

use ort::ep::ExecutionProvider;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::IoBinding;
use ort::session::{
    builder::{GraphOptimizationLevel, SessionBuilder},
    Session,
};
use ort::value::Tensor;

use super::backend::BackendKind;
use crate::Qwen3TtsError;

fn ort_err(err: impl std::fmt::Display) -> Qwen3TtsError {
    Qwen3TtsError::Ort(err.to_string())
}

fn ensure_ort_init() -> Result<(), Qwen3TtsError> {
    static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();
    ORT_INIT
        .get_or_init(|| {
            let _ = ort::init().commit();
            Ok(())
        })
        .as_ref()
        .map_err(|err| Qwen3TtsError::Ort(err.clone()))?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestedExecutionProvider {
    Auto,
    Explicit(VocoderExecutionProvider),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionProviderParseError {
    Unknown,
    MissingFeature(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocoderExecutionProvider {
    Cpu,
    #[cfg(feature = "acl")]
    Acl,
    #[cfg(feature = "armnn")]
    ArmNn,
    #[cfg(feature = "azure")]
    Azure,
    #[cfg(feature = "cann")]
    Cann,
    #[cfg(feature = "coreml")]
    CoreMl,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "directml")]
    DirectMl,
    #[cfg(feature = "migraphx")]
    MigraphX,
    #[cfg(feature = "nnapi")]
    Nnapi,
    #[cfg(feature = "nvrtx")]
    Nvrtx,
    #[cfg(feature = "onednn")]
    OneDnn,
    #[cfg(feature = "openvino")]
    OpenVino,
    #[cfg(feature = "qnn")]
    Qnn,
    #[cfg(feature = "rknpu")]
    Rknpu,
    #[cfg(feature = "tensorrt")]
    TensorRt,
    #[cfg(feature = "tvm")]
    Tvm,
    #[cfg(feature = "vitis")]
    Vitis,
    #[cfg(feature = "webgpu")]
    WebGpu,
    #[cfg(feature = "xnnpack")]
    Xnnpack,
}

impl VocoderExecutionProvider {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "acl")]
            Self::Acl => "acl",
            #[cfg(feature = "armnn")]
            Self::ArmNn => "armnn",
            #[cfg(feature = "azure")]
            Self::Azure => "azure",
            #[cfg(feature = "cann")]
            Self::Cann => "cann",
            #[cfg(feature = "coreml")]
            Self::CoreMl => "coreml",
            #[cfg(feature = "cuda")]
            Self::Cuda => "cuda",
            #[cfg(feature = "directml")]
            Self::DirectMl => "directml",
            #[cfg(feature = "migraphx")]
            Self::MigraphX => "migraphx",
            #[cfg(feature = "nnapi")]
            Self::Nnapi => "nnapi",
            #[cfg(feature = "nvrtx")]
            Self::Nvrtx => "nvrtx",
            #[cfg(feature = "onednn")]
            Self::OneDnn => "onednn",
            #[cfg(feature = "openvino")]
            Self::OpenVino => "openvino",
            #[cfg(feature = "qnn")]
            Self::Qnn => "qnn",
            #[cfg(feature = "rknpu")]
            Self::Rknpu => "rknpu",
            #[cfg(feature = "tensorrt")]
            Self::TensorRt => "tensorrt",
            #[cfg(feature = "tvm")]
            Self::Tvm => "tvm",
            #[cfg(feature = "vitis")]
            Self::Vitis => "vitis",
            #[cfg(feature = "webgpu")]
            Self::WebGpu => "webgpu",
            #[cfg(feature = "xnnpack")]
            Self::Xnnpack => "xnnpack",
        }
    }

    #[must_use]
    pub fn display_str(self) -> &'static str {
        match self {
            Self::Cpu => "ORT/CPU",
            #[cfg(feature = "acl")]
            Self::Acl => "ORT/ACL",
            #[cfg(feature = "armnn")]
            Self::ArmNn => "ORT/ArmNN",
            #[cfg(feature = "azure")]
            Self::Azure => "ORT/Azure",
            #[cfg(feature = "cann")]
            Self::Cann => "ORT/CANN",
            #[cfg(feature = "coreml")]
            Self::CoreMl => "ORT/CoreML",
            #[cfg(feature = "cuda")]
            Self::Cuda => "ORT/CUDA",
            #[cfg(feature = "directml")]
            Self::DirectMl => "ORT/DirectML",
            #[cfg(feature = "migraphx")]
            Self::MigraphX => "ORT/MIGraphX",
            #[cfg(feature = "nnapi")]
            Self::Nnapi => "ORT/NNAPI",
            #[cfg(feature = "nvrtx")]
            Self::Nvrtx => "ORT/NVRTX",
            #[cfg(feature = "onednn")]
            Self::OneDnn => "ORT/oneDNN",
            #[cfg(feature = "openvino")]
            Self::OpenVino => "ORT/OpenVINO",
            #[cfg(feature = "qnn")]
            Self::Qnn => "ORT/QNN",
            #[cfg(feature = "rknpu")]
            Self::Rknpu => "ORT/RKNPU",
            #[cfg(feature = "tensorrt")]
            Self::TensorRt => "ORT/TensorRT",
            #[cfg(feature = "tvm")]
            Self::Tvm => "ORT/TVM",
            #[cfg(feature = "vitis")]
            Self::Vitis => "ORT/Vitis",
            #[cfg(feature = "webgpu")]
            Self::WebGpu => "ORT/WebGPU",
            #[cfg(feature = "xnnpack")]
            Self::Xnnpack => "ORT/XNNPACK",
        }
    }

    #[must_use]
    fn expected_values() -> &'static str {
        "cpu, acl, armnn, azure, cann, coreml, cuda, directml, migraphx, nnapi, nvrtx, onednn, openvino, qnn, rknpu, tensorrt, tvm, vitis, webgpu, xnnpack"
    }

    fn parse_token(value: &str) -> Result<Self, ExecutionProviderParseError> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "acl" => {
                #[cfg(feature = "acl")]
                {
                    Ok(Self::Acl)
                }
                #[cfg(not(feature = "acl"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("acl"))
                }
            }
            "armnn" => {
                #[cfg(feature = "armnn")]
                {
                    Ok(Self::ArmNn)
                }
                #[cfg(not(feature = "armnn"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("armnn"))
                }
            }
            "azure" => {
                #[cfg(feature = "azure")]
                {
                    Ok(Self::Azure)
                }
                #[cfg(not(feature = "azure"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("azure"))
                }
            }
            "cann" => {
                #[cfg(feature = "cann")]
                {
                    Ok(Self::Cann)
                }
                #[cfg(not(feature = "cann"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("cann"))
                }
            }
            "coreml" => {
                #[cfg(feature = "coreml")]
                {
                    Ok(Self::CoreMl)
                }
                #[cfg(not(feature = "coreml"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("coreml"))
                }
            }
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Self::Cuda)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("cuda"))
                }
            }
            "directml" => {
                #[cfg(feature = "directml")]
                {
                    Ok(Self::DirectMl)
                }
                #[cfg(not(feature = "directml"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("directml"))
                }
            }
            "migraphx" => {
                #[cfg(feature = "migraphx")]
                {
                    Ok(Self::MigraphX)
                }
                #[cfg(not(feature = "migraphx"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("migraphx"))
                }
            }
            "nnapi" => {
                #[cfg(feature = "nnapi")]
                {
                    Ok(Self::Nnapi)
                }
                #[cfg(not(feature = "nnapi"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("nnapi"))
                }
            }
            "nvrtx" => {
                #[cfg(feature = "nvrtx")]
                {
                    Ok(Self::Nvrtx)
                }
                #[cfg(not(feature = "nvrtx"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("nvrtx"))
                }
            }
            "onednn" => {
                #[cfg(feature = "onednn")]
                {
                    Ok(Self::OneDnn)
                }
                #[cfg(not(feature = "onednn"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("onednn"))
                }
            }
            "openvino" => {
                #[cfg(feature = "openvino")]
                {
                    Ok(Self::OpenVino)
                }
                #[cfg(not(feature = "openvino"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("openvino"))
                }
            }
            "qnn" => {
                #[cfg(feature = "qnn")]
                {
                    Ok(Self::Qnn)
                }
                #[cfg(not(feature = "qnn"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("qnn"))
                }
            }
            "rknpu" => {
                #[cfg(feature = "rknpu")]
                {
                    Ok(Self::Rknpu)
                }
                #[cfg(not(feature = "rknpu"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("rknpu"))
                }
            }
            "tensorrt" => {
                #[cfg(feature = "tensorrt")]
                {
                    Ok(Self::TensorRt)
                }
                #[cfg(not(feature = "tensorrt"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("tensorrt"))
                }
            }
            "tvm" => {
                #[cfg(feature = "tvm")]
                {
                    Ok(Self::Tvm)
                }
                #[cfg(not(feature = "tvm"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("tvm"))
                }
            }
            "vitis" => {
                #[cfg(feature = "vitis")]
                {
                    Ok(Self::Vitis)
                }
                #[cfg(not(feature = "vitis"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("vitis"))
                }
            }
            "webgpu" => {
                #[cfg(feature = "webgpu")]
                {
                    Ok(Self::WebGpu)
                }
                #[cfg(not(feature = "webgpu"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("webgpu"))
                }
            }
            "xnnpack" => {
                #[cfg(feature = "xnnpack")]
                {
                    Ok(Self::Xnnpack)
                }
                #[cfg(not(feature = "xnnpack"))]
                {
                    Err(ExecutionProviderParseError::MissingFeature("xnnpack"))
                }
            }
            _ => Err(ExecutionProviderParseError::Unknown),
        }
    }
}

fn parse_requested_execution_provider() -> Result<RequestedExecutionProvider, Qwen3TtsError> {
    let Some(raw) = env::var_os("QWEN3_TTS_VOCODER_EP") else {
        return Ok(RequestedExecutionProvider::Auto);
    };
    let value = raw.to_string_lossy();
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(RequestedExecutionProvider::Auto),
        other => match VocoderExecutionProvider::parse_token(other) {
            Ok(provider) => Ok(RequestedExecutionProvider::Explicit(provider)),
            Err(ExecutionProviderParseError::MissingFeature(feature)) => {
                Err(Qwen3TtsError::InvalidInput(format!(
                    "unsupported QWEN3_TTS_VOCODER_EP={other}; binary was built without the {feature} feature"
                )))
            }
            Err(ExecutionProviderParseError::Unknown) => Err(Qwen3TtsError::InvalidInput(
                format!(
                    "unsupported QWEN3_TTS_VOCODER_EP={other}; expected auto or one of {}",
                    VocoderExecutionProvider::expected_values()
                ),
            )),
        },
    }
}

fn default_auto_execution_provider_order() -> Vec<VocoderExecutionProvider> {
    let mut order = Vec::new();

    #[cfg(target_vendor = "apple")]
    {
        #[cfg(feature = "coreml")]
        {
            order.push(VocoderExecutionProvider::CoreMl);
        }
    }

    #[cfg(all(not(target_vendor = "apple"), feature = "cuda"))]
    {
        order.push(VocoderExecutionProvider::Cuda);
    }

    #[cfg(all(not(target_vendor = "apple"), feature = "nvrtx"))]
    {
        order.push(VocoderExecutionProvider::Nvrtx);
    }

    #[cfg(all(not(target_vendor = "apple"), feature = "tensorrt"))]
    {
        order.push(VocoderExecutionProvider::TensorRt);
    }

    #[cfg(all(target_os = "windows", feature = "directml"))]
    {
        order.push(VocoderExecutionProvider::DirectMl);
    }

    order.push(VocoderExecutionProvider::Cpu);
    order
}

fn parse_auto_execution_provider_order() -> Result<Vec<VocoderExecutionProvider>, Qwen3TtsError> {
    let var = match env::var("QWEN3_TTS_VOCODER_EP_FALLBACK") {
        Ok(s) if !s.trim().is_empty() => s,
        _ => return Ok(default_auto_execution_provider_order()),
    };
    let mut order = Vec::new();
    for token in var.split(',') {
        let value = token.trim().to_ascii_lowercase();
        if value.is_empty() {
            continue;
        }
        let provider = match VocoderExecutionProvider::parse_token(&value) {
            Ok(provider) => provider,
            Err(ExecutionProviderParseError::MissingFeature(feature)) => {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "QWEN3_TTS_VOCODER_EP_FALLBACK includes {value}, but the binary was built without the {feature} feature"
                )));
            }
            Err(ExecutionProviderParseError::Unknown) => {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "QWEN3_TTS_VOCODER_EP_FALLBACK: unknown EP '{value}' (expected one of {})",
                    VocoderExecutionProvider::expected_values()
                )));
            }
        };
        if !order.contains(&provider) {
            order.push(provider);
        }
    }
    if order.is_empty() {
        return Err(Qwen3TtsError::InvalidInput(
            "QWEN3_TTS_VOCODER_EP_FALLBACK must contain at least one execution provider".into(),
        ));
    }
    Ok(order)
}

#[derive(Debug, Clone)]
pub struct VocoderConfig {
    pub sample_rate: i32,
    pub n_codebooks: i32,
    pub codebook_size: i32,
    pub codebook_dim: i32,
    pub latent_dim: i32,
    pub hidden_dim: i32,
    pub n_pre_tfm_layers: i32,
    pub n_heads: i32,
    pub ffn_dim: i32,
    pub decoder_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            n_codebooks: 16,
            codebook_size: 2_048,
            codebook_dim: 256,
            latent_dim: 1_024,
            hidden_dim: 512,
            n_pre_tfm_layers: 8,
            n_heads: 16,
            ffn_dim: 1_024,
            decoder_dim: 1_536,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VocoderGraphTemplate {
    n_frames: usize,
}

struct CachedVocoderTemplate {
    input: Tensor<i64>,
    binding: IoBinding,
}

struct VocoderSessionState {
    session: Session,
    templates: HashMap<usize, CachedVocoderTemplate>,
}

pub struct Vocoder {
    config: VocoderConfig,
    model_path: PathBuf,
    execution_provider: VocoderExecutionProvider,
    sessions: Mutex<HashMap<usize, VocoderSessionState>>,
}

impl Vocoder {
    pub fn load_from_onnx(path: impl AsRef<Path>) -> Result<Self, Qwen3TtsError> {
        let path = path.as_ref().to_path_buf();
        if !path.is_file() {
            return Err(Qwen3TtsError::ModelFile(path));
        }

        ensure_ort_init()?;
        let (default_session, execution_provider) = Self::build_session(&path, 1)?;
        let mut sessions = HashMap::new();
        sessions.insert(
            1,
            VocoderSessionState {
                session: default_session,
                templates: HashMap::new(),
            },
        );

        Ok(Self {
            config: VocoderConfig::default(),
            model_path: path,
            execution_provider,
            sessions: Mutex::new(sessions),
        })
    }

    #[must_use]
    pub fn primary_backend_kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    #[must_use]
    pub fn execution_provider(&self) -> VocoderExecutionProvider {
        self.execution_provider
    }

    #[must_use]
    pub fn execution_provider_label(&self) -> &'static str {
        self.execution_provider.display_str()
    }

    #[must_use]
    pub fn config(&self) -> &VocoderConfig {
        &self.config
    }

    pub fn decode(
        &self,
        codes: &[i32],
        n_frames: usize,
        thread_count: usize,
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        let mut template = self.build_decode_template(n_frames)?;
        self.decode_with_template(&mut template, codes, thread_count)
    }

    pub fn build_decode_template(
        &self,
        n_frames: usize,
    ) -> Result<VocoderGraphTemplate, Qwen3TtsError> {
        Ok(VocoderGraphTemplate { n_frames })
    }

    pub fn decode_with_template(
        &self,
        template: &mut VocoderGraphTemplate,
        codes: &[i32],
        thread_count: usize,
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        let n_codebooks = self.config.n_codebooks as usize;
        let expected_codes = template
            .n_frames
            .checked_mul(n_codebooks)
            .ok_or_else(|| Qwen3TtsError::InvalidInput("vocoder input shape overflow".into()))?;
        if codes.len() != expected_codes {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "expected {} codec ids for {} frames with {} codebooks, got {}",
                expected_codes,
                template.n_frames,
                n_codebooks,
                codes.len()
            )));
        }

        let key = thread_count.max(1);
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| Qwen3TtsError::InvalidInput("failed to lock ORT session cache".into()))?;
        if let std::collections::hash_map::Entry::Vacant(e) = sessions.entry(key) {
            let (session, actual_ep) = Self::build_session(&self.model_path, key)?;
            if actual_ep != self.execution_provider {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "ORT execution provider mismatch across sessions: expected {}, got {}",
                    self.execution_provider.as_str(),
                    actual_ep.as_str()
                )));
            }
            e.insert(VocoderSessionState {
                session,
                templates: HashMap::new(),
            });
        }

        let state = sessions
            .get_mut(&key)
            .ok_or_else(|| Qwen3TtsError::InvalidInput("missing ORT session".into()))?;
        if !state.templates.contains_key(&template.n_frames) {
            let cached = Self::create_cached_template(
                &state.session,
                template.n_frames,
                self.config.n_codebooks as usize,
                &self.model_path,
            )?;
            state.templates.insert(template.n_frames, cached);
        }
        let cached = state
            .templates
            .get_mut(&template.n_frames)
            .ok_or_else(|| Qwen3TtsError::InvalidInput("missing vocoder template".into()))?;
        Self::write_codes_into_template(&mut cached.input, codes);
        let input_name = state
            .session
            .inputs()
            .first()
            .ok_or_else(|| Qwen3TtsError::InvalidOnnx(self.model_path.clone()))?
            .name()
            .to_owned();
        cached
            .binding
            .bind_input(input_name, &cached.input)
            .map_err(ort_err)?;
        let outputs = state
            .session
            .run_binding(&cached.binding)
            .map_err(ort_err)?;
        if outputs.len() < 2 {
            return Err(Qwen3TtsError::InvalidOnnx(self.model_path.clone()));
        }

        let (_audio_shape, audio_values) =
            outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        let (_length_shape, audio_lengths) =
            outputs[1].try_extract_tensor::<i64>().map_err(ort_err)?;
        let sample_count = audio_lengths
            .first()
            .copied()
            .unwrap_or(audio_values.len() as i64)
            .clamp(0, audio_values.len() as i64) as usize;
        Ok(audio_values[..sample_count].to_vec())
    }

    fn create_cached_template(
        session: &Session,
        n_frames: usize,
        n_codebooks: usize,
        model_path: &Path,
    ) -> Result<CachedVocoderTemplate, Qwen3TtsError> {
        let input = Tensor::<i64>::new(
            &ort::memory::Allocator::default(),
            [1usize, n_frames, n_codebooks],
        )
        .map_err(ort_err)?;
        let mut binding = session.create_binding().map_err(ort_err)?;
        let cpu_output = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )
        .map_err(ort_err)?;
        let audio_name = session
            .outputs()
            .first()
            .ok_or_else(|| Qwen3TtsError::InvalidOnnx(model_path.to_path_buf()))?
            .name()
            .to_owned();
        let length_name = session
            .outputs()
            .get(1)
            .ok_or_else(|| Qwen3TtsError::InvalidOnnx(model_path.to_path_buf()))?
            .name()
            .to_owned();
        binding
            .bind_output_to_device(audio_name, &cpu_output)
            .map_err(ort_err)?;
        binding
            .bind_output_to_device(length_name, &cpu_output)
            .map_err(ort_err)?;
        Ok(CachedVocoderTemplate { input, binding })
    }

    fn write_codes_into_template(input: &mut Tensor<i64>, codes: &[i32]) {
        let input_ptr = input.data_ptr_mut().cast::<i64>();
        let input_values = unsafe { slice::from_raw_parts_mut(input_ptr, codes.len()) };
        for (dst, src) in input_values.iter_mut().zip(codes.iter().copied()) {
            *dst = i64::from(src);
        }
    }

    fn build_session(
        path: &Path,
        thread_count: usize,
    ) -> Result<(Session, VocoderExecutionProvider), Qwen3TtsError> {
        ensure_ort_init()?;

        match parse_requested_execution_provider()? {
            RequestedExecutionProvider::Explicit(provider) => {
                let session = Self::build_session_for_provider(path, thread_count, provider, true)?;
                Ok((session, provider))
            }
            RequestedExecutionProvider::Auto => {
                let order = parse_auto_execution_provider_order()?;
                let mut last_error = None;
                for provider in order {
                    match Self::build_session_for_provider(path, thread_count, provider, false) {
                        Ok(session) => return Ok((session, provider)),
                        Err(err) => last_error = Some(err),
                    }
                }
                Err(last_error.unwrap_or_else(|| {
                    Qwen3TtsError::InvalidInput(
                        "QWEN3_TTS_VOCODER_EP_FALLBACK did not contain a usable execution provider"
                            .into(),
                    )
                }))
            }
        }
    }

    fn build_session_for_provider(
        path: &Path,
        thread_count: usize,
        provider: VocoderExecutionProvider,
        required: bool,
    ) -> Result<Session, Qwen3TtsError> {
        let mut builder = Session::builder().map_err(ort_err)?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?;
        if thread_count > 0 {
            builder = builder.with_intra_threads(thread_count).map_err(ort_err)?;
        }
        Self::register_execution_provider(&mut builder, provider, required)?;
        builder.commit_from_file(path).map_err(ort_err)
    }

    #[allow(unused_variables)]
    fn register_execution_provider(
        builder: &mut SessionBuilder,
        provider: VocoderExecutionProvider,
        required: bool,
    ) -> Result<(), Qwen3TtsError> {
        match provider {
            VocoderExecutionProvider::Cpu => Ok(()),
            #[cfg(feature = "acl")]
            VocoderExecutionProvider::Acl => Self::register_acl(builder, required),
            #[cfg(feature = "armnn")]
            VocoderExecutionProvider::ArmNn => Self::register_armnn(builder, required),
            #[cfg(feature = "azure")]
            VocoderExecutionProvider::Azure => Self::register_azure(builder, required),
            #[cfg(feature = "cann")]
            VocoderExecutionProvider::Cann => Self::register_cann(builder, required),
            #[cfg(feature = "coreml")]
            VocoderExecutionProvider::CoreMl => Self::register_coreml(builder, required),
            #[cfg(feature = "cuda")]
            VocoderExecutionProvider::Cuda => Self::register_cuda(builder, required),
            #[cfg(feature = "directml")]
            VocoderExecutionProvider::DirectMl => Self::register_directml(builder, required),
            #[cfg(feature = "migraphx")]
            VocoderExecutionProvider::MigraphX => Self::register_migraphx(builder, required),
            #[cfg(feature = "nnapi")]
            VocoderExecutionProvider::Nnapi => Self::register_nnapi(builder, required),
            #[cfg(feature = "nvrtx")]
            VocoderExecutionProvider::Nvrtx => Self::register_nvrtx(builder, required),
            #[cfg(feature = "onednn")]
            VocoderExecutionProvider::OneDnn => Self::register_onednn(builder, required),
            #[cfg(feature = "openvino")]
            VocoderExecutionProvider::OpenVino => Self::register_openvino(builder, required),
            #[cfg(feature = "qnn")]
            VocoderExecutionProvider::Qnn => Self::register_qnn(builder, required),
            #[cfg(feature = "rknpu")]
            VocoderExecutionProvider::Rknpu => Self::register_rknpu(builder, required),
            #[cfg(feature = "tensorrt")]
            VocoderExecutionProvider::TensorRt => Self::register_tensorrt(builder, required),
            #[cfg(feature = "tvm")]
            VocoderExecutionProvider::Tvm => Self::register_tvm(builder, required),
            #[cfg(feature = "vitis")]
            VocoderExecutionProvider::Vitis => Self::register_vitis(builder, required),
            #[cfg(feature = "webgpu")]
            VocoderExecutionProvider::WebGpu => Self::register_webgpu(builder, required),
            #[cfg(feature = "xnnpack")]
            VocoderExecutionProvider::Xnnpack => Self::register_xnnpack(builder, required),
        }
    }

    #[allow(dead_code)]
    fn register_ort_execution_provider<E: ExecutionProvider>(
        builder: &mut SessionBuilder,
        required: bool,
        provider: E,
        provider_token: &'static str,
        provider_label: &'static str,
        supported_platforms: &'static str,
    ) -> Result<(), Qwen3TtsError> {
        if !provider.supported_by_platform() {
            if required {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "QWEN3_TTS_VOCODER_EP={provider_token} is only supported on {supported_platforms}"
                )));
            }
            return Err(Qwen3TtsError::InvalidInput(format!(
                "{provider_token} EP is not supported on this platform"
            )));
        }
        if !provider.is_available().map_err(ort_err)? {
            if required {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "QWEN3_TTS_VOCODER_EP={provider_token} requested, but this build of ONNX Runtime does not include {provider_label}"
                )));
            }
            return Err(Qwen3TtsError::InvalidInput(format!(
                "{provider_token} EP is not available in this ONNX Runtime build"
            )));
        }
        provider.register(builder).map_err(ort_err)
    }

    #[cfg(feature = "acl")]
    fn register_acl(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::ACL::default(),
            "acl",
            "ACL",
            "Arm platforms",
        )
    }

    #[cfg(feature = "armnn")]
    #[allow(deprecated)]
    fn register_armnn(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::ArmNN::default(),
            "armnn",
            "ArmNN",
            "Arm platforms",
        )
    }

    #[cfg(feature = "azure")]
    fn register_azure(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::Azure::default(),
            "azure",
            "Azure",
            "supported native platforms",
        )
    }

    #[cfg(feature = "cann")]
    fn register_cann(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::CANN::default(),
            "cann",
            "CANN",
            "Linux",
        )
    }

    #[cfg(feature = "coreml")]
    fn register_coreml(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::CoreML::default(),
            "coreml",
            "CoreML",
            "Apple platforms",
        )
    }

    #[cfg(feature = "cuda")]
    fn register_cuda(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::CUDA::default(),
            "cuda",
            "CUDA",
            "Windows or Linux",
        )
    }

    #[cfg(feature = "directml")]
    fn register_directml(
        builder: &mut SessionBuilder,
        required: bool,
    ) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::DirectML::default(),
            "directml",
            "DirectML",
            "Windows",
        )
    }

    #[cfg(feature = "migraphx")]
    fn register_migraphx(
        builder: &mut SessionBuilder,
        required: bool,
    ) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::MIGraphX::default(),
            "migraphx",
            "MIGraphX",
            "Linux",
        )
    }

    #[cfg(feature = "nnapi")]
    fn register_nnapi(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::NNAPI::default(),
            "nnapi",
            "NNAPI",
            "Android",
        )
    }

    #[cfg(feature = "nvrtx")]
    fn register_nvrtx(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::NVRTX::default(),
            "nvrtx",
            "NVTX",
            "supported native platforms",
        )
    }

    #[cfg(feature = "onednn")]
    fn register_onednn(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::OneDNN::default(),
            "onednn",
            "oneDNN",
            "supported native platforms",
        )
    }

    #[cfg(feature = "openvino")]
    fn register_openvino(
        builder: &mut SessionBuilder,
        required: bool,
    ) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::OpenVINO::default(),
            "openvino",
            "OpenVINO",
            "x86_64 Windows or Linux",
        )
    }

    #[cfg(feature = "qnn")]
    fn register_qnn(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::QNN::default(),
            "qnn",
            "QNN",
            "Windows, Linux, or Android",
        )
    }

    #[cfg(feature = "rknpu")]
    fn register_rknpu(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::RKNPU::default(),
            "rknpu",
            "RKNPU",
            "Linux",
        )
    }

    #[cfg(feature = "tensorrt")]
    fn register_tensorrt(
        builder: &mut SessionBuilder,
        required: bool,
    ) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::TensorRT::default(),
            "tensorrt",
            "TensorRT",
            "Windows or Linux",
        )
    }

    #[cfg(feature = "tvm")]
    fn register_tvm(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::TVM::default(),
            "tvm",
            "TVM",
            "supported native platforms",
        )
    }

    #[cfg(feature = "vitis")]
    fn register_vitis(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::Vitis::default(),
            "vitis",
            "Vitis",
            "Linux",
        )
    }

    #[cfg(feature = "webgpu")]
    fn register_webgpu(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::WebGPU::default(),
            "webgpu",
            "WebGPU",
            "supported native platforms",
        )
    }

    #[cfg(feature = "xnnpack")]
    fn register_xnnpack(builder: &mut SessionBuilder, required: bool) -> Result<(), Qwen3TtsError> {
        Self::register_ort_execution_provider(
            builder,
            required,
            ort::ep::XNNPACK::default(),
            "xnnpack",
            "XNNPACK",
            "supported Arm or x86_64 platforms",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env_var(name: &str, value: Option<&str>, f: impl FnOnce()) {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let old = env::var_os(name);
        match value {
            Some(value) => unsafe { env::set_var(name, value) },
            None => unsafe { env::remove_var(name) },
        }
        f();
        match old {
            Some(old) => unsafe { env::set_var(name, old) },
            None => unsafe { env::remove_var(name) },
        }
    }

    #[test]
    fn parses_cpu_explicit_ep() {
        with_env_var("QWEN3_TTS_VOCODER_EP", Some("cpu"), || {
            assert_eq!(
                parse_requested_execution_provider().expect("parse should succeed"),
                RequestedExecutionProvider::Explicit(VocoderExecutionProvider::Cpu)
            );
        });
    }

    #[test]
    fn deduplicates_fallback_order() {
        with_env_var("QWEN3_TTS_VOCODER_EP_FALLBACK", Some("cpu,cpu"), || {
            assert_eq!(
                parse_auto_execution_provider_order().expect("fallback parse should succeed"),
                vec![VocoderExecutionProvider::Cpu]
            );
        });
    }

    #[test]
    fn rejects_unknown_fallback_ep() {
        with_env_var("QWEN3_TTS_VOCODER_EP_FALLBACK", Some("mystery"), || {
            let err = parse_auto_execution_provider_order().expect_err("parse should fail");
            let msg = err.to_string();
            assert!(msg.contains("unknown EP 'mystery'"));
            assert!(msg.contains("xnnpack"));
        });
    }
}
