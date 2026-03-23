//! ONNX Runtime vocoder wrapper for exported Qwen3-TTS speech tokenizer decoder.

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

#[cfg(any(feature = "coreml", feature = "directml"))]
use ort::ep::ExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
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
pub enum VocoderExecutionProvider {
    Cpu,
    #[cfg(feature = "coreml")]
    CoreMl,
    #[cfg(feature = "directml")]
    DirectMl,
}

impl VocoderExecutionProvider {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "coreml")]
            Self::CoreMl => "coreml",
            #[cfg(feature = "directml")]
            Self::DirectMl => "directml",
        }
    }

    #[must_use]
    pub fn display_str(self) -> &'static str {
        match self {
            Self::Cpu => "ORT/CPU",
            #[cfg(feature = "coreml")]
            Self::CoreMl => "ORT/CoreML",
            #[cfg(feature = "directml")]
            Self::DirectMl => "ORT/DirectML",
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
        "cpu" => Ok(RequestedExecutionProvider::Explicit(
            VocoderExecutionProvider::Cpu,
        )),
        "coreml" => {
            #[cfg(feature = "coreml")]
            {
                Ok(RequestedExecutionProvider::Explicit(
                    VocoderExecutionProvider::CoreMl,
                ))
            }
            #[cfg(not(feature = "coreml"))]
            {
                Err(Qwen3TtsError::InvalidInput(
                    "unsupported QWEN3_TTS_VOCODER_EP=coreml; binary was built without the coreml feature"
                        .into(),
                ))
            }
        }
        "directml" => {
            #[cfg(feature = "directml")]
            {
                Ok(RequestedExecutionProvider::Explicit(
                    VocoderExecutionProvider::DirectMl,
                ))
            }
            #[cfg(not(feature = "directml"))]
            {
                Err(Qwen3TtsError::InvalidInput(
                    "unsupported QWEN3_TTS_VOCODER_EP=directml; binary was built without the directml feature"
                        .into(),
                ))
            }
        }
        other => Err(Qwen3TtsError::InvalidInput(format!(
            "unsupported QWEN3_TTS_VOCODER_EP={other}; expected auto, cpu, coreml, or directml"
        ))),
    }
}

fn default_auto_execution_provider_order() -> Vec<VocoderExecutionProvider> {
    #[cfg(target_vendor = "apple")]
    {
        #[cfg(feature = "coreml")]
        {
            vec![
                VocoderExecutionProvider::CoreMl,
                VocoderExecutionProvider::Cpu,
            ]
        }
        #[cfg(not(feature = "coreml"))]
        {
            vec![VocoderExecutionProvider::Cpu]
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        #[cfg(target_os = "windows")]
        {
            #[cfg(feature = "directml")]
            {
                vec![
                    VocoderExecutionProvider::DirectMl,
                    VocoderExecutionProvider::Cpu,
                ]
            }
            #[cfg(not(feature = "directml"))]
            {
                vec![VocoderExecutionProvider::Cpu]
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            vec![VocoderExecutionProvider::Cpu]
        }
    }
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
        let provider = match value.as_str() {
            "cpu" => VocoderExecutionProvider::Cpu,
            "coreml" => {
                #[cfg(feature = "coreml")]
                {
                    VocoderExecutionProvider::CoreMl
                }
                #[cfg(not(feature = "coreml"))]
                {
                    return Err(Qwen3TtsError::InvalidInput(
                        "QWEN3_TTS_VOCODER_EP_FALLBACK includes coreml, but the binary was built without the coreml feature"
                            .into(),
                    ));
                }
            }
            "directml" => {
                #[cfg(feature = "directml")]
                {
                    VocoderExecutionProvider::DirectMl
                }
                #[cfg(not(feature = "directml"))]
                {
                    return Err(Qwen3TtsError::InvalidInput(
                        "QWEN3_TTS_VOCODER_EP_FALLBACK includes directml, but the binary was built without the directml feature"
                            .into(),
                    ));
                }
            }
            other => {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "QWEN3_TTS_VOCODER_EP_FALLBACK: unknown EP '{other}' (expected cpu, coreml, or directml)"
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

pub struct Vocoder {
    config: VocoderConfig,
    model_path: PathBuf,
    execution_provider: VocoderExecutionProvider,
    sessions: Mutex<HashMap<usize, Session>>,
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
        sessions.insert(1, default_session);

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
            e.insert(session);
        }

        let session = sessions
            .get_mut(&key)
            .ok_or_else(|| Qwen3TtsError::InvalidInput("missing ORT session".into()))?;
        let shape = vec![1usize, template.n_frames, n_codebooks];
        let input_codes = codes
            .iter()
            .copied()
            .map(i64::from)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let input = Tensor::from_array((shape, input_codes)).map_err(ort_err)?;
        let outputs = session.run(ort::inputs![input]).map_err(ort_err)?;
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
        builder: &mut ort::session::builder::SessionBuilder,
        provider: VocoderExecutionProvider,
        required: bool,
    ) -> Result<(), Qwen3TtsError> {
        match provider {
            VocoderExecutionProvider::Cpu => Ok(()),
            #[cfg(feature = "coreml")]
            VocoderExecutionProvider::CoreMl => Self::register_coreml(builder, required),
            #[cfg(feature = "directml")]
            VocoderExecutionProvider::DirectMl => Self::register_directml(builder, required),
        }
    }

    #[cfg(feature = "coreml")]
    fn register_coreml(
        builder: &mut ort::session::builder::SessionBuilder,
        required: bool,
    ) -> Result<(), Qwen3TtsError> {
        let coreml = ort::ep::CoreML::default();
        if !coreml.supported_by_platform() {
            if required {
                return Err(Qwen3TtsError::InvalidInput(
                    "QWEN3_TTS_VOCODER_EP=coreml is only supported on Apple platforms".into(),
                ));
            }
            return Err(Qwen3TtsError::InvalidInput(
                "coreml EP is not supported on this platform".into(),
            ));
        }
        if !coreml.is_available().map_err(ort_err)? {
            if required {
                return Err(Qwen3TtsError::InvalidInput(
                    "QWEN3_TTS_VOCODER_EP=coreml requested, but this build of ONNX Runtime does not include CoreML".into(),
                ));
            }
            return Err(Qwen3TtsError::InvalidInput(
                "coreml EP is not available in this ONNX Runtime build".into(),
            ));
        }
        match coreml.register(builder) {
            Ok(()) => Ok(()),
            Err(err) => Err(ort_err(err)),
        }
    }

    #[cfg(not(feature = "coreml"))]
    #[allow(dead_code)]
    fn register_coreml(
        _builder: &mut ort::session::builder::SessionBuilder,
        _required: bool,
    ) -> Result<(), Qwen3TtsError> {
        Err(Qwen3TtsError::InvalidInput(
            "coreml EP is unavailable because the binary was built without the coreml feature"
                .into(),
        ))
    }

    #[cfg(feature = "directml")]
    fn register_directml(
        builder: &mut ort::session::builder::SessionBuilder,
        required: bool,
    ) -> Result<(), Qwen3TtsError> {
        let directml = ort::ep::DirectML::default();
        if !directml.supported_by_platform() {
            if required {
                return Err(Qwen3TtsError::InvalidInput(
                    "QWEN3_TTS_VOCODER_EP=directml is only supported on Windows".into(),
                ));
            }
            return Err(Qwen3TtsError::InvalidInput(
                "directml EP is not supported on this platform".into(),
            ));
        }
        if !directml.is_available().map_err(ort_err)? {
            if required {
                return Err(Qwen3TtsError::InvalidInput(
                    "QWEN3_TTS_VOCODER_EP=directml requested, but this build of ONNX Runtime does not include DirectML".into(),
                ));
            }
            return Err(Qwen3TtsError::InvalidInput(
                "directml EP is not available in this ONNX Runtime build".into(),
            ));
        }
        match directml.register(builder) {
            Ok(()) => Ok(()),
            Err(err) => Err(ort_err(err)),
        }
    }

    #[cfg(not(feature = "directml"))]
    #[allow(dead_code)]
    fn register_directml(
        _builder: &mut ort::session::builder::SessionBuilder,
        _required: bool,
    ) -> Result<(), Qwen3TtsError> {
        Err(Qwen3TtsError::InvalidInput(
            "directml EP is unavailable because the binary was built without the directml feature"
                .into(),
        ))
    }
}
