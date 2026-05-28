//! Qwen3 TTS (GGUF + GGML) — native inference library.
//!
//! Output sample rate for the published Qwen3-TTS checkpoints.
pub const SAMPLE_RATE_HZ: u32 = 24_000;

mod custom_voice;
mod error;
mod model;
pub mod pipeline;
mod synthesis_profile;
mod voice_clone_prompt;

pub use custom_voice::{CustomVoiceMetadata, VoiceModelKind};
pub use error::Qwen3TtsError;
pub use model::{load_and_validate, GgufFile, ModelPaths};
pub use pipeline::audio_code_encoder::{AudioCodeEncoder, AudioCodeEncoderConfig};
pub use pipeline::backend::BackendKind;
pub use pipeline::speaker_encoder::{SpeakerEncoder, SpeakerEncoderConfig};
pub use pipeline::tokenizer::{TextTokenizer, TokenizerConfig};
pub use pipeline::tts_transformer::{
    CodePredDebugStep, CodePredTopLogit, CodecRollout, CodecRolloutSubTimings,
    IclPrefillConditioning, PrefillConditioning, PrefillForwardOutputs, PreparedPrefillInputs,
    SelectedCodecFrame, TtsTransformer, TtsTransformerConfig, VocoderChunk,
};
pub use pipeline::vocoder::{
    Vocoder, VocoderConfig, VocoderExecutionProvider, VocoderGraphTemplate,
};
pub use synthesis_profile::SynthesisStageTimings;
pub use voice_clone_prompt::{
    TensorF32, TensorI32, VoiceClonePromptV2, VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
};

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub(crate) fn trace_stage(label: &str) {
    if std::env::var_os("QWEN3_TTS_TRACE").is_some() {
        eprintln!("[qts-trace] {label}");
    }
}

pub(crate) fn ensure_ort_init() -> Result<(), Qwen3TtsError> {
    static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();
    ORT_INIT
        .get_or_init(|| {
            ensure_ort_dylib_path().map_err(|err| err.to_string())?;
            let _ = ort::init().commit();
            Ok(())
        })
        .as_ref()
        .map(|_| ())
        .map_err(|err| Qwen3TtsError::Ort(err.clone()))
}

fn ensure_ort_dylib_path() -> Result<(), Qwen3TtsError> {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return Ok(());
    }

    #[cfg(windows)]
    {
        let mut candidates = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("onnxruntime.dll"));
            }
        }
        if let Ok(dir) = std::env::current_dir() {
            candidates.push(dir.join("onnxruntime.dll"));
            candidates.push(dir.join("target").join("release").join("onnxruntime.dll"));
        }

        if let Some(path) = candidates.into_iter().find(|path| path.is_file()) {
            unsafe { std::env::set_var("ORT_DYLIB_PATH", path) };
            return Ok(());
        }

        return Err(Qwen3TtsError::Ort(
            "ONNX Runtime DLL not found. Put onnxruntime.dll next to qts_cli.exe or set ORT_DYLIB_PATH to its full path.".into(),
        ));
    }

    #[cfg(not(windows))]
    {
        Ok(())
    }
}

/// User-facing synthesis parameters (stable for future `gdext` bindings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TalkerKvMode {
    #[default]
    F16,
    TurboQuant,
}

impl TalkerKvMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::TurboQuant => "turboquant",
        }
    }

    pub fn parse(value: &str) -> Result<Self, Qwen3TtsError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "turboquant" | "turbo" | "q8_0" | "q8" => Ok(Self::TurboQuant),
            other => Err(Qwen3TtsError::InvalidInput(format!(
                "unknown talker KV mode '{other}' (expected f16 or turboquant)"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynthesizeRequest {
    pub text: String,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub max_audio_frames: usize,
    pub thread_count: usize,
    pub repetition_penalty: f32,
    /// Codec language id (e.g. 2050=en, 2055=zh, 2058=ja).
    pub language_id: i32,
    /// Number of CPU threads for vocoder decode when pipelining is enabled.
    /// Defaults to 4. Set to 0 to derive a backend-agnostic fallback from `thread_count`.
    pub vocoder_thread_count: usize,
    /// When > 0, pipeline transformer (GPU) and vocoder (CPU) by processing
    /// vocoder chunks of this many frames in a background thread while the
    /// transformer continues generating. Set to 0 to disable (sequential).
    pub vocoder_chunk_size: usize,
    /// Experimental talker KV cache storage mode.
    pub talker_kv_mode: TalkerKvMode,
}

impl Default for SynthesizeRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            max_audio_frames: 4096,
            thread_count: 4,
            repetition_penalty: 1.05,
            language_id: 2050,
            vocoder_thread_count: 4,
            vocoder_chunk_size: 0,
            talker_kv_mode: TalkerKvMode::F16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynthesizeResult {
    pub pcm_f32: Vec<f32>,
    pub sample_rate_hz: u32,
    /// Best-effort: frame count not yet exposed by C API; 0 for now.
    pub generated_frames: usize,
}

#[derive(Debug, Clone)]
pub struct SynthesizeDebugResult {
    pub synthesis: SynthesizeResult,
    pub codec_frames: Vec<Vec<i32>>,
    pub talker_hidden_states: Vec<Vec<f32>>,
    pub prefix_frame_count: usize,
    pub debug_step_embeddings: Vec<Vec<f32>>,
    pub debug_trailing_rows: Vec<Vec<f32>>,
    pub debug_code_pred_steps: Vec<Vec<CodePredDebugStep>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingSynthesizeResult {
    pub sample_rate_hz: u32,
    pub generated_frames: usize,
    pub generated_samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthesisProgressStage {
    Preparing,
    Prefill,
    Rollout,
    Vocoder,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynthesisProgress {
    pub stage: SynthesisProgressStage,
    pub generated_frames: usize,
    pub max_frames: usize,
}

impl SynthesisProgress {
    #[must_use]
    pub fn new(stage: SynthesisProgressStage, generated_frames: usize, max_frames: usize) -> Self {
        Self {
            stage,
            generated_frames,
            max_frames,
        }
    }

    #[must_use]
    pub fn rollout(generated_frames: usize, max_frames: usize) -> Self {
        Self::new(
            SynthesisProgressStage::Rollout,
            generated_frames,
            max_frames,
        )
    }
}

struct PreparedSynthesis<'a> {
    prepared_inputs: PreparedPrefillInputs,
    prompt_frames: Vec<Vec<i32>>,
    prefix_frame_count: usize,
    speaker_encode: std::time::Duration,
    tokenize: std::time::Duration,
    prefill_build: std::time::Duration,
    _speaker_embedding: Option<SpeakerEmbeddingStorage<'a>>,
}

enum SpeakerEmbeddingStorage<'a> {
    Borrowed(&'a [f32]),
    Owned(Vec<f32>),
}

impl<'a> SpeakerEmbeddingStorage<'a> {
    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Borrowed(values) => values,
            Self::Owned(values) => values,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CustomVoiceConditioning<'a> {
    speaker: &'a str,
    instruct: Option<&'a str>,
}

pub struct Qwen3TtsEngine {
    paths: ModelPaths,
    voice_model_kind: VoiceModelKind,
    custom_voice: Option<CustomVoiceMetadata>,
    tokenizer: TextTokenizer,
    transformer: TtsTransformer,
    vocoder: Vocoder,
    speaker_encoder: SpeakerEncoder,
    audio_code_encoder: Mutex<Option<AudioCodeEncoder>>,
}

impl Qwen3TtsEngine {
    pub fn load(paths: ModelPaths) -> Result<Self, Qwen3TtsError> {
        trace_stage("load: validate paths");
        load_and_validate(&paths)?;
        trace_stage("load: model kind");
        let voice_model_kind = VoiceModelKind::load(&paths)?;
        trace_stage("load: custom voice metadata");
        let custom_voice = CustomVoiceMetadata::load(&paths)?;
        trace_stage("load: vocoder");
        let vocoder = Vocoder::load_from_onnx(&paths.vocoder_onnx)?;
        trace_stage("load: open main gguf");
        let main = GgufFile::open(&paths.main_gguf)?;
        trace_stage("load: tokenizer");
        let tokenizer = TextTokenizer::load_from_gguf(&main)?;
        trace_stage("load: transformer");
        let transformer = TtsTransformer::load_from_gguf(&main)?;
        trace_stage("load: speaker encoder");
        let speaker_encoder = SpeakerEncoder::new(transformer.config().hidden_size as usize)?;
        trace_stage("load: done");
        Ok(Self {
            paths,
            voice_model_kind,
            custom_voice,
            tokenizer,
            transformer,
            vocoder,
            speaker_encoder,
            audio_code_encoder: Mutex::new(None),
        })
    }

    pub fn from_model_dir(dir: impl AsRef<std::path::Path>) -> Result<Self, Qwen3TtsError> {
        Self::load(ModelPaths::from_model_dir(dir))
    }

    pub fn model_paths(&self) -> &ModelPaths {
        &self.paths
    }

    #[must_use]
    pub fn voice_model_kind(&self) -> VoiceModelKind {
        self.voice_model_kind
    }

    #[must_use]
    pub fn custom_voice_metadata(&self) -> Option<&CustomVoiceMetadata> {
        self.custom_voice.as_ref()
    }

    #[must_use]
    pub fn tokenizer(&self) -> &TextTokenizer {
        &self.tokenizer
    }

    #[must_use]
    pub fn transformer(&self) -> &TtsTransformer {
        &self.transformer
    }

    #[must_use]
    pub fn vocoder(&self) -> &Vocoder {
        &self.vocoder
    }

    #[must_use]
    pub fn vocoder_execution_provider(&self) -> VocoderExecutionProvider {
        self.vocoder.execution_provider()
    }

    #[must_use]
    pub fn vocoder_backend_label(&self) -> &'static str {
        self.vocoder.execution_provider_label()
    }

    #[must_use]
    pub fn speaker_encoder(&self) -> &SpeakerEncoder {
        &self.speaker_encoder
    }

    #[must_use]
    pub fn primary_backend_kind(&self) -> BackendKind {
        self.transformer.primary_backend_kind()
    }

    #[must_use]
    pub fn encode_for_tts(&self, text: &str) -> Vec<i32> {
        self.tokenizer.encode_for_tts(text)
    }

    pub fn encode_reference_speaker(&self, wav_bytes: &[u8]) -> Result<Vec<f32>, Qwen3TtsError> {
        self.speaker_encoder.encode_wav_bytes(wav_bytes)
    }

    pub fn encode_reference_audio_codes(
        &self,
        wav_bytes: &[u8],
    ) -> Result<TensorI32, Qwen3TtsError> {
        let mut encoder = self
            .audio_code_encoder
            .lock()
            .map_err(|_| Qwen3TtsError::InvalidInput("audio code encoder mutex poisoned".into()))?;
        if encoder.is_none() {
            trace_stage("audio-code-encoder: load");
            *encoder = Some(AudioCodeEncoder::load_from_onnx(
                &self.paths.tokenizer_encoder_onnx,
            )?);
            trace_stage("audio-code-encoder: loaded");
        }
        let encoder = encoder.as_ref().expect("audio code encoder loaded");
        trace_stage("audio-code-encoder: encode wav");
        encoder.encode_wav_bytes(wav_bytes)
    }

    #[must_use]
    pub fn speaker_embedding_size(&self) -> usize {
        self.transformer.config().hidden_size as usize
    }

    pub fn decode_voice_clone_prompt(
        &self,
        pb_bytes: &[u8],
    ) -> Result<VoiceClonePromptV2, Qwen3TtsError> {
        let prompt = VoiceClonePromptV2::from_protobuf_bytes(pb_bytes)?;
        self.validate_speaker_embedding(prompt.speaker_embedding())?;
        if let Some((_, codebooks)) = prompt.ref_code_shape() {
            if codebooks != self.transformer.config().n_codebooks as usize {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "voice clone prompt ref_code must have {} codebooks per frame",
                    self.transformer.config().n_codebooks
                )));
            }
        }
        Ok(prompt)
    }

    pub fn synthesize_with_voice_clone_prompt(
        &self,
        req: &SynthesizeRequest,
        prompt: &VoiceClonePromptV2,
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.validate_speaker_embedding(prompt.speaker_embedding())?;
        self.synthesize_impl(req, None, Some(prompt), None, None, None)
    }

    pub fn synthesize(&self, req: &SynthesizeRequest) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.synthesize_impl(req, None, None, None, None, None)
    }

    pub fn synthesize_with_custom_voice(
        &self,
        req: &SynthesizeRequest,
        speaker: &str,
        instruct: Option<&str>,
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.synthesize_impl(
            req,
            None,
            None,
            Some(CustomVoiceConditioning { speaker, instruct }),
            None,
            None,
        )
    }

    pub fn synthesize_with_voice_design(
        &self,
        req: &SynthesizeRequest,
        instruct: &str,
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.synthesize_impl(req, None, None, None, Some(instruct), None)
    }

    pub fn synthesize_with_voice_design_speaker_embedding(
        &self,
        req: &SynthesizeRequest,
        instruct: &str,
        speaker_embedding: &[f32],
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.validate_speaker_embedding(speaker_embedding)?;
        self.synthesize_impl(
            req,
            Some(speaker_embedding),
            None,
            None,
            Some(instruct),
            None,
        )
    }

    pub fn synthesize_debug(
        &self,
        req: &SynthesizeRequest,
    ) -> Result<SynthesizeDebugResult, Qwen3TtsError> {
        self.synthesize_debug_impl(req, None, None, None, None)
    }

    pub fn synthesize_with_voice_clone_prompt_debug(
        &self,
        req: &SynthesizeRequest,
        prompt: &VoiceClonePromptV2,
    ) -> Result<SynthesizeDebugResult, Qwen3TtsError> {
        self.validate_speaker_embedding(prompt.speaker_embedding())?;
        self.synthesize_debug_impl(req, None, Some(prompt), None, None)
    }

    pub fn synthesize_with_custom_voice_debug(
        &self,
        req: &SynthesizeRequest,
        speaker: &str,
        instruct: Option<&str>,
    ) -> Result<SynthesizeDebugResult, Qwen3TtsError> {
        self.synthesize_debug_impl(
            req,
            None,
            None,
            Some(CustomVoiceConditioning { speaker, instruct }),
            None,
        )
    }

    pub fn synthesize_with_voice_design_debug(
        &self,
        req: &SynthesizeRequest,
        instruct: &str,
    ) -> Result<SynthesizeDebugResult, Qwen3TtsError> {
        self.synthesize_debug_impl(req, None, None, None, Some(instruct))
    }

    pub fn synthesize_streaming<S>(
        &self,
        req: &SynthesizeRequest,
        sink: &mut S,
    ) -> Result<StreamingSynthesizeResult, Qwen3TtsError>
    where
        S: StreamingSynthesis + Send,
    {
        self.synthesize_streaming_impl(req, None, None, None, None, sink, None)
    }

    pub fn synthesize_with_voice_clone_prompt_streaming<S>(
        &self,
        req: &SynthesizeRequest,
        prompt: &VoiceClonePromptV2,
        sink: &mut S,
    ) -> Result<StreamingSynthesizeResult, Qwen3TtsError>
    where
        S: StreamingSynthesis + Send,
    {
        self.validate_speaker_embedding(prompt.speaker_embedding())?;
        self.synthesize_streaming_impl(req, None, Some(prompt), None, None, sink, None)
    }

    /// Same as [`Self::synthesize`], plus wall-clock timings per pipeline stage.
    pub fn synthesize_with_profile(
        &self,
        req: &SynthesizeRequest,
    ) -> Result<(SynthesizeResult, SynthesisStageTimings), Qwen3TtsError> {
        let mut timings = SynthesisStageTimings::default();
        let result = self.synthesize_impl(req, None, None, None, None, Some(&mut timings))?;
        Ok((result, timings))
    }

    /// Same as [`Self::synthesize_with_voice_clone_prompt`], plus stage timings.
    pub fn synthesize_with_voice_clone_prompt_profile(
        &self,
        req: &SynthesizeRequest,
        prompt: &VoiceClonePromptV2,
    ) -> Result<(SynthesizeResult, SynthesisStageTimings), Qwen3TtsError> {
        self.validate_speaker_embedding(prompt.speaker_embedding())?;
        let mut timings = SynthesisStageTimings::default();
        let result =
            self.synthesize_impl(req, None, Some(prompt), None, None, Some(&mut timings))?;
        Ok((result, timings))
    }

    pub fn synthesize_with_custom_voice_profile(
        &self,
        req: &SynthesizeRequest,
        speaker: &str,
        instruct: Option<&str>,
    ) -> Result<(SynthesizeResult, SynthesisStageTimings), Qwen3TtsError> {
        let mut timings = SynthesisStageTimings::default();
        let result = self.synthesize_impl(
            req,
            None,
            None,
            Some(CustomVoiceConditioning { speaker, instruct }),
            None,
            Some(&mut timings),
        )?;
        Ok((result, timings))
    }

    pub fn synthesize_with_voice_design_profile(
        &self,
        req: &SynthesizeRequest,
        instruct: &str,
    ) -> Result<(SynthesizeResult, SynthesisStageTimings), Qwen3TtsError> {
        let mut timings = SynthesisStageTimings::default();
        let result =
            self.synthesize_impl(req, None, None, None, Some(instruct), Some(&mut timings))?;
        Ok((result, timings))
    }

    pub fn synthesize_with_progress(
        &self,
        req: &SynthesizeRequest,
        mut progress: impl FnMut(SynthesisProgress),
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.synthesize_impl_with_progress(req, None, None, None, None, None, &mut progress)
    }

    pub fn synthesize_with_custom_voice_progress(
        &self,
        req: &SynthesizeRequest,
        speaker: &str,
        instruct: Option<&str>,
        mut progress: impl FnMut(SynthesisProgress),
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.synthesize_impl_with_progress(
            req,
            None,
            None,
            Some(CustomVoiceConditioning { speaker, instruct }),
            None,
            None,
            &mut progress,
        )
    }

    pub fn synthesize_with_voice_design_progress(
        &self,
        req: &SynthesizeRequest,
        instruct: &str,
        mut progress: impl FnMut(SynthesisProgress),
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.synthesize_impl_with_progress(
            req,
            None,
            None,
            None,
            Some(instruct),
            None,
            &mut progress,
        )
    }

    pub fn synthesize_with_voice_design_speaker_embedding_progress(
        &self,
        req: &SynthesizeRequest,
        instruct: &str,
        speaker_embedding: &[f32],
        mut progress: impl FnMut(SynthesisProgress),
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.validate_speaker_embedding(speaker_embedding)?;
        self.synthesize_impl_with_progress(
            req,
            Some(speaker_embedding),
            None,
            None,
            Some(instruct),
            None,
            &mut progress,
        )
    }

    pub fn synthesize_with_voice_clone_prompt_progress(
        &self,
        req: &SynthesizeRequest,
        prompt: &VoiceClonePromptV2,
        mut progress: impl FnMut(SynthesisProgress),
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.validate_speaker_embedding(prompt.speaker_embedding())?;
        self.synthesize_impl_with_progress(req, None, Some(prompt), None, None, None, &mut progress)
    }

    fn synthesize_impl(
        &self,
        req: &SynthesizeRequest,
        speaker_embedding_override: Option<&[f32]>,
        voice_clone_prompt: Option<&VoiceClonePromptV2>,
        custom_voice: Option<CustomVoiceConditioning<'_>>,
        voice_design_instruct: Option<&str>,
        timings: Option<&mut SynthesisStageTimings>,
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.synthesize_impl_with_progress(
            req,
            speaker_embedding_override,
            voice_clone_prompt,
            custom_voice,
            voice_design_instruct,
            timings,
            &mut |_| {},
        )
    }

    fn synthesize_impl_with_progress(
        &self,
        req: &SynthesizeRequest,
        speaker_embedding_override: Option<&[f32]>,
        voice_clone_prompt: Option<&VoiceClonePromptV2>,
        custom_voice: Option<CustomVoiceConditioning<'_>>,
        voice_design_instruct: Option<&str>,
        timings: Option<&mut SynthesisStageTimings>,
        progress: &mut dyn FnMut(SynthesisProgress),
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        progress(SynthesisProgress::new(
            SynthesisProgressStage::Preparing,
            0,
            req.max_audio_frames,
        ));
        let prepared = self.prepare_synthesis(
            req,
            speaker_embedding_override,
            voice_clone_prompt,
            custom_voice,
            voice_design_instruct,
        )?;

        progress(SynthesisProgress::new(
            SynthesisProgressStage::Prefill,
            0,
            req.max_audio_frames,
        ));
        if req.vocoder_chunk_size > 0 {
            let result = self.synthesize_pipelined(req, &prepared, timings)?;
            progress(SynthesisProgress::new(
                SynthesisProgressStage::Done,
                result.generated_frames,
                req.max_audio_frames,
            ));
            Ok(result)
        } else {
            self.synthesize_sequential_with_progress(req, &prepared, timings, progress)
        }
    }

    fn synthesize_streaming_impl<S>(
        &self,
        req: &SynthesizeRequest,
        speaker_embedding_override: Option<&[f32]>,
        voice_clone_prompt: Option<&VoiceClonePromptV2>,
        custom_voice: Option<CustomVoiceConditioning<'_>>,
        voice_design_instruct: Option<&str>,
        sink: &mut S,
        timings: Option<&mut SynthesisStageTimings>,
    ) -> Result<StreamingSynthesizeResult, Qwen3TtsError>
    where
        S: StreamingSynthesis + Send,
    {
        let prepared = self.prepare_synthesis(
            req,
            speaker_embedding_override,
            voice_clone_prompt,
            custom_voice,
            voice_design_instruct,
        )?;

        if req.vocoder_chunk_size > 0 {
            self.synthesize_pipelined_streaming(req, &prepared, sink, timings)
        } else {
            self.synthesize_sequential_streaming(req, &prepared, sink, timings)
        }
    }

    fn synthesize_debug_impl(
        &self,
        req: &SynthesizeRequest,
        speaker_embedding_override: Option<&[f32]>,
        voice_clone_prompt: Option<&VoiceClonePromptV2>,
        custom_voice: Option<CustomVoiceConditioning<'_>>,
        voice_design_instruct: Option<&str>,
    ) -> Result<SynthesizeDebugResult, Qwen3TtsError> {
        trace_stage("synthesize-debug: prepare");
        let prepared = self.prepare_synthesis(
            req,
            speaker_embedding_override,
            voice_clone_prompt,
            custom_voice,
            voice_design_instruct,
        )?;
        trace_stage("synthesize-debug: rollout");
        let codec_rollout = self.transformer.rollout_codec_frames_kv(
            &prepared.prepared_inputs.prefill_embd,
            &prepared.prepared_inputs.trailing_text_hidden,
            &prepared.prepared_inputs.tts_pad_embed,
            &prepared.prompt_frames,
            req.talker_kv_mode,
            req.thread_count,
            req.max_audio_frames,
            req.repetition_penalty,
            req.temperature,
            req.top_k,
            req.top_p,
        )?;

        trace_stage("synthesize-debug: flatten");
        let generated_frames = codec_rollout
            .frames
            .len()
            .saturating_sub(prepared.prefix_frame_count);
        let flattened_codes = codec_rollout
            .frames
            .iter()
            .flat_map(|frame| frame.codebook_tokens.iter().copied())
            .collect::<Vec<_>>();
        trace_stage("synthesize-debug: vocoder decode");
        let pcm_all = self.vocoder.decode(
            &flattened_codes,
            codec_rollout.frames.len(),
            req.thread_count,
        )?;
        trace_stage("synthesize-debug: done");
        let pcm_f32 = if prepared.prefix_frame_count == 0 || codec_rollout.frames.is_empty() {
            pcm_all
        } else {
            let cut = prepared
                .prefix_frame_count
                .saturating_mul(pcm_all.len())
                .checked_div(codec_rollout.frames.len())
                .unwrap_or(0)
                .min(pcm_all.len());
            pcm_all[cut..].to_vec()
        };

        Ok(SynthesizeDebugResult {
            synthesis: SynthesizeResult {
                pcm_f32,
                sample_rate_hz: self.vocoder.config().sample_rate as u32,
                generated_frames,
            },
            codec_frames: codec_rollout
                .frames
                .iter()
                .skip(prepared.prefix_frame_count)
                .map(|frame| frame.codebook_tokens.clone())
                .collect(),
            talker_hidden_states: codec_rollout
                .frames
                .iter()
                .skip(prepared.prefix_frame_count)
                .map(|frame| frame.hidden_state.clone())
                .collect(),
            prefix_frame_count: prepared.prefix_frame_count,
            debug_step_embeddings: codec_rollout.debug_step_embeddings,
            debug_trailing_rows: codec_rollout.debug_trailing_rows,
            debug_code_pred_steps: match codec_rollout
                .frames
                .iter()
                .skip(prepared.prefix_frame_count)
                .take(2)
                .map(|frame| {
                    self.transformer.debug_code_predictor_recompute(
                        &frame.hidden_state,
                        frame.codebook_0_token,
                        req.thread_count,
                        8,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
            {
                Ok(steps) => steps,
                Err(err) => {
                    eprintln!(
                        "warning: failed to recompute debug code predictor steps; continuing without codepred_first2: {err}"
                    );
                    Vec::new()
                }
            },
        })
    }

    fn prepare_synthesis<'a>(
        &self,
        req: &SynthesizeRequest,
        speaker_embedding_override: Option<&'a [f32]>,
        voice_clone_prompt: Option<&'a VoiceClonePromptV2>,
        custom_voice: Option<CustomVoiceConditioning<'a>>,
        voice_design_instruct: Option<&'a str>,
    ) -> Result<PreparedSynthesis<'a>, Qwen3TtsError> {
        let conditioning_count = usize::from(voice_clone_prompt.is_some())
            + usize::from(custom_voice.is_some())
            + usize::from(voice_design_instruct.is_some());
        if conditioning_count > 1 {
            return Err(Qwen3TtsError::InvalidInput(
                "voice clone, custom voice, and voice design modes cannot be combined".into(),
            ));
        }

        let speaker_encode = std::time::Duration::ZERO;
        let t_tok = Instant::now();
        let prompt_frames = if let Some(prompt) = voice_clone_prompt {
            prompt
                .ref_code_shape()
                .map_or_else(Vec::new, |(frames, codebooks)| {
                    let values = prompt.ref_code_values().unwrap_or(&[]);
                    (0..frames)
                        .map(|frame_idx| {
                            let start = frame_idx * codebooks;
                            let end = start + codebooks;
                            values[start..end].to_vec()
                        })
                        .collect::<Vec<_>>()
                })
        } else {
            Vec::new()
        };
        let prefix_frame_count = prompt_frames.len();
        let tokenize = t_tok.elapsed();

        let t_prefill = Instant::now();
        let (prepared_inputs, speaker_embedding) = if let Some(custom_voice) = custom_voice {
            let metadata = self.custom_voice.as_ref().ok_or_else(|| {
                Qwen3TtsError::InvalidInput(
                    "loaded model does not include custom voice speaker metadata".into(),
                )
            })?;
            let speaker_token_id = metadata.speaker_token_id(custom_voice.speaker)?;
            let effective_language_id =
                metadata.resolve_language_id(req.language_id, custom_voice.speaker);
            let speaker_embedding = Some(SpeakerEmbeddingStorage::Owned(
                self.transformer
                    .lookup_codec_embedding_row(speaker_token_id, req.thread_count)?,
            ));
            let text_tokens = self.tokenizer.encode_for_tts(&req.text);
            let instruct_tokens = custom_voice
                .instruct
                .filter(|text| !text.trim().is_empty())
                .map_or_else(Vec::new, |text| {
                    self.tokenizer.encode_instruct_for_tts(text)
                });
            (
                self.transformer.build_prefill_inputs(
                    PrefillConditioning {
                        text_tokens: &text_tokens,
                        instruct_tokens: &instruct_tokens,
                        speaker_embd: speaker_embedding
                            .as_ref()
                            .map(SpeakerEmbeddingStorage::as_slice),
                        ref_codebook_0: &[],
                        language_id: effective_language_id,
                    },
                    req.thread_count,
                )?,
                speaker_embedding,
            )
        } else if let Some(instruct) = voice_design_instruct {
            if !self.voice_model_kind.is_voice_design() {
                return Err(Qwen3TtsError::InvalidInput(
                    "voice design requires a voice_design model".into(),
                ));
            }
            let instruct = instruct.trim();
            if instruct.is_empty() {
                return Err(Qwen3TtsError::InvalidInput(
                    "voice design requires a non-empty instruct description".into(),
                ));
            }
            let text_tokens = self.tokenizer.encode_for_tts(&req.text);
            let instruct_tokens = self.tokenizer.encode_instruct_for_tts(instruct);
            let speaker_embedding =
                speaker_embedding_override.map(SpeakerEmbeddingStorage::Borrowed);
            (
                self.transformer.build_prefill_inputs(
                    PrefillConditioning {
                        text_tokens: &text_tokens,
                        instruct_tokens: &instruct_tokens,
                        speaker_embd: speaker_embedding
                            .as_ref()
                            .map(SpeakerEmbeddingStorage::as_slice),
                        ref_codebook_0: &[],
                        language_id: req.language_id,
                    },
                    req.thread_count,
                )?,
                speaker_embedding,
            )
        } else if let Some(prompt) = voice_clone_prompt {
            let speaker_embedding = if let Some(speaker_embedding) = speaker_embedding_override {
                Some(SpeakerEmbeddingStorage::Borrowed(speaker_embedding))
            } else {
                Some(SpeakerEmbeddingStorage::Borrowed(
                    prompt.speaker_embedding(),
                ))
            };
            if prompt.icl_mode {
                let text_tokens = self.tokenizer.encode_for_tts(&req.text);
                let ref_text_tokens = self.tokenizer.encode_ref_for_tts(&prompt.ref_text);
                (
                    self.transformer.build_icl_prefill_inputs(
                        IclPrefillConditioning {
                            text_tokens: &text_tokens,
                            ref_text_tokens: &ref_text_tokens,
                            speaker_embd: speaker_embedding
                                .as_ref()
                                .map(SpeakerEmbeddingStorage::as_slice),
                            ref_code_frames: &prompt_frames,
                            language_id: req.language_id,
                        },
                        req.thread_count,
                    )?,
                    speaker_embedding,
                )
            } else {
                let ref_codebook_0 = prompt_frames
                    .iter()
                    .filter_map(|frame| frame.first().copied())
                    .collect::<Vec<_>>();
                let text_tokens = self.tokenizer.encode_for_tts(&req.text);
                (
                    self.transformer.build_prefill_inputs(
                        PrefillConditioning {
                            text_tokens: &text_tokens,
                            instruct_tokens: &[],
                            speaker_embd: speaker_embedding
                                .as_ref()
                                .map(SpeakerEmbeddingStorage::as_slice),
                            ref_codebook_0: &ref_codebook_0,
                            language_id: req.language_id,
                        },
                        req.thread_count,
                    )?,
                    speaker_embedding,
                )
            }
        } else {
            let speaker_embedding = if let Some(speaker_embedding) = speaker_embedding_override {
                Some(SpeakerEmbeddingStorage::Borrowed(speaker_embedding))
            } else {
                Some(SpeakerEmbeddingStorage::Owned(vec![
                    0.0f32;
                    self.speaker_embedding_size()
                ]))
            };
            let text_tokens = self.tokenizer.encode_for_tts(&req.text);
            (
                self.transformer.build_prefill_inputs(
                    PrefillConditioning {
                        text_tokens: &text_tokens,
                        instruct_tokens: &[],
                        speaker_embd: speaker_embedding
                            .as_ref()
                            .map(SpeakerEmbeddingStorage::as_slice),
                        ref_codebook_0: &[],
                        language_id: req.language_id,
                    },
                    req.thread_count,
                )?,
                speaker_embedding,
            )
        };
        let prefill_build = t_prefill.elapsed();

        Ok(PreparedSynthesis {
            prepared_inputs,
            prompt_frames,
            prefix_frame_count,
            speaker_encode,
            tokenize,
            prefill_build,
            _speaker_embedding: speaker_embedding,
        })
    }

    fn synthesize_sequential(
        &self,
        req: &SynthesizeRequest,
        prepared: &PreparedSynthesis<'_>,
        timings: Option<&mut SynthesisStageTimings>,
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        self.synthesize_sequential_with_progress(req, prepared, timings, &mut |_| {})
    }

    fn synthesize_sequential_with_progress(
        &self,
        req: &SynthesizeRequest,
        prepared: &PreparedSynthesis<'_>,
        timings: Option<&mut SynthesisStageTimings>,
        progress: &mut dyn FnMut(SynthesisProgress),
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        let t_roll = Instant::now();
        let codec_rollout = self.transformer.rollout_codec_frames_kv_with_progress(
            &prepared.prepared_inputs.prefill_embd,
            &prepared.prepared_inputs.trailing_text_hidden,
            &prepared.prepared_inputs.tts_pad_embed,
            &prepared.prompt_frames,
            req.talker_kv_mode,
            req.thread_count,
            req.max_audio_frames,
            req.repetition_penalty,
            req.temperature,
            req.top_k,
            req.top_p,
            progress,
        )?;
        let codec_rollout_dur = t_roll.elapsed();

        let generated_frames = codec_rollout
            .frames
            .len()
            .saturating_sub(prepared.prefix_frame_count);

        let t_post = Instant::now();
        let flattened_codes = codec_rollout
            .frames
            .iter()
            .flat_map(|frame| frame.codebook_tokens.iter().copied())
            .collect::<Vec<_>>();
        let flatten_dur = t_post.elapsed();

        progress(SynthesisProgress::new(
            SynthesisProgressStage::Vocoder,
            generated_frames,
            req.max_audio_frames,
        ));
        let t_voc = Instant::now();
        let pcm_all = self.vocoder.decode(
            &flattened_codes,
            codec_rollout.frames.len(),
            req.thread_count,
        )?;
        let vocoder_decode = t_voc.elapsed();

        let t_trim = Instant::now();
        let pcm_f32 = if prepared.prefix_frame_count == 0 || codec_rollout.frames.is_empty() {
            pcm_all
        } else {
            let cut = prepared
                .prefix_frame_count
                .saturating_mul(pcm_all.len())
                .checked_div(codec_rollout.frames.len())
                .unwrap_or(0)
                .min(pcm_all.len());
            pcm_all[cut..].to_vec()
        };
        let post = t_trim.elapsed() + flatten_dur;

        let sample_rate_hz = self.vocoder.config().sample_rate as u32;
        progress(SynthesisProgress::new(
            SynthesisProgressStage::Done,
            generated_frames,
            req.max_audio_frames,
        ));
        if let Some(t) = timings {
            t.speaker_encode = prepared.speaker_encode;
            t.tokenize = prepared.tokenize;
            t.prefill_build = prepared.prefill_build;
            t.codec_rollout = codec_rollout_dur;
            t.vocoder_decode = vocoder_decode;
            t.post = post;
            t.codec_rollout_detail = codec_rollout.sub_timings;
            t.first_frame_latency = prepared.speaker_encode
                + prepared.tokenize
                + prepared.prefill_build
                + codec_rollout.first_frame_elapsed;
            t.generated_samples = pcm_f32.len();
            t.sample_rate_hz = sample_rate_hz;
        }

        Ok(SynthesizeResult {
            pcm_f32,
            sample_rate_hz,
            generated_frames,
        })
    }

    fn synthesize_sequential_streaming<S>(
        &self,
        req: &SynthesizeRequest,
        prepared: &PreparedSynthesis<'_>,
        sink: &mut S,
        timings: Option<&mut SynthesisStageTimings>,
    ) -> Result<StreamingSynthesizeResult, Qwen3TtsError>
    where
        S: StreamingSynthesis + Send,
    {
        let result = self.synthesize_sequential(req, prepared, timings)?;
        sink.push_pcm_chunk(&result.pcm_f32)?;
        Ok(StreamingSynthesizeResult {
            sample_rate_hz: result.sample_rate_hz,
            generated_frames: result.generated_frames,
            generated_samples: result.pcm_f32.len(),
        })
    }

    /// Pipeline transformer (GPU/main thread) with vocoder (CPU/background thread).
    ///
    /// The transformer generates frames autoregressively; every `chunk_size` generated
    /// frames the codebook tokens are sent to a vocoder worker that decodes them in
    /// parallel on CPU. When the transformer finishes the last chunk is flushed and
    /// all audio is concatenated.
    ///
    /// To avoid audible clicks at chunk boundaries the vocoder thread uses
    /// **overlap-and-add**: each chunk (after the first) is decoded with a few
    /// extra context frames from the previous chunk prepended, then the
    /// overlapping audio region is linearly crossfaded with the previous
    /// chunk's tail.
    fn synthesize_pipelined(
        &self,
        req: &SynthesizeRequest,
        prepared: &PreparedSynthesis<'_>,
        timings: Option<&mut SynthesisStageTimings>,
    ) -> Result<SynthesizeResult, Qwen3TtsError> {
        use std::sync::mpsc;

        let chunk_size = req.vocoder_chunk_size;
        let thread_count = req.thread_count;
        let vocoder_thread_count = if req.vocoder_thread_count > 0 {
            req.vocoder_thread_count
        } else {
            (thread_count / 2).max(1)
        };

        let (chunk_tx, chunk_rx) = mpsc::sync_channel::<VocoderChunk>(2);

        let prompt_chunk = if prepared.prefix_frame_count > 0 {
            let codes = prepared
                .prompt_frames
                .iter()
                .flat_map(|f| f.iter().copied())
                .collect::<Vec<_>>();
            Some(VocoderChunk {
                codes,
                n_frames: prepared.prefix_frame_count,
            })
        } else {
            None
        };

        let t_pipeline_start = Instant::now();

        std::thread::scope(|s| {
            let vocoder = &self.vocoder;
            let n_codebooks = vocoder.config().n_codebooks as usize;

            let vocoder_handle = s.spawn(
                move || -> Result<(Vec<f32>, std::time::Duration), Qwen3TtsError> {
                    const OVERLAP_FRAMES: usize = 3;
                    let standard_frames = OVERLAP_FRAMES + chunk_size;

                    let t_voc_start = Instant::now();
                    let mut all_pcm = Vec::<f32>::new();
                    let mut prev_codes: Vec<i32> = Vec::new();
                    let mut prev_n_frames: usize = 0;
                    let mut template: Option<VocoderGraphTemplate> = None;

                    if let Some(prompt_chunk) = prompt_chunk {
                        let audio = vocoder.decode(
                            &prompt_chunk.codes,
                            prompt_chunk.n_frames,
                            vocoder_thread_count,
                        )?;
                        all_pcm.extend_from_slice(&audio);
                        prev_n_frames = prompt_chunk.n_frames;
                        prev_codes = prompt_chunk.codes;
                    }

                    while let Ok(chunk) = chunk_rx.recv() {
                        let ctx_frames = OVERLAP_FRAMES.min(prev_n_frames);

                        let mut combined_buf = Vec::new();
                        let (codes_slice, decode_frames) = if ctx_frames > 0 {
                            combined_buf.reserve(ctx_frames * n_codebooks + chunk.codes.len());
                            let ctx_start = prev_codes.len() - ctx_frames * n_codebooks;
                            combined_buf.extend_from_slice(&prev_codes[ctx_start..]);
                            combined_buf.extend_from_slice(&chunk.codes);
                            (combined_buf.as_slice(), ctx_frames + chunk.n_frames)
                        } else {
                            (chunk.codes.as_slice(), chunk.n_frames)
                        };

                        let audio = if decode_frames == standard_frames {
                            if template.is_none() {
                                template = Some(vocoder.build_decode_template(standard_frames)?);
                            }
                            vocoder.decode_with_template(
                                template.as_mut().unwrap(),
                                codes_slice,
                                vocoder_thread_count,
                            )?
                        } else {
                            vocoder.decode(codes_slice, decode_frames, vocoder_thread_count)?
                        };

                        if ctx_frames > 0 && !all_pcm.is_empty() {
                            let total_frames = ctx_frames + chunk.n_frames;
                            let overlap_samples =
                                (audio.len() * ctx_frames / total_frames).min(all_pcm.len());
                            let start = all_pcm.len() - overlap_samples;
                            for i in 0..overlap_samples {
                                let t = (i as f32 + 0.5) / overlap_samples as f32;
                                all_pcm[start + i] = all_pcm[start + i] * (1.0 - t) + audio[i] * t;
                            }
                            all_pcm.extend_from_slice(&audio[overlap_samples..]);
                        } else {
                            all_pcm.extend_from_slice(&audio);
                        }

                        prev_n_frames = chunk.n_frames;
                        prev_codes = chunk.codes;
                    }

                    Ok((all_pcm, t_voc_start.elapsed()))
                },
            );

            let t_roll = Instant::now();
            let codec_rollout = self.transformer.rollout_codec_frames_kv_streaming(
                &prepared.prepared_inputs.prefill_embd,
                &prepared.prepared_inputs.trailing_text_hidden,
                &prepared.prepared_inputs.tts_pad_embed,
                &prepared.prompt_frames,
                req.talker_kv_mode,
                thread_count,
                req.max_audio_frames,
                req.repetition_penalty,
                req.temperature,
                req.top_k,
                req.top_p,
                chunk_size,
                &chunk_tx,
            );
            let codec_rollout_dur = t_roll.elapsed();
            drop(chunk_tx);

            let codec_rollout = codec_rollout?;
            let generated_frames = codec_rollout
                .frames
                .len()
                .saturating_sub(prepared.prefix_frame_count);

            let (pcm_all, vocoder_decode) = vocoder_handle.join().unwrap()?;

            let pipeline_wall_clock = t_pipeline_start.elapsed();

            let t_trim = Instant::now();
            let pcm_f32 = if prepared.prefix_frame_count == 0 || pcm_all.is_empty() {
                pcm_all
            } else {
                let total_frames = prepared.prefix_frame_count + generated_frames;
                let cut = prepared
                    .prefix_frame_count
                    .saturating_mul(pcm_all.len())
                    .checked_div(total_frames)
                    .unwrap_or(0)
                    .min(pcm_all.len());
                pcm_all[cut..].to_vec()
            };
            let post = t_trim.elapsed();

            let sample_rate_hz = self.vocoder.config().sample_rate as u32;
            if let Some(t) = timings {
                t.speaker_encode = prepared.speaker_encode;
                t.tokenize = prepared.tokenize;
                t.prefill_build = prepared.prefill_build;
                t.codec_rollout = codec_rollout_dur;
                t.vocoder_decode = vocoder_decode;
                t.post = post;
                t.codec_rollout_detail = codec_rollout.sub_timings;
                let sequential_sum = codec_rollout_dur + vocoder_decode;
                t.pipeline_overlap = sequential_sum.saturating_sub(pipeline_wall_clock);
                t.first_frame_latency = prepared.speaker_encode
                    + prepared.tokenize
                    + prepared.prefill_build
                    + codec_rollout.first_frame_elapsed;
                t.generated_samples = pcm_f32.len();
                t.sample_rate_hz = sample_rate_hz;
            }

            Ok(SynthesizeResult {
                pcm_f32,
                sample_rate_hz,
                generated_frames,
            })
        })
    }

    fn synthesize_pipelined_streaming<S>(
        &self,
        req: &SynthesizeRequest,
        prepared: &PreparedSynthesis<'_>,
        sink: &mut S,
        timings: Option<&mut SynthesisStageTimings>,
    ) -> Result<StreamingSynthesizeResult, Qwen3TtsError>
    where
        S: StreamingSynthesis + Send,
    {
        use std::sync::mpsc;

        let chunk_size = req.vocoder_chunk_size;
        let thread_count = req.thread_count;
        let vocoder_thread_count = if req.vocoder_thread_count > 0 {
            req.vocoder_thread_count
        } else {
            (thread_count / 2).max(1)
        };

        let (chunk_tx, chunk_rx) = mpsc::sync_channel::<VocoderChunk>(2);

        let t_pipeline_start = Instant::now();

        std::thread::scope(|s| {
            let vocoder = &self.vocoder;
            let n_codebooks = vocoder.config().n_codebooks as usize;

            let vocoder_handle = s.spawn(
                move || -> Result<(usize, std::time::Duration), Qwen3TtsError> {
                    const OVERLAP_FRAMES: usize = 1;
                    let standard_frames = OVERLAP_FRAMES + chunk_size;
                    let t_voc_start = Instant::now();
                    let mut total_samples = 0usize;
                    let mut all_pcm = Vec::<f32>::new();
                    let mut emitted_samples = 0usize;
                    let mut prev_codes: Vec<i32> = Vec::new();
                    let mut prev_n_frames: usize = 0;
                    let mut template: Option<VocoderGraphTemplate> = None;

                    while let Ok(chunk) = chunk_rx.recv() {
                        let ctx_frames = OVERLAP_FRAMES.min(prev_n_frames);

                        let mut combined_buf = Vec::new();
                        let (codes_slice, decode_frames) = if ctx_frames > 0 {
                            combined_buf.reserve(ctx_frames * n_codebooks + chunk.codes.len());
                            let ctx_start = prev_codes.len() - ctx_frames * n_codebooks;
                            combined_buf.extend_from_slice(&prev_codes[ctx_start..]);
                            combined_buf.extend_from_slice(&chunk.codes);
                            (combined_buf.as_slice(), ctx_frames + chunk.n_frames)
                        } else {
                            (chunk.codes.as_slice(), chunk.n_frames)
                        };

                        let audio = if decode_frames == standard_frames {
                            if template.is_none() {
                                template = Some(vocoder.build_decode_template(standard_frames)?);
                            }
                            vocoder.decode_with_template(
                                template.as_mut().unwrap(),
                                codes_slice,
                                vocoder_thread_count,
                            )?
                        } else {
                            vocoder.decode(codes_slice, decode_frames, vocoder_thread_count)?
                        };

                        let current_chunk_samples = if ctx_frames > 0 && !all_pcm.is_empty() {
                            let total_frames = ctx_frames + chunk.n_frames;
                            let overlap_samples =
                                (audio.len() * ctx_frames / total_frames).min(all_pcm.len());
                            let start = all_pcm.len() - overlap_samples;
                            for i in 0..overlap_samples {
                                let t = (i as f32 + 0.5) / overlap_samples as f32;
                                all_pcm[start + i] = all_pcm[start + i] * (1.0 - t) + audio[i] * t;
                            }
                            let appended = audio.len().saturating_sub(overlap_samples);
                            all_pcm.extend_from_slice(&audio[overlap_samples..]);
                            appended
                        } else {
                            let appended = audio.len();
                            all_pcm.extend_from_slice(&audio);
                            appended
                        };

                        let hold_back_frames = OVERLAP_FRAMES.min(chunk.n_frames);
                        let hold_back_samples = if chunk.n_frames == 0 {
                            0
                        } else {
                            current_chunk_samples
                                .saturating_mul(hold_back_frames)
                                .checked_div(chunk.n_frames)
                                .unwrap_or(0)
                                .min(all_pcm.len().saturating_sub(emitted_samples))
                        };
                        let finalized_end = all_pcm.len().saturating_sub(hold_back_samples);

                        if finalized_end > emitted_samples {
                            let finalized = &all_pcm[emitted_samples..finalized_end];
                            total_samples += finalized.len();
                            sink.push_pcm_chunk(finalized)?;
                            emitted_samples = finalized_end;
                        }

                        prev_n_frames = chunk.n_frames;
                        prev_codes = chunk.codes;
                    }

                    if all_pcm.len() > emitted_samples {
                        let finalized = &all_pcm[emitted_samples..];
                        total_samples += finalized.len();
                        sink.push_pcm_chunk(finalized)?;
                    }

                    Ok((total_samples, t_voc_start.elapsed()))
                },
            );

            let t_roll = Instant::now();
            let codec_rollout = self.transformer.rollout_codec_frames_kv_streaming(
                &prepared.prepared_inputs.prefill_embd,
                &prepared.prepared_inputs.trailing_text_hidden,
                &prepared.prepared_inputs.tts_pad_embed,
                &prepared.prompt_frames,
                req.talker_kv_mode,
                thread_count,
                req.max_audio_frames,
                req.repetition_penalty,
                req.temperature,
                req.top_k,
                req.top_p,
                chunk_size,
                &chunk_tx,
            );
            let codec_rollout_dur = t_roll.elapsed();
            drop(chunk_tx);

            let codec_rollout = codec_rollout?;
            let generated_frames = codec_rollout
                .frames
                .len()
                .saturating_sub(prepared.prefix_frame_count);

            let (generated_samples, vocoder_decode) = vocoder_handle.join().unwrap()?;

            let pipeline_wall_clock = t_pipeline_start.elapsed();
            let post = std::time::Duration::ZERO;
            let sample_rate_hz = self.vocoder.config().sample_rate as u32;
            if let Some(t) = timings {
                t.speaker_encode = prepared.speaker_encode;
                t.tokenize = prepared.tokenize;
                t.prefill_build = prepared.prefill_build;
                t.codec_rollout = codec_rollout_dur;
                t.vocoder_decode = vocoder_decode;
                t.post = post;
                t.codec_rollout_detail = codec_rollout.sub_timings;
                let sequential_sum = codec_rollout_dur + vocoder_decode;
                t.pipeline_overlap = sequential_sum.saturating_sub(pipeline_wall_clock);
                t.first_frame_latency = prepared.speaker_encode
                    + prepared.tokenize
                    + prepared.prefill_build
                    + codec_rollout.first_frame_elapsed;
                t.generated_samples = generated_samples;
                t.sample_rate_hz = sample_rate_hz;
            }

            Ok(StreamingSynthesizeResult {
                sample_rate_hz,
                generated_frames,
                generated_samples,
            })
        })
    }

    fn validate_speaker_embedding(&self, speaker_embedding: &[f32]) -> Result<(), Qwen3TtsError> {
        let expected = self.speaker_embedding_size();
        if speaker_embedding.len() != expected {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "speaker embedding must have {expected} elements"
            )));
        }
        Ok(())
    }
}

pub trait StreamingSynthesis {
    fn push_pcm_chunk(&mut self, pcm_f32: &[f32]) -> Result<(), Qwen3TtsError>;
}

impl<F> StreamingSynthesis for F
where
    F: FnMut(&[f32]) -> Result<(), Qwen3TtsError>,
{
    fn push_pcm_chunk(&mut self, pcm_f32: &[f32]) -> Result<(), Qwen3TtsError> {
        self(pcm_f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults() {
        let r = SynthesizeRequest::default();
        assert_eq!(r.temperature, 0.9);
        assert_eq!(r.top_k, 50);
        assert_eq!(r.language_id, 2050);
        assert_eq!(r.talker_kv_mode, TalkerKvMode::F16);
    }

    #[test]
    fn talker_kv_mode_parse_accepts_aliases() {
        assert_eq!(TalkerKvMode::parse("f16").unwrap(), TalkerKvMode::F16);
        assert_eq!(
            TalkerKvMode::parse("turboquant").unwrap(),
            TalkerKvMode::TurboQuant
        );
        assert_eq!(TalkerKvMode::parse("q8").unwrap(), TalkerKvMode::TurboQuant);
    }
}
