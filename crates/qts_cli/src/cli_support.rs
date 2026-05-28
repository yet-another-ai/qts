use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use qts::{
    Qwen3TtsEngine, SynthesizeRequest, TalkerKvMode, TensorF32, VoiceClonePromptV2,
    VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
};

#[derive(Debug, Clone, Default)]
pub(crate) struct RuntimeBackendOverrides {
    ggml_backend: Option<String>,
    ggml_backend_fallback: Option<String>,
    vocoder_ep: Option<String>,
    vocoder_ep_fallback: Option<String>,
}

impl RuntimeBackendOverrides {
    pub(crate) fn parse_flag(&mut self, args: &[String], idx: &mut usize) -> Result<bool> {
        match args[*idx].as_str() {
            "--backend" => {
                self.ggml_backend = Some(value_arg(args, idx, "--backend")?);
                Ok(true)
            }
            "--backend-fallback" => {
                self.ggml_backend_fallback = Some(value_arg(args, idx, "--backend-fallback")?);
                Ok(true)
            }
            "--vocoder-ep" => {
                self.vocoder_ep = Some(value_arg(args, idx, "--vocoder-ep")?);
                Ok(true)
            }
            "--vocoder-ep-fallback" => {
                self.vocoder_ep_fallback = Some(value_arg(args, idx, "--vocoder-ep-fallback")?);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub(crate) fn apply_env_overrides(&self) {
        if let Some(value) = &self.ggml_backend {
            unsafe { env::set_var("QWEN3_TTS_BACKEND", value) };
        }
        if let Some(value) = &self.ggml_backend_fallback {
            unsafe { env::set_var("QWEN3_TTS_BACKEND_FALLBACK", value) };
        }
        if let Some(value) = &self.vocoder_ep {
            unsafe { env::set_var("QWEN3_TTS_VOCODER_EP", value) };
        }
        if let Some(value) = &self.vocoder_ep_fallback {
            unsafe { env::set_var("QWEN3_TTS_VOCODER_EP_FALLBACK", value) };
        }
    }

    pub(crate) fn describe(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(value) = &self.ggml_backend {
            parts.push(format!("backend={value}"));
        }
        if let Some(value) = &self.ggml_backend_fallback {
            parts.push(format!("backend_fallback={value}"));
        }
        if let Some(value) = &self.vocoder_ep {
            parts.push(format!("vocoder_ep={value}"));
        }
        if let Some(value) = &self.vocoder_ep_fallback {
            parts.push(format!("vocoder_ep_fallback={value}"));
        }
        (!parts.is_empty()).then(|| parts.join(" "))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CommonSynthesisArgs {
    pub(crate) model_dir: PathBuf,
    pub(crate) text: Option<String>,
    pub(crate) out_path: Option<PathBuf>,
    pub(crate) dump_codec_frames_path: Option<PathBuf>,
    pub(crate) voice_clone_prompt: Option<PathBuf>,
    pub(crate) voice_clone_wav: Option<PathBuf>,
    pub(crate) voice_clone_ref_text: Option<String>,
    pub(crate) speaker: Option<String>,
    pub(crate) instruct: Option<String>,
    pub(crate) thread_count: usize,
    pub(crate) max_audio_frames: Option<usize>,
    pub(crate) temperature: f32,
    pub(crate) top_k: i32,
    pub(crate) top_p: f32,
    pub(crate) repetition_penalty: f32,
    pub(crate) language_id: i32,
    pub(crate) vocoder_thread_count: usize,
    pub(crate) vocoder_chunk_size: usize,
    pub(crate) talker_kv_mode: TalkerKvMode,
    pub(crate) runtime_backends: RuntimeBackendOverrides,
}

impl CommonSynthesisArgs {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            model_dir: default_model_dir()?,
            text: None,
            out_path: None,
            dump_codec_frames_path: None,
            voice_clone_prompt: None,
            voice_clone_wav: None,
            voice_clone_ref_text: None,
            speaker: None,
            instruct: None,
            thread_count: 4,
            max_audio_frames: None,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            language_id: 2050,
            vocoder_thread_count: 4,
            vocoder_chunk_size: 0,
            talker_kv_mode: parse_talker_kv_mode_env()?,
            runtime_backends: RuntimeBackendOverrides::default(),
        })
    }

    pub(crate) fn parse_flag(&mut self, args: &[String], idx: &mut usize) -> Result<bool> {
        if self.runtime_backends.parse_flag(args, idx)? {
            return Ok(true);
        }
        match args[*idx].as_str() {
            "--model-dir" => {
                self.model_dir = PathBuf::from(value_arg(args, idx, "--model-dir")?);
                Ok(true)
            }
            "--text" => {
                self.text = Some(value_arg(args, idx, "--text")?);
                Ok(true)
            }
            "--out" => {
                self.out_path = Some(PathBuf::from(value_arg(args, idx, "--out")?));
                Ok(true)
            }
            "--dump-codec-frames" => {
                self.dump_codec_frames_path =
                    Some(PathBuf::from(value_arg(args, idx, "--dump-codec-frames")?));
                Ok(true)
            }
            "--voice-clone-prompt" => {
                self.voice_clone_prompt =
                    Some(PathBuf::from(value_arg(args, idx, "--voice-clone-prompt")?));
                Ok(true)
            }
            "--voice-clone-wav" => {
                self.voice_clone_wav =
                    Some(PathBuf::from(value_arg(args, idx, "--voice-clone-wav")?));
                Ok(true)
            }
            "--voice-clone-ref-text" => {
                self.voice_clone_ref_text = Some(value_arg(args, idx, "--voice-clone-ref-text")?);
                Ok(true)
            }
            "--speaker" => {
                self.speaker = Some(value_arg(args, idx, "--speaker")?);
                Ok(true)
            }
            "--instruct" => {
                self.instruct = Some(value_arg(args, idx, "--instruct")?);
                Ok(true)
            }
            "--threads" => {
                self.thread_count = parse_value_arg(args, idx, "--threads")?;
                Ok(true)
            }
            "--frames" => {
                self.max_audio_frames = Some(parse_value_arg(args, idx, "--frames")?);
                Ok(true)
            }
            "--temperature" => {
                self.temperature = parse_value_arg(args, idx, "--temperature")?;
                Ok(true)
            }
            "--top-k" => {
                self.top_k = parse_value_arg(args, idx, "--top-k")?;
                Ok(true)
            }
            "--top-p" => {
                self.top_p = parse_value_arg(args, idx, "--top-p")?;
                Ok(true)
            }
            "--repetition-penalty" => {
                self.repetition_penalty = parse_value_arg(args, idx, "--repetition-penalty")?;
                Ok(true)
            }
            "--language-id" => {
                self.language_id = parse_value_arg(args, idx, "--language-id")?;
                Ok(true)
            }
            "--vocoder-threads" => {
                self.vocoder_thread_count = parse_value_arg(args, idx, "--vocoder-threads")?;
                Ok(true)
            }
            "--chunk-size" => {
                self.vocoder_chunk_size = parse_value_arg(args, idx, "--chunk-size")?;
                Ok(true)
            }
            "--talker-kv-mode" => {
                self.talker_kv_mode =
                    TalkerKvMode::parse(&value_arg(args, idx, "--talker-kv-mode")?)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub(crate) fn validate_conditioning(&self) -> Result<()> {
        let clone_modes = usize::from(self.voice_clone_prompt.is_some())
            + usize::from(self.voice_clone_wav.is_some());
        if clone_modes > 1 {
            anyhow::bail!("--voice-clone-prompt cannot be combined with --voice-clone-wav");
        }
        if clone_modes > 0 && self.speaker.is_some() {
            anyhow::bail!("voice clone flags cannot be combined with --speaker");
        }
        if clone_modes > 0 && self.instruct.is_some() {
            anyhow::bail!("voice clone flags cannot be combined with --instruct");
        }
        if self.voice_clone_ref_text.is_some() && self.voice_clone_wav.is_none() {
            anyhow::bail!("--voice-clone-ref-text requires --voice-clone-wav");
        }
        Ok(())
    }

    pub(crate) fn require_text(&self) -> Result<String> {
        self.text.clone().context("--text is required")
    }

    pub(crate) fn require_out_path(&self) -> Result<PathBuf> {
        self.out_path.clone().context("--out is required")
    }

    pub(crate) fn build_request(&self, text: String) -> Result<SynthesizeRequest> {
        self.validate_conditioning()?;
        Ok(SynthesizeRequest {
            text,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            max_audio_frames: self.max_audio_frames.unwrap_or(256),
            thread_count: self.thread_count,
            repetition_penalty: self.repetition_penalty,
            language_id: self.language_id,
            vocoder_thread_count: self.vocoder_thread_count,
            vocoder_chunk_size: self.vocoder_chunk_size,
            talker_kv_mode: self.talker_kv_mode,
        })
    }
}

fn parse_talker_kv_mode_env() -> Result<TalkerKvMode> {
    match env::var("QWEN3_TTS_TALKER_KV_MODE") {
        Ok(value) if !value.trim().is_empty() => Ok(TalkerKvMode::parse(&value)?),
        _ => Ok(TalkerKvMode::F16),
    }
}

pub(crate) fn load_engine(
    model_dir: &Path,
    runtime_backends: &RuntimeBackendOverrides,
) -> Result<Qwen3TtsEngine> {
    runtime_backends.apply_env_overrides();
    Qwen3TtsEngine::from_model_dir(model_dir)
        .with_context(|| format!("failed to load model dir {}", model_dir.display()))
}

pub(crate) fn default_model_dir() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("qwen3-tts-cli manifest has no workspace parent")?;
    Ok(workspace_root.join("models/qwen3-tts-bundle"))
}

pub(crate) fn value_arg(args: &[String], idx: &mut usize, flag: &str) -> Result<String> {
    *idx += 1;
    let value = args
        .get(*idx)
        .with_context(|| format!("missing value for {flag}"))?
        .clone();
    *idx += 1;
    Ok(value)
}

pub(crate) fn parse_value_arg<T>(args: &[String], idx: &mut usize, flag: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = value_arg(args, idx, flag)?;
    value
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid value for {flag}: {err}"))
}

pub(crate) fn encode_wav_f32(sample_rate_hz: u32, pcm_f32: &[f32]) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec).context("failed to create WAV")?;
        for sample in pcm_f32.iter().copied() {
            let clamped = sample.clamp(-1.0, 1.0);
            writer
                .write_sample((clamped * i16::MAX as f32) as i16)
                .context("failed to write WAV sample")?;
        }
        writer.finalize().context("failed to finalize WAV")?;
    }
    Ok(cursor.into_inner())
}

pub(crate) fn write_wav_f32(path: &Path, sample_rate_hz: u32, pcm_f32: &[f32]) -> Result<()> {
    let bytes = encode_wav_f32(sample_rate_hz, pcm_f32)?;
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

pub(crate) fn build_wav_only_voice_clone_prompt(
    engine: &Qwen3TtsEngine,
    model_dir: &Path,
    source: impl Into<String>,
    wav_bytes: &[u8],
) -> Result<VoiceClonePromptV2> {
    let speaker_embedding = engine.encode_reference_speaker(wav_bytes)?;
    let prompt = VoiceClonePromptV2 {
        schema_version: VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
        source: source.into(),
        model_id: model_dir.display().to_string(),
        speaker_encoder_sample_rate_hz: 16_000,
        x_vector_only_mode: true,
        icl_mode: false,
        ref_text: String::new(),
        ref_code: None,
        ref_spk_embedding: Some(TensorF32 {
            shape: vec![speaker_embedding.len() as u32],
            values: speaker_embedding,
        }),
    };
    prompt.validate()?;
    Ok(prompt)
}

pub(crate) fn build_icl_voice_clone_prompt(
    engine: &Qwen3TtsEngine,
    model_dir: &Path,
    source: impl Into<String>,
    wav_bytes: &[u8],
    ref_text: &str,
) -> Result<VoiceClonePromptV2> {
    let speaker_embedding = engine.encode_reference_speaker(wav_bytes)?;
    let ref_code = engine.encode_reference_audio_codes(wav_bytes).with_context(|| {
        format!(
            "failed to encode reference audio codes; ensure qwen3-tts-tokenizer-encoder.onnx exists in {}",
            model_dir.display()
        )
    })?;
    let prompt = VoiceClonePromptV2 {
        schema_version: VOICE_CLONE_PROMPT_V2_SCHEMA_VERSION,
        source: source.into(),
        model_id: model_dir.display().to_string(),
        speaker_encoder_sample_rate_hz: 16_000,
        x_vector_only_mode: false,
        icl_mode: true,
        ref_text: ref_text.to_string(),
        ref_code: Some(ref_code),
        ref_spk_embedding: Some(TensorF32 {
            shape: vec![speaker_embedding.len() as u32],
            values: speaker_embedding,
        }),
    };
    prompt.validate()?;
    Ok(prompt)
}
