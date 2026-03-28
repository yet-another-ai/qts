use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use qts::{Qwen3TtsEngine, SynthesizeRequest, TalkerKvMode};

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
    pub(crate) voice_clone_prompt: Option<PathBuf>,
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
            voice_clone_prompt: None,
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
            "--voice-clone-prompt" => {
                self.voice_clone_prompt =
                    Some(PathBuf::from(value_arg(args, idx, "--voice-clone-prompt")?));
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
