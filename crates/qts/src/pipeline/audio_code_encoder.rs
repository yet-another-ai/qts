//! ONNX Runtime wrapper for Qwen3-TTS speech-tokenizer encoder.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use hound::{SampleFormat, WavReader};
use ort::session::{
    builder::{GraphOptimizationLevel, SessionBuilder},
    Session,
};
use ort::value::Tensor;

use super::{soxr_resampler::resample_soxr_hq, vocoder::VocoderExecutionProvider};
use crate::{Qwen3TtsError, TensorI32};

fn ort_err(err: impl std::fmt::Display) -> Qwen3TtsError {
    Qwen3TtsError::Ort(err.to_string())
}

#[derive(Debug, Clone)]
pub struct AudioCodeEncoderConfig {
    pub input_sample_rate: u32,
    pub n_codebooks: usize,
}

impl Default for AudioCodeEncoderConfig {
    fn default() -> Self {
        Self {
            input_sample_rate: 24_000,
            n_codebooks: 16,
        }
    }
}

pub struct AudioCodeEncoder {
    config: AudioCodeEncoderConfig,
    model_path: PathBuf,
    session: Mutex<Session>,
}

impl AudioCodeEncoder {
    pub fn load_from_onnx(path: impl AsRef<Path>) -> Result<Self, Qwen3TtsError> {
        let path = path.as_ref().to_path_buf();
        if !path.is_file() {
            return Err(Qwen3TtsError::ModelFile(path));
        }
        crate::ensure_ort_init()?;
        let session = SessionBuilder::new()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(ort_err)?
            .with_memory_pattern(false)
            .map_err(ort_err)?
            .with_intra_threads(1)
            .map_err(ort_err)?
            .commit_from_file(&path)
            .map_err(ort_err)?;

        Ok(Self {
            config: AudioCodeEncoderConfig::default(),
            model_path: path,
            session: Mutex::new(session),
        })
    }

    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    #[must_use]
    pub fn execution_provider(&self) -> VocoderExecutionProvider {
        VocoderExecutionProvider::Cpu
    }

    pub fn encode_wav_bytes(&self, wav_bytes: &[u8]) -> Result<TensorI32, Qwen3TtsError> {
        let (samples, sample_rate) = read_wav_mono_f32(wav_bytes)?;
        let samples = if sample_rate == self.config.input_sample_rate {
            samples
        } else {
            resample_soxr_hq(&samples, sample_rate, self.config.input_sample_rate)?
        };
        self.encode_samples(&samples)
    }

    fn encode_samples(&self, samples: &[f32]) -> Result<TensorI32, Qwen3TtsError> {
        let input =
            Tensor::from_array(([1usize, samples.len()], samples.to_vec())).map_err(ort_err)?;

        let mut session = self.session.lock().map_err(|_| {
            Qwen3TtsError::InvalidInput("audio code encoder session mutex poisoned".into())
        })?;
        let outputs = session.run(ort::inputs![input]).map_err(ort_err)?;
        if outputs.len() == 0 {
            return Err(Qwen3TtsError::Ort(
                "audio code encoder produced no outputs".into(),
            ));
        }
        let output = outputs.get("audio_codes").unwrap_or(&outputs[0]);
        let (shape, values) = output.try_extract_tensor::<i64>().map_err(ort_err)?;
        let dims = shape
            .iter()
            .map(|dim| usize::try_from(*dim).unwrap_or(0))
            .collect::<Vec<_>>();

        let (frames, codebooks, flat) = match dims.as_slice() {
            [1, frames, codebooks] => (*frames, *codebooks, values.iter().copied()),
            [frames, codebooks] => (*frames, *codebooks, values.iter().copied()),
            other => {
                return Err(Qwen3TtsError::Ort(format!(
                    "unexpected audio code encoder output shape: {other:?}"
                )));
            }
        };
        if codebooks != self.config.n_codebooks {
            return Err(Qwen3TtsError::Ort(format!(
                "audio code encoder returned {codebooks} codebooks, expected {}",
                self.config.n_codebooks
            )));
        }

        Ok(TensorI32 {
            shape: vec![frames as u32, codebooks as u32],
            values: flat.map(|value| value as i32).collect(),
        })
    }
}

fn read_wav_mono_f32(wav_bytes: &[u8]) -> Result<(Vec<f32>, u32), Qwen3TtsError> {
    let cursor = std::io::Cursor::new(wav_bytes);
    let mut reader =
        WavReader::new(cursor).map_err(|err| Qwen3TtsError::InvalidInput(err.to_string()))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let values = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| Qwen3TtsError::InvalidInput(err.to_string()))?,
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.map(|v| f32::from(v) / f32::from(i16::MAX)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| Qwen3TtsError::InvalidInput(err.to_string()))?,
        (SampleFormat::Int, 24 | 32) => reader
            .samples::<i32>()
            .map(|sample| sample.map(|v| v as f32 / i32::MAX as f32))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| Qwen3TtsError::InvalidInput(err.to_string()))?,
        _ => {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "unsupported WAV format: {:?} {} bits",
                spec.sample_format, spec.bits_per_sample
            )));
        }
    };

    if channels == 1 {
        return Ok((values, spec.sample_rate));
    }
    let mono = values
        .chunks(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}
