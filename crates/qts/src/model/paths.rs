use std::path::{Path, PathBuf};

/// Paths to the main GGUF plus vocoder artifacts.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// Main TTS checkpoint (e.g. `qwen3-tts-0.6b-f16.gguf`).
    pub main_gguf: PathBuf,
    /// Vocoder artifact exported for ONNX Runtime.
    pub vocoder_onnx: PathBuf,
    /// Optional speech-tokenizer encoder exported for ONNX Runtime.
    pub tokenizer_encoder_onnx: PathBuf,
    /// Optional copied source config for model-type specific runtime behavior.
    pub config_json: PathBuf,
}

impl ModelPaths {
    pub fn new(
        main_gguf: PathBuf,
        vocoder_onnx: PathBuf,
        tokenizer_encoder_onnx: PathBuf,
        config_json: PathBuf,
    ) -> Self {
        Self {
            main_gguf,
            vocoder_onnx,
            tokenizer_encoder_onnx,
            config_json,
        }
    }

    /// Resolve under a single directory using the conventional filenames.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self {
            main_gguf: choose_model_file(
                dir,
                &[
                    "qwen3-tts-0.6b-f16.gguf",
                    "qwen3-tts-0.6b-q8_0.gguf",
                    "qwen3-tts-1.7b-customvoice-f16.gguf",
                    "qwen3-tts-1.7b-customvoice-q8_0.gguf",
                    "qwen3-tts-1.7b-voicedesign-f16.gguf",
                    "qwen3-tts-1.7b-voicedesign-q8_0.gguf",
                    "qwen3-tts-0.6b-q6_k.gguf",
                    "qwen3-tts-0.6b-q5_k.gguf",
                    "qwen3-tts-0.6b-q4_k.gguf",
                ],
            ),
            vocoder_onnx: choose_vocoder_file(dir),
            tokenizer_encoder_onnx: dir.join("qwen3-tts-tokenizer-encoder.onnx"),
            config_json: dir.join("config.json"),
        }
    }

    pub fn main_exists(&self) -> bool {
        self.main_gguf.is_file()
    }

    pub fn vocoder_exists(&self) -> bool {
        self.vocoder_onnx.is_file()
    }

    pub fn tokenizer_encoder_exists(&self) -> bool {
        self.tokenizer_encoder_onnx.is_file()
    }
}

fn choose_model_file(dir: &Path, candidates: &[&str]) -> PathBuf {
    for candidate in candidates {
        let path = dir.join(candidate);
        if path.is_file() {
            return path;
        }
    }

    dir.join(candidates[0])
}

fn choose_vocoder_file(dir: &Path) -> PathBuf {
    let local = dir.join("qwen3-tts-vocoder.onnx");
    if local.is_file() {
        return local;
    }

    let Some(parent) = dir.parent() else {
        return local;
    };
    for sibling in ["Qwen3-TTS-12Hz-0.6B-Base", "Qwen3-TTS-12Hz-1.7B-Base"] {
        let path = parent.join(sibling).join("qwen3-tts-vocoder.onnx");
        if path.is_file() {
            return path;
        }
    }

    local
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_model_dir_names() {
        let p = ModelPaths::from_model_dir("/models");
        assert!(p.main_gguf.ends_with("qwen3-tts-0.6b-f16.gguf"));
        assert!(p.vocoder_onnx.ends_with("qwen3-tts-vocoder.onnx"));
        assert!(p.config_json.ends_with("config.json"));
    }

    #[test]
    fn from_model_dir_prefers_existing_main_quantized_files() {
        let dir = std::env::temp_dir().join("qwen_tts_native_paths_quantized");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("qwen3-tts-0.6b-q4_k.gguf"), b"").unwrap();

        let p = ModelPaths::from_model_dir(&dir);
        assert!(p.main_gguf.ends_with("qwen3-tts-0.6b-q4_k.gguf"));
        assert!(p.vocoder_onnx.ends_with("qwen3-tts-vocoder.onnx"));

        let _ = std::fs::remove_dir_all(dir);
    }
}
