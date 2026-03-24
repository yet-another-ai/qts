//! Layer-B style checks: requires real model artifacts on disk.
//!
//! ```text
//! export QWEN3_TTS_MODEL_DIR=/path/to/models   # contains qwen3-tts-0.6b-f16.gguf + qwen3-tts-vocoder.onnx
//! cargo test -p qts integration_ -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use qts::{Qwen3TtsEngine, SynthesizeRequest, VoiceClonePromptV2};

fn require_model_dir() -> PathBuf {
    std::env::var("QWEN3_TTS_MODEL_DIR")
        .map(PathBuf::from)
        .expect("QWEN3_TTS_MODEL_DIR must be set when running ignored integration tests")
}

fn load_fixture_prompt(engine: &Qwen3TtsEngine, name: &str) -> VoiceClonePromptV2 {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|err| {
        panic!("failed to read fixture {}: {err}", path.display());
    });
    engine
        .decode_voice_clone_prompt(&bytes)
        .expect("decode voice clone prompt fixture")
}

#[test]
#[ignore = "set QWEN3_TTS_MODEL_DIR to run"]
fn integration_loads_models() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load models");
    assert!(engine.model_paths().main_exists());
    assert!(engine.model_paths().vocoder_exists());
    assert!(!engine.encode_for_tts("hello").is_empty());
}

#[test]
#[ignore = "set QWEN3_TTS_MODEL_DIR to run"]
fn integration_synthesize_direct_path_audio() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load");
    let req = SynthesizeRequest {
        text: "hello".into(),
        max_audio_frames: 4,
        ..Default::default()
    };
    let result = engine.synthesize(&req).expect("synthesize audio");
    assert_eq!(result.sample_rate_hz, 24_000);
    assert!(result.generated_frames > 0);
    assert!(!result.pcm_f32.is_empty());
}

#[test]
#[ignore = "set QWEN3_TTS_MODEL_DIR to run"]
fn integration_voice_clone_prompt_xvector_mode() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load");
    let prompt = load_fixture_prompt(&engine, "sample1.xvector.voice-clone-prompt.pb");
    assert!(prompt.x_vector_only_mode);
    assert!(!prompt.icl_mode);
    assert!(prompt.ref_code.is_none());
    assert_eq!(
        prompt.speaker_embedding_dim(),
        Some(engine.speaker_embedding_size())
    );
    let req = SynthesizeRequest {
        text: "hello".into(),
        max_audio_frames: 4,
        ..Default::default()
    };
    let result = engine
        .synthesize_with_voice_clone_prompt(&req, &prompt)
        .expect("synthesize xvector prompt");
    assert_eq!(result.sample_rate_hz, 24_000);
    assert!(result.generated_frames > 0);
    assert!(!result.pcm_f32.is_empty());
}

#[test]
#[ignore = "set QWEN3_TTS_MODEL_DIR to run"]
fn integration_voice_clone_prompt_icl_mode() {
    let dir = require_model_dir();
    let engine = Qwen3TtsEngine::from_model_dir(&dir).expect("load");
    let prompt = load_fixture_prompt(&engine, "sample1.icl.voice-clone-prompt.pb");
    assert!(!prompt.x_vector_only_mode);
    assert!(prompt.icl_mode);
    assert!(prompt.ref_code.is_some());
    assert!(!prompt.ref_text.is_empty());
    assert_eq!(
        prompt.speaker_embedding_dim(),
        Some(engine.speaker_embedding_size())
    );
    assert_eq!(
        prompt.ref_code_shape(),
        Some((105, engine.transformer().config().n_codebooks as usize))
    );
    let req = SynthesizeRequest {
        text: "world".into(),
        max_audio_frames: 4,
        ..Default::default()
    };
    let result = engine
        .synthesize_with_voice_clone_prompt(&req, &prompt)
        .expect("synthesize icl prompt");
    assert_eq!(result.sample_rate_hz, 24_000);
    assert!(result.generated_frames > 0);
    assert!(!result.pcm_f32.is_empty());
}
