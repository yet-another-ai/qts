use std::path::PathBuf;

use qts::{AudioCodeEncoder, VoiceClonePromptV2};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

#[test]
fn native_audio_code_encoder_matches_fixture_when_enabled() {
    let Ok(model_dir) = std::env::var("QWEN3_TTS_TEST_MODEL_DIR") else {
        eprintln!("skipping: QWEN3_TTS_TEST_MODEL_DIR is not set");
        return;
    };
    let encoder_path = PathBuf::from(model_dir).join("qwen3-tts-tokenizer-encoder.onnx");
    if !encoder_path.is_file() {
        eprintln!(
            "skipping: tokenizer encoder ONNX not found at {}",
            encoder_path.display()
        );
        return;
    }

    let encoder = AudioCodeEncoder::load_from_onnx(&encoder_path).unwrap();
    let wav_bytes = std::fs::read(fixture_path("sample1.wav")).unwrap();
    let actual = encoder.encode_wav_bytes(&wav_bytes).unwrap();

    let prompt_bytes = std::fs::read(fixture_path("sample1.icl.voice-clone-prompt.pb")).unwrap();
    let prompt = VoiceClonePromptV2::from_protobuf_bytes(&prompt_bytes).unwrap();
    let expected = prompt.ref_code.expect("fixture has ref_code");

    assert_eq!(actual.shape, expected.shape);
    let diff_count = actual
        .values
        .iter()
        .zip(expected.values.iter())
        .filter(|(actual, expected)| actual != expected)
        .count();
    assert_eq!(
        diff_count,
        0,
        "native audio code encoder differs from fixture in {diff_count}/{} tokens",
        expected.values.len()
    );
}
