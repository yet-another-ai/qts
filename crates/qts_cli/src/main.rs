use std::env;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use qts::{CodePredDebugStep, Qwen3TtsEngine, SynthesisStageTimings, VoiceClonePromptV2};
use serde::Serialize;

mod cli_support;
mod tui;

use crate::cli_support::{
    build_icl_voice_clone_prompt as build_icl_voice_clone_prompt_from_bytes,
    build_wav_only_voice_clone_prompt as build_wav_only_voice_clone_prompt_from_bytes, load_engine,
    parse_value_arg, write_wav_f32, CommonSynthesisArgs,
};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("synthesize") => run_synthesize(args.collect()),
        Some("profile") => run_profile(args.collect()),
        Some("tui") => tui::run(args.collect()),
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn run_synthesize(args: Vec<String>) -> Result<()> {
    let mut common = CommonSynthesisArgs::new()?;

    let mut idx = 0;
    while idx < args.len() {
        if common.parse_flag(&args, &mut idx)? {
            continue;
        }
        bail!("unknown synthesize argument: {}", args[idx]);
    }

    common.validate_conditioning()?;
    let text = common.require_text()?;
    let out_path = common.require_out_path()?;
    let engine = load_engine(&common.model_dir, &common.runtime_backends)?;
    let request = common.build_request(text)?;
    let conditioning = load_synthesis_conditioning(&engine, &common)?;
    if common.dump_codec_frames_path.is_none() {
        let result = match &conditioning {
            LoadedConditioning::VoiceClonePrompt(prompt) => {
                engine.synthesize_with_voice_clone_prompt(&request, prompt)?
            }
            LoadedConditioning::CustomVoice { speaker, instruct } => {
                engine.synthesize_with_custom_voice(&request, speaker, instruct.as_deref())?
            }
            LoadedConditioning::VoiceDesign { instruct } => {
                engine.synthesize_with_voice_design(&request, instruct)?
            }
            LoadedConditioning::None => engine.synthesize(&request)?,
        };
        write_wav_f32(&out_path, result.sample_rate_hz, &result.pcm_f32)?;
        eprintln!(
            "wrote synthesis: path={} sample_rate={} frames={} samples={}",
            out_path.display(),
            result.sample_rate_hz,
            result.generated_frames,
            result.pcm_f32.len()
        );
        return Ok(());
    }

    let debug_result = if let LoadedConditioning::VoiceClonePrompt(prompt) = &conditioning {
        engine.synthesize_with_voice_clone_prompt_debug(&request, prompt)?
    } else if let LoadedConditioning::CustomVoice { speaker, instruct } = &conditioning {
        engine.synthesize_with_custom_voice_debug(&request, speaker, instruct.as_deref())?
    } else if let LoadedConditioning::VoiceDesign { instruct } = &conditioning {
        engine.synthesize_with_voice_design_debug(&request, instruct)?
    } else {
        engine.synthesize_debug(&request)?
    };
    let result = debug_result.synthesis;

    write_wav_f32(&out_path, result.sample_rate_hz, &result.pcm_f32)?;
    if let Some(path) = &common.dump_codec_frames_path {
        write_codec_frames_json(
            path,
            &debug_result.codec_frames,
            &debug_result.talker_hidden_states,
            &debug_result.debug_trailing_rows,
            &debug_result.debug_step_embeddings,
            &debug_result.debug_code_pred_steps,
        )?;
        eprintln!(
            "wrote codec frames: path={} frames={} prefix_frames={}",
            path.display(),
            debug_result.codec_frames.len(),
            debug_result.prefix_frame_count
        );
    }
    eprintln!(
        "wrote synthesis: path={} sample_rate={} frames={} samples={}",
        out_path.display(),
        result.sample_rate_hz,
        result.generated_frames,
        result.pcm_f32.len()
    );
    Ok(())
}

fn run_profile(args: Vec<String>) -> Result<()> {
    if args.iter().any(|a| matches!(a.as_str(), "--help" | "-h")) {
        eprintln!(
            "qwen3-tts-cli profile — print per-stage synthesis timings (wall clock)\n\n\
             usage:\n  profile --text TEXT [--model-dir DIR] [--runs N] [--out OUT.wav] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language-id N] [--speaker NAME] [--instruct TEXT] [--vocoder-threads N] [--chunk-size N] [--talker-kv-mode f16|turboquant] [--voice-clone-prompt PATH | --voice-clone-wav REF.wav [--voice-clone-ref-text TEXT]] [--backend auto|cpu|metal|vulkan] [--backend-fallback LIST] [--vocoder-ep auto|cpu|cuda|nvrtx|tensorrt|coreml|directml] [--vocoder-ep-fallback LIST]\n\n\
             CLI flags override environment variables.\n\
             Default transformer auto chain: Apple = metal,vulkan,cpu ; others = vulkan,cpu.\n\
             Default vocoder auto chain: Apple = coreml,cpu ; Windows = cuda,nvrtx,tensorrt,directml,cpu ; Linux/others = cuda,nvrtx,tensorrt,cpu.\n\n\
             If --frames is omitted, the CLI derives a text-length-based max frame budget.\n\
             --runs N averages stage times over N full synthesize passes (default 1).\n\
             --out writes WAV from the first pass only when --runs > 1."
        );
        return Ok(());
    }

    let mut common = CommonSynthesisArgs::new()?;
    let mut runs = 1usize;

    let mut idx = 0;
    while idx < args.len() {
        if common.parse_flag(&args, &mut idx)? {
            continue;
        }
        match args[idx].as_str() {
            "--runs" => {
                runs = parse_value_arg(&args, &mut idx, "--runs")?;
            }
            other => bail!("unknown profile argument: {other}"),
        }
    }

    if runs == 0 {
        bail!("--runs must be >= 1");
    }

    common.validate_conditioning()?;
    let text = common.require_text()?;
    let engine = load_engine(&common.model_dir, &common.runtime_backends)?;
    let request = common.build_request(text)?;
    let conditioning = load_synthesis_conditioning(&engine, &common)?;

    let mut samples = Vec::with_capacity(runs);
    let mut first_result = None;

    for run_idx in 0..runs {
        let (result, timings) = match &conditioning {
            LoadedConditioning::VoiceClonePrompt(prompt) => {
                engine.synthesize_with_voice_clone_prompt_profile(&request, prompt)?
            }
            LoadedConditioning::CustomVoice { speaker, instruct } => engine
                .synthesize_with_custom_voice_profile(&request, speaker, instruct.as_deref())?,
            LoadedConditioning::VoiceDesign { instruct } => {
                engine.synthesize_with_voice_design_profile(&request, instruct)?
            }
            LoadedConditioning::None => engine.synthesize_with_profile(&request)?,
        };
        if run_idx == 0 {
            first_result = Some(result);
        }
        samples.push(timings);
    }

    if let Some(path) = common.out_path {
        let result = first_result.context("internal: missing first synthesis result")?;
        write_wav_f32(&path, result.sample_rate_hz, &result.pcm_f32)?;
        eprintln!(
            "wrote profile run #1 WAV: path={} sample_rate={} frames={}",
            path.display(),
            result.sample_rate_hz,
            result.generated_frames
        );
    }

    let summary = if runs == 1 {
        samples[0].format_table()
    } else {
        let avg = SynthesisStageTimings::average(&samples).expect("non-empty samples");
        format!("averaged over {runs} runs\n{}", avg.format_table())
    };
    eprint!("{summary}");

    Ok(())
}

enum LoadedConditioning {
    None,
    VoiceClonePrompt(VoiceClonePromptV2),
    CustomVoice {
        speaker: String,
        instruct: Option<String>,
    },
    VoiceDesign {
        instruct: String,
    },
}

fn load_synthesis_conditioning(
    engine: &Qwen3TtsEngine,
    args: &CommonSynthesisArgs,
) -> Result<LoadedConditioning> {
    if let Some(path) = &args.voice_clone_prompt {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let prompt = engine.decode_voice_clone_prompt(&bytes)?;
        return Ok(LoadedConditioning::VoiceClonePrompt(prompt));
    }
    if let Some(path) = &args.voice_clone_wav {
        let prompt = if let Some(ref_text) = &args.voice_clone_ref_text {
            build_icl_voice_clone_prompt(engine, args, path, ref_text)?
        } else {
            build_wav_only_voice_clone_prompt(engine, args, path)?
        };
        return Ok(LoadedConditioning::VoiceClonePrompt(prompt));
    }
    if let Some(speaker) = &args.speaker {
        if engine.custom_voice_metadata().is_none() {
            bail!(
                "speaker '{}' requested, but {} does not expose custom voice metadata",
                speaker,
                args.model_dir.display()
            );
        }
        return Ok(LoadedConditioning::CustomVoice {
            speaker: speaker.clone(),
            instruct: args.instruct.clone(),
        });
    }
    if let Some(instruct) = &args.instruct {
        if engine.voice_model_kind().is_voice_design() {
            return Ok(LoadedConditioning::VoiceDesign {
                instruct: instruct.clone(),
            });
        }
        bail!("--instruct without --speaker requires a voice_design model");
    }
    Ok(LoadedConditioning::None)
}

fn build_wav_only_voice_clone_prompt(
    engine: &Qwen3TtsEngine,
    args: &CommonSynthesisArgs,
    path: &Path,
) -> Result<VoiceClonePromptV2> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    build_wav_only_voice_clone_prompt_from_bytes(
        engine,
        &args.model_dir,
        path.display().to_string(),
        &bytes,
    )
}

fn build_icl_voice_clone_prompt(
    engine: &Qwen3TtsEngine,
    args: &CommonSynthesisArgs,
    path: &Path,
    ref_text: &str,
) -> Result<VoiceClonePromptV2> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    build_icl_voice_clone_prompt_from_bytes(
        engine,
        &args.model_dir,
        path.display().to_string(),
        &bytes,
        ref_text,
    )
}

#[derive(Debug, Serialize)]
struct CodecFramesDump<'a> {
    frames: usize,
    first10: Vec<Vec<i32>>,
    all: &'a [Vec<i32>],
    hidden_first2: Vec<Vec<f32>>,
    trailing_first2: Vec<Vec<f32>>,
    step_embd_first2: Vec<Vec<f32>>,
    codepred_first2: Vec<Vec<SerializableCodePredDebugStep>>,
}

#[derive(Debug, Serialize)]
struct SerializableCodePredDebugStep {
    codebook_index: usize,
    selected_token: i32,
    top_logits: Vec<SerializableTopLogit>,
}

#[derive(Debug, Serialize)]
struct SerializableTopLogit {
    token: i32,
    logit: f32,
}

fn write_codec_frames_json(
    path: &Path,
    codec_frames: &[Vec<i32>],
    talker_hidden_states: &[Vec<f32>],
    debug_trailing_rows: &[Vec<f32>],
    debug_step_embeddings: &[Vec<f32>],
    debug_code_pred_steps: &[Vec<CodePredDebugStep>],
) -> Result<()> {
    let payload = CodecFramesDump {
        frames: codec_frames.len(),
        first10: codec_frames.iter().take(10).cloned().collect(),
        all: codec_frames,
        hidden_first2: talker_hidden_states.iter().take(2).cloned().collect(),
        trailing_first2: debug_trailing_rows.iter().take(2).cloned().collect(),
        step_embd_first2: debug_step_embeddings.iter().take(2).cloned().collect(),
        codepred_first2: debug_code_pred_steps
            .iter()
            .take(2)
            .map(|steps| {
                steps
                    .iter()
                    .map(|step| SerializableCodePredDebugStep {
                        codebook_index: step.codebook_index,
                        selected_token: step.selected_token,
                        top_logits: step
                            .top_logits
                            .iter()
                            .map(|item| SerializableTopLogit {
                                token: item.token,
                                logit: item.logit,
                            })
                            .collect(),
                    })
                    .collect()
            })
            .collect(),
    };
    let json = serde_json::to_vec(&payload).context("failed to serialize codec frames JSON")?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn print_usage() {
    eprintln!(
        "usage:\n  cargo run -p qwen3-tts-cli -- synthesize --text TEXT --out OUT.wav [--model-dir DIR] [--speaker NAME] [--instruct TEXT] [--voice-clone-prompt prompt.pb | --voice-clone-wav REF.wav [--voice-clone-ref-text TEXT]] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language-id N] [--vocoder-threads N] [--chunk-size N] [--talker-kv-mode f16|turboquant] [--backend auto|cpu|metal|vulkan] [--backend-fallback LIST] [--vocoder-ep auto|cpu|acl|armnn|azure|cann|coreml|cuda|directml|migraphx|nnapi|nvrtx|onednn|openvino|qnn|rknpu|tensorrt|tvm|vitis|webgpu|xnnpack] [--vocoder-ep-fallback LIST]\n  cargo run -p qwen3-tts-cli -- profile --text TEXT [--model-dir DIR] [--runs N] [--out OUT.wav] [--speaker NAME] [--instruct TEXT] [--voice-clone-prompt prompt.pb | --voice-clone-wav REF.wav [--voice-clone-ref-text TEXT]] (same tuning flags as synthesize plus backend flags)\n  cargo run -p qwen3-tts-cli -- tui [--model-dir DIR] [--voice-clone-prompt prompt.pb] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language-id N] [--vocoder-threads N] [--chunk-size N] [--talker-kv-mode f16|turboquant] [--backend auto|cpu|metal|vulkan] [--backend-fallback LIST] [--vocoder-ep auto|cpu|acl|armnn|azure|cann|coreml|cuda|directml|migraphx|nnapi|nvrtx|onednn|openvino|qnn|rknpu|tensorrt|tvm|vitis|webgpu|xnnpack] [--vocoder-ep-fallback LIST]\n\nCLI flags override environment variables.\nDefault transformer auto chain: Apple = metal,vulkan,cpu ; others = vulkan,cpu.\nDefault vocoder auto chain: Apple = coreml,cpu ; Windows = cuda,nvrtx,tensorrt,directml,cpu ; Linux/others = cuda,nvrtx,tensorrt,cpu.\n\nEnv fallback remains available: QWEN3_TTS_BACKEND / QWEN3_TTS_BACKEND_FALLBACK / QWEN3_TTS_VOCODER_EP / QWEN3_TTS_VOCODER_EP_FALLBACK / QWEN3_TTS_TALKER_KV_MODE\n\n--voice-clone-wav alone uses pure Rust x-vector clone. Add --voice-clone-ref-text to use the native ICL clone path with qwen3-tts-tokenizer-encoder.onnx and bundled soxr audio-code extraction.\n\nIf --frames is omitted, synthesize/profile derive a text-length-based max frame budget.\n\nOr from the repo root (see .cargo/config.toml): cargo xtask bench … / cargo xtask profile …"
    );
}
