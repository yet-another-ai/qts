use std::env;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use qts::{Qwen3TtsEngine, SynthesisStageTimings, VoiceClonePromptV2};

mod cli_support;
mod tui;

use crate::cli_support::{load_engine, parse_value_arg, CommonSynthesisArgs};

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

    let result = if let LoadedConditioning::VoiceClonePrompt(prompt) = &conditioning {
        engine.synthesize_with_voice_clone_prompt(&request, prompt)?
    } else {
        engine.synthesize(&request)?
    };

    write_wav_f32(&out_path, result.sample_rate_hz, &result.pcm_f32)?;
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
             usage:\n  profile --text TEXT [--model-dir DIR] [--runs N] [--out OUT.wav] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language-id N] [--vocoder-threads N] [--chunk-size N] [--talker-kv-mode f16|turboquant] [--voice-clone-prompt PATH] [--backend auto|cpu|metal|vulkan] [--backend-fallback LIST] [--vocoder-ep auto|cpu|cuda|nvrtx|tensorrt|coreml|directml] [--vocoder-ep-fallback LIST]\n\n\
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
    Ok(LoadedConditioning::None)
}

fn write_wav_f32(path: &Path, sample_rate_hz: u32, pcm_f32: &[f32]) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("failed to create {}", path.display()))?;
    for sample in pcm_f32.iter().copied() {
        let clamped = sample.clamp(-1.0, 1.0);
        writer
            .write_sample((clamped * i16::MAX as f32) as i16)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("failed to finalize {}", path.display()))?;
    Ok(())
}

fn print_usage() {
    eprintln!(
        "usage:\n  cargo run -p qwen3-tts-cli -- synthesize --text TEXT --out OUT.wav [--model-dir DIR] [--voice-clone-prompt prompt.pb] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language-id N] [--vocoder-threads N] [--chunk-size N] [--talker-kv-mode f16|turboquant] [--backend auto|cpu|metal|vulkan] [--backend-fallback LIST] [--vocoder-ep auto|cpu|acl|armnn|azure|cann|coreml|cuda|directml|migraphx|nnapi|nvrtx|onednn|openvino|qnn|rknpu|tensorrt|tvm|vitis|webgpu|xnnpack] [--vocoder-ep-fallback LIST]\n  cargo run -p qwen3-tts-cli -- profile --text TEXT [--model-dir DIR] [--runs N] [--out OUT.wav] [--voice-clone-prompt] (same tuning flags as synthesize plus backend flags)\n  cargo run -p qwen3-tts-cli -- tui [--model-dir DIR] [--voice-clone-prompt prompt.pb] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language-id N] [--vocoder-threads N] [--chunk-size N] [--talker-kv-mode f16|turboquant] [--backend auto|cpu|metal|vulkan] [--backend-fallback LIST] [--vocoder-ep auto|cpu|acl|armnn|azure|cann|coreml|cuda|directml|migraphx|nnapi|nvrtx|onednn|openvino|qnn|rknpu|tensorrt|tvm|vitis|webgpu|xnnpack] [--vocoder-ep-fallback LIST]\n\nCLI flags override environment variables.\nDefault transformer auto chain: Apple = metal,vulkan,cpu ; others = vulkan,cpu.\nDefault vocoder auto chain: Apple = coreml,cpu ; Windows = cuda,nvrtx,tensorrt,directml,cpu ; Linux/others = cuda,nvrtx,tensorrt,cpu.\n\nEnv fallback remains available: QWEN3_TTS_BACKEND / QWEN3_TTS_BACKEND_FALLBACK / QWEN3_TTS_VOCODER_EP / QWEN3_TTS_VOCODER_EP_FALLBACK / QWEN3_TTS_TALKER_KV_MODE\n\nIf --frames is omitted, synthesize/profile derive a text-length-based max frame budget.\n\nOr from the repo root (see .cargo/config.toml): cargo xtask bench … / cargo xtask profile …"
    );
}
