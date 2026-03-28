use std::env;
use std::hint::black_box;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use qts::{Qwen3TtsEngine, SynthesizeRequest, TalkerKvMode};

struct BenchConfig {
    backend_label: String,
    model_dir: PathBuf,
    text: String,
    thread_count: usize,
    max_audio_frames: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    talker_kv_mode: TalkerKvMode,
}

impl BenchConfig {
    fn from_env() -> Self {
        Self {
            backend_label: env::var("QWEN3_TTS_BENCH_BACKEND").unwrap_or_else(|_| "cpu".into()),
            model_dir: PathBuf::from(
                env::var("QWEN3_TTS_BENCH_MODEL_DIR")
                    .expect("QWEN3_TTS_BENCH_MODEL_DIR must be set for criterion benchmarks"),
            ),
            text: env::var("QWEN3_TTS_BENCH_TEXT").unwrap_or_else(|_| "hello".into()),
            thread_count: parse_env("QWEN3_TTS_BENCH_THREADS", 4usize),
            max_audio_frames: parse_env("QWEN3_TTS_BENCH_MAX_AUDIO_FRAMES", 16usize),
            temperature: parse_env("QWEN3_TTS_BENCH_TEMPERATURE", 0.0f32),
            top_k: parse_env("QWEN3_TTS_BENCH_TOP_K", 0i32),
            top_p: parse_env("QWEN3_TTS_BENCH_TOP_P", 1.0f32),
            talker_kv_mode: env::var("QWEN3_TTS_BENCH_TALKER_KV_MODE")
                .ok()
                .map(|value| {
                    TalkerKvMode::parse(&value).expect("invalid QWEN3_TTS_BENCH_TALKER_KV_MODE")
                })
                .unwrap_or(TalkerKvMode::F16),
        }
    }
}

fn parse_env<T>(key: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn bench_synthesize(c: &mut Criterion) {
    let cfg = BenchConfig::from_env();
    let engine = Qwen3TtsEngine::from_model_dir(&cfg.model_dir).unwrap_or_else(|err| {
        panic!(
            "failed to load model dir {}: {err}",
            cfg.model_dir.display()
        )
    });
    let req = SynthesizeRequest {
        text: cfg.text.clone(),
        thread_count: cfg.thread_count,
        max_audio_frames: cfg.max_audio_frames,
        temperature: cfg.temperature,
        top_k: cfg.top_k,
        top_p: cfg.top_p,
        talker_kv_mode: cfg.talker_kv_mode,
        ..Default::default()
    };

    let mut group = c.benchmark_group("synthesize");
    group.bench_function(
        BenchmarkId::new(
            &cfg.backend_label,
            format!(
                "threads={} frames={} talker_kv={}",
                cfg.thread_count,
                cfg.max_audio_frames,
                cfg.talker_kv_mode.as_str()
            ),
        ),
        |b| {
            b.iter(|| {
                let result = engine.synthesize(&req).expect("synthesis benchmark failed");
                black_box(result.generated_frames);
                black_box(result.pcm_f32.len());
            });
        },
    );
    group.finish();
}

criterion_group!(benches, bench_synthesize);
criterion_main!(benches);
