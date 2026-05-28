use std::collections::HashMap;
use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use axum::body::Body;
use axum::extract::{Multipart, Path as AxumPath, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use qts::{
    Qwen3TtsEngine, SynthesisProgress, SynthesisProgressStage, SynthesizeRequest, SynthesizeResult,
    TalkerKvMode, VoiceClonePromptV2,
};
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[path = "../cli_support.rs"]
mod cli_support;

use cli_support::{
    build_icl_voice_clone_prompt, build_wav_only_voice_clone_prompt, default_model_dir,
    encode_wav_f32, load_engine, parse_value_arg, value_arg, RuntimeBackendOverrides,
};

#[tokio::main]
async fn main() -> Result<()> {
    let config = ServerConfig::parse(env::args().skip(1).collect())?;
    config.validate()?;

    let jobs = Arc::new(Mutex::new(HashMap::new()));
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();

    let worker_jobs = Arc::clone(&jobs);
    let worker_config = config.clone();
    std::thread::spawn(move || run_worker(worker_config, worker_jobs, cmd_rx, ready_tx));

    match ready_rx
        .recv()
        .context("server worker did not report readiness")?
    {
        Ok(()) => {}
        Err(err) => bail!("{err}"),
    }

    let state = AppState {
        config: Arc::new(config.clone()),
        jobs,
        command_tx: Arc::new(Mutex::new(cmd_tx)),
        next_job_id: Arc::new(AtomicU64::new(1)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/audio/speech", post(openai_speech))
        .route("/v1/qts/audio/jobs", post(create_job))
        .route("/v1/qts/audio/jobs/multipart", post(create_multipart_job))
        .route("/v1/qts/audio/jobs/{job_id}", get(get_job))
        .route("/v1/qts/audio/jobs/{job_id}/audio", get(get_job_audio))
        .with_state(state);

    let addr = SocketAddr::new(config.host, config.port);
    eprintln!(
        "qts_server listening on http://{addr} mode={} model={}",
        config.mode.name(),
        config.model_dir.display()
    );
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Clone)]
struct AppState {
    config: Arc<ServerConfig>,
    jobs: JobStore,
    command_tx: Arc<Mutex<mpsc::Sender<WorkerCommand>>>,
    next_job_id: Arc<AtomicU64>,
}

type JobStore = Arc<Mutex<HashMap<String, JobInfo>>>;

const LONG_TEXT_TARGET_CHARS: usize = 120;
const LONG_TEXT_MAX_CHARS: usize = 160;
const LONG_TEXT_SHORT_CHARS: usize = 60;
const LONG_TEXT_SINGLE_PASS_CHARS: usize = 220;
const LONG_TEXT_MIN_TAIL_CHARS: usize = 45;

#[derive(Clone)]
struct ServerConfig {
    model_dir: PathBuf,
    host: std::net::IpAddr,
    port: u16,
    mode: FixedMode,
    request_defaults: RequestDefaults,
    runtime_backends: RuntimeBackendOverrides,
}

#[derive(Clone)]
struct RequestDefaults {
    thread_count: usize,
    max_audio_frames: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    repetition_penalty: f32,
    language_id: i32,
    vocoder_thread_count: usize,
    vocoder_chunk_size: usize,
    talker_kv_mode: TalkerKvMode,
}

#[derive(Clone)]
enum FixedMode {
    None,
    Custom {
        speaker: String,
        instruct: Option<String>,
    },
    Design {
        instruct: Option<String>,
    },
    Clone {
        prompt_path: Option<PathBuf>,
        wav_path: Option<PathBuf>,
        ref_text: Option<String>,
    },
}

impl FixedMode {
    fn parse(value: &str, args: &ModeArgs) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "custom" => Ok(Self::Custom {
                speaker: args
                    .speaker
                    .clone()
                    .context("--speaker is required when --mode custom")?,
                instruct: args.instruct.clone(),
            }),
            "design" => Ok(Self::Design {
                instruct: args.instruct.clone(),
            }),
            "clone" => Ok(Self::Clone {
                prompt_path: args.voice_clone_prompt.clone(),
                wav_path: args.voice_clone_wav.clone(),
                ref_text: args.voice_clone_ref_text.clone(),
            }),
            other => bail!("unknown --mode {other}; expected none, custom, design, clone"),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Custom { .. } => "custom",
            Self::Design { .. } => "design",
            Self::Clone { .. } => "clone",
        }
    }
}

#[derive(Default)]
struct ModeArgs {
    mode: Option<String>,
    speaker: Option<String>,
    instruct: Option<String>,
    voice_clone_prompt: Option<PathBuf>,
    voice_clone_wav: Option<PathBuf>,
    voice_clone_ref_text: Option<String>,
}

impl ServerConfig {
    fn parse(args: Vec<String>) -> Result<Self> {
        let mut model_dir = default_model_dir()?;
        let mut host = "127.0.0.1".parse()?;
        let mut port = 8080u16;
        let mut mode_args = ModeArgs::default();
        let mut runtime_backends = RuntimeBackendOverrides::default();
        let mut defaults = RequestDefaults {
            thread_count: 4,
            max_audio_frames: 256,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            language_id: 2050,
            vocoder_thread_count: 4,
            vocoder_chunk_size: 0,
            talker_kv_mode: TalkerKvMode::F16,
        };

        let mut idx = 0;
        while idx < args.len() {
            if runtime_backends.parse_flag(&args, &mut idx)? {
                continue;
            }
            match args[idx].as_str() {
                "--model-dir" => {
                    model_dir = PathBuf::from(value_arg(&args, &mut idx, "--model-dir")?)
                }
                "--host" => host = value_arg(&args, &mut idx, "--host")?.parse()?,
                "--port" => port = parse_value_arg(&args, &mut idx, "--port")?,
                "--mode" => mode_args.mode = Some(value_arg(&args, &mut idx, "--mode")?),
                "--speaker" => mode_args.speaker = Some(value_arg(&args, &mut idx, "--speaker")?),
                "--instruct" => {
                    mode_args.instruct = Some(value_arg(&args, &mut idx, "--instruct")?)
                }
                "--voice-clone-prompt" => {
                    mode_args.voice_clone_prompt = Some(PathBuf::from(value_arg(
                        &args,
                        &mut idx,
                        "--voice-clone-prompt",
                    )?));
                }
                "--voice-clone-wav" => {
                    mode_args.voice_clone_wav = Some(PathBuf::from(value_arg(
                        &args,
                        &mut idx,
                        "--voice-clone-wav",
                    )?));
                }
                "--voice-clone-ref-text" => {
                    mode_args.voice_clone_ref_text =
                        Some(value_arg(&args, &mut idx, "--voice-clone-ref-text")?);
                }
                "--threads" => {
                    defaults.thread_count = parse_value_arg(&args, &mut idx, "--threads")?
                }
                "--frames" => {
                    defaults.max_audio_frames = parse_value_arg(&args, &mut idx, "--frames")?;
                }
                "--temperature" => {
                    defaults.temperature = parse_value_arg(&args, &mut idx, "--temperature")?;
                }
                "--top-k" => defaults.top_k = parse_value_arg(&args, &mut idx, "--top-k")?,
                "--top-p" => defaults.top_p = parse_value_arg(&args, &mut idx, "--top-p")?,
                "--repetition-penalty" => {
                    defaults.repetition_penalty =
                        parse_value_arg(&args, &mut idx, "--repetition-penalty")?;
                }
                "--language-id" => {
                    defaults.language_id = parse_value_arg(&args, &mut idx, "--language-id")?;
                }
                "--vocoder-threads" => {
                    defaults.vocoder_thread_count =
                        parse_value_arg(&args, &mut idx, "--vocoder-threads")?;
                }
                "--chunk-size" => {
                    defaults.vocoder_chunk_size = parse_value_arg(&args, &mut idx, "--chunk-size")?;
                }
                "--talker-kv-mode" => {
                    defaults.talker_kv_mode =
                        TalkerKvMode::parse(&value_arg(&args, &mut idx, "--talker-kv-mode")?)?;
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => bail!("unknown qts_server argument: {other}"),
            }
        }

        let mode = FixedMode::parse(
            &mode_args
                .mode
                .clone()
                .context("--mode is required; choose none, custom, design, or clone")?,
            &mode_args,
        )?;

        Ok(Self {
            model_dir,
            host,
            port,
            mode,
            request_defaults: defaults,
            runtime_backends,
        })
    }

    fn validate(&self) -> Result<()> {
        if let FixedMode::Clone {
            prompt_path,
            wav_path,
            ..
        } = &self.mode
        {
            if prompt_path.is_some() && wav_path.is_some() {
                bail!("--voice-clone-prompt cannot be combined with --voice-clone-wav");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct SpeechRequest {
    text: Option<String>,
    input: Option<String>,
    model: Option<String>,
    voice: Option<String>,
    instructions: Option<String>,
    response_format: Option<String>,
    speed: Option<f32>,
    language_id: Option<i32>,
    frames: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    vocoder_threads: Option<usize>,
    chunk_size: Option<usize>,
    talker_kv_mode: Option<String>,
    qts_conditioning: Option<QtsConditioning>,
}

#[derive(Debug, Clone, Deserialize)]
struct QtsConditioning {
    #[serde(rename = "type")]
    kind: Option<String>,
    speaker: Option<String>,
    instruct: Option<String>,
    prompt_path: Option<PathBuf>,
    wav_path: Option<PathBuf>,
    ref_text: Option<String>,
}

#[derive(Debug, Clone)]
struct JobPayload {
    request: SpeechRequest,
    upload_wav: Option<Vec<u8>>,
}

struct WorkerCommand {
    job_id: String,
    payload: JobPayload,
    response: Option<tokio::sync::oneshot::Sender<Result<AudioPayload, String>>>,
}

#[derive(Debug, Clone)]
struct AudioPayload {
    bytes: Vec<u8>,
    sample_rate_hz: u32,
    generated_frames: usize,
}

#[derive(Debug, Clone, Serialize)]
struct JobInfo {
    job_id: String,
    status: String,
    stage: String,
    progress: f32,
    generated_frames: usize,
    max_frames: usize,
    sample_rate_hz: Option<u32>,
    error: Option<String>,
    created_at_ms: u128,
    started_at_ms: Option<u128>,
    finished_at_ms: Option<u128>,
    #[serde(skip)]
    audio: Option<Vec<u8>>,
}

#[derive(Debug, Serialize)]
struct CreateJobResponse {
    job_id: String,
    status: String,
    progress_url: String,
    audio_url: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    mode: &'static str,
    model_dir: String,
    queued_or_running_jobs: usize,
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let queued_or_running_jobs = state
        .jobs
        .lock()
        .expect("job lock poisoned")
        .values()
        .filter(|job| job.status == "queued" || job.status == "running")
        .count();
    Json(HealthResponse {
        status: "ok",
        mode: state.config.mode.name(),
        model_dir: state.config.model_dir.display().to_string(),
        queued_or_running_jobs,
    })
}

async fn openai_speech(
    State(state): State<AppState>,
    Json(request): Json<SpeechRequest>,
) -> Result<Response, ApiError> {
    validate_response_format(&request)?;
    let (job_id, rx) = submit_job(&state, request, None, true)?;
    match rx.expect("sync response channel").await {
        Ok(Ok(audio)) => audio_response(audio),
        Ok(Err(err)) => Err(ApiError::bad_request(err)),
        Err(_) => Err(ApiError::internal(format!(
            "job {job_id} response channel closed"
        ))),
    }
}

async fn create_job(
    State(state): State<AppState>,
    Json(request): Json<SpeechRequest>,
) -> Result<Json<CreateJobResponse>, ApiError> {
    validate_response_format(&request)?;
    let (job_id, _) = submit_job(&state, request, None, false)?;
    Ok(Json(CreateJobResponse {
        status: "queued".into(),
        progress_url: format!("/v1/qts/audio/jobs/{job_id}"),
        audio_url: format!("/v1/qts/audio/jobs/{job_id}/audio"),
        job_id,
    }))
}

async fn create_multipart_job(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<CreateJobResponse>, ApiError> {
    let mut request = None;
    let mut ref_wav = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| ApiError::bad_request(err.to_string()))?
    {
        let name = field.name().unwrap_or_default().to_string();
        let bytes = field
            .bytes()
            .await
            .map_err(|err| ApiError::bad_request(err.to_string()))?;
        match name.as_str() {
            "request" => {
                request = Some(
                    serde_json::from_slice::<SpeechRequest>(&bytes)
                        .map_err(|err| ApiError::bad_request(err.to_string()))?,
                );
            }
            "ref_wav" => ref_wav = Some(bytes.to_vec()),
            _ => {}
        }
    }
    let request =
        request.ok_or_else(|| ApiError::bad_request("multipart field `request` is required"))?;
    validate_response_format(&request)?;
    let (job_id, _) = submit_job(&state, request, ref_wav, false)?;
    Ok(Json(CreateJobResponse {
        status: "queued".into(),
        progress_url: format!("/v1/qts/audio/jobs/{job_id}"),
        audio_url: format!("/v1/qts/audio/jobs/{job_id}/audio"),
        job_id,
    }))
}

async fn get_job(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<JobInfo>, ApiError> {
    let jobs = state.jobs.lock().expect("job lock poisoned");
    let job = jobs
        .get(&job_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found(format!("job not found: {job_id}")))?;
    Ok(Json(job))
}

async fn get_job_audio(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Response, ApiError> {
    let jobs = state.jobs.lock().expect("job lock poisoned");
    let job = jobs
        .get(&job_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found(format!("job not found: {job_id}")))?;
    if job.status == "failed" {
        return Err(ApiError::bad_request(
            job.error.unwrap_or_else(|| "job failed".into()),
        ));
    }
    let bytes = job
        .audio
        .ok_or_else(|| ApiError::conflict("job is not complete"))?;
    audio_response(AudioPayload {
        bytes,
        sample_rate_hz: job.sample_rate_hz.unwrap_or(qts::SAMPLE_RATE_HZ),
        generated_frames: job.generated_frames,
    })
}

fn submit_job(
    state: &AppState,
    request: SpeechRequest,
    upload_wav: Option<Vec<u8>>,
    sync: bool,
) -> Result<
    (
        String,
        Option<tokio::sync::oneshot::Receiver<Result<AudioPayload, String>>>,
    ),
    ApiError,
> {
    validate_mode_request(state.config.mode.name(), &request)
        .map_err(|err| ApiError::bad_request(err.to_string()))?;
    let job_id = state.next_job_id.fetch_add(1, Ordering::SeqCst).to_string();
    let max_frames = request
        .frames
        .unwrap_or(state.config.request_defaults.max_audio_frames);
    let now = unix_ms();
    state.jobs.lock().expect("job lock poisoned").insert(
        job_id.clone(),
        JobInfo {
            job_id: job_id.clone(),
            status: "queued".into(),
            stage: "queued".into(),
            progress: 0.0,
            generated_frames: 0,
            max_frames,
            sample_rate_hz: None,
            error: None,
            created_at_ms: now,
            started_at_ms: None,
            finished_at_ms: None,
            audio: None,
        },
    );
    let (tx, rx) = if sync {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    let cmd = WorkerCommand {
        job_id: job_id.clone(),
        payload: JobPayload {
            request,
            upload_wav,
        },
        response: tx,
    };
    state
        .command_tx
        .lock()
        .expect("command tx lock poisoned")
        .send(cmd)
        .map_err(|_| ApiError::internal("server worker is not running"))?;
    Ok((job_id, rx))
}

fn run_worker(
    config: ServerConfig,
    jobs: JobStore,
    cmd_rx: mpsc::Receiver<WorkerCommand>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) {
    let engine = match load_engine(&config.model_dir, &config.runtime_backends) {
        Ok(engine) => {
            let _ = ready_tx.send(Ok(()));
            engine
        }
        Err(err) => {
            let _ = ready_tx.send(Err(err.to_string()));
            return;
        }
    };

    while let Ok(cmd) = cmd_rx.recv() {
        mark_running(&jobs, &cmd.job_id);
        let result = synthesize_job(&engine, &config, &jobs, &cmd.job_id, cmd.payload);
        match result {
            Ok(audio) => {
                mark_done(&jobs, &cmd.job_id, &audio);
                if let Some(response) = cmd.response {
                    let _ = response.send(Ok(audio));
                }
            }
            Err(err) => {
                let msg = err.to_string();
                mark_failed(&jobs, &cmd.job_id, msg.clone());
                if let Some(response) = cmd.response {
                    let _ = response.send(Err(msg));
                }
            }
        }
    }
}

fn synthesize_job(
    engine: &Qwen3TtsEngine,
    config: &ServerConfig,
    jobs: &JobStore,
    job_id: &str,
    payload: JobPayload,
) -> Result<AudioPayload> {
    let mut req = build_request(&config.request_defaults, &payload.request)?;
    if payload.request.frames.is_none() {
        req.max_audio_frames =
            segment_max_audio_frames(&req.text, None, config.request_defaults.max_audio_frames);
    }
    let segments = split_long_speech_text(&req.text);
    if segments.len() > 1 {
        return synthesize_segmented_job(engine, config, jobs, job_id, &payload, &req, segments);
    }
    let mut progress_cb = |progress: SynthesisProgress| update_progress(jobs, job_id, progress);
    let result = synthesize_single_segment(engine, config, &payload, &req, &mut progress_cb)?;
    let bytes = encode_wav_f32(result.sample_rate_hz, &result.pcm_f32)?;
    Ok(AudioPayload {
        bytes,
        sample_rate_hz: result.sample_rate_hz,
        generated_frames: result.generated_frames,
    })
}

fn synthesize_segmented_job(
    engine: &Qwen3TtsEngine,
    config: &ServerConfig,
    jobs: &JobStore,
    job_id: &str,
    payload: &JobPayload,
    base_req: &SynthesizeRequest,
    segments: Vec<String>,
) -> Result<AudioPayload> {
    let frame_budgets = segments
        .iter()
        .map(|segment| {
            segment_max_audio_frames(segment, payload.request.frames, base_req.max_audio_frames)
        })
        .collect::<Vec<_>>();
    let total_max_frames = frame_budgets.iter().copied().sum::<usize>().max(1);
    update_progress(
        jobs,
        job_id,
        SynthesisProgress::new(SynthesisProgressStage::Preparing, 0, total_max_frames),
    );

    let mut pcm_f32 = Vec::new();
    let mut sample_rate_hz = None;
    let mut completed_frames = 0usize;
    let mut total_generated_frames = 0usize;
    let mut design_speaker_embedding = None;
    let mut design_prompt = None;

    for (idx, segment) in segments.iter().enumerate() {
        let mut req = base_req.clone();
        req.text = segment.clone();
        req.max_audio_frames = frame_budgets[idx];
        let completed_before_segment = completed_frames;
        let mut progress_cb = |progress: SynthesisProgress| {
            update_segmented_progress(
                jobs,
                job_id,
                progress,
                completed_before_segment,
                total_max_frames,
            )
        };
        let result = synthesize_segment_with_optional_design_embedding(
            engine,
            config,
            payload,
            &req,
            design_prompt.as_ref(),
            design_speaker_embedding.as_deref(),
            &mut progress_cb,
        )
        .map_err(|err| {
            anyhow::anyhow!(
                "failed to synthesize segment {}/{}: {err:#}",
                idx + 1,
                segments.len()
            )
        })?;
        if let Some(sample_rate) = sample_rate_hz {
            if sample_rate != result.sample_rate_hz {
                bail!(
                    "segment {} sample rate changed from {} to {}",
                    idx + 1,
                    sample_rate,
                    result.sample_rate_hz
                );
            }
        } else {
            sample_rate_hz = Some(result.sample_rate_hz);
        }
        completed_frames = completed_frames.saturating_add(req.max_audio_frames);
        total_generated_frames = total_generated_frames.saturating_add(result.generated_frames);
        if idx == 0 && is_design_mode(config) && design_speaker_embedding.is_none() {
            let first_segment_wav = encode_wav_f32(result.sample_rate_hz, &result.pcm_f32)?;
            if let Ok(prompt) = build_icl_voice_clone_prompt(
                engine,
                &config.model_dir,
                "generated:first_segment",
                &first_segment_wav,
                segment,
            ) {
                design_prompt = Some(prompt);
            }
            design_speaker_embedding = Some(engine.encode_reference_speaker(&first_segment_wav)?);
        }
        pcm_f32.extend_from_slice(&result.pcm_f32);
    }

    update_progress(
        jobs,
        job_id,
        SynthesisProgress::new(
            SynthesisProgressStage::Vocoder,
            total_generated_frames,
            total_max_frames,
        ),
    );
    let sample_rate_hz = sample_rate_hz.unwrap_or(qts::SAMPLE_RATE_HZ);
    let bytes = encode_wav_f32(sample_rate_hz, &pcm_f32)?;
    Ok(AudioPayload {
        bytes,
        sample_rate_hz,
        generated_frames: total_generated_frames,
    })
}

fn synthesize_segment_with_optional_design_embedding(
    engine: &Qwen3TtsEngine,
    config: &ServerConfig,
    payload: &JobPayload,
    req: &SynthesizeRequest,
    design_prompt: Option<&VoiceClonePromptV2>,
    design_speaker_embedding: Option<&[f32]>,
    progress_cb: &mut dyn FnMut(SynthesisProgress),
) -> Result<SynthesizeResult> {
    if let (FixedMode::Design { instruct }, Some(speaker_embedding)) =
        (&config.mode, design_speaker_embedding)
    {
        validate_mode_request(config.mode.name(), &payload.request)?;
        let effective_instruct = payload
            .request
            .qts_conditioning
            .as_ref()
            .and_then(|c| c.instruct.as_deref())
            .or(payload.request.instructions.as_deref())
            .or(instruct.as_deref())
            .context("design mode requires `instructions`, `qts_conditioning.instruct`, or startup --instruct")?;
        if let Some(prompt) = design_prompt {
            return Ok(engine.synthesize_with_voice_clone_prompt_progress(
                req,
                prompt,
                progress_cb,
            )?);
        }
        return Ok(
            engine.synthesize_with_voice_design_speaker_embedding_progress(
                req,
                effective_instruct,
                speaker_embedding,
                progress_cb,
            )?,
        );
    }
    synthesize_single_segment(engine, config, payload, req, progress_cb)
}

fn is_design_mode(config: &ServerConfig) -> bool {
    matches!(config.mode, FixedMode::Design { .. })
}

fn synthesize_single_segment(
    engine: &Qwen3TtsEngine,
    config: &ServerConfig,
    payload: &JobPayload,
    req: &SynthesizeRequest,
    progress_cb: &mut dyn FnMut(SynthesisProgress),
) -> Result<SynthesizeResult> {
    let result = match &config.mode {
        FixedMode::None => {
            validate_mode_request(config.mode.name(), &payload.request)?;
            engine.synthesize_with_progress(req, progress_cb)?
        }
        FixedMode::Custom { speaker, instruct } => {
            validate_mode_request(config.mode.name(), &payload.request)?;
            let request_instruct = payload
                .request
                .qts_conditioning
                .as_ref()
                .and_then(|c| c.instruct.as_deref())
                .or(payload.request.instructions.as_deref())
                .or(instruct.as_deref());
            if let Some(request_speaker) = payload
                .request
                .qts_conditioning
                .as_ref()
                .and_then(|c| c.speaker.as_deref())
            {
                if request_speaker != speaker {
                    bail!(
                        "server is fixed to custom speaker '{}', request speaker '{}' is not allowed",
                        speaker,
                        request_speaker
                    );
                }
            }
            engine.synthesize_with_custom_voice_progress(
                req,
                speaker,
                request_instruct,
                progress_cb,
            )?
        }
        FixedMode::Design { instruct } => {
            validate_mode_request(config.mode.name(), &payload.request)?;
            let effective_instruct = payload
                .request
                .qts_conditioning
                .as_ref()
                .and_then(|c| c.instruct.as_deref())
                .or(payload.request.instructions.as_deref())
                .or(instruct.as_deref())
                .context("design mode requires `instructions`, `qts_conditioning.instruct`, or startup --instruct")?;
            engine.synthesize_with_voice_design_progress(req, effective_instruct, progress_cb)?
        }
        FixedMode::Clone { .. } => {
            validate_mode_request(config.mode.name(), &payload.request)?;
            let prompt = resolve_clone_prompt(engine, config, &payload)?;
            engine.synthesize_with_voice_clone_prompt_progress(req, &prompt, progress_cb)?
        }
    };
    Ok(result)
}

fn build_request(defaults: &RequestDefaults, request: &SpeechRequest) -> Result<SynthesizeRequest> {
    let text = request
        .text
        .clone()
        .or_else(|| request.input.clone())
        .context("`input` or `text` is required")?;
    if let Some(speed) = request.speed {
        if (speed - 1.0).abs() > f32::EPSILON {
            bail!("OpenAI `speed` is accepted only as 1.0 in this version");
        }
    }
    Ok(SynthesizeRequest {
        text,
        temperature: request.temperature.unwrap_or(defaults.temperature),
        top_p: request.top_p.unwrap_or(defaults.top_p),
        top_k: request.top_k.unwrap_or(defaults.top_k),
        max_audio_frames: request.frames.unwrap_or(defaults.max_audio_frames),
        thread_count: defaults.thread_count,
        repetition_penalty: request
            .repetition_penalty
            .unwrap_or(defaults.repetition_penalty),
        language_id: request.language_id.unwrap_or(defaults.language_id),
        vocoder_thread_count: request
            .vocoder_threads
            .unwrap_or(defaults.vocoder_thread_count),
        vocoder_chunk_size: request.chunk_size.unwrap_or(defaults.vocoder_chunk_size),
        talker_kv_mode: match &request.talker_kv_mode {
            Some(value) => TalkerKvMode::parse(value)?,
            None => defaults.talker_kv_mode,
        },
    })
}

fn split_long_speech_text(text: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.chars().count() <= LONG_TEXT_SINGLE_PASS_CHARS {
        return vec![trimmed.to_string()];
    }

    let mut segments = Vec::new();
    let mut current = String::new();
    let mut current_chars = 0usize;
    let mut last_soft_boundary = None;

    for ch in trimmed.chars() {
        current.push(ch);
        current_chars += 1;
        if is_soft_speech_boundary(ch) {
            last_soft_boundary = Some(current.len());
        }

        if current_chars >= LONG_TEXT_TARGET_CHARS && is_soft_speech_boundary(ch) {
            push_segment(&mut segments, &mut current);
            current_chars = 0;
            last_soft_boundary = None;
            continue;
        }

        if current_chars >= LONG_TEXT_MAX_CHARS {
            if let Some(boundary) = last_soft_boundary {
                let tail = current.split_off(boundary);
                push_segment_text(&mut segments, &current);
                current = tail;
                current_chars = current.chars().count();
                last_soft_boundary = None;
            } else {
                push_segment(&mut segments, &mut current);
                current_chars = 0;
            }
        }
    }
    push_segment(&mut segments, &mut current);
    merge_short_tail_segment(&mut segments);

    if segments.is_empty() {
        vec![trimmed.to_string()]
    } else {
        segments
    }
}

fn merge_short_tail_segment(segments: &mut Vec<String>) {
    if segments.len() < 2 {
        return;
    }
    let tail_len = segments
        .last()
        .map(|segment| segment.chars().count())
        .unwrap_or_default();
    if tail_len >= LONG_TEXT_MIN_TAIL_CHARS {
        return;
    }
    let tail = segments.pop().unwrap_or_default();
    if let Some(previous) = segments.last_mut() {
        if !previous.ends_with('\n') && !tail.starts_with('\n') {
            previous.push('\n');
        }
        previous.push_str(&tail);
    }
}

fn push_segment(segments: &mut Vec<String>, current: &mut String) {
    push_segment_text(segments, current);
    current.clear();
}

fn push_segment_text(segments: &mut Vec<String>, text: &str) {
    let segment = text.trim();
    if !segment.is_empty() {
        segments.push(segment.to_string());
    }
}

fn is_soft_speech_boundary(ch: char) -> bool {
    matches!(
        ch,
        '。' | '！' | '？' | '；' | '，' | '、' | '.' | '!' | '?' | ';' | ',' | '\n' | '\r'
    )
}

fn segment_max_audio_frames(
    segment: &str,
    requested_frames: Option<usize>,
    default_frames: usize,
) -> usize {
    if let Some(frames) = requested_frames {
        return frames.max(1);
    }
    let char_count = segment.chars().count();
    if char_count <= LONG_TEXT_SHORT_CHARS {
        default_frames.max(256)
    } else if char_count <= LONG_TEXT_TARGET_CHARS {
        default_frames.max(384)
    } else {
        default_frames.max(512)
    }
}

fn validate_mode_request(fixed_mode: &str, request: &SpeechRequest) -> Result<()> {
    if let Some(kind) = request
        .qts_conditioning
        .as_ref()
        .and_then(|c| c.kind.as_deref())
    {
        if kind != fixed_mode {
            bail!("server mode is fixed to `{fixed_mode}`, request conditioning type `{kind}` is not allowed");
        }
    }
    Ok(())
}

fn resolve_clone_prompt(
    engine: &Qwen3TtsEngine,
    config: &ServerConfig,
    payload: &JobPayload,
) -> Result<VoiceClonePromptV2> {
    if let Some(bytes) = &payload.upload_wav {
        let ref_text = payload
            .request
            .qts_conditioning
            .as_ref()
            .and_then(|c| c.ref_text.as_deref());
        return clone_prompt_from_bytes(
            engine,
            &config.model_dir,
            "upload:ref_wav",
            bytes,
            ref_text,
        );
    }

    if let Some(conditioning) = &payload.request.qts_conditioning {
        if let Some(path) = &conditioning.prompt_path {
            let bytes =
                fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
            return Ok(engine.decode_voice_clone_prompt(&bytes)?);
        }
        if let Some(path) = &conditioning.wav_path {
            return clone_prompt_from_path(
                engine,
                &config.model_dir,
                path,
                conditioning.ref_text.as_deref(),
            );
        }
    }

    if let FixedMode::Clone {
        prompt_path,
        wav_path,
        ref_text,
    } = &config.mode
    {
        if let Some(path) = prompt_path {
            let bytes =
                fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
            return Ok(engine.decode_voice_clone_prompt(&bytes)?);
        }
        if let Some(path) = wav_path {
            return clone_prompt_from_path(engine, &config.model_dir, path, ref_text.as_deref());
        }
    }

    bail!("clone mode requires qts_conditioning.prompt_path, qts_conditioning.wav_path, multipart ref_wav, or startup clone source")
}

fn clone_prompt_from_path(
    engine: &Qwen3TtsEngine,
    model_dir: &Path,
    path: &Path,
    ref_text: Option<&str>,
) -> Result<VoiceClonePromptV2> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    clone_prompt_from_bytes(
        engine,
        model_dir,
        path.display().to_string(),
        &bytes,
        ref_text,
    )
}

fn clone_prompt_from_bytes(
    engine: &Qwen3TtsEngine,
    model_dir: &Path,
    source: impl Into<String>,
    bytes: &[u8],
    ref_text: Option<&str>,
) -> Result<VoiceClonePromptV2> {
    if let Some(ref_text) = ref_text.filter(|text| !text.trim().is_empty()) {
        build_icl_voice_clone_prompt(engine, model_dir, source, bytes, ref_text)
    } else {
        build_wav_only_voice_clone_prompt(engine, model_dir, source, bytes)
    }
}

fn validate_response_format(request: &SpeechRequest) -> Result<(), ApiError> {
    if let Some(format) = request.response_format.as_deref() {
        if !matches!(format, "wav" | "pcm") {
            return Err(ApiError::bad_request(
                "only response_format `wav` and `pcm` are accepted; this server currently returns WAV for both",
            ));
        }
    }
    let _ = (&request.model, &request.voice);
    Ok(())
}

fn audio_response(audio: AudioPayload) -> Result<Response, ApiError> {
    let mut response = Response::new(Body::from(audio.bytes));
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
    response.headers_mut().insert(
        "x-qts-sample-rate",
        HeaderValue::from_str(&audio.sample_rate_hz.to_string())
            .map_err(|err| ApiError::internal(err.to_string()))?,
    );
    response.headers_mut().insert(
        "x-qts-generated-frames",
        HeaderValue::from_str(&audio.generated_frames.to_string())
            .map_err(|err| ApiError::internal(err.to_string()))?,
    );
    Ok(response)
}

fn mark_running(jobs: &JobStore, job_id: &str) {
    if let Some(job) = jobs.lock().expect("job lock poisoned").get_mut(job_id) {
        job.status = "running".into();
        job.stage = "preparing".into();
        job.progress = 0.01;
        job.started_at_ms = Some(unix_ms());
    }
}

fn update_progress(jobs: &JobStore, job_id: &str, progress: SynthesisProgress) {
    if let Some(job) = jobs.lock().expect("job lock poisoned").get_mut(job_id) {
        job.status = "running".into();
        job.stage = match progress.stage {
            SynthesisProgressStage::Preparing => "preparing",
            SynthesisProgressStage::Prefill => "prefill",
            SynthesisProgressStage::Rollout => "rollout",
            SynthesisProgressStage::Vocoder => "vocoder",
            SynthesisProgressStage::Done => "done",
        }
        .into();
        job.generated_frames = progress.generated_frames;
        job.max_frames = progress.max_frames;
        job.progress = match progress.stage {
            SynthesisProgressStage::Preparing => 0.02,
            SynthesisProgressStage::Prefill => 0.05,
            SynthesisProgressStage::Rollout => {
                let denom = progress.max_frames.max(1) as f32;
                0.05 + 0.85 * (progress.generated_frames as f32 / denom).clamp(0.0, 1.0)
            }
            SynthesisProgressStage::Vocoder => 0.95,
            SynthesisProgressStage::Done => 1.0,
        };
    }
}

fn update_segmented_progress(
    jobs: &JobStore,
    job_id: &str,
    progress: SynthesisProgress,
    completed_frames: usize,
    total_max_frames: usize,
) {
    let segment_max_frames = progress.max_frames.max(1);
    let local_ratio = match progress.stage {
        SynthesisProgressStage::Preparing => 0.02,
        SynthesisProgressStage::Prefill => 0.05,
        SynthesisProgressStage::Rollout => {
            progress.generated_frames as f32 / segment_max_frames as f32
        }
        SynthesisProgressStage::Vocoder => 0.98,
        SynthesisProgressStage::Done => 1.0,
    }
    .clamp(0.0, 1.0);
    let segment_progress_frames = (local_ratio * segment_max_frames as f32).round() as usize;
    let adjusted_generated_frames = completed_frames
        .saturating_add(segment_progress_frames)
        .min(total_max_frames);
    if let Some(job) = jobs.lock().expect("job lock poisoned").get_mut(job_id) {
        job.status = "running".into();
        job.stage = match progress.stage {
            SynthesisProgressStage::Preparing => "preparing",
            SynthesisProgressStage::Prefill => "prefill",
            SynthesisProgressStage::Rollout => "rollout",
            SynthesisProgressStage::Vocoder => "vocoder",
            SynthesisProgressStage::Done => "rollout",
        }
        .into();
        job.generated_frames = adjusted_generated_frames;
        job.max_frames = total_max_frames;
        let global_ratio = adjusted_generated_frames as f32 / total_max_frames.max(1) as f32;
        job.progress = (0.02 + 0.93 * global_ratio).clamp(job.progress, 0.99);
    }
}

fn mark_done(jobs: &JobStore, job_id: &str, audio: &AudioPayload) {
    if let Some(job) = jobs.lock().expect("job lock poisoned").get_mut(job_id) {
        job.status = "done".into();
        job.stage = "done".into();
        job.progress = 1.0;
        job.generated_frames = audio.generated_frames;
        job.sample_rate_hz = Some(audio.sample_rate_hz);
        job.finished_at_ms = Some(unix_ms());
        job.audio = Some(audio.bytes.clone());
    }
}

fn mark_failed(jobs: &JobStore, job_id: &str, error: String) {
    if let Some(job) = jobs.lock().expect("job lock poisoned").get_mut(job_id) {
        job.status = "failed".into();
        job.stage = "failed".into();
        job.error = Some(error);
        job.finished_at_ms = Some(unix_ms());
    }
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        #[derive(Serialize)]
        struct ErrorBody {
            error: String,
        }
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

fn print_usage() {
    eprintln!(
        "usage:\n  qts_server --mode none|custom|design|clone [--model-dir DIR] [--host 127.0.0.1] [--port 8080] [--speaker NAME] [--instruct TEXT] [--voice-clone-prompt prompt.pb | --voice-clone-wav REF.wav [--voice-clone-ref-text TEXT]] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language-id N] [--vocoder-threads N] [--chunk-size N] [--talker-kv-mode f16|turboquant] [--backend auto|cpu|metal|vulkan] [--backend-fallback LIST] [--vocoder-ep auto|cpu|cuda|directml|nvrtx|tensorrt] [--vocoder-ep-fallback LIST]\n\nserver mode is fixed at startup. Requests may provide mode-specific fields, but cannot switch conditioning type."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_long_speech_text_keeps_short_text_as_one_segment() {
        let segments = split_long_speech_text("你好呀，今天过得怎么样？");
        assert_eq!(segments, vec!["你好呀，今天过得怎么样？"]);
    }

    #[test]
    fn split_long_speech_text_keeps_medium_text_as_one_segment() {
        let text = "新华社北京二月十七日电　新华社奉中国政府之命，于一九七九年二月十七日发布声明如下：\n\n越南当局无视中国方面的一再警告，最近连续出动武装部队，侵犯中国领土，袭击中国边防人员和边境居民，局势急剧恶化，严重威胁我国边疆的和平和安全。中国边防部队在忍无可忍的情况下，被迫奋起还击。";
        let segments = split_long_speech_text(text);
        assert_eq!(segments, vec![text.trim()]);
    }

    #[test]
    fn split_long_speech_text_prefers_sentence_boundaries() {
        let text = "第一段内容，".repeat(50) + "最后一句。";
        let segments = split_long_speech_text(&text);
        assert!(segments.len() > 1);
        assert!(segments
            .iter()
            .all(|segment| segment.chars().count() <= LONG_TEXT_MAX_CHARS));
        assert_eq!(
            segments.concat().replace(char::is_whitespace, ""),
            text.replace(char::is_whitespace, "")
        );
    }

    #[test]
    fn split_long_speech_text_merges_short_tail() {
        let text = "This is a sentence with enough words to create a long segment, ".repeat(8)
            + "short tail.";
        let segments = split_long_speech_text(&text);
        assert!(segments.len() > 1);
        assert!(segments
            .last()
            .map(|segment| segment.chars().count() >= LONG_TEXT_MIN_TAIL_CHARS)
            .unwrap_or(false));
        assert_eq!(
            without_whitespace(&segments.concat()),
            without_whitespace(&text)
        );
    }

    #[test]
    fn split_long_speech_text_hard_splits_when_no_boundary_exists() {
        let text = "长".repeat(LONG_TEXT_MAX_CHARS * 2 + 17);
        let segments = split_long_speech_text(&text);
        assert_eq!(segments.len(), 2);
        assert!(segments.iter().all(
            |segment| segment.chars().count() <= LONG_TEXT_MAX_CHARS + LONG_TEXT_MIN_TAIL_CHARS
        ));
        assert_eq!(
            without_whitespace(&segments.concat()),
            without_whitespace(&text)
        );
    }

    fn without_whitespace(text: &str) -> String {
        text.chars().filter(|ch| !ch.is_whitespace()).collect()
    }

    #[test]
    fn segment_max_audio_frames_uses_request_frames_when_present() {
        assert_eq!(segment_max_audio_frames("短句", Some(123), 256), 123);
        assert_eq!(segment_max_audio_frames("短句", Some(0), 256), 1);
    }

    #[test]
    fn segment_max_audio_frames_scales_for_longer_segments() {
        assert_eq!(segment_max_audio_frames(&"短".repeat(40), None, 256), 256);
        assert_eq!(segment_max_audio_frames(&"中".repeat(90), None, 256), 384);
        assert_eq!(segment_max_audio_frames(&"长".repeat(140), None, 256), 512);
    }

    #[test]
    fn segmented_progress_offsets_rollout_frames() {
        let jobs = Arc::new(Mutex::new(HashMap::new()));
        jobs.lock().unwrap().insert(
            "1".into(),
            JobInfo {
                job_id: "1".into(),
                status: "running".into(),
                stage: "rollout".into(),
                progress: 0.0,
                generated_frames: 0,
                max_frames: 10,
                sample_rate_hz: None,
                error: None,
                created_at_ms: 0,
                started_at_ms: Some(0),
                finished_at_ms: None,
                audio: None,
            },
        );

        update_segmented_progress(&jobs, "1", SynthesisProgress::rollout(25, 100), 200, 500);
        let job = jobs.lock().unwrap().get("1").cloned().unwrap();
        assert_eq!(job.generated_frames, 225);
        assert_eq!(job.max_frames, 500);
        assert!(job.progress > 0.05);
        assert!(job.progress < 0.95);
    }

    #[test]
    fn segmented_progress_does_not_jump_to_global_vocoder_near_done() {
        let jobs = Arc::new(Mutex::new(HashMap::new()));
        jobs.lock().unwrap().insert(
            "1".into(),
            JobInfo {
                job_id: "1".into(),
                status: "running".into(),
                stage: "rollout".into(),
                progress: 0.0,
                generated_frames: 0,
                max_frames: 10,
                sample_rate_hz: None,
                error: None,
                created_at_ms: 0,
                started_at_ms: Some(0),
                finished_at_ms: None,
                audio: None,
            },
        );

        update_segmented_progress(
            &jobs,
            "1",
            SynthesisProgress::new(SynthesisProgressStage::Vocoder, 94, 384),
            0,
            1408,
        );
        let job = jobs.lock().unwrap().get("1").cloned().unwrap();
        assert_eq!(job.stage, "vocoder");
        assert!(job.progress < 0.30);
    }
}
