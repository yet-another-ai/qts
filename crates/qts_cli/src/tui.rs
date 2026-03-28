use std::collections::VecDeque;
use std::fs;
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
#[cfg(windows)]
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};

use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, Stream, StreamConfig, SupportedStreamConfig};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use qts::{
    Qwen3TtsEngine, Qwen3TtsError, StreamingSynthesizeResult, SynthesizeRequest, TalkerKvMode,
    VoiceClonePromptV2, SAMPLE_RATE_HZ,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;
#[cfg(windows)]
use windows_sys::Win32::Foundation::{
    DuplicateHandle, DUPLICATE_SAME_ACCESS, HANDLE, INVALID_HANDLE_VALUE,
};
#[cfg(windows)]
use windows_sys::Win32::System::Console::{GetStdHandle, SetStdHandle, STD_ERROR_HANDLE};
#[cfg(windows)]
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use crate::cli_support::{
    default_model_dir, load_engine, parse_value_arg, value_arg, RuntimeBackendOverrides,
};

const MAX_LOG_LINES: usize = 12;
const WARMUP_TEXT: &str = "TTS warmup.";
const WARMUP_MAX_AUDIO_FRAMES: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LanguageOption {
    label: &'static str,
    short: &'static str,
    id: i32,
}

const LANGUAGE_OPTIONS: [LanguageOption; 3] = [
    LanguageOption {
        label: "English",
        short: "en",
        id: 2050,
    },
    LanguageOption {
        label: "Chinese",
        short: "zh",
        id: 2055,
    },
    LanguageOption {
        label: "Japanese",
        short: "ja",
        id: 2058,
    },
];

fn language_option_from_name(name: &str) -> Option<LanguageOption> {
    match name.trim().to_ascii_lowercase().as_str() {
        "en" | "eng" | "english" => Some(LANGUAGE_OPTIONS[0]),
        "zh" | "cn" | "zh-cn" | "chinese" => Some(LANGUAGE_OPTIONS[1]),
        "ja" | "jp" | "japanese" => Some(LANGUAGE_OPTIONS[2]),
        _ => None,
    }
}

fn language_option_from_id(id: i32) -> Option<LanguageOption> {
    LANGUAGE_OPTIONS
        .iter()
        .copied()
        .find(|option| option.id == id)
}

fn format_language(language_id: i32) -> String {
    match language_option_from_id(language_id) {
        Some(option) => format!("lang={}({})", option.short, option.id),
        None => format!("lang=id:{language_id}"),
    }
}

fn next_language_id(language_id: i32) -> i32 {
    match LANGUAGE_OPTIONS
        .iter()
        .position(|option| option.id == language_id)
    {
        Some(idx) => LANGUAGE_OPTIONS[(idx + 1) % LANGUAGE_OPTIONS.len()].id,
        None => LANGUAGE_OPTIONS[0].id,
    }
}

pub(crate) fn run(args: Vec<String>) -> Result<()> {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        print_tui_usage();
        return Ok(());
    }

    let mut config = TuiConfig::parse(args)?;
    let _stderr_guard = StderrSilencer::new()?;
    let engine = load_engine(&config.model_dir, &config.runtime_backends)?;
    let conditioning = load_conditioning(&engine, &config)?;
    let playback = PlaybackStream::new()?;
    let mut terminal = TerminalSession::new()?;
    let mut app = App::new(
        format!(
            "Transformer: {} | Vocoder: {}",
            engine.primary_backend_kind().as_str(),
            engine.vocoder_backend_label()
        ),
        config.describe(),
        conditioning.describe(),
        playback.describe(),
    );
    app.status = "Warming up synthesis pipeline...".to_string();
    terminal.draw(&app)?;

    match warmup_engine(&engine, &conditioning, &config, &playback) {
        Ok(summary) => {
            app.push_log(format!(
                "Warmup complete. first_finalized_audio={:.1}ms synth={:.1}ms",
                summary.first_chunk_latency_ms, summary.synthesis_ms
            ));
            app.status = "Ready.".to_string();
        }
        Err(err) => {
            app.push_log(format!("warmup failed: {err:#}"));
            app.status = "Ready (warmup failed).".to_string();
        }
    }

    app.push_log("Model loaded. Enter a sentence and press Enter to synthesize.");
    if config.vocoder_chunk_size == 0 {
        app.push_log("Chunked vocoder streaming is disabled; playback will start after synthesis.");
    } else {
        app.push_log(format!(
            "Streaming playback enabled with vocoder chunk size {} frame(s).",
            config.vocoder_chunk_size
        ));
    }

    loop {
        terminal.draw(&app)?;
        if !event::poll(Duration::from_millis(100))? {
            continue;
        }

        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Esc => break,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
            KeyCode::Backspace => {
                app.input.pop();
            }
            KeyCode::Enter => {
                let text = app.input.trim().to_string();
                app.input.clear();
                if text.is_empty() {
                    continue;
                }
                if matches!(text.as_str(), ":q" | ":quit" | "exit" | "quit") {
                    break;
                }

                app.push_log(format!("> {text}"));
                app.status = "Synthesizing and streaming playback...".to_string();
                terminal.draw(&app)?;

                match synthesize_and_play(&engine, &conditioning, &config, &playback, &text) {
                    Ok(summary) => {
                        app.status = format!(
                            "Ready. First finalized audio {:.1} ms, synth {:.1} ms, utterance {:.2} s.",
                            summary.first_chunk_latency_ms,
                            summary.synthesis_ms,
                            summary.audio_seconds
                        );
                        app.push_log(summary.describe());
                    }
                    Err(err) => {
                        app.status = "Last synthesis failed.".to_string();
                        app.push_log(format!("error: {err:#}"));
                    }
                }
            }
            KeyCode::F(2) => {
                config.language_id = next_language_id(config.language_id);
                app.config_line = config.describe();
                app.push_log(format!(
                    "Language changed to {}.",
                    format_language(config.language_id)
                ));
            }
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.input.push(ch);
            }
            _ => {}
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct TuiConfig {
    model_dir: PathBuf,
    voice_clone_prompt: Option<PathBuf>,
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
    runtime_backends: RuntimeBackendOverrides,
}

impl TuiConfig {
    fn parse(args: Vec<String>) -> Result<Self> {
        let mut config = Self {
            model_dir: default_model_dir()?,
            voice_clone_prompt: None,
            thread_count: 4,
            max_audio_frames: 256,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            language_id: 2050,
            vocoder_thread_count: 4,
            vocoder_chunk_size: 4,
            talker_kv_mode: TalkerKvMode::F16,
            runtime_backends: RuntimeBackendOverrides::default(),
        };

        let mut idx = 0;
        while idx < args.len() {
            if config.runtime_backends.parse_flag(&args, &mut idx)? {
                continue;
            }
            match args[idx].as_str() {
                "--model-dir" => {
                    config.model_dir = PathBuf::from(value_arg(&args, &mut idx, "--model-dir")?);
                }
                "--voice-clone-prompt" => {
                    config.voice_clone_prompt = Some(PathBuf::from(value_arg(
                        &args,
                        &mut idx,
                        "--voice-clone-prompt",
                    )?));
                }
                "--threads" => {
                    config.thread_count = parse_value_arg(&args, &mut idx, "--threads")?;
                }
                "--frames" => {
                    config.max_audio_frames = parse_value_arg(&args, &mut idx, "--frames")?;
                }
                "--temperature" => {
                    config.temperature = parse_value_arg(&args, &mut idx, "--temperature")?;
                }
                "--top-k" => {
                    config.top_k = parse_value_arg(&args, &mut idx, "--top-k")?;
                }
                "--top-p" => {
                    config.top_p = parse_value_arg(&args, &mut idx, "--top-p")?;
                }
                "--repetition-penalty" => {
                    config.repetition_penalty =
                        parse_value_arg(&args, &mut idx, "--repetition-penalty")?;
                }
                "--language-id" => {
                    config.language_id = parse_value_arg(&args, &mut idx, "--language-id")?;
                }
                "--language" => {
                    let value = value_arg(&args, &mut idx, "--language")?;
                    config.language_id = language_option_from_name(&value)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                            "unsupported value for --language: {value} (expected en, zh, or ja)"
                        )
                        })?
                        .id;
                }
                "--vocoder-threads" => {
                    config.vocoder_thread_count =
                        parse_value_arg(&args, &mut idx, "--vocoder-threads")?;
                }
                "--chunk-size" => {
                    config.vocoder_chunk_size = parse_value_arg(&args, &mut idx, "--chunk-size")?;
                }
                "--talker-kv-mode" => {
                    config.talker_kv_mode =
                        TalkerKvMode::parse(&value_arg(&args, &mut idx, "--talker-kv-mode")?)?;
                }
                other => bail!("unknown tui argument: {other}"),
            }
        }

        Ok(config)
    }

    fn describe(&self) -> String {
        let mut parts = vec![format!(
            "threads={} frames={} chunk_size={} temp={:.2} top_k={} top_p={:.2} {}",
            self.thread_count,
            self.max_audio_frames,
            self.vocoder_chunk_size,
            self.temperature,
            self.top_k,
            self.top_p,
            format_language(self.language_id)
        )];
        parts.push(format!("talker_kv={}", self.talker_kv_mode.as_str()));
        if let Some(runtime) = self.runtime_backends.describe() {
            parts.push(runtime);
        }
        parts.join(" ")
    }

    fn make_request(&self, text: &str) -> SynthesizeRequest {
        SynthesizeRequest {
            text: text.to_string(),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            max_audio_frames: self.max_audio_frames,
            thread_count: self.thread_count,
            repetition_penalty: self.repetition_penalty,
            language_id: self.language_id,
            vocoder_thread_count: self.vocoder_thread_count,
            vocoder_chunk_size: self.vocoder_chunk_size,
            talker_kv_mode: self.talker_kv_mode,
        }
    }
}

enum Conditioning {
    None,
    VoiceClonePrompt(VoiceClonePromptV2),
}

impl Conditioning {
    fn describe(&self) -> String {
        match self {
            Self::None => "Conditioning: none".to_string(),
            Self::VoiceClonePrompt(prompt) => format!(
                "Conditioning: voice-clone prompt{}",
                if prompt.icl_mode { " with ICL" } else { "" }
            ),
        }
    }
}

fn load_conditioning(engine: &Qwen3TtsEngine, config: &TuiConfig) -> Result<Conditioning> {
    if let Some(path) = &config.voice_clone_prompt {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let prompt = engine.decode_voice_clone_prompt(&bytes)?;
        return Ok(Conditioning::VoiceClonePrompt(prompt));
    }
    Ok(Conditioning::None)
}

#[derive(Default)]
struct App {
    input: String,
    logs: VecDeque<String>,
    status: String,
    backend_line: String,
    config_line: String,
    conditioning_line: String,
    playback_line: String,
}

impl App {
    fn new(
        backend_line: String,
        config_line: String,
        conditioning_line: String,
        playback_line: String,
    ) -> Self {
        Self {
            input: String::new(),
            logs: VecDeque::new(),
            status: "Ready.".to_string(),
            backend_line,
            config_line,
            conditioning_line,
            playback_line,
        }
    }

    fn push_log(&mut self, line: impl Into<String>) {
        self.logs.push_back(line.into());
        while self.logs.len() > MAX_LOG_LINES {
            self.logs.pop_front();
        }
    }
}

struct TerminalSession {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalSession {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }

    fn draw(&mut self, app: &App) -> Result<()> {
        self.terminal.draw(|frame| {
            let area = frame.area();
            let sections = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(6),
                    Constraint::Min(8),
                    Constraint::Length(3),
                    Constraint::Length(3),
                ])
                .split(area);

            let header = Paragraph::new(vec![
                Line::from(
                    "Enter to synthesize. F2 cycles language. Esc/Ctrl-C quits. :q also exits.",
                ),
                Line::from(app.backend_line.as_str()),
                Line::from(app.config_line.as_str()),
                Line::from(app.conditioning_line.as_str()),
                Line::from(app.playback_line.as_str()),
            ])
            .block(
                Block::default()
                    .title("Qwen3 TTS TUI")
                    .borders(Borders::ALL),
            );
            frame.render_widget(header, sections[0]);

            let log_lines = if app.logs.is_empty() {
                vec![Line::from("No utterances yet.")]
            } else {
                app.logs
                    .iter()
                    .map(|line| Line::from(line.as_str()))
                    .collect::<Vec<_>>()
            };
            let log_panel = Paragraph::new(log_lines)
                .block(Block::default().title("Session").borders(Borders::ALL))
                .wrap(Wrap { trim: false });
            frame.render_widget(log_panel, sections[1]);

            let status = Paragraph::new(app.status.as_str())
                .style(Style::default().add_modifier(Modifier::BOLD))
                .block(Block::default().title("Status").borders(Borders::ALL))
                .wrap(Wrap { trim: false });
            frame.render_widget(status, sections[2]);

            let input = Paragraph::new(app.input.as_str())
                .block(Block::default().title("Input").borders(Borders::ALL))
                .wrap(Wrap { trim: false });
            frame.render_widget(input, sections[3]);
        })?;
        Ok(())
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
    }
}

#[derive(Default)]
struct PlaybackQueue {
    samples: VecDeque<f32>,
}

struct PlaybackStream {
    stream: Stream,
    queue: Arc<(Mutex<PlaybackQueue>, Condvar)>,
    sample_rate_hz: u32,
    channels: usize,
    device_name: String,
}

impl PlaybackStream {
    fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("failed to open a default output device")?;
        let device_name = device.description()?.name().to_string();
        let supported = pick_output_config(&device)?;
        let stream_config = supported.config();
        let sample_rate_hz = supported.sample_rate();
        let channels = stream_config.channels as usize;
        let queue = Arc::new((Mutex::new(PlaybackQueue::default()), Condvar::new()));
        let stream = build_stream(&device, &supported, Arc::clone(&queue), channels)?;
        stream.play()?;
        Ok(Self {
            stream,
            queue,
            sample_rate_hz,
            channels,
            device_name,
        })
    }

    fn describe(&self) -> String {
        let _ = &self.stream;
        format!(
            "Playback: {} @ {} Hz, {} channel(s)",
            self.device_name, self.sample_rate_hz, self.channels
        )
    }

    fn begin_sink(&self, input_rate_hz: u32) -> StreamingPlaybackSink {
        StreamingPlaybackSink {
            queue: Arc::clone(&self.queue),
            resampler: LinearResampler::new(input_rate_hz, self.sample_rate_hz),
        }
    }

    fn wait_until_empty(&self) -> Result<()> {
        let (lock, cvar) = &*self.queue;
        let mut state = lock
            .lock()
            .map_err(|_| anyhow::anyhow!("playback queue lock poisoned"))?;
        while !state.samples.is_empty() {
            state = cvar
                .wait(state)
                .map_err(|_| anyhow::anyhow!("playback queue wait poisoned"))?;
        }
        Ok(())
    }
}

struct StreamingPlaybackSink {
    queue: Arc<(Mutex<PlaybackQueue>, Condvar)>,
    resampler: LinearResampler,
}

impl StreamingPlaybackSink {
    fn push_chunk(&mut self, pcm_f32: &[f32]) -> Result<(), Qwen3TtsError> {
        let mut samples = self.resampler.push_chunk(pcm_f32);
        if samples.is_empty() {
            return Ok(());
        }
        let (lock, cvar) = &*self.queue;
        let mut state = lock
            .lock()
            .map_err(|_| Qwen3TtsError::InvalidInput("playback queue lock poisoned".into()))?;
        state.samples.extend(samples.drain(..));
        cvar.notify_all();
        Ok(())
    }

    fn finish(&mut self) -> Result<(), Qwen3TtsError> {
        let tail = self.resampler.finish();
        if tail.is_empty() {
            return Ok(());
        }
        let (lock, cvar) = &*self.queue;
        let mut state = lock
            .lock()
            .map_err(|_| Qwen3TtsError::InvalidInput("playback queue lock poisoned".into()))?;
        state.samples.extend(tail);
        cvar.notify_all();
        Ok(())
    }
}

struct MutedPlaybackSink {
    resampler: LinearResampler,
}

impl MutedPlaybackSink {
    fn new(input_rate_hz: u32, output_rate_hz: u32) -> Self {
        Self {
            resampler: LinearResampler::new(input_rate_hz, output_rate_hz),
        }
    }

    fn push_chunk(&mut self, pcm_f32: &[f32]) {
        let _ = self.resampler.push_chunk(pcm_f32);
    }

    fn finish(&mut self) {
        let _ = self.resampler.finish();
    }
}

struct LinearResampler {
    input_rate_hz: u32,
    output_rate_hz: u32,
    source_pos: f64,
    last_sample: Option<f32>,
}

impl LinearResampler {
    fn new(input_rate_hz: u32, output_rate_hz: u32) -> Self {
        Self {
            input_rate_hz,
            output_rate_hz,
            source_pos: 0.0,
            last_sample: None,
        }
    }

    fn push_chunk(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        if self.input_rate_hz == self.output_rate_hz {
            self.last_sample = input.last().copied();
            return input.to_vec();
        }

        let mut source = Vec::with_capacity(input.len() + usize::from(self.last_sample.is_some()));
        if let Some(sample) = self.last_sample {
            source.push(sample);
        }
        source.extend_from_slice(input);
        self.last_sample = source.last().copied();
        if source.len() < 2 {
            return Vec::new();
        }

        let step = self.input_rate_hz as f64 / self.output_rate_hz as f64;
        let limit = (source.len() - 1) as f64;
        let mut output = Vec::new();
        while self.source_pos < limit {
            let idx = self.source_pos.floor() as usize;
            let frac = (self.source_pos - idx as f64) as f32;
            let a = source[idx];
            let b = source[idx + 1];
            output.push(a + (b - a) * frac);
            self.source_pos += step;
        }
        self.source_pos -= limit;
        output
    }

    fn finish(&mut self) -> Vec<f32> {
        self.last_sample
            .take()
            .map_or_else(Vec::new, |sample| vec![sample])
    }
}

fn pick_output_config(device: &cpal::Device) -> Result<SupportedStreamConfig> {
    let desired = SAMPLE_RATE_HZ;
    if let Ok(configs) = device.supported_output_configs() {
        for config in configs {
            if config.min_sample_rate() <= desired && config.max_sample_rate() >= desired {
                return Ok(config.with_sample_rate(desired));
            }
        }
    }
    device
        .default_output_config()
        .context("failed to pick an output stream config")
}

fn build_stream(
    device: &cpal::Device,
    supported: &SupportedStreamConfig,
    queue: Arc<(Mutex<PlaybackQueue>, Condvar)>,
    channels: usize,
) -> Result<Stream> {
    let config: StreamConfig = supported.config();
    let err_fn = |err| eprintln!("cpal stream error: {err}");
    match supported.sample_format() {
        SampleFormat::F32 => build_stream_inner::<f32>(device, &config, queue, channels, err_fn),
        SampleFormat::I16 => build_stream_inner::<i16>(device, &config, queue, channels, err_fn),
        SampleFormat::U16 => build_stream_inner::<u16>(device, &config, queue, channels, err_fn),
        other => bail!("unsupported output sample format: {other:?}"),
    }
}

fn build_stream_inner<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    queue: Arc<(Mutex<PlaybackQueue>, Condvar)>,
    channels: usize,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _| write_output(data, channels, &queue),
        err_fn,
        None,
    )?;
    Ok(stream)
}

fn write_output<T>(output: &mut [T], channels: usize, queue: &Arc<(Mutex<PlaybackQueue>, Condvar)>)
where
    T: SizedSample + FromSample<f32>,
{
    let (lock, cvar) = &**queue;
    let mut state = match lock.lock() {
        Ok(state) => state,
        Err(_) => {
            for slot in output.iter_mut() {
                *slot = T::from_sample(0.0);
            }
            return;
        }
    };

    for frame in output.chunks_mut(channels) {
        let sample = state.samples.pop_front().unwrap_or(0.0);
        let value = T::from_sample(sample);
        for slot in frame.iter_mut() {
            *slot = value;
        }
    }

    if state.samples.is_empty() {
        cvar.notify_all();
    }
}

struct PlaybackSummary {
    first_chunk_latency_ms: f64,
    synthesis_ms: f64,
    audio_seconds: f64,
    generated_frames: usize,
}

impl PlaybackSummary {
    fn describe(&self) -> String {
        format!(
            "first_finalized_audio={:.1}ms synth={:.1}ms audio={:.2}s frames={}",
            self.first_chunk_latency_ms,
            self.synthesis_ms,
            self.audio_seconds,
            self.generated_frames
        )
    }
}

fn synthesize_and_play(
    engine: &Qwen3TtsEngine,
    conditioning: &Conditioning,
    config: &TuiConfig,
    playback: &PlaybackStream,
    text: &str,
) -> Result<PlaybackSummary> {
    let request = config.make_request(text);
    let start = Instant::now();
    let mut first_chunk_latency: Option<Duration> = None;
    let mut sink = playback.begin_sink(SAMPLE_RATE_HZ);

    let mut stream_sink = |pcm_f32: &[f32]| -> Result<(), Qwen3TtsError> {
        if !pcm_f32.is_empty() && first_chunk_latency.is_none() {
            first_chunk_latency = Some(start.elapsed());
        }
        sink.push_chunk(pcm_f32)
    };

    let result: StreamingSynthesizeResult = match conditioning {
        Conditioning::None => engine.synthesize_streaming(&request, &mut stream_sink)?,
        Conditioning::VoiceClonePrompt(prompt) => engine
            .synthesize_with_voice_clone_prompt_streaming(&request, prompt, &mut stream_sink)?,
    };
    sink.finish()?;
    let synthesis_elapsed = start.elapsed();
    playback.wait_until_empty()?;

    Ok(PlaybackSummary {
        first_chunk_latency_ms: first_chunk_latency
            .unwrap_or(synthesis_elapsed)
            .as_secs_f64()
            * 1_000.0,
        synthesis_ms: synthesis_elapsed.as_secs_f64() * 1_000.0,
        audio_seconds: result.generated_samples as f64 / result.sample_rate_hz as f64,
        generated_frames: result.generated_frames,
    })
}

fn warmup_engine(
    engine: &Qwen3TtsEngine,
    conditioning: &Conditioning,
    config: &TuiConfig,
    playback: &PlaybackStream,
) -> Result<PlaybackSummary> {
    let mut request = config.make_request(WARMUP_TEXT);
    request.max_audio_frames = request.max_audio_frames.clamp(1, WARMUP_MAX_AUDIO_FRAMES);

    let start = Instant::now();
    let mut first_chunk_latency: Option<Duration> = None;
    let mut sink = MutedPlaybackSink::new(SAMPLE_RATE_HZ, playback.sample_rate_hz);
    let mut stream_sink = |pcm_f32: &[f32]| -> Result<(), Qwen3TtsError> {
        if !pcm_f32.is_empty() && first_chunk_latency.is_none() {
            first_chunk_latency = Some(start.elapsed());
        }
        sink.push_chunk(pcm_f32);
        Ok(())
    };

    let result: StreamingSynthesizeResult = match conditioning {
        Conditioning::None => engine.synthesize_streaming(&request, &mut stream_sink)?,
        Conditioning::VoiceClonePrompt(prompt) => engine
            .synthesize_with_voice_clone_prompt_streaming(&request, prompt, &mut stream_sink)?,
    };
    sink.finish();
    let synthesis_elapsed = start.elapsed();

    Ok(PlaybackSummary {
        first_chunk_latency_ms: first_chunk_latency
            .unwrap_or(synthesis_elapsed)
            .as_secs_f64()
            * 1_000.0,
        synthesis_ms: synthesis_elapsed.as_secs_f64() * 1_000.0,
        audio_seconds: result.generated_samples as f64 / result.sample_rate_hz as f64,
        generated_frames: result.generated_frames,
    })
}

fn print_tui_usage() {
    eprintln!(
        "qwen3-tts-cli tui — interactive terminal mode with direct cpal playback\n\n\
         usage:\n  tui [--model-dir DIR] [--voice-clone-prompt prompt.pb] [--threads N] [--frames N] [--temperature F] [--top-k N] [--top-p F] [--repetition-penalty F] [--language en|zh|ja | --language-id N] [--vocoder-threads N] [--chunk-size N] [--talker-kv-mode f16|turboquant] [--backend auto|cpu|metal|vulkan] [--backend-fallback LIST] [--vocoder-ep auto|cpu|acl|armnn|azure|cann|coreml|cuda|directml|migraphx|nnapi|nvrtx|onednn|openvino|qnn|rknpu|tensorrt|tvm|vitis|webgpu|xnnpack] [--vocoder-ep-fallback LIST]\n\n\
         CLI flags override environment variables.\n\
         Default transformer auto chain: Apple = metal,vulkan,cpu ; others = vulkan,cpu.\n\
         Default vocoder auto chain: Apple = coreml,cpu ; Windows = cuda,nvrtx,tensorrt,directml,cpu ; Linux/others = cuda,nvrtx,tensorrt,cpu.\n\n\
         The model is loaded once and reused for every input line.\n\
         Press F2 in the TUI to cycle between English, Chinese, and Japanese.\n\
         --chunk-size controls how many codec frames are vocoded per streamed chunk (default 4).\n\
         Set --chunk-size 0 to disable chunked playback and decode after full synthesis.\n\
         Experimental note: --talker-kv-mode turboquant keeps the KV cache on the selected backend, but quantizes on the host before upload."
    );
}

#[cfg(unix)]
struct StderrSilencer {
    saved_stderr: OwnedFd,
}

#[cfg(unix)]
impl StderrSilencer {
    fn new() -> Result<Self> {
        let null = fs::OpenOptions::new()
            .write(true)
            .open("/dev/null")
            .context("failed to open /dev/null for stderr silencing")?;
        let stderr_fd = io::stderr().as_raw_fd();
        let saved = unsafe { libc::dup(stderr_fd) };
        if saved < 0 {
            return Err(anyhow::anyhow!("failed to duplicate stderr fd"));
        }
        if unsafe { libc::dup2(null.as_raw_fd(), stderr_fd) } < 0 {
            unsafe {
                libc::close(saved);
            }
            return Err(anyhow::anyhow!("failed to redirect stderr to /dev/null"));
        }
        let saved_stderr = unsafe { OwnedFd::from_raw_fd(saved) };
        Ok(Self { saved_stderr })
    }
}

#[cfg(unix)]
impl Drop for StderrSilencer {
    fn drop(&mut self) {
        let stderr_fd = io::stderr().as_raw_fd();
        unsafe {
            libc::dup2(self.saved_stderr.as_raw_fd(), stderr_fd);
        }
    }
}

#[cfg(windows)]
struct StderrSilencer {
    saved_stderr: OwnedHandle,
    _null_stderr: fs::File,
}

#[cfg(windows)]
impl StderrSilencer {
    fn new() -> Result<Self> {
        let null = fs::OpenOptions::new()
            .write(true)
            .open("NUL")
            .context("failed to open NUL for stderr silencing")?;

        let current_process = unsafe { GetCurrentProcess() };
        let stderr_handle = unsafe { GetStdHandle(STD_ERROR_HANDLE) };
        if stderr_handle.is_null() || stderr_handle == INVALID_HANDLE_VALUE {
            return Err(anyhow::anyhow!("failed to fetch stderr handle"));
        }

        let mut saved: HANDLE = std::ptr::null_mut();
        let duplicated = unsafe {
            DuplicateHandle(
                current_process,
                stderr_handle,
                current_process,
                &mut saved,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            )
        };
        if duplicated == 0 || saved.is_null() {
            return Err(anyhow::anyhow!("failed to duplicate stderr handle"));
        }

        let redirected = unsafe { SetStdHandle(STD_ERROR_HANDLE, null.as_raw_handle() as _) };
        if redirected == 0 {
            let _ = unsafe { OwnedHandle::from_raw_handle(saved as _) };
            return Err(anyhow::anyhow!("failed to redirect stderr to NUL"));
        }

        let saved_stderr = unsafe { OwnedHandle::from_raw_handle(saved as _) };
        Ok(Self {
            saved_stderr,
            _null_stderr: null,
        })
    }
}

#[cfg(windows)]
impl Drop for StderrSilencer {
    fn drop(&mut self) {
        unsafe {
            SetStdHandle(STD_ERROR_HANDLE, self.saved_stderr.as_raw_handle() as _);
        }
    }
}
