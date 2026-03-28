//! Wall-clock stage timings for one synthesis pass (tokenizer / transformer / vocoder, etc.).

use std::fmt::Write as _;
use std::time::Duration;

use crate::pipeline::tts_transformer::CodecRolloutSubTimings;

/// Per-stage wall times for a single [`crate::Qwen3TtsEngine::synthesize_with_profile`] call.
///
/// When pipelining is disabled, stages are sequential and their sum approximates
/// end-to-end latency. With pipelining (`vocoder_chunk_size > 0`), `codec_rollout`
/// and `vocoder_decode` overlap; use [`total()`] which accounts for this via
/// `pipeline_overlap`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SynthesisStageTimings {
    /// Reference WAV → speaker embedding (zero if not used).
    pub speaker_encode: Duration,
    /// Text → token ids (includes voice-clone ref+target path when applicable).
    pub tokenize: Duration,
    /// `build_prefill_inputs` (text projection + codec input layout).
    pub prefill_build: Duration,
    /// `rollout_codec_frames_kv` (talker prefill + autoregressive codec steps).
    pub codec_rollout: Duration,
    /// Vocoder decode to PCM.
    pub vocoder_decode: Duration,
    /// Flatten codec rows + optional ICL prefix trim on PCM.
    pub post: Duration,
    /// Time saved by overlapping transformer and vocoder (pipelining).
    /// Zero when pipelining is disabled.
    pub pipeline_overlap: Duration,
    /// Time from synthesis start to first generated codec frame (includes
    /// speaker encode + tokenize + prefill + first frame prediction).
    /// This is the theoretical minimum latency for streaming playback.
    pub first_frame_latency: Duration,
    /// Sub-component breakdown within the `codec_rollout` stage.
    pub codec_rollout_detail: CodecRolloutSubTimings,
    /// Number of PCM samples in the final output (after prefix trim).
    pub generated_samples: usize,
    /// Output sample rate (Hz).
    pub sample_rate_hz: u32,
}

impl SynthesisStageTimings {
    #[must_use]
    pub fn total(&self) -> Duration {
        let sum = self.speaker_encode
            + self.tokenize
            + self.prefill_build
            + self.codec_rollout
            + self.vocoder_decode
            + self.post;
        sum.saturating_sub(self.pipeline_overlap)
    }

    /// Arithmetic mean of each stage (wall clock). Empty slice returns `None`.
    #[must_use]
    pub fn average(samples: &[Self]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let n = samples.len() as u128;
        let avg = |field: fn(&Self) -> Duration| {
            let sum: u128 = samples.iter().map(|s| field(s).as_nanos()).sum();
            let nanos = (sum / n).min(u128::from(u64::MAX)) as u64;
            Duration::from_nanos(nanos)
        };
        let avg_usize = |field: fn(&Self) -> usize| {
            let sum: u128 = samples.iter().map(|s| field(s) as u128).sum();
            (sum / n) as usize
        };
        Some(Self {
            speaker_encode: avg(|s| s.speaker_encode),
            tokenize: avg(|s| s.tokenize),
            prefill_build: avg(|s| s.prefill_build),
            codec_rollout: avg(|s| s.codec_rollout),
            vocoder_decode: avg(|s| s.vocoder_decode),
            post: avg(|s| s.post),
            pipeline_overlap: avg(|s| s.pipeline_overlap),
            first_frame_latency: avg(|s| s.first_frame_latency),
            codec_rollout_detail: CodecRolloutSubTimings {
                talker_prefill: avg(|s| s.codec_rollout_detail.talker_prefill),
                talker_steps: avg(|s| s.codec_rollout_detail.talker_steps),
                code_pred_total: avg(|s| s.codec_rollout_detail.code_pred_total),
                kv_writeback: avg(|s| s.codec_rollout_detail.kv_writeback),
                kv_download: avg(|s| s.codec_rollout_detail.kv_download),
                kv_quantize: avg(|s| s.codec_rollout_detail.kv_quantize),
                kv_upload: avg(|s| s.codec_rollout_detail.kv_upload),
                talker_kv_bytes: avg_usize(|s| s.codec_rollout_detail.talker_kv_bytes),
            },
            generated_samples: avg_usize(|s| s.generated_samples),
            sample_rate_hz: samples[0].sample_rate_hz,
        })
    }

    /// Multi-line table: stage name, milliseconds, % of total.
    #[must_use]
    pub fn format_table(&self) -> String {
        let total = self.total();
        let total_ms = duration_ms(total);
        let mut out = String::new();
        let _ = writeln!(
            out,
            "synthesis stage timings (wall clock, single run){}",
            if total_ms > 0.0 {
                String::new()
            } else {
                " — total 0ms".to_string()
            }
        );
        let rows: [(&str, Duration); 6] = [
            ("speaker_encode", self.speaker_encode),
            ("tokenize", self.tokenize),
            ("prefill_build", self.prefill_build),
            ("codec_rollout (transformer)", self.codec_rollout),
            ("vocoder_decode", self.vocoder_decode),
            ("post (flatten/trim)", self.post),
        ];
        for (name, d) in rows {
            let ms = duration_ms(d);
            let pct = if total.is_zero() {
                0.0
            } else {
                100.0 * ms / total_ms.max(f64::EPSILON)
            };
            let _ = writeln!(out, "  {name:<28} {ms:>10.3} ms  ({pct:>5.1}%)");
        }
        let d = &self.codec_rollout_detail;
        let has_detail = !d.talker_prefill.is_zero()
            || !d.talker_steps.is_zero()
            || !d.code_pred_total.is_zero()
            || !d.kv_writeback.is_zero()
            || !d.kv_download.is_zero()
            || !d.kv_quantize.is_zero()
            || !d.kv_upload.is_zero()
            || d.talker_kv_bytes > 0;
        if has_detail {
            let sub_rows: [(&str, Duration); 7] = [
                ("  talker_prefill", d.talker_prefill),
                ("  talker_steps", d.talker_steps),
                ("  code_pred_total", d.code_pred_total),
                ("  kv_writeback", d.kv_writeback),
                ("  kv_download", d.kv_download),
                ("  kv_quantize", d.kv_quantize),
                ("  kv_upload", d.kv_upload),
            ];
            for (name, sd) in sub_rows {
                let _ = writeln!(out, "    {name:<26} {:>10.3} ms", duration_ms(sd));
            }
            if d.talker_kv_bytes > 0 {
                let mib = d.talker_kv_bytes as f64 / (1024.0 * 1024.0);
                let _ = writeln!(out, "    {:<26} {:>10.3} MiB", "talker_kv_bytes", mib);
            }
        }
        if !self.pipeline_overlap.is_zero() {
            let overlap_ms = duration_ms(self.pipeline_overlap);
            let _ = writeln!(
                out,
                "  {:<28} {:>+10.3} ms  (pipeline overlap)",
                "pipeline_overlap", -overlap_ms
            );
        }
        if !self.first_frame_latency.is_zero() {
            let _ = writeln!(
                out,
                "  {:<28} {:>10.3} ms",
                "first_frame_latency",
                duration_ms(self.first_frame_latency)
            );
        }
        let _ = writeln!(out, "  {:<28} {:>10.3} ms  (100.0%)", "total", total_ms);
        if self.sample_rate_hz > 0 && self.generated_samples > 0 {
            let audio_secs = self.generated_samples as f64 / self.sample_rate_hz as f64;
            let audio_ms = audio_secs * 1_000.0;
            let rtf = if audio_secs > 0.0 {
                total.as_secs_f64() / audio_secs
            } else {
                0.0
            };
            let _ = writeln!(
                out,
                "  {:<28} {:>10.3} ms  (RTF = {:.3}x)",
                "audio duration", audio_ms, rtf
            );
        }
        out
    }
}

fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_table_zero_total() {
        let t = SynthesisStageTimings::default();
        let s = t.format_table();
        assert!(s.contains("speaker_encode"));
        assert!(s.contains("total"));
    }

    #[test]
    fn average_two_samples() {
        let a = SynthesisStageTimings {
            tokenize: Duration::from_millis(2),
            codec_rollout: Duration::from_millis(8),
            ..Default::default()
        };
        let b = SynthesisStageTimings {
            tokenize: Duration::from_millis(4),
            codec_rollout: Duration::from_millis(12),
            ..Default::default()
        };
        let m = SynthesisStageTimings::average(&[a, b]).unwrap();
        assert_eq!(m.tokenize, Duration::from_millis(3));
        assert_eq!(m.codec_rollout, Duration::from_millis(10));
    }
}
