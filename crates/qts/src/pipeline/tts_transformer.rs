//! TTS transformer metadata, weights, and prefill-input construction from GGUF.

use std::cmp::max;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use qts_ggml::sys;
use rand::RngExt;

use super::backend::{
    execute_graph, ggml_soft_max_ext_with_diag_mask_cache, graph_metadata_mem_size, slice_as_bytes,
    slice_as_bytes_mut, BackendKind, BackendSet, OwnedBuffer, TensorDownload, TensorUpload,
};
use crate::model::GgufFile;
use crate::{Qwen3TtsError, TalkerKvMode};

#[derive(Debug, Clone)]
pub struct TtsTransformerConfig {
    pub text_vocab_size: i32,
    pub text_embd_dim: i32,
    pub hidden_size: i32,
    pub n_layers: i32,
    pub n_attention_heads: i32,
    pub n_key_value_heads: i32,
    pub intermediate_size: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub codec_vocab_size: i32,
    pub n_codebooks: i32,
    pub code_pred_layers: i32,
    pub code_pred_vocab_size: i32,
    pub codec_pad_id: i32,
    pub codec_bos_id: i32,
    pub codec_eos_id: i32,
    pub tts_bos_token_id: i32,
    pub tts_eos_token_id: i32,
    pub tts_pad_token_id: i32,
    pub codec_think_id: i32,
    pub codec_nothink_id: i32,
    pub codec_think_bos_id: i32,
    pub codec_think_eos_id: i32,
    pub english_language_id: i32,
}

impl Default for TtsTransformerConfig {
    fn default() -> Self {
        Self {
            text_vocab_size: 151_936,
            text_embd_dim: 2_048,
            hidden_size: 1_024,
            n_layers: 28,
            n_attention_heads: 16,
            n_key_value_heads: 8,
            intermediate_size: 3_072,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            codec_vocab_size: 3_072,
            n_codebooks: 16,
            code_pred_layers: 5,
            code_pred_vocab_size: 2_048,
            codec_pad_id: 2_148,
            codec_bos_id: 2_149,
            codec_eos_id: 2_150,
            tts_bos_token_id: 151_672,
            tts_eos_token_id: 151_673,
            tts_pad_token_id: 151_671,
            codec_think_id: 2_154,
            codec_nothink_id: 2_155,
            codec_think_bos_id: 2_156,
            codec_think_eos_id: 2_157,
            english_language_id: 2_050,
        }
    }
}

pub struct TtsTransformer {
    config: TtsTransformerConfig,
    talker: TalkerWeights,
    code_pred: CodePredWeights,
}

#[derive(Debug, Clone)]
pub struct PreparedPrefillInputs {
    pub prefill_embd: Vec<f32>,
    pub trailing_text_hidden: Vec<f32>,
    pub tts_pad_embed: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct PrefillConditioning<'a> {
    pub text_tokens: &'a [i32],
    pub speaker_embd: Option<&'a [f32]>,
    pub ref_codebook_0: &'a [i32],
    pub language_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct IclPrefillConditioning<'a> {
    pub text_tokens: &'a [i32],
    pub ref_text_tokens: &'a [i32],
    pub speaker_embd: Option<&'a [f32]>,
    pub ref_code_frames: &'a [Vec<i32>],
    pub language_id: i32,
}

#[derive(Debug, Clone)]
pub struct PrefillForwardOutputs {
    pub hidden_states: Vec<f32>,
    pub logits: Vec<f32>,
    pub n_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct SelectedCodecFrame {
    pub codebook_0_token: i32,
    pub codebook_tokens: Vec<i32>,
    pub hidden_state: Vec<f32>,
    pub logits: Vec<f32>,
}

/// Sub-component wall-clock breakdown within a single codec rollout.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodecRolloutSubTimings {
    /// Talker prefill graph execution.
    pub talker_prefill: Duration,
    /// Sum of talker autoregressive step graph executions (excluding KV write-back).
    pub talker_steps: Duration,
    /// Sum of code-predictor forward passes (prefill + per-codebook steps, all frames).
    pub code_pred_total: Duration,
    /// Sum of host-side KV cache write-back (download + quantize + upload).
    pub kv_writeback: Duration,
    /// Sum of host-side KV downloads prior to quantization.
    pub kv_download: Duration,
    /// Sum of host-side quantization time for talker K/V rows.
    pub kv_quantize: Duration,
    /// Sum of cache uploads after host quantization.
    pub kv_upload: Duration,
    /// Total bytes reserved for talker K/V cache storage.
    pub talker_kv_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct CodecRollout {
    pub frames: Vec<SelectedCodecFrame>,
    pub first_frame_elapsed: Duration,
    pub sub_timings: CodecRolloutSubTimings,
}

pub struct VocoderChunk {
    pub codes: Vec<i32>,
    pub n_frames: usize,
}

#[derive(Debug, Clone)]
struct StepForwardOutputs {
    hidden_state: Vec<f32>,
    logits: Vec<f32>,
    kv_writeback_elapsed: Duration,
    kv_download_elapsed: Duration,
    kv_quantize_elapsed: Duration,
    kv_upload_elapsed: Duration,
}

#[derive(Debug, Clone)]
struct CodePredStepOutputs {
    logits: Vec<f32>,
}

struct KvWritebackTensorDownloads {
    layer_idx: usize,
    token_start: usize,
    n_tokens: usize,
    k_tensor: *mut sys::ggml_tensor,
    v_tensor: *mut sys::ggml_tensor,
    k_data: Vec<f32>,
    v_data: Vec<f32>,
}

impl TtsTransformer {
    fn recent_codebook0_tokens_from_frames(frames: &[SelectedCodecFrame]) -> Vec<i32> {
        frames.iter().map(|frame| frame.codebook_0_token).collect()
    }

    fn recent_codebook0_tokens_from_prompt(prompt_frames: &[Vec<i32>]) -> Vec<i32> {
        prompt_frames
            .iter()
            .filter_map(|frame| frame.first().copied())
            .collect()
    }

    pub fn load_from_gguf(file: &GgufFile) -> Result<Self, Qwen3TtsError> {
        let mut cfg = TtsTransformerConfig::default();
        cfg.text_vocab_size = get_u32_any(
            file,
            &["qwen3-tts.text.vocab_size", "qwen3-tts.text_vocab_size"],
            cfg.text_vocab_size,
        );
        cfg.text_embd_dim = get_u32_any(
            file,
            &["qwen3-tts.text.embedding_dim", "qwen3-tts.text_hidden_size"],
            cfg.text_embd_dim,
        );
        cfg.hidden_size = get_u32_any(
            file,
            &[
                "qwen3-tts.talker.embedding_length",
                "qwen3-tts.embedding_length",
            ],
            cfg.hidden_size,
        );
        cfg.n_layers = get_u32_any(
            file,
            &["qwen3-tts.talker.block_count", "qwen3-tts.block_count"],
            cfg.n_layers,
        );
        cfg.n_attention_heads = get_u32_any(
            file,
            &[
                "qwen3-tts.talker.attention.head_count",
                "qwen3-tts.attention.head_count",
            ],
            cfg.n_attention_heads,
        );
        cfg.n_key_value_heads = get_u32_any(
            file,
            &[
                "qwen3-tts.talker.attention.head_count_kv",
                "qwen3-tts.attention.head_count_kv",
            ],
            cfg.n_key_value_heads,
        );
        cfg.intermediate_size = get_u32_any(
            file,
            &[
                "qwen3-tts.talker.feed_forward_length",
                "qwen3-tts.feed_forward_length",
            ],
            cfg.intermediate_size,
        );
        cfg.head_dim = get_u32_any(
            file,
            &[
                "qwen3-tts.talker.attention.key_length",
                "qwen3-tts.attention.key_length",
            ],
            cfg.head_dim,
        );
        cfg.rms_norm_eps = get_f32_any(
            file,
            &[
                "qwen3-tts.talker.attention.layer_norm_rms_epsilon",
                "qwen3-tts.attention.layer_norm_rms_epsilon",
            ],
            cfg.rms_norm_eps,
        );
        cfg.rope_theta = get_f32_any(
            file,
            &[
                "qwen3-tts.talker.rope.freq_base",
                "qwen3-tts.rope.freq_base",
            ],
            cfg.rope_theta,
        );
        cfg.codec_vocab_size = get_u32_any(
            file,
            &["qwen3-tts.talker.codec_vocab_size", "qwen3-tts.vocab_size"],
            cfg.codec_vocab_size,
        );
        cfg.n_codebooks = get_u32_any(
            file,
            &[
                "qwen3-tts.talker.num_codebooks",
                "qwen3-tts.num_code_groups",
            ],
            cfg.n_codebooks,
        );
        cfg.code_pred_layers = get_u32_any(
            file,
            &[
                "qwen3-tts.code_pred.layer_count",
                "qwen3-tts.code_predictor.layer_count",
            ],
            cfg.code_pred_layers,
        );
        cfg.code_pred_vocab_size = get_u32_any(
            file,
            &[
                "qwen3-tts.code_pred.vocab_size",
                "qwen3-tts.code_predictor.vocab_size",
            ],
            cfg.code_pred_vocab_size,
        );
        cfg.codec_pad_id = get_u32_any(file, &["qwen3-tts.codec.pad_id"], cfg.codec_pad_id);
        cfg.codec_bos_id = get_u32_any(file, &["qwen3-tts.codec.bos_id"], cfg.codec_bos_id);
        cfg.codec_eos_id = get_u32_any(
            file,
            &["qwen3-tts.codec.eos_id", "qwen3-tts.codec.eos_token_id"],
            cfg.codec_eos_id,
        );
        cfg.tts_bos_token_id = get_u32_any(
            file,
            &[
                "qwen3-tts.tts_bos_token_id",
                "qwen3-tts.tts.bos_token_id",
                "qwen3-tts.tts.bos_id",
            ],
            cfg.tts_bos_token_id,
        );
        cfg.tts_eos_token_id = get_u32_any(
            file,
            &[
                "qwen3-tts.tts_eos_token_id",
                "qwen3-tts.tts.eos_token_id",
                "qwen3-tts.tts.eos_id",
            ],
            cfg.tts_eos_token_id,
        );
        cfg.tts_pad_token_id = get_u32_any(
            file,
            &[
                "qwen3-tts.tts_pad_token_id",
                "qwen3-tts.tts.pad_token_id",
                "qwen3-tts.tts.pad_id",
            ],
            cfg.tts_pad_token_id,
        );
        cfg.codec_think_id = get_u32_any(
            file,
            &["qwen3-tts.codec.think_id", "qwen3-tts.codec_think_id"],
            cfg.codec_think_id,
        );
        cfg.codec_nothink_id = get_u32_any(
            file,
            &["qwen3-tts.codec.nothink_id", "qwen3-tts.codec_nothink_id"],
            cfg.codec_nothink_id,
        );
        cfg.codec_think_bos_id = get_u32_any(
            file,
            &[
                "qwen3-tts.codec.think_bos_id",
                "qwen3-tts.codec_think_bos_id",
            ],
            cfg.codec_think_bos_id,
        );
        cfg.codec_think_eos_id = get_u32_any(
            file,
            &[
                "qwen3-tts.codec.think_eos_id",
                "qwen3-tts.codec_think_eos_id",
            ],
            cfg.codec_think_eos_id,
        );
        cfg.english_language_id = get_u32_any(
            file,
            &[
                "qwen3-tts.language.english_id",
                "qwen3-tts.codec.language.english_id",
                "qwen3-tts.language_id",
            ],
            cfg.english_language_id,
        );

        let talker_backends = BackendSet::new()?;
        let code_pred_backends = BackendSet::new()?;
        let talker = TalkerWeights::load(file, &cfg, talker_backends)?;
        let code_pred = CodePredWeights::load(file, &cfg, code_pred_backends)?;

        Ok(Self {
            config: cfg,
            talker,
            code_pred,
        })
    }

    #[must_use]
    pub fn config(&self) -> &TtsTransformerConfig {
        &self.config
    }

    #[must_use]
    pub fn primary_backend_kind(&self) -> BackendKind {
        self.talker._backends.primary_kind()
    }

    pub fn build_prefill_inputs(
        &self,
        conditioning: PrefillConditioning<'_>,
        thread_count: usize,
    ) -> Result<PreparedPrefillInputs, Qwen3TtsError> {
        let text_tokens = conditioning.text_tokens;
        let speaker_embd = conditioning.speaker_embd;
        let ref_codebook_0 = conditioning.ref_codebook_0;
        let language_id = conditioning.language_id;
        if text_tokens.len() < 4 {
            return Err(Qwen3TtsError::InvalidInput(
                "need at least 4 text tokens for prefill".into(),
            ));
        }

        let hidden_size = self.config.hidden_size as usize;
        if let Some(speaker_embd) = speaker_embd {
            if speaker_embd.len() != hidden_size {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "speaker embedding must have {hidden_size} elements"
                )));
            }
        }
        for &token in ref_codebook_0 {
            if token < 0 || token >= self.config.codec_vocab_size {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "reference codec token {token} out of range 0..{}",
                    self.config.codec_vocab_size - 1
                )));
            }
        }

        let special_tokens = [
            self.config.tts_bos_token_id,
            self.config.tts_eos_token_id,
            self.config.tts_pad_token_id,
        ];
        let mut projected_text_tokens =
            Vec::with_capacity(special_tokens.len() + text_tokens.len());
        projected_text_tokens.extend_from_slice(&special_tokens);
        projected_text_tokens.extend_from_slice(text_tokens);
        let projected_text = self.project_text_tokens(&projected_text_tokens, thread_count)?;
        let special_proj = &projected_text[..hidden_size * special_tokens.len()];
        let text_proj = &projected_text[hidden_size * special_tokens.len()..];
        let tts_bos_embed = special_proj[0..hidden_size].to_vec();
        let tts_eos_embed = special_proj[hidden_size..hidden_size * 2].to_vec();
        let tts_pad_embed = special_proj[hidden_size * 2..hidden_size * 3].to_vec();
        let role_embed = text_proj[..hidden_size * 3].to_vec();

        let codec_prefill_tokens = if language_id < 0 {
            vec![
                self.config.codec_nothink_id,
                self.config.codec_think_bos_id,
                self.config.codec_think_eos_id,
            ]
        } else {
            vec![
                self.config.codec_think_id,
                self.config.codec_think_bos_id,
                language_id,
                self.config.codec_think_eos_id,
            ]
        };
        let codec_tail_tokens = [self.config.codec_pad_id, self.config.codec_bos_id];
        let mut all_codec_tokens = Vec::with_capacity(
            codec_prefill_tokens.len() + ref_codebook_0.len() + codec_tail_tokens.len(),
        );
        all_codec_tokens.extend_from_slice(&codec_prefill_tokens);
        all_codec_tokens.extend_from_slice(ref_codebook_0);
        all_codec_tokens.extend_from_slice(&codec_tail_tokens);
        let all_codec_embed = self.lookup_codec_embedding_rows(&all_codec_tokens, thread_count)?;
        let codec_prefill_embed_len = codec_prefill_tokens.len() * hidden_size;
        let codec_prompt_embed_len = ref_codebook_0.len() * hidden_size;
        let codec_prefill_embed = &all_codec_embed[..codec_prefill_embed_len];
        let codec_prompt_embed = &all_codec_embed
            [codec_prefill_embed_len..codec_prefill_embed_len + codec_prompt_embed_len];
        let codec_tail_embed = &all_codec_embed[codec_prefill_embed_len + codec_prompt_embed_len..];

        let codec_input_len = codec_prefill_tokens.len()
            + ref_codebook_0.len()
            + usize::from(speaker_embd.is_some())
            + 2;
        let mut codec_input_embedding = vec![0.0f32; codec_input_len * hidden_size];
        codec_input_embedding[..codec_prefill_embed.len()].copy_from_slice(codec_prefill_embed);
        let mut dst_token = codec_prefill_tokens.len();

        if !codec_prompt_embed.is_empty() {
            let dst = &mut codec_input_embedding
                [dst_token * hidden_size..(dst_token + ref_codebook_0.len()) * hidden_size];
            dst.copy_from_slice(codec_prompt_embed);
            dst_token += ref_codebook_0.len();
        }

        if let Some(speaker_embd) = speaker_embd {
            let dst =
                &mut codec_input_embedding[dst_token * hidden_size..(dst_token + 1) * hidden_size];
            dst.copy_from_slice(speaker_embd);
            dst_token += 1;
        }

        codec_input_embedding[dst_token * hidden_size..(dst_token + 2) * hidden_size]
            .copy_from_slice(codec_tail_embed);

        let codec_plus_overlay_len = codec_input_len - 1;
        let mut codec_plus_overlay = vec![0.0f32; codec_plus_overlay_len * hidden_size];
        for token_idx in 0..codec_plus_overlay_len {
            let overlay = if token_idx == codec_plus_overlay_len - 1 {
                &tts_bos_embed
            } else {
                &tts_pad_embed
            };
            let codec_row =
                &codec_input_embedding[token_idx * hidden_size..(token_idx + 1) * hidden_size];
            let out_row =
                &mut codec_plus_overlay[token_idx * hidden_size..(token_idx + 1) * hidden_size];
            for h in 0..hidden_size {
                out_row[h] = overlay[h] + codec_row[h];
            }
        }

        let first_text_embed = &text_proj[3 * hidden_size..4 * hidden_size];
        let codec_bos_embed = &codec_input_embedding
            [(codec_input_len - 1) * hidden_size..codec_input_len * hidden_size];
        let mut first_text_plus_codec_bos = vec![0.0f32; hidden_size];
        for h in 0..hidden_size {
            first_text_plus_codec_bos[h] = first_text_embed[h] + codec_bos_embed[h];
        }

        let prefill_len = 3 + codec_plus_overlay_len + 1;
        let mut prefill_embd = vec![0.0f32; prefill_len * hidden_size];
        prefill_embd[..role_embed.len()].copy_from_slice(&role_embed);
        let codec_offset = 3 * hidden_size;
        prefill_embd[codec_offset..codec_offset + codec_plus_overlay.len()]
            .copy_from_slice(&codec_plus_overlay);
        let final_offset = (prefill_len - 1) * hidden_size;
        prefill_embd[final_offset..final_offset + hidden_size]
            .copy_from_slice(&first_text_plus_codec_bos);

        let trailing_token_count = text_tokens.len().saturating_sub(9);
        let trailing_text_proj = if trailing_token_count > 0 {
            text_proj[4 * hidden_size..(4 + trailing_token_count) * hidden_size].to_vec()
        } else {
            Vec::new()
        };

        let trailing_len = trailing_token_count + 1;
        let mut trailing_text_hidden = vec![0.0f32; trailing_len * hidden_size];
        if !trailing_text_proj.is_empty() {
            trailing_text_hidden[..trailing_text_proj.len()].copy_from_slice(&trailing_text_proj);
        }
        let trailing_eos_offset = (trailing_len - 1) * hidden_size;
        trailing_text_hidden[trailing_eos_offset..trailing_eos_offset + hidden_size]
            .copy_from_slice(&tts_eos_embed);

        Ok(PreparedPrefillInputs {
            prefill_embd,
            trailing_text_hidden,
            tts_pad_embed,
        })
    }

    pub fn build_icl_prefill_inputs(
        &self,
        conditioning: IclPrefillConditioning<'_>,
        thread_count: usize,
    ) -> Result<PreparedPrefillInputs, Qwen3TtsError> {
        let text_tokens = conditioning.text_tokens;
        let ref_text_tokens = conditioning.ref_text_tokens;
        let ref_code_frames = conditioning.ref_code_frames;
        let speaker_embd = conditioning.speaker_embd;
        let language_id = conditioning.language_id;
        if text_tokens.len() < 9 {
            return Err(Qwen3TtsError::InvalidInput(
                "ICL target text prompt must contain at least one text token".into(),
            ));
        }
        if ref_text_tokens.len() < 6 {
            return Err(Qwen3TtsError::InvalidInput(
                "ICL reference text prompt must contain at least one text token".into(),
            ));
        }

        let hidden_size = self.config.hidden_size as usize;
        if let Some(speaker_embd) = speaker_embd {
            if speaker_embd.len() != hidden_size {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "speaker embedding must have {hidden_size} elements"
                )));
            }
        }
        for frame in ref_code_frames {
            self.validate_codebook_frame(frame)?;
        }

        let special_tokens = [
            self.config.tts_bos_token_id,
            self.config.tts_eos_token_id,
            self.config.tts_pad_token_id,
        ];
        let special_proj = self.project_text_tokens(&special_tokens, thread_count)?;
        let tts_bos_embed = special_proj[0..hidden_size].to_vec();
        let tts_eos_embed = special_proj[hidden_size..hidden_size * 2].to_vec();
        let tts_pad_embed = special_proj[hidden_size * 2..hidden_size * 3].to_vec();

        let role_embed = self.project_text_tokens(&text_tokens[..3], thread_count)?;
        let target_text_tokens = &text_tokens[3..text_tokens.len() - 5];
        let ref_text_tokens = &ref_text_tokens[3..ref_text_tokens.len() - 2];
        let mut icl_text_tokens =
            Vec::with_capacity(ref_text_tokens.len() + target_text_tokens.len());
        icl_text_tokens.extend_from_slice(ref_text_tokens);
        icl_text_tokens.extend_from_slice(target_text_tokens);
        let mut text_embed = self.project_text_tokens(&icl_text_tokens, thread_count)?;
        text_embed.extend_from_slice(&tts_eos_embed);

        let codec_prefill_tokens = if language_id < 0 {
            vec![
                self.config.codec_nothink_id,
                self.config.codec_think_bos_id,
                self.config.codec_think_eos_id,
            ]
        } else {
            vec![
                self.config.codec_think_id,
                self.config.codec_think_bos_id,
                language_id,
                self.config.codec_think_eos_id,
            ]
        };
        let codec_tail_tokens = [self.config.codec_pad_id, self.config.codec_bos_id];
        let mut all_codec_tokens =
            Vec::with_capacity(codec_prefill_tokens.len() + codec_tail_tokens.len());
        all_codec_tokens.extend_from_slice(&codec_prefill_tokens);
        all_codec_tokens.extend_from_slice(&codec_tail_tokens);
        let all_codec_embed = self.lookup_codec_embedding_rows(&all_codec_tokens, thread_count)?;
        let codec_prefill_embed_len = codec_prefill_tokens.len() * hidden_size;
        let codec_prefill_embed = &all_codec_embed[..codec_prefill_embed_len];
        let codec_tail_embed = &all_codec_embed[codec_prefill_embed_len..];

        let codec_input_len = codec_prefill_tokens.len() + usize::from(speaker_embd.is_some()) + 2;
        let mut codec_input_embedding = vec![0.0f32; codec_input_len * hidden_size];
        codec_input_embedding[..codec_prefill_embed.len()].copy_from_slice(codec_prefill_embed);
        let mut dst_token = codec_prefill_tokens.len();

        if let Some(speaker_embd) = speaker_embd {
            let dst =
                &mut codec_input_embedding[dst_token * hidden_size..(dst_token + 1) * hidden_size];
            dst.copy_from_slice(speaker_embd);
            dst_token += 1;
        }

        codec_input_embedding[dst_token * hidden_size..(dst_token + 2) * hidden_size]
            .copy_from_slice(codec_tail_embed);

        let codec_plus_overlay_len = codec_input_len - 1;
        let mut codec_plus_overlay = vec![0.0f32; codec_plus_overlay_len * hidden_size];
        for token_idx in 0..codec_plus_overlay_len {
            let overlay = if token_idx == codec_plus_overlay_len - 1 {
                &tts_bos_embed
            } else {
                &tts_pad_embed
            };
            let codec_row =
                &codec_input_embedding[token_idx * hidden_size..(token_idx + 1) * hidden_size];
            let out_row =
                &mut codec_plus_overlay[token_idx * hidden_size..(token_idx + 1) * hidden_size];
            for h in 0..hidden_size {
                out_row[h] = overlay[h] + codec_row[h];
            }
        }

        let codec_bos_embed = &codec_input_embedding
            [(codec_input_len - 1) * hidden_size..codec_input_len * hidden_size];
        let codec_len = ref_code_frames.len() + 1;
        let mut codec_embed = vec![0.0f32; codec_len * hidden_size];
        codec_embed[..hidden_size].copy_from_slice(codec_bos_embed);
        for (frame_idx, frame) in ref_code_frames.iter().enumerate() {
            let row = self.sum_codec_frame_embeddings(frame, thread_count)?;
            let dst = (frame_idx + 1) * hidden_size..(frame_idx + 2) * hidden_size;
            codec_embed[dst].copy_from_slice(&row);
        }

        let text_len = text_embed.len() / hidden_size;
        let mut icl_input_embed = vec![0.0f32; codec_len * hidden_size];
        if text_len > codec_len {
            for row_idx in 0..codec_len {
                let text_row = &text_embed[row_idx * hidden_size..(row_idx + 1) * hidden_size];
                let codec_row = &codec_embed[row_idx * hidden_size..(row_idx + 1) * hidden_size];
                let out_row =
                    &mut icl_input_embed[row_idx * hidden_size..(row_idx + 1) * hidden_size];
                for h in 0..hidden_size {
                    out_row[h] = text_row[h] + codec_row[h];
                }
            }
        } else {
            for row_idx in 0..codec_len {
                let text_row = if row_idx < text_len {
                    &text_embed[row_idx * hidden_size..(row_idx + 1) * hidden_size]
                } else {
                    &tts_pad_embed
                };
                let codec_row = &codec_embed[row_idx * hidden_size..(row_idx + 1) * hidden_size];
                let out_row =
                    &mut icl_input_embed[row_idx * hidden_size..(row_idx + 1) * hidden_size];
                for h in 0..hidden_size {
                    out_row[h] = text_row[h] + codec_row[h];
                }
            }
        }

        let trailing_text_hidden = if text_len > codec_len {
            text_embed[codec_len * hidden_size..].to_vec()
        } else {
            tts_pad_embed.clone()
        };

        let prefill_len = 3 + codec_plus_overlay_len + codec_len;
        let mut prefill_embd = vec![0.0f32; prefill_len * hidden_size];
        prefill_embd[..role_embed.len()].copy_from_slice(&role_embed);
        let codec_offset = 3 * hidden_size;
        prefill_embd[codec_offset..codec_offset + codec_plus_overlay.len()]
            .copy_from_slice(&codec_plus_overlay);
        let icl_offset = codec_offset + codec_plus_overlay.len();
        prefill_embd[icl_offset..icl_offset + icl_input_embed.len()]
            .copy_from_slice(&icl_input_embed);

        Ok(PreparedPrefillInputs {
            prefill_embd,
            trailing_text_hidden,
            tts_pad_embed,
        })
    }

    pub fn forward_prefill(
        &self,
        prefill_embd: &[f32],
        thread_count: usize,
    ) -> Result<PrefillForwardOutputs, Qwen3TtsError> {
        let hidden_size = self.config.hidden_size as usize;
        if prefill_embd.is_empty() || !prefill_embd.len().is_multiple_of(hidden_size) {
            return Err(Qwen3TtsError::InvalidInput(
                "prefill embedding shape is invalid".into(),
            ));
        }

        let n_tokens = prefill_embd.len() / hidden_size;
        let graph_nodes = 4096;
        let ctx = ComputeContext::new_graph(graph_nodes)?;
        let graph = unsafe { sys::ggml_new_graph_custom(ctx.as_ptr(), graph_nodes, false) };
        let graph = NonNull::new(graph).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate prefill graph".into())
        })?;

        let inp_prefill_embd = unsafe {
            sys::ggml_new_tensor_2d(
                ctx.as_ptr(),
                sys::ggml_type_GGML_TYPE_F32,
                hidden_size as i64,
                n_tokens as i64,
            )
        };
        let inp_prefill_embd = NonNull::new(inp_prefill_embd).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate prefill input".into())
        })?;

        let inp_pos = unsafe {
            sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_I32, n_tokens as i64)
        };
        let inp_pos = NonNull::new(inp_pos).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate position input".into())
        })?;
        let positions = (0..n_tokens as i32).collect::<Vec<_>>();

        let mut inp_l = inp_prefill_embd.as_ptr();
        let kq_scale = 1.0f32 / (self.config.head_dim as f32).sqrt();
        let mut attn_softmax_mask: Option<(*mut sys::ggml_tensor, Vec<f32>)> = None;

        for layer in &self.talker.layers {
            let mut cur =
                unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_l, self.config.rms_norm_eps) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, layer.attn_norm.as_ptr()) };

            let mut q_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_q.as_ptr(), cur) };
            let mut k_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_k.as_ptr(), cur) };
            let mut v_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_v.as_ptr(), cur) };

            q_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    q_cur,
                    self.config.head_dim as i64,
                    self.config.n_attention_heads as i64,
                    n_tokens as i64,
                )
            };
            k_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    k_cur,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    n_tokens as i64,
                )
            };
            v_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    v_cur,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    n_tokens as i64,
                )
            };

            if let Some(attn_q_norm) = layer.attn_q_norm {
                q_cur =
                    unsafe { sys::ggml_rms_norm(ctx.as_ptr(), q_cur, self.config.rms_norm_eps) };
                q_cur = unsafe { sys::ggml_mul(ctx.as_ptr(), q_cur, attn_q_norm.as_ptr()) };
            }
            if let Some(attn_k_norm) = layer.attn_k_norm {
                k_cur =
                    unsafe { sys::ggml_rms_norm(ctx.as_ptr(), k_cur, self.config.rms_norm_eps) };
                k_cur = unsafe { sys::ggml_mul(ctx.as_ptr(), k_cur, attn_k_norm.as_ptr()) };
            }

            q_cur = unsafe {
                sys::ggml_rope_ext(
                    ctx.as_ptr(),
                    q_cur,
                    inp_pos.as_ptr(),
                    std::ptr::null_mut(),
                    self.config.head_dim,
                    sys::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    self.config.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            };
            k_cur = unsafe {
                sys::ggml_rope_ext(
                    ctx.as_ptr(),
                    k_cur,
                    inp_pos.as_ptr(),
                    std::ptr::null_mut(),
                    self.config.head_dim,
                    sys::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    self.config.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            };

            let q = unsafe { sys::ggml_permute(ctx.as_ptr(), q_cur, 0, 2, 1, 3) };
            let k = unsafe { sys::ggml_permute(ctx.as_ptr(), k_cur, 0, 2, 1, 3) };
            let mut v = unsafe { sys::ggml_permute(ctx.as_ptr(), v_cur, 0, 2, 1, 3) };

            let mut kq = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), k, q) };
            kq = unsafe { sys::ggml_scale(ctx.as_ptr(), kq, kq_scale) };
            kq = unsafe {
                ggml_soft_max_ext_with_diag_mask_cache(ctx.as_ptr(), kq, 0, &mut attn_softmax_mask)
            };

            v = unsafe { sys::ggml_cont(ctx.as_ptr(), sys::ggml_transpose(ctx.as_ptr(), v)) };

            let mut kqv = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), v, kq) };
            kqv = unsafe { sys::ggml_permute(ctx.as_ptr(), kqv, 0, 2, 1, 3) };
            cur = unsafe {
                sys::ggml_cont_2d(
                    ctx.as_ptr(),
                    kqv,
                    (self.config.n_attention_heads * self.config.head_dim) as i64,
                    n_tokens as i64,
                )
            };

            cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_output.as_ptr(), cur) };
            cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_l) };
            let inp_ff = cur;

            cur = unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_ff, self.config.rms_norm_eps) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, layer.ffn_norm.as_ptr()) };

            let mut gate = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.ffn_gate.as_ptr(), cur) };
            let up = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.ffn_up.as_ptr(), cur) };
            gate = unsafe { sys::ggml_silu(ctx.as_ptr(), gate) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), gate, up) };

            let ffn_down_f32 = layer
                .ffn_down_f32
                .map(NonNull::as_ptr)
                .unwrap_or_else(|| unsafe {
                    sys::ggml_cast(
                        ctx.as_ptr(),
                        layer.ffn_down.as_ptr(),
                        sys::ggml_type_GGML_TYPE_F32,
                    )
                });
            cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), ffn_down_f32, cur) };
            inp_l = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_ff) };
        }

        let mut hidden_states =
            unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_l, self.config.rms_norm_eps) };
        hidden_states = unsafe {
            sys::ggml_mul(
                ctx.as_ptr(),
                hidden_states,
                self.talker.output_norm.as_ptr(),
            )
        };
        hidden_states = unsafe { sys::ggml_cont(ctx.as_ptr(), hidden_states) };
        let mut logits = unsafe {
            sys::ggml_mul_mat(ctx.as_ptr(), self.talker.codec_head.as_ptr(), hidden_states)
        };
        logits = unsafe { sys::ggml_cont(ctx.as_ptr(), logits) };

        unsafe {
            sys::ggml_build_forward_expand(graph.as_ptr(), hidden_states);
            sys::ggml_build_forward_expand(graph.as_ptr(), logits);
        }
        let hidden_elems = hidden_size * n_tokens;
        let logits_elems = self.config.codec_vocab_size as usize * n_tokens;
        let mut hidden_data = vec![0.0f32; hidden_elems];
        let mut logits_data = vec![0.0f32; logits_elems];
        let mut uploads = vec![
            TensorUpload {
                tensor: inp_prefill_embd.as_ptr(),
                bytes: slice_as_bytes(prefill_embd),
            },
            TensorUpload {
                tensor: inp_pos.as_ptr(),
                bytes: slice_as_bytes(positions.as_slice()),
            },
        ];
        if let Some((t, data)) = &attn_softmax_mask {
            uploads.push(TensorUpload {
                tensor: *t,
                bytes: slice_as_bytes(data.as_slice()),
            });
        }
        execute_graph(
            &self.talker._backends,
            graph,
            uploads.as_slice(),
            &mut [
                TensorDownload {
                    tensor: hidden_states,
                    bytes: slice_as_bytes_mut(hidden_data.as_mut_slice()),
                },
                TensorDownload {
                    tensor: logits,
                    bytes: slice_as_bytes_mut(logits_data.as_mut_slice()),
                },
            ],
            thread_count,
            "prefill graph execution failed",
        )?;

        Ok(PrefillForwardOutputs {
            hidden_states: hidden_data,
            logits: logits_data,
            n_tokens,
        })
    }

    pub fn select_codec_frame_from_prefill(
        &self,
        outputs: &PrefillForwardOutputs,
        repetition_penalty: f32,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        recent_tokens: &[i32],
    ) -> Result<SelectedCodecFrame, Qwen3TtsError> {
        let hidden_size = self.config.hidden_size as usize;
        let vocab_size = self.config.codec_vocab_size as usize;
        if outputs.n_tokens == 0
            || outputs.hidden_states.len() != outputs.n_tokens * hidden_size
            || outputs.logits.len() != outputs.n_tokens * vocab_size
        {
            return Err(Qwen3TtsError::InvalidInput(
                "prefill outputs shape is invalid".into(),
            ));
        }

        let hidden_offset = (outputs.n_tokens - 1) * hidden_size;
        let logits_offset = (outputs.n_tokens - 1) * vocab_size;
        let hidden_state =
            outputs.hidden_states[hidden_offset..hidden_offset + hidden_size].to_vec();
        let mut logits = outputs.logits[logits_offset..logits_offset + vocab_size].to_vec();
        let suppress_start = vocab_size.saturating_sub(1024);
        for (token, logit) in logits.iter_mut().enumerate().skip(suppress_start) {
            if token as i32 != self.config.codec_eos_id {
                *logit = f32::NEG_INFINITY;
            }
        }
        let codebook_0_token = select_token(
            &logits,
            repetition_penalty,
            temperature,
            top_k,
            top_p,
            recent_tokens,
        )?;

        Ok(SelectedCodecFrame {
            codebook_0_token,
            codebook_tokens: vec![codebook_0_token],
            hidden_state,
            logits,
        })
    }

    pub fn predict_remaining_codebooks_recompute(
        &self,
        hidden_state: &[f32],
        codebook_0_token: i32,
        thread_count: usize,
        temperature: f32,
        top_k: i32,
        top_p: f32,
    ) -> Result<Vec<i32>, Qwen3TtsError> {
        let hidden_size = self.config.hidden_size as usize;
        if hidden_state.len() != hidden_size {
            return Err(Qwen3TtsError::InvalidInput(
                "code predictor hidden state shape is invalid".into(),
            ));
        }

        let mut codebook_tokens = Vec::with_capacity(self.config.n_codebooks as usize);
        codebook_tokens.push(codebook_0_token);

        while codebook_tokens.len() < self.config.n_codebooks as usize {
            let prev_codes = &codebook_tokens[1..];
            let logits = self.forward_code_pred_sequence_recompute(
                hidden_state,
                codebook_0_token,
                prev_codes,
                thread_count,
            )?;
            let token = select_token(&logits, 1.0, temperature, top_k, top_p, &[])?;
            codebook_tokens.push(token);
        }

        Ok(codebook_tokens)
    }

    pub fn predict_remaining_codebooks_kv(
        &self,
        hidden_state: &[f32],
        codebook_0_token: i32,
        thread_count: usize,
        temperature: f32,
        top_k: i32,
        top_p: f32,
    ) -> Result<Vec<i32>, Qwen3TtsError> {
        let cache = CodePredKvCache::new(
            &self.config,
            self.config.n_codebooks as usize,
            self.code_pred._backends.clone(),
        )?;
        self.predict_remaining_codebooks_kv_with_cache(
            hidden_state,
            codebook_0_token,
            thread_count,
            temperature,
            top_k,
            top_p,
            &cache,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_remaining_codebooks_kv_with_cache(
        &self,
        hidden_state: &[f32],
        codebook_0_token: i32,
        thread_count: usize,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        cache: &CodePredKvCache,
    ) -> Result<Vec<i32>, Qwen3TtsError> {
        let hidden_size = self.config.hidden_size as usize;
        if hidden_state.len() != hidden_size {
            return Err(Qwen3TtsError::InvalidInput(
                "code predictor hidden state shape is invalid".into(),
            ));
        }

        let mut codebook_tokens = Vec::with_capacity(self.config.n_codebooks as usize);
        codebook_tokens.push(codebook_0_token);

        let first = self.forward_code_pred_prefill_cached(
            hidden_state,
            codebook_0_token,
            thread_count,
            cache,
        )?;
        codebook_tokens.push(select_token(
            &first.logits,
            1.0,
            temperature,
            top_k,
            top_p,
            &[],
        )?);

        while codebook_tokens.len() < self.config.n_codebooks as usize {
            let generation_step = codebook_tokens.len() - 1;
            let prev_code = *codebook_tokens.last().ok_or_else(|| {
                Qwen3TtsError::InvalidInput("code predictor lost previous code".into())
            })?;
            let outputs = self.forward_code_pred_step_cached(
                prev_code,
                generation_step + 1,
                generation_step,
                thread_count,
                cache,
            )?;
            codebook_tokens.push(select_token(
                &outputs.logits,
                1.0,
                temperature,
                top_k,
                top_p,
                &[],
            )?);
        }

        Ok(codebook_tokens)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_next_codec_frame_recompute(
        &self,
        history_embd: &[f32],
        step_embd: &[f32],
        thread_count: usize,
        repetition_penalty: f32,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        recent_tokens: &[i32],
    ) -> Result<(Vec<f32>, SelectedCodecFrame), Qwen3TtsError> {
        let hidden_size = self.config.hidden_size as usize;
        if history_embd.is_empty() || !history_embd.len().is_multiple_of(hidden_size) {
            return Err(Qwen3TtsError::InvalidInput(
                "history embedding shape is invalid".into(),
            ));
        }
        if step_embd.len() != hidden_size {
            return Err(Qwen3TtsError::InvalidInput(
                "step embedding shape is invalid".into(),
            ));
        }

        let mut extended_history = Vec::with_capacity(history_embd.len() + step_embd.len());
        extended_history.extend_from_slice(history_embd);
        extended_history.extend_from_slice(step_embd);

        let outputs = self.forward_prefill(&extended_history, thread_count)?;
        let selected = self.select_codec_frame_from_prefill(
            &outputs,
            repetition_penalty,
            temperature,
            top_k,
            top_p,
            recent_tokens,
        )?;
        let codebook_tokens = self.predict_remaining_codebooks_recompute(
            &selected.hidden_state,
            selected.codebook_0_token,
            thread_count,
            temperature,
            top_k,
            top_p,
        )?;

        Ok((
            extended_history,
            SelectedCodecFrame {
                codebook_tokens,
                ..selected
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rollout_codec_frames_recompute(
        &self,
        prefill_embd: &[f32],
        trailing_text_hidden: &[f32],
        tts_pad_embed: &[f32],
        prompt_frames: &[Vec<i32>],
        thread_count: usize,
        max_frames: usize,
        repetition_penalty: f32,
        temperature: f32,
        top_k: i32,
        top_p: f32,
    ) -> Result<CodecRollout, Qwen3TtsError> {
        if max_frames == 0 {
            return Ok(CodecRollout {
                frames: Vec::new(),
                first_frame_elapsed: Duration::ZERO,
                sub_timings: CodecRolloutSubTimings::default(),
            });
        }
        let t_rollout_start = Instant::now();

        let prefill_outputs = self.forward_prefill(prefill_embd, thread_count)?;
        for frame in prompt_frames {
            self.validate_codebook_frame(frame)?;
        }
        let prompt_recent_tokens = Self::recent_codebook0_tokens_from_prompt(prompt_frames);
        let first = self.select_codec_frame_from_prefill(
            &prefill_outputs,
            repetition_penalty,
            temperature,
            top_k,
            top_p,
            &prompt_recent_tokens,
        )?;
        if first.codebook_0_token == self.config.codec_eos_id {
            return Ok(CodecRollout {
                first_frame_elapsed: t_rollout_start.elapsed(),
                frames: prompt_frames
                    .iter()
                    .map(|codebook_tokens| SelectedCodecFrame {
                        codebook_0_token: codebook_tokens[0],
                        codebook_tokens: codebook_tokens.clone(),
                        hidden_state: Vec::new(),
                        logits: Vec::new(),
                    })
                    .collect(),
                sub_timings: CodecRolloutSubTimings::default(),
            });
        }
        let first = SelectedCodecFrame {
            codebook_tokens: self.predict_remaining_codebooks_recompute(
                &first.hidden_state,
                first.codebook_0_token,
                thread_count,
                temperature,
                top_k,
                top_p,
            )?,
            ..first
        };

        let mut frames = prompt_frames
            .iter()
            .map(|codebook_tokens| SelectedCodecFrame {
                codebook_0_token: codebook_tokens[0],
                codebook_tokens: codebook_tokens.clone(),
                hidden_state: Vec::new(),
                logits: Vec::new(),
            })
            .collect::<Vec<_>>();
        frames.push(first);
        let first_frame_elapsed = t_rollout_start.elapsed();
        let mut history_embd = prefill_embd.to_vec();
        let hidden_size = self.config.hidden_size as usize;
        if !trailing_text_hidden.len().is_multiple_of(hidden_size) {
            return Err(Qwen3TtsError::InvalidInput(
                "trailing text hidden shape is invalid".into(),
            ));
        }
        if tts_pad_embed.len() != hidden_size {
            return Err(Qwen3TtsError::InvalidInput(
                "tts pad embedding shape is invalid".into(),
            ));
        }
        let trailing_len = trailing_text_hidden.len() / hidden_size;

        while frames.len() < max_frames {
            let generated_frames = frames.len().saturating_sub(prompt_frames.len());
            let recent_tokens = Self::recent_codebook0_tokens_from_frames(&frames);
            let prev_token = frames
                .last()
                .map(|frame| frame.codebook_0_token)
                .ok_or_else(|| Qwen3TtsError::InvalidInput("rollout lost previous token".into()))?;

            if prev_token == self.config.codec_eos_id {
                break;
            }

            let trailing_idx = generated_frames.saturating_sub(1);
            let trailing_row = if trailing_idx < trailing_len {
                &trailing_text_hidden[trailing_idx * hidden_size..(trailing_idx + 1) * hidden_size]
            } else {
                tts_pad_embed
            };
            let step_embd = self.build_talker_step_embedding(
                &frames
                    .last()
                    .ok_or_else(|| {
                        Qwen3TtsError::InvalidInput("rollout lost previous frame".into())
                    })?
                    .codebook_tokens,
                trailing_row,
                thread_count,
            )?;
            let (next_history, next_frame) = self.generate_next_codec_frame_recompute(
                &history_embd,
                &step_embd,
                thread_count,
                repetition_penalty,
                temperature,
                top_k,
                top_p,
                &recent_tokens,
            )?;
            if next_frame.codebook_0_token == self.config.codec_eos_id {
                break;
            }
            history_embd = next_history;
            frames.push(next_frame);
        }

        Ok(CodecRollout {
            frames,
            first_frame_elapsed,
            sub_timings: CodecRolloutSubTimings::default(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rollout_codec_frames_kv(
        &self,
        prefill_embd: &[f32],
        trailing_text_hidden: &[f32],
        tts_pad_embed: &[f32],
        prompt_frames: &[Vec<i32>],
        talker_kv_mode: TalkerKvMode,
        thread_count: usize,
        max_frames: usize,
        repetition_penalty: f32,
        temperature: f32,
        top_k: i32,
        top_p: f32,
    ) -> Result<CodecRollout, Qwen3TtsError> {
        if max_frames == 0 {
            return Ok(CodecRollout {
                frames: Vec::new(),
                first_frame_elapsed: Duration::ZERO,
                sub_timings: CodecRolloutSubTimings::default(),
            });
        }
        let t_rollout_start = Instant::now();

        let hidden_size = self.config.hidden_size as usize;
        if prefill_embd.is_empty() || !prefill_embd.len().is_multiple_of(hidden_size) {
            return Err(Qwen3TtsError::InvalidInput(
                "prefill embedding shape is invalid".into(),
            ));
        }
        if !trailing_text_hidden.len().is_multiple_of(hidden_size) {
            return Err(Qwen3TtsError::InvalidInput(
                "trailing text hidden shape is invalid".into(),
            ));
        }
        if tts_pad_embed.len() != hidden_size {
            return Err(Qwen3TtsError::InvalidInput(
                "tts pad embedding shape is invalid".into(),
            ));
        }
        for frame in prompt_frames {
            self.validate_codebook_frame(frame)?;
        }

        let prefill_len = prefill_embd.len() / hidden_size;
        let trailing_len = trailing_text_hidden.len() / hidden_size;
        let required_ctx = max(256, prefill_len + max_frames + 16);
        let cache = TalkerKvCache::new(
            &self.config,
            required_ctx,
            self.talker._backends.clone(),
            talker_kv_mode,
        )?;
        let code_pred_cache = CodePredKvCache::new(
            &self.config,
            self.config.n_codebooks as usize,
            self.code_pred._backends.clone(),
        )?;

        let t_prefill = Instant::now();
        let first_outputs = self.forward_prefill_cached(prefill_embd, thread_count, &cache)?;
        let talker_prefill_dur = t_prefill.elapsed();

        let mut logits = first_outputs.logits;
        let suppress_start = self.config.codec_vocab_size.saturating_sub(1024) as usize;
        for (token, logit) in logits.iter_mut().enumerate().skip(suppress_start) {
            if token as i32 != self.config.codec_eos_id {
                *logit = f32::NEG_INFINITY;
            }
        }
        let prompt_recent_tokens = Self::recent_codebook0_tokens_from_prompt(prompt_frames);
        let first_codebook_0 = select_token(
            &logits,
            repetition_penalty,
            temperature,
            top_k,
            top_p,
            &prompt_recent_tokens,
        )?;
        if first_codebook_0 == self.config.codec_eos_id {
            return Ok(CodecRollout {
                first_frame_elapsed: t_rollout_start.elapsed(),
                frames: prompt_frames
                    .iter()
                    .map(|codebook_tokens| SelectedCodecFrame {
                        codebook_0_token: codebook_tokens[0],
                        codebook_tokens: codebook_tokens.clone(),
                        hidden_state: Vec::new(),
                        logits: Vec::new(),
                    })
                    .collect(),
                sub_timings: CodecRolloutSubTimings {
                    talker_prefill: talker_prefill_dur,
                    talker_kv_bytes: cache.total_bytes(),
                    ..Default::default()
                },
            });
        }

        let mut code_pred_dur = Duration::ZERO;
        let t_cp = Instant::now();
        let first_codes = self.predict_remaining_codebooks_kv_with_cache(
            &first_outputs.hidden_state,
            first_codebook_0,
            thread_count,
            temperature,
            top_k,
            top_p,
            &code_pred_cache,
        )?;
        code_pred_dur += t_cp.elapsed();

        let mut frames = prompt_frames
            .iter()
            .map(|codebook_tokens| SelectedCodecFrame {
                codebook_0_token: codebook_tokens[0],
                codebook_tokens: codebook_tokens.clone(),
                hidden_state: Vec::new(),
                logits: Vec::new(),
            })
            .collect::<Vec<_>>();
        frames.push(SelectedCodecFrame {
            codebook_0_token: first_codebook_0,
            codebook_tokens: first_codes,
            hidden_state: first_outputs.hidden_state,
            logits,
        });
        let first_frame_elapsed = t_rollout_start.elapsed();
        let mut n_past = prefill_len;

        let mut talker_steps_dur = Duration::ZERO;
        let mut kv_writeback_dur = Duration::ZERO;
        let mut kv_download_dur = Duration::ZERO;
        let mut kv_quantize_dur = Duration::ZERO;
        let mut kv_upload_dur = Duration::ZERO;

        while frames.len() < max_frames {
            let generated_frames = frames.len().saturating_sub(prompt_frames.len());
            let recent_tokens = Self::recent_codebook0_tokens_from_frames(&frames);
            let trailing_idx = generated_frames.saturating_sub(1);
            let trailing_row = if trailing_idx < trailing_len {
                &trailing_text_hidden[trailing_idx * hidden_size..(trailing_idx + 1) * hidden_size]
            } else {
                tts_pad_embed
            };
            let step_embd = self.build_talker_step_embedding(
                &frames
                    .last()
                    .ok_or_else(|| {
                        Qwen3TtsError::InvalidInput("rollout lost previous frame".into())
                    })?
                    .codebook_tokens,
                trailing_row,
                thread_count,
            )?;
            let t_step = Instant::now();
            let step_outputs =
                self.forward_step_cached(&step_embd, n_past, thread_count, &cache)?;
            let step_elapsed = t_step.elapsed();
            kv_writeback_dur += step_outputs.kv_writeback_elapsed;
            kv_download_dur += step_outputs.kv_download_elapsed;
            kv_quantize_dur += step_outputs.kv_quantize_elapsed;
            kv_upload_dur += step_outputs.kv_upload_elapsed;
            talker_steps_dur += step_elapsed.saturating_sub(step_outputs.kv_writeback_elapsed);

            let mut logits = step_outputs.logits;
            for (token, logit) in logits.iter_mut().enumerate().skip(suppress_start) {
                if token as i32 != self.config.codec_eos_id {
                    *logit = f32::NEG_INFINITY;
                }
            }
            let codebook_0_token = select_token(
                &logits,
                repetition_penalty,
                temperature,
                top_k,
                top_p,
                &recent_tokens,
            )?;
            if codebook_0_token == self.config.codec_eos_id {
                break;
            }
            let t_cp = Instant::now();
            let codebook_tokens = self.predict_remaining_codebooks_kv_with_cache(
                &step_outputs.hidden_state,
                codebook_0_token,
                thread_count,
                temperature,
                top_k,
                top_p,
                &code_pred_cache,
            )?;
            code_pred_dur += t_cp.elapsed();

            frames.push(SelectedCodecFrame {
                codebook_0_token,
                codebook_tokens,
                hidden_state: step_outputs.hidden_state,
                logits,
            });
            n_past += 1;
        }

        Ok(CodecRollout {
            frames,
            first_frame_elapsed,
            sub_timings: CodecRolloutSubTimings {
                talker_prefill: talker_prefill_dur,
                talker_steps: talker_steps_dur,
                code_pred_total: code_pred_dur,
                kv_writeback: kv_writeback_dur,
                kv_download: kv_download_dur,
                kv_quantize: kv_quantize_dur,
                kv_upload: kv_upload_dur,
                talker_kv_bytes: cache.total_bytes(),
            },
        })
    }

    /// Streaming variant that sends `VocoderChunk`s to `chunk_tx` every
    /// `chunk_size` generated frames so the vocoder can decode in parallel.
    #[allow(clippy::too_many_arguments)]
    pub fn rollout_codec_frames_kv_streaming(
        &self,
        prefill_embd: &[f32],
        trailing_text_hidden: &[f32],
        tts_pad_embed: &[f32],
        prompt_frames: &[Vec<i32>],
        talker_kv_mode: TalkerKvMode,
        thread_count: usize,
        max_frames: usize,
        repetition_penalty: f32,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        chunk_size: usize,
        chunk_tx: &std::sync::mpsc::SyncSender<VocoderChunk>,
    ) -> Result<CodecRollout, Qwen3TtsError> {
        if max_frames == 0 {
            return Ok(CodecRollout {
                frames: Vec::new(),
                first_frame_elapsed: Duration::ZERO,
                sub_timings: CodecRolloutSubTimings::default(),
            });
        }
        let t_rollout_start = Instant::now();

        let hidden_size = self.config.hidden_size as usize;
        if prefill_embd.is_empty() || !prefill_embd.len().is_multiple_of(hidden_size) {
            return Err(Qwen3TtsError::InvalidInput(
                "prefill embedding shape is invalid".into(),
            ));
        }
        if !trailing_text_hidden.len().is_multiple_of(hidden_size) {
            return Err(Qwen3TtsError::InvalidInput(
                "trailing text hidden shape is invalid".into(),
            ));
        }
        if tts_pad_embed.len() != hidden_size {
            return Err(Qwen3TtsError::InvalidInput(
                "tts pad embedding shape is invalid".into(),
            ));
        }
        for frame in prompt_frames {
            self.validate_codebook_frame(frame)?;
        }

        let prefill_len = prefill_embd.len() / hidden_size;
        let trailing_len = trailing_text_hidden.len() / hidden_size;
        let required_ctx = max(256, prefill_len + max_frames + 16);
        let cache = TalkerKvCache::new(
            &self.config,
            required_ctx,
            self.talker._backends.clone(),
            talker_kv_mode,
        )?;
        let code_pred_cache = CodePredKvCache::new(
            &self.config,
            self.config.n_codebooks as usize,
            self.code_pred._backends.clone(),
        )?;

        let t_prefill = Instant::now();
        let first_outputs = self.forward_prefill_cached(prefill_embd, thread_count, &cache)?;
        let talker_prefill_dur = t_prefill.elapsed();

        let mut logits = first_outputs.logits;
        let suppress_start = self.config.codec_vocab_size.saturating_sub(1024) as usize;
        for (token, logit) in logits.iter_mut().enumerate().skip(suppress_start) {
            if token as i32 != self.config.codec_eos_id {
                *logit = f32::NEG_INFINITY;
            }
        }
        let prompt_recent_tokens = Self::recent_codebook0_tokens_from_prompt(prompt_frames);
        let first_codebook_0 = select_token(
            &logits,
            repetition_penalty,
            temperature,
            top_k,
            top_p,
            &prompt_recent_tokens,
        )?;
        if first_codebook_0 == self.config.codec_eos_id {
            return Ok(CodecRollout {
                first_frame_elapsed: t_rollout_start.elapsed(),
                frames: prompt_frames
                    .iter()
                    .map(|codebook_tokens| SelectedCodecFrame {
                        codebook_0_token: codebook_tokens[0],
                        codebook_tokens: codebook_tokens.clone(),
                        hidden_state: Vec::new(),
                        logits: Vec::new(),
                    })
                    .collect(),
                sub_timings: CodecRolloutSubTimings {
                    talker_prefill: talker_prefill_dur,
                    talker_kv_bytes: cache.total_bytes(),
                    ..Default::default()
                },
            });
        }

        let mut code_pred_dur = Duration::ZERO;
        let t_cp = Instant::now();
        let first_codes = self.predict_remaining_codebooks_kv_with_cache(
            &first_outputs.hidden_state,
            first_codebook_0,
            thread_count,
            temperature,
            top_k,
            top_p,
            &code_pred_cache,
        )?;
        code_pred_dur += t_cp.elapsed();

        let mut frames = prompt_frames
            .iter()
            .map(|codebook_tokens| SelectedCodecFrame {
                codebook_0_token: codebook_tokens[0],
                codebook_tokens: codebook_tokens.clone(),
                hidden_state: Vec::new(),
                logits: Vec::new(),
            })
            .collect::<Vec<_>>();
        frames.push(SelectedCodecFrame {
            codebook_0_token: first_codebook_0,
            codebook_tokens: first_codes,
            hidden_state: first_outputs.hidden_state,
            logits,
        });
        let first_frame_elapsed = t_rollout_start.elapsed();
        let mut n_past = prefill_len;

        let prefix_frame_count = prompt_frames.len();
        let chunk_size = chunk_size.max(1);
        let mut unsent_start = prefix_frame_count;

        let flush_chunk =
            |frames: &[SelectedCodecFrame],
             start: usize,
             end: usize,
             tx: &std::sync::mpsc::SyncSender<VocoderChunk>| {
                if end <= start {
                    return;
                }
                let codes = frames[start..end]
                    .iter()
                    .flat_map(|f| f.codebook_tokens.iter().copied())
                    .collect::<Vec<_>>();
                let _ = tx.send(VocoderChunk {
                    codes,
                    n_frames: end - start,
                });
            };

        let mut talker_steps_dur = Duration::ZERO;
        let mut kv_writeback_dur = Duration::ZERO;
        let mut kv_download_dur = Duration::ZERO;
        let mut kv_quantize_dur = Duration::ZERO;
        let mut kv_upload_dur = Duration::ZERO;

        while frames.len() < max_frames {
            let generated_frames = frames.len().saturating_sub(prompt_frames.len());
            let recent_tokens = Self::recent_codebook0_tokens_from_frames(&frames);
            let trailing_idx = generated_frames.saturating_sub(1);
            let trailing_row = if trailing_idx < trailing_len {
                &trailing_text_hidden[trailing_idx * hidden_size..(trailing_idx + 1) * hidden_size]
            } else {
                tts_pad_embed
            };
            let step_embd = self.build_talker_step_embedding(
                &frames
                    .last()
                    .ok_or_else(|| {
                        Qwen3TtsError::InvalidInput("rollout lost previous frame".into())
                    })?
                    .codebook_tokens,
                trailing_row,
                thread_count,
            )?;
            let t_step = Instant::now();
            let step_outputs =
                self.forward_step_cached(&step_embd, n_past, thread_count, &cache)?;
            let step_elapsed = t_step.elapsed();
            kv_writeback_dur += step_outputs.kv_writeback_elapsed;
            kv_download_dur += step_outputs.kv_download_elapsed;
            kv_quantize_dur += step_outputs.kv_quantize_elapsed;
            kv_upload_dur += step_outputs.kv_upload_elapsed;
            talker_steps_dur += step_elapsed.saturating_sub(step_outputs.kv_writeback_elapsed);

            let mut logits = step_outputs.logits;
            for (token, logit) in logits.iter_mut().enumerate().skip(suppress_start) {
                if token as i32 != self.config.codec_eos_id {
                    *logit = f32::NEG_INFINITY;
                }
            }
            let codebook_0_token = select_token(
                &logits,
                repetition_penalty,
                temperature,
                top_k,
                top_p,
                &recent_tokens,
            )?;
            if codebook_0_token == self.config.codec_eos_id {
                break;
            }
            let t_cp = Instant::now();
            let codebook_tokens = self.predict_remaining_codebooks_kv_with_cache(
                &step_outputs.hidden_state,
                codebook_0_token,
                thread_count,
                temperature,
                top_k,
                top_p,
                &code_pred_cache,
            )?;
            code_pred_dur += t_cp.elapsed();

            frames.push(SelectedCodecFrame {
                codebook_0_token,
                codebook_tokens,
                hidden_state: step_outputs.hidden_state,
                logits,
            });
            n_past += 1;

            let generated_since_last = frames.len() - unsent_start;
            if generated_since_last >= chunk_size {
                flush_chunk(&frames, unsent_start, frames.len(), chunk_tx);
                unsent_start = frames.len();
            }
        }

        if frames.len() > unsent_start {
            flush_chunk(&frames, unsent_start, frames.len(), chunk_tx);
        }

        Ok(CodecRollout {
            frames,
            first_frame_elapsed,
            sub_timings: CodecRolloutSubTimings {
                talker_prefill: talker_prefill_dur,
                talker_steps: talker_steps_dur,
                code_pred_total: code_pred_dur,
                kv_writeback: kv_writeback_dur,
                kv_download: kv_download_dur,
                kv_quantize: kv_quantize_dur,
                kv_upload: kv_upload_dur,
                talker_kv_bytes: cache.total_bytes(),
            },
        })
    }

    fn validate_codebook_frame(&self, codebook_tokens: &[i32]) -> Result<(), Qwen3TtsError> {
        if codebook_tokens.len() != self.config.n_codebooks as usize {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "reference codec frame must contain {} codebooks",
                self.config.n_codebooks
            )));
        }
        for (idx, &token) in codebook_tokens.iter().enumerate() {
            let vocab_size = if idx == 0 {
                self.config.codec_vocab_size
            } else {
                self.config.code_pred_vocab_size
            };
            if token < 0 || token >= vocab_size {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "reference codec token {token} at codebook {idx} out of range 0..{}",
                    vocab_size - 1
                )));
            }
        }
        Ok(())
    }

    fn project_text_tokens(
        &self,
        text_tokens: &[i32],
        thread_count: usize,
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        if text_tokens.is_empty() {
            return Ok(Vec::new());
        }

        let graph_nodes = 16;
        let ctx = ComputeContext::new_graph(graph_nodes)?;
        let graph = unsafe { sys::ggml_new_graph_custom(ctx.as_ptr(), graph_nodes, false) };
        let graph = NonNull::new(graph).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate text projection graph".into())
        })?;

        let inp_tokens = unsafe {
            sys::ggml_new_tensor_1d(
                ctx.as_ptr(),
                sys::ggml_type_GGML_TYPE_I32,
                text_tokens.len() as i64,
            )
        };
        let inp_tokens = NonNull::new(inp_tokens).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate text token input".into())
        })?;
        let mut cur = unsafe {
            sys::ggml_get_rows(
                ctx.as_ptr(),
                self.talker.text_embd.as_ptr(),
                inp_tokens.as_ptr(),
            )
        };
        cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), self.talker.text_proj_fc1.as_ptr(), cur) };
        cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, self.talker.text_proj_fc1_bias.as_ptr()) };
        cur = unsafe { sys::ggml_silu(ctx.as_ptr(), cur) };
        cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), self.talker.text_proj_fc2.as_ptr(), cur) };
        cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, self.talker.text_proj_fc2_bias.as_ptr()) };
        cur = unsafe { sys::ggml_cast(ctx.as_ptr(), cur, sys::ggml_type_GGML_TYPE_F32) };

        unsafe {
            sys::ggml_build_forward_expand(graph.as_ptr(), cur);
        }
        let elem_count = self.config.hidden_size as usize * text_tokens.len();
        let mut data = vec![0.0f32; elem_count];
        execute_graph(
            &self.talker._backends,
            graph,
            &[TensorUpload {
                tensor: inp_tokens.as_ptr(),
                bytes: slice_as_bytes(text_tokens),
            }],
            &mut [TensorDownload {
                tensor: cur,
                bytes: slice_as_bytes_mut(data.as_mut_slice()),
            }],
            thread_count,
            "text projection graph execution failed",
        )?;
        Ok(data)
    }

    fn lookup_codec_embedding_rows(
        &self,
        token_ids: &[i32],
        thread_count: usize,
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        for &token_id in token_ids {
            if token_id < 0 || token_id >= self.config.codec_vocab_size {
                return Err(Qwen3TtsError::InvalidInput(format!(
                    "codec token {token_id} out of range 0..{}",
                    self.config.codec_vocab_size - 1
                )));
            }
        }

        let hidden_size = self.config.hidden_size as usize;
        if let Some(codec_table) = self.talker.codec_embd_cpu.as_ref() {
            let mut data = vec![0.0f32; hidden_size * token_ids.len()];
            for (row_idx, &token_id) in token_ids.iter().enumerate() {
                let token = token_id as usize;
                let src = token * hidden_size..(token + 1) * hidden_size;
                let dst = row_idx * hidden_size..(row_idx + 1) * hidden_size;
                if src.end > codec_table.len() {
                    return Err(Qwen3TtsError::InvalidInput(
                        "codec embedding CPU table shape is invalid".into(),
                    ));
                }
                data[dst].copy_from_slice(&codec_table[src]);
            }
            return Ok(data);
        }

        let graph_nodes = 8;
        let ctx = ComputeContext::new_graph(graph_nodes)?;
        let graph = unsafe { sys::ggml_new_graph_custom(ctx.as_ptr(), graph_nodes, false) };
        let graph = NonNull::new(graph).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate embedding graph".into())
        })?;

        let inp_tokens = unsafe {
            sys::ggml_new_tensor_1d(
                ctx.as_ptr(),
                sys::ggml_type_GGML_TYPE_I32,
                token_ids.len() as i64,
            )
        };
        let inp_tokens = NonNull::new(inp_tokens).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate codec token input".into())
        })?;
        let mut rows = unsafe {
            sys::ggml_get_rows(
                ctx.as_ptr(),
                self.talker.codec_embd.as_ptr(),
                inp_tokens.as_ptr(),
            )
        };
        rows = unsafe { sys::ggml_cast(ctx.as_ptr(), rows, sys::ggml_type_GGML_TYPE_F32) };
        unsafe {
            sys::ggml_build_forward_expand(graph.as_ptr(), rows);
        }
        let elem_count = hidden_size * token_ids.len();
        let mut data = vec![0.0f32; elem_count];
        execute_graph(
            &self.talker._backends,
            graph,
            &[TensorUpload {
                tensor: inp_tokens.as_ptr(),
                bytes: slice_as_bytes(token_ids),
            }],
            &mut [TensorDownload {
                tensor: rows,
                bytes: slice_as_bytes_mut(data.as_mut_slice()),
            }],
            thread_count,
            "codec embedding graph execution failed",
        )?;
        Ok(data)
    }

    fn sum_codec_frame_embeddings(
        &self,
        codebook_tokens: &[i32],
        thread_count: usize,
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        self.validate_codebook_frame(codebook_tokens)?;
        let hidden_size = self.config.hidden_size as usize;

        if let (Some(codec_table), Some(pred_tables)) = (
            self.talker.codec_embd_cpu.as_ref(),
            self.code_pred.codec_embd_cpu.as_ref(),
        ) {
            let expected_pred = codebook_tokens.len().saturating_sub(1);
            if pred_tables.len() == expected_pred {
                let mut out = vec![0.0f32; hidden_size];
                let token0 = codebook_tokens[0] as usize;
                let row0 = token0 * hidden_size..(token0 + 1) * hidden_size;
                if row0.end <= codec_table.len() {
                    out.copy_from_slice(&codec_table[row0]);
                    let mut cpu_ok = true;
                    for (cb_idx, &token) in codebook_tokens[1..].iter().enumerate() {
                        let tok = token as usize;
                        let row = tok * hidden_size..(tok + 1) * hidden_size;
                        let table = &pred_tables[cb_idx];
                        if row.end > table.len() {
                            cpu_ok = false;
                            break;
                        }
                        for i in 0..hidden_size {
                            out[i] += table[row.start + i];
                        }
                    }
                    if cpu_ok {
                        return Ok(out);
                    }
                }
            }
        }

        let mut out = self.lookup_codec_embedding_rows(&[codebook_tokens[0]], thread_count)?;
        for (cb_idx, &token) in codebook_tokens[1..].iter().enumerate() {
            let row = self.lookup_code_pred_embedding_row(cb_idx, token, thread_count)?;
            for i in 0..hidden_size {
                out[i] += row[i];
            }
        }
        Ok(out)
    }

    fn forward_code_pred_sequence_recompute(
        &self,
        hidden_state: &[f32],
        codebook_0_token: i32,
        prev_codes: &[i32],
        thread_count: usize,
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        let hidden_size = self.config.hidden_size as usize;
        if hidden_state.len() != hidden_size {
            return Err(Qwen3TtsError::InvalidInput(
                "code predictor hidden state shape is invalid".into(),
            ));
        }
        if prev_codes.len() >= self.config.n_codebooks as usize {
            return Err(Qwen3TtsError::InvalidInput(
                "too many previous code predictor tokens".into(),
            ));
        }

        let mut sequence_embd = Vec::with_capacity((2 + prev_codes.len()) * hidden_size);
        sequence_embd.extend_from_slice(hidden_state);
        sequence_embd.extend_from_slice(
            &self.lookup_codec_embedding_rows(&[codebook_0_token], thread_count)?,
        );
        for (cb_idx, &token) in prev_codes.iter().enumerate() {
            sequence_embd.extend_from_slice(&self.lookup_code_pred_embedding_row(
                cb_idx,
                token,
                thread_count,
            )?);
        }

        let n_tokens = sequence_embd.len() / hidden_size;
        let graph_nodes = 2048;
        let ctx = ComputeContext::new_graph(graph_nodes)?;
        let graph = unsafe { sys::ggml_new_graph_custom(ctx.as_ptr(), graph_nodes, false) };
        let graph = NonNull::new(graph).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate code predictor graph".into())
        })?;

        let inp_embd = unsafe {
            sys::ggml_new_tensor_2d(
                ctx.as_ptr(),
                sys::ggml_type_GGML_TYPE_F32,
                hidden_size as i64,
                n_tokens as i64,
            )
        };
        let inp_embd = NonNull::new(inp_embd).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate code predictor input".into())
        })?;

        let inp_pos = unsafe {
            sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_I32, n_tokens as i64)
        };
        let inp_pos = NonNull::new(inp_pos).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate code predictor positions".into())
        })?;
        let positions = (0..n_tokens as i32).collect::<Vec<_>>();

        let mut inp_l = inp_embd.as_ptr();
        let kq_scale = 1.0f32 / (self.config.head_dim as f32).sqrt();
        let mut attn_softmax_mask: Option<(*mut sys::ggml_tensor, Vec<f32>)> = None;
        for layer in &self.code_pred.layers {
            let mut cur =
                unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_l, self.config.rms_norm_eps) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, layer.attn_norm.as_ptr()) };

            let mut q_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_q.as_ptr(), cur) };
            let mut k_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_k.as_ptr(), cur) };
            let mut v_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_v.as_ptr(), cur) };

            q_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    q_cur,
                    self.config.head_dim as i64,
                    self.config.n_attention_heads as i64,
                    n_tokens as i64,
                )
            };
            k_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    k_cur,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    n_tokens as i64,
                )
            };
            v_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    v_cur,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    n_tokens as i64,
                )
            };

            if let Some(attn_q_norm) = layer.attn_q_norm {
                q_cur =
                    unsafe { sys::ggml_rms_norm(ctx.as_ptr(), q_cur, self.config.rms_norm_eps) };
                q_cur = unsafe { sys::ggml_mul(ctx.as_ptr(), q_cur, attn_q_norm.as_ptr()) };
            }
            if let Some(attn_k_norm) = layer.attn_k_norm {
                k_cur =
                    unsafe { sys::ggml_rms_norm(ctx.as_ptr(), k_cur, self.config.rms_norm_eps) };
                k_cur = unsafe { sys::ggml_mul(ctx.as_ptr(), k_cur, attn_k_norm.as_ptr()) };
            }

            q_cur = unsafe {
                sys::ggml_rope_ext(
                    ctx.as_ptr(),
                    q_cur,
                    inp_pos.as_ptr(),
                    std::ptr::null_mut(),
                    self.config.head_dim,
                    sys::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    self.config.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            };
            k_cur = unsafe {
                sys::ggml_rope_ext(
                    ctx.as_ptr(),
                    k_cur,
                    inp_pos.as_ptr(),
                    std::ptr::null_mut(),
                    self.config.head_dim,
                    sys::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    self.config.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            };

            let q = unsafe { sys::ggml_permute(ctx.as_ptr(), q_cur, 0, 2, 1, 3) };
            let k = unsafe { sys::ggml_permute(ctx.as_ptr(), k_cur, 0, 2, 1, 3) };
            let mut v = unsafe { sys::ggml_permute(ctx.as_ptr(), v_cur, 0, 2, 1, 3) };

            let mut kq = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), k, q) };
            kq = unsafe { sys::ggml_scale(ctx.as_ptr(), kq, kq_scale) };
            kq = unsafe {
                ggml_soft_max_ext_with_diag_mask_cache(ctx.as_ptr(), kq, 0, &mut attn_softmax_mask)
            };

            v = unsafe { sys::ggml_cont(ctx.as_ptr(), sys::ggml_transpose(ctx.as_ptr(), v)) };

            let mut kqv = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), v, kq) };
            kqv = unsafe { sys::ggml_permute(ctx.as_ptr(), kqv, 0, 2, 1, 3) };
            cur = unsafe {
                sys::ggml_cont_2d(
                    ctx.as_ptr(),
                    kqv,
                    (self.config.n_attention_heads * self.config.head_dim) as i64,
                    n_tokens as i64,
                )
            };

            cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_output.as_ptr(), cur) };
            cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_l) };
            let inp_ff = cur;

            cur = unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_ff, self.config.rms_norm_eps) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, layer.ffn_norm.as_ptr()) };
            let mut gate = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.ffn_gate.as_ptr(), cur) };
            let up = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.ffn_up.as_ptr(), cur) };
            gate = unsafe { sys::ggml_silu(ctx.as_ptr(), gate) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), gate, up) };

            let ffn_down_f32 = layer
                .ffn_down_f32
                .map(NonNull::as_ptr)
                .unwrap_or_else(|| unsafe {
                    sys::ggml_cast(
                        ctx.as_ptr(),
                        layer.ffn_down.as_ptr(),
                        sys::ggml_type_GGML_TYPE_F32,
                    )
                });
            cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), ffn_down_f32, cur) };
            inp_l = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_ff) };
        }

        let mut cur = unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_l, self.config.rms_norm_eps) };
        cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, self.code_pred.output_norm.as_ptr()) };
        let last_hidden = unsafe {
            sys::ggml_view_2d(
                ctx.as_ptr(),
                cur,
                hidden_size as i64,
                1,
                (*cur).nb[1],
                (n_tokens - 1) * (*cur).nb[1],
            )
        };
        let head = self
            .code_pred
            .heads
            .get(prev_codes.len())
            .ok_or_else(|| Qwen3TtsError::InvalidInput("missing code predictor head".into()))?;
        let mut logits = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), head.as_ptr(), last_hidden) };
        logits = unsafe { sys::ggml_cont(ctx.as_ptr(), logits) };
        unsafe {
            sys::ggml_build_forward_expand(graph.as_ptr(), logits);
        }
        let vocab = self.config.code_pred_vocab_size as usize;
        let mut logits_data = vec![0.0f32; vocab];
        let mut uploads = vec![
            TensorUpload {
                tensor: inp_embd.as_ptr(),
                bytes: slice_as_bytes(sequence_embd.as_slice()),
            },
            TensorUpload {
                tensor: inp_pos.as_ptr(),
                bytes: slice_as_bytes(positions.as_slice()),
            },
        ];
        if let Some((t, data)) = &attn_softmax_mask {
            uploads.push(TensorUpload {
                tensor: *t,
                bytes: slice_as_bytes(data.as_slice()),
            });
        }
        execute_graph(
            &self.code_pred._backends,
            graph,
            uploads.as_slice(),
            &mut [TensorDownload {
                tensor: logits,
                bytes: slice_as_bytes_mut(logits_data.as_mut_slice()),
            }],
            thread_count,
            "code predictor graph execution failed",
        )?;
        Ok(logits_data)
    }

    fn lookup_code_pred_embedding_row(
        &self,
        cb_idx: usize,
        token_id: i32,
        thread_count: usize,
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        if token_id < 0 || token_id >= self.config.code_pred_vocab_size {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "code predictor token {token_id} out of range 0..{} for codebook {}",
                self.config.code_pred_vocab_size - 1,
                cb_idx + 1
            )));
        }
        let weight = self.code_pred.embeddings.get(cb_idx).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("missing code predictor embedding".into())
        })?;
        let hidden_size = self.config.hidden_size as usize;
        let graph_nodes = 16;
        let ctx = ComputeContext::new_graph(graph_nodes)?;
        let graph = unsafe { sys::ggml_new_graph_custom(ctx.as_ptr(), graph_nodes, false) };
        let graph = NonNull::new(graph).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate code predictor embedding graph".into())
        })?;
        let inp_token =
            unsafe { sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_I32, 1) };
        let inp_token = NonNull::new(inp_token).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate code predictor embedding input".into())
        })?;
        let mut rows =
            unsafe { sys::ggml_get_rows(ctx.as_ptr(), weight.as_ptr(), inp_token.as_ptr()) };
        rows = unsafe { sys::ggml_cast(ctx.as_ptr(), rows, sys::ggml_type_GGML_TYPE_F32) };
        unsafe {
            sys::ggml_build_forward_expand(graph.as_ptr(), rows);
        }
        let mut data = vec![0.0f32; hidden_size];
        execute_graph(
            &self.code_pred._backends,
            graph,
            &[TensorUpload {
                tensor: inp_token.as_ptr(),
                bytes: slice_as_bytes(std::slice::from_ref(&token_id)),
            }],
            &mut [TensorDownload {
                tensor: rows,
                bytes: slice_as_bytes_mut(data.as_mut_slice()),
            }],
            thread_count,
            "code predictor embedding lookup failed",
        )?;
        Ok(data)
    }

    fn forward_code_pred_prefill_cached(
        &self,
        hidden_state: &[f32],
        codebook_0_token: i32,
        thread_count: usize,
        cache: &CodePredKvCache,
    ) -> Result<CodePredStepOutputs, Qwen3TtsError> {
        let hidden_size = self.config.hidden_size as usize;
        if hidden_state.len() != hidden_size {
            return Err(Qwen3TtsError::InvalidInput(
                "code predictor hidden state shape is invalid".into(),
            ));
        }
        if codebook_0_token < 0 || codebook_0_token >= self.config.codec_vocab_size {
            return Err(Qwen3TtsError::InvalidInput(format!(
                "codec token {codebook_0_token} out of range 0..{}",
                self.config.codec_vocab_size - 1
            )));
        }

        let graph_nodes = 2048;
        let ctx = ComputeContext::new_graph(graph_nodes)?;
        let graph = unsafe { sys::ggml_new_graph_custom(ctx.as_ptr(), graph_nodes, false) };
        let graph = NonNull::new(graph).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate code predictor prefill graph".into())
        })?;

        let inp_hidden = unsafe {
            sys::ggml_new_tensor_1d(
                ctx.as_ptr(),
                sys::ggml_type_GGML_TYPE_F32,
                hidden_size as i64,
            )
        };
        let inp_hidden = NonNull::new(inp_hidden).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate code predictor hidden input".into())
        })?;

        let inp_cb0_embd = unsafe {
            sys::ggml_new_tensor_2d(
                ctx.as_ptr(),
                sys::ggml_type_GGML_TYPE_F32,
                hidden_size as i64,
                1,
            )
        };
        let inp_cb0_embd = NonNull::new(inp_cb0_embd).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate code predictor cb0 embedding".into())
        })?;

        let inp_pos =
            unsafe { sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_I32, 2) };
        let inp_pos = NonNull::new(inp_pos).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate code predictor positions".into())
        })?;
        let positions = [0i32, 1i32];

        let hidden_2d = unsafe {
            sys::ggml_reshape_2d(ctx.as_ptr(), inp_hidden.as_ptr(), hidden_size as i64, 1)
        };
        let mut inp_l =
            unsafe { sys::ggml_concat(ctx.as_ptr(), hidden_2d, inp_cb0_embd.as_ptr(), 1) };
        let kq_scale = 1.0f32 / (self.config.head_dim as f32).sqrt();
        let mut attn_softmax_mask: Option<(*mut sys::ggml_tensor, Vec<f32>)> = None;

        for (layer_idx, layer) in self.code_pred.layers.iter().enumerate() {
            let mut cur =
                unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_l, self.config.rms_norm_eps) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, layer.attn_norm.as_ptr()) };
            let mut q_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_q.as_ptr(), cur) };
            let mut k_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_k.as_ptr(), cur) };
            let mut v_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_v.as_ptr(), cur) };
            q_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    q_cur,
                    self.config.head_dim as i64,
                    self.config.n_attention_heads as i64,
                    2,
                )
            };
            k_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    k_cur,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    2,
                )
            };
            v_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    v_cur,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    2,
                )
            };

            if let Some(attn_q_norm) = layer.attn_q_norm {
                q_cur =
                    unsafe { sys::ggml_rms_norm(ctx.as_ptr(), q_cur, self.config.rms_norm_eps) };
                q_cur = unsafe { sys::ggml_mul(ctx.as_ptr(), q_cur, attn_q_norm.as_ptr()) };
            }
            if let Some(attn_k_norm) = layer.attn_k_norm {
                k_cur =
                    unsafe { sys::ggml_rms_norm(ctx.as_ptr(), k_cur, self.config.rms_norm_eps) };
                k_cur = unsafe { sys::ggml_mul(ctx.as_ptr(), k_cur, attn_k_norm.as_ptr()) };
            }

            q_cur = unsafe {
                sys::ggml_rope_ext(
                    ctx.as_ptr(),
                    q_cur,
                    inp_pos.as_ptr(),
                    std::ptr::null_mut(),
                    self.config.head_dim,
                    sys::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    self.config.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            };
            k_cur = unsafe {
                sys::ggml_rope_ext(
                    ctx.as_ptr(),
                    k_cur,
                    inp_pos.as_ptr(),
                    std::ptr::null_mut(),
                    self.config.head_dim,
                    sys::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    self.config.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            };

            let k_cache = cache.k_cache[layer_idx].as_ptr();
            let v_cache = cache.v_cache[layer_idx].as_ptr();
            let k_cache_view = unsafe {
                sys::ggml_view_3d(
                    ctx.as_ptr(),
                    k_cache,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    2,
                    (*k_cache).nb[1],
                    (*k_cache).nb[2],
                    0,
                )
            };
            let v_cache_view = unsafe {
                sys::ggml_view_3d(
                    ctx.as_ptr(),
                    v_cache,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    2,
                    (*v_cache).nb[1],
                    (*v_cache).nb[2],
                    0,
                )
            };
            unsafe {
                sys::ggml_build_forward_expand(
                    graph.as_ptr(),
                    sys::ggml_cpy(ctx.as_ptr(), k_cur, k_cache_view),
                );
                sys::ggml_build_forward_expand(
                    graph.as_ptr(),
                    sys::ggml_cpy(ctx.as_ptr(), v_cur, v_cache_view),
                );
            }

            let q = unsafe { sys::ggml_permute(ctx.as_ptr(), q_cur, 0, 2, 1, 3) };
            let k = unsafe { sys::ggml_permute(ctx.as_ptr(), k_cur, 0, 2, 1, 3) };
            let mut v = unsafe { sys::ggml_permute(ctx.as_ptr(), v_cur, 0, 2, 1, 3) };
            let mut kq = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), k, q) };
            kq = unsafe { sys::ggml_scale(ctx.as_ptr(), kq, kq_scale) };
            kq = unsafe {
                ggml_soft_max_ext_with_diag_mask_cache(ctx.as_ptr(), kq, 0, &mut attn_softmax_mask)
            };
            v = unsafe { sys::ggml_cont(ctx.as_ptr(), sys::ggml_transpose(ctx.as_ptr(), v)) };
            let mut kqv = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), v, kq) };
            kqv = unsafe { sys::ggml_permute(ctx.as_ptr(), kqv, 0, 2, 1, 3) };
            cur = unsafe {
                sys::ggml_cont_2d(
                    ctx.as_ptr(),
                    kqv,
                    (self.config.n_attention_heads * self.config.head_dim) as i64,
                    2,
                )
            };
            cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_output.as_ptr(), cur) };
            cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_l) };
            let inp_ff = cur;
            cur = unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_ff, self.config.rms_norm_eps) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, layer.ffn_norm.as_ptr()) };
            let mut gate = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.ffn_gate.as_ptr(), cur) };
            let up = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.ffn_up.as_ptr(), cur) };
            gate = unsafe { sys::ggml_silu(ctx.as_ptr(), gate) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), gate, up) };
            let ffn_down_f32 = layer
                .ffn_down_f32
                .map(NonNull::as_ptr)
                .unwrap_or_else(|| unsafe {
                    sys::ggml_cast(
                        ctx.as_ptr(),
                        layer.ffn_down.as_ptr(),
                        sys::ggml_type_GGML_TYPE_F32,
                    )
                });
            cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), ffn_down_f32, cur) };
            inp_l = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_ff) };
        }

        let mut cur = unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_l, self.config.rms_norm_eps) };
        cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, self.code_pred.output_norm.as_ptr()) };
        let last_hidden = unsafe {
            sys::ggml_view_2d(
                ctx.as_ptr(),
                cur,
                hidden_size as i64,
                1,
                (*cur).nb[1],
                hidden_size * std::mem::size_of::<f32>(),
            )
        };
        let mut logits = unsafe {
            sys::ggml_mul_mat(ctx.as_ptr(), self.code_pred.heads[0].as_ptr(), last_hidden)
        };
        logits = unsafe { sys::ggml_cont(ctx.as_ptr(), logits) };
        unsafe {
            sys::ggml_build_forward_expand(graph.as_ptr(), logits);
        }

        let mut logits_data = vec![0.0f32; self.config.code_pred_vocab_size as usize];
        let cb0_embd = self.lookup_codec_embedding_rows(&[codebook_0_token], thread_count)?;
        let mut uploads = vec![
            TensorUpload {
                tensor: inp_hidden.as_ptr(),
                bytes: slice_as_bytes(hidden_state),
            },
            TensorUpload {
                tensor: inp_cb0_embd.as_ptr(),
                bytes: slice_as_bytes(cb0_embd.as_slice()),
            },
            TensorUpload {
                tensor: inp_pos.as_ptr(),
                bytes: slice_as_bytes(&positions),
            },
        ];
        if let Some((t, data)) = &attn_softmax_mask {
            uploads.push(TensorUpload {
                tensor: *t,
                bytes: slice_as_bytes(data.as_slice()),
            });
        }
        execute_graph(
            &cache._backends,
            graph,
            uploads.as_slice(),
            &mut [TensorDownload {
                tensor: logits,
                bytes: slice_as_bytes_mut(logits_data.as_mut_slice()),
            }],
            thread_count,
            "code predictor prefill graph execution failed",
        )?;
        Ok(CodePredStepOutputs {
            logits: logits_data,
        })
    }

    fn forward_code_pred_step_cached(
        &self,
        prev_code: i32,
        n_past: usize,
        generation_step: usize,
        thread_count: usize,
        cache: &CodePredKvCache,
    ) -> Result<CodePredStepOutputs, Qwen3TtsError> {
        if generation_step == 0 || generation_step >= self.code_pred.heads.len() {
            return Err(Qwen3TtsError::InvalidInput(
                "code predictor generation step out of range".into(),
            ));
        }
        if n_past >= cache.n_ctx {
            return Err(Qwen3TtsError::InvalidInput(
                "code predictor context length exceeded".into(),
            ));
        }

        let hidden_size = self.config.hidden_size as usize;
        let graph_nodes = 2048;
        let ctx = ComputeContext::new_graph(graph_nodes)?;
        let graph = unsafe { sys::ggml_new_graph_custom(ctx.as_ptr(), graph_nodes, false) };
        let graph = NonNull::new(graph).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate code predictor step graph".into())
        })?;

        let inp_code =
            unsafe { sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_I32, 1) };
        let inp_code = NonNull::new(inp_code).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate code predictor token input".into())
        })?;

        let inp_pos =
            unsafe { sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_I32, 1) };
        let inp_pos = NonNull::new(inp_pos).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate code predictor position input".into())
        })?;
        let pos = n_past as i32;

        let weight = self
            .code_pred
            .embeddings
            .get(generation_step - 1)
            .ok_or_else(|| {
                Qwen3TtsError::InvalidInput("missing code predictor embedding".into())
            })?;
        let mut inp_l =
            unsafe { sys::ggml_get_rows(ctx.as_ptr(), weight.as_ptr(), inp_code.as_ptr()) };
        inp_l = unsafe { sys::ggml_reshape_2d(ctx.as_ptr(), inp_l, hidden_size as i64, 1) };
        let kq_scale = 1.0f32 / (self.config.head_dim as f32).sqrt();
        let mut attn_softmax_mask: Option<(*mut sys::ggml_tensor, Vec<f32>)> = None;

        for (layer_idx, layer) in self.code_pred.layers.iter().enumerate() {
            let mut cur =
                unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_l, self.config.rms_norm_eps) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, layer.attn_norm.as_ptr()) };
            let mut q_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_q.as_ptr(), cur) };
            let mut k_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_k.as_ptr(), cur) };
            let mut v_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_v.as_ptr(), cur) };
            q_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    q_cur,
                    self.config.head_dim as i64,
                    self.config.n_attention_heads as i64,
                    1,
                )
            };
            k_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    k_cur,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    1,
                )
            };
            v_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    v_cur,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    1,
                )
            };

            if let Some(attn_q_norm) = layer.attn_q_norm {
                q_cur =
                    unsafe { sys::ggml_rms_norm(ctx.as_ptr(), q_cur, self.config.rms_norm_eps) };
                q_cur = unsafe { sys::ggml_mul(ctx.as_ptr(), q_cur, attn_q_norm.as_ptr()) };
            }
            if let Some(attn_k_norm) = layer.attn_k_norm {
                k_cur =
                    unsafe { sys::ggml_rms_norm(ctx.as_ptr(), k_cur, self.config.rms_norm_eps) };
                k_cur = unsafe { sys::ggml_mul(ctx.as_ptr(), k_cur, attn_k_norm.as_ptr()) };
            }

            q_cur = unsafe {
                sys::ggml_rope_ext(
                    ctx.as_ptr(),
                    q_cur,
                    inp_pos.as_ptr(),
                    std::ptr::null_mut(),
                    self.config.head_dim,
                    sys::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    self.config.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            };
            k_cur = unsafe {
                sys::ggml_rope_ext(
                    ctx.as_ptr(),
                    k_cur,
                    inp_pos.as_ptr(),
                    std::ptr::null_mut(),
                    self.config.head_dim,
                    sys::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    self.config.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            };

            let k_cache = cache.k_cache[layer_idx].as_ptr();
            let v_cache = cache.v_cache[layer_idx].as_ptr();
            let k_cache_view = unsafe {
                sys::ggml_view_3d(
                    ctx.as_ptr(),
                    k_cache,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    1,
                    (*k_cache).nb[1],
                    (*k_cache).nb[2],
                    n_past * (*k_cache).nb[2],
                )
            };
            let v_cache_view = unsafe {
                sys::ggml_view_3d(
                    ctx.as_ptr(),
                    v_cache,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    1,
                    (*v_cache).nb[1],
                    (*v_cache).nb[2],
                    n_past * (*v_cache).nb[2],
                )
            };
            unsafe {
                sys::ggml_build_forward_expand(
                    graph.as_ptr(),
                    sys::ggml_cpy(ctx.as_ptr(), k_cur, k_cache_view),
                );
                sys::ggml_build_forward_expand(
                    graph.as_ptr(),
                    sys::ggml_cpy(ctx.as_ptr(), v_cur, v_cache_view),
                );
            }

            let n_kv = n_past + 1;
            let mut k = unsafe {
                sys::ggml_view_3d(
                    ctx.as_ptr(),
                    k_cache,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    n_kv as i64,
                    (*k_cache).nb[1],
                    (*k_cache).nb[2],
                    0,
                )
            };
            let mut v = unsafe {
                sys::ggml_view_3d(
                    ctx.as_ptr(),
                    v_cache,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    n_kv as i64,
                    (*v_cache).nb[1],
                    (*v_cache).nb[2],
                    0,
                )
            };
            let q = unsafe { sys::ggml_permute(ctx.as_ptr(), q_cur, 0, 2, 1, 3) };
            k = unsafe { sys::ggml_permute(ctx.as_ptr(), k, 0, 2, 1, 3) };
            v = unsafe { sys::ggml_permute(ctx.as_ptr(), v, 0, 2, 1, 3) };
            let mut kq = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), k, q) };
            kq = unsafe { sys::ggml_scale(ctx.as_ptr(), kq, kq_scale) };
            kq = unsafe {
                ggml_soft_max_ext_with_diag_mask_cache(
                    ctx.as_ptr(),
                    kq,
                    n_past as i32,
                    &mut attn_softmax_mask,
                )
            };
            v = unsafe { sys::ggml_cont(ctx.as_ptr(), sys::ggml_transpose(ctx.as_ptr(), v)) };
            let mut kqv = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), v, kq) };
            kqv = unsafe { sys::ggml_permute(ctx.as_ptr(), kqv, 0, 2, 1, 3) };
            cur = unsafe {
                sys::ggml_cont_2d(
                    ctx.as_ptr(),
                    kqv,
                    (self.config.n_attention_heads * self.config.head_dim) as i64,
                    1,
                )
            };
            cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_output.as_ptr(), cur) };
            cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_l) };
            let inp_ff = cur;
            cur = unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_ff, self.config.rms_norm_eps) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, layer.ffn_norm.as_ptr()) };
            let mut gate = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.ffn_gate.as_ptr(), cur) };
            let up = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.ffn_up.as_ptr(), cur) };
            gate = unsafe { sys::ggml_silu(ctx.as_ptr(), gate) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), gate, up) };
            let ffn_down_f32 = layer
                .ffn_down_f32
                .map(NonNull::as_ptr)
                .unwrap_or_else(|| unsafe {
                    sys::ggml_cast(
                        ctx.as_ptr(),
                        layer.ffn_down.as_ptr(),
                        sys::ggml_type_GGML_TYPE_F32,
                    )
                });
            cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), ffn_down_f32, cur) };
            inp_l = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_ff) };
        }

        let mut cur = unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_l, self.config.rms_norm_eps) };
        cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, self.code_pred.output_norm.as_ptr()) };
        let mut logits = unsafe {
            sys::ggml_mul_mat(
                ctx.as_ptr(),
                self.code_pred.heads[generation_step].as_ptr(),
                cur,
            )
        };
        logits = unsafe { sys::ggml_cont(ctx.as_ptr(), logits) };
        unsafe {
            sys::ggml_build_forward_expand(graph.as_ptr(), logits);
        }

        let mut logits_data = vec![0.0f32; self.config.code_pred_vocab_size as usize];
        let mut uploads = vec![
            TensorUpload {
                tensor: inp_code.as_ptr(),
                bytes: slice_as_bytes(std::slice::from_ref(&prev_code)),
            },
            TensorUpload {
                tensor: inp_pos.as_ptr(),
                bytes: slice_as_bytes(std::slice::from_ref(&pos)),
            },
        ];
        if let Some((t, data)) = &attn_softmax_mask {
            uploads.push(TensorUpload {
                tensor: *t,
                bytes: slice_as_bytes(data.as_slice()),
            });
        }
        execute_graph(
            &cache._backends,
            graph,
            uploads.as_slice(),
            &mut [TensorDownload {
                tensor: logits,
                bytes: slice_as_bytes_mut(logits_data.as_mut_slice()),
            }],
            thread_count,
            "code predictor step graph execution failed",
        )?;
        Ok(CodePredStepOutputs {
            logits: logits_data,
        })
    }

    fn build_talker_step_embedding(
        &self,
        codebook_tokens: &[i32],
        trailing_row: &[f32],
        thread_count: usize,
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        let hidden_size = self.config.hidden_size as usize;
        if codebook_tokens.len() != self.config.n_codebooks as usize {
            return Err(Qwen3TtsError::InvalidInput(
                "full codebook frame is required for step embedding".into(),
            ));
        }
        if trailing_row.len() != hidden_size {
            return Err(Qwen3TtsError::InvalidInput(
                "trailing row shape is invalid".into(),
            ));
        }
        let mut cpu_sum = self.sum_codec_frame_embeddings(codebook_tokens, thread_count)?;
        for i in 0..hidden_size {
            cpu_sum[i] += trailing_row[i];
        }
        if self.talker.codec_embd_cpu.is_some() && self.code_pred.codec_embd_cpu.is_some() {
            return Ok(cpu_sum);
        }

        let graph_nodes = 128;
        let ctx = ComputeContext::new_graph(graph_nodes)?;
        let graph = unsafe { sys::ggml_new_graph_custom(ctx.as_ptr(), graph_nodes, false) };
        let graph = NonNull::new(graph).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate step embedding graph".into())
        })?;

        let inp_trailing = unsafe {
            sys::ggml_new_tensor_2d(
                ctx.as_ptr(),
                sys::ggml_type_GGML_TYPE_F32,
                hidden_size as i64,
                1,
            )
        };
        let inp_trailing = NonNull::new(inp_trailing).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate trailing row input".into())
        })?;
        let inp_cb0 =
            unsafe { sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_I32, 1) };
        let inp_cb0 = NonNull::new(inp_cb0).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate codec token input".into())
        })?;

        let mut cur = unsafe {
            sys::ggml_get_rows(
                ctx.as_ptr(),
                self.talker.codec_embd.as_ptr(),
                inp_cb0.as_ptr(),
            )
        };
        cur = unsafe { sys::ggml_cast(ctx.as_ptr(), cur, sys::ggml_type_GGML_TYPE_F32) };

        let mut code_inputs = Vec::with_capacity(codebook_tokens.len().saturating_sub(1));
        for (cb_idx, weight) in self.code_pred.embeddings.iter().enumerate() {
            let inp_code =
                unsafe { sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_I32, 1) };
            let inp_code = NonNull::new(inp_code).ok_or_else(|| {
                Qwen3TtsError::InvalidInput("failed to allocate code predictor token input".into())
            })?;
            let mut embd =
                unsafe { sys::ggml_get_rows(ctx.as_ptr(), weight.as_ptr(), inp_code.as_ptr()) };
            embd = unsafe { sys::ggml_cast(ctx.as_ptr(), embd, sys::ggml_type_GGML_TYPE_F32) };
            cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, embd) };
            code_inputs.push((cb_idx, inp_code));
        }
        cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_trailing.as_ptr()) };

        unsafe {
            sys::ggml_build_forward_expand(graph.as_ptr(), cur);
        }

        let mut step_embd = vec![0.0f32; hidden_size];
        let mut uploads = Vec::with_capacity(code_inputs.len() + 2);
        uploads.push(TensorUpload {
            tensor: inp_trailing.as_ptr(),
            bytes: slice_as_bytes(trailing_row),
        });
        uploads.push(TensorUpload {
            tensor: inp_cb0.as_ptr(),
            bytes: slice_as_bytes(std::slice::from_ref(&codebook_tokens[0])),
        });
        for (cb_idx, inp_code) in &code_inputs {
            uploads.push(TensorUpload {
                tensor: inp_code.as_ptr(),
                bytes: slice_as_bytes(std::slice::from_ref(&codebook_tokens[*cb_idx + 1])),
            });
        }

        execute_graph(
            &self.talker._backends,
            graph,
            uploads.as_slice(),
            &mut [TensorDownload {
                tensor: cur,
                bytes: slice_as_bytes_mut(step_embd.as_mut_slice()),
            }],
            thread_count,
            "step embedding graph execution failed",
        )?;
        Ok(step_embd)
    }

    fn forward_prefill_cached(
        &self,
        prefill_embd: &[f32],
        thread_count: usize,
        cache: &TalkerKvCache,
    ) -> Result<StepForwardOutputs, Qwen3TtsError> {
        let hidden_size = self.config.hidden_size as usize;
        if prefill_embd.is_empty() || !prefill_embd.len().is_multiple_of(hidden_size) {
            return Err(Qwen3TtsError::InvalidInput(
                "prefill embedding shape is invalid".into(),
            ));
        }

        let n_tokens = prefill_embd.len() / hidden_size;
        if n_tokens > cache.n_ctx {
            return Err(Qwen3TtsError::InvalidInput(
                "talker context length exceeded".into(),
            ));
        }

        let graph_nodes = 4096;
        let ctx = ComputeContext::new_graph(graph_nodes)?;
        let graph = unsafe { sys::ggml_new_graph_custom(ctx.as_ptr(), graph_nodes, false) };
        let graph = NonNull::new(graph).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate cached prefill graph".into())
        })?;

        let inp_prefill_embd = unsafe {
            sys::ggml_new_tensor_2d(
                ctx.as_ptr(),
                sys::ggml_type_GGML_TYPE_F32,
                hidden_size as i64,
                n_tokens as i64,
            )
        };
        let inp_prefill_embd = NonNull::new(inp_prefill_embd).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate prefill input".into())
        })?;

        let inp_pos = unsafe {
            sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_I32, n_tokens as i64)
        };
        let inp_pos = NonNull::new(inp_pos).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate position input".into())
        })?;
        let positions = (0..n_tokens as i32).collect::<Vec<_>>();

        let mut inp_l = inp_prefill_embd.as_ptr();
        let kq_scale = 1.0f32 / (self.config.head_dim as f32).sqrt();
        let mut attn_softmax_mask: Option<(*mut sys::ggml_tensor, Vec<f32>)> = None;
        let mut kv_writeback = Vec::<KvWritebackTensorDownloads>::new();

        for (layer_idx, layer) in self.talker.layers.iter().enumerate() {
            let mut cur =
                unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_l, self.config.rms_norm_eps) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, layer.attn_norm.as_ptr()) };

            let mut q_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_q.as_ptr(), cur) };
            let mut k_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_k.as_ptr(), cur) };
            let mut v_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_v.as_ptr(), cur) };

            q_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    q_cur,
                    self.config.head_dim as i64,
                    self.config.n_attention_heads as i64,
                    n_tokens as i64,
                )
            };
            k_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    k_cur,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    n_tokens as i64,
                )
            };
            v_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    v_cur,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    n_tokens as i64,
                )
            };

            if let Some(attn_q_norm) = layer.attn_q_norm {
                q_cur =
                    unsafe { sys::ggml_rms_norm(ctx.as_ptr(), q_cur, self.config.rms_norm_eps) };
                q_cur = unsafe { sys::ggml_mul(ctx.as_ptr(), q_cur, attn_q_norm.as_ptr()) };
            }
            if let Some(attn_k_norm) = layer.attn_k_norm {
                k_cur =
                    unsafe { sys::ggml_rms_norm(ctx.as_ptr(), k_cur, self.config.rms_norm_eps) };
                k_cur = unsafe { sys::ggml_mul(ctx.as_ptr(), k_cur, attn_k_norm.as_ptr()) };
            }

            q_cur = unsafe {
                sys::ggml_rope_ext(
                    ctx.as_ptr(),
                    q_cur,
                    inp_pos.as_ptr(),
                    std::ptr::null_mut(),
                    self.config.head_dim,
                    sys::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    self.config.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            };
            k_cur = unsafe {
                sys::ggml_rope_ext(
                    ctx.as_ptr(),
                    k_cur,
                    inp_pos.as_ptr(),
                    std::ptr::null_mut(),
                    self.config.head_dim,
                    sys::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    self.config.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            };

            match cache.storage() {
                TalkerKvStorage::F16 => {
                    let k_cache = cache.k_cache[layer_idx].as_ptr();
                    let v_cache = cache.v_cache[layer_idx].as_ptr();
                    let k_cache_view = unsafe {
                        sys::ggml_view_3d(
                            ctx.as_ptr(),
                            k_cache,
                            self.config.head_dim as i64,
                            self.config.n_key_value_heads as i64,
                            n_tokens as i64,
                            (*k_cache).nb[1],
                            (*k_cache).nb[2],
                            0,
                        )
                    };
                    let v_cache_view = unsafe {
                        sys::ggml_view_3d(
                            ctx.as_ptr(),
                            v_cache,
                            self.config.head_dim as i64,
                            self.config.n_key_value_heads as i64,
                            n_tokens as i64,
                            (*v_cache).nb[1],
                            (*v_cache).nb[2],
                            0,
                        )
                    };
                    unsafe {
                        sys::ggml_build_forward_expand(
                            graph.as_ptr(),
                            sys::ggml_cpy(ctx.as_ptr(), k_cur, k_cache_view),
                        );
                        sys::ggml_build_forward_expand(
                            graph.as_ptr(),
                            sys::ggml_cpy(ctx.as_ptr(), v_cur, v_cache_view),
                        );
                    }
                }
                TalkerKvStorage::TurboQuantQ8_0 => {
                    let k_store = unsafe { sys::ggml_cont(ctx.as_ptr(), k_cur) };
                    let v_store = unsafe { sys::ggml_cont(ctx.as_ptr(), v_cur) };
                    unsafe {
                        sys::ggml_build_forward_expand(graph.as_ptr(), k_store);
                        sys::ggml_build_forward_expand(graph.as_ptr(), v_store);
                    }
                    let rows = self.config.n_key_value_heads as usize
                        * n_tokens
                        * self.config.head_dim as usize;
                    kv_writeback.push(KvWritebackTensorDownloads {
                        layer_idx,
                        token_start: 0,
                        n_tokens,
                        k_tensor: k_store,
                        v_tensor: v_store,
                        k_data: vec![0.0; rows],
                        v_data: vec![0.0; rows],
                    });
                }
            }

            let q = unsafe { sys::ggml_permute(ctx.as_ptr(), q_cur, 0, 2, 1, 3) };
            let k = unsafe { sys::ggml_permute(ctx.as_ptr(), k_cur, 0, 2, 1, 3) };
            let mut v = unsafe { sys::ggml_permute(ctx.as_ptr(), v_cur, 0, 2, 1, 3) };

            let mut kq = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), k, q) };
            kq = unsafe { sys::ggml_scale(ctx.as_ptr(), kq, kq_scale) };
            kq = unsafe {
                ggml_soft_max_ext_with_diag_mask_cache(ctx.as_ptr(), kq, 0, &mut attn_softmax_mask)
            };

            v = unsafe { sys::ggml_cont(ctx.as_ptr(), sys::ggml_transpose(ctx.as_ptr(), v)) };

            let mut kqv = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), v, kq) };
            kqv = unsafe { sys::ggml_permute(ctx.as_ptr(), kqv, 0, 2, 1, 3) };
            cur = unsafe {
                sys::ggml_cont_2d(
                    ctx.as_ptr(),
                    kqv,
                    (self.config.n_attention_heads * self.config.head_dim) as i64,
                    n_tokens as i64,
                )
            };

            cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_output.as_ptr(), cur) };
            cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_l) };
            let inp_ff = cur;

            cur = unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_ff, self.config.rms_norm_eps) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, layer.ffn_norm.as_ptr()) };

            let mut gate = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.ffn_gate.as_ptr(), cur) };
            let up = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.ffn_up.as_ptr(), cur) };
            gate = unsafe { sys::ggml_silu(ctx.as_ptr(), gate) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), gate, up) };

            let ffn_down_f32 = layer
                .ffn_down_f32
                .map(NonNull::as_ptr)
                .unwrap_or_else(|| unsafe {
                    sys::ggml_cast(
                        ctx.as_ptr(),
                        layer.ffn_down.as_ptr(),
                        sys::ggml_type_GGML_TYPE_F32,
                    )
                });
            cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), ffn_down_f32, cur) };
            inp_l = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_ff) };
        }

        let mut hidden_states =
            unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_l, self.config.rms_norm_eps) };
        hidden_states = unsafe {
            sys::ggml_mul(
                ctx.as_ptr(),
                hidden_states,
                self.talker.output_norm.as_ptr(),
            )
        };
        let last_hidden = unsafe {
            sys::ggml_view_2d(
                ctx.as_ptr(),
                hidden_states,
                hidden_size as i64,
                1,
                (*hidden_states).nb[1],
                (n_tokens - 1) * (*hidden_states).nb[1],
            )
        };
        let last_hidden = unsafe { sys::ggml_cont(ctx.as_ptr(), last_hidden) };
        let mut logits = unsafe {
            sys::ggml_mul_mat(ctx.as_ptr(), self.talker.codec_head.as_ptr(), last_hidden)
        };
        logits = unsafe { sys::ggml_cont(ctx.as_ptr(), logits) };

        unsafe {
            sys::ggml_build_forward_expand(graph.as_ptr(), last_hidden);
            sys::ggml_build_forward_expand(graph.as_ptr(), logits);
        }

        let mut hidden_data = vec![0.0f32; hidden_size];
        let mut logits_data = vec![0.0f32; self.config.codec_vocab_size as usize];
        let mut uploads = vec![
            TensorUpload {
                tensor: inp_prefill_embd.as_ptr(),
                bytes: slice_as_bytes(prefill_embd),
            },
            TensorUpload {
                tensor: inp_pos.as_ptr(),
                bytes: slice_as_bytes(positions.as_slice()),
            },
        ];
        if let Some((t, data)) = &attn_softmax_mask {
            uploads.push(TensorUpload {
                tensor: *t,
                bytes: slice_as_bytes(data.as_slice()),
            });
        }
        let mut downloads = vec![
            TensorDownload {
                tensor: last_hidden,
                bytes: slice_as_bytes_mut(hidden_data.as_mut_slice()),
            },
            TensorDownload {
                tensor: logits,
                bytes: slice_as_bytes_mut(logits_data.as_mut_slice()),
            },
        ];
        execute_graph(
            &cache._backends,
            graph,
            uploads.as_slice(),
            downloads.as_mut_slice(),
            thread_count,
            "cached prefill graph execution failed",
        )?;
        if cache.storage().is_quantized() {
            let layout = QuantizedKvWriteLayout {
                token_start: 0,
                n_tokens,
                rows_per_token: self.config.n_key_value_heads as usize,
                row_len: self.config.head_dim as usize,
            };
            for mut pending in kv_writeback {
                unsafe {
                    sys::ggml_backend_tensor_get(
                        pending.k_tensor,
                        pending.k_data.as_mut_ptr().cast(),
                        0,
                        std::mem::size_of_val(pending.k_data.as_slice()),
                    );
                    sys::ggml_backend_tensor_get(
                        pending.v_tensor,
                        pending.v_data.as_mut_ptr().cast(),
                        0,
                        std::mem::size_of_val(pending.v_data.as_slice()),
                    );
                }
                let _ = cache.quantized_write_layer(
                    pending.layer_idx,
                    QuantizedKvWriteLayout {
                        token_start: pending.token_start,
                        n_tokens: pending.n_tokens,
                        ..layout
                    },
                    pending.k_data.as_slice(),
                    pending.v_data.as_slice(),
                )?;
            }
        }

        Ok(StepForwardOutputs {
            hidden_state: hidden_data,
            logits: logits_data,
            kv_writeback_elapsed: Duration::ZERO,
            kv_download_elapsed: Duration::ZERO,
            kv_quantize_elapsed: Duration::ZERO,
            kv_upload_elapsed: Duration::ZERO,
        })
    }

    #[allow(dead_code)]
    fn forward_step_cached(
        &self,
        step_embd: &[f32],
        n_past: usize,
        thread_count: usize,
        cache: &TalkerKvCache,
    ) -> Result<StepForwardOutputs, Qwen3TtsError> {
        let hidden_size = self.config.hidden_size as usize;
        if step_embd.len() != hidden_size {
            return Err(Qwen3TtsError::InvalidInput(
                "step embedding shape is invalid".into(),
            ));
        }
        if n_past >= cache.n_ctx {
            return Err(Qwen3TtsError::InvalidInput(
                "talker context length exceeded".into(),
            ));
        }

        let graph_nodes = 4096;
        let ctx = ComputeContext::new_graph(graph_nodes)?;
        let graph = unsafe { sys::ggml_new_graph_custom(ctx.as_ptr(), graph_nodes, false) };
        let graph = NonNull::new(graph)
            .ok_or_else(|| Qwen3TtsError::InvalidInput("failed to allocate step graph".into()))?;

        let inp_step = unsafe {
            sys::ggml_new_tensor_2d(
                ctx.as_ptr(),
                sys::ggml_type_GGML_TYPE_F32,
                hidden_size as i64,
                1,
            )
        };
        let inp_step = NonNull::new(inp_step)
            .ok_or_else(|| Qwen3TtsError::InvalidInput("failed to allocate step input".into()))?;

        let inp_pos =
            unsafe { sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_I32, 1) };
        let inp_pos = NonNull::new(inp_pos).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate step position".into())
        })?;
        let pos = n_past as i32;

        let mut inp_l = inp_step.as_ptr();
        let kq_scale = 1.0f32 / (self.config.head_dim as f32).sqrt();
        let mut attn_softmax_mask: Option<(*mut sys::ggml_tensor, Vec<f32>)> = None;
        let mut kv_writeback = Vec::<KvWritebackTensorDownloads>::new();
        for (layer_idx, layer) in self.talker.layers.iter().enumerate() {
            let mut cur =
                unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_l, self.config.rms_norm_eps) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, layer.attn_norm.as_ptr()) };

            let mut q_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_q.as_ptr(), cur) };
            let mut k_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_k.as_ptr(), cur) };
            let mut v_cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_v.as_ptr(), cur) };

            q_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    q_cur,
                    self.config.head_dim as i64,
                    self.config.n_attention_heads as i64,
                    1,
                )
            };
            k_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    k_cur,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    1,
                )
            };
            v_cur = unsafe {
                sys::ggml_reshape_3d(
                    ctx.as_ptr(),
                    v_cur,
                    self.config.head_dim as i64,
                    self.config.n_key_value_heads as i64,
                    1,
                )
            };

            if let Some(attn_q_norm) = layer.attn_q_norm {
                q_cur =
                    unsafe { sys::ggml_rms_norm(ctx.as_ptr(), q_cur, self.config.rms_norm_eps) };
                q_cur = unsafe { sys::ggml_mul(ctx.as_ptr(), q_cur, attn_q_norm.as_ptr()) };
            }
            if let Some(attn_k_norm) = layer.attn_k_norm {
                k_cur =
                    unsafe { sys::ggml_rms_norm(ctx.as_ptr(), k_cur, self.config.rms_norm_eps) };
                k_cur = unsafe { sys::ggml_mul(ctx.as_ptr(), k_cur, attn_k_norm.as_ptr()) };
            }

            q_cur = unsafe {
                sys::ggml_rope_ext(
                    ctx.as_ptr(),
                    q_cur,
                    inp_pos.as_ptr(),
                    std::ptr::null_mut(),
                    self.config.head_dim,
                    sys::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    self.config.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            };
            k_cur = unsafe {
                sys::ggml_rope_ext(
                    ctx.as_ptr(),
                    k_cur,
                    inp_pos.as_ptr(),
                    std::ptr::null_mut(),
                    self.config.head_dim,
                    sys::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    self.config.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
            };

            let k_cache = cache.k_cache[layer_idx].as_ptr();
            let v_cache = cache.v_cache[layer_idx].as_ptr();
            let (mut k, mut v) = match cache.storage() {
                TalkerKvStorage::F16 => {
                    let k_cache_view = unsafe {
                        sys::ggml_view_3d(
                            ctx.as_ptr(),
                            k_cache,
                            self.config.head_dim as i64,
                            self.config.n_key_value_heads as i64,
                            1,
                            (*k_cache).nb[1],
                            (*k_cache).nb[2],
                            n_past * (*k_cache).nb[2],
                        )
                    };
                    let v_cache_view = unsafe {
                        sys::ggml_view_3d(
                            ctx.as_ptr(),
                            v_cache,
                            self.config.head_dim as i64,
                            self.config.n_key_value_heads as i64,
                            1,
                            (*v_cache).nb[1],
                            (*v_cache).nb[2],
                            n_past * (*v_cache).nb[2],
                        )
                    };
                    unsafe {
                        sys::ggml_build_forward_expand(
                            graph.as_ptr(),
                            sys::ggml_cpy(ctx.as_ptr(), k_cur, k_cache_view),
                        );
                        sys::ggml_build_forward_expand(
                            graph.as_ptr(),
                            sys::ggml_cpy(ctx.as_ptr(), v_cur, v_cache_view),
                        );
                    }

                    let n_kv = n_past + 1;
                    let k = unsafe {
                        sys::ggml_view_3d(
                            ctx.as_ptr(),
                            k_cache,
                            self.config.head_dim as i64,
                            self.config.n_key_value_heads as i64,
                            n_kv as i64,
                            (*k_cache).nb[1],
                            (*k_cache).nb[2],
                            0,
                        )
                    };
                    let v = unsafe {
                        sys::ggml_view_3d(
                            ctx.as_ptr(),
                            v_cache,
                            self.config.head_dim as i64,
                            self.config.n_key_value_heads as i64,
                            n_kv as i64,
                            (*v_cache).nb[1],
                            (*v_cache).nb[2],
                            0,
                        )
                    };
                    (k, v)
                }
                TalkerKvStorage::TurboQuantQ8_0 => {
                    let k_store = unsafe { sys::ggml_cont(ctx.as_ptr(), k_cur) };
                    let v_store = unsafe { sys::ggml_cont(ctx.as_ptr(), v_cur) };
                    unsafe {
                        sys::ggml_build_forward_expand(graph.as_ptr(), k_store);
                        sys::ggml_build_forward_expand(graph.as_ptr(), v_store);
                    }
                    let rows =
                        self.config.n_key_value_heads as usize * self.config.head_dim as usize;
                    kv_writeback.push(KvWritebackTensorDownloads {
                        layer_idx,
                        token_start: n_past,
                        n_tokens: 1,
                        k_tensor: k_store,
                        v_tensor: v_store,
                        k_data: vec![0.0; rows],
                        v_data: vec![0.0; rows],
                    });
                    if n_past == 0 {
                        (k_cur, v_cur)
                    } else {
                        let k_prefix_q = unsafe {
                            sys::ggml_view_3d(
                                ctx.as_ptr(),
                                k_cache,
                                self.config.head_dim as i64,
                                self.config.n_key_value_heads as i64,
                                n_past as i64,
                                (*k_cache).nb[1],
                                (*k_cache).nb[2],
                                0,
                            )
                        };
                        let v_prefix_q = unsafe {
                            sys::ggml_view_3d(
                                ctx.as_ptr(),
                                v_cache,
                                self.config.head_dim as i64,
                                self.config.n_key_value_heads as i64,
                                n_past as i64,
                                (*v_cache).nb[1],
                                (*v_cache).nb[2],
                                0,
                            )
                        };
                        let k_prefix = unsafe {
                            sys::ggml_cast(ctx.as_ptr(), k_prefix_q, sys::ggml_type_GGML_TYPE_F32)
                        };
                        let v_prefix = unsafe {
                            sys::ggml_cast(ctx.as_ptr(), v_prefix_q, sys::ggml_type_GGML_TYPE_F32)
                        };
                        let k = unsafe { sys::ggml_concat(ctx.as_ptr(), k_prefix, k_cur, 2) };
                        let v = unsafe { sys::ggml_concat(ctx.as_ptr(), v_prefix, v_cur, 2) };
                        (k, v)
                    }
                }
            };

            let q = unsafe { sys::ggml_permute(ctx.as_ptr(), q_cur, 0, 2, 1, 3) };
            k = unsafe { sys::ggml_permute(ctx.as_ptr(), k, 0, 2, 1, 3) };
            v = unsafe { sys::ggml_permute(ctx.as_ptr(), v, 0, 2, 1, 3) };

            let mut kq = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), k, q) };
            kq = unsafe { sys::ggml_scale(ctx.as_ptr(), kq, kq_scale) };
            kq = unsafe {
                ggml_soft_max_ext_with_diag_mask_cache(
                    ctx.as_ptr(),
                    kq,
                    n_past as i32,
                    &mut attn_softmax_mask,
                )
            };

            v = unsafe { sys::ggml_cont(ctx.as_ptr(), sys::ggml_transpose(ctx.as_ptr(), v)) };
            let mut kqv = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), v, kq) };
            kqv = unsafe { sys::ggml_permute(ctx.as_ptr(), kqv, 0, 2, 1, 3) };
            cur = unsafe {
                sys::ggml_cont_2d(
                    ctx.as_ptr(),
                    kqv,
                    (self.config.n_attention_heads * self.config.head_dim) as i64,
                    1,
                )
            };

            cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.attn_output.as_ptr(), cur) };
            cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_l) };
            let inp_ff = cur;

            cur = unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_ff, self.config.rms_norm_eps) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), cur, layer.ffn_norm.as_ptr()) };
            let mut gate = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.ffn_gate.as_ptr(), cur) };
            let up = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), layer.ffn_up.as_ptr(), cur) };
            gate = unsafe { sys::ggml_silu(ctx.as_ptr(), gate) };
            cur = unsafe { sys::ggml_mul(ctx.as_ptr(), gate, up) };

            let ffn_down_f32 = layer
                .ffn_down_f32
                .map(NonNull::as_ptr)
                .unwrap_or_else(|| unsafe {
                    sys::ggml_cast(
                        ctx.as_ptr(),
                        layer.ffn_down.as_ptr(),
                        sys::ggml_type_GGML_TYPE_F32,
                    )
                });
            cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), ffn_down_f32, cur) };
            inp_l = unsafe { sys::ggml_add(ctx.as_ptr(), cur, inp_ff) };
        }

        let mut hidden_state =
            unsafe { sys::ggml_rms_norm(ctx.as_ptr(), inp_l, self.config.rms_norm_eps) };
        hidden_state =
            unsafe { sys::ggml_mul(ctx.as_ptr(), hidden_state, self.talker.output_norm.as_ptr()) };
        hidden_state = unsafe { sys::ggml_cont(ctx.as_ptr(), hidden_state) };
        let mut logits = unsafe {
            sys::ggml_mul_mat(ctx.as_ptr(), self.talker.codec_head.as_ptr(), hidden_state)
        };
        logits = unsafe { sys::ggml_cont(ctx.as_ptr(), logits) };
        unsafe {
            sys::ggml_build_forward_expand(graph.as_ptr(), hidden_state);
            sys::ggml_build_forward_expand(graph.as_ptr(), logits);
        }

        let mut hidden = vec![0.0f32; hidden_size];
        let mut logits_data = vec![0.0f32; self.config.codec_vocab_size as usize];
        let mut uploads = vec![
            TensorUpload {
                tensor: inp_step.as_ptr(),
                bytes: slice_as_bytes(step_embd),
            },
            TensorUpload {
                tensor: inp_pos.as_ptr(),
                bytes: slice_as_bytes(std::slice::from_ref(&pos)),
            },
        ];
        if let Some((t, data)) = &attn_softmax_mask {
            uploads.push(TensorUpload {
                tensor: *t,
                bytes: slice_as_bytes(data.as_slice()),
            });
        }
        let mut downloads = vec![
            TensorDownload {
                tensor: hidden_state,
                bytes: slice_as_bytes_mut(hidden.as_mut_slice()),
            },
            TensorDownload {
                tensor: logits,
                bytes: slice_as_bytes_mut(logits_data.as_mut_slice()),
            },
        ];
        execute_graph(
            &cache._backends,
            graph,
            uploads.as_slice(),
            downloads.as_mut_slice(),
            thread_count,
            "step graph execution failed",
        )?;
        let (kv_writeback_elapsed, kv_download_elapsed, kv_quantize_elapsed, kv_upload_elapsed) =
            if cache.storage().is_quantized() {
                let t_writeback = Instant::now();
                let t_download = Instant::now();
                for pending in &mut kv_writeback {
                    unsafe {
                        sys::ggml_backend_tensor_get(
                            pending.k_tensor,
                            pending.k_data.as_mut_ptr().cast(),
                            0,
                            std::mem::size_of_val(pending.k_data.as_slice()),
                        );
                        sys::ggml_backend_tensor_get(
                            pending.v_tensor,
                            pending.v_data.as_mut_ptr().cast(),
                            0,
                            std::mem::size_of_val(pending.v_data.as_slice()),
                        );
                    }
                }
                let download_elapsed = t_download.elapsed();
                let mut quantize_elapsed = Duration::ZERO;
                let mut upload_elapsed = Duration::ZERO;
                let base_layout = QuantizedKvWriteLayout {
                    token_start: 0,
                    n_tokens: 0,
                    rows_per_token: self.config.n_key_value_heads as usize,
                    row_len: self.config.head_dim as usize,
                };
                for pending in kv_writeback {
                    let (quantize, upload) = cache.quantized_write_layer(
                        pending.layer_idx,
                        QuantizedKvWriteLayout {
                            token_start: pending.token_start,
                            n_tokens: pending.n_tokens,
                            ..base_layout
                        },
                        pending.k_data.as_slice(),
                        pending.v_data.as_slice(),
                    )?;
                    quantize_elapsed += quantize;
                    upload_elapsed += upload;
                }
                let total = t_writeback.elapsed();
                (total, download_elapsed, quantize_elapsed, upload_elapsed)
            } else {
                (
                    Duration::ZERO,
                    Duration::ZERO,
                    Duration::ZERO,
                    Duration::ZERO,
                )
            };
        Ok(StepForwardOutputs {
            hidden_state: hidden,
            logits: logits_data,
            kv_writeback_elapsed,
            kv_download_elapsed,
            kv_quantize_elapsed,
            kv_upload_elapsed,
        })
    }
}

/// Pre-built, pre-allocated step graph whose tensor shapes are independent of `n_past`.
///
/// Attention reads `K_cache[0..n_kv_max-1]` (the cache slots filled so far **plus** garbage
fn get_u32_any(file: &GgufFile, keys: &[&str], default: i32) -> i32 {
    for key in keys {
        if file.key_index(key).is_some() {
            return file.get_u32(key, default as u32) as i32;
        }
    }
    default
}

fn get_f32_any(file: &GgufFile, keys: &[&str], default: f32) -> f32 {
    for key in keys {
        if file.key_index(key).is_some() {
            return file.get_f32(key, default);
        }
    }
    default
}

struct TalkerWeights {
    _ctx: OwnedContext,
    _backends: BackendSet,
    _buffer: OwnedBuffer,
    text_embd: NonNull<sys::ggml_tensor>,
    text_proj_fc1: NonNull<sys::ggml_tensor>,
    text_proj_fc1_bias: NonNull<sys::ggml_tensor>,
    text_proj_fc2: NonNull<sys::ggml_tensor>,
    text_proj_fc2_bias: NonNull<sys::ggml_tensor>,
    codec_embd: NonNull<sys::ggml_tensor>,
    /// Row-major `[codec_vocab_size * hidden_size]` when GGUF stores `codec_embd` as F16/F32.
    codec_embd_cpu: Option<Vec<f32>>,
    output_norm: NonNull<sys::ggml_tensor>,
    codec_head: NonNull<sys::ggml_tensor>,
    layers: Vec<TalkerLayerWeights>,
}

impl TalkerWeights {
    fn load(
        file: &GgufFile,
        cfg: &TtsTransformerConfig,
        backends: BackendSet,
    ) -> Result<Self, Qwen3TtsError> {
        unsafe {
            sys::ggml_cpu_init();
        }
        let tensor_count = 8 + cfg.n_layers as usize * 12;
        let ctx = OwnedContext::new_for_tensor_metadata(tensor_count)?;

        let text_embd = load_tensor_into_context(file, ctx.as_ptr(), "talker.text_embd.weight")?;
        let text_proj_fc1 =
            load_tensor_into_context(file, ctx.as_ptr(), "talker.text_proj.fc1.weight")?;
        let text_proj_fc1_bias =
            load_tensor_into_context(file, ctx.as_ptr(), "talker.text_proj.fc1.bias")?;
        let text_proj_fc2 =
            load_tensor_into_context(file, ctx.as_ptr(), "talker.text_proj.fc2.weight")?;
        let text_proj_fc2_bias =
            load_tensor_into_context(file, ctx.as_ptr(), "talker.text_proj.fc2.bias")?;
        let codec_embd = load_tensor_into_context(file, ctx.as_ptr(), "talker.codec_embd.weight")?;
        let hidden_u = cfg.hidden_size as usize;
        let codec_vocab_u = cfg.codec_vocab_size as usize;
        let codec_embd_cpu = try_read_embedding_matrix_f32(
            file,
            "talker.codec_embd.weight",
            codec_embd,
            hidden_u,
            codec_vocab_u,
        );
        let output_norm =
            load_tensor_into_context(file, ctx.as_ptr(), "talker.output_norm.weight")?;
        let codec_head = load_tensor_into_context(file, ctx.as_ptr(), "talker.codec_head.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for layer_idx in 0..cfg.n_layers {
            let prefix = format!("talker.blk.{layer_idx}.");
            let ffn_down_name = prefix.clone() + "ffn_down.weight";
            layers.push(TalkerLayerWeights {
                attn_norm: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_norm.weight"),
                )?,
                attn_q_norm: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_q_norm.weight"),
                )?,
                attn_k_norm: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_k_norm.weight"),
                )?,
                attn_q: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_q.weight"),
                )?,
                attn_k: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_k.weight"),
                )?,
                attn_v: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_v.weight"),
                )?,
                attn_output: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_output.weight"),
                )?,
                ffn_norm: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "ffn_norm.weight"),
                )?,
                ffn_gate: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "ffn_gate.weight"),
                )?,
                ffn_up: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "ffn_up.weight"),
                )?,
                ffn_down: load_tensor_into_context(file, ctx.as_ptr(), &ffn_down_name)?,
                ffn_down_f32: load_tensor_f32_into_context(file, ctx.as_ptr(), &ffn_down_name)?,
            });
        }

        let buffer = OwnedBuffer::alloc(ctx.as_ptr(), backends.primary_ptr())?;
        for (name, tensor) in [
            ("talker.text_embd.weight", text_embd),
            ("talker.text_proj.fc1.weight", text_proj_fc1),
            ("talker.text_proj.fc1.bias", text_proj_fc1_bias),
            ("talker.text_proj.fc2.weight", text_proj_fc2),
            ("talker.text_proj.fc2.bias", text_proj_fc2_bias),
            ("talker.codec_embd.weight", codec_embd),
            ("talker.output_norm.weight", output_norm),
            ("talker.codec_head.weight", codec_head),
        ] {
            let (_, raw) = file.read_tensor_bytes(name)?;
            unsafe {
                sys::ggml_backend_tensor_set(tensor.as_ptr(), raw.as_ptr().cast(), 0, raw.len());
            }
        }
        for (layer_idx, layer) in layers.iter().enumerate() {
            for (suffix, tensor) in [
                ("attn_norm.weight", Some(layer.attn_norm)),
                ("attn_q_norm.weight", layer.attn_q_norm),
                ("attn_k_norm.weight", layer.attn_k_norm),
                ("attn_q.weight", Some(layer.attn_q)),
                ("attn_k.weight", Some(layer.attn_k)),
                ("attn_v.weight", Some(layer.attn_v)),
                ("attn_output.weight", Some(layer.attn_output)),
                ("ffn_norm.weight", Some(layer.ffn_norm)),
                ("ffn_gate.weight", Some(layer.ffn_gate)),
                ("ffn_up.weight", Some(layer.ffn_up)),
                ("ffn_down.weight", Some(layer.ffn_down)),
            ] {
                if let Some(tensor) = tensor {
                    let name = format!("talker.blk.{layer_idx}.{suffix}");
                    let (_, raw) = file.read_tensor_bytes(&name)?;
                    unsafe {
                        sys::ggml_backend_tensor_set(
                            tensor.as_ptr(),
                            raw.as_ptr().cast(),
                            0,
                            raw.len(),
                        );
                    }
                }
            }
            if let Some(tensor) = layer.ffn_down_f32 {
                let name = format!("talker.blk.{layer_idx}.ffn_down.weight");
                let (_, raw) = file.read_tensor_f32(&name)?;
                unsafe {
                    sys::ggml_backend_tensor_set(
                        tensor.as_ptr(),
                        raw.as_ptr().cast(),
                        0,
                        std::mem::size_of_val(raw.as_slice()),
                    );
                }
            }
        }

        Ok(Self {
            _ctx: ctx,
            _backends: backends,
            _buffer: buffer,
            text_embd,
            text_proj_fc1,
            text_proj_fc1_bias,
            text_proj_fc2,
            text_proj_fc2_bias,
            codec_embd,
            codec_embd_cpu,
            output_norm,
            codec_head,
            layers,
        })
    }
}

struct TalkerLayerWeights {
    attn_norm: NonNull<sys::ggml_tensor>,
    attn_q_norm: Option<NonNull<sys::ggml_tensor>>,
    attn_k_norm: Option<NonNull<sys::ggml_tensor>>,
    attn_q: NonNull<sys::ggml_tensor>,
    attn_k: NonNull<sys::ggml_tensor>,
    attn_v: NonNull<sys::ggml_tensor>,
    attn_output: NonNull<sys::ggml_tensor>,
    ffn_norm: NonNull<sys::ggml_tensor>,
    ffn_gate: NonNull<sys::ggml_tensor>,
    ffn_up: NonNull<sys::ggml_tensor>,
    ffn_down: NonNull<sys::ggml_tensor>,
    ffn_down_f32: Option<NonNull<sys::ggml_tensor>>,
}

struct CodePredWeights {
    _ctx: OwnedContext,
    _backends: BackendSet,
    _buffer: OwnedBuffer,
    embeddings: Vec<NonNull<sys::ggml_tensor>>,
    /// Per codebook row-major `[code_pred_vocab_size * hidden_size]`, only when all tables are F16/F32 in GGUF.
    codec_embd_cpu: Option<Vec<Vec<f32>>>,
    output_norm: NonNull<sys::ggml_tensor>,
    heads: Vec<NonNull<sys::ggml_tensor>>,
    layers: Vec<TalkerLayerWeights>,
}

struct TalkerKvCache {
    _ctx: OwnedContext,
    _backends: BackendSet,
    _buffer: OwnedBuffer,
    k_cache: Vec<NonNull<sys::ggml_tensor>>,
    v_cache: Vec<NonNull<sys::ggml_tensor>>,
    storage: TalkerKvStorage,
    n_ctx: usize,
}

#[derive(Debug, Clone, Copy)]
struct QuantizedKvWriteLayout {
    token_start: usize,
    n_tokens: usize,
    rows_per_token: usize,
    row_len: usize,
}

struct CodePredKvCache {
    _ctx: OwnedContext,
    _backends: BackendSet,
    _buffer: OwnedBuffer,
    k_cache: Vec<NonNull<sys::ggml_tensor>>,
    v_cache: Vec<NonNull<sys::ggml_tensor>>,
    n_ctx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TalkerKvStorage {
    F16,
    TurboQuantQ8_0,
}

impl TalkerKvStorage {
    fn from_mode(mode: TalkerKvMode) -> Self {
        match mode {
            TalkerKvMode::F16 => Self::F16,
            TalkerKvMode::TurboQuant => Self::TurboQuantQ8_0,
        }
    }

    fn tensor_type(self) -> sys::ggml_type {
        match self {
            Self::F16 => sys::ggml_type_GGML_TYPE_F16,
            Self::TurboQuantQ8_0 => sys::ggml_type_GGML_TYPE_Q8_0,
        }
    }

    fn is_quantized(self) -> bool {
        matches!(self, Self::TurboQuantQ8_0)
    }
}

impl TalkerKvCache {
    fn new(
        cfg: &TtsTransformerConfig,
        n_ctx: usize,
        backends: BackendSet,
        mode: TalkerKvMode,
    ) -> Result<Self, Qwen3TtsError> {
        unsafe {
            sys::ggml_cpu_init();
        }
        let storage = TalkerKvStorage::from_mode(mode);
        let ctx = OwnedContext::new_for_tensor_metadata(cfg.n_layers as usize * 2)?;
        let mut k_cache = Vec::with_capacity(cfg.n_layers as usize);
        let mut v_cache = Vec::with_capacity(cfg.n_layers as usize);
        for _ in 0..cfg.n_layers {
            let k = unsafe {
                sys::ggml_new_tensor_3d(
                    ctx.as_ptr(),
                    storage.tensor_type(),
                    cfg.head_dim as i64,
                    cfg.n_key_value_heads as i64,
                    n_ctx as i64,
                )
            };
            let v = unsafe {
                sys::ggml_new_tensor_3d(
                    ctx.as_ptr(),
                    storage.tensor_type(),
                    cfg.head_dim as i64,
                    cfg.n_key_value_heads as i64,
                    n_ctx as i64,
                )
            };
            k_cache.push(
                NonNull::new(k).ok_or_else(|| {
                    Qwen3TtsError::InvalidInput("failed to allocate K cache".into())
                })?,
            );
            v_cache.push(
                NonNull::new(v).ok_or_else(|| {
                    Qwen3TtsError::InvalidInput("failed to allocate V cache".into())
                })?,
            );
        }
        let buffer = OwnedBuffer::alloc(ctx.as_ptr(), backends.primary_ptr())?;
        unsafe {
            let nbytes = sys::ggml_nbytes(k_cache[0].as_ptr());
            let zeros = vec![0u8; nbytes];
            for k in &k_cache {
                sys::ggml_backend_tensor_set(k.as_ptr(), zeros.as_ptr().cast(), 0, nbytes);
            }
            for v in &v_cache {
                sys::ggml_backend_tensor_set(v.as_ptr(), zeros.as_ptr().cast(), 0, nbytes);
            }
        }
        Ok(Self {
            _ctx: ctx,
            _backends: backends,
            _buffer: buffer,
            k_cache,
            v_cache,
            storage,
            n_ctx,
        })
    }

    fn storage(&self) -> TalkerKvStorage {
        self.storage
    }

    fn total_bytes(&self) -> usize {
        let per_tensor = self
            .k_cache
            .first()
            .map(|tensor| unsafe { sys::ggml_nbytes(tensor.as_ptr()) })
            .unwrap_or(0);
        per_tensor.saturating_mul(self.k_cache.len() + self.v_cache.len())
    }

    fn quantized_write_layer(
        &self,
        layer_idx: usize,
        layout: QuantizedKvWriteLayout,
        k_src: &[f32],
        v_src: &[f32],
    ) -> Result<(Duration, Duration), Qwen3TtsError> {
        debug_assert!(self.storage.is_quantized());
        let k = self.k_cache.get(layer_idx).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("talker K cache layer out of range".into())
        })?;
        let v = self.v_cache.get(layer_idx).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("talker V cache layer out of range".into())
        })?;
        let (k_quantize, k_upload) =
            quantized_tensor_write_rows(k.as_ptr(), self.storage.tensor_type(), layout, k_src)?;
        let (v_quantize, v_upload) =
            quantized_tensor_write_rows(v.as_ptr(), self.storage.tensor_type(), layout, v_src)?;
        Ok((k_quantize + v_quantize, k_upload + v_upload))
    }
}

fn quantized_tensor_write_rows(
    tensor: *mut sys::ggml_tensor,
    quant_type: sys::ggml_type,
    layout: QuantizedKvWriteLayout,
    src: &[f32],
) -> Result<(Duration, Duration), Qwen3TtsError> {
    let QuantizedKvWriteLayout {
        token_start,
        n_tokens,
        rows_per_token,
        row_len,
    } = layout;
    if src.len()
        != n_tokens
            .saturating_mul(rows_per_token)
            .saturating_mul(row_len)
    {
        return Err(Qwen3TtsError::InvalidInput(
            "quantized KV source rows had an unexpected size".into(),
        ));
    }
    let block_size = unsafe { sys::ggml_blck_size(quant_type) as usize };
    if block_size == 0 || !row_len.is_multiple_of(block_size) {
        return Err(Qwen3TtsError::InvalidInput(format!(
            "quantized KV row length {row_len} is incompatible with block size {block_size}"
        )));
    }
    let type_size = unsafe { sys::ggml_type_size(quant_type) };
    let bytes_per_row = row_len / block_size * type_size;
    let total_bytes = n_tokens
        .saturating_mul(rows_per_token)
        .saturating_mul(bytes_per_row);
    let mut quantized = vec![0u8; total_bytes];
    let t_quantize = Instant::now();
    let written = unsafe {
        sys::ggml_quantize_chunk(
            quant_type,
            src.as_ptr(),
            quantized.as_mut_ptr().cast(),
            0,
            (n_tokens * rows_per_token) as i64,
            row_len as i64,
            std::ptr::null(),
        )
    };
    let quantize_elapsed = t_quantize.elapsed();
    if written != quantized.len() {
        return Err(Qwen3TtsError::InvalidInput(format!(
            "quantized KV write produced {written} bytes, expected {}",
            quantized.len()
        )));
    }
    let offset = unsafe { token_start * (*tensor).nb[2] };
    let t_upload = Instant::now();
    unsafe {
        sys::ggml_backend_tensor_set(tensor, quantized.as_ptr().cast(), offset, quantized.len());
    }
    Ok((quantize_elapsed, t_upload.elapsed()))
}

impl CodePredKvCache {
    fn new(
        cfg: &TtsTransformerConfig,
        n_ctx: usize,
        backends: BackendSet,
    ) -> Result<Self, Qwen3TtsError> {
        unsafe {
            sys::ggml_cpu_init();
        }
        let ctx = OwnedContext::new_for_tensor_metadata(cfg.code_pred_layers as usize * 2)?;
        let mut k_cache = Vec::with_capacity(cfg.code_pred_layers as usize);
        let mut v_cache = Vec::with_capacity(cfg.code_pred_layers as usize);
        for _ in 0..cfg.code_pred_layers {
            let k = unsafe {
                sys::ggml_new_tensor_3d(
                    ctx.as_ptr(),
                    sys::ggml_type_GGML_TYPE_F16,
                    cfg.head_dim as i64,
                    cfg.n_key_value_heads as i64,
                    n_ctx as i64,
                )
            };
            let v = unsafe {
                sys::ggml_new_tensor_3d(
                    ctx.as_ptr(),
                    sys::ggml_type_GGML_TYPE_F16,
                    cfg.head_dim as i64,
                    cfg.n_key_value_heads as i64,
                    n_ctx as i64,
                )
            };
            k_cache.push(NonNull::new(k).ok_or_else(|| {
                Qwen3TtsError::InvalidInput("failed to allocate code predictor K cache".into())
            })?);
            v_cache.push(NonNull::new(v).ok_or_else(|| {
                Qwen3TtsError::InvalidInput("failed to allocate code predictor V cache".into())
            })?);
        }
        let buffer = OwnedBuffer::alloc(ctx.as_ptr(), backends.primary_ptr())?;
        Ok(Self {
            _ctx: ctx,
            _backends: backends,
            _buffer: buffer,
            k_cache,
            v_cache,
            n_ctx,
        })
    }
}

impl CodePredWeights {
    fn load(
        file: &GgufFile,
        cfg: &TtsTransformerConfig,
        backends: BackendSet,
    ) -> Result<Self, Qwen3TtsError> {
        unsafe {
            sys::ggml_cpu_init();
        }
        let per_codebook = (cfg.n_codebooks - 1) as usize;
        let tensor_count = 1 + per_codebook * 2 + cfg.code_pred_layers as usize * 12;
        let ctx = OwnedContext::new_for_tensor_metadata(tensor_count)?;

        let mut embeddings = Vec::with_capacity(per_codebook);
        let mut heads = Vec::with_capacity(per_codebook);
        for cb_idx in 0..per_codebook {
            embeddings.push(load_tensor_into_context(
                file,
                ctx.as_ptr(),
                &format!("code_pred.codec_embd.{cb_idx}.weight"),
            )?);
            heads.push(load_tensor_into_context(
                file,
                ctx.as_ptr(),
                &format!("code_pred.lm_head.{cb_idx}.weight"),
            )?);
        }
        let hidden_u = cfg.hidden_size as usize;
        let pred_vocab_u = cfg.code_pred_vocab_size as usize;
        let mut codec_embd_cpu: Option<Vec<Vec<f32>>> = Some(Vec::with_capacity(per_codebook));
        for (cb_idx, embedding) in embeddings.iter().enumerate().take(per_codebook) {
            let name = format!("code_pred.codec_embd.{cb_idx}.weight");
            let row =
                try_read_embedding_matrix_f32(file, &name, *embedding, hidden_u, pred_vocab_u);
            match (&mut codec_embd_cpu, row) {
                (Some(rows), Some(data)) => rows.push(data),
                _ => {
                    codec_embd_cpu = None;
                    break;
                }
            }
        }
        if codec_embd_cpu.as_ref().map(|v| v.len()) != Some(per_codebook) {
            codec_embd_cpu = None;
        }
        let output_norm =
            load_tensor_into_context(file, ctx.as_ptr(), "code_pred.output_norm.weight")?;
        let mut layers = Vec::with_capacity(cfg.code_pred_layers as usize);
        for layer_idx in 0..cfg.code_pred_layers {
            let prefix = format!("code_pred.blk.{layer_idx}.");
            let ffn_down_name = prefix.clone() + "ffn_down.weight";
            layers.push(TalkerLayerWeights {
                attn_norm: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_norm.weight"),
                )?,
                attn_q_norm: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_q_norm.weight"),
                )?,
                attn_k_norm: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_k_norm.weight"),
                )?,
                attn_q: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_q.weight"),
                )?,
                attn_k: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_k.weight"),
                )?,
                attn_v: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_v.weight"),
                )?,
                attn_output: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_output.weight"),
                )?,
                ffn_norm: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "ffn_norm.weight"),
                )?,
                ffn_gate: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "ffn_gate.weight"),
                )?,
                ffn_up: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "ffn_up.weight"),
                )?,
                ffn_down: load_tensor_into_context(file, ctx.as_ptr(), &ffn_down_name)?,
                ffn_down_f32: load_tensor_f32_into_context(file, ctx.as_ptr(), &ffn_down_name)?,
            });
        }

        let buffer = OwnedBuffer::alloc(ctx.as_ptr(), backends.primary_ptr())?;
        for cb_idx in 0..per_codebook {
            for name in [
                format!("code_pred.codec_embd.{cb_idx}.weight"),
                format!("code_pred.lm_head.{cb_idx}.weight"),
            ] {
                let tensor = if name.contains("codec_embd") {
                    embeddings[cb_idx]
                } else {
                    heads[cb_idx]
                };
                let (_, raw) = file.read_tensor_bytes(&name)?;
                unsafe {
                    sys::ggml_backend_tensor_set(
                        tensor.as_ptr(),
                        raw.as_ptr().cast(),
                        0,
                        raw.len(),
                    );
                }
            }
        }
        let (_, raw) = file.read_tensor_bytes("code_pred.output_norm.weight")?;
        unsafe {
            sys::ggml_backend_tensor_set(output_norm.as_ptr(), raw.as_ptr().cast(), 0, raw.len());
        }
        for (layer_idx, layer) in layers.iter().enumerate() {
            for (suffix, tensor) in [
                ("attn_norm.weight", Some(layer.attn_norm)),
                ("attn_q_norm.weight", layer.attn_q_norm),
                ("attn_k_norm.weight", layer.attn_k_norm),
                ("attn_q.weight", Some(layer.attn_q)),
                ("attn_k.weight", Some(layer.attn_k)),
                ("attn_v.weight", Some(layer.attn_v)),
                ("attn_output.weight", Some(layer.attn_output)),
                ("ffn_norm.weight", Some(layer.ffn_norm)),
                ("ffn_gate.weight", Some(layer.ffn_gate)),
                ("ffn_up.weight", Some(layer.ffn_up)),
                ("ffn_down.weight", Some(layer.ffn_down)),
            ] {
                if let Some(tensor) = tensor {
                    let name = format!("code_pred.blk.{layer_idx}.{suffix}");
                    let (_, raw) = file.read_tensor_bytes(&name)?;
                    unsafe {
                        sys::ggml_backend_tensor_set(
                            tensor.as_ptr(),
                            raw.as_ptr().cast(),
                            0,
                            raw.len(),
                        );
                    }
                }
            }
            if let Some(tensor) = layer.ffn_down_f32 {
                let name = format!("code_pred.blk.{layer_idx}.ffn_down.weight");
                let (_, raw) = file.read_tensor_f32(&name)?;
                unsafe {
                    sys::ggml_backend_tensor_set(
                        tensor.as_ptr(),
                        raw.as_ptr().cast(),
                        0,
                        std::mem::size_of_val(raw.as_slice()),
                    );
                }
            }
        }

        Ok(Self {
            _ctx: ctx,
            _backends: backends,
            _buffer: buffer,
            embeddings,
            codec_embd_cpu,
            output_norm,
            heads,
            layers,
        })
    }
}

struct OwnedContext {
    raw: NonNull<sys::ggml_context>,
}

impl OwnedContext {
    fn new(mem_size: usize, no_alloc: bool) -> Result<Self, Qwen3TtsError> {
        let raw = unsafe {
            sys::ggml_init(sys::ggml_init_params {
                mem_size,
                mem_buffer: std::ptr::null_mut(),
                no_alloc,
            })
        };
        let raw = NonNull::new(raw).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to initialize ggml context".into())
        })?;
        Ok(Self { raw })
    }

    fn new_for_tensor_metadata(n_tensors: usize) -> Result<Self, Qwen3TtsError> {
        Self::new(
            max(1, n_tensors) * unsafe { sys::ggml_tensor_overhead() },
            true,
        )
    }

    fn as_ptr(&self) -> *mut sys::ggml_context {
        self.raw.as_ptr()
    }
}

impl Drop for OwnedContext {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_free(self.raw.as_ptr());
        }
    }
}

struct ComputeContext(OwnedContext);

impl ComputeContext {
    fn new_graph(max_nodes: usize) -> Result<Self, Qwen3TtsError> {
        Ok(Self(OwnedContext::new(
            graph_metadata_mem_size(max_nodes),
            true,
        )?))
    }

    fn as_ptr(&self) -> *mut sys::ggml_context {
        self.0.as_ptr()
    }
}

/// When the GGUF tensor is F32/F16 and shaped with `ne[0] == hidden`, `ne[1] == vocab`,
/// return row-major `f32` weights in the same layout as `ggml_get_rows` on that tensor.
fn try_read_embedding_matrix_f32(
    file: &GgufFile,
    name: &str,
    tensor: NonNull<sys::ggml_tensor>,
    hidden: usize,
    vocab: usize,
) -> Option<Vec<f32>> {
    let n0 = unsafe { (*tensor.as_ptr()).ne[0] as usize };
    let n1 = unsafe { (*tensor.as_ptr()).ne[1] as usize };
    if n0 != hidden || n1 != vocab {
        return None;
    }
    let (_, data) = file.read_tensor_f32(name).ok()?;
    (data.len() == hidden * vocab).then_some(data)
}

fn load_tensor_into_context(
    file: &GgufFile,
    ctx: *mut sys::ggml_context,
    name: &str,
) -> Result<NonNull<sys::ggml_tensor>, Qwen3TtsError> {
    let info = file.tensor_info(name)?;
    let mut ne = [1i64; 4];
    for (idx, dim) in info.dims.iter().copied().enumerate() {
        ne[idx] = dim as i64;
    }
    let tensor = unsafe { sys::ggml_new_tensor(ctx, info.ty, info.dims.len() as i32, ne.as_ptr()) };
    NonNull::new(tensor).ok_or_else(|| Qwen3TtsError::InvalidTensor(name.into()))
}

fn load_tensor_f32_into_context(
    file: &GgufFile,
    ctx: *mut sys::ggml_context,
    name: &str,
) -> Result<Option<NonNull<sys::ggml_tensor>>, Qwen3TtsError> {
    let info = match file.tensor_info(name) {
        Ok(info) => info,
        Err(Qwen3TtsError::MissingTensor(_)) => return Ok(None),
        Err(err) => return Err(err),
    };
    // Decide availability based on metadata only: this helper is for tensors
    // that are actually stored as F32 in the GGUF file.
    if info.ty != sys::ggml_type_GGML_TYPE_F32 {
        return Ok(None);
    }
    let mut ne = [1i64; 4];
    for (idx, dim) in info.dims.iter().copied().enumerate() {
        ne[idx] = dim as i64;
    }
    let tensor = unsafe {
        sys::ggml_new_tensor(
            ctx,
            sys::ggml_type_GGML_TYPE_F32,
            info.dims.len() as i32,
            ne.as_ptr(),
        )
    };
    Ok(NonNull::new(tensor))
}

fn load_optional_tensor_into_context(
    file: &GgufFile,
    ctx: *mut sys::ggml_context,
    name: &str,
) -> Result<Option<NonNull<sys::ggml_tensor>>, Qwen3TtsError> {
    match file.tensor_info(name) {
        Ok(info) => {
            let mut ne = [1i64; 4];
            for (idx, dim) in info.dims.iter().copied().enumerate() {
                ne[idx] = dim as i64;
            }
            let tensor =
                unsafe { sys::ggml_new_tensor(ctx, info.ty, info.dims.len() as i32, ne.as_ptr()) };
            Ok(NonNull::new(tensor))
        }
        Err(Qwen3TtsError::MissingTensor(_)) => Ok(None),
        Err(err) => Err(err),
    }
}

fn select_token(
    logits: &[f32],
    repetition_penalty: f32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    recent_tokens: &[i32],
) -> Result<i32, Qwen3TtsError> {
    if logits.is_empty() {
        return Err(Qwen3TtsError::InvalidInput(
            "logits must not be empty".into(),
        ));
    }

    let mut adjusted = logits.to_vec();
    if repetition_penalty != 1.0 {
        let mut unique_tokens = recent_tokens.to_vec();
        unique_tokens.sort_unstable();
        unique_tokens.dedup();
        for token in unique_tokens {
            let idx = token as usize;
            if idx >= adjusted.len() {
                continue;
            }
            if adjusted[idx] > 0.0 {
                adjusted[idx] /= repetition_penalty;
            } else {
                adjusted[idx] *= repetition_penalty;
            }
        }
    }

    if temperature <= 0.0 {
        let (best_idx, best_value) = adjusted
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or_else(|| Qwen3TtsError::InvalidInput("failed to select token".into()))?;
        if !best_value.is_finite() {
            return Err(Qwen3TtsError::InvalidInput(
                "all candidate logits were filtered out".into(),
            ));
        }
        return Ok(best_idx as i32);
    }

    if temperature > 0.0 {
        for logit in &mut adjusted {
            *logit /= temperature;
        }
    }

    if top_k > 0 && (top_k as usize) < adjusted.len() {
        let mut ranked = adjusted
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<(usize, f32)>>();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let threshold = ranked[top_k as usize - 1].1;
        for logit in &mut adjusted {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    if top_p.is_finite() && top_p > 0.0 && top_p < 1.0 {
        let mut ranked = adjusted
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<(usize, f32)>>();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let max_logit = ranked
            .iter()
            .map(|(_, logit)| *logit)
            .find(|logit| logit.is_finite())
            .ok_or_else(|| Qwen3TtsError::InvalidInput("all logits are filtered out".into()))?;

        let exp_values = ranked
            .iter()
            .map(|(_, logit)| {
                if logit.is_finite() {
                    (logit - max_logit).exp()
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let total = exp_values.iter().sum::<f32>();
        if total <= 0.0 {
            return Err(Qwen3TtsError::InvalidInput(
                "all logits are filtered out".into(),
            ));
        }
        let mut cumulative = 0.0f32;
        let mut keep = vec![false; adjusted.len()];
        for ((idx, logit), prob) in ranked.into_iter().zip(exp_values.into_iter()) {
            if !logit.is_finite() {
                break;
            }
            cumulative += prob / total;
            keep[idx] = true;
            if cumulative >= top_p {
                break;
            }
        }

        if keep.iter().any(|value| *value) {
            for (idx, logit) in adjusted.iter_mut().enumerate() {
                if !keep[idx] {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
    }

    let mut ranked = adjusted
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .collect::<Vec<_>>();
    if ranked.is_empty() {
        return Err(Qwen3TtsError::InvalidInput(
            "all candidate logits were filtered out".into(),
        ));
    }
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let max_logit = ranked[0].1;
    let probs = ranked
        .iter()
        .map(|(_, logit)| (logit - max_logit).exp())
        .collect::<Vec<_>>();
    let total = probs.iter().sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return Err(Qwen3TtsError::InvalidInput(
            "invalid probability mass during sampling".into(),
        ));
    }
    let mut target = rand::rng().random::<f32>() * total;
    for ((idx, _), prob) in ranked.into_iter().zip(probs.into_iter()) {
        target -= prob;
        if target <= 0.0 {
            return Ok(idx as i32);
        }
    }

    adjusted
        .iter()
        .copied()
        .enumerate()
        .rfind(|(_, logit)| logit.is_finite())
        .map(|(idx, _)| idx as i32)
        .ok_or_else(|| Qwen3TtsError::InvalidInput("failed to sample token".into()))
}

#[cfg(test)]
mod tests {
    use super::select_token;

    #[test]
    fn select_token_prefers_best_logit() {
        let token = select_token(&[0.1, 2.0, 1.0], 1.0, 0.0, 0, 1.0, &[]).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn select_token_applies_repetition_penalty() {
        let token = select_token(&[5.0, 4.0], 2.0, 0.0, 0, 1.0, &[0]).unwrap();
        assert_eq!(token, 1);
    }
}
