use std::collections::{BTreeSet, HashMap};
use std::fs;

use serde::{Deserialize, Deserializer};

use crate::{ModelPaths, Qwen3TtsError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceModelKind {
    Base,
    CustomVoice,
    VoiceDesign,
}

impl VoiceModelKind {
    pub fn load(paths: &ModelPaths) -> Result<Self, Qwen3TtsError> {
        if !paths.config_json.is_file() {
            return Ok(Self::Base);
        }

        let config_text = fs::read_to_string(&paths.config_json)?;
        let parsed: Qwen3TtsConfig = serde_json::from_str(&config_text).map_err(|err| {
            Qwen3TtsError::InvalidInput(format!(
                "failed to parse {}: {err}",
                paths.config_json.display()
            ))
        })?;

        Ok(match parsed.tts_model_type.as_deref() {
            Some("custom_voice") => Self::CustomVoice,
            Some("voice_design") => Self::VoiceDesign,
            _ => Self::Base,
        })
    }

    #[must_use]
    pub fn is_custom_voice(self) -> bool {
        matches!(self, Self::CustomVoice)
    }

    #[must_use]
    pub fn is_voice_design(self) -> bool {
        matches!(self, Self::VoiceDesign)
    }
}

#[derive(Debug, Clone)]
pub struct CustomVoiceMetadata {
    speaker_ids: HashMap<String, i32>,
    speaker_dialects: HashMap<String, String>,
    codec_language_ids: HashMap<String, i32>,
    supported_speakers: Vec<String>,
}

impl CustomVoiceMetadata {
    pub fn load(paths: &ModelPaths) -> Result<Option<Self>, Qwen3TtsError> {
        if !paths.config_json.is_file() {
            return Ok(None);
        }

        let config_text = fs::read_to_string(&paths.config_json)?;
        let parsed: Qwen3TtsConfig = serde_json::from_str(&config_text).map_err(|err| {
            Qwen3TtsError::InvalidInput(format!(
                "failed to parse {}: {err}",
                paths.config_json.display()
            ))
        })?;

        if parsed.tts_model_type.as_deref() != Some("custom_voice") {
            return Ok(None);
        }

        let talker = parsed.talker_config.ok_or_else(|| {
            Qwen3TtsError::InvalidInput(format!(
                "custom voice config missing talker_config: {}",
                paths.config_json.display()
            ))
        })?;

        let mut speaker_ids = HashMap::new();
        let mut supported_speakers = BTreeSet::new();
        for (speaker, token_id) in talker.spk_id {
            let normalized = speaker.trim().to_ascii_lowercase();
            if normalized.is_empty() {
                continue;
            }
            speaker_ids.insert(normalized.clone(), token_id);
            supported_speakers.insert(normalized);
        }

        let mut speaker_dialects = HashMap::new();
        for (speaker, dialect) in talker.spk_is_dialect {
            let Some(dialect_name) = dialect.into_dialect_name() else {
                continue;
            };
            let normalized = speaker.trim().to_ascii_lowercase();
            if normalized.is_empty() {
                continue;
            }
            speaker_dialects.insert(normalized, dialect_name.to_ascii_lowercase());
        }

        let codec_language_ids = talker
            .codec_language_id
            .into_iter()
            .map(|(name, id)| (name.trim().to_ascii_lowercase(), id))
            .collect::<HashMap<_, _>>();

        Ok(Some(Self {
            speaker_ids,
            speaker_dialects,
            codec_language_ids,
            supported_speakers: supported_speakers.into_iter().collect(),
        }))
    }

    pub fn speaker_token_id(&self, speaker: &str) -> Result<i32, Qwen3TtsError> {
        let normalized = speaker.trim().to_ascii_lowercase();
        self.speaker_ids.get(&normalized).copied().ok_or_else(|| {
            Qwen3TtsError::InvalidInput(format!(
                "unsupported custom voice speaker '{speaker}' (supported: {})",
                self.supported_speakers.join(", ")
            ))
        })
    }

    pub fn resolve_language_id(&self, requested_language_id: i32, speaker: &str) -> i32 {
        let Some(chinese_language_id) = self.codec_language_ids.get("chinese").copied() else {
            return requested_language_id;
        };
        if requested_language_id != chinese_language_id {
            return requested_language_id;
        }

        let normalized = speaker.trim().to_ascii_lowercase();
        let Some(dialect_name) = self.speaker_dialects.get(&normalized) else {
            return requested_language_id;
        };
        self.codec_language_ids
            .get(dialect_name)
            .copied()
            .unwrap_or(requested_language_id)
    }

    #[must_use]
    pub fn supported_speakers(&self) -> &[String] {
        &self.supported_speakers
    }
}

#[derive(Debug, Deserialize)]
struct Qwen3TtsConfig {
    #[serde(default)]
    tts_model_type: Option<String>,
    #[serde(default)]
    talker_config: Option<TalkerConfig>,
}

#[derive(Debug, Deserialize)]
struct TalkerConfig {
    #[serde(default)]
    spk_id: HashMap<String, i32>,
    #[serde(default)]
    spk_is_dialect: HashMap<String, DialectValue>,
    #[serde(default)]
    codec_language_id: HashMap<String, i32>,
}

#[derive(Debug)]
enum DialectValue {
    Disabled,
    DialectName(String),
}

impl<'de> Deserialize<'de> for DialectValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::Bool(_) | serde_json::Value::Null => Ok(Self::Disabled),
            serde_json::Value::String(text) => Ok(Self::DialectName(text)),
            other => Err(serde::de::Error::custom(format!(
                "unsupported dialect value: {other}"
            ))),
        }
    }
}

impl DialectValue {
    fn into_dialect_name(self) -> Option<String> {
        match self {
            Self::Disabled => None,
            Self::DialectName(value) if !value.trim().is_empty() => Some(value),
            Self::DialectName(_) => None,
        }
    }
}
