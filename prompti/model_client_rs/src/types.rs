use serde::{Deserialize, Serialize};
use std::fmt;

/// Model identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(String);

impl ModelId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ModelId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for ModelId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}

/// Provider identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderId(String);

impl ProviderId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ProviderId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for ProviderId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}

/// Predefined provider IDs
impl ProviderId {
    pub const OPENAI: &'static str = "openai";
    pub const ANTHROPIC: &'static str = "anthropic";
    pub const OPENROUTER: &'static str = "openrouter";
    pub const OLLAMA: &'static str = "ollama";
}

/// Predefined model IDs
impl ModelId {
    // OpenAI models
    pub const GPT_4: &'static str = "gpt-4";
    pub const GPT_4_TURBO: &'static str = "gpt-4-turbo-preview";
    pub const GPT_4O: &'static str = "gpt-4o";
    pub const GPT_4O_MINI: &'static str = "gpt-4o-mini";
    pub const GPT_3_5_TURBO: &'static str = "gpt-3.5-turbo";
    
    // Anthropic models
    pub const CLAUDE_3_OPUS: &'static str = "claude-3-opus-20240229";
    pub const CLAUDE_3_SONNET: &'static str = "claude-3-sonnet-20240229";
    pub const CLAUDE_3_HAIKU: &'static str = "claude-3-haiku-20240307";
    pub const CLAUDE_3_5_SONNET: &'static str = "claude-3-5-sonnet-20241022";
    pub const CLAUDE_3_5_HAIKU: &'static str = "claude-3-5-haiku-20241022";
}

/// Request ID for tracking
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(String);

impl RequestId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    pub fn from_string(id: String) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

impl Default for TokenUsage {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }
    }
} 