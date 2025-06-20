use crate::config::{ClientConfig, ModelConfig, ProviderConfig};
use crate::models::{ChatMessage, ChatRequest, ChatResponse, StreamingChatResponse};
use crate::providers::{OpenAIProvider, ClaudeProvider, Provider};
use crate::types::{ModelId, ProviderId};
use crate::error::{ModelResult, ModelError};
use async_trait::async_trait;
use reqwest::Client;
use std::sync::Arc;
use std::pin::Pin;
use futures::Stream;

pub struct ModelClient {
    provider: Arc<dyn Provider>,
}

impl ModelClient {
    pub fn new(config: &ClientConfig) -> ModelResult<Self> {
        let client = Client::new();
        let provider: Arc<dyn Provider> = match config.provider.id.as_str() {
            "openai" => Arc::new(OpenAIProvider {
                api_key: config.api_key.clone().or_else(|| config.provider.api_key.clone()).ok_or_else(|| ModelError::Configuration("Missing OpenAI API key".to_string()))?,
                api_base: config.api_base.clone().or_else(|| config.provider.api_base.clone()).unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
                client,
            }),
            "anthropic" => Arc::new(ClaudeProvider {
                api_key: config.api_key.clone().or_else(|| config.provider.api_key.clone()).ok_or_else(|| ModelError::Configuration("Missing Claude API key".to_string()))?,
                api_base: config.api_base.clone().or_else(|| config.provider.api_base.clone()).unwrap_or_else(|| "https://api.anthropic.com/v1".to_string()),
                client,
            }),
            other => return Err(ModelError::Provider(format!("Unknown provider: {}", other))),
        };
        Ok(Self { provider })
    }

    pub async fn chat(&self, req: &ChatRequest) -> ModelResult<ChatResponse> {
        self.provider.chat(req).await
    }

    pub async fn chat_stream(&self, req: &ChatRequest) -> ModelResult<Pin<Box<dyn Stream<Item = ModelResult<StreamingChatResponse>> + Send>>> {
        self.provider.chat_stream(req).await
    }
} 