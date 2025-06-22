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
use std::time::Instant;
use futures::StreamExt;
use metrics::{counter, gauge, histogram};

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
        let provider = self.provider.id().to_string();
        let model = req.model.clone();
        gauge!("llm_inflight_requests", 1.0, "provider" => provider.clone(), "is_error" => "false");
        let start = Instant::now();
        let resp = self.provider.chat(req).await;
        histogram!("llm_request_latency_seconds", start.elapsed().as_secs_f64(), "provider" => provider.clone());
        gauge!("llm_inflight_requests", -1.0, "provider" => provider.clone(), "is_error" => "false");
        match &resp {
            Ok(r) => {
                counter!("llm_requests_total", 1, "provider" => provider.clone(), "result" => "success", "is_error" => "false");
                if let Some(usage) = &r.usage {
                    counter!("llm_prompt_tokens_total", usage.prompt_tokens as u64, "provider" => provider.clone(), "model" => model.clone());
                    counter!("llm_completion_tokens_total", usage.completion_tokens as u64, "provider" => provider.clone(), "model" => model.clone());
                }
            }
            Err(_) => {
                counter!("llm_requests_total", 1, "provider" => provider.clone(), "result" => "error", "is_error" => "true");
            }
        }
        resp
    }

    pub async fn chat_stream(&self, req: &ChatRequest) -> ModelResult<Pin<Box<dyn Stream<Item = ModelResult<StreamingChatResponse>> + Send>>> {
        let provider = self.provider.id().to_string();
        let model = req.model.clone();
        gauge!("llm_inflight_requests", 1.0, "provider" => provider.clone(), "is_error" => "false");
        let start = Instant::now();
        match self.provider.chat_stream(req).await {
            Ok(stream) => {
                let provider_cl = provider.clone();
                let model_cl = model.clone();
                let mut first = true;
                let mut last = start;
                let wrapped = stream.inspect(move |res| {
                    if res.is_ok() {
                        let now = Instant::now();
                        if first {
                            histogram!("llm_first_token_latency_seconds", now.duration_since(start).as_secs_f64(), "provider" => provider_cl.clone(), "model" => model_cl.clone());
                            first = false;
                        } else {
                            histogram!("llm_stream_intertoken_gap_seconds", now.duration_since(last).as_secs_f64(), "provider" => provider_cl.clone(), "model" => model_cl.clone());
                        }
                        last = now;
                    }
                });
                histogram!("llm_request_latency_seconds", start.elapsed().as_secs_f64(), "provider" => provider.clone());
                gauge!("llm_inflight_requests", -1.0, "provider" => provider.clone(), "is_error" => "false");
                counter!("llm_requests_total", 1, "provider" => provider.clone(), "result" => "success", "is_error" => "false");
                Ok(Box::pin(wrapped))
            }
            Err(e) => {
                histogram!("llm_request_latency_seconds", start.elapsed().as_secs_f64(), "provider" => provider.clone());
                gauge!("llm_inflight_requests", -1.0, "provider" => provider.clone(), "is_error" => "true");
                counter!("llm_requests_total", 1, "provider" => provider.clone(), "result" => "error", "is_error" => "true");
                Err(e)
            }
        }
    }
}
