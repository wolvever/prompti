use crate::models::{ChatMessage, ChatRequest, ChatResponse, StreamingChatResponse};
use crate::types::{ModelId, ProviderId};
use crate::error::{ModelResult, ModelError};
use async_trait::async_trait;
use reqwest::{Client, Response};
use serde_json::json;
use std::pin::Pin;
use futures::{Stream, StreamExt};

#[async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> ProviderId;
    async fn chat(&self, req: &ChatRequest) -> ModelResult<ChatResponse>;
    async fn chat_stream(&self, req: &ChatRequest) -> ModelResult<Pin<Box<dyn Stream<Item = ModelResult<StreamingChatResponse>> + Send>>>;
}

pub struct OpenAIProvider {
    pub api_key: String,
    pub api_base: String,
    pub client: Client,
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn id(&self) -> ProviderId {
        ProviderId::new("openai")
    }

    async fn chat(&self, req: &ChatRequest) -> ModelResult<ChatResponse> {
        let url = format!("{}/v1/chat/completions", self.api_base.trim_end_matches('/'));
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(req)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(ModelError::UnexpectedStatus(resp.status(), resp.text().await.unwrap_or_default()));
        }
        let chat_resp: ChatResponse = resp.json().await?;
        Ok(chat_resp)
    }

    async fn chat_stream(&self, req: &ChatRequest) -> ModelResult<Pin<Box<dyn Stream<Item = ModelResult<StreamingChatResponse>> + Send>>> {
        let url = format!("{}/v1/chat/completions", self.api_base.trim_end_matches('/'));
        let mut req = req.clone();
        req.stream = Some(true);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(ModelError::UnexpectedStatus(resp.status(), resp.text().await.unwrap_or_default()));
        }
        let stream = sse_stream(resp);
        Ok(Box::pin(stream))
    }
}

pub struct ClaudeProvider {
    pub api_key: String,
    pub api_base: String,
    pub client: Client,
}

#[async_trait]
impl Provider for ClaudeProvider {
    fn id(&self) -> ProviderId {
        ProviderId::new("anthropic")
    }

    async fn chat(&self, req: &ChatRequest) -> ModelResult<ChatResponse> {
        let url = format!("{}/v1/messages", self.api_base.trim_end_matches('/'));
        let mut anthropic_req = json!({
            "model": req.model,
            "max_tokens": req.max_tokens.unwrap_or(1024),
            "messages": req.messages,
            "stream": false,
        });
        // Add other fields as needed
        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&anthropic_req)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(ModelError::UnexpectedStatus(resp.status(), resp.text().await.unwrap_or_default()));
        }
        let chat_resp: ChatResponse = resp.json().await?;
        Ok(chat_resp)
    }

    async fn chat_stream(&self, req: &ChatRequest) -> ModelResult<Pin<Box<dyn Stream<Item = ModelResult<StreamingChatResponse>> + Send>>> {
        let url = format!("{}/v1/messages", self.api_base.trim_end_matches('/'));
        let mut anthropic_req = json!({
            "model": req.model,
            "max_tokens": req.max_tokens.unwrap_or(1024),
            "messages": req.messages,
            "stream": true,
        });
        // Add other fields as needed
        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&anthropic_req)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(ModelError::UnexpectedStatus(resp.status(), resp.text().await.unwrap_or_default()));
        }
        let stream = sse_stream(resp);
        Ok(Box::pin(stream))
    }
}

fn sse_stream(resp: Response) -> impl Stream<Item = ModelResult<StreamingChatResponse>> + Send {
    use futures::stream;
    // Placeholder: In real code, parse SSE events from resp.bytes_stream()
    // Here, just return an empty stream for now
    stream::empty()
} 