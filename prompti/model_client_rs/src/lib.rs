//! A comprehensive Rust ModelClient implementation supporting OpenAI and Claude
//! 
//! This library provides a unified interface for interacting with various LLM providers,
//! with support for streaming, function calling, and advanced error handling.

pub mod client;
pub mod config;
pub mod error;
pub mod models;
pub mod providers;
pub mod streaming;
pub mod types;
pub mod utils;

// Re-export main types for convenience
pub use client::ModelClient;
pub use config::{ClientConfig, ModelConfig, ProviderConfig};
pub use error::{ModelError, ModelResult};
pub use models::{ChatMessage, ChatResponse, FunctionCall, ToolCall};
pub use providers::{OpenAIProvider, ClaudeProvider, Provider};
pub use streaming::{StreamingResponse, ResponseStream};
pub use types::{ModelId, ProviderId};

/// Initialize the logging system
pub fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
} 