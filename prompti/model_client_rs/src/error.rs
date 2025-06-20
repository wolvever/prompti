use reqwest::StatusCode;
use serde_json;
use std::io;
use thiserror::Error;
use tokio::task::JoinError;

pub type ModelResult<T> = std::result::Result<T, ModelError>;

/// Main error type for the ModelClient
#[derive(Error, Debug)]
pub enum ModelError {
    /// Stream disconnected before completion
    #[error("stream disconnected before completion: {0}")]
    Stream(String),

    /// Request timed out
    #[error("request timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// Unexpected HTTP status code
    #[error("unexpected status {0}: {1}")]
    UnexpectedStatus(StatusCode, String),

    /// Retry limit exceeded
    #[error("exceeded retry limit ({0} attempts), last status: {1}")]
    RetryLimit(u32, StatusCode),

    /// Rate limit exceeded
    #[error("rate limit exceeded: {0}")]
    RateLimit(String),

    /// Context window exceeded
    #[error("context window exceeded: input tokens {0}, max tokens {1}")]
    ContextWindowExceeded(u32, u32),

    /// Authentication error
    #[error("authentication failed: {0}")]
    Authentication(String),

    /// Invalid request parameters
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// Provider-specific error
    #[error("provider error: {0}")]
    Provider(String),

    /// Model not found or not available
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// Function calling error
    #[error("function calling error: {0}")]
    FunctionCall(String),

    /// Token counting error
    #[error("token counting error: {0}")]
    TokenCount(String),

    /// Configuration error
    #[error("configuration error: {0}")]
    Configuration(String),

    /// Environment variable error
    #[error("missing environment variable: {0}")]
    EnvVar(#[from] EnvVarError),

    // Automatic conversions for common external error types
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Url(#[from] url::ParseError),

    #[error(transparent)]
    TokioJoin(#[from] JoinError),

    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

/// Environment variable error
#[derive(Debug)]
pub struct EnvVarError {
    /// Name of the environment variable that is missing
    pub var: String,
    /// Optional instructions to help the user get a valid value
    pub instructions: Option<String>,
}

impl std::fmt::Display for EnvVarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Missing environment variable: `{}`", self.var)?;
        if let Some(instructions) = &self.instructions {
            write!(f, ". {}", instructions)?;
        }
        Ok(())
    }
}

impl std::error::Error for EnvVarError {}

impl ModelError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            ModelError::Stream(_) => true,
            ModelError::Timeout(_) => true,
            ModelError::RateLimit(_) => true,
            ModelError::UnexpectedStatus(status, _) => {
                status.is_server_error() || *status == StatusCode::TOO_MANY_REQUESTS
            }
            _ => false,
        }
    }

    /// Check if the error is a client error (should not retry)
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            ModelError::Authentication(_)
                | ModelError::InvalidRequest(_)
                | ModelError::ModelNotFound(_)
                | ModelError::ContextWindowExceeded(_, _)
                | ModelError::Configuration(_)
                | ModelError::EnvVar(_)
        )
    }

    /// Get the HTTP status code if available
    pub fn status_code(&self) -> Option<StatusCode> {
        match self {
            ModelError::UnexpectedStatus(status, _) => Some(*status),
            ModelError::RetryLimit(_, status) => Some(*status),
            _ => None,
        }
    }
} 