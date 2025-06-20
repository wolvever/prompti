use crate::models::StreamingChatResponse;
use crate::error::ModelResult;
use futures::Stream;
use std::pin::Pin;

/// A streaming response from an LLM provider
pub type StreamingResponse = Pin<Box<dyn Stream<Item = ModelResult<StreamingChatResponse>> + Send>>;

/// A response stream that can be consumed
pub type ResponseStream = StreamingResponse; 