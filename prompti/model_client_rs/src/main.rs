use clap::{Arg, ArgAction, Command};
use model_client_rs::{ModelClient, config::{ClientConfig, ProviderConfig, ModelConfig}};
use model_client_rs::models::{ChatMessage, ChatRequest, MessageRole};
use serde_json;
use std::fs;
use std::io::{self, Write};
use tokio;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("model-client-rs")
        .version("0.1.0")
        .about("Rust-based model client for LLM providers")
        .arg(
            Arg::new("request-file")
                .long("request-file")
                .value_name("FILE")
                .help("JSON file containing the request")
                .required(true)
                .action(ArgAction::Set)
        )
        .arg(
            Arg::new("stream")
                .long("stream")
                .help("Enable streaming output")
                .action(ArgAction::SetTrue)
        )
        .get_matches();

    // Read request from file
    let request_file: &String = matches.get_one("request-file").expect("required");
    let request_data: serde_json::Value = serde_json::from_str(&fs::read_to_string(request_file)?)?;
    
    // Extract configuration
    let provider = request_data["provider"].as_str().unwrap_or("openai");
    let model = request_data["model"].as_str().unwrap_or("gpt-3.5-turbo");
    let parameters = request_data["parameters"].as_object().cloned().unwrap_or_default();
    
    // Get API key from environment
    let api_key = match provider {
        "openai" => std::env::var("OPENAI_API_KEY"),
        "anthropic" => std::env::var("ANTHROPIC_API_KEY"),
        _ => std::env::var("API_KEY"),
    }.map_err(|_| "API key not found in environment")?;
    
    // Create client configuration
    let provider_config = ProviderConfig {
        id: provider.into(),
        api_key: Some(api_key.clone()),
        api_base: parameters
            .get("api_base")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
    };

    let model_config = ModelConfig {
        id: model.into(),
        temperature: None,
        max_tokens: None,
        top_p: None,
        n: None,
        stop: None,
    };

    let client_config = ClientConfig {
        provider: provider_config,
        model: model_config,
        api_key: Some(api_key),
        api_base: None,
        timeout_secs: None,
    };
    
    // Create model client
    let client = ModelClient::new(&client_config)?;
    
    // Convert messages
    let messages: Vec<ChatMessage> = request_data["messages"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .map(|msg| {
            let role_str = msg["role"].as_str().unwrap_or("user");
            let role = match role_str {
                "system" => MessageRole::System,
                "assistant" => MessageRole::Assistant,
                "tool" => MessageRole::Tool,
                _ => MessageRole::User,
            };
            ChatMessage {
                role,
                content: msg["content"].as_str().unwrap_or("").to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }
        })
        .collect();
    
    // Create chat request
    let mut chat_request = ChatRequest::new(model.to_string(), messages);
    if let Some(temp) = parameters.get("temperature").and_then(|v| v.as_f64()) {
        chat_request = chat_request.with_temperature(temp as f32);
    }
    if let Some(max) = parameters.get("max_tokens").and_then(|v| v.as_u64()) {
        chat_request = chat_request.with_max_tokens(max as u32);
    }
    chat_request = chat_request.with_stream(matches.get_flag("stream"));
    
    // Execute request
    if matches.get_flag("stream") {
        // Streaming response
        let mut stream = client.chat_stream(&chat_request).await?;
        
        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => {
                    if let Some(choice) = response.choices.first() {
                        if let Some(delta) = &choice.delta {
                            let output = serde_json::json!({
                                "content": delta.content,
                                "role": "assistant"
                            });
                            println!("{}", serde_json::to_string(&output)?);
                            io::stdout().flush()?;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error in stream: {}", e);
                    break;
                }
            }
        }
    } else {
        // Non-streaming response
        let response = client.chat(&chat_request).await?;
        if let Some(content) = response.get_content() {
            let output = serde_json::json!({
                "content": content,
                "role": "assistant"
            });
            println!("{}", serde_json::to_string(&output)?);
        }
    }
    
    Ok(())
} 