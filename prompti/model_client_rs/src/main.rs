use clap::{App, Arg};
use model_client_rs::{ModelClient, config::{ClientConfig, ProviderConfig}};
use model_client_rs::models::{ChatMessage, ChatRequest};
use serde_json;
use std::fs;
use std::io::{self, Write};
use tokio;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("model-client-rs")
        .version("0.1.0")
        .about("Rust-based model client for LLM providers")
        .arg(
            Arg::new("request-file")
                .long("request-file")
                .value_name("FILE")
                .help("JSON file containing the request")
                .required(true)
        )
        .arg(
            Arg::new("stream")
                .long("stream")
                .help("Enable streaming output")
                .takes_value(false)
        )
        .get_matches();

    // Read request from file
    let request_file = matches.value_of("request-file").unwrap();
    let request_data: serde_json::Value = serde_json::from_str(&fs::read_to_string(request_file)?)?;
    
    // Extract configuration
    let provider = request_data["provider"].as_str().unwrap_or("openai");
    let model = request_data["model"].as_str().unwrap_or("gpt-3.5-turbo");
    let parameters = request_data["parameters"].as_object().unwrap_or(&serde_json::Map::new());
    
    // Get API key from environment
    let api_key = match provider {
        "openai" => std::env::var("OPENAI_API_KEY"),
        "anthropic" => std::env::var("ANTHROPIC_API_KEY"),
        _ => std::env::var("API_KEY"),
    }.map_err(|_| "API key not found in environment")?;
    
    // Create client configuration
    let provider_config = ProviderConfig {
        id: provider.to_string(),
        api_key: Some(api_key.clone()),
        api_base: parameters.get("api_base").and_then(|v| v.as_str()).map(|s| s.to_string()),
    };
    
    let client_config = ClientConfig {
        api_key: Some(api_key),
        api_base: None,
        provider: provider_config,
    };
    
    // Create model client
    let client = ModelClient::new(&client_config)?;
    
    // Convert messages
    let messages: Vec<ChatMessage> = request_data["messages"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .map(|msg| {
            ChatMessage {
                role: msg["role"].as_str().unwrap_or("user").to_string(),
                content: msg["content"].as_str().unwrap_or("").to_string(),
            }
        })
        .collect();
    
    // Create chat request
    let chat_request = ChatRequest {
        messages,
        model: model.to_string(),
        temperature: parameters.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.7),
        max_tokens: parameters.get("max_tokens").and_then(|v| v.as_u64()).map(|v| v as u32),
        stream: matches.is_present("stream"),
        ..Default::default()
    };
    
    // Execute request
    if matches.is_present("stream") {
        // Streaming response
        let mut stream = client.chat_stream(&chat_request).await?;
        
        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => {
                    if let Some(content) = response.content {
                        let output = serde_json::json!({
                            "content": content,
                            "role": "assistant"
                        });
                        println!("{}", serde_json::to_string(&output)?);
                        io::stdout().flush()?;
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
        if let Some(content) = response.content {
            let output = serde_json::json!({
                "content": content,
                "role": "assistant"
            });
            println!("{}", serde_json::to_string(&output)?);
        }
    }
    
    Ok(())
} 