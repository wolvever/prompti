# Rust Model Client

This directory contains a Rust implementation of the model client that provides high-performance LLM interactions.

## Building

This crate exposes both a Python extension and an optional CLI binary.
To build and install the Python bindings locally:

```bash
cd prompti/prompti/model_client_rs
maturin develop --release
```

The standalone binary can still be built with:

```bash
cargo build --release
```

## Usage

The Rust client can be used in two ways:

### 1. As a Python wrapper

```python
from prompti.model_client import RustModelClient, RustModelConfig
from prompti.message import Message

# Create the client
client = RustModelClient()

# Create messages
messages = [
    Message(role="user", content="Hello, how are you?", kind="text")
]

# Create model configuration
model_cfg = RustModelConfig(
    api_key="your-api-key",
    provider="openai",
    model="gpt-3.5-turbo"
)

# Run the model
async for response in client.run(messages, model_cfg):
    print(response.content)
```

### 2. As a standalone binary

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key"

# Create a request file
cat > request.json << EOF
{
  "messages": [
    {"role": "user", "content": "Hello, how are you?", "kind": "text"}
  ],
  "provider": "openai",
  "model": "gpt-3.5-turbo",
  "parameters": {
    "temperature": 0.7
  }
}
EOF

# Run the client
./target/release/model-client-rs --request-file request.json --stream
```

## Features

- **High Performance**: Rust implementation for fast, memory-efficient operations
- **Streaming Support**: Real-time streaming of LLM responses
- **Multiple Providers**: Support for OpenAI and Anthropic Claude
- **Error Handling**: Comprehensive error handling and retry logic
- **Observability**: Built-in metrics and tracing support

## Configuration

The Rust client supports the following configuration options:

- `api_key`: Your API key for the provider
- `api_base`: Custom API base URL (optional)
- `provider`: Provider name ("openai" or "anthropic")
- `model`: Model name (e.g., "gpt-3.5-turbo", "claude-3-sonnet")
- `temperature`: Sampling temperature (0.0 to 2.0)
- `max_tokens`: Maximum tokens to generate
- `stream`: Enable streaming responses

## Development

### Prerequisites

- Rust 1.70+ with Cargo
- Python 3.8+ (for the wrapper)

### Building for Development

```bash
cargo build
```

### Running Tests

```bash
cargo test
```

### Code Formatting

```bash
cargo fmt
```

### Linting

```bash
cargo clippy
```

## Architecture

The Rust client consists of several modules:

- `client.rs`: Main client implementation
- `config.rs`: Configuration structures
- `models.rs`: Data models for requests/responses
- `providers.rs`: Provider-specific implementations
- `error.rs`: Error handling
- `types.rs`: Type definitions
- `streaming.rs`: Streaming response handling
- `utils.rs`: Utility functions
- `main.rs`: Command-line interface

## Integration

The Rust client integrates seamlessly with the Python ``prompti`` framework:

1. The Python wrapper (`RustModelClient`) exposes the `model_client_rs` library through PyO3 bindings.
2. Requests and responses are exchanged using JSON structures.
3. Streaming responses are yielded asynchronously without spawning a subprocess.

This design retains Rust performance while providing a smooth Python API.
