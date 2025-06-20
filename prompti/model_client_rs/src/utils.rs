use crate::error::ModelResult;
use serde_json::Value;

/// Utility functions for the model client

/// Parse a JSON string into a Value
pub fn parse_json(json_str: &str) -> ModelResult<Value> {
    serde_json::from_str(json_str).map_err(|e| crate::error::ModelError::Serialization(e.to_string()))
}

/// Convert a Value to a JSON string
pub fn to_json(value: &Value) -> ModelResult<String> {
    serde_json::to_string(value).map_err(|e| crate::error::ModelError::Serialization(e.to_string()))
}

/// Extract a string value from a JSON object
pub fn get_string(value: &Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(|s| s.to_string())
}

/// Extract a number value from a JSON object
pub fn get_number(value: &Value, key: &str) -> Option<f64> {
    value.get(key)?.as_f64()
} 