// Tokenizer format fixer utility
use anyhow::Result;
use serde_json::{Value, Map};
use std::fs;
use std::path::Path;

pub fn fix_tokenizer_format(tokenizer_path: &str) -> Result<()> {
    let path = Path::new(tokenizer_path);

    if !path.exists() {
        return Ok(()); // Nothing to fix if file doesn't exist
    }

    // Read the tokenizer file
    let content = fs::read_to_string(path)?;
    let mut tokenizer: Value = serde_json::from_str(&content)?;

    // Fix the decoder section if it has extra fields
    if let Some(decoder) = tokenizer.get_mut("decoder") {
        if let Value::Object(decoder_obj) = decoder {
            // Keep only the "type" field for ByteLevel decoder
            if decoder_obj.get("type") == Some(&Value::String("ByteLevel".to_string())) {
                let mut new_decoder = Map::new();
                new_decoder.insert("type".to_string(), Value::String("ByteLevel".to_string()));
                *decoder = Value::Object(new_decoder);

                tracing::info!("Fixed decoder section in tokenizer at {}", tokenizer_path);
            }
        }
    }

    // Fix added_tokens section - ensure each token has add_prefix_space field
    if let Some(added_tokens) = tokenizer.get_mut("added_tokens") {
        if let Value::Array(tokens_array) = added_tokens {
            let mut fixed_tokens = false;
            for token in tokens_array {
                if let Value::Object(token_obj) = token {
                    // Add add_prefix_space field if it doesn't exist
                    if !token_obj.contains_key("add_prefix_space") {
                        token_obj.insert("add_prefix_space".to_string(), Value::Bool(false));
                        fixed_tokens = true;
                    }
                }
            }
            if fixed_tokens {
                tracing::info!("Fixed added_tokens section in tokenizer at {}", tokenizer_path);
            }
        }
    }

    // Write the fixed tokenizer back
    let fixed_content = serde_json::to_string_pretty(&tokenizer)?;
    fs::write(path, fixed_content)?;

    tracing::info!("Successfully fixed tokenizer format at {}", tokenizer_path);
    Ok(())
}