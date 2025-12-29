use crate::bpe_tokenizer::BPETokenizer;
use anyhow::{Result, anyhow};
use std::path::Path;
use serde_json;

/// Wrapper that makes our BPE tokenizer compatible with the existing training infrastructure
#[derive(Debug, Clone)]
pub struct BPETokenizerWrapper {
    pub tokenizer: BPETokenizer,
}

impl BPETokenizerWrapper {
    /// Create a new BPE tokenizer wrapper
    pub fn new() -> Self {
        let mut tokenizer = BPETokenizer::new();

        // Add the special tokens expected by the training system
        tokenizer.add_special_token("<|system|>");
        tokenizer.add_special_token("<|user|>");
        tokenizer.add_special_token("<|assistant|>");

        Self { tokenizer }
    }

    /// Train the tokenizer on conversation data and save it
    pub fn train_and_save(
        conversations: &[crate::training::ConversationData],
        output_path: &str,
        target_vocab_size: usize
    ) -> Result<Self> {
        let mut wrapper = Self::new();

        // Convert conversations to training text
        let training_texts = wrapper.conversations_to_texts(conversations);

        // Train the BPE tokenizer
        wrapper.tokenizer.train(&training_texts, target_vocab_size)?;

        // Save the tokenizer
        let tokenizer_path = Path::new(output_path).join("bpe_tokenizer.json");
        std::fs::create_dir_all(output_path)?;
        wrapper.tokenizer.save(&tokenizer_path)?;

        // Also save a compatible tokenizer.json for backward compatibility
        wrapper.save_compatible_format(output_path)?;

        Ok(wrapper)
    }

    /// Load an existing BPE tokenizer
    pub fn load(output_path: &str) -> Result<Self> {
        let tokenizer_path = Path::new(output_path).join("bpe_tokenizer.json");

        if !tokenizer_path.exists() {
            return Err(anyhow!("BPE tokenizer file not found: {}", tokenizer_path.display()));
        }

        let tokenizer = BPETokenizer::load(&tokenizer_path)?;
        Ok(Self { tokenizer })
    }

    /// Convert conversations to training texts
    pub fn conversations_to_texts(&self, conversations: &[crate::training::ConversationData]) -> Vec<String> {
        let mut texts = Vec::new();

        for conv in conversations {
            let mut formatted = String::new();

            for message in &conv.messages {
                let role_token = match message.role.to_lowercase().as_str() {
                    "system" => "<|system|>",
                    "user" => "<|user|>",
                    "assistant" => "<|assistant|>",
                    _ => "<|user|>",
                };
                formatted.push_str(&format!("{} {}\n", role_token, message.content));
            }
            formatted.push_str("<|endoftext|>");

            if !formatted.trim().is_empty() {
                texts.push(formatted);
            }
        }

        texts
    }

    /// Save in a format compatible with the existing system
    pub fn save_compatible_format(&self, output_path: &str) -> Result<()> {
        let tokenizer_path = Path::new(output_path).join("tokenizer.json");

        // Create a minimal compatible JSON structure
        let mut vocab_map = serde_json::Map::new();
        for (token, id) in &self.tokenizer.vocab {
            vocab_map.insert(token.clone(), serde_json::Value::Number((*id).into()));
        }

        let mut merges_array = Vec::new();
        for (first, second) in &self.tokenizer.merges {
            merges_array.push(format!("{} {}", first, second));
        }

        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 0,
                    "content": "<|endoftext|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true,
                    "add_prefix_space": false
                },
                {
                    "id": 1,
                    "content": "<|pad|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true,
                    "add_prefix_space": false
                },
                {
                    "id": self.tokenizer.special_tokens.get("<|system|>").copied().unwrap_or(2),
                    "content": "<|system|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": true,
                    "special": true,
                    "add_prefix_space": false
                },
                {
                    "id": self.tokenizer.special_tokens.get("<|user|>").copied().unwrap_or(3),
                    "content": "<|user|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": true,
                    "special": true,
                    "add_prefix_space": false
                },
                {
                    "id": self.tokenizer.special_tokens.get("<|assistant|>").copied().unwrap_or(4),
                    "content": "<|assistant|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": true,
                    "special": true,
                    "add_prefix_space": false
                }
            ],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel"
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": "<|unk|>",
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": vocab_map,
                "merges": merges_array
            }
        });

        let json_str = serde_json::to_string_pretty(&tokenizer_json)?;
        std::fs::write(tokenizer_path, json_str)?;

        Ok(())
    }

    // Interface methods compatible with tokenizers crate

    /// Get vocabulary size (compatible with tokenizers::Tokenizer::get_vocab_size)
    pub fn get_vocab_size(&self, _with_added_tokens: bool) -> usize {
        self.tokenizer.vocab_size()
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<BPEEncoding> {
        let token_ids = self.tokenizer.encode(text);
        let tokens = self.ids_to_tokens(&token_ids);
        Ok(BPEEncoding {
            ids: token_ids,
            tokens
        })
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        Ok(self.tokenizer.decode(token_ids))
    }

    /// Convert token IDs to token strings
    fn ids_to_tokens(&self, token_ids: &[u32]) -> Vec<String> {
        token_ids.iter()
            .map(|&id| self.tokenizer.id_to_token_str(id).unwrap_or("<|unk|>").to_string())
            .collect()
    }
}

/// A simple encoding result compatible with tokenizers::Encoding
#[derive(Debug, Clone)]
pub struct BPEEncoding {
    pub ids: Vec<u32>,
    pub tokens: Vec<String>,
}

impl BPEEncoding {
    /// Get token IDs (compatible with tokenizers::Encoding::get_ids)
    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    /// Get token count (compatible with tokenizers::Encoding::len)
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Get tokens
    pub fn get_tokens(&self) -> &[String] {
        &self.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{ConversationData, TrainingMessage};

    #[test]
    fn test_bpe_wrapper_basic() {
        let mut wrapper = BPETokenizerWrapper::new();

        // Create test conversation data
        let conversations = vec![
            ConversationData {
                messages: vec![
                    TrainingMessage {
                        role: "user".to_string(),
                        content: "Hello world".to_string(),
                    },
                    TrainingMessage {
                        role: "assistant".to_string(),
                        content: "Hi there!".to_string(),
                    },
                ],
            },
        ];

        // Train the tokenizer
        let texts = wrapper.conversations_to_texts(&conversations);
        wrapper.tokenizer.train(&texts, 1000).unwrap();

        // Test encoding/decoding
        let encoding = wrapper.encode("<|user|> Hello <|assistant|> Hi!", false).unwrap();
        let decoded = wrapper.decode(encoding.get_ids(), false).unwrap();

        println!("Encoded: {:?}", encoding.get_ids());
        println!("Decoded: {}", decoded);

        assert!(!encoding.get_ids().is_empty());
        assert!(decoded.contains("Hello"));
    }
}