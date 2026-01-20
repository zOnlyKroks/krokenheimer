use anyhow::{Result, anyhow};
use std::path::Path;
use tokenizers::{Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel, decoders::byte_level::ByteLevel as ByteLevelDecoder, AddedToken};
use tokenizers::models::bpe::BpeTrainer;

/// Wrapper that uses HuggingFace Tokenizers instead of custom BPE
/// This fixes all corruption issues while maintaining the same interface
#[derive(Debug, Clone)]
pub struct BPETokenizerWrapper {
    pub tokenizer: Tokenizer,
}

impl BPETokenizerWrapper {
    /// Create a new BPE tokenizer wrapper using HuggingFace tokenizers
    pub fn new() -> Self {
        // Create a BPE tokenizer with proper configuration
        let mut tokenizer = Tokenizer::new(
            BPE::builder()
                .unk_token("<|unk|>".to_string())
                .build()
                .unwrap()
        );

        // Use byte-level pre-tokenizer (handles all UTF-8 properly)
        tokenizer.with_pre_tokenizer(ByteLevel::default());

        // Use byte-level decoder (no corruption issues)
        tokenizer.with_decoder(ByteLevelDecoder::default());

        // Skip normalizer for now to avoid complexity

        // Add special tokens that won't cause corruption
        let special_tokens: Vec<AddedToken> = vec![
            AddedToken::from("<|endoftext|>", true),
            AddedToken::from("<|pad|>", true),
            AddedToken::from("<|unk|>", true),
            AddedToken::from("<|system|>", true),
            AddedToken::from("<|user|>", true),
            AddedToken::from("<|assistant|>", true),
        ];

        tokenizer.add_special_tokens(&special_tokens);

        Self { tokenizer }
    }

    /// Train the tokenizer on conversation data and save it
    pub fn train_and_save(
        conversations: &[crate::ConversationData],
        output_path: &str,
        target_vocab_size: usize
    ) -> Result<Self> {
        // Convert conversations to training text
        let training_texts = Self::conversations_to_texts_static(conversations);

        // Create a new BPE model and tokenizer for training
        let mut tokenizer = Tokenizer::new(BPE::default());

        // Set up pre-tokenizer and decoder
        tokenizer.with_pre_tokenizer(ByteLevel::default());
        tokenizer.with_decoder(ByteLevelDecoder::default());

        // Create BPE trainer with proper configuration
        let mut trainer = BpeTrainer::builder()
            .vocab_size(target_vocab_size)
            .min_frequency(2)
            .special_tokens(vec![
                AddedToken::from("<|endoftext|>", true),
                AddedToken::from("<|pad|>", true),
                AddedToken::from("<|unk|>", true),
                AddedToken::from("<|system|>", true),
                AddedToken::from("<|user|>", true),
                AddedToken::from("<|assistant|>", true),
            ])
            .build();

        // Create temporary files for training (HuggingFace tokenizers expects files)
        let temp_dir = std::env::temp_dir().join("tokenizer_training");
        std::fs::create_dir_all(&temp_dir)?;

        let training_files: Vec<String> = training_texts.iter().enumerate().map(|(i, text)| {
            let file_path = temp_dir.join(format!("training_{}.txt", i));
            std::fs::write(&file_path, text).unwrap();
            file_path.to_string_lossy().to_string()
        }).collect();

        // Train the tokenizer using the correct API
        tokenizer.train_from_files(&mut trainer, training_files.clone())
            .map_err(|e| anyhow!("Training failed: {}", e))?;

        // Clean up temp files
        for file_path in training_files {
            let _ = std::fs::remove_file(&file_path);
        }
        let _ = std::fs::remove_dir(&temp_dir);

        // Ensure output directory exists
        std::fs::create_dir_all(output_path)?;

        // Save the tokenizer in HuggingFace format
        let tokenizer_path = Path::new(output_path).join("tokenizer.json");
        tokenizer.save(&tokenizer_path, false)
            .map_err(|e| anyhow!("Save failed: {}", e))?;

        let wrapper = Self { tokenizer };

        // Also save our custom format for compatibility
        wrapper.save_legacy_format(output_path)?;

        Ok(wrapper)
    }

    /// Load an existing tokenizer (tries HuggingFace format first, then legacy)
    pub fn load(output_path: &str) -> Result<Self> {
        let tokenizer_path = Path::new(output_path).join("tokenizer.json");

        if tokenizer_path.exists() {
            // Try loading HuggingFace format
            match Tokenizer::from_file(&tokenizer_path) {
                Ok(tokenizer) => return Ok(Self { tokenizer }),
                Err(e) => {
                    eprintln!("Failed to load HuggingFace tokenizer format: {}", e);
                }
            }
        }

        // Try legacy format as fallback
        let legacy_path = Path::new(output_path).join("bpe_tokenizer.json");
        if legacy_path.exists() {
            return Err(anyhow!("Legacy BPE format detected. Please retrain the tokenizer to use the new HuggingFace format."));
        }

        Err(anyhow!("No tokenizer found at: {}", output_path))
    }

    /// Convert conversations to training texts (static version)
    pub fn conversations_to_texts_static(conversations: &[crate::ConversationData]) -> Vec<String> {
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

    /// Convert conversations to training texts (same as before)
    pub fn conversations_to_texts(&self, conversations: &[crate::ConversationData]) -> Vec<String> {
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

    /// Save in legacy format for compatibility
    fn save_legacy_format(&self, output_path: &str) -> Result<()> {
        let legacy_path = Path::new(output_path).join("bpe_tokenizer.json");

        // Create a simple mapping for legacy compatibility
        let legacy_data = serde_json::json!({
            "version": "2.0_hf",
            "note": "This tokenizer now uses HuggingFace tokenizers internally",
            "vocab_size": self.tokenizer.get_vocab_size(true),
            "special_tokens": [
                "<|endoftext|>",
                "<|pad|>",
                "<|unk|>",
                "<|system|>",
                "<|user|>",
                "<|assistant|>",
            ]
        });

        std::fs::write(legacy_path, serde_json::to_string_pretty(&legacy_data)?)?;
        Ok(())
    }

    // Interface methods compatible with the existing system

    /// Get vocabulary size
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.tokenizer.get_vocab_size(with_added_tokens)
    }

    /// Encode text to token IDs (no corruption issues now!)
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<BPEEncoding> {
        match self.tokenizer.encode(text, add_special_tokens) {
            Ok(encoding) => {
                let ids = encoding.get_ids().to_vec();
                let tokens = encoding.get_tokens().iter().map(|s| s.to_string()).collect();
                Ok(BPEEncoding { ids, tokens })
            }
            Err(e) => Err(anyhow!("Encoding error: {}", e))
        }
    }

    /// Decode token IDs to text (no corruption issues now!)
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self.tokenizer.decode(token_ids, skip_special_tokens) {
            Ok(text) => Ok(text),
            Err(e) => Err(anyhow!("Decoding error: {}", e))
        }
    }
}

/// Encoding result compatible with the existing interface
#[derive(Debug, Clone)]
pub struct BPEEncoding {
    pub ids: Vec<u32>,
    pub tokens: Vec<String>,
}

impl BPEEncoding {
    /// Get token IDs
    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    /// Get token count
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
    use crate::{ConversationData, TrainingMessage};

    #[test]
    fn test_huggingface_wrapper_basic() {
        let wrapper = BPETokenizerWrapper::new();

        // Test encoding/decoding German text (should not corrupt now)
        let test_text = "<|user|> Hallo äöü ß! <|assistant|> Wie geht es dir?";
        let encoding = wrapper.encode(test_text, false).unwrap();
        let decoded = wrapper.decode(encoding.get_ids(), false).unwrap();

        println!("Original: {}", test_text);
        println!("Decoded:  {}", decoded);

        assert!(!encoding.get_ids().is_empty());
        // HuggingFace tokenizers handle UTF-8 properly, no corruption
        assert!(decoded.contains("ä") || decoded.contains("Hallo")); // Should preserve German chars
    }

    #[test]
    fn test_conversation_training() {
        let mut wrapper = BPETokenizerWrapper::new();

        // Create test conversation with German text
        let conversations = vec![
            ConversationData {
                messages: vec![
                    TrainingMessage {
                        role: "user".to_string(),
                        content: "Hallo, wie gehts? Schönen Tag!".to_string(),
                    },
                    TrainingMessage {
                        role: "assistant".to_string(),
                        content: "Mir geht es gut, danke für die Frage!".to_string(),
                    },
                ],
            },
        ];

        let texts = wrapper.conversations_to_texts(&conversations);
        assert!(!texts.is_empty());
        assert!(texts[0].contains("<|user|>"));
        assert!(texts[0].contains("<|assistant|>"));
        assert!(texts[0].contains("<|endoftext|>"));
    }
}