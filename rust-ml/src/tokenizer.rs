// Tokenizer utilities and helper functions
use anyhow::Result;
use tokenizers::Tokenizer;

pub struct TokenizerUtils;

impl TokenizerUtils {
    /// Validate tokenizer file exists and is loadable
    pub fn validate_tokenizer(tokenizer_path: &str) -> Result<bool> {
        match Tokenizer::from_file(tokenizer_path) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get tokenizer vocabulary size
    pub fn get_vocab_size(tokenizer_path: &str) -> Result<usize> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        Ok(tokenizer.get_vocab_size(false))
    }

    /// Estimate token count for text
    pub fn estimate_tokens(tokenizer_path: &str, text: &str) -> Result<usize> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        let encoding = tokenizer.encode(text, false)?;
        Ok(encoding.len())
    }

    /// Truncate text to fit within token limit
    pub fn truncate_to_tokens(tokenizer_path: &str, text: &str, max_tokens: usize) -> Result<String> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        let encoding = tokenizer.encode(text, false)?;

        if encoding.len() <= max_tokens {
            return Ok(text.to_string());
        }

        // Truncate tokens and decode back
        let truncated_ids: Vec<u32> = encoding.get_ids()[..max_tokens].to_vec();
        let truncated_text = tokenizer.decode(&truncated_ids, false)?;

        Ok(truncated_text)
    }
}