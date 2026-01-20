use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use std::path::Path;

/// A complete BPE (Byte Pair Encoding) tokenizer implementation from scratch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    /// Maps tokens to their IDs
    pub vocab: HashMap<String, u32>,
    /// Maps IDs back to tokens
    pub id_to_token: HashMap<u32, String>,
    /// Ordered list of merge operations (token1, token2) -> merged_token
    pub merges: Vec<(String, String)>,
    /// Special tokens (won't be split further)
    pub special_tokens: HashMap<String, u32>,
    /// Next available token ID
    next_id: u32,
    /// End of word token (used for proper word boundaries)
    pub eos_token: String,
    /// Padding token
    pub pad_token: String,
    /// Unknown token
    pub unk_token: String,
}

impl BPETokenizer {
    /// Create a new BPE tokenizer
    pub fn new() -> Self {
        let mut tokenizer = Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            merges: Vec::new(),
            special_tokens: HashMap::new(),
            next_id: 0,
            eos_token: "<|endoftext|>".to_string(),
            pad_token: "<|pad|>".to_string(),
            unk_token: "<|unk|>".to_string(),
        };

        // Add basic special tokens
        tokenizer.add_special_token(&tokenizer.eos_token.clone());
        tokenizer.add_special_token(&tokenizer.pad_token.clone());
        tokenizer.add_special_token(&tokenizer.unk_token.clone());

        tokenizer
    }

    /// Add a special token that won't be split during BPE
    pub fn add_special_token(&mut self, token: &str) {
        if !self.vocab.contains_key(token) {
            let id = self.next_id;
            self.vocab.insert(token.to_string(), id);
            self.id_to_token.insert(id, token.to_string());
            self.special_tokens.insert(token.to_string(), id);
            self.next_id += 1;
        }
    }

    /// Train the BPE tokenizer on a corpus of text
    pub fn train(&mut self, texts: &[String], target_vocab_size: usize) -> Result<()> {
        if texts.is_empty() {
            return Err(anyhow!("Cannot train on empty corpus"));
        }

        println!("Training BPE tokenizer on {} texts, target vocab size: {}",
                 texts.len(), target_vocab_size);

        // Step 1: Initialize character-level vocabulary from the corpus
        self.initialize_vocab_from_corpus(texts)?;

        let initial_vocab_size = self.vocab.len();
        println!("Initial character-level vocabulary size: {}", initial_vocab_size);

        // Step 2: Prepare word frequency counts
        let word_freqs = self.get_word_frequencies(texts);
        println!("Unique words in corpus: {}", word_freqs.len());

        // Step 3: Split words into characters
        let mut word_splits = self.initialize_word_splits(&word_freqs);

        // Step 4: Iteratively merge most frequent pairs
        let target_merges = target_vocab_size - initial_vocab_size;
        println!("Will perform {} merge operations", target_merges);

        for merge_step in 0..target_merges {
            // Count all adjacent pairs across all word splits
            let pair_frequencies = self.count_pairs(&word_splits, &word_freqs);

            if pair_frequencies.is_empty() {
                println!("No more pairs to merge at step {}", merge_step);
                break;
            }

            // Find the most frequent pair
            let most_frequent_pair = pair_frequencies.iter()
                .max_by_key(|(_, &count)| count)
                .map(|(pair, _)| pair.clone());

            if let Some((first, second)) = most_frequent_pair {
                let merged = format!("{}{}", first, second);

                // Add new merged token to vocabulary
                let new_id = self.next_id;
                self.vocab.insert(merged.clone(), new_id);
                self.id_to_token.insert(new_id, merged.clone());
                self.next_id += 1;

                // Record the merge operation
                self.merges.push((first.clone(), second.clone()));

                // Update all word splits that contain this pair
                self.apply_merge(&mut word_splits, &first, &second, &merged);

                if (merge_step + 1) % 1000 == 0 {
                    let freq = pair_frequencies.get(&(first.clone(), second.clone())).unwrap_or(&0);
                    println!("Merge step {}: {} + {} -> {} (freq: {})",
                             merge_step + 1, first, second, merged, freq);
                }
            } else {
                break;
            }
        }

        println!("Training completed! Final vocabulary size: {}", self.vocab.len());
        Ok(())
    }

    /// Initialize the vocabulary with all characters found in the corpus
    fn initialize_vocab_from_corpus(&mut self, texts: &[String]) -> Result<()> {
        let mut chars = std::collections::HashSet::new();

        for text in texts {
            for ch in text.chars() {
                chars.insert(ch.to_string());
            }
        }

        // Add basic Unicode characters found in text first (prioritize)
        for ch_str in &chars {
            if !self.vocab.contains_key(ch_str) {
                let id = self.next_id;
                self.vocab.insert(ch_str.clone(), id);
                self.id_to_token.insert(id, ch_str.clone());
                self.next_id += 1;
            }
        }

        // Only add common byte-level tokens for handling edge cases
        let common_bytes = [
            0x20, // Space
            0x0A, // Newline
            0x09, // Tab
            0x21, 0x22, 0x27, // !, ", '
            0x28, 0x29, // (, )
            0x2C, 0x2E, // ,, .
            0x3A, 0x3B, 0x3F, // :, ;, ?
        ];

        for &byte_val in &common_bytes {
            let byte_char = char::from_u32(byte_val as u32).unwrap_or('?');
            let byte_token = format!("Ġ{}", byte_char); // Use GPT-style prefix
            if !self.vocab.contains_key(&byte_token) {
                let id = self.next_id;
                self.vocab.insert(byte_token.clone(), id);
                self.id_to_token.insert(id, byte_token);
                self.next_id += 1;
            }
        }

        // Add unknown token fallback for rare bytes
        let unk_byte_token = "<|byte|>".to_string();
        if !self.vocab.contains_key(&unk_byte_token) {
            let id = self.next_id;
            self.vocab.insert(unk_byte_token.clone(), id);
            self.id_to_token.insert(id, unk_byte_token);
            self.next_id += 1;
        }

        Ok(())
    }

    /// Get word frequencies from the corpus
    fn get_word_frequencies(&self, texts: &[String]) -> HashMap<String, usize> {
        let mut word_freqs = HashMap::new();

        for text in texts {
            // Simple whitespace tokenization with word ending markers
            for word in text.split_whitespace() {
                if !word.is_empty() {
                    // Add end-of-word marker for proper BPE boundaries
                    let word_with_eow = format!("{}</w>", word);
                    *word_freqs.entry(word_with_eow).or_insert(0) += 1;
                }
            }
        }

        word_freqs
    }

    /// Initialize word splits (each word split into characters)
    fn initialize_word_splits(&self, word_freqs: &HashMap<String, usize>) -> HashMap<String, Vec<String>> {
        let mut word_splits = HashMap::new();

        for word in word_freqs.keys() {
            let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            word_splits.insert(word.clone(), chars);
        }

        word_splits
    }

    /// Count frequencies of all adjacent pairs across word splits
    fn count_pairs(
        &self,
        word_splits: &HashMap<String, Vec<String>>,
        word_freqs: &HashMap<String, usize>
    ) -> HashMap<(String, String), usize> {
        let mut pair_counts = HashMap::new();

        for (word, splits) in word_splits {
            let word_freq = word_freqs.get(word).unwrap_or(&0);

            // Count adjacent pairs in this word
            for window in splits.windows(2) {
                if let [first, second] = window {
                    let pair = (first.clone(), second.clone());
                    *pair_counts.entry(pair).or_insert(0) += word_freq;
                }
            }
        }

        pair_counts
    }

    /// Apply a merge operation to all word splits
    fn apply_merge(
        &self,
        word_splits: &mut HashMap<String, Vec<String>>,
        first: &str,
        second: &str,
        merged: &str,
    ) {
        for splits in word_splits.values_mut() {
            let mut i = 0;
            while i < splits.len().saturating_sub(1) {
                if splits[i] == first && splits[i + 1] == second {
                    // Merge the two tokens
                    splits[i] = merged.to_string();
                    splits.remove(i + 1);
                }
                i += 1;
            }
        }
    }

    /// Encode text into token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let mut result = Vec::new();

        // First, handle special tokens
        let remaining_text = text;
        let mut pos = 0;

        // Scan for special tokens
        while pos < remaining_text.len() {
            let mut found_special = false;

            // Check if any special token starts at current position
            for special_token in self.special_tokens.keys() {
                if remaining_text[pos..].starts_with(special_token) {
                    if let Some(&token_id) = self.special_tokens.get(special_token) {
                        result.push(token_id);
                    }
                    pos += special_token.len();
                    found_special = true;
                    break;
                }
            }

            if !found_special {
                // Extract the next "word" (until next special token or whitespace)
                let word_end = self.find_next_boundary(&remaining_text[pos..]);
                let word = &remaining_text[pos..pos + word_end];

                if !word.trim().is_empty() {
                    let word_tokens = self.encode_word(word);
                    result.extend(word_tokens);
                }

                pos += word_end;
            }
        }

        result
    }

    /// Find the boundary of the next word/token
    fn find_next_boundary(&self, text: &str) -> usize {
        // Check for special tokens first
        for special_token in self.special_tokens.keys() {
            if text.starts_with(special_token) {
                return special_token.len();
            }
        }

        // Otherwise find next whitespace or special token
        let mut end = 0;
        for (i, ch) in text.char_indices() {
            if ch.is_whitespace() {
                return if i == 0 { 1 } else { i };
            }

            // Check if any special token starts here
            for special_token in self.special_tokens.keys() {
                if text[i..].starts_with(special_token) {
                    return if i == 0 { special_token.len() } else { i };
                }
            }

            end = i + ch.len_utf8();
        }

        end.max(1)
    }

    /// Encode a single word using learned BPE merges
    fn encode_word(&self, word: &str) -> Vec<u32> {
        if word.is_empty() {
            return vec![];
        }

        // Add end-of-word marker for consistency with training
        let word_with_eow = format!("{}</w>", word.trim());

        // Start with character splits
        let mut tokens: Vec<String> = word_with_eow.chars().map(|c| c.to_string()).collect();

        // Apply all learned merges in order
        for (first, second) in &self.merges {
            let merged = format!("{}{}", first, second);

            // Apply this merge wherever possible
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if tokens[i] == *first && tokens[i + 1] == *second {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        // Convert tokens to IDs
        tokens.into_iter()
            .map(|token| self.vocab.get(&token).copied().unwrap_or_else(|| {
                // Handle unknown tokens
                self.special_tokens.get(&self.unk_token).copied().unwrap_or(0)
            }))
            .collect()
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut tokens = Vec::new();

        for &token_id in token_ids {
            if let Some(token) = self.id_to_token.get(&token_id) {
                tokens.push(token.as_str());
            } else {
                // Handle unknown token IDs gracefully
                tokens.push("<|unk|>");
            }
        }

        // Join tokens and clean up BPE artifacts
        let mut result = tokens.join("");

        // Clean up BPE-specific artifacts
        result = result.replace("</w>", " ");     // End of word marker
        result = result.replace("<|byte|>", "");  // Remove byte fallback tokens
        result = result.replace("Ġ", " ");        // GPT-style space prefix

        // Clean up control characters and invalid sequences
        result = result.chars()
            .filter(|c| {
                // Keep most characters, filter only problematic control chars
                !c.is_control() || *c == '\n' || *c == '\t' || *c == ' '
            })
            .collect();

        // Remove special tokens that shouldn't appear in normal text
        result = result.replace("<0x", " ");  // Remove hex byte representations
        result = result.replace(">", " ");    // Clean up remaining brackets

        // Clean up multiple spaces and normalize whitespace
        let words: Vec<&str> = result.split_whitespace().collect();
        result = words.join(" ");

        // Handle German characters properly (since this appears to be German text)
        result = result.replace("Ã¤", "ä");
        result = result.replace("Ã¶", "ö");
        result = result.replace("Ã¼", "ü");
        result = result.replace("Ã", "ß");

        result.trim().to_string()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get token for ID
    pub fn id_to_token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Get ID for token
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Save the tokenizer to disk
    pub fn save(&self, path: &Path) -> Result<()> {
        let json_str = serde_json::to_string_pretty(self)
            .map_err(|e| anyhow!("Failed to serialize tokenizer: {}", e))?;

        std::fs::write(path, json_str)
            .map_err(|e| anyhow!("Failed to write tokenizer to {}: {}", path.display(), e))?;

        Ok(())
    }

    /// Load the tokenizer from disk
    pub fn load(path: &Path) -> Result<Self> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read tokenizer from {}: {}", path.display(), e))?;

        let tokenizer: BPETokenizer = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to deserialize tokenizer: {}", e))?;

        Ok(tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_bpe_training() {
        let mut tokenizer = BPETokenizer::new();

        let texts = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "world peace".to_string(),
        ];

        tokenizer.train(&texts, 300).unwrap();

        // Test encoding
        let tokens = tokenizer.encode("hello world");
        assert!(!tokens.is_empty());

        // Test decoding
        let decoded = tokenizer.decode(&tokens);
        assert!(decoded.contains("hello") && decoded.contains("world"));

        println!("Vocab size: {}", tokenizer.vocab_size());
    }

    #[test]
    fn test_special_tokens() {
        let mut tokenizer = BPETokenizer::new();
        tokenizer.add_special_token("<|user|>");
        tokenizer.add_special_token("<|assistant|>");

        let text = "<|user|> Hello! <|assistant|> Hi there!";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);

        assert!(decoded.contains("<|user|>"));
        assert!(decoded.contains("<|assistant|>"));
    }
}