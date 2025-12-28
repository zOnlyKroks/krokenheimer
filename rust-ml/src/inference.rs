use candle_core::{Device, Tensor, Result as CandleResult};
use candle_transformers::models::gpt2::{Gpt2, Config as Gpt2Config};
use tokenizers::Tokenizer;
use anyhow::{Result, Context, anyhow};
use std::path::Path;
use serde_json;

pub struct InferenceService {
    model: Gpt2,
    tokenizer: Tokenizer,
    device: Device,
    config: Gpt2Config,
}

impl InferenceService {
    pub fn new(model_path: &str) -> Result<Self> {
        tracing::info!("Loading model from: {}", model_path);

        // Initialize device (CPU for now, can be extended for GPU)
        let device = Device::Cpu;

        // Load tokenizer
        let tokenizer_path = Path::new(model_path).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer from {}: {}", tokenizer_path.display(), e))?;

        // Load model config
        let config_path = Path::new(model_path).join("config.json");
        let config_data = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config from {}", config_path.display()))?;

        let config: Gpt2Config = serde_json::from_str(&config_data)
            .with_context(|| "Failed to parse model config")?;

        // Load model weights
        let weights_path = Path::new(model_path).join("model.safetensors");
        let weights = if weights_path.exists() {
            // Try SafeTensors first (preferred)
            candle_core::safetensors::load(&weights_path, &device)?
        } else {
            // Fallback to PyTorch format
            let pytorch_path = Path::new(model_path).join("pytorch_model.bin");
            if pytorch_path.exists() {
                // For now, we'll need to convert PyTorch weights to Candle format
                // This is a simplified implementation - in practice you'd need proper conversion
                return Err(anyhow!("PyTorch model conversion not implemented yet. Please retrain with Candle."));
            } else {
                return Err(anyhow!("No model weights found in {}", model_path));
            }
        };

        // Initialize model
        let model = Gpt2::load(&weights, &config, &device)
            .with_context(|| "Failed to initialize GPT-2 model")?;

        tracing::info!("Model loaded successfully with {} parameters", config.n_embd * config.n_layer);

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    pub fn generate(&self, prompt: &str, max_length: usize, temperature: f32) -> Result<String> {
        tracing::debug!("Generating text for prompt: {} chars", prompt.len());

        // Tokenize input
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let input_ids = encoding.get_ids();
        if input_ids.is_empty() {
            return Err(anyhow!("Empty input after tokenization"));
        }

        // Convert to tensor
        let input_tensor = Tensor::new(input_ids, &self.device)?
            .unsqueeze(0)?; // Add batch dimension

        let mut current_tokens = input_tensor;
        let mut generated_tokens = Vec::new();

        // Generation loop
        for _ in 0..max_length {
            // Forward pass
            let logits = self.model.forward(&current_tokens)?;

            // Get last token logits
            let (batch_size, seq_len, vocab_size) = logits.dims3()?;
            let next_token_logits = logits
                .narrow(1, seq_len - 1, 1)?  // Get last position
                .squeeze(1)?;  // Remove sequence dimension

            // Apply temperature
            let scaled_logits = if temperature > 0.0 {
                (&next_token_logits / temperature)?
            } else {
                next_token_logits
            };

            // Sample next token
            let next_token = if temperature > 0.0 {
                self.sample_token(&scaled_logits)?
            } else {
                // Greedy decoding
                scaled_logits.argmax(candle_core::D::Minus1)?
                    .to_scalar::<u32>()? as u32
            };

            // Check for end token
            if let Some(eos_token) = self.tokenizer.get_vocab(true).get("[SEP]")
                .or_else(|| self.tokenizer.get_vocab(true).get("<|endoftext|>")) {
                if next_token == *eos_token {
                    break;
                }
            }

            generated_tokens.push(next_token);

            // Update current tokens
            let next_token_tensor = Tensor::new(&[next_token], &self.device)?
                .unsqueeze(0)?;
            current_tokens = Tensor::cat(&[&current_tokens, &next_token_tensor], 1)?;

            // Limit context window
            if current_tokens.dim(1)? > self.config.n_positions {
                let start = current_tokens.dim(1)? - self.config.n_positions;
                current_tokens = current_tokens.narrow(1, start, self.config.n_positions)?;
            }
        }

        // Decode generated tokens
        let generated_text = self.tokenizer.decode(&generated_tokens, false)
            .map_err(|e| anyhow!("Decoding failed: {}", e))?;

        let cleaned_text = self.clean_response(&generated_text);

        tracing::debug!("Generated {} tokens: {}", generated_tokens.len(), cleaned_text);

        Ok(cleaned_text)
    }

    fn sample_token(&self, logits: &Tensor) -> CandleResult<u32> {
        // Convert logits to probabilities
        let probabilities = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;

        // Sample using multinomial distribution (simplified)
        // For now, we'll use a simple random sampling based on probabilities
        let probs_vec = probabilities.to_vec1::<f32>()?;

        let mut rng = rand::thread_rng();
        let random_value: f32 = rand::Rng::gen(&mut rng);

        let mut cumulative = 0.0;
        for (i, prob) in probs_vec.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return Ok(i as u32);
            }
        }

        // Fallback to last token
        Ok((probs_vec.len() - 1) as u32)
    }

    fn clean_response(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        // Remove special tokens
        cleaned = cleaned.replace("<|endoftext|>", "");
        cleaned = cleaned.replace("<|assistant|>", "");
        cleaned = cleaned.replace("<|system|>", "");
        cleaned = cleaned.replace("<|user|>", "");
        cleaned = cleaned.replace("<|pad|>", "");

        // Remove bot name prefix
        if cleaned.to_lowercase().starts_with("krokenheimer:") {
            cleaned = cleaned[13..].trim_start().to_string();
        }

        // Clean up whitespace
        cleaned = cleaned.trim().to_string();

        // Take first meaningful line if multiple lines
        let lines: Vec<&str> = cleaned.lines()
            .map(|l| l.trim())
            .filter(|l| l.len() > 2)
            .collect();

        if let Some(first_line) = lines.first() {
            cleaned = first_line.to_string();
        }

        // Limit length
        if cleaned.len() > 300 {
            cleaned = format!("{}...", &cleaned[..297]);
        }

        // Fallback for empty responses
        if cleaned.is_empty() {
            cleaned = "👍".to_string();
        }

        cleaned
    }

    pub fn get_model_info(&self) -> (usize, usize, usize) {
        (
            self.config.n_layer,
            self.config.n_embd,
            self.config.n_positions,
        )
    }
}