use candle_core::{Device, Tensor, Result as CandleResult};
use candle_nn::{Linear, Module, VarBuilder};
use tokenizers::Tokenizer;
use anyhow::{Result, Context, anyhow};
use std::path::Path;
use serde_json;
use rand::Rng;
use crate::model::Gpt2Config;

// Simple transformer model implementation since candle_transformers::gpt2 is not available
pub struct SimpleTransformer {
    embedding: candle_nn::Embedding,
    layers: Vec<TransformerLayer>,
    ln_f: candle_nn::LayerNorm,
    lm_head: Linear,
}

struct TransformerLayer {
    attention: MultiHeadAttention,
    mlp: MLP,
    ln_1: candle_nn::LayerNorm,
    ln_2: candle_nn::LayerNorm,
}

struct MultiHeadAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
}

struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl SimpleTransformer {
    pub fn load(_weights: &std::collections::HashMap<String, Tensor>, config: &Gpt2Config, vb: VarBuilder) -> Result<Self, candle_core::Error> {
        // This is a simplified implementation - for a real model you'd need proper weight loading
        let embedding = candle_nn::embedding(config.vocab_size, config.n_embd, vb.pp("wte"))?;
        let ln_f = candle_nn::layer_norm(config.n_embd, config.layer_norm_epsilon, vb.pp("ln_f"))?;
        let lm_head = candle_nn::linear(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;

        let mut layers = Vec::new();
        for i in 0..config.n_layer {
            let layer_vb = vb.pp(&format!("h.{}", i));
            let layer = TransformerLayer {
                attention: MultiHeadAttention {
                    c_attn: candle_nn::linear(config.n_embd, 3 * config.n_embd, layer_vb.pp("attn.c_attn"))?,
                    c_proj: candle_nn::linear(config.n_embd, config.n_embd, layer_vb.pp("attn.c_proj"))?,
                    n_head: config.n_head,
                    n_embd: config.n_embd,
                },
                mlp: MLP {
                    c_fc: candle_nn::linear(config.n_embd, config.n_inner.unwrap_or(4 * config.n_embd), layer_vb.pp("mlp.c_fc"))?,
                    c_proj: candle_nn::linear(config.n_inner.unwrap_or(4 * config.n_embd), config.n_embd, layer_vb.pp("mlp.c_proj"))?,
                },
                ln_1: candle_nn::layer_norm(config.n_embd, config.layer_norm_epsilon, layer_vb.pp("ln_1"))?,
                ln_2: candle_nn::layer_norm(config.n_embd, config.layer_norm_epsilon, layer_vb.pp("ln_2"))?,
            };
            layers.push(layer);
        }

        Ok(Self {
            embedding,
            layers,
            ln_f,
            lm_head,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let mut hidden_states = self.embedding.forward(input_ids)?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        hidden_states = self.ln_f.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states)
    }
}

impl TransformerLayer {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x.clone();
        let x = self.ln_1.forward(x)?;
        let attn_output = self.attention.forward(&x)?;
        let x = (&residual + &attn_output)?;

        let residual = x.clone();
        let x = self.ln_2.forward(&x)?;
        let mlp_output = self.mlp.forward(&x)?;
        &residual + &mlp_output
    }
}

impl MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let qkv = self.c_attn.forward(x)?;
        // In real attention: split QKV, do attention computation
        // For simplification: just take the first n_embd dimensions (Q part)
        let (_batch_size, _seq_len, _full_dim) = qkv.dims3()?;
        let q_only = qkv.narrow(2, 0, self.n_embd)?; // Take first n_embd dimensions
        self.c_proj.forward(&q_only)
    }
}

impl MLP {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = self.c_fc.forward(x)?;
        // Simple GELU approximation using tanh
        let x = self.gelu_approximation(&x)?;
        self.c_proj.forward(&x)
    }

    fn gelu_approximation(&self, x: &Tensor) -> CandleResult<Tensor> {
        // GELU approximation: x * sigmoid(1.702 * x)
        let scale = Tensor::from_slice(&[1.702f32], &[], x.device())?;
        let scaled_x = x.broadcast_mul(&scale)?;
        let sigmoid_x = candle_nn::ops::sigmoid(&scaled_x)?;
        x.broadcast_mul(&sigmoid_x)
    }
}

pub struct InferenceService {
    model: SimpleTransformer,
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
        let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("Failed to load tokenizer: {}", e);
                tracing::info!("Attempting to fix tokenizer format...");

                // Try to fix the tokenizer format
                if let Err(fix_error) = crate::tokenizer_fixer::fix_tokenizer_format(&tokenizer_path.to_string_lossy()) {
                    tracing::error!("Failed to fix tokenizer format: {}", fix_error);
                    return Err(anyhow!("Failed to load tokenizer from {}: {}", tokenizer_path.display(), e));
                }

                // Try loading again after fix
                Tokenizer::from_file(&tokenizer_path)
                    .map_err(|e2| anyhow!("Failed to load tokenizer from {} even after format fix: {}", tokenizer_path.display(), e2))?
            }
        };

        // Load model config
        let config_path = Path::new(model_path).join("config.json");
        let config_data = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config from {}", config_path.display()))?;

        let config: Gpt2Config = serde_json::from_str(&config_data)
            .with_context(|| "Failed to parse model config")?;

        // Initialize model with weights
        let mut var_map = candle_nn::VarMap::new();
        let weights_path = Path::new(model_path).join("model.safetensors");

        if weights_path.exists() {
            // Load trained weights
            var_map.load(&weights_path)
                .map_err(|e| anyhow!("Failed to load model weights from {}: {}", weights_path.display(), e))?;
            tracing::info!("Loaded trained model weights");
        } else {
            tracing::warn!("No trained weights found, using randomly initialized model");
        }

        let var_builder = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
        let empty_weights = std::collections::HashMap::new();
        let model = SimpleTransformer::load(&empty_weights, &config, var_builder)
            .map_err(|e| anyhow!("Failed to initialize transformer model: {}", e))?;

        tracing::info!("Model loaded successfully with {} parameters", config.n_embd * config.n_layer);

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    pub fn generate(&self, prompt: &str, max_length: usize, temperature: f32) -> Result<String> {
        tracing::info!("Generating text for prompt: '{}' (max_length: {}, temp: {})", prompt, max_length, temperature);

        // Tokenize input
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let input_ids = encoding.get_ids();
        tracing::info!("Tokenized input: {:?}", input_ids);

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
            let (_batch_size, seq_len, _vocab_size) = logits.dims3()?;
            let next_token_logits = logits
                .narrow(1, seq_len - 1, 1)?  // Get last position
                .squeeze(1)?;  // Remove sequence dimension

            // Apply temperature
            let scaled_logits = if temperature > 0.0 {
                let temp_tensor = Tensor::from_slice(&[temperature], &[], next_token_logits.device())?;
                next_token_logits.broadcast_div(&temp_tensor)?
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
            let vocab = self.tokenizer.get_vocab(true);
            if let Some(eos_token) = vocab.get("[SEP]")
                .or_else(|| vocab.get("<|endoftext|>")) {
                if next_token == *eos_token {
                    break;
                }
            }

            generated_tokens.push(next_token);
            tracing::debug!("Generated token {}: {}", generated_tokens.len(), next_token);

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
        tracing::info!("Decoding {} tokens: {:?}", generated_tokens.len(), generated_tokens);
        let generated_text = self.tokenizer.decode(&generated_tokens, false)
            .map_err(|e| anyhow!("Decoding failed: {}", e))?;

        tracing::info!("Raw decoded text: '{}'", generated_text);
        let cleaned_text = self.clean_response(&generated_text);
        tracing::info!("Cleaned text: '{}'", cleaned_text);

        Ok(cleaned_text)
    }

    fn sample_token(&self, logits: &Tensor) -> CandleResult<u32> {
        // Convert logits to probabilities
        let probabilities = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;

        // Sample using multinomial distribution (simplified)
        // For now, we'll use a simple random sampling based on probabilities
        // Squeeze to convert from [1, vocab_size] to [vocab_size]
        let probabilities_1d = probabilities.squeeze(0)?;
        let probs_vec = probabilities_1d.to_vec1::<f32>()?;

        let mut rng = rand::rng();
        let random_value: f32 = rng.random();

        let mut cumulative = 0.0;
        for (i, prob) in probs_vec.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                let token_id = i as u32;
                // Bounds check to prevent garbage token generation
                let vocab = self.tokenizer.get_vocab(false);
                if token_id < vocab.len() as u32 {
                    return Ok(token_id);
                }
            }
        }

        // Safe fallback - return a common token (space or period)
        let vocab = self.tokenizer.get_vocab(false);
        if let Some(&space_token) = vocab.get(" ") {
            Ok(space_token)
        } else if let Some(&period_token) = vocab.get(".") {
            Ok(period_token)
        } else {
            // Absolute fallback to first token
            Ok(0)
        }
    }

    fn clean_response(&self, text: &str) -> String {
        let mut cleaned = text.to_string();

        // Remove special tokens
        cleaned = cleaned.replace("<|endoftext|>", "");
        cleaned = cleaned.replace("<|assistant|>", "");
        cleaned = cleaned.replace("<|system|>", "");
        cleaned = cleaned.replace("<|user|>", "");
        cleaned = cleaned.replace("<|pad|>", "");
        cleaned = cleaned.replace("[PAD]", "");
        cleaned = cleaned.replace("[UNK]", "");

        // Remove only actual control characters, keep most Unicode
        cleaned = cleaned.chars()
            .filter(|c| {
                // Keep most characters, only filter control chars
                !c.is_control() || c.is_whitespace()
            })
            .collect();

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

    // Debug function to test tokenization
    pub fn debug_tokenization(&self, text: &str) -> Result<()> {
        tracing::info!("=== TOKENIZATION DEBUG ===");
        tracing::info!("Input text: '{}'", text);

        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let tokens = encoding.get_ids();
        tracing::info!("Token IDs: {:?}", tokens);

        let decoded = self.tokenizer.decode(tokens, false)
            .map_err(|e| anyhow!("Decoding failed: {}", e))?;
        tracing::info!("Decoded back: '{}'", decoded);

        let vocab = self.tokenizer.get_vocab(false);
        tracing::info!("Vocab size: {}", vocab.len());

        // Show first few vocab entries
        let mut vocab_items: Vec<(String, u32)> = vocab.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        vocab_items.sort_by_key(|(_, v)| *v);

        tracing::info!("First 20 vocab tokens:");
        for (token, id) in vocab_items.iter().take(20) {
            tracing::info!("  {}: '{}'", id, token);
        }

        Ok(())
    }
}