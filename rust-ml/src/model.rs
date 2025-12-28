// Model utilities and configuration
use candle_nn::Activation;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::path::Path;

// Define our own GPT-2 config since candle_transformers::models::gpt2 is not available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_inner: Option<usize>,
    pub activation_function: Activation,
    pub resid_pdrop: f64,
    pub embd_pdrop: f64,
    pub attn_pdrop: f64,
    pub layer_norm_epsilon: f64,
    pub initializer_range: f64,
    pub scale_attn_weights: bool,
    pub use_cache: bool,
    pub scale_attn_by_inverse_layer_idx: bool,
    pub reorder_and_upcast_attn: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_type: String,
    pub model_version: String,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_positions: usize,
    pub training_date: String,
    pub training_messages: usize,
    pub training_epochs: u32,
}

impl ModelMetadata {
    pub fn new(config: &Gpt2Config, training_messages: usize, epochs: u32) -> Self {
        Self {
            model_type: "gpt2-candle".to_string(),
            model_version: "0.1.0".to_string(),
            vocab_size: config.vocab_size,
            n_layer: config.n_layer,
            n_embd: config.n_embd,
            n_positions: config.n_positions,
            training_date: chrono::Utc::now().to_rfc3339(),
            training_messages,
            training_epochs: epochs,
        }
    }

    pub fn load(model_path: &str) -> Result<Self> {
        let metadata_path = Path::new(model_path).join("metadata.json");
        let content = std::fs::read_to_string(metadata_path)?;
        Ok(serde_json::from_str(&content)?)
    }

    pub fn save(&self, model_path: &str) -> Result<()> {
        let metadata_path = Path::new(model_path).join("metadata.json");
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(metadata_path, content)?;
        Ok(())
    }
}

pub struct ModelUtils;

impl ModelUtils {
    /// Check if model directory contains all required files
    pub fn validate_model_directory(model_path: &str) -> Result<bool> {
        let required_files = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
        ];

        for file in &required_files {
            let file_path = Path::new(model_path).join(file);
            if !file_path.exists() {
                tracing::warn!("Missing required model file: {}", file);
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Get model size in parameters (approximate)
    pub fn estimate_model_size(config: &Gpt2Config) -> u64 {
        let embedding_params = config.vocab_size * config.n_embd;
        let attention_params = config.n_layer * config.n_embd * config.n_embd * 4; // QKV + output
        let feedforward_params = config.n_layer * config.n_embd * config.n_inner.unwrap_or(4 * config.n_embd) * 2;
        let layer_norm_params = config.n_layer * config.n_embd * 2; // 2 layer norms per layer
        let final_layer_norm = config.n_embd;

        (embedding_params + attention_params + feedforward_params + layer_norm_params + final_layer_norm) as u64
    }

    /// Create optimized config for fast CPU training
    pub fn create_fast_config(vocab_size: usize) -> Gpt2Config {
        Gpt2Config {
            vocab_size,
            n_positions: 512,     // Shorter context for speed
            n_embd: 512,         // Smaller embedding
            n_layer: 8,          // Fewer layers
            n_head: 8,           // Fewer heads
            n_inner: Some(2048), // Smaller FFN
            activation_function: Activation::Gelu,
            resid_pdrop: 0.1,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            scale_attn_weights: true,
            use_cache: true,
            scale_attn_by_inverse_layer_idx: false,
            reorder_and_upcast_attn: false,
        }
    }

    /// Create quality config for better performance (larger model)
    pub fn create_quality_config(vocab_size: usize) -> Gpt2Config {
        Gpt2Config {
            vocab_size,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_inner: Some(3072),
            activation_function: Activation::Gelu,
            resid_pdrop: 0.1,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            scale_attn_weights: true,
            use_cache: true,
            scale_attn_by_inverse_layer_idx: false,
            reorder_and_upcast_attn: false,
        }
    }
}