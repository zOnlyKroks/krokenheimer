use candle_core::{Device, Tensor, Result as CandleResult};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, loss::cross_entropy, Activation, ParamsAdamW};
use tokenizers::Tokenizer;
use anyhow::{Result, Context, anyhow};
use std::path::Path;
use serde_json;
use std::collections::HashMap;
use ahash::AHashMap;
use crate::model::Gpt2Config;
use crate::inference::SimpleTransformer;

pub struct TrainingService {
    device: Device,
}

#[derive(serde::Deserialize)]
struct TrainingMessage {
    role: String,
    content: String,
}

#[derive(serde::Deserialize)]
struct ConversationData {
    messages: Vec<TrainingMessage>,
}

impl TrainingService {
    pub fn new() -> Self {
        let device = Device::Cpu;
        tracing::info!("Training service initialized with device: {:?}", device);

        std::env::set_var("RAYON_NUM_THREADS", num_cpus::get().to_string());
        tracing::info!("Utilizing {} CPU threads for parallel processing", num_cpus::get());

        Self { device }
    }

    pub fn train(&mut self, training_data_path: &str, output_path: &str, epochs: u32) -> Result<()> {
        tracing::info!("Starting training with {} epochs", epochs);

        let conversations = self.load_training_data(training_data_path)?;
        tracing::info!("Loaded {} conversations", conversations.len());

        // Create and test tokenizer FIRST
        let tokenizer = self.create_simple_tokenizer(&conversations, output_path)?;

        // Test tokenizer immediately
        tracing::info!("Testing tokenizer...");
        if let Ok(encoding) = tokenizer.encode("Hello", true) {
            let tokens = encoding.get_tokens();
            tracing::info!("Tokenizer test: 'Hello' -> {} tokens: {:?}",
                          tokens.len(), tokens);
        }

        // Continue with training...
        let tokenized_data = self.tokenize_conversations(&conversations, &tokenizer)?;

        let validation_split = 0.2;
        let split_index = ((tokenized_data.len() as f32) * (1.0 - validation_split)) as usize;
        let (train_data, val_data) = tokenized_data.split_at(split_index);

        tracing::info!("Data split: {} training, {} validation",
                      train_data.len(), val_data.len());

        let config = self.create_model_config(tokenizer.get_vocab_size(false));

        let var_map = match self.load_existing_weights_if_available(output_path)? {
            Some(loaded_var_map) => {
                tracing::info!("✅ Continuing from checkpoint");
                loaded_var_map
            }
            None => {
                tracing::info!("🆕 Starting from scratch");
                VarMap::new()
            }
        };

        let var_builder = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &self.device);
        let weights = HashMap::new();
        let model = SimpleTransformer::load(&weights, &config, var_builder)?;

        self.train_model(&model, &var_map, train_data, val_data, epochs, &config)?;
        self.save_model(&model, &tokenizer, &config, &var_map, output_path)?;

        tracing::info!("Training completed successfully!");
        Ok(())
    }

    // FIXED: Create a working character-level tokenizer
    fn create_simple_tokenizer(&self, conversations: &[ConversationData], output_path: &str) -> Result<Tokenizer> {
        tracing::info!("Creating character-level tokenizer...");

        // Collect ALL characters from conversations
        let mut chars = std::collections::HashSet::new();

        // Add basic ASCII first
        for c in ' '..='~' {  // ASCII printable range
            chars.insert(c);
        }

        // Add characters from your data
        for conv in conversations {
            for message in &conv.messages {
                let text = format!("{}: {}", message.role, message.content);
                for c in text.chars() {
                    chars.insert(c);
                }
            }
        }

        // Convert to sorted vector for deterministic ordering
        let mut sorted_chars: Vec<char> = chars.into_iter().collect();
        sorted_chars.sort();

        // Build vocabulary mapping
        let mut vocab = AHashMap::new();
        vocab.insert("[PAD]".to_string(), 0u32);
        vocab.insert("[UNK]".to_string(), 1u32);
        vocab.insert("<|endoftext|>".to_string(), 2u32);

        let mut id = 3u32;
        for &c in &sorted_chars {
            vocab.insert(c.to_string(), id);
            id += 1;
        }

        tracing::info!("Vocabulary size: {}", vocab.len());

        // Create a SIMPLE tokenizer - using BPE with character-level tokens
        let mut tokenizer = Tokenizer::new(
            tokenizers::models::bpe::BPE::builder()
                .vocab_and_merges(vocab, vec![]) // No merges = character-level
                .unk_token("[UNK]".to_string())
                .build()
                .map_err(|e| anyhow!("Failed to build BPE tokenizer: {}", e))?
        );

        // Use byte-level pre-tokenizer which handles all characters
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::byte_level::ByteLevel::default()));

        // Save tokenizer
        let tokenizer_path = Path::new(output_path).join("tokenizer.json");
        std::fs::create_dir_all(output_path)
            .with_context(|| format!("Failed to create output directory: {}", output_path))?;

        tokenizer.save(&tokenizer_path, false)
            .map_err(|e| anyhow!("Failed to save tokenizer: {}", e))?;

        tracing::info!("✅ Tokenizer saved to: {}", tokenizer_path.display());

        // Test it right away
        let test_texts = ["Hello", "こんにちは", "Hello world!", "User: Hello\nAssistant: Hi there"];
        for test_text in test_texts.iter() {
            if let Ok(encoding) = tokenizer.encode(*test_text, true) {
                let tokens = encoding.get_tokens();
                let ids = encoding.get_ids();
                tracing::info!("Test '{}' -> {} tokens: {:?} (ids: {:?})",
                              test_text, tokens.len(), tokens, ids);
            }
        }

        Ok(tokenizer)
    }


    // Rest of your methods remain mostly the same, but ensure tokenization works:
    fn tokenize_conversations(&self, conversations: &[ConversationData], tokenizer: &Tokenizer) -> Result<Vec<Vec<u32>>> {
        tracing::info!("Tokenizing conversations...");

        let mut tokenized_data = Vec::new();
        let mut total_tokens = 0;

        for (i, conv) in conversations.iter().enumerate() {
            let mut text = String::new();
            for message in &conv.messages {
                text.push_str(&format!("{}: {}\n", message.role, message.content));
            }
            text.push_str("<|endoftext|>");

            match tokenizer.encode(text, false) {
                Ok(encoding) => {
                    let token_ids = encoding.get_ids().to_vec();
                    if !token_ids.is_empty() {
                        total_tokens += token_ids.len();
                        tokenized_data.push(token_ids);
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to tokenize conversation {}: {}", i, e);
                    // Try with individual messages if whole conversation fails
                    let mut all_message_ids = Vec::new();
                    for message in &conv.messages {
                        let msg_text = format!("{}: {}", message.role, message.content);
                        if let Ok(msg_encoding) = tokenizer.encode(msg_text, false) {
                            all_message_ids.extend(msg_encoding.get_ids());
                        }
                    }
                    if !all_message_ids.is_empty() {
                        all_message_ids.push(2); // <|endoftext|> token
                        total_tokens += all_message_ids.len();
                        tokenized_data.push(all_message_ids);
                    }
                }
            }
        }

        tracing::info!("Tokenized {} conversations, total tokens: {}",
                      tokenized_data.len(), total_tokens);

        // Analyze token distribution
        let avg_tokens = total_tokens as f32 / tokenized_data.len() as f32;
        let max_tokens = tokenized_data.iter().map(|v| v.len()).max().unwrap_or(0);
        let min_tokens = tokenized_data.iter().map(|v| v.len()).min().unwrap_or(0);

        tracing::info!("Token stats: avg={:.1}, min={}, max={}",
                      avg_tokens, min_tokens, max_tokens);

        Ok(tokenized_data)
    }

    // Fixed prepare_batch to handle padding correctly
    fn prepare_batch(&self, batch: &[Vec<u32>], max_length: usize) -> Result<(Tensor, Tensor)> {
        if batch.is_empty() {
            return Ok((
                Tensor::zeros((1, 1), candle_core::DType::U32, &self.device)?,
                Tensor::zeros((1, 1), candle_core::DType::U32, &self.device)?
            ));
        }

        let batch_size = batch.len();
        let mut padded_inputs = Vec::with_capacity(batch_size * max_length);
        let mut padded_labels = Vec::with_capacity(batch_size * max_length);

        for sequence in batch {
            let seq_len = sequence.len().min(max_length);

            if seq_len < 2 {
                // Skip sequences that are too short for training
                continue;
            }

            // Input: everything except last token
            let input_len = (seq_len - 1).min(max_length - 1);
            for &token in sequence[..input_len].iter() {
                padded_inputs.push(token);
            }
            // Pad input
            for _ in input_len..max_length {
                padded_inputs.push(0); // [PAD] token
            }

            // Labels: everything except first token (shifted by 1)
            let label_len = seq_len - 1;
            for &token in sequence[1..seq_len].iter() {
                padded_labels.push(token);
            }
            // Pad labels
            for _ in label_len..max_length {
                padded_labels.push(0); // [PAD] token
            }
        }

        // Handle case where all sequences were too short
        if padded_inputs.is_empty() {
            return Ok((
                Tensor::zeros((1, 1), candle_core::DType::U32, &self.device)?,
                Tensor::zeros((1, 1), candle_core::DType::U32, &self.device)?
            ));
        }

        let input_tensor = Tensor::from_vec(
            padded_inputs,
            (batch_size, max_length),
            &self.device
        )?;

        let label_tensor = Tensor::from_vec(
            padded_labels,
            (batch_size, max_length),
            &self.device
        )?;

        Ok((input_tensor, label_tensor))
    }

    // Add back missing methods
    fn load_training_data(&self, file_path: &str) -> Result<Vec<ConversationData>> {
        tracing::info!("Loading training data from: {}", file_path);

        let content = std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read training data from {}", file_path))?;

        let mut conversations = Vec::new();
        let mut filtered_count = 0;

        for (line_num, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<ConversationData>(line) {
                Ok(conv_data) => {
                    // Apply minimal filtering
                    if !conv_data.messages.is_empty() {
                        conversations.push(conv_data);
                    } else {
                        filtered_count += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!("Skipping invalid JSON on line {}: {}", line_num + 1, e);
                    continue;
                }
            }
        }

        if conversations.is_empty() {
            return Err(anyhow!("No valid conversations found in training data"));
        }

        tracing::info!("Data quality: {} conversations retained, {} filtered out", conversations.len(), filtered_count);
        Ok(conversations)
    }

    fn create_model_config(&self, vocab_size: usize) -> Gpt2Config {
        Gpt2Config {
            vocab_size,
            n_positions: 2048,   // Maximum sequence length
            n_embd: 768,         // Standard embedding size
            n_layer: 12,         // Reasonable depth
            n_head: 12,          // Reduced from 16 attention heads
            n_inner: Some(3072), // Reduced from 4096 (4x embedding)
            activation_function: Activation::Gelu,
            resid_pdrop: 0.1,    // Standard dropout for stable training
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

    fn train_model(
        &self,
        model: &SimpleTransformer,
        var_map: &VarMap,
        train_data: &[Vec<u32>],
        val_data: &[Vec<u32>],
        epochs: u32,
        _config: &Gpt2Config,
    ) -> Result<()> {
        tracing::info!("Starting model training...");

        let batch_size = 16;  // Safe batch size to prevent memory issues
        let max_sequence_length = 1024;  // Longer sequences for better context understanding

        // Create optimizer with current API
        let mut optimizer = AdamW::new(
            var_map.all_vars(),
            ParamsAdamW {
                lr: 5e-5,  // Conservative learning rate to prevent divergence
                beta1: 0.9,   // Standard momentum parameter
                beta2: 0.999, // Standard second moment decay
                eps: 1e-8,
                weight_decay: 0.01, // Standard weight decay for stability
            }
        )?;

        let total_batches = (train_data.len() + batch_size - 1) / batch_size;
        tracing::info!("Training configuration: batch_size={}, sequence_length={}, total_batches={}",
                      batch_size, max_sequence_length, total_batches);

        // Early stopping variables
        let mut best_val_loss = f32::INFINITY;
        let mut patience_counter = 0;
        let patience = 3; // Stop after 3 epochs without improvement

        for epoch in 1..=epochs {
            tracing::info!("Epoch {}/{} - Processing {} batches", epoch, epochs, total_batches);
            let mut total_loss = 0.0;
            let mut batch_count = 0;
            let epoch_start = std::time::Instant::now();

            // Training phase
            for batch_start in (0..train_data.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(train_data.len());
                let batch = &train_data[batch_start..batch_end];

                // Prepare batch tensors
                let (input_ids, labels) = self.prepare_batch(batch, max_sequence_length)?;

                // Check if batch is empty by checking tensor dimensions
                let batch_dims = input_ids.dims();
                if batch_dims.len() == 0 || batch_dims.iter().any(|&d| d == 0) {
                    continue;
                }

                // Forward pass
                let logits = model.forward(&input_ids)?;

                // Calculate loss
                let loss = self.calculate_loss(&logits, &labels)?;
                total_loss += loss.to_scalar::<f32>()?;
                batch_count += 1;

                // Backward pass
                let grads = loss.backward()?;
                optimizer.step(&grads)?;

                if batch_count % 10 == 0 {
                    let elapsed = epoch_start.elapsed();
                    let batches_per_sec = batch_count as f64 / elapsed.as_secs_f64();
                    let progress = batch_count as f64 / total_batches as f64 * 100.0;
                    tracing::info!("Batch {}/{} ({:.1}%): loss={:.4}, {:.2} batches/sec",
                                  batch_count, total_batches, progress, loss.to_scalar::<f32>()?, batches_per_sec);
                }
            }

            let epoch_duration = epoch_start.elapsed();
            let avg_train_loss = total_loss / batch_count as f32;
            let batches_per_sec = batch_count as f64 / epoch_duration.as_secs_f64();

            // Validation phase
            let val_loss = self.evaluate_on_validation_set(model, val_data, batch_size, max_sequence_length)?;

            let remaining_epochs = epochs - epoch;
            let estimated_remaining = epoch_duration * remaining_epochs;

            tracing::info!("Epoch {} completed in {:.1}s. Train Loss: {:.4}, Val Loss: {:.4}, Speed: {:.1} batches/sec, ETA: {:.1}min",
                          epoch, epoch_duration.as_secs_f64(), avg_train_loss, val_loss, batches_per_sec,
                          estimated_remaining.as_secs_f64() / 60.0);

            // Early stopping check
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience_counter = 0;
                tracing::info!("✅ New best validation loss: {:.4}", best_val_loss);
            } else {
                patience_counter += 1;
                tracing::warn!("⚠️ Validation loss did not improve. Patience: {}/{}", patience_counter, patience);

                if patience_counter >= patience {
                    tracing::info!("🛑 Early stopping triggered after {} epochs without improvement", patience);
                    break;
                }
            }
        }

        Ok(())
    }

    fn evaluate_on_validation_set(
        &self,
        model: &SimpleTransformer,
        val_data: &[Vec<u32>],
        batch_size: usize,
        max_sequence_length: usize,
    ) -> Result<f32> {
        let mut total_val_loss = 0.0;
        let mut val_batch_count = 0;

        // Process validation data in batches (no gradient computation)
        for batch_start in (0..val_data.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(val_data.len());
            let batch = &val_data[batch_start..batch_end];

            // Prepare batch tensors
            let (input_ids, labels) = self.prepare_batch(batch, max_sequence_length)?;

            // Check if batch is valid
            let batch_dims = input_ids.dims();
            if batch_dims.len() == 0 || batch_dims.iter().any(|&d| d == 0) {
                continue;
            }

            // Forward pass only (no backward pass for validation)
            let logits = model.forward(&input_ids)?;

            // Calculate validation loss
            let loss = self.calculate_loss(&logits, &labels)?;
            total_val_loss += loss.to_scalar::<f32>()?;
            val_batch_count += 1;
        }

        if val_batch_count == 0 {
            Ok(f32::INFINITY) // Return high loss if no valid batches
        } else {
            Ok(total_val_loss / val_batch_count as f32)
        }
    }

    fn load_existing_weights_if_available(&self, output_path: &str) -> Result<Option<VarMap>> {
        let standard_model_path = Path::new(output_path).join("model.safetensors");

        if standard_model_path.exists() {
            tracing::info!("Found existing trained model at: {}", standard_model_path.display());
            tracing::info!("Loading model checkpoint for continued training...");

            // Load the VarMap from safetensors
            let mut var_map = VarMap::new();
            match var_map.load(standard_model_path) {
                Ok(()) => {
                    tracing::info!("✅ Successfully loaded model checkpoint!");
                    tracing::info!("Continuing training from existing weights");
                    return Ok(Some(var_map));
                }
                Err(e) => {
                    tracing::error!("❌ Failed to load model checkpoint: {}", e);
                    tracing::warn!("Starting fresh training instead");
                }
            }
        }

        tracing::info!("No existing model found - starting fresh training");
        Ok(None)
    }

    fn save_model(&self, _model: &SimpleTransformer, tokenizer: &Tokenizer, config: &Gpt2Config, var_map: &VarMap, output_path: &str) -> Result<()> {
        tracing::info!("Saving model...");

        // Create output directory
        std::fs::create_dir_all(output_path)
            .with_context(|| format!("Failed to create output directory: {}", output_path))?;

        // Save weights
        let weights_path = Path::new(output_path).join("model.safetensors");
        var_map.save(&weights_path)
            .map_err(|e| anyhow!("Failed to save model weights: {}", e))?;

        // Save config
        let config_path = Path::new(output_path).join("config.json");
        let config_json = serde_json::to_string_pretty(config)
            .with_context(|| "Failed to serialize config")?;
        std::fs::write(&config_path, config_json)
            .with_context(|| format!("Failed to write config to {}", config_path.display()))?;

        // Save tokenizer
        let tokenizer_path = Path::new(output_path).join("tokenizer.json");
        tokenizer.save(&tokenizer_path, false)
            .map_err(|e| anyhow!("Failed to save tokenizer: {}", e))?;

        tracing::info!("✅ Model saved successfully to: {}", output_path);
        Ok(())
    }

    fn calculate_loss(&self, logits: &Tensor, labels: &Tensor) -> CandleResult<Tensor> {
        // Reshape for cross-entropy loss
        let (batch_size, seq_len, vocab_size) = logits.dims3()?;
        let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
        let labels_flat = labels.flatten_all()?;

        // Calculate cross-entropy loss
        cross_entropy(&logits_flat, &labels_flat)
    }
}