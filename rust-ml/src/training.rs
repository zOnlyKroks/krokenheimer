use candle_core::{Device, Tensor, Result as CandleResult, DType};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, loss::cross_entropy, Activation, ParamsAdamW};
use tokenizers::Tokenizer;
use anyhow::{Result, Context, anyhow};
use std::path::Path;
use serde_json;
use std::collections::HashMap;
use crate::model::Gpt2Config;
use crate::inference::SimpleTransformer;
use crate::bpe_wrapper::BPETokenizerWrapper;

pub struct TrainingService {
    device: Device,
}

#[derive(serde::Deserialize)]
pub struct TrainingMessage {
    pub role: String,
    pub content: String,
}

#[derive(serde::Deserialize)]
pub struct ConversationData {
    pub messages: Vec<TrainingMessage>,
}

impl TrainingService {
    pub fn new() -> Self {
        // Try GPU first, fall back to CPU
        let device = Device::cuda_if_available(0)
            .unwrap_or_else(|_| {
                tracing::info!("No GPU available, using CPU");
                Device::Cpu
            });

        // Optimize CPU usage
        std::env::set_var("RAYON_NUM_THREADS", std::thread::available_parallelism()
            .map(|n| n.get().to_string())
            .unwrap_or_else(|_| "1".to_string()));

        tracing::info!("Training service initialized with device: {:?}", device);
        tracing::info!("Utilizing {} CPU threads", std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1));

        Self { device }
    }

    pub fn train(&mut self, training_data_path: &str, output_path: &str, epochs: u32) -> Result<()> {
        tracing::info!("Starting training with {} epochs", epochs);

        // Load and preprocess data
        let conversations = self.load_training_data(training_data_path)?;
        tracing::info!("Loaded {} conversations", conversations.len());

        // Create or load tokenizer
        let tokenizer = self.create_or_load_tokenizer(&conversations, output_path)?;

        // Tokenize efficiently
        let tokenized_data = self.tokenize_conversations_batch(&conversations, &tokenizer)?;

        // Split data
        let validation_split = 0.15;
        let split_index = ((tokenized_data.len() as f32) * (1.0 - validation_split)) as usize;
        let (train_data, val_data) = tokenized_data.split_at(split_index);

        tracing::info!("Data split: {} training, {} validation sequences",
                      train_data.len(), val_data.len());

        // Create optimized model config
        let config = self.create_optimized_config(tokenizer.get_vocab_size(false));

        // Load or create model
        let (var_map, is_continuing) = self.load_or_create_weights(output_path)?;
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &self.device);
        let weights = HashMap::new();
        let model = SimpleTransformer::load(&weights, &config, var_builder)?;

        // Train with optimizations
        self.train_model_optimized(&model, &var_map, train_data, val_data, epochs, &config, is_continuing, output_path)?;

        // Save model
        self.save_model(&model, &tokenizer, &config, &var_map, output_path)?;

        tracing::info!("Training completed successfully!");
        Ok(())
    }

    fn create_or_load_tokenizer(&self, conversations: &[ConversationData], output_path: &str) -> Result<BPETokenizerWrapper> {
        let bpe_tokenizer_path = Path::new(output_path).join("bpe_tokenizer.json");
        let tokenizer_json_path = Path::new(output_path).join("tokenizer.json");

        // Try to load existing BPE tokenizer first
        if bpe_tokenizer_path.exists() {
            tracing::info!("Loading existing BPE tokenizer from: {}", bpe_tokenizer_path.display());
            return BPETokenizerWrapper::load(output_path);
        }

        // Try to load existing tokenizer.json for backward compatibility
        if tokenizer_json_path.exists() {
            tracing::info!("Found existing tokenizer.json, attempting to load...");
            if let Ok(old_tokenizer) = Tokenizer::from_file(&tokenizer_json_path) {
                tracing::info!("Loaded existing tokenizer with {} tokens", old_tokenizer.get_vocab_size(false));
                // Create a wrapper that mimics the old tokenizer for compatibility
                return self.create_wrapper_from_existing_tokenizer(old_tokenizer);
            }
        }

        tracing::info!("🚀 Training custom BPE tokenizer from scratch on {} conversations...", conversations.len());

        // Determine target vocabulary size based on data size
        let target_vocab_size = self.calculate_optimal_vocab_size(conversations);

        tracing::info!("Target vocabulary size: {}", target_vocab_size);

        // Train our custom BPE tokenizer
        let tokenizer_wrapper = BPETokenizerWrapper::train_and_save(conversations, output_path, target_vocab_size)?;

        let final_vocab_size = tokenizer_wrapper.get_vocab_size(false);
        tracing::info!("✅ Custom BPE tokenizer trained and saved with {} tokens", final_vocab_size);
        tracing::info!("   - Vocabulary learned directly from your conversation data");
        tracing::info!("   - No pre-trained weights used - completely custom!");

        Ok(tokenizer_wrapper)
    }

    fn calculate_optimal_vocab_size(&self, conversations: &[ConversationData]) -> usize {
        // More sophisticated vocabulary size calculation
        let mut total_chars = 0;
        let mut unique_tokens = std::collections::HashSet::new();
        let mut unique_chars = std::collections::HashSet::new();
        let mut total_messages = 0;

        for conv in conversations {
            for message in &conv.messages {
                total_chars += message.content.len();
                total_messages += 1;

                // Collect all unique characters
                for ch in message.content.chars() {
                    unique_chars.insert(ch);
                }

                // Better tokenization: split on whitespace AND punctuation
                let content = &message.content;
                let mut current_token = String::new();

                for ch in content.chars() {
                    if ch.is_alphanumeric() || ch == '_' || ch == '@' || ch == '#' {
                        current_token.push(ch);
                    } else {
                        // End of alphanumeric token
                        if !current_token.is_empty() {
                            unique_tokens.insert(current_token.clone());
                            current_token.clear();
                        }
                        // Also add the punctuation/symbol as a token if it's significant
                        if !ch.is_whitespace() {
                            unique_tokens.insert(ch.to_string());
                        }
                    }
                }
                // Don't forget the last token
                if !current_token.is_empty() {
                    unique_tokens.insert(current_token);
                }

                // Also add case variations for common words
                for word in content.split_whitespace() {
                    if word.len() > 2 {
                        unique_tokens.insert(word.to_string()); // Original case
                        unique_tokens.insert(word.to_lowercase()); // Lowercase
                        if word.chars().next().unwrap().is_lowercase() {
                            unique_tokens.insert(format!("{}{}",
                                word.chars().next().unwrap().to_uppercase().to_string(),
                                &word[1..])); // Title case
                        }
                    }
                }
            }
        }

        let unique_token_count = unique_tokens.len();
        let unique_char_count = unique_chars.len();
        let avg_message_length = total_chars as f64 / total_messages.max(1) as f64;

        tracing::info!("📊 Vocabulary analysis:");
        tracing::info!("  - Unique tokens (better counting): {}", unique_token_count);
        tracing::info!("  - Unique characters: {}", unique_char_count);
        tracing::info!("  - Total characters: {}", total_chars);
        tracing::info!("  - Average message length: {:.1} chars", avg_message_length);

        // Base vocabulary: Unicode chars + byte representations
        let base_size = 1000.max(unique_char_count * 2);

        // Scale based on unique tokens (more realistic now)
        let token_factor = (unique_token_count as f64 * 0.3).min(10000.0) as usize;

        // Scale based on corpus diversity
        let diversity_factor = if avg_message_length > 100.0 {
            // Longer messages suggest more complex vocabulary
            (total_chars as f64 / 8000.0).min(3000.0) as usize
        } else {
            // Shorter messages, more conservative
            (total_chars as f64 / 15000.0).min(1500.0) as usize
        };

        let calculated_size = base_size + token_factor + diversity_factor;

        // More generous bounds for chat data
        let final_size = calculated_size.max(3000).min(20000);

        tracing::info!("  - Calculated vocab size: {} (base: {}, tokens: {}, diversity: {})",
                      final_size, base_size, token_factor, diversity_factor);

        final_size
    }

    fn create_wrapper_from_existing_tokenizer(&self, _tokenizer: Tokenizer) -> Result<BPETokenizerWrapper> {
        // This is a fallback - if we find an old tokenizer.json, we can't easily convert it
        // to our BPE format, so we'll return an error and suggest retraining
        Err(anyhow!(
            "Found existing tokenizer.json but it's not compatible with the new BPE implementation. \
             Please delete the existing tokenizer files and retrain to use the new custom BPE tokenizer."
        ))
    }


    fn tokenize_conversations_batch(&self, conversations: &[ConversationData], tokenizer: &BPETokenizerWrapper) -> Result<Vec<Vec<u32>>> {
        tracing::info!("Tokenizing conversations in batch...");

        let mut tokenized_data = Vec::with_capacity(conversations.len());
        let mut total_tokens = 0;
        let mut skipped = 0;

        // Process conversations in chunks for better memory usage
        let chunk_size = 500;

        for chunk in conversations.chunks(chunk_size) {
            for conv in chunk {
                // Format conversation with special tokens
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

                match tokenizer.encode(&*formatted, false) {
                    Ok(encoding) => {
                        let token_ids = encoding.get_ids().to_vec();

                        // Filter out very short sequences and ensure reasonable length
                        if token_ids.len() >= 8 && token_ids.len() <= 512 {
                            total_tokens += token_ids.len();
                            tokenized_data.push(token_ids);
                        } else {
                            skipped += 1;
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Failed to tokenize conversation: {}", e);
                        skipped += 1;
                    }
                }
            }
        }

        // Sort by length for more efficient batching (optional but helps with padding efficiency)
        tokenized_data.sort_by_key(|seq| seq.len());

        tracing::info!("Tokenized {} conversations (skipped {}), total tokens: {}, avg length: {:.1}",
                     tokenized_data.len(), skipped, total_tokens,
                     total_tokens as f32 / tokenized_data.len().max(1) as f32);

        if tokenized_data.is_empty() {
            return Err(anyhow!("No valid tokenized conversations found"));
        }

        Ok(tokenized_data)
    }

    fn load_or_create_weights(&self, output_path: &str) -> Result<(VarMap, bool)> {
        let model_path = Path::new(output_path).join("model.safetensors");

        if model_path.exists() {
            tracing::info!("Found existing model at: {}", model_path.display());
            let mut var_map = VarMap::new();

            match var_map.load(&model_path) {
                Ok(()) => {
                    tracing::info!("✅ Successfully loaded model checkpoint");
                    return Ok((var_map, true));
                }
                Err(e) => {
                    tracing::warn!("Failed to load checkpoint: {}, starting fresh", e);
                }
            }
        }

        tracing::info!("🆕 Starting from scratch");
        Ok((VarMap::new(), false))
    }

    fn create_optimized_config(&self, vocab_size: usize) -> Gpt2Config {
        // Optimized model config for better quality
        Gpt2Config {
            vocab_size,
            n_positions: 512,
            n_embd: 384,
            n_layer: 8,
            n_head: 12,
            n_inner: Some(1024),
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

    fn train_model_optimized(
        &self,
        model: &SimpleTransformer,
        var_map: &VarMap,
        train_data: &[Vec<u32>],
        val_data: &[Vec<u32>],
        epochs: u32,
        config: &Gpt2Config,
        is_continuing: bool,
        output_path: &str,
    ) -> Result<()> {
        tracing::info!("Starting model training...");

        // Optimized hyperparameters
        let batch_size = 8;  // Smaller batches for better gradient estimates
        let max_sequence_length = config.n_positions.min(384);

        // Calculate total batches
        let total_batches = (train_data.len() + batch_size - 1) / batch_size;

        tracing::info!("Training config: batch_size={}, seq_len={}, total_batches={}",
                      batch_size, max_sequence_length, total_batches);

        // Create optimizer
        let mut optimizer = AdamW::new(
            var_map.all_vars(),
            ParamsAdamW {
                lr: if is_continuing { 5e-6 } else { 1e-4 },
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            }
        )?;

        // Early stopping
        let mut best_val_loss = f32::INFINITY;
        let mut patience_counter = 0;
        let patience = 4;

        for epoch in 1..=epochs {
            tracing::info!("Epoch {}/{} - Processing {} batches", epoch, epochs, total_batches);

            let epoch_start = std::time::Instant::now();
            let mut total_loss = 0.0;
            let mut processed_batches = 0;

            // Shuffle training data each epoch
            let mut train_indices: Vec<usize> = (0..train_data.len()).collect();
            fastrand::shuffle(&mut train_indices);

            // Training loop
            for batch_idx in 0..total_batches {
                let batch_start = batch_idx * batch_size;
                let batch_end = (batch_start + batch_size).min(train_indices.len());

                // Get batch indices
                let batch_indices = &train_indices[batch_start..batch_end];

                // Prepare batch
                let batch_data: Vec<&Vec<u32>> = batch_indices.iter()
                    .map(|&idx| &train_data[idx])
                    .collect();

                let (input_ids, labels) = self.prepare_batch_fast(&batch_data, max_sequence_length)?;

                // Skip if batch is invalid
                if input_ids.dims().iter().any(|&d| d == 0) {
                    continue;
                }

                // Forward pass
                let logits = model.forward(&input_ids)?;

                // Calculate loss
                let loss = self.calculate_loss_optimized(&logits, &labels)?;
                let loss_value = loss.to_scalar::<f32>()?;
                total_loss += loss_value;
                processed_batches += 1;

                // Backward pass
                let grads = loss.backward()?;
                optimizer.step(&grads)?;

                // Log progress
                if processed_batches % (total_batches.max(1) / 20).max(1) == 0 {
                    let elapsed = epoch_start.elapsed();
                    let batches_per_sec = processed_batches as f64 / elapsed.as_secs_f64();
                    let progress = (processed_batches as f64 / total_batches as f64) * 100.0;

                    tracing::info!("Batch {}/{} ({:.1}%): loss={:.4}, {:.2} batches/sec",
                                  processed_batches, total_batches, progress, loss_value, batches_per_sec);
                }
            }

            let epoch_duration = epoch_start.elapsed();
            let avg_train_loss = if processed_batches > 0 {
                total_loss / processed_batches as f32
            } else {
                0.0
            };

            // Validation
            let val_loss = self.evaluate_validation_fast(model, val_data, batch_size, max_sequence_length)?;

            let batches_per_sec = processed_batches as f64 / epoch_duration.as_secs_f64();
            let remaining_epochs = epochs - epoch;
            let eta_minutes = (epoch_duration * remaining_epochs).as_secs_f64() / 60.0;

            tracing::info!("Epoch {} completed in {:.1}s. Train Loss: {:.4}, Val Loss: {:.4}, Speed: {:.1} batches/sec, ETA: {:.1}min",
                          epoch, epoch_duration.as_secs_f64(), avg_train_loss, val_loss,
                          batches_per_sec, eta_minutes);

            // Early stopping check
            if val_loss < best_val_loss - 0.001 {
                best_val_loss = val_loss;
                patience_counter = 0;
                tracing::info!("✅ New best validation loss: {:.4}", best_val_loss);

                // Save checkpoint on improvement
                let checkpoint_path = Path::new(output_path).join(format!("checkpoint_epoch_{}.safetensors", epoch));
                if let Err(e) = var_map.save(&checkpoint_path) {
                    tracing::warn!("Failed to save checkpoint: {}", e);
                } else {
                    tracing::info!("💾 Checkpoint saved: {}", checkpoint_path.display());
                }
            } else {
                patience_counter += 1;
                tracing::warn!("⚠️ No improvement. Patience: {}/{}", patience_counter, patience);

                if patience_counter >= patience {
                    tracing::info!("🛑 Early stopping at epoch {}", epoch);
                    break;
                }
            }
        }

        Ok(())
    }

    fn prepare_batch_fast(&self, batch: &[&Vec<u32>], max_length: usize) -> Result<(Tensor, Tensor)> {
        if batch.is_empty() {
            return Ok((
                Tensor::zeros((1, 1), DType::U32, &self.device)?,
                Tensor::zeros((1, 1), DType::U32, &self.device)?
            ));
        }

        let batch_size = batch.len();
        let mut inputs = Vec::with_capacity(batch_size * max_length);
        let mut labels = Vec::with_capacity(batch_size * max_length);

        for sequence in batch {
            let seq_len = sequence.len().min(max_length);

            if seq_len < 2 {
                // Pad with zeros for sequences that are too short
                for _ in 0..max_length {
                    inputs.push(0);
                    labels.push(0);
                }
                continue;
            }

            // Both inputs and labels should have the same length
            let actual_len = (seq_len - 1).min(max_length);

            // Input: indices 0..actual_len (all except last token)
            for &token in &sequence[..actual_len] {
                inputs.push(token);
            }
            for _ in actual_len..max_length {
                inputs.push(0); // Pad
            }

            // Labels: indices 1..1+actual_len (all except first token)
            for &token in &sequence[1..1+actual_len] {
                labels.push(token);
            }
            for _ in actual_len..max_length {
                labels.push(0); // Pad
            }
        }

        let input_tensor = Tensor::from_vec(inputs, (batch_size, max_length), &self.device)?;
        let label_tensor = Tensor::from_vec(labels, (batch_size, max_length), &self.device)?;

        Ok((input_tensor, label_tensor))
    }

    fn evaluate_validation_fast(
        &self,
        model: &SimpleTransformer,
        val_data: &[Vec<u32>],
        batch_size: usize,
        max_length: usize,
    ) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        // Use smaller batch for more thorough validation
        let val_batch_size = batch_size.min(4);

        for chunk in val_data.chunks(val_batch_size) {
            let batch_refs: Vec<&Vec<u32>> = chunk.iter().collect();
            let (input_ids, labels) = self.prepare_batch_fast(&batch_refs, max_length)?;

            if input_ids.dims().iter().any(|&d| d == 0) {
                continue;
            }

            let logits = model.forward(&input_ids)?;
            let loss = self.calculate_loss_optimized(&logits, &labels)?;

            total_loss += loss.to_scalar::<f32>()?;
            batch_count += 1;
        }

        if batch_count == 0 {
            Ok(f32::INFINITY)
        } else {
            Ok(total_loss / batch_count as f32)
        }
    }

    fn calculate_loss_optimized(&self, logits: &Tensor, labels: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, seq_len, vocab_size) = logits.dims3()?;

        // Reshape for loss calculation
        let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
        let labels_flat = labels.flatten_all()?;

        // Use cross entropy
        cross_entropy(&logits_flat, &labels_flat)
    }

    fn load_training_data(&self, file_path: &str) -> Result<Vec<ConversationData>> {
        tracing::info!("Loading training data from: {}", file_path);

        let content = std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read training data from {}", file_path))?;

        let mut conversations = Vec::new();
        let mut filtered = 0;

        for (line_num, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<ConversationData>(line) {
                Ok(conv) => {
                    // Filter: require at least 2 messages and reasonable content
                    let has_content = conv.messages.iter()
                        .any(|msg| msg.content.len() > 5);

                    if conv.messages.len() >= 2 && has_content {
                        conversations.push(conv);
                    } else {
                        filtered += 1;
                    }
                }
                Err(e) => {
                    if line_num < 5 { // Only warn for first few errors
                        tracing::debug!("Invalid JSON on line {}: {}", line_num + 1, e);
                    }
                }
            }
        }

        tracing::info!("Loaded {} conversations (filtered {} invalid)", conversations.len(), filtered);

        if conversations.is_empty() {
            return Err(anyhow!("No valid conversations found"));
        }

        Ok(conversations)
    }

    fn save_model(&self, _model: &SimpleTransformer, tokenizer: &BPETokenizerWrapper, config: &Gpt2Config, var_map: &VarMap, output_path: &str) -> Result<()> {
        tracing::info!("Saving final model...");

        std::fs::create_dir_all(output_path)
            .with_context(|| format!("Failed to create output directory: {}", output_path))?;

        // Save weights
        let weights_path = Path::new(output_path).join("model.safetensors");
        var_map.save(&weights_path)
            .map_err(|e| anyhow!("Failed to save weights: {}", e))?;

        // Save config
        let config_path = Path::new(output_path).join("config.json");
        let config_json = serde_json::to_string_pretty(config)?;
        std::fs::write(&config_path, config_json)
            .with_context(|| format!("Failed to write config to {}", config_path.display()))?;

        // Save tokenizer (BPE tokenizer is already saved during training)
        let bpe_tokenizer_path = Path::new(output_path).join("bpe_tokenizer.json");
        tokenizer.tokenizer.save(&bpe_tokenizer_path)
            .map_err(|e| anyhow!("Failed to save BPE tokenizer: {}", e))?;

        // Also save compatible tokenizer.json
        tokenizer.save_compatible_format(output_path)
            .map_err(|e| anyhow!("Failed to save compatible tokenizer format: {}", e))?;

        tracing::info!("✅ Model saved to: {}", output_path);
        Ok(())
    }
}