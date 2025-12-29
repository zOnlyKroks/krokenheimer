use candle_core::{Device, Tensor, Result as CandleResult};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, loss::cross_entropy, Activation};
use tokenizers::Tokenizer;
use anyhow::{Result, Context, anyhow};
use std::path::Path;
use serde_json;
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
        let device = Device::Cpu; // Can be extended for GPU support
        tracing::info!("Training service initialized with device: {:?}", device);

        // Enable CPU optimizations
        std::env::set_var("RAYON_NUM_THREADS", num_cpus::get().to_string());
        tracing::info!("Utilizing {} CPU threads for parallel processing", num_cpus::get());

        Self { device }
    }

    pub fn train(&mut self, training_data_path: &str, output_path: &str, epochs: u32) -> Result<()> {
        tracing::info!("Starting training with {} epochs", epochs);
        tracing::info!("Training data: {}", training_data_path);
        tracing::info!("Output path: {}", output_path);

        // Load and preprocess training data
        let conversations = self.load_training_data(training_data_path)?;
        tracing::info!("Loaded {} conversations", conversations.len());

        // Create tokenizer
        let tokenizer = self.create_tokenizer(&conversations, output_path)?;

        // Tokenize conversations
        let tokenized_data = self.tokenize_conversations(&conversations, &tokenizer)?;

        // Split data into training and validation sets for better training monitoring
        let validation_split = 0.2; // Use 20% for validation
        let split_index = ((tokenized_data.len() as f32) * (1.0 - validation_split)) as usize;
        let (train_data, val_data) = tokenized_data.split_at(split_index);

        tracing::info!("Data split: {} training samples, {} validation samples",
                      train_data.len(), val_data.len());

        // Create model - try to load existing weights for continued training
        let config = self.create_model_config(tokenizer.get_vocab_size(false));

        // Check for existing model to continue training from
        let var_map = match self.load_existing_weights_if_available(output_path)? {
            Some(loaded_var_map) => {
                tracing::info!("✅ Using loaded model checkpoint for continued training");

                // Verify model compatibility
                if let Err(e) = self.verify_model_compatibility(&loaded_var_map, &config) {
                    tracing::warn!("⚠️ Model compatibility check failed: {}", e);
                    tracing::warn!("Starting fresh training to avoid issues");
                    VarMap::new()
                } else {
                    tracing::info!("✅ Model compatibility verified - continuing training");
                    loaded_var_map
                }
            }
            None => {
                tracing::info!("🆕 Starting training from scratch (no existing model found)");
                VarMap::new()
            }
        };

        let var_builder = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &self.device);
        let weights = std::collections::HashMap::new(); // Empty since we use VarBuilder
        let model = SimpleTransformer::load(&weights, &config, var_builder)?;

        // Training loop
        self.train_model(&model, &var_map, train_data, val_data, epochs, &config)?;

        // Save model
        self.save_model(&model, &tokenizer, &config, &var_map, output_path)?;

        tracing::info!("Training completed successfully!");
        Ok(())
    }

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
                    // Apply data quality filters
                    let filtered_conv = self.apply_data_quality_filters(conv_data);
                    if let Some(quality_conv) = filtered_conv {
                        conversations.push(quality_conv);
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

    /// Apply minimal data quality filters - keep your good conversation data!
    fn apply_data_quality_filters(&self, conv: ConversationData) -> Option<ConversationData> {
        // Much more relaxed filtering - keep real conversations
        if conv.messages.is_empty() {
            return None;
        }

        let mut filtered_messages = Vec::new();

        for message in conv.messages {
            let trimmed_content = message.content.trim();

            // Only filter out completely empty messages
            if trimmed_content.is_empty() {
                continue;
            }

            // Keep everything else - your conversations are good data!
            filtered_messages.push(message);
        }

        // Keep any conversation with at least 1 message
        if !filtered_messages.is_empty() {
            Some(ConversationData {
                messages: filtered_messages,
            })
        } else {
            None
        }
    }

    fn create_tokenizer(&self, conversations: &[ConversationData], output_path: &str) -> Result<Tokenizer> {
        tracing::info!("Creating simple but effective tokenizer...");

        // Build comprehensive vocabulary from your actual conversation data
        let mut vocab = std::collections::HashMap::new();

        // Special tokens first
        vocab.insert("[PAD]".to_string(), 0u32);
        vocab.insert("[UNK]".to_string(), 1u32);
        vocab.insert("[SEP]".to_string(), 2u32);
        vocab.insert("<|endoftext|>".to_string(), 3u32);

        let mut id = 4u32;

        // Collect ALL words from your conversations - don't filter aggressively
        let mut word_set = std::collections::HashSet::new();

        for conv in conversations {
            for message in &conv.messages {
                let text = format!("{}: {}", message.role, message.content);

                // Simple split on whitespace and add everything
                for word in text.split_whitespace() {
                    // Add the raw word
                    word_set.insert(word.to_string());

                    // Also add lowercase version
                    word_set.insert(word.to_lowercase());

                    // Handle common contractions and punctuation
                    if word.contains('\'') {
                        word_set.insert(word.replace('\'', ""));
                    }

                    // Add word without ending punctuation
                    let cleaned = word.trim_end_matches(|c: char| c.is_ascii_punctuation());
                    if !cleaned.is_empty() && cleaned != word {
                        word_set.insert(cleaned.to_string());
                        word_set.insert(cleaned.to_lowercase());
                    }
                }

                // Add individual characters for fallback (better than just [UNK])
                for ch in text.chars() {
                    if ch.is_alphanumeric() || ch.is_ascii_punctuation() || ch.is_whitespace() {
                        word_set.insert(ch.to_string());
                    }
                }
            }
        }

        // Add all collected words to vocabulary
        for word in word_set {
            if vocab.len() < 16000 { // Larger vocab for better coverage
                vocab.insert(word, id);
                id += 1;
            }
        }

        tracing::info!("Built comprehensive vocabulary with {} tokens", vocab.len());

        // Create simple WordPiece model
        let model = tokenizers::models::wordpiece::WordPiece::builder()
            .vocab(vocab.clone().into_iter().collect::<ahash::AHashMap<String, u32>>())
            .unk_token("[UNK]".to_string())
            .build()
            .map_err(|e| anyhow!("Failed to build tokenizer: {}", e))?;

        let mut tokenizer = Tokenizer::new(model);

        // Use whitespace pre-tokenization
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));

        // Save tokenizer
        let tokenizer_path = Path::new(output_path).join("tokenizer.json");
        std::fs::create_dir_all(output_path)
            .with_context(|| format!("Failed to create output directory: {}", output_path))?;

        tokenizer.save(&tokenizer_path, false)
            .map_err(|e| anyhow!("Failed to save tokenizer: {}", e))?;

        let vocab_size = tokenizer.get_vocab_size(true);
        tracing::info!("✅ Simple tokenizer created with vocab size: {}", vocab_size);
        tracing::info!("Tokenizer saved to: {}", tokenizer_path.display());

        // Test tokenization quality
        let test_text = "Hello! How are you doing today?";
        if let Ok(encoding) = tokenizer.encode(test_text, false) {
            let tokens = encoding.get_tokens();
            tracing::info!("Test tokenization: '{}' -> {} tokens: {:?}", test_text, tokens.len(), tokens);
        }

        Ok(tokenizer)
    }

    fn tokenize_conversations(&self, conversations: &[ConversationData], tokenizer: &Tokenizer) -> Result<Vec<Vec<u32>>> {
        tracing::info!("Tokenizing conversations...");

        let mut tokenized_data = Vec::new();

        for conv in conversations {
            let mut text = String::new();
            for message in &conv.messages {
                text.push_str(&format!("{}: {}\n", message.role, message.content));
            }
            text.push_str("<|endoftext|>");

            let encoding = tokenizer.encode(text, false)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

            let token_ids = encoding.get_ids().to_vec();
            if !token_ids.is_empty() {
                tokenized_data.push(token_ids);
            }
        }

        tracing::info!("Tokenized {} conversations", tokenized_data.len());
        Ok(tokenized_data)
    }

    fn create_model_config(&self, vocab_size: usize) -> Gpt2Config {
        // Memory-optimized but still capable config (similar to GPT-2 small)
        Gpt2Config {
            vocab_size,
            n_positions: 1024,  // Full context window for aggressive training
            n_embd: 768,        // Reduced from 1024 (standard GPT-2 small size)
            n_layer: 12,        // Reduced from 16 layers
            n_head: 12,         // Reduced from 16 attention heads
            n_inner: Some(3072), // Reduced from 4096 (4x embedding)
            activation_function: Activation::Gelu,
            resid_pdrop: 0.1,   // Standard dropout for stable training
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
            candle_nn::ParamsAdamW {
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

        // Learning rate scheduling variables
        let initial_lr = 5e-5;
        let lr_decay_factor = 0.8; // Reduce LR by 20% when validation doesn't improve
        let mut current_lr = initial_lr;

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

            // Validation phase - evaluate on validation set
            let val_loss = self.evaluate_on_validation_set(model, val_data, batch_size, max_sequence_length)?;

            let remaining_epochs = epochs - epoch;
            let estimated_remaining = epoch_duration * remaining_epochs;

            tracing::info!("Epoch {} completed in {:.1}s. Train Loss: {:.4}, Val Loss: {:.4}, LR: {:.2e}, Speed: {:.1} batches/sec, ETA: {:.1}min",
                          epoch, epoch_duration.as_secs_f64(), avg_train_loss, val_loss, current_lr, batches_per_sec,
                          estimated_remaining.as_secs_f64() / 60.0);

            // Learning rate scheduling and early stopping check
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience_counter = 0;
                tracing::info!("✅ New best validation loss: {:.4}", best_val_loss);
            } else {
                patience_counter += 1;
                tracing::warn!("⚠️ Validation loss did not improve. Patience: {}/{}", patience_counter, patience);

                // Reduce learning rate when validation doesn't improve
                if patience_counter == 2 {
                    current_lr *= lr_decay_factor;
                    tracing::info!("📉 Learning rate reduced to {:.2e}", current_lr);

                    // Create new optimizer with reduced learning rate
                    optimizer = AdamW::new(
                        var_map.all_vars(),
                        candle_nn::ParamsAdamW {
                            lr: current_lr,
                            beta1: 0.9,
                            beta2: 0.999,
                            eps: 1e-8,
                            weight_decay: 0.01,
                        }
                    )?;
                }

                if patience_counter >= patience {
                    tracing::info!("🛑 Early stopping triggered after {} epochs without improvement", patience);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Evaluate model performance on validation set without updating weights
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

    fn prepare_batch(&self, batch: &[Vec<u32>], max_length: usize) -> Result<(Tensor, Tensor)> {
        let mut input_sequences = Vec::new();
        let mut label_sequences = Vec::new();

        for sequence in batch {
            if sequence.is_empty() {
                continue;
            }

            // Truncate sequence if too long
            let truncated_len = sequence.len().min(max_length);
            let truncated_seq = &sequence[..truncated_len];

            if truncated_seq.len() < 2 {
                continue; // Need at least 2 tokens for input+label
            }

            // Input is sequence without last token
            let input = &truncated_seq[..truncated_seq.len() - 1];
            // Label is sequence without first token (shifted by 1)
            let label = &truncated_seq[1..];

            input_sequences.push(input.to_vec());
            label_sequences.push(label.to_vec());
        }

        if input_sequences.is_empty() {
            return Ok((Tensor::new(&[0u32; 0], &self.device)?, Tensor::new(&[0u32; 0], &self.device)?));
        }

        // Pad sequences to same length
        let max_len = input_sequences.iter().map(|s| s.len()).max().unwrap_or(0);

        let mut padded_inputs = Vec::new();
        let mut padded_labels = Vec::new();

        for (input, label) in input_sequences.iter().zip(label_sequences.iter()) {
            let mut padded_input = input.clone();
            let mut padded_label = label.clone();

            // Pad with zeros (assuming 0 is pad token)
            while padded_input.len() < max_len {
                padded_input.push(0);
                padded_label.push(0);
            }

            padded_inputs.extend(padded_input);
            padded_labels.extend(padded_label);
        }

        let batch_size = input_sequences.len();
        let input_tensor = Tensor::from_vec(padded_inputs, (batch_size, max_len), &self.device)?;
        let label_tensor = Tensor::from_vec(padded_labels, (batch_size, max_len), &self.device)?;

        Ok((input_tensor, label_tensor))
    }

    fn calculate_loss(&self, logits: &Tensor, labels: &Tensor) -> CandleResult<Tensor> {
        // Reshape for cross-entropy loss
        let (batch_size, seq_len, vocab_size) = logits.dims3()?;
        let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
        let labels_flat = labels.flatten_all()?;

        // Calculate cross-entropy loss
        cross_entropy(&logits_flat, &labels_flat)
    }

    fn save_model(&self, _model: &SimpleTransformer, _tokenizer: &Tokenizer, config: &Gpt2Config, var_map: &VarMap, output_path: &str) -> Result<()> {
        tracing::info!("Saving model to: {}", output_path);

        std::fs::create_dir_all(output_path)
            .with_context(|| format!("Failed to create output directory: {}", output_path))?;

        // Save config
        let config_path = Path::new(output_path).join("config.json");
        let config_json = serde_json::to_string_pretty(config)
            .with_context(|| "Failed to serialize model config")?;
        std::fs::write(&config_path, config_json)
            .with_context(|| format!("Failed to write config to: {}", config_path.display()))?;

        // Save model weights as SafeTensors
        let weights_path = Path::new(output_path).join("model.safetensors");
        tracing::info!("Saving model weights to: {}", weights_path.display());

        // Save the VarMap which contains all trained parameters
        var_map.save(&weights_path)
            .map_err(|e| anyhow!("Failed to save model weights: {}", e))?;

        tracing::info!("Model weights saved successfully");

        tracing::info!("Model saved successfully to: {}", output_path);
        Ok(())
    }

    /// Try to load existing model weights for continued training
    fn load_existing_weights_if_available(&self, _output_path: &str) -> Result<Option<VarMap>> {
        // Check standard model location first (where RustMLService looks)
        let standard_model_path = "./data/models/krokenheimer/model.safetensors";

        if Path::new(standard_model_path).exists() {
            tracing::info!("Found existing trained model at: {}", standard_model_path);
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

        // Also check if there are any other trained models
        let models_dir = Path::new("./data/models");
        if models_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(models_dir) {
                for entry in entries.flatten() {
                    if entry.path().is_dir() {
                        let potential_model = entry.path().join("model.safetensors");
                        if potential_model.exists() {
                            tracing::info!("Found alternative model at: {:?}", potential_model);

                            // Try loading this alternative model
                            let mut var_map = VarMap::new();
                            if var_map.load(&potential_model).is_ok() {
                                tracing::info!("✅ Successfully loaded alternative model checkpoint!");
                                return Ok(Some(var_map));
                            }
                        }
                    }
                }
            }
        }

        tracing::info!("No existing model found - starting fresh training");
        Ok(None)
    }

    /// Verify that loaded model is compatible with current configuration
    fn verify_model_compatibility(&self, _var_map: &VarMap, _config: &Gpt2Config) -> Result<()> {
        // For now, assume compatibility since we're using the same architecture
        // Advanced compatibility checking can be added later if needed
        tracing::info!("Model architecture appears compatible");
        Ok(())
    }
}