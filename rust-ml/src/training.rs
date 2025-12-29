use candle_core::{Device, Tensor, Result as CandleResult, DType};
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

        // Enable CPU optimizations
        #[cfg(feature = "cpu")]
        {
            std::env::set_var("ACCELERATE_USE_ACCELERATE", "1");
            std::env::set_var("ACCELERATE_DISABLE_AVX2", "0");
        }

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
        let validation_split = 0.1; // Reduced from 0.2 to use more training data
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
        self.train_model_optimized(&model, &var_map, train_data, val_data, epochs, &config, is_continuing)?;

        // Save model
        self.save_model(&model, &tokenizer, &config, &var_map, output_path)?;

        tracing::info!("Training completed successfully!");
        Ok(())
    }

    fn create_or_load_tokenizer(&self, conversations: &[ConversationData], output_path: &str) -> Result<Tokenizer> {
        let tokenizer_path = Path::new(output_path).join("tokenizer.json");

        // Try to load existing tokenizer first
        if tokenizer_path.exists() {
            tracing::info!("Loading existing tokenizer from: {}", tokenizer_path.display());
            return Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow!("Failed to load tokenizer: {}", e));
        }

        tracing::info!("Creating new character-level tokenizer...");

        // Create vocabulary from data (optimized collection)
        let mut vocab = AHashMap::new();

        // Add special tokens
        vocab.insert("[PAD]".to_string(), 0u32);
        vocab.insert("[UNK]".to_string(), 1u32);
        vocab.insert("<|endoftext|>".to_string(), 2u32);

        // Collect characters more efficiently
        let mut char_set = std::collections::HashSet::new();

        // Pre-allocate with expected size
        char_set.reserve(500);

        // Add common ASCII characters first
        for c in ' '..='~' {
            char_set.insert(c);
        }

        // Add characters from conversations
        for conv in conversations {
            for msg in &conv.messages {
                let formatted = format!("{}: {}", msg.role, msg.content);
                for c in formatted.chars() {
                    char_set.insert(c);
                }
            }
        }

        // Convert to sorted vector
        let mut sorted_chars: Vec<char> = char_set.into_iter().collect();
        sorted_chars.sort();

        // Add to vocabulary
        let mut id = 3u32;
        for c in sorted_chars {
            vocab.insert(c.to_string(), id);
            id += 1;
        }

        tracing::info!("Created vocabulary with {} tokens", vocab.len());

        // Create tokenizer
        let mut tokenizer = Tokenizer::new(
            tokenizers::models::bpe::BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .unk_token("[UNK]".to_string())
                .build()
                .map_err(|e| anyhow!("Failed to build BPE tokenizer: {}", e))?
        );

        // Use byte-level pre-tokenizer
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::byte_level::ByteLevel::default()));

        // Save tokenizer
        std::fs::create_dir_all(output_path)
            .with_context(|| format!("Failed to create output directory: {}", output_path))?;

        tokenizer.save(&tokenizer_path, false)
            .map_err(|e| anyhow!("Failed to save tokenizer: {}", e))?;

        tracing::info!("✅ Tokenizer saved to: {}", tokenizer_path.display());

        Ok(tokenizer)
    }

    fn tokenize_conversations_batch(&self, conversations: &[ConversationData], tokenizer: &Tokenizer) -> Result<Vec<Vec<u32>>> {
        tracing::info!("Tokenizing conversations in batch...");

        let mut tokenized_data = Vec::with_capacity(conversations.len());
        let mut total_tokens = 0;

        // Process conversations in parallel chunks
        let chunk_size = 100; // Process 100 conversations at a time

        for chunk in conversations.chunks(chunk_size) {
            for conv in chunk {
                let mut text = String::new();
                for message in &conv.messages {
                    text.push_str(&format!("{}: {}\n", message.role, message.content));
                }
                text.push_str("<|endoftext|>");

                if let Ok(encoding) = tokenizer.encode(text, false) {
                    let token_ids = encoding.get_ids().to_vec();
                    if !token_ids.is_empty() && token_ids.len() >= 10 { // Skip very short sequences
                        total_tokens += token_ids.len();
                        tokenized_data.push(token_ids);
                    }
                }
            }
        }

        // Sort by length for more efficient batching (optional)
        tokenized_data.sort_by_key(|seq| seq.len());

        tracing::info!("Tokenized {} conversations, total tokens: {}", tokenized_data.len(), total_tokens);

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
        // SMALLER MODEL FOR FASTER TRAINING
        Gpt2Config {
            vocab_size,
            n_positions: 512,    // Reduced from 2048 - shorter sequences are much faster
            n_embd: 256,         // Reduced from 768 - smaller embeddings = less computation
            n_layer: 6,          // Reduced from 12 - fewer layers
            n_head: 8,           // Reduced from 12 - fewer attention heads
            n_inner: Some(512),  // Reduced from 3072 - much smaller feedforward
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
    ) -> Result<()> {
        tracing::info!("Starting optimized model training...");

        // OPTIMIZED HYPERPARAMETERS
        let batch_size = 32;  // Increased batch size (but smaller model allows this)
        let max_sequence_length = config.n_positions.min(256); // Use shorter sequences

        // Calculate total batches
        let total_batches = (train_data.len() + batch_size - 1) / batch_size;

        tracing::info!("Optimized training config: batch_size={}, seq_len={}, total_batches={}",
                      batch_size, max_sequence_length, total_batches);

        // Create optimizer with warmup
        let mut optimizer = AdamW::new(
            var_map.all_vars(),
            ParamsAdamW {
                lr: if is_continuing { 1e-5 } else { 3e-4 }, // Higher LR for fresh training
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            }
        )?;

        // Early stopping
        let mut best_val_loss = f32::INFINITY;
        let mut patience_counter = 0;
        let patience = 2; // Reduced patience

        for epoch in 1..=epochs {
            tracing::info!("Epoch {}/{} - Processing {} batches", epoch, epochs, total_batches);

            let epoch_start = std::time::Instant::now();
            let mut total_loss = 0.0;
            let mut processed_batches = 0;

            // Shuffle training data each epoch
            let mut train_indices: Vec<usize> = (0..train_data.len()).collect();
            fastrand::shuffle(&mut train_indices);

            // Training loop with progress tracking
            for batch_idx in 0..total_batches {
                let batch_start = batch_idx * batch_size;
                let batch_end = (batch_start + batch_size).min(train_indices.len());

                // Get batch indices
                let batch_indices = &train_indices[batch_start..batch_end];

                // Prepare batch more efficiently
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

                // Log progress every 5% of batches
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
            if val_loss < best_val_loss - 0.001 { // Require meaningful improvement
                best_val_loss = val_loss;
                patience_counter = 0;
                tracing::info!("✅ New best validation loss: {:.4}", best_val_loss);

                // Save checkpoint on improvement
                let checkpoint_path = format!("{}/checkpoint_epoch_{}.safetensors", "models", epoch);
                if let Err(e) = var_map.save(&checkpoint_path) {
                    tracing::warn!("Failed to save checkpoint: {}", e);
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

        // Use smaller batch for validation if needed
        let val_batch_size = batch_size.min(8);

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

    fn save_model(&self, _model: &SimpleTransformer, tokenizer: &Tokenizer, config: &Gpt2Config, var_map: &VarMap, output_path: &str) -> Result<()> {
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

        // Save tokenizer
        let tokenizer_path = Path::new(output_path).join("tokenizer.json");
        tokenizer.save(&tokenizer_path, false)
            .map_err(|e| anyhow!("Failed to save tokenizer: {}", e))?;

        tracing::info!("✅ Model saved to: {}", output_path);
        Ok(())
    }
}