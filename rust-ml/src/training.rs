use candle_core::{Device, Tensor, Result as CandleResult};
use candle_transformers::models::gpt2::{Gpt2, Config as Gpt2Config};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, loss::cross_entropy, Activation};
use tokenizers::{Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel,
                 decoders::byte_level::ByteLevel as ByteLevelDecoder, trainers::BpeTrainer,
                 processors::byte_level::ByteLevel as ByteLevelProcessor};
use anyhow::{Result, Context, anyhow};
use std::path::Path;
use serde_json;

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

        // Create model
        let config = self.create_model_config(tokenizer.get_vocab_size(false));
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &self.device);
        let model = Gpt2::new(&config, var_builder)?;

        // Training loop
        self.train_model(&model, &var_map, &tokenized_data, epochs, &config)?;

        // Save model
        self.save_model(&model, &tokenizer, &config, output_path)?;

        tracing::info!("Training completed successfully!");
        Ok(())
    }

    fn load_training_data(&self, file_path: &str) -> Result<Vec<ConversationData>> {
        tracing::info!("Loading training data from: {}", file_path);

        let content = std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read training data from {}", file_path))?;

        let mut conversations = Vec::new();
        for (line_num, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<ConversationData>(line) {
                Ok(conv_data) => conversations.push(conv_data),
                Err(e) => {
                    tracing::warn!("Skipping invalid JSON on line {}: {}", line_num + 1, e);
                    continue;
                }
            }
        }

        if conversations.is_empty() {
            return Err(anyhow!("No valid conversations found in training data"));
        }

        Ok(conversations)
    }

    fn create_tokenizer(&self, conversations: &[ConversationData], output_path: &str) -> Result<Tokenizer> {
        tracing::info!("Creating custom tokenizer...");

        // Collect all text for tokenizer training
        let mut texts = Vec::new();
        for conv in conversations {
            for message in &conv.messages {
                texts.push(message.content.clone());
            }
        }

        tracing::info!("Training tokenizer on {} messages", texts.len());

        // Create BPE tokenizer
        let mut tokenizer = Tokenizer::new(BPE::default());

        // Set pre-tokenizer
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));

        // Set decoder
        tokenizer.with_decoder(Some(ByteLevelDecoder::new(false, false, true)));

        // Create trainer
        let mut trainer = BpeTrainer::builder()
            .vocab_size(8000)
            .min_frequency(2)
            .special_tokens(vec![
                "[PAD]".to_string(),
                "[UNK]".to_string(),
                "[SEP]".to_string(),
                "<|endoftext|>".to_string(),
            ])
            .build();

        // Train tokenizer
        // Write texts to temporary files for training
        let temp_dir = std::env::temp_dir().join("tokenizer_training");
        std::fs::create_dir_all(&temp_dir)?;
        let mut temp_files = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            let temp_file = temp_dir.join(format!("text_{}.txt", i));
            std::fs::write(&temp_file, text)?;
            temp_files.push(temp_file.to_string_lossy().to_string());
        }

        tokenizer.train(&temp_files, Some(&mut trainer))
            .map_err(|e| anyhow!("Tokenizer training failed: {}", e))?;

        // Clean up temp files
        std::fs::remove_dir_all(&temp_dir).ok();

        // Set post-processor
        tokenizer.with_post_processor(Some(ByteLevelProcessor::new(false, false, true)));

        // Save tokenizer
        let tokenizer_path = Path::new(output_path).join("tokenizer.json");
        std::fs::create_dir_all(output_path)
            .with_context(|| format!("Failed to create output directory: {}", output_path))?;

        tokenizer.save(&tokenizer_path, false)
            .map_err(|e| anyhow!("Failed to save tokenizer: {}", e))?;

        tracing::info!("Tokenizer saved to: {}", tokenizer_path.display());
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

    fn train_model(
        &self,
        model: &Gpt2,
        optimizer: &mut AdamW,
        tokenized_data: &[Vec<u32>],
        epochs: u32,
        config: &Gpt2Config,
    ) -> Result<()> {
        tracing::info!("Starting model training...");

        let batch_size = 4;  // Small batch size for CPU training
        let max_sequence_length = 512;  // Limit sequence length for memory

        for epoch in 1..=epochs {
            tracing::info!("Epoch {}/{}", epoch, epochs);
            let mut total_loss = 0.0;
            let mut batch_count = 0;

            // Process data in batches
            for batch_start in (0..tokenized_data.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(tokenized_data.len());
                let batch = &tokenized_data[batch_start..batch_end];

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
                optimizer.zero_grad()?;
                loss.backward()?;
                optimizer.step()?;

                if batch_count % 10 == 0 {
                    tracing::info!("Batch {}: loss = {:.4}", batch_count, loss.to_scalar::<f32>()?);
                }
            }

            let avg_loss = total_loss / batch_count as f32;
            tracing::info!("Epoch {} completed. Average loss: {:.4}", epoch, avg_loss);
        }

        Ok(())
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

    fn save_model(&self, model: &Gpt2, tokenizer: &Tokenizer, config: &Gpt2Config, output_path: &str) -> Result<()> {
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
        model.save_safetensors(&weights_path)
            .map_err(|e| anyhow!("Failed to save model weights: {}", e))?;

        tracing::info!("Model saved successfully to: {}", output_path);
        Ok(())
    }
}