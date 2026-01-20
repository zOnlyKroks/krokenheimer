use neon::prelude::*;

mod tokenizer;
mod bpe_tokenizer;
mod bpe_wrapper;
mod utils;
mod simple_burn;

// Simple training data structures for BPE wrapper
#[derive(serde::Deserialize, Clone)]
pub struct TrainingMessage {
    pub role: String,
    pub content: String,
}

#[derive(serde::Deserialize, Clone)]
pub struct ConversationData {
    pub messages: Vec<TrainingMessage>,
}

use simple_burn::{SimpleBurnService, BurnInferenceService};
use std::sync::Mutex;

// Node.js API exports
#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Inference functions
    cx.export_function("loadModel", load_model)?;
    cx.export_function("generateText", generate_text)?;
    cx.export_function("generateMentionResponse", generate_mention_response)?;

    // Training functions
    cx.export_function("trainModel", train_model)?;
    cx.export_function("checkTrainingStatus", check_training_status)?;
    cx.export_function("shouldStartTraining", should_start_training)?;

    // Utility functions
    cx.export_function("getModelInfo", get_model_info)?;
    cx.export_function("getConfig", get_config)?;

    // Note: BPE testing removed for simplified implementation

    Ok(())
}

// Global service instances using simplified Burn (CPU-only)
static TRAINING_SERVICE: Mutex<Option<SimpleBurnService>> = Mutex::new(None);
static INFERENCE_SERVICE: Mutex<Option<BurnInferenceService>> = Mutex::new(None);

// Initialize inference service (Burn-based)
fn load_model(mut cx: FunctionContext) -> JsResult<JsBoolean> {
    let model_path = cx.argument::<JsString>(0)?.value(&mut cx);

    tracing::info!("Loading Burn-based model from: {}", model_path);

    match BurnInferenceService::load_from_path(&model_path) {
        Ok(service) => {
            let mut inference_lock = INFERENCE_SERVICE.lock().unwrap();
            *inference_lock = Some(service);
            tracing::info!("Burn model loaded successfully");
            Ok(cx.boolean(true))
        }
        Err(e) => {
            tracing::error!("Failed to load Burn model: {}", e);
            Ok(cx.boolean(false))
        }
    }
}

// Generate text function (Burn-based)
fn generate_text(mut cx: FunctionContext) -> JsResult<JsString> {
    let prompt = cx.argument::<JsString>(0)?.value(&mut cx);
    let max_length = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let temperature = cx.argument::<JsNumber>(2)?.value(&mut cx) as f32;

    let inference_lock = INFERENCE_SERVICE.lock().unwrap();
    match inference_lock.as_ref() {
        Some(service) => {
            match service.generate(&prompt, max_length, temperature) {
                Ok(generated_text) => {
                    tracing::info!("Generated text successfully");
                    Ok(cx.string(generated_text))
                }
                Err(e) => {
                    tracing::error!("Text generation failed: {}", e);
                    Ok(cx.string(format!("Error: {}", e)))
                }
            }
        }
        None => {
            tracing::warn!("No model loaded for text generation");
            Ok(cx.string("Error: No model loaded. Call loadModel() first."))
        }
    }
}

// Generate mention response (Burn-based)
fn generate_mention_response(mut cx: FunctionContext) -> JsResult<JsString> {
    let _context = cx.argument::<JsArray>(0)?;
    let _message_content = cx.argument::<JsString>(1)?.value(&mut cx);
    let _author_name = cx.argument::<JsString>(2)?.value(&mut cx);

    // TODO: Implement Burn-based mention response generation
    tracing::warn!("Burn-based mention response generation not yet implemented");

    // For now, return a placeholder response
    Ok(cx.string("👍"))
}

// Training functions (Burn-based)
fn train_model(mut cx: FunctionContext) -> JsResult<JsBoolean> {
    let training_data_path = cx.argument::<JsString>(0)?.value(&mut cx);
    let output_path = cx.argument::<JsString>(1)?.value(&mut cx);
    let epochs = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;

    // Initialize training service if needed
    let mut service_lock = TRAINING_SERVICE.lock().unwrap();
    if service_lock.is_none() {
        *service_lock = Some(SimpleBurnService::new());
    }

    if let Some(ref mut service) = service_lock.as_mut() {
        match service.train(&training_data_path, &output_path, epochs) {
            Ok(()) => {
                tracing::info!("Simple Burn-based training completed successfully");
                Ok(cx.boolean(true))
            }
            Err(e) => {
                tracing::error!("Simple Burn-based training failed: {}", e);
                Ok(cx.boolean(false))
            }
        }
    } else {
        tracing::error!("Failed to initialize Simple Burn training service");
        Ok(cx.boolean(false))
    }
}

fn check_training_status(mut cx: FunctionContext) -> JsResult<JsObject> {
    let obj = cx.empty_object();
    let training_in_progress = cx.boolean(false); // TODO: Implement actual status checking
    obj.set(&mut cx, "training_in_progress", training_in_progress)?;
    Ok(obj)
}

fn should_start_training(mut cx: FunctionContext) -> JsResult<JsObject> {
    let message_count = cx.argument::<JsNumber>(0)?.value(&mut cx) as u32;
    let last_train_count = cx.argument::<JsNumber>(1)?.value(&mut cx) as u32;
    let threshold = cx.argument::<JsNumber>(2)?.value(&mut cx) as u32;

    let should_train = (message_count - last_train_count) >= threshold;

    let obj = cx.empty_object();
    let should_train_js = cx.boolean(should_train);
    let reason = if should_train {
        cx.string(format!("Ready to train with {} new messages", message_count - last_train_count))
    } else {
        cx.string(format!("Only {} new messages (need {})", message_count - last_train_count, threshold))
    };

    obj.set(&mut cx, "shouldTrain", should_train_js)?;
    obj.set(&mut cx, "reason", reason)?;
    Ok(obj)
}

fn get_model_info(mut cx: FunctionContext) -> JsResult<JsObject> {
    let obj = cx.empty_object();

    let inference_lock = INFERENCE_SERVICE.lock().unwrap();
    match inference_lock.as_ref() {
        Some(service) => {
            let (model_name, version, vocab_size) = service.get_info();
            let model_type = cx.string(model_name);
            let version_js = cx.string(version);
            let vocab_size_js = cx.number(vocab_size as f64);

            obj.set(&mut cx, "model", model_type)?;
            obj.set(&mut cx, "version", version_js)?;
            obj.set(&mut cx, "vocabSize", vocab_size_js)?;
            let loaded_js = cx.boolean(true);
            obj.set(&mut cx, "loaded", loaded_js)?;
        }
        None => {
            let model_type = cx.string("krokenheimer-rust (burn)");
            let version = cx.string("0.2.0");

            obj.set(&mut cx, "model", model_type)?;
            obj.set(&mut cx, "version", version)?;
            let loaded_js = cx.boolean(false);
            obj.set(&mut cx, "loaded", loaded_js)?;
        }
    }

    Ok(obj)
}

fn get_config(mut cx: FunctionContext) -> JsResult<JsObject> {
    let obj = cx.empty_object();
    let model = cx.string("krokenheimer-rust (burn)");
    let temperature = cx.number(0.3);
    let max_tokens = cx.number(100);
    let context_window = cx.number(512);  // Updated to match our Burn config

    obj.set(&mut cx, "model", model)?;
    obj.set(&mut cx, "temperature", temperature)?;
    obj.set(&mut cx, "maxTokens", max_tokens)?;
    obj.set(&mut cx, "contextWindow", context_window)?;
    Ok(obj)
}

