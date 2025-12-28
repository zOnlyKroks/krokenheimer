use neon::prelude::*;

mod inference;
mod training;
mod tokenizer;
mod model;
mod utils;

use inference::InferenceService;
use training::TrainingService;

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

    Ok(())
}

// Global inference service instance
static mut INFERENCE_SERVICE: Option<InferenceService> = None;
static mut TRAINING_SERVICE: Option<TrainingService> = None;

// Initialize inference service
fn load_model(mut cx: FunctionContext) -> JsResult<JsBoolean> {
    let model_path = cx.argument::<JsString>(0)?.value(&mut cx);

    match InferenceService::new(&model_path) {
        Ok(service) => {
            unsafe {
                INFERENCE_SERVICE = Some(service);
            }
            Ok(cx.boolean(true))
        }
        Err(e) => {
            tracing::error!("Failed to load model: {}", e);
            Ok(cx.boolean(false))
        }
    }
}

// Generate text function
fn generate_text(mut cx: FunctionContext) -> JsResult<JsString> {
    let prompt = cx.argument::<JsString>(0)?.value(&mut cx);
    let max_length = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let temperature = cx.argument::<JsNumber>(2)?.value(&mut cx) as f32;

    unsafe {
        if let Some(ref service) = INFERENCE_SERVICE {
            match service.generate(&prompt, max_length, temperature) {
                Ok(result) => Ok(cx.string(result)),
                Err(e) => {
                    tracing::error!("Generation failed: {}", e);
                    Ok(cx.string("👍")) // Fallback response
                }
            }
        } else {
            tracing::error!("Inference service not initialized");
            Ok(cx.string("👍"))
        }
    }
}

// Generate mention response
fn generate_mention_response(mut cx: FunctionContext) -> JsResult<JsString> {
    let _context = cx.argument::<JsArray>(0)?;
    let message_content = cx.argument::<JsString>(1)?.value(&mut cx);
    let author_name = cx.argument::<JsString>(2)?.value(&mut cx);

    // Parse context array (simplified for now)
    let prompt = format!("{}: {}\nKrokenheimer: ", author_name, message_content);

    unsafe {
        if let Some(ref service) = INFERENCE_SERVICE {
            match service.generate(&prompt, 100, 0.9) {
                Ok(result) => Ok(cx.string(result)),
                Err(e) => {
                    tracing::error!("Mention response failed: {}", e);
                    Ok(cx.string("👍"))
                }
            }
        } else {
            Ok(cx.string("👍"))
        }
    }
}

// Training functions
fn train_model(mut cx: FunctionContext) -> JsResult<JsBoolean> {
    let training_data_path = cx.argument::<JsString>(0)?.value(&mut cx);
    let output_path = cx.argument::<JsString>(1)?.value(&mut cx);
    let epochs = cx.argument::<JsNumber>(2)?.value(&mut cx) as u32;

    // Initialize training service if needed
    unsafe {
        if TRAINING_SERVICE.is_none() {
            TRAINING_SERVICE = Some(TrainingService::new());
        }

        if let Some(ref mut service) = TRAINING_SERVICE {
            match service.train(&training_data_path, &output_path, epochs) {
                Ok(()) => Ok(cx.boolean(true)),
                Err(e) => {
                    tracing::error!("Training failed: {}", e);
                    Ok(cx.boolean(false))
                }
            }
        } else {
            Ok(cx.boolean(false))
        }
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
    let model_type = cx.string("krokenheimer-rust (candle)");
    let version = cx.string("0.1.0");

    obj.set(&mut cx, "model", model_type)?;
    obj.set(&mut cx, "version", version)?;
    Ok(obj)
}

fn get_config(mut cx: FunctionContext) -> JsResult<JsObject> {
    let obj = cx.empty_object();
    let model = cx.string("krokenheimer-rust (candle)");
    let temperature = cx.number(0.9);
    let max_tokens = cx.number(100);
    let context_window = cx.number(1024);

    obj.set(&mut cx, "model", model)?;
    obj.set(&mut cx, "temperature", temperature)?;
    obj.set(&mut cx, "maxTokens", max_tokens)?;
    obj.set(&mut cx, "contextWindow", context_window)?;
    Ok(obj)
}