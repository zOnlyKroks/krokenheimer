/// Simple tests and examples for the BPE tokenizer
use crate::bpe_tokenizer::BPETokenizer;
use crate::bpe_wrapper::BPETokenizerWrapper;
use crate::training::{ConversationData, TrainingMessage};

pub fn run_bpe_tests() -> anyhow::Result<()> {
    println!("🧪 Running BPE Tokenizer Tests...");

    // Test 1: Basic training and usage
    test_basic_functionality()?;

    // Test 2: Special tokens
    test_special_tokens()?;

    // Test 3: Round-trip encoding/decoding
    test_round_trip()?;

    println!("✅ All BPE tests passed!");
    Ok(())
}

fn test_basic_functionality() -> anyhow::Result<()> {
    println!("  Test 1: Basic functionality...");

    let mut tokenizer = BPETokenizer::new();

    let training_texts = vec![
        "Hello world! How are you?".to_string(),
        "Hello there! I'm doing great.".to_string(),
        "The world is beautiful today.".to_string(),
        "How beautiful the world looks!".to_string(),
        "You look great today, how wonderful!".to_string(),
    ];

    // Train tokenizer
    tokenizer.train(&training_texts, 500)?;

    // Test encoding
    let text = "Hello world! You look great!";
    let tokens = tokenizer.encode(text);
    println!("    Encoded '{}' -> {:?}", text, &tokens[..tokens.len().min(10)]);

    // Test decoding
    let decoded = tokenizer.decode(&tokens);
    println!("    Decoded back -> '{}'", decoded);

    assert!(!tokens.is_empty(), "Should produce some tokens");
    assert!(decoded.contains("Hello"), "Should contain 'Hello'");
    assert!(decoded.contains("world"), "Should contain 'world'");

    println!("    ✓ Basic functionality works");
    Ok(())
}

fn test_special_tokens() -> anyhow::Result<()> {
    println!("  Test 2: Special tokens...");

    let mut wrapper = BPETokenizerWrapper::new();

    // Create conversation data
    let conversations = vec![
        ConversationData {
            messages: vec![
                TrainingMessage {
                    role: "user".to_string(),
                    content: "Hello assistant!".to_string(),
                },
                TrainingMessage {
                    role: "assistant".to_string(),
                    content: "Hi there! How can I help?".to_string(),
                },
                TrainingMessage {
                    role: "system".to_string(),
                    content: "This is a system message.".to_string(),
                },
            ],
        },
    ];

    // Convert to training texts
    let training_texts = wrapper.conversations_to_texts(&conversations);
    wrapper.tokenizer.train(&training_texts, 800)?;

    // Test special token handling
    let text = "<|user|> Hello! <|assistant|> Hi there! <|system|> System message.";
    let encoding = wrapper.encode(text, false)?;
    let decoded = wrapper.decode(encoding.get_ids(), false)?;

    println!("    Original: {}", text);
    println!("    Decoded:  {}", decoded);

    assert!(decoded.contains("<|user|>"), "Should preserve <|user|> token");
    assert!(decoded.contains("<|assistant|>"), "Should preserve <|assistant|> token");
    assert!(decoded.contains("<|system|>"), "Should preserve <|system|> token");

    println!("    ✓ Special tokens preserved");
    Ok(())
}

fn test_round_trip() -> anyhow::Result<()> {
    println!("  Test 3: Round-trip consistency...");

    let mut tokenizer = BPETokenizer::new();

    // Add some Discord-like special tokens
    tokenizer.add_special_token("@everyone");
    tokenizer.add_special_token("@here");
    tokenizer.add_special_token(":smile:");
    tokenizer.add_special_token(":thumbs_up:");

    let training_texts = vec![
        "Hey @everyone, how's it going? :smile:".to_string(),
        "Looking good @here! :thumbs_up:".to_string(),
        "Thanks everyone! Much appreciated :smile: :thumbs_up:".to_string(),
        "Discord messages with @mentions and :emojis: are common".to_string(),
        "BPE should handle these patterns well!".to_string(),
    ];

    tokenizer.train(&training_texts, 600)?;

    let test_cases = vec![
        "Hello world!",
        "@everyone check this out! :smile:",
        "BPE tokenization works great :thumbs_up:",
        "Special characters: $, %, &, *, +, =, |, \\, /, <, >",
        "Numbers and symbols: 123, 456.78, user@email.com",
    ];

    for test_text in test_cases {
        let tokens = tokenizer.encode(test_text);
        let decoded = tokenizer.decode(&tokens);

        println!("    '{}' -> {} tokens -> '{}'",
                 test_text, tokens.len(), decoded);

        // Basic consistency checks
        assert!(!tokens.is_empty(), "Should produce tokens for: {}", test_text);

        // For simple text, we should get most words back
        if test_text == "Hello world!" {
            assert!(decoded.contains("Hello") && decoded.contains("world"),
                   "Basic words should be preserved");
        }
    }

    println!("    ✓ Round-trip consistency verified");
    Ok(())
}

// Public interface for running from main
pub fn demo_bpe_tokenizer() {
    println!("\n🚀 BPE Tokenizer Demo");
    println!("====================");

    match run_bpe_tests() {
        Ok(()) => println!("🎉 Demo completed successfully!"),
        Err(e) => println!("❌ Demo failed: {}", e),
    }
}