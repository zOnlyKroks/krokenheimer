#!/usr/bin/env python3
"""
Generate script for Krokenheimer Discord bot
Uses the trained-from-scratch GPT-2 model for text generation
"""
import argparse
import sys
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_model_and_tokenizer(model_path):
    """Load the trained model and tokenizer"""
    try:
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)

        # Load model
        model = GPT2LMHeadModel.from_pretrained(model_path)

        # Set to evaluation mode
        model.eval()

        # Move to CPU (since we're running in container)
        model = model.to('cpu')

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

def generate_text(model, tokenizer, prompt, max_length, temperature):
    """Generate text using the trained model"""
    try:
        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        # Decode the generated text
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the generated text
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        return generated

    except Exception as e:
        print(f"Error generating text: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Generate text using trained Krokenheimer model')
    parser.add_argument('model_path', help='Path to the trained model directory')
    parser.add_argument('--prompt', required=True, help='Input prompt for text generation')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling')

    args = parser.parse_args()

    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Check if required model files exist
    required_files = ['config.json', 'pytorch_model.bin']
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(args.model_path, file)):
            missing_files.append(file)

    if missing_files:
        print(f"Error: Missing model files: {', '.join(missing_files)}", file=sys.stderr)
        sys.exit(1)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Generate text
    generated_text = generate_text(
        model,
        tokenizer,
        args.prompt,
        args.max_length,
        args.temperature
    )

    # Output the generated text
    print(generated_text)

if __name__ == '__main__':
    main()