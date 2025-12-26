#!/usr/bin/env python3
"""
Generate text using YOUR trained-from-scratch model
Uses custom tokenizer and model weights learned from Discord
"""

import torch
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Generate text from trained model')
    parser.add_argument('model_path', type=str, help='Path to trained model directory')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.9, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')

    args = parser.parse_args()

    # Import transformers
    try:
        from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}", file=sys.stderr)
        print("Please train the model first with: !llmtrain now", file=sys.stderr)
        sys.exit(1)

    try:
        # Load custom tokenizer
        tokenizer_path = f"{args.model_path}/tokenizer.json"
        if not os.path.exists(tokenizer_path):
            print(f"Error: Tokenizer not found at {tokenizer_path}", file=sys.stderr)
            sys.exit(1)

        tokenizer_obj = Tokenizer.from_file(tokenizer_path)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            bos_token='<|endoftext|>',
            eos_token='<|endoftext|>',
            unk_token='<|endoftext|>',
            pad_token='<|pad|>'
        )

        # Load model
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
        model.eval()

        # Get model's max position embeddings
        max_position_embeddings = model.config.n_positions  # Usually 512

        # Encode prompt
        input_ids = tokenizer.encode(args.prompt, return_tensors='pt')

        # Check if prompt is too long - truncate if needed
        if input_ids.shape[1] > max_position_embeddings - args.max_length:
            # Leave room for generation
            max_prompt_length = max_position_embeddings - args.max_length
            print(f"Warning: Truncating prompt from {input_ids.shape[1]} to {max_prompt_length} tokens", file=sys.stderr)
            input_ids = input_ids[:, -max_prompt_length:]

        # Calculate safe max_length
        safe_max_length = min(
            input_ids.shape[1] + args.max_length,
            max_position_embeddings
        )

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=safe_max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Prevent repetition
            )

        # Decode output
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

        # Find where assistant's response starts (after the <|assistant|> token)
        assistant_marker = '<|assistant|>'
        if assistant_marker in generated_text:
            # Get text after the last <|assistant|> marker
            parts = generated_text.split(assistant_marker)
            generated_part = parts[-1].strip()
        else:
            # Fallback: extract after prompt
            prompt_length = len(args.prompt)
            generated_part = generated_text[prompt_length:].strip()

        # Import regex for cleaning
        import re

        # Remove all special tokens (complete tokens)
        special_tokens = ['<|endoftext|>', '<|assistant|>', '<|system|>', '<|user|>', '<|pad|>']
        for token in special_tokens:
            generated_part = generated_part.replace(token, '')

        # Remove partial tokens (missing opening <)
        partial_tokens = ['|endoftext|>', '|assistant|>', '|system|>', '|user|>', '|pad|>']
        for token in partial_tokens:
            generated_part = generated_part.replace(token, '')

        # Remove tokens missing closing >
        partial_tokens_2 = ['<|endoftext|', '<|assistant|', '<|system|', '<|user|', '<|pad|']
        for token in partial_tokens_2:
            generated_part = generated_part.replace(token, '')

        # Remove any remaining angle bracket token patterns (case insensitive)
        generated_part = re.sub(r'<\|[^>|]+\|>', '', generated_part, flags=re.IGNORECASE)
        generated_part = re.sub(r'\|[^>|]+\|>', '', generated_part, flags=re.IGNORECASE)

        # Final cleanup: remove any fragments
        generated_part = re.sub(r'[<|]+(assistant|system|user|pad|endoftext)[|>]+', '', generated_part, flags=re.IGNORECASE)

        # Clean up whitespace
        generated_part = generated_part.strip()

        # Split by newlines and take first non-empty line
        lines = [line.strip() for line in generated_part.split('\n') if line.strip()]
        if lines:
            generated_part = lines[0]
        else:
            generated_part = ''

        # Only print if we have actual content
        if generated_part:
            print(generated_part)
        else:
            print("Sorry, I'm not sure what to say.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Generation error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
