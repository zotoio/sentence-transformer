#!/usr/bin/env python3
"""
Fine-tune the all-MiniLM-L6-v2 sentence transformer model on markdown documents.

This script demonstrates unsupervised training using TSDAE (Transformer-based 
Sequential Denoising Auto-Encoder) which is effective for domain adaptation
without labeled data.
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import Dataset
import re
from tqdm import tqdm

from download_dataset import load_local_markdown_files


def clean_markdown(text: str) -> str:
    """
    Clean markdown text by removing formatting while preserving content.
    """
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    
    # Remove images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # Convert links to just text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove headers formatting but keep text
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic markers
    text = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    
    # Remove bullet points
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    
    # Remove numbered lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text


def extract_sentences(documents: list[dict], min_length: int = 20, max_length: int = 512) -> list[str]:
    """
    Extract clean sentences from markdown documents.
    
    Args:
        documents: List of document dicts with 'text' key
        min_length: Minimum sentence length in characters
        max_length: Maximum sentence length in characters
    
    Returns:
        List of clean sentences
    """
    sentences = []
    
    for doc in tqdm(documents, desc="Extracting sentences"):
        cleaned = clean_markdown(doc["text"])
        
        # Split into paragraphs first
        paragraphs = cleaned.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            
            # Skip empty or very short paragraphs
            if len(para) < min_length:
                continue
            
            # If paragraph is short enough, use as-is
            if len(para) <= max_length:
                sentences.append(para)
            else:
                # Split long paragraphs into sentences
                # Simple sentence splitting on . ! ?
                parts = re.split(r'(?<=[.!?])\s+', para)
                
                current = ""
                for part in parts:
                    if len(current) + len(part) <= max_length:
                        current = (current + " " + part).strip()
                    else:
                        if len(current) >= min_length:
                            sentences.append(current)
                        current = part
                
                if len(current) >= min_length:
                    sentences.append(current)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)
    
    print(f"Extracted {len(unique_sentences)} unique sentences")
    return unique_sentences


def train_with_tsdae(
    sentences: list[str],
    model_name: str = "models/all-MiniLM-L6-v2",
    output_dir: str = "models/finetuned-minilm",
    epochs: int = 1,
    batch_size: int = 8,
):
    """
    Train using TSDAE (unsupervised denoising auto-encoder).
    
    This is ideal for domain adaptation when you don't have labeled data.
    Uses the DataLoader-based training API for TSDAE compatibility.
    """
    print(f"\nLoading base model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Preparing {len(sentences)} sentences for TSDAE training...")
    
    # Create training examples - TSDAE uses single sentences
    train_examples = [InputExample(texts=[s, s]) for s in sentences]
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # TSDAE loss for unsupervised training
    train_loss = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=model_name,
        tie_encoder_decoder=True
    )
    
    # Calculate warmup steps
    warmup_steps = int(len(train_dataloader) * 0.1)
    
    print(f"\nStarting TSDAE training...")
    print(f"  - {len(train_examples)} training examples")
    print(f"  - {len(train_dataloader)} batches per epoch")
    print(f"  - {epochs} epochs")
    print(f"  - {warmup_steps} warmup steps")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train using the fit method (more compatible with TSDAE)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=output_dir,
        use_amp=True,  # Mixed precision
    )
    
    print(f"\n✓ Model saved to {output_dir}")
    
    return model


def train_with_contrastive(
    sentences: list[str],
    model_name: str = "models/all-MiniLM-L6-v2",
    output_dir: str = "models/finetuned-minilm-contrastive",
    epochs: int = 1,
    batch_size: int = 64,
):
    """
    Train using Multiple Negatives Ranking Loss (contrastive learning).
    
    Creates positive pairs from adjacent sentences in the same document.
    """
    print(f"\nLoading base model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Create pairs from adjacent sentences (assuming they're related)
    print("Creating sentence pairs for contrastive training...")
    pairs = []
    for i in range(0, len(sentences) - 1, 2):
        pairs.append(InputExample(texts=[sentences[i], sentences[i + 1]]))
    
    print(f"Created {len(pairs)} training pairs")
    
    train_dataloader = DataLoader(pairs, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    warmup_steps = int(len(train_dataloader) * 0.1)
    
    print(f"\nStarting contrastive training...")
    print(f"  - {len(pairs)} training pairs")
    print(f"  - {len(train_dataloader)} batches per epoch")
    print(f"  - {epochs} epochs")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=output_dir,
        use_amp=True,
    )
    
    print(f"\n✓ Model saved to {output_dir}")
    return model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train sentence transformer on markdown")
    parser.add_argument(
        "--data-dir",
        default="data/markdown",
        help="Directory containing markdown files"
    )
    parser.add_argument(
        "--output-dir",
        default="models/finetuned-minilm",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--method",
        choices=["tsdae", "contrastive"],
        default="tsdae",
        help="Training method (tsdae for unsupervised, contrastive for pair-based)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8 for TSDAE)"
    )
    parser.add_argument(
        "--model-name",
        default="models/all-MiniLM-L6-v2",
        help="Base model to fine-tune"
    )
    
    args = parser.parse_args()
    
    # Load documents
    print(f"Loading markdown files from {args.data_dir}...")
    documents = load_local_markdown_files(args.data_dir)
    
    # Extract sentences
    sentences = extract_sentences(documents)
    
    if len(sentences) < 100:
        print("Warning: Very few sentences extracted. Check your data.")
        return
    
    # Train
    if args.method == "tsdae":
        train_with_tsdae(
            sentences,
            model_name=args.model_name,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        train_with_contrastive(
            sentences,
            model_name=args.model_name,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
