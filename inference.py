#!/usr/bin/env python3
"""
Inference examples using the sentence transformer model.

Demonstrates how to:
- Load the base or fine-tuned model
- Generate embeddings for text
- Compute semantic similarity
- Perform semantic search
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
from pathlib import Path


def load_model(model_path: str = None) -> SentenceTransformer:
    """
    Load the sentence transformer model.
    
    Args:
        model_path: Path to fine-tuned model, or None for base model
    
    Returns:
        Loaded SentenceTransformer model
    """
    if model_path and Path(model_path).exists():
        print(f"Loading model from {model_path}")
        return SentenceTransformer(model_path)
    else:
        # Default to local base model
        local_model = Path("models/all-MiniLM-L6-v2")
        if local_model.exists():
            print(f"Loading base model from {local_model}")
            return SentenceTransformer(str(local_model))
        else:
            print("Loading base all-MiniLM-L6-v2 model from HuggingFace")
            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def generate_embeddings(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        model: The sentence transformer model
        texts: List of text strings
    
    Returns:
        numpy array of embeddings (shape: [num_texts, 384])
    """
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def compute_similarity(model: SentenceTransformer, text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts.
    """
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()


def semantic_search(
    model: SentenceTransformer,
    query: str,
    corpus: list[str],
    top_k: int = 5
) -> list[dict]:
    """
    Perform semantic search to find most similar texts.
    
    Args:
        model: The sentence transformer model
        query: Query text
        corpus: List of texts to search
        top_k: Number of results to return
    
    Returns:
        List of dicts with 'text', 'score', and 'index' keys
    """
    # Encode query and corpus
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
    
    # Compute similarities
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    
    results = []
    for hit in hits:
        results.append({
            "text": corpus[hit["corpus_id"]],
            "score": hit["score"],
            "index": hit["corpus_id"]
        })
    
    return results


def demo():
    """
    Run a demonstration of the model capabilities.
    """
    # Try to load fine-tuned model, fall back to base
    model = load_model("models/finetuned-minilm/final")
    
    print("\n" + "=" * 60)
    print("SENTENCE TRANSFORMER DEMO")
    print("=" * 60)
    
    # Demo 1: Generate embeddings
    print("\n1. Generating embeddings...")
    sample_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "The weather today is sunny and warm.",
        "Natural language processing deals with text and speech.",
    ]
    
    embeddings = generate_embeddings(model, sample_texts)
    print(f"   Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Demo 2: Similarity comparison
    print("\n2. Computing similarities...")
    pairs = [
        ("Machine learning is powerful.", "AI and ML are transforming industries."),
        ("Machine learning is powerful.", "The cat sat on the mat."),
        ("How to install Python?", "Guide to setting up Python environment"),
    ]
    
    for text1, text2 in pairs:
        sim = compute_similarity(model, text1, text2)
        print(f"   '{text1[:40]}...' vs '{text2[:40]}...'")
        print(f"   Similarity: {sim:.4f}")
        print()
    
    # Demo 3: Semantic search
    print("\n3. Semantic search demo...")
    corpus = [
        "Installing dependencies with pip",
        "How to create a virtual environment in Python",
        "Deep learning with TensorFlow and PyTorch",
        "Building REST APIs with FastAPI",
        "Database migrations with Alembic",
        "Unit testing best practices",
        "Deploying applications to Kubernetes",
        "Git branching strategies",
    ]
    
    query = "How do I set up my Python project?"
    print(f"   Query: '{query}'")
    print("   Top results:")
    
    results = semantic_search(model, query, corpus, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['text']} (score: {result['score']:.4f})")
    
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    demo()

