#!/usr/bin/env python3
"""
Web UI for testing sentence transformer models.

Provides an interactive interface to:
- Compare different models (base vs fine-tuned)
- Compute text similarity
- Perform semantic search
- Visualize embeddings
"""

import gradio as gr
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import numpy as np
import json


def get_available_models() -> list[str]:
    """Get list of available models in the models directory."""
    models_dir = Path("models")
    models = []
    
    if models_dir.exists():
        for model_path in models_dir.iterdir():
            if model_path.is_dir():
                # Check if it's a valid model directory
                if (model_path / "config.json").exists() or (model_path / "modules.json").exists():
                    models.append(str(model_path))
    
    # Add HuggingFace model as fallback
    if not models:
        models.append("sentence-transformers/all-MiniLM-L6-v2")
    
    return models


# Global model cache
_model_cache = {}


def load_model(model_path: str) -> SentenceTransformer:
    """Load model with caching."""
    if model_path not in _model_cache:
        _model_cache[model_path] = SentenceTransformer(model_path)
    return _model_cache[model_path]


def compute_similarity(text1: str, text2: str, model_path: str) -> tuple[float, str]:
    """Compute cosine similarity between two texts."""
    if not text1.strip() or not text2.strip():
        return 0.0, "Please enter both texts."
    
    model = load_model(model_path)
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    # Interpretation
    if similarity > 0.8:
        interpretation = "🟢 Very similar / Same meaning"
    elif similarity > 0.5:
        interpretation = "🟡 Related / Similar topic"
    elif similarity > 0.2:
        interpretation = "🟠 Somewhat related"
    else:
        interpretation = "🔴 Unrelated / Different topics"
    
    return round(similarity, 4), interpretation


def semantic_search(query: str, corpus_text: str, model_path: str, top_k: int = 5) -> str:
    """Perform semantic search on a corpus."""
    if not query.strip() or not corpus_text.strip():
        return "Please enter a query and corpus."
    
    # Split corpus into sentences/paragraphs
    corpus = [line.strip() for line in corpus_text.split('\n') if line.strip()]
    
    if not corpus:
        return "Corpus is empty."
    
    model = load_model(model_path)
    
    # Encode
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    
    # Search
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=min(top_k, len(corpus)))[0]
    
    # Format results
    results = []
    for i, hit in enumerate(hits, 1):
        score = hit['score']
        text = corpus[hit['corpus_id']]
        
        # Score indicator
        if score > 0.7:
            indicator = "🟢"
        elif score > 0.4:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        results.append(f"{i}. {indicator} **Score: {score:.4f}**\n   {text}")
    
    return "\n\n".join(results)


def batch_similarity(texts_input: str, model_path: str) -> str:
    """Compute pairwise similarity matrix for multiple texts."""
    if not texts_input.strip():
        return "Please enter texts (one per line)."
    
    texts = [line.strip() for line in texts_input.split('\n') if line.strip()]
    
    if len(texts) < 2:
        return "Please enter at least 2 texts."
    
    model = load_model(model_path)
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    # Compute similarity matrix
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    
    # Format as table
    result = "### Similarity Matrix\n\n"
    
    # Header
    result += "| | " + " | ".join([f"**{i+1}**" for i in range(len(texts))]) + " |\n"
    result += "|---" + "|---" * len(texts) + "|\n"
    
    # Rows
    for i, text in enumerate(texts):
        row_values = [f"{sim_matrix[i][j]:.2f}" for j in range(len(texts))]
        result += f"| **{i+1}** | " + " | ".join(row_values) + " |\n"
    
    result += "\n### Texts\n"
    for i, text in enumerate(texts):
        result += f"{i+1}. {text[:100]}{'...' if len(text) > 100 else ''}\n"
    
    return result


def get_embedding_info(text: str, model_path: str) -> str:
    """Get embedding information for a text."""
    if not text.strip():
        return "Please enter text."
    
    model = load_model(model_path)
    embedding = model.encode(text, convert_to_numpy=True)
    
    info = f"""### Embedding Information

**Model**: {model_path}
**Dimension**: {len(embedding)}
**Text length**: {len(text)} characters

### Embedding Statistics
- **Min**: {embedding.min():.4f}
- **Max**: {embedding.max():.4f}
- **Mean**: {embedding.mean():.4f}
- **Std**: {embedding.std():.4f}
- **L2 Norm**: {np.linalg.norm(embedding):.4f}

### First 10 dimensions
```
{embedding[:10].round(4).tolist()}
```
"""
    return info


def compare_models(text1: str, text2: str, model1_path: str, model2_path: str) -> str:
    """Compare similarity scores between two models."""
    if not text1.strip() or not text2.strip():
        return "Please enter both texts."
    
    if model1_path == model2_path:
        return "Please select two different models to compare."
    
    # Model 1
    model1 = load_model(model1_path)
    emb1 = model1.encode([text1, text2], convert_to_tensor=True)
    sim1 = util.cos_sim(emb1[0], emb1[1]).item()
    
    # Model 2
    model2 = load_model(model2_path)
    emb2 = model2.encode([text1, text2], convert_to_tensor=True)
    sim2 = util.cos_sim(emb2[0], emb2[1]).item()
    
    diff = sim2 - sim1
    
    result = f"""### Model Comparison

| Model | Similarity Score |
|-------|-----------------|
| {Path(model1_path).name} | **{sim1:.4f}** |
| {Path(model2_path).name} | **{sim2:.4f}** |

**Difference**: {diff:+.4f} {'(Model 2 higher)' if diff > 0 else '(Model 1 higher)' if diff < 0 else '(Same)'}

### Texts
- **Text 1**: {text1[:200]}{'...' if len(text1) > 200 else ''}
- **Text 2**: {text2[:200]}{'...' if len(text2) > 200 else ''}
"""
    return result


# Sample corpus for demo
SAMPLE_CORPUS = """How to install Python on Windows
Setting up a virtual environment with venv
Introduction to machine learning with scikit-learn
Deep learning with PyTorch tutorial
Building REST APIs with FastAPI
Database migrations with Alembic
Unit testing best practices in Python
Deploying applications to Kubernetes
Git branching strategies for teams
Docker container basics
Natural language processing with transformers
Computer vision with OpenCV
Web scraping with BeautifulSoup
Data visualization with matplotlib
Pandas dataframe operations"""


def create_ui():
    """Create the Gradio interface."""
    available_models = get_available_models()
    default_model = available_models[0] if available_models else ""
    
    with gr.Blocks(
        title="Sentence Transformer Test UI",
        analytics_enabled=False,
    ) as demo:
        # Custom CSS to hide Gradio branding and settings
        gr.HTML("""
        <style>
            /* Hide settings button */
            .settings { display: none !important; }
            button.settings { display: none !important; }
            
            /* Hide Gradio footer/branding */
            footer { display: none !important; }
            .gradio-container footer { display: none !important; }
            
            /* Hide "Built with Gradio" and any external links */
            a[href*="gradio.app"] { display: none !important; }
            .built-with { display: none !important; }
            
            /* Hide share button if present */
            .share-button { display: none !important; }
        </style>
        """)
        gr.Markdown("""
        # 🔤 Sentence Transformer Test UI
        
        Test and compare sentence transformer models for semantic similarity and search.
        """)
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Models", scale=0)
            models_status = gr.Markdown(f"**Models:** {', '.join([f'`{m}`' for m in available_models])}")
        
        with gr.Tab("🔍 Similarity"):
            gr.Markdown("### Compute similarity between two texts")
            with gr.Row():
                with gr.Column():
                    sim_text1 = gr.Textbox(
                        label="Text 1",
                        placeholder="Enter first text...",
                        lines=3,
                        value="Machine learning is a subset of artificial intelligence."
                    )
                    sim_text2 = gr.Textbox(
                        label="Text 2",
                        placeholder="Enter second text...",
                        lines=3,
                        value="AI and ML are transforming how we build software."
                    )
                    sim_model = gr.Dropdown(
                        choices=available_models,
                        value=default_model,
                        label="Model"
                    )
                    sim_btn = gr.Button("Compute Similarity", variant="primary")
                
                with gr.Column():
                    sim_score = gr.Number(label="Similarity Score")
                    sim_interpretation = gr.Textbox(label="Interpretation")
            
            sim_btn.click(
                compute_similarity,
                inputs=[sim_text1, sim_text2, sim_model],
                outputs=[sim_score, sim_interpretation]
            )
        
        with gr.Tab("🔎 Semantic Search"):
            gr.Markdown("### Search for similar texts in a corpus")
            with gr.Row():
                with gr.Column():
                    search_query = gr.Textbox(
                        label="Query",
                        placeholder="Enter search query...",
                        value="How do I set up my Python development environment?"
                    )
                    search_corpus = gr.Textbox(
                        label="Corpus (one text per line)",
                        placeholder="Enter texts to search...",
                        lines=10,
                        value=SAMPLE_CORPUS
                    )
                    search_model = gr.Dropdown(
                        choices=available_models,
                        value=default_model,
                        label="Model"
                    )
                    search_k = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="Top K results"
                    )
                    search_btn = gr.Button("Search", variant="primary")
                
                with gr.Column():
                    search_results = gr.Markdown(label="Results")
            
            search_btn.click(
                semantic_search,
                inputs=[search_query, search_corpus, search_model, search_k],
                outputs=[search_results]
            )
        
        with gr.Tab("📊 Batch Similarity"):
            gr.Markdown("### Compute pairwise similarity for multiple texts")
            with gr.Row():
                with gr.Column():
                    batch_texts = gr.Textbox(
                        label="Texts (one per line)",
                        placeholder="Enter texts...",
                        lines=8,
                        value="Python is a programming language\nJava is also a programming language\nThe weather is sunny today\nMachine learning uses data"
                    )
                    batch_model = gr.Dropdown(
                        choices=available_models,
                        value=default_model,
                        label="Model"
                    )
                    batch_btn = gr.Button("Compute Matrix", variant="primary")
                
                with gr.Column():
                    batch_results = gr.Markdown(label="Similarity Matrix")
            
            batch_btn.click(
                batch_similarity,
                inputs=[batch_texts, batch_model],
                outputs=[batch_results]
            )
        
        with gr.Tab("📐 Embedding Info"):
            gr.Markdown("### Inspect embedding properties")
            with gr.Row():
                with gr.Column():
                    emb_text = gr.Textbox(
                        label="Text",
                        placeholder="Enter text...",
                        lines=3,
                        value="This is a sample sentence to analyze."
                    )
                    emb_model = gr.Dropdown(
                        choices=available_models,
                        value=default_model,
                        label="Model"
                    )
                    emb_btn = gr.Button("Analyze", variant="primary")
                
                with gr.Column():
                    emb_info = gr.Markdown(label="Embedding Info")
            
            emb_btn.click(
                get_embedding_info,
                inputs=[emb_text, emb_model],
                outputs=[emb_info]
            )
        
        with gr.Tab("⚖️ Compare Models"):
            gr.Markdown("### Compare similarity scores between different models")
            with gr.Row():
                with gr.Column():
                    cmp_text1 = gr.Textbox(
                        label="Text 1",
                        placeholder="Enter first text...",
                        lines=3,
                        value="How to deploy a Python application?"
                    )
                    cmp_text2 = gr.Textbox(
                        label="Text 2",
                        placeholder="Enter second text...",
                        lines=3,
                        value="Deploying Python apps to production servers"
                    )
                    with gr.Row():
                        cmp_model1 = gr.Dropdown(
                            choices=available_models,
                            value=available_models[0] if available_models else "",
                            label="Model 1"
                        )
                        cmp_model2 = gr.Dropdown(
                            choices=available_models,
                            value=available_models[-1] if len(available_models) > 1 else (available_models[0] if available_models else ""),
                            label="Model 2"
                        )
                    cmp_btn = gr.Button("Compare", variant="primary")
                
                with gr.Column():
                    cmp_results = gr.Markdown(label="Comparison Results")
            
            cmp_btn.click(
                compare_models,
                inputs=[cmp_text1, cmp_text2, cmp_model1, cmp_model2],
                outputs=[cmp_results]
            )
        
        def refresh_models():
            """Refresh available models list."""
            models = get_available_models()
            status = f"**Models:** {', '.join([f'`{m}`' for m in models])}"
            dropdown_update = gr.update(choices=models)
            return [status] + [dropdown_update] * 6
        
        refresh_btn.click(
            refresh_models,
            inputs=[],
            outputs=[models_status, sim_model, search_model, batch_model, emb_model, cmp_model1, cmp_model2]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

