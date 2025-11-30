# Sentence Transformer - Markdown Training

Fine-tune the `all-MiniLM-L6-v2` sentence transformer model on a dataset of markdown files.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Download Dataset

Download ~20,000 markdown files from the GitHub code dataset:

```bash
python download_dataset.py --target-files 20000
```

Options:
- `--output-dir`: Output directory (default: `data/markdown`)
- `--target-files`: Number of files to download (default: 20000)
- `--no-streaming`: Disable streaming mode (not recommended)

### Train the Model

Train using TSDAE (unsupervised):

```bash
python train.py --method tsdae --epochs 1
```

Train using contrastive learning:

```bash
python train.py --method contrastive --epochs 1
```

Options:
- `--data-dir`: Directory containing markdown files
- `--output-dir`: Output directory for trained model
- `--method`: Training method (`tsdae` or `contrastive`)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--model-name`: Base model to fine-tune

### Run Inference

```bash
python inference.py
```

### Launch Web UI

```bash
python web_ui.py
```

Then open http://localhost:7860 in your browser.

**Features:**
- Compute similarity between texts
- Semantic search over a corpus
- Batch similarity matrix
- Embedding inspection
- Compare different models (base vs fine-tuned)

## Model Information

### Base Model: all-MiniLM-L6-v2

- **Dimensions**: 384
- **Max Sequence Length**: 256 tokens
- **Performance**: Excellent balance of speed and quality
- **Use Cases**: Semantic search, clustering, similarity comparison

### Training Methods

#### TSDAE (Transformer-based Sequential Denoising Auto-Encoder)

Best for unsupervised domain adaptation. The model learns to reconstruct sentences from noisy inputs, improving its understanding of your domain's vocabulary and structure.

#### Contrastive Learning (Multiple Negatives Ranking Loss)

Creates positive pairs from related sentences and trains the model to distinguish them from negative examples. Good when you have naturally paired data.

## Project Structure

```
sentence-transformer/
├── requirements.txt      # Python dependencies
├── download_dataset.py   # Download markdown files
├── train.py             # Training script
├── inference.py         # Inference examples
├── models/
│   └── all-MiniLM-L6-v2/ # Base model (committed)
├── data/                # Downloaded data (gitignored)
│   └── markdown/        # Markdown files
└── models/finetuned-*/  # Trained models
```

## Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended for training)
- ~10GB disk space for dataset

## License

MIT

