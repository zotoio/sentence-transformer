#!/usr/bin/env python3
"""
Download and prepare a text dataset for sentence transformer training.

Uses the 'HuggingFaceFW/fineweb-edu' dataset which contains high-quality
educational web content - excellent for training sentence embeddings.
"""

import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import json


def download_fineweb_edu(
    output_dir: str = "data/markdown",
    target_files: int = 20000,
):
    """
    Download educational text content from the fineweb-edu dataset.
    
    Args:
        output_dir: Directory to save text files
        target_files: Target number of files to download
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading 'HuggingFaceFW/fineweb-edu' (sample-10BT) dataset...")
    print("This contains high-quality educational web content.")
    print("Dataset is ~20GB, using streaming mode...")
    
    # Load the sample subset in streaming mode
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    
    print(f"\nDownloading up to {target_files} documents...")
    
    file_count = 0
    metadata = []
    pbar = tqdm(total=target_files, desc="Downloading")
    
    for item in dataset:
        if file_count >= target_files:
            break
            
        content = item.get("text", "")
        url = item.get("url", "unknown")
        doc_id = item.get("id", f"doc_{file_count}")
        
        # Skip very short or very long content
        if len(content) < 200 or len(content) > 50000:
            continue
        
        # Create filename
        safe_name = f"{file_count:05d}_doc.md"
        file_path = output_path / safe_name
        
        # Save the content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        metadata.append({
            "file": safe_name,
            "source": url,
            "id": str(doc_id),
            "size": len(content)
        })
        
        file_count += 1
        pbar.update(1)
    
    pbar.close()
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Downloaded {file_count} documents to {output_path}")
    print(f"✓ Metadata saved to {metadata_path}")
    
    # Calculate total size
    total_size = sum(m["size"] for m in metadata)
    print(f"✓ Total text size: {total_size / 1024 / 1024:.1f} MB")
    
    return file_count


def download_wikipedia(
    output_dir: str = "data/markdown",
    target_files: int = 20000,
):
    """
    Download from Wikipedia dataset as an alternative.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading 'wikipedia' dataset (20220301.en)...")
    
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True
    )
    
    print(f"\nDownloading up to {target_files} articles...")
    
    file_count = 0
    metadata = []
    pbar = tqdm(total=target_files, desc="Downloading")
    
    for item in dataset:
        if file_count >= target_files:
            break
            
        content = item.get("text", "")
        title = item.get("title", "unknown")
        
        # Skip very short or very long content
        if len(content) < 500 or len(content) > 100000:
            continue
        
        # Create safe filename from title
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title[:50])
        safe_name = f"{file_count:05d}_{safe_title.strip().replace(' ', '_')}.md"
        file_path = output_path / safe_name
        
        # Format as markdown with title
        markdown_content = f"# {title}\n\n{content}"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        metadata.append({
            "file": safe_name,
            "title": title,
            "size": len(content)
        })
        
        file_count += 1
        pbar.update(1)
    
    pbar.close()
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Downloaded {file_count} Wikipedia articles to {output_path}")
    print(f"✓ Metadata saved to {metadata_path}")
    
    return file_count


def download_arxiv_abstracts(
    output_dir: str = "data/markdown",
    target_files: int = 20000,
):
    """
    Download arXiv paper abstracts - great for technical/scientific content.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading 'ccdv/arxiv-summarization' dataset...")
    
    dataset = load_dataset(
        "ccdv/arxiv-summarization",
        split="train",
        streaming=True
    )
    
    print(f"\nDownloading up to {target_files} papers...")
    
    file_count = 0
    metadata = []
    pbar = tqdm(total=target_files, desc="Downloading")
    
    for item in dataset:
        if file_count >= target_files:
            break
            
        abstract = item.get("abstract", "")
        article = item.get("article", "")
        
        # Use abstract + beginning of article
        content = abstract
        if article and len(content) < 1000:
            content = f"{abstract}\n\n{article[:5000]}"
        
        if len(content) < 200:
            continue
        
        safe_name = f"{file_count:05d}_arxiv.md"
        file_path = output_path / safe_name
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        metadata.append({
            "file": safe_name,
            "size": len(content)
        })
        
        file_count += 1
        pbar.update(1)
    
    pbar.close()
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Downloaded {file_count} arXiv papers to {output_path}")
    
    return file_count


def load_local_markdown_files(data_dir: str = "data/markdown") -> list[dict]:
    """
    Load downloaded markdown files for training.
    
    Returns:
        List of dicts with 'text' and 'source' keys
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found. "
            "Run the download command first."
        )
    
    documents = []
    
    for md_file in tqdm(list(data_path.glob("*.md")), desc="Loading files"):
        with open(md_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        documents.append({
            "text": content,
            "source": md_file.name
        })
    
    print(f"Loaded {len(documents)} documents")
    return documents


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download text dataset for training")
    parser.add_argument(
        "--output-dir", 
        default="data/markdown",
        help="Output directory for files"
    )
    parser.add_argument(
        "--target-files",
        type=int,
        default=20000,
        help="Number of files to download (default: 20000)"
    )
    parser.add_argument(
        "--source",
        choices=["fineweb", "wikipedia", "arxiv"],
        default="fineweb",
        help="Data source to use (default: fineweb)"
    )
    
    args = parser.parse_args()
    
    if args.source == "fineweb":
        download_fineweb_edu(
            output_dir=args.output_dir,
            target_files=args.target_files,
        )
    elif args.source == "wikipedia":
        download_wikipedia(
            output_dir=args.output_dir,
            target_files=args.target_files,
        )
    else:
        download_arxiv_abstracts(
            output_dir=args.output_dir,
            target_files=args.target_files,
        )
