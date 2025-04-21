#!/usr/bin/env python3
"""
Demo script for MITRE TTP categorization using the FineTuneST model.
This script allows you to paste a paragraph of cybersecurity text and get the associated MITRE TTP categories.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_categories(categories_path):
    """Load TTP categories for reference"""
    logger.info(f"Loading TTP categories from {categories_path}")
    with open(categories_path, 'r') as f:
        categories = json.load(f)
    return categories

def categorize_text(text, model, categories_data):
    """Categorize a single text input against all TTP categories"""
    # Create list of all category examples
    category_texts = []
    category_labels = []
    
    for category, examples in categories_data.items():
        for example in examples:
            category_texts.append(example)
            category_labels.append(category)
    
    # Encode category examples and input text
    logger.info("Computing embeddings for categories and input text...")
    category_embeddings = model.encode(category_texts, show_progress_bar=False)
    text_embedding = model.encode([text], show_progress_bar=False)[0]
    
    # Calculate similarities with all category examples
    similarities = util.cos_sim(text_embedding, category_embeddings).numpy()[0]
    
    # Sort by similarity score (highest first)
    sorted_indices = np.argsort(-similarities)
    
    # Return top 3 matches
    results = []
    for idx in sorted_indices[:3]:
        results.append({
            "category": category_labels[idx],
            "matched_example": category_texts[idx],
            "similarity_score": float(similarities[idx])
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Categorize cybersecurity text using MITRE ATT&CK framework")
    
    parser.add_argument("--model_path", type=str, default="checkpoints/model",
                        help="Path to the fine-tuned model directory (default: checkpoints/model)")
    parser.add_argument("--categories_file", type=str, default="data/categories.json",
                        help="Path to categories json file (default: data/categories.json)")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to analyze (if not provided, will prompt for input)")
    
    args = parser.parse_args()
    
    # Set model path
    model_path = args.model_path
    
    # Check if model path exists
    if not Path(model_path).exists():
        logger.error(f"Model path does not exist: {model_path}")
        logger.info("Falling back to using the base sentence-transformer model 'all-MiniLM-L6-v2'")
        model_path = 'all-MiniLM-L6-v2'
    
    # Load the model
    try:
        logger.info(f"Loading model from {model_path}")
        model = SentenceTransformer(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Falling back to using the base sentence-transformer model 'all-MiniLM-L6-v2'")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load categories
    categories_path = args.categories_file
    if not Path(categories_path).exists():
        logger.error(f"Categories file does not exist: {categories_path}")
        sys.exit(1)
    
    categories_data = load_categories(categories_path)
    
    # Get input text
    if args.text:
        input_text = args.text
    else:
        print("\n" + "="*50)
        print("Cybersecurity Text to MITRE ATT&CK TTP Analyzer")
        print("="*50)
        print("\nEnter or paste your cybersecurity text (press Ctrl+D or Ctrl+Z when finished):")
        lines = []
        try:
            for line in sys.stdin:
                lines.append(line)
        except KeyboardInterrupt:
            print("\nInput terminated.")
        input_text = ''.join(lines)
    
    if not input_text.strip():
        logger.error("No input text provided.")
        sys.exit(1)
    
    print("\nAnalyzing text:")
    print("-"*50)
    print(input_text.strip())
    print("-"*50)
    
    # Categorize the text
    results = categorize_text(input_text.strip(), model, categories_data)
    
    # Display results
    print("\nTop 3 MITRE ATT&CK TTPs:")
    print("="*50)
    
    for i, result in enumerate(results, 1):
        category = result["category"]
        similarity = result["similarity_score"] * 100
        print(f"{i}. {category} (Confidence: {similarity:.2f}%)")
        print(f"   Example: {result['matched_example']}")
        print()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()