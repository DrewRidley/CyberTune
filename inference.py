#!/usr/bin/env python3
"""
This script demonstrates how to use a fine-tuned sentence transformer model
for analyzing cybersecurity texts to identify and correlate TTPs.
"""

import os
import json
import logging
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Example cybersecurity texts for inference
EXAMPLE_TEXTS = [
    # Initial Access
    "The attack began with a phishing campaign targeting finance department employees with malicious Excel attachments.",
    "Employees received spear phishing emails with links to fake login pages resembling the company's VPN portal.",
    
    # Execution
    "After gaining access, the threat actor executed PowerShell scripts to establish persistence and disable security controls.",
    "The malware created scheduled tasks to execute the payload during system startup.",
    
    # Persistence
    "The attackers maintained access by creating new service entries that launched at system boot.",
    "Registry run keys were modified to ensure the malware would execute when users logged into the system.",
    
    # Lateral Movement
    "Using compromised credentials, the attackers moved laterally through the network via RDP connections.",
    "The threat actor used pass-the-hash techniques to access other systems without needing the actual passwords.",
    
    # Exfiltration
    "Data was compressed and encrypted before being exfiltrated through DNS tunneling to avoid detection.",
    "The attackers used steganography to hide stolen data within image files before sending them to external servers."
]

def load_categories(categories_path):
    """Load TTP categories for reference"""
    with open(categories_path, 'r') as f:
        categories = json.load(f)
    return categories

def analyze_text(model_path, texts=None, categories_file=None, output_dir=None):
    """Analyze cybersecurity text with the fine-tuned model"""
    # Load the model
    logger.info(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)
    
    # Use example texts if none provided
    if texts is None:
        texts = EXAMPLE_TEXTS
        logger.info("Using example texts for analysis")
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(model_path, "inference")
    os.makedirs(output_dir, exist_ok=True)
    
    # Encode the texts
    logger.info("Computing embeddings for input texts")
    text_embeddings = model.encode(texts, show_progress_bar=True)
    
    # Calculate similarity matrix
    similarity_matrix = util.cos_sim(text_embeddings, text_embeddings).numpy()
    
    # Save similarity matrix
    with open(os.path.join(output_dir, "similarity_matrix.json"), "w") as f:
        json.dump({
            "texts": texts,
            "similarity_matrix": similarity_matrix.tolist()
        }, f, indent=2)
    
    # Visualize similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Cosine Similarity')
    plt.title("Cosine Similarity Between Text Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_matrix.png"))
    
    # Cluster the texts based on similarity
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is zero
    
    # Fit clustering model - using parameters compatible with all scikit-learn versions
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.5,
        metric='precomputed',
        linkage='average'
    )
    
    # Fit clustering model
    clustering_model.fit(distance_matrix)
    clusters = clustering_model.labels_
    
    # Organize results by cluster
    cluster_results = {}
    for i, (text, cluster) in enumerate(zip(texts, clusters)):
        # Convert NumPy integer to Python integer for JSON serialization
        cluster_id = int(cluster)
        if cluster_id not in cluster_results:
            cluster_results[cluster_id] = []
        cluster_results[cluster_id].append({
            "id": i,
            "text": text
        })
    
    # Save clustering results
    with open(os.path.join(output_dir, "clusters.json"), "w") as f:
        json.dump({
            "num_clusters": len(cluster_results),
            "clusters": cluster_results
        }, f, indent=2)
    
    # Print clustering results
    logger.info(f"Found {len(cluster_results)} clusters:")
    for cluster_id, items in cluster_results.items():
        logger.info(f"\nCluster {cluster_id}:")
        for item in items:
            logger.info(f"  - {item['text']}")
    
    # If categories file is provided, try to map clusters to TTP categories
    if categories_file and Path(categories_file).exists():
        logger.info(f"Loading TTP categories from {categories_file}")
        categories = load_categories(categories_file)
        
        # Create list of all category examples
        category_texts = []
        category_labels = []
        
        for category, examples in categories.items():
            for example in examples:
                category_texts.append(example)
                category_labels.append(category)
        
        # Encode category examples
        logger.info("Computing embeddings for category examples")
        category_embeddings = model.encode(category_texts, show_progress_bar=True)
        
        # For each text, find the most similar category
        logger.info("Matching texts to TTP categories")
        text_categories = []
        
        for text_embedding in text_embeddings:
            # Calculate similarities with all category examples
            similarities = util.cos_sim(text_embedding, category_embeddings).numpy()[0]
            
            # Find the index of the most similar category example
            most_similar_idx = np.argmax(similarities)
            
            # Get the corresponding category
            matched_category = category_labels[most_similar_idx]
            matched_example = category_texts[most_similar_idx]
            similarity_score = similarities[most_similar_idx]
            
            text_categories.append({
                "category": matched_category,
                "matched_example": matched_example,
                "similarity_score": float(similarity_score)
            })
        
        # Save category mapping results
        with open(os.path.join(output_dir, "category_mapping.json"), "w") as f:
            json.dump({
                "results": [
                    {
                        "text": text,
                        "category": cat["category"],
                        "matched_example": cat["matched_example"],
                        "similarity_score": cat["similarity_score"]
                    }
                    for text, cat in zip(texts, text_categories)
                ]
            }, f, indent=2)
        
        # Print category mapping
        logger.info("\nCategory Mapping:")
        for text, category_info in zip(texts, text_categories):
            logger.info(f"\nText: {text}")
            logger.info(f"Mapped Category: {category_info['category']}")
            logger.info(f"Similarity Score: {category_info['similarity_score']:.4f}")
    
    logger.info(f"Analysis results saved to {output_dir}")
    
    return {
        "clusters": cluster_results,
        "categories": text_categories if categories_file and Path(categories_file).exists() else None
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use a fine-tuned Sentence Transformer model for cybersecurity text analysis")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to a file containing texts to analyze (one per line)")
    parser.add_argument("--categories_file", type=str, default="data/categories.json",
                        help="Path to categories json file (default: data/categories.json)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save inference results (default: model_path/inference)")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not Path(args.model_path).exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        exit(1)
    
    # Load input texts if provided
    texts = None
    if args.input_file and Path(args.input_file).exists():
        with open(args.input_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} texts from {args.input_file}")
    
    # Run analysis
    results = analyze_text(args.model_path, texts, args.categories_file, args.output_dir)
    
    print(f"Analysis completed successfully!") 