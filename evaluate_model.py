#!/usr/bin/env python3
"""
This script evaluates a fine-tuned sentence transformer model on cybersecurity TTP test data.
It calculates similarity scores, accuracy, precision, recall, and F1-score.
"""

import os
import json
import logging
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(data_path):
    """Load the test dataset"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_model(model_path, test_data_path, output_dir=None):
    """Evaluate the model on test data"""
    # Load the model
    logger.info(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)
    
    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    test_data = load_test_data(test_data_path)
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(model_path, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract pairs
    sentences1 = [item["sentence1"] for item in test_data]
    sentences2 = [item["sentence2"] for item in test_data]
    true_labels = [item["label"] for item in test_data]
    
    # Compute embeddings
    logger.info("Computing embeddings for test data")
    embeddings1 = model.encode(sentences1, show_progress_bar=True)
    embeddings2 = model.encode(sentences2, show_progress_bar=True)
    
    # Compute cosine similarities
    cosine_scores = [util.cos_sim(emb1, emb2).item() for emb1, emb2 in zip(embeddings1, embeddings2)]
    
    # Convert to binary predictions using threshold of 0.5
    # (typical threshold for similarity, can be tuned)
    predicted_labels = [1.0 if score >= 0.5 else 0.0 for score in cosine_scores]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='binary', zero_division=0
    )
    
    # Print results
    logger.info(f"Evaluation Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Save results to file
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "details": [
            {
                "sentence1": s1,
                "sentence2": s2,
                "true_label": float(tl),
                "predicted_label": float(pl),
                "similarity_score": float(cs)
            }
            for s1, s2, tl, pl, cs in zip(sentences1, sentences2, true_labels, predicted_labels, cosine_scores)
        ]
    }
    
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot similarity score distribution
    plt.figure(figsize=(10, 6))
    
    # Separate positive and negative pairs
    positive_scores = [score for score, label in zip(cosine_scores, true_labels) if label == 1.0]
    negative_scores = [score for score, label in zip(cosine_scores, true_labels) if label == 0.0]
    
    # Plot histograms
    plt.hist(positive_scores, alpha=0.7, label='Positive Pairs (Same Category)', bins=20, range=(-0.1, 1.1))
    plt.hist(negative_scores, alpha=0.7, label='Negative Pairs (Different Categories)', bins=20, range=(-0.1, 1.1))
    
    plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    plt.title("Distribution of Similarity Scores")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "similarity_distribution.png"))
    
    # Plot confusion matrix
    confusion = {
        "true_positives": sum(1 for pl, tl in zip(predicted_labels, true_labels) if pl == 1.0 and tl == 1.0),
        "false_positives": sum(1 for pl, tl in zip(predicted_labels, true_labels) if pl == 1.0 and tl == 0.0),
        "true_negatives": sum(1 for pl, tl in zip(predicted_labels, true_labels) if pl == 0.0 and tl == 0.0),
        "false_negatives": sum(1 for pl, tl in zip(predicted_labels, true_labels) if pl == 0.0 and tl == 1.0)
    }
    
    # Create a 2x2 confusion matrix plot
    cm = np.array([
        [confusion["true_negatives"], confusion["false_positives"]],
        [confusion["false_negatives"], confusion["true_positives"]]
    ])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    classes = ["Negative (0)", "Positive (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    logger.info(f"Evaluation results saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Sentence Transformer model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--test_data", type=str, default="data/test.json",
                        help="Path to test data (default: data/test.json)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (default: model_path/evaluation)")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not Path(args.model_path).exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        exit(1)
    
    # Check if test data exists
    if not Path(args.test_data).exists():
        logger.error(f"Test data file does not exist: {args.test_data}")
        exit(1)
    
    # Evaluate the model
    results = evaluate_model(args.model_path, args.test_data, args.output_dir)
    
    print(f"Evaluation completed successfully!") 