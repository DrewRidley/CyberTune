#!/usr/bin/env python3
"""
This script evaluates the performance of the fine-tuned model against the baseline model
using unseen test data to demonstrate the effectiveness of fine-tuning.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, util
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def load_test_data(test_data_path):
    """Load test data from JSON file"""
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    return test_data

def evaluate_models(args):
    """Compare fine-tuned model with baseline model on test data"""
    print(f"Loading test data from {args.test_data}")
    test_data = load_test_data(args.test_data)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load models
    print(f"Loading baseline model: {args.base_model}")
    base_model = SentenceTransformer(args.base_model)
    
    print(f"Loading fine-tuned model: {args.fine_tuned_model}")
    fine_tuned_model = SentenceTransformer(args.fine_tuned_model)
    
    # Extract sentence pairs and true labels
    sentences1 = [item['sentence1'] for item in test_data]
    sentences2 = [item['sentence2'] for item in test_data]
    true_labels = np.array([float(item['label']) for item in test_data])
    true_classes = (true_labels >= 0.5).astype(int)  # Convert similarity scores to binary classes
    
    # Calculate embeddings for both models
    print("Calculating embeddings with baseline model...")
    base_embeddings1 = base_model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True)
    base_embeddings2 = base_model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True)
    
    print("Calculating embeddings with fine-tuned model...")
    ft_embeddings1 = fine_tuned_model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True)
    ft_embeddings2 = fine_tuned_model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True)
    
    # Calculate cosine similarities
    print("Calculating similarities...")
    base_similarities = util.pytorch_cos_sim(base_embeddings1, base_embeddings2).diagonal().cpu().numpy()
    ft_similarities = util.pytorch_cos_sim(ft_embeddings1, ft_embeddings2).diagonal().cpu().numpy()
    
    # Convert similarities to predicted classes
    base_predicted_classes = (base_similarities >= 0.5).astype(int)
    ft_predicted_classes = (ft_similarities >= 0.5).astype(int)
    
    # Calculate metrics
    base_accuracy = accuracy_score(true_classes, base_predicted_classes)
    ft_accuracy = accuracy_score(true_classes, ft_predicted_classes)
    
    base_precision, base_recall, base_f1, _ = precision_recall_fscore_support(true_classes, base_predicted_classes, average='binary')
    ft_precision, ft_recall, ft_f1, _ = precision_recall_fscore_support(true_classes, ft_predicted_classes, average='binary')
    
    # Print results
    print("\n===== Performance Comparison =====")
    print(f"Test samples: {len(test_data)}")
    print("\nBaseline Model Performance:")
    print(f"Accuracy: {base_accuracy:.4f}")
    print(f"Precision: {base_precision:.4f}")
    print(f"Recall: {base_recall:.4f}")
    print(f"F1 Score: {base_f1:.4f}")
    
    print("\nFine-tuned Model Performance:")
    print(f"Accuracy: {ft_accuracy:.4f}")
    print(f"Precision: {ft_precision:.4f}")
    print(f"Recall: {ft_recall:.4f}")
    print(f"F1 Score: {ft_f1:.4f}")
    
    print("\nImprovement:")
    print(f"Accuracy: {(ft_accuracy - base_accuracy) * 100:.2f}% absolute")
    print(f"F1 Score: {(ft_f1 - base_f1) * 100:.2f}% absolute")
    
    # Save results to file
    with open(output_dir / "performance_comparison.txt", "w") as f:
        f.write("===== Performance Comparison =====\n")
        f.write(f"Test samples: {len(test_data)}\n\n")
        f.write("Baseline Model Performance:\n")
        f.write(f"Accuracy: {base_accuracy:.4f}\n")
        f.write(f"Precision: {base_precision:.4f}\n")
        f.write(f"Recall: {base_recall:.4f}\n")
        f.write(f"F1 Score: {base_f1:.4f}\n\n")
        
        f.write("Fine-tuned Model Performance:\n")
        f.write(f"Accuracy: {ft_accuracy:.4f}\n")
        f.write(f"Precision: {ft_precision:.4f}\n")
        f.write(f"Recall: {ft_recall:.4f}\n")
        f.write(f"F1 Score: {ft_f1:.4f}\n\n")
        
        f.write("Improvement:\n")
        f.write(f"Accuracy: {(ft_accuracy - base_accuracy) * 100:.2f}% absolute\n")
        f.write(f"F1 Score: {(ft_f1 - base_f1) * 100:.2f}% absolute\n")
    
    # Create visualizations
    
    # 1. Confusion matrices
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    base_cm = confusion_matrix(true_classes, base_predicted_classes)
    ConfusionMatrixDisplay(base_cm, display_labels=['Not Similar', 'Similar']).plot(ax=plt.gca(), values_format='d', cmap='Blues')
    plt.title(f"Baseline Model\nAccuracy: {base_accuracy:.4f}")
    
    plt.subplot(1, 2, 2)
    ft_cm = confusion_matrix(true_classes, ft_predicted_classes)
    ConfusionMatrixDisplay(ft_cm, display_labels=['Not Similar', 'Similar']).plot(ax=plt.gca(), values_format='d', cmap='Blues')
    plt.title(f"Fine-tuned Model\nAccuracy: {ft_accuracy:.4f}")
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_comparison.png", dpi=300)
    
    # 2. Similarity distribution comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data={'True Positives': base_similarities[true_classes == 1], 
                      'True Negatives': base_similarities[true_classes == 0]}, 
                bins=25, element="step", common_norm=False)
    plt.axvline(x=0.5, color='red', linestyle='--')
    plt.title("Baseline Model Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(data={'True Positives': ft_similarities[true_classes == 1], 
                      'True Negatives': ft_similarities[true_classes == 0]}, 
                bins=25, element="step", common_norm=False)
    plt.axvline(x=0.5, color='red', linestyle='--')
    plt.title("Fine-tuned Model Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "similarity_distribution_comparison.png", dpi=300)
    
    # 3. Performance metrics bar chart
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    base_scores = [base_accuracy, base_precision, base_recall, base_f1]
    ft_scores = [ft_accuracy, ft_precision, ft_recall, ft_f1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, base_scores, width, label='Baseline Model')
    plt.bar(x + width/2, ft_scores, width, label='Fine-tuned Model')
    
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(base_scores):
        plt.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center')
    
    for i, v in enumerate(ft_scores):
        plt.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_metrics_comparison.png", dpi=300)
    
    # 4. Real-world example evaluation
    if args.example_texts:
        print("\n===== Testing with Real-world Examples =====")
        real_world_examples = []
        with open(args.example_texts, 'r') as f:
            real_world_examples = [line.strip() for line in f if line.strip()]
        
        if not real_world_examples:
            print("No examples found in the example texts file.")
            return
        
        # Generate all pairwise combinations
        example_pairs = []
        for i in range(len(real_world_examples)):
            for j in range(i+1, len(real_world_examples)):
                example_pairs.append((real_world_examples[i], real_world_examples[j]))
        
        # Calculate similarities with both models
        examples1 = [pair[0] for pair in example_pairs]
        examples2 = [pair[1] for pair in example_pairs]
        
        base_ex_embeddings1 = base_model.encode(examples1, convert_to_tensor=True)
        base_ex_embeddings2 = base_model.encode(examples2, convert_to_tensor=True)
        
        ft_ex_embeddings1 = fine_tuned_model.encode(examples1, convert_to_tensor=True)
        ft_ex_embeddings2 = fine_tuned_model.encode(examples2, convert_to_tensor=True)
        
        base_ex_similarities = util.pytorch_cos_sim(base_ex_embeddings1, base_ex_embeddings2).diagonal().cpu().numpy()
        ft_ex_similarities = util.pytorch_cos_sim(ft_ex_embeddings1, ft_ex_embeddings2).diagonal().cpu().numpy()
        
        # Output the results for real-world examples
        with open(output_dir / "real_world_example_comparison.txt", "w") as f:
            f.write("===== Real-world Example Similarity Comparison =====\n\n")
            
            for i, pair in enumerate(example_pairs):
                f.write(f"Example Pair {i+1}:\n")
                f.write(f"Text 1: {pair[0][:100]}...\n")
                f.write(f"Text 2: {pair[1][:100]}...\n")
                f.write(f"Baseline Model Similarity: {base_ex_similarities[i]:.4f}\n")
                f.write(f"Fine-tuned Model Similarity: {ft_ex_similarities[i]:.4f}\n")
                f.write(f"Difference: {(ft_ex_similarities[i] - base_ex_similarities[i]):.4f}\n\n")
        
        print(f"Real-world example comparison saved to {output_dir}/real_world_example_comparison.txt")
    
    print(f"\nAll comparison results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fine-tuned model with baseline on unseen data")
    
    parser.add_argument("--base_model", type=str, default="all-mpnet-base-v2",
                        help="Base model without fine-tuning (default: all-mpnet-base-v2)")
    parser.add_argument("--fine_tuned_model", type=str, required=True,
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--test_data", type=str, default="data/combined_test.json",
                        help="Path to test data JSON file (default: data/combined_test.json)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results (default: evaluation_results)")
    parser.add_argument("--example_texts", type=str,
                        help="Path to a file containing real-world example texts (one per line)")
    
    args = parser.parse_args()
    
    evaluate_models(args) 