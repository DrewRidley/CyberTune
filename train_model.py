#!/usr/bin/env python3
"""
This script fine-tunes a sentence transformer model on a cybersecurity TTP dataset.
It uses a contrastive learning approach with a similarity loss function.
"""

import os
import json
import logging
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load JSON dataset"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def create_examples(pairs):
    """Convert data pairs to InputExample objects"""
    examples = []
    for pair in pairs:
        examples.append(
            InputExample(
                texts=[pair['sentence1'], pair['sentence2']],
                label=float(pair['label'])
            )
        )
    return examples

def plot_training_loss(loss_values, output_path):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "training_loss.png"))
    plt.close()

def train_model(args):
    """Fine-tune the sentence transformer model"""
    # Create output directory
    output_path = os.path.join(args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_path, exist_ok=True)
    
    # Load training and evaluation data
    logger.info(f"Loading training data from {args.train_data}")
    train_data = load_data(args.train_data)
    train_examples = create_examples(train_data)
    
    logger.info(f"Loading evaluation data from {args.dev_data}")
    dev_data = load_data(args.dev_data)
    dev_examples = create_examples(dev_data)
    
    # Create evaluator
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name='sts-dev')
    
    # Load base model
    logger.info(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    
    # Define loss function
    logger.info(f"Using loss function: {args.loss}")
    if args.loss == 'cosine':
        train_loss = losses.CosineSimilarityLoss(model)
    elif args.loss == 'contrastive':
        train_loss = losses.ContrastiveLoss(model)
    elif args.loss == 'online_contrastive':
        train_loss = losses.OnlineContrastiveLoss(model)
    elif args.loss == 'triplet':
        train_loss = losses.TripletLoss(model)
    elif args.loss == 'multiple_negatives_ranking':
        train_loss = losses.MultipleNegativesRankingLoss(model)
    else:
        logger.error(f"Unknown loss function: {args.loss}")
        return
    
    # Set warmup steps
    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)
    
    # Train the model
    logger.info(f"Starting training with {len(train_examples)} examples for {args.epochs} epochs")
    logger.info(f"Evaluation will be performed every {args.eval_steps} steps")
    
    train_loss_values = []
    
    # Train with callback to collect loss values
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=args.eval_steps,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        callback=lambda score, epoch, steps: train_loss_values.append(score)
    )
    
    # Plot training loss
    plot_training_loss(train_loss_values, output_path)
    
    # Save a simple model card
    with open(os.path.join(output_path, 'model_card.md'), 'w') as f:
        f.write(f"# Cybersecurity TTP Fine-tuned Sentence Transformer\n\n")
        f.write(f"Base model: {args.base_model}\n")
        f.write(f"Fine-tuned for cybersecurity TTPs (Tactics, Techniques, and Procedures)\n\n")
        f.write(f"## Training Parameters\n\n")
        f.write(f"- Loss function: {args.loss}\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Learning rate: {args.lr}\n")
        f.write(f"- Training examples: {len(train_examples)}\n")
        f.write(f"- Evaluation examples: {len(dev_examples)}\n")
    
    logger.info(f"Model saved to {output_path}")
    
    return model, output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Sentence Transformer model for cybersecurity TTPs")
    
    parser.add_argument("--base_model", type=str, default="all-MiniLM-L6-v2", 
                        help="Base model to fine-tune (default: all-MiniLM-L6-v2)")
    parser.add_argument("--train_data", type=str, default="data/train.json", 
                        help="Path to training data (default: data/train.json)")
    parser.add_argument("--dev_data", type=str, default="data/dev.json", 
                        help="Path to development/validation data (default: data/dev.json)")
    parser.add_argument("--output_dir", type=str, default="models", 
                        help="Directory to save the fine-tuned model (default: models)")
    parser.add_argument("--epochs", type=int, default=4, 
                        help="Number of training epochs (default: 4)")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Training batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-5, 
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--eval_steps", type=int, default=1000, 
                        help="Evaluation steps (default: 1000)")
    parser.add_argument("--loss", type=str, default="cosine", 
                        choices=["cosine", "contrastive", "online_contrastive", "triplet", "multiple_negatives_ranking"],
                        help="Loss function to use (default: cosine)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    model, model_path = train_model(args)
    
    print(f"Training completed! Model saved to: {model_path}") 