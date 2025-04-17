#!/usr/bin/env python3
"""
Advanced training script implementing several performance enhancement strategies:
1. Longer training with early stopping
2. Hard negative mining
3. Better hyperparameter tuning
4. Option for stronger base models
"""

import os
import json
import logging
import argparse
import random
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def perform_hard_negative_mining(model, train_data, num_hard_negatives=100):
    """Find hard negative examples from the training data"""
    logger.info(f"Performing hard negative mining to find {num_hard_negatives} challenging examples...")
    
    # Get sentence pairs with label 0 (negative pairs)
    negative_pairs = [pair for pair in train_data if pair['label'] < 0.5]
    if len(negative_pairs) < num_hard_negatives:
        logger.warning(f"Not enough negative pairs for mining. Found only {len(negative_pairs)}.")
        return train_data
    
    # Encode all negative sentences
    sentences1 = [pair['sentence1'] for pair in negative_pairs]
    sentences2 = [pair['sentence2'] for pair in negative_pairs]
    
    embeddings1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True)
    
    # Calculate similarities
    from sentence_transformers import util
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    
    # Get scores for each negative pair
    pair_scores = []
    for i in range(len(negative_pairs)):
        pair_scores.append((i, cosine_scores[i][i].item()))
    
    # Sort by similarity score (higher = harder negative)
    pair_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top N hard negatives
    hard_negative_indices = [idx for idx, _ in pair_scores[:num_hard_negatives]]
    hard_negatives = [negative_pairs[idx] for idx in hard_negative_indices]
    
    # Get positive pairs
    positive_pairs = [pair for pair in train_data if pair['label'] >= 0.5]
    
    # Combine positive pairs with hard negatives
    enhanced_data = positive_pairs + hard_negatives
    
    logger.info(f"Enhanced dataset created with {len(positive_pairs)} positive and {len(hard_negatives)} hard negative examples")
    return enhanced_data

def create_balanced_triplet_examples(train_data, num_triplets=5000):
    """Create balanced triplet examples from the training data"""
    logger.info(f"Creating {num_triplets} balanced triplet examples...")
    
    # Separate positive and negative pairs
    positive_pairs = [pair for pair in train_data if pair['label'] >= 0.5]
    negative_pairs = [pair for pair in train_data if pair['label'] < 0.5]
    
    if len(positive_pairs) == 0 or len(negative_pairs) == 0:
        logger.warning("Cannot create triplets: need both positive and negative pairs.")
        return []
    
    # Create dictionary of sentences by category (assuming pairs within same category are positive)
    sentences_by_category = {}
    for pair in positive_pairs:
        # Extract category from sentence if available
        # This assumes the category is available in the data or can be inferred
        category = pair.get('category', 'default_category')
        
        if category not in sentences_by_category:
            sentences_by_category[category] = []
        
        # Add both sentences to this category
        if pair['sentence1'] not in sentences_by_category[category]:
            sentences_by_category[category].append(pair['sentence1'])
        if pair['sentence2'] not in sentences_by_category[category]:
            sentences_by_category[category].append(pair['sentence2'])
    
    # Create triplets: (anchor, positive, negative)
    triplets = []
    categories = list(sentences_by_category.keys())
    
    for _ in range(min(num_triplets, len(positive_pairs) * 2)):
        # Select a random category
        category = random.choice(categories)
        if len(sentences_by_category[category]) < 2:
            continue
            
        # Select anchor and positive from same category
        anchor, positive = random.sample(sentences_by_category[category], 2)
        
        # Select negative from different category
        neg_category = random.choice([c for c in categories if c != category])
        while not sentences_by_category.get(neg_category, []):
            neg_category = random.choice([c for c in categories if c != category])
            
        negative = random.choice(sentences_by_category[neg_category])
        
        triplets.append((anchor, positive, negative))
    
    logger.info(f"Created {len(triplets)} triplet examples")
    return triplets

def create_sbert_triplet_examples(train_data, num_triplets=None):
    """Convert data to the InputExample for triplet loss"""
    # First try to create balanced triplets if we have category information
    triplets = create_balanced_triplet_examples(train_data, num_triplets)
    
    examples = []
    for anchor, positive, negative in triplets:
        examples.append(InputExample(texts=[anchor, positive, negative]))
        
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

def create_training_loss_callback(loss_values):
    """Create a callback to record training loss values"""
    def callback(score, epoch, steps):
        loss_values.append(score)
    return callback

def train_model(args):
    """Fine-tune the sentence transformer model with advanced settings"""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = os.path.join(args.output_dir, f"advanced_{args.strategy}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # Log training configuration
    logger.info(f"Training configuration:")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Training strategy: {args.strategy}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Loss function: {args.loss}")
    
    # Load training and evaluation data
    logger.info(f"Loading training data from {args.train_data}")
    train_data = load_data(args.train_data)
    
    logger.info(f"Loading evaluation data from {args.dev_data}")
    dev_data = load_data(args.dev_data)
    dev_examples = create_examples(dev_data)
    
    # Load base model
    logger.info(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)
    
    # Apply training strategy
    if args.strategy == 'hard_negative_mining':
        logger.info("Applying hard negative mining strategy")
        # Initial model to find hard negatives
        train_data = perform_hard_negative_mining(model, train_data, num_hard_negatives=args.hard_negatives)
        train_examples = create_examples(train_data)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        
    elif args.strategy == 'triplet':
        logger.info("Using triplet loss with carefully selected triplets")
        # Create triplet examples
        train_examples = create_sbert_triplet_examples(train_data, num_triplets=args.triplets)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        
    else:  # default strategy
        logger.info("Using default training strategy")
        train_examples = create_examples(train_data)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    
    # Create evaluator
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name='sts-dev')
    
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
    
    # Set warmup steps (10% of training steps)
    steps_per_epoch = len(train_dataloader)
    warmup_steps = int(steps_per_epoch * args.epochs * args.warmup_ratio)
    total_steps = steps_per_epoch * args.epochs
    
    logger.info(f"Training steps per epoch: {steps_per_epoch}")
    logger.info(f"Warmup steps: {warmup_steps} ({args.warmup_ratio * 100:.1f}% of total)")
    logger.info(f"Total training steps: {total_steps}")
    
    # Configure early stopping
    early_stopping_patience = args.patience
    best_score = -1
    early_stopping_counter = 0
    best_model_path = os.path.join(output_path, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    
    # Set up training loss tracking
    train_loss_values = []
    eval_scores = []
    
    # Define save_best_model callback
    def save_best_model_callback(score, epoch, steps):
        nonlocal best_score, early_stopping_counter
        
        # Record evaluation score
        eval_scores.append(score)
        
        # Check if this is the best score
        if score > best_score:
            logger.info(f"New best score: {score:.4f} (prev: {best_score:.4f})")
            best_score = score
            early_stopping_counter = 0
            
            # Save the best model
            model.save(best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
        else:
            early_stopping_counter += 1
            logger.info(f"No improvement for {early_stopping_counter}/{early_stopping_patience} evaluations")
            
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered! Best score: {best_score:.4f}")
                # Use RuntimeError to stop training
                raise RuntimeError("Early stopping triggered")
    
    # Train the model with callbacks
    logger.info(f"Starting advanced training with {len(train_examples)} examples for up to {args.epochs} epochs")
    logger.info(f"Evaluation will be performed every {args.eval_steps} steps")
    
    train_loss_callback = create_training_loss_callback(train_loss_values)
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=args.epochs,
            evaluation_steps=args.eval_steps,
            warmup_steps=warmup_steps,
            scheduler='warmuplinear',
            optimizer_params={'lr': args.lr},
            show_progress_bar=True,
            checkpoint_path=os.path.join(output_path, 'checkpoints'),
            checkpoint_save_steps=args.eval_steps * 2,
            callback=lambda score, epoch, steps: (train_loss_callback(score, epoch, steps), 
                                                  save_best_model_callback(score, epoch, steps))
        )
    except RuntimeError as e:
        if str(e) == "Early stopping triggered":
            logger.info("Training stopped early due to no improvement in validation score.")
        else:
            raise e
    
    # Create plots
    # 1. Plot training loss
    plot_training_loss(train_loss_values, output_path)
    
    # 2. Plot evaluation scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(eval_scores)), eval_scores, marker='o')
    plt.title("Evaluation Scores During Training")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Cosine Similarity Score")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "evaluation_scores.png"))
    plt.close()
    
    # Save a model card
    with open(os.path.join(output_path, 'model_card.md'), 'w') as f:
        f.write(f"# Advanced Cybersecurity TTP Fine-tuned Sentence Transformer\n\n")
        f.write(f"Base model: {args.base_model}\n")
        f.write(f"Fine-tuned for cybersecurity TTPs (Tactics, Techniques, and Procedures)\n\n")
        f.write(f"## Training Parameters\n\n")
        f.write(f"- Training strategy: {args.strategy}\n")
        f.write(f"- Loss function: {args.loss}\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Learning rate: {args.lr}\n")
        f.write(f"- Warmup ratio: {args.warmup_ratio}\n")
        f.write(f"- Best evaluation score: {best_score:.4f}\n")
        f.write(f"- Training examples: {len(train_examples)}\n")
        f.write(f"- Evaluation examples: {len(dev_examples)}\n")
        f.write(f"\n## Dataset\n\n")
        f.write(f"This model was trained on an enhanced dataset of cybersecurity TTPs including real-world examples.\n")
        f.write(f"Training strategy '{args.strategy}' was used to improve the model's performance.\n")
    
    logger.info(f"Model trained and saved to {output_path}")
    logger.info(f"Best model saved to {best_model_path} with score: {best_score:.4f}")
    
    # Copy the best model to output_path for consistency
    if os.path.exists(best_model_path):
        # Load best model for return
        best_model = SentenceTransformer(best_model_path)
        return best_model, best_model_path
    
    return model, output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an advanced sentence transformer model for cybersecurity TTPs")
    
    # Model configuration
    parser.add_argument("--base_model", type=str, default="all-mpnet-base-v2", 
                        help="Base model to fine-tune (default: all-mpnet-base-v2)")
    
    # Data configuration
    parser.add_argument("--train_data", type=str, default="data/combined_train.json", 
                        help="Path to combined training data")
    parser.add_argument("--dev_data", type=str, default="data/combined_dev.json", 
                        help="Path to combined development/validation data")
    parser.add_argument("--output_dir", type=str, default="models", 
                        help="Directory to save the fine-tuned model")
    
    # Training strategy
    parser.add_argument("--strategy", type=str, default="default", 
                        choices=["default", "hard_negative_mining", "triplet"],
                        help="Training strategy to use")
    
    # Hard negative mining parameters
    parser.add_argument("--hard_negatives", type=int, default=200, 
                        help="Number of hard negative examples to mine (for hard_negative_mining strategy)")
    
    # Triplet parameters
    parser.add_argument("--triplets", type=int, default=3000, 
                        help="Number of triplets to create (for triplet strategy)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=15, 
                        help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=250, 
                        help="Steps between evaluations")
    parser.add_argument("--loss", type=str, default="multiple_negatives_ranking", 
                        choices=["cosine", "contrastive", "online_contrastive", "triplet", "multiple_negatives_ranking"],
                        help="Loss function to use")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, 
                        help="Ratio of total steps used for warmup")
    parser.add_argument("--patience", type=int, default=5, 
                        help="Number of evaluations with no improvement after which training will be stopped")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    model, model_path = train_model(args) 