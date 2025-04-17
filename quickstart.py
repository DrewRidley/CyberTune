#!/usr/bin/env python3
"""
This script runs the entire fine-tuning process for a sentence transformer model
on cybersecurity TTPs from data creation to inference.
"""

import os
import argparse
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO)
logger = logging.getLogger(__name__)

def run_step(command, description):
    """Run a command and log its output"""
    logger.info(f"=== {description} ===")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if process.returncode == 0:
        logger.info(process.stdout)
        logger.info(f"✅ {description} completed successfully")
        return True
    else:
        logger.error(f"❌ {description} failed with error code {process.returncode}")
        logger.error(process.stderr)
        return False

def main(args):
    """Run the entire workflow"""
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Step 1: Create dataset
    if not run_step("python create_dataset.py", "Dataset Creation"):
        return
    
    # Step 2: Train model
    train_cmd = (
        f"python train_model.py "
        f"--base_model {args.base_model} "
        f"--epochs {args.epochs} "
        f"--batch_size {args.batch_size} "
        f"--loss {args.loss}"
    )
    if not run_step(train_cmd, "Model Training"):
        return
    
    # Find the most recently created model directory
    model_dirs = sorted(Path("models").glob("*"), key=os.path.getmtime, reverse=True)
    if not model_dirs:
        logger.error("❌ No model directory found")
        return
    
    model_path = str(model_dirs[0])
    logger.info(f"Using most recent model: {model_path}")
    
    # Step 3: Evaluate model
    eval_cmd = f"python evaluate_model.py --model_path {model_path}"
    if not run_step(eval_cmd, "Model Evaluation"):
        return
    
    # Step 4: Run inference
    inference_cmd = f"python inference.py --model_path {model_path}"
    if not run_step(inference_cmd, "Model Inference"):
        return
    
    logger.info("\n🎉 Complete workflow executed successfully! 🎉")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Evaluation results: {model_path}/evaluation")
    logger.info(f"Inference results: {model_path}/inference")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full sentence transformer fine-tuning pipeline")
    
    parser.add_argument("--base_model", type=str, default="all-MiniLM-L6-v2", 
                        help="Base model to fine-tune (default: all-MiniLM-L6-v2)")
    parser.add_argument("--epochs", type=int, default=4, 
                        help="Number of training epochs (default: 4)")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Training batch size (default: 16)")
    parser.add_argument("--loss", type=str, default="cosine", 
                        choices=["cosine", "contrastive", "online_contrastive", "triplet", "multiple_negatives_ranking"],
                        help="Loss function to use (default: cosine)")
    
    args = parser.parse_args()
    main(args) 