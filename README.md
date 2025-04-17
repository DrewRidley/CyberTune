# Fine-Tuning Sentence Transformers for Cybersecurity TTPs

This project demonstrates how to fine-tune sentence transformer models to identify and correlate cybersecurity Tactics, Techniques, and Procedures (TTPs) from text.

## Overview

Sentence Transformers are powerful models that convert text into meaningful embeddings, allowing for semantic similarity comparisons. This project fine-tunes these models specifically for the cybersecurity domain, focusing on:

- Identifying text describing TTPs in cybersecurity reports
- Correlating similar TTPs across different reports
- Creating embeddings that cluster related attack patterns

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/finetune-st.git
cd finetune-st

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The project includes several scripts:

1. `create_dataset.py` - Generates training data by processing cybersecurity reports
2. `train_model.py` - Fine-tunes a sentence transformer model
3. `evaluate_model.py` - Tests the model's performance
4. `inference.py` - Uses the fine-tuned model to analyze new texts

Run them in sequence:

```bash
python create_dataset.py
python train_model.py
python evaluate_model.py
python inference.py
```

## Data

The project uses a combination of:
- Public cybersecurity reports for training data
- MITRE ATT&CK framework for TTP categorization
- Synthetic examples for data augmentation

## How It Works

1. Text samples are collected and labeled according to their TTP categories
2. Pairs or triplets are created for contrastive learning
3. A pre-trained sentence transformer is fine-tuned using these pairs
4. The model learns to map similar TTPs closer in the embedding space

## License

This project is released under the MIT License. 