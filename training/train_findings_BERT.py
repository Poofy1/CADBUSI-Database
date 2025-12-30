"""
Training script for BERT-based ultrasound findings classification.

This script:
1. Loads prepared training data
2. Initializes the BERT model
3. Trains with class-weighted loss
4. Validates and saves best model
5. Logs metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import json
import os
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.ML_processing.findings_BERT import UltrasoundBERTClassifier, load_label_encodings

# Paths
DATA_DIR = os.path.join(SCRIPT_DIR, 'dataset', 'bert_data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'src', 'ML_processing', 'models')
ENCODINGS_PATH = os.path.join(MODELS_DIR, 'bert_encodings.json')

# Training configuration
CONFIG = {
    'model_name': 'emilyalsentzer/Bio_ClinicalBERT',
    'max_length': 512,
    'batch_size': 8,  # Small batch size due to limited data
    'learning_rate': 2e-5,
    'num_epochs': 20,
    'warmup_steps': 50,
    'freeze_bert': True,  # Freeze BERT layers to prevent overfitting on small dataset
    'dropout': 0.3,  # Higher dropout for regularization
    'early_stopping_patience': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
}

# Features to classify
FEATURES = ['margin', 'shape', 'orientation', 'echo', 'posterior', 'boundary']


class UltrasoundFindingsDataset(Dataset):
    """PyTorch Dataset for ultrasound findings."""

    def __init__(self, csv_path, tokenizer, label_encodings, max_length=512):
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV file with data
            tokenizer: BERT tokenizer
            label_encodings: Label encodings dictionary
            max_length: Maximum sequence length
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.label_encodings = label_encodings
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get a single example."""
        row = self.df.iloc[idx]

        # Get findings text
        findings_text = row['FINDINGS']
        if pd.isna(findings_text):
            findings_text = ""

        # Tokenize
        encoding = self.tokenizer(
            findings_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get labels for each feature
        labels = {}
        for feature in FEATURES:
            label_value = row[feature]

            # Convert to index using label_to_idx mapping
            if pd.isna(label_value):
                label_idx = 0  # 0 = None/not mentioned
            else:
                label_idx = self.label_encodings[feature]['label_to_idx'].get(label_value, 0)

            labels[feature] = label_idx

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }


def calculate_class_weights(df, label_encodings):
    """
    Calculate class weights for handling class imbalance.

    Uses inverse frequency weighting.
    """
    class_weights = {}

    for feature in FEATURES:
        num_classes = label_encodings[feature]['num_classes']

        # Count occurrences of each class
        counts = np.zeros(num_classes)

        for idx, row in df.iterrows():
            label_value = row[feature]

            if pd.isna(label_value):
                label_idx = 0
            else:
                label_idx = label_encodings[feature]['label_to_idx'].get(label_value, 0)

            counts[label_idx] += 1

        # Calculate weights (inverse frequency, with smoothing)
        # Add 1 to avoid division by zero
        weights = 1.0 / (counts + 1.0)

        # Normalize weights
        weights = weights / weights.sum() * num_classes

        class_weights[feature] = torch.FloatTensor(weights)

    return class_weights


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = {feature: [] for feature in FEATURES}
    all_labels = {feature: [] for feature in FEATURES}

    # Define loss functions with class weights for each feature
    criteria = {}
    for feature in FEATURES:
        criteria[feature] = nn.CrossEntropyLoss(weight=class_weights[feature].to(device))

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = {feature: batch['labels'][feature].to(device) for feature in FEATURES}

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Calculate loss for each feature and sum
        loss = 0
        for feature in FEATURES:
            feature_loss = criteria[feature](outputs[feature], labels[feature])
            loss += feature_loss

        # Backward pass
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()
        scheduler.step()

        # Track loss
        total_loss += loss.item()

        # Track predictions for metrics
        for feature in FEATURES:
            preds = torch.argmax(outputs[feature], dim=1)
            all_predictions[feature].extend(preds.cpu().numpy())
            all_labels[feature].extend(labels[feature].cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracies = {}
    for feature in FEATURES:
        accuracies[feature] = accuracy_score(all_labels[feature], all_predictions[feature])

    return avg_loss, accuracies


def validate(model, dataloader, device, class_weights):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_predictions = {feature: [] for feature in FEATURES}
    all_labels = {feature: [] for feature in FEATURES}

    # Define loss functions
    criteria = {}
    for feature in FEATURES:
        criteria[feature] = nn.CrossEntropyLoss(weight=class_weights[feature].to(device))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = {feature: batch['labels'][feature].to(device) for feature in FEATURES}

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Calculate loss
            loss = 0
            for feature in FEATURES:
                feature_loss = criteria[feature](outputs[feature], labels[feature])
                loss += feature_loss

            total_loss += loss.item()

            # Track predictions
            for feature in FEATURES:
                preds = torch.argmax(outputs[feature], dim=1)
                all_predictions[feature].extend(preds.cpu().numpy())
                all_labels[feature].extend(labels[feature].cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracies = {}
    f1_scores = {}

    for feature in FEATURES:
        accuracies[feature] = accuracy_score(all_labels[feature], all_predictions[feature])
        f1_scores[feature] = f1_score(all_labels[feature], all_predictions[feature], average='macro', zero_division=0)

    return avg_loss, accuracies, f1_scores


def print_metrics(split_name, loss, accuracies, f1_scores=None):
    """Print training/validation metrics."""
    print(f"\n{split_name} Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Per-feature accuracy:")
    for feature in FEATURES:
        acc = accuracies[feature]
        if f1_scores:
            f1 = f1_scores[feature]
            print(f"    {feature:12s}: {acc:.4f} (F1: {f1:.4f})")
        else:
            print(f"    {feature:12s}: {acc:.4f}")

    # Overall accuracy (average across features)
    overall_acc = np.mean(list(accuracies.values()))
    print(f"  Overall avg accuracy: {overall_acc:.4f}")


def main():
    """Main training function."""
    print("="*80)
    print("BERT MODEL TRAINING FOR ULTRASOUND FINDINGS")
    print("="*80)

    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    # Print configuration
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # Load label encodings
    print(f"\nLoading label encodings from {ENCODINGS_PATH}...")
    label_encodings = load_label_encodings(ENCODINGS_PATH)

    # Load tokenizer
    print(f"Loading tokenizer: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = UltrasoundFindingsDataset(
        os.path.join(DATA_DIR, 'train.csv'),
        tokenizer,
        label_encodings,
        CONFIG['max_length']
    )

    val_dataset = UltrasoundFindingsDataset(
        os.path.join(DATA_DIR, 'val.csv'),
        tokenizer,
        label_encodings,
        CONFIG['max_length']
    )

    print(f"  Train size: {len(train_dataset)}")
    print(f"  Val size:   {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0  # Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Calculate class weights
    print("\nCalculating class weights for imbalanced data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    class_weights = calculate_class_weights(train_df, label_encodings)

    # Initialize model
    print(f"\nInitializing model: {CONFIG['model_name']}...")
    model = UltrasoundBERTClassifier(
        model_name=CONFIG['model_name'],
        label_encodings=label_encodings,
        dropout=CONFIG['dropout'],
        freeze_bert=CONFIG['freeze_bert']
    )
    model.to(CONFIG['device'])

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])

    total_steps = len(train_loader) * CONFIG['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )

    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        print("-" * 80)

        # Train
        train_loss, train_accuracies = train_epoch(
            model, train_loader, optimizer, scheduler, CONFIG['device'], class_weights
        )

        # Validate
        val_loss, val_accuracies, val_f1_scores = validate(
            model, val_loader, CONFIG['device'], class_weights
        )

        # Print metrics
        print_metrics("Train", train_loss, train_accuracies)
        print_metrics("Validation", val_loss, val_accuracies, val_f1_scores)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            model_path = os.path.join(MODELS_DIR, 'findings_bert_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracies': val_accuracies,
                'val_f1_scores': val_f1_scores,
                'config': CONFIG
            }, model_path)

            print(f"\n[OK] Best model saved to {model_path}")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    print("\n" + "="*80)
    print("[OK] TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(MODELS_DIR, 'findings_bert_best.pt')}")
    print("\nNext steps:")
    print("  1. Test the model on the test set")
    print("  2. Run evaluate_bert_vs_regex.py to compare with regex parser")


if __name__ == "__main__":
    main()
