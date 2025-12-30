"""
BERT-based ultrasound findings classification module.

This module provides:
1. UltrasoundBERTClassifier: Multi-label classification model for ultrasound features
2. Inference functions compatible with the regex parser API
3. Batch processing for efficient classification
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import json
import os
import pandas as pd
from typing import Dict, Optional, List


class UltrasoundBERTClassifier(nn.Module):
    """
    Multi-label BERT classifier for ultrasound findings.

    Predicts 6 ultrasound features from radiology text:
    - margin
    - shape
    - orientation
    - echo
    - posterior
    - boundary
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        label_encodings: Dict = None,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        """
        Initialize the BERT classifier.

        Args:
            model_name: Hugging Face model identifier
            label_encodings: Dictionary with label encodings for each feature
            dropout: Dropout probability for regularization
            freeze_bert: If True, freeze BERT weights (only train classification heads)
        """
        super(UltrasoundBERTClassifier, self).__init__()

        # Load pre-trained BERT (use safetensors to avoid torch.load security issues)
        self.bert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.hidden_size = self.bert.config.hidden_size

        # Freeze BERT weights if requested (helps with small datasets)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Create classification heads for each feature
        # Each head predicts one feature with variable number of classes
        self.label_encodings = label_encodings

        if label_encodings is not None:
            self.margin_head = nn.Linear(self.hidden_size, label_encodings['margin']['num_classes'])
            self.shape_head = nn.Linear(self.hidden_size, label_encodings['shape']['num_classes'])
            self.orientation_head = nn.Linear(self.hidden_size, label_encodings['orientation']['num_classes'])
            self.echo_head = nn.Linear(self.hidden_size, label_encodings['echo']['num_classes'])
            self.posterior_head = nn.Linear(self.hidden_size, label_encodings['posterior']['num_classes'])
            self.boundary_head = nn.Linear(self.hidden_size, label_encodings['boundary']['num_classes'])

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs from tokenizer (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Dictionary with logits for each feature
        """
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Apply dropout
        cls_embedding = self.dropout(cls_embedding)

        # Pass through each classification head
        margin_logits = self.margin_head(cls_embedding)
        shape_logits = self.shape_head(cls_embedding)
        orientation_logits = self.orientation_head(cls_embedding)
        echo_logits = self.echo_head(cls_embedding)
        posterior_logits = self.posterior_head(cls_embedding)
        boundary_logits = self.boundary_head(cls_embedding)

        return {
            'margin': margin_logits,
            'shape': shape_logits,
            'orientation': orientation_logits,
            'echo': echo_logits,
            'posterior': posterior_logits,
            'boundary': boundary_logits,
        }

    def predict(self, input_ids, attention_mask):
        """
        Make predictions (convert logits to class indices).

        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask

        Returns:
            Dictionary with predicted class indices for each feature
        """
        # Get logits
        logits = self.forward(input_ids, attention_mask)

        # Convert to predictions (argmax)
        predictions = {}
        for feature, feature_logits in logits.items():
            predictions[feature] = torch.argmax(feature_logits, dim=1)

        return predictions


def load_label_encodings(encodings_path: str) -> Dict:
    """Load label encodings from JSON file."""
    with open(encodings_path, 'r') as f:
        encodings_json = json.load(f)

    # Convert string keys back to appropriate types
    encodings = {}
    for feature, encoding_dict in encodings_json.items():
        encodings[feature] = {
            'label_to_idx': {(None if k == 'None' else k): v for k, v in encoding_dict['label_to_idx'].items()},
            'idx_to_label': {int(k): (None if v is None else v) for k, v in encoding_dict['idx_to_label'].items()},
            'num_classes': encoding_dict['num_classes']
        }

    return encodings


def load_model(model_path: str, encodings_path: str, device: str = 'cuda') -> tuple:
    """
    Load a trained BERT model and its label encodings.

    Args:
        model_path: Path to saved model weights (.pt file)
        encodings_path: Path to label encodings JSON file
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Tuple of (model, tokenizer, label_encodings)
    """
    # Load label encodings
    label_encodings = load_label_encodings(encodings_path)

    # Initialize model
    model = UltrasoundBERTClassifier(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        label_encodings=label_encodings
    )

    # Load trained weights (use weights_only for security)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=True)

    return model, tokenizer, label_encodings


def parse_findings_bert(
    findings_text: str,
    model: UltrasoundBERTClassifier,
    tokenizer,
    label_encodings: Dict,
    device: str = 'cuda',
    max_length: int = 512
) -> Dict:
    """
    Parse ultrasound findings using BERT model.

    Args:
        findings_text: Raw radiology findings text
        model: Trained BERT model
        tokenizer: BERT tokenizer
        label_encodings: Label encodings dictionary
        device: Device ('cuda' or 'cpu')
        max_length: Maximum sequence length

    Returns:
        Dictionary with predicted features (same format as regex parser)
    """
    if pd.isna(findings_text) or not findings_text.strip():
        return {feature: None for feature in ['margin', 'shape', 'orientation', 'echo', 'posterior', 'boundary']}

    # Tokenize input
    inputs = tokenizer(
        findings_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Make prediction
    with torch.no_grad():
        predictions = model.predict(input_ids, attention_mask)

    # Convert predictions to labels
    features = {}
    for feature_name, pred_idx in predictions.items():
        idx = pred_idx.item()
        label = label_encodings[feature_name]['idx_to_label'][idx]
        features[feature_name] = label

    return features


def add_ultrasound_classifications_bert(
    radiology_df: pd.DataFrame,
    model_path: str,
    encodings_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 16,
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Add BERT-based ultrasound classifications to a radiology dataframe.

    This function mirrors the API of add_ultrasound_classifications from findings_parser.py

    Args:
        radiology_df: Dataframe with radiology data
        model_path: Path to trained model weights
        encodings_path: Path to label encodings JSON
        output_path: Optional path to save results (not used, for API compatibility)
        batch_size: Batch size for inference
        device: Device ('cuda' or 'cpu')

    Returns:
        Updated dataframe with classification columns
    """
    print(f"Loading BERT model from {model_path}...")
    model, tokenizer, label_encodings = load_model(model_path, encodings_path, device)

    # Filter for ultrasound modality and RIGHT or LEFT laterality (matching regex parser)
    mask = (radiology_df['MODALITY'] == 'US') & (radiology_df['Study_Laterality'].isin(['RIGHT', 'LEFT']))
    filtered_df = radiology_df[mask].copy()

    print(f"Processing {len(filtered_df)} records with BERT model...")

    # Ensure all feature columns exist
    for feature_type in ['margin', 'shape', 'orientation', 'echo', 'posterior', 'boundary']:
        if feature_type not in radiology_df.columns:
            radiology_df[feature_type] = None

    # Process in batches for efficiency
    features_list = []
    indices = filtered_df.index.tolist()
    findings_texts = filtered_df['FINDINGS'].tolist()

    for i in range(0, len(findings_texts), batch_size):
        batch_texts = findings_texts[i:i+batch_size]
        batch_indices = indices[i:i+batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Make predictions
        with torch.no_grad():
            predictions = model.predict(input_ids, attention_mask)

        # Convert to labels and update dataframe
        for j, idx in enumerate(batch_indices):
            for feature_name, pred_indices in predictions.items():
                pred_idx = pred_indices[j].item()
                label = label_encodings[feature_name]['idx_to_label'][pred_idx]
                radiology_df.loc[idx, feature_name] = label

    print(f"BERT classification complete.")

    return radiology_df
