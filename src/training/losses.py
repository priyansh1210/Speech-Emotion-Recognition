"""
Combined loss for multimodal SER.

  L_total = L_emotion + vad_weight * L_vad

L_emotion: CrossEntropyLoss with class weights (handles class imbalance)
L_vad:     MSELoss on Valence/Arousal/Dominance scores
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


class SERLoss(nn.Module):
    def __init__(
        self,
        vad_weight: float = 0.3,
        class_weights: torch.Tensor = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.vad_weight = vad_weight

        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.emotion_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.vad_loss = nn.MSELoss()

    def forward(
        self,
        emotion_logits: torch.Tensor,  # [B, num_classes]
        vad_scores: torch.Tensor,      # [B, 3]
        emotion_targets: torch.Tensor, # [B]
        vad_targets: torch.Tensor,     # [B, 3]
    ) -> dict:
        l_emotion = self.emotion_loss(emotion_logits, emotion_targets)
        l_vad = self.vad_loss(vad_scores, vad_targets)
        l_total = l_emotion + self.vad_weight * l_vad

        return {
            "total": l_total,
            "emotion": l_emotion,
            "vad": l_vad,
        }


def compute_class_weights(csv_path: str, emotion_to_idx: dict) -> torch.Tensor:
    """Compute sklearn balanced class weights from the training CSV.

    Args:
        csv_path: Path to master CSV
        emotion_to_idx: Dict mapping emotion string → int index

    Returns:
        Tensor of shape [num_classes]
    """
    df = pd.read_csv(csv_path)
    labels = df["emotion"].map(emotion_to_idx).dropna().astype(int).values
    classes = np.array(sorted(emotion_to_idx.values()))
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)
