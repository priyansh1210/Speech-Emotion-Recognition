"""
Full Multimodal SER model.

Combines AudioModel + TextModel + FusionModule into a single end-to-end
network with three output heads:
  1. emotion_logits  [B, num_classes]   — 6-class emotion classification
  2. vad_scores      [B, 3]             — Valence/Arousal/Dominance regression
  3. audio_attn      [B, T']            — Frame-level attention (XAI)
  4. token_attn      [B, L]             — Token-level attention (XAI)
"""

import torch
import torch.nn as nn

from src.models.audio_model import AudioModel
from src.models.text_model import TextModel
from src.models.fusion import FusionModule


class SERModel(nn.Module):
    """Multimodal Speech Emotion Recognition model.

    Novelty: Returns audio and text attention weights alongside predictions
    for Attention-Based Emotion Localization (XAI).
    """

    def __init__(
        self,
        num_classes: int = 6,
        roberta_model: str = "roberta-base",
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        fusion_dim: int = 256,
        dropout: float = 0.4,
        freeze_roberta_layers: int = 6,
    ):
        super().__init__()

        self.audio_model = AudioModel(
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            dropout=dropout,
        )
        self.text_model = TextModel(
            model_name=roberta_model,
            dropout=dropout,
            freeze_layers=freeze_roberta_layers,
        )
        self.fusion = FusionModule(
            audio_dim=self.audio_model.out_dim,   # 512
            text_dim=self.text_model.out_dim,     # 768
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

        # Emotion classification head
        self.emotion_head = nn.Linear(fusion_dim, num_classes)

        # VAD regression head (output scaled to [1,5] via sigmoid)
        self.vad_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

        self._init_heads()

    def _init_heads(self) -> None:
        nn.init.xavier_uniform_(self.emotion_head.weight)
        nn.init.zeros_(self.emotion_head.bias)

    def forward(
        self,
        audio: torch.Tensor,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> dict:
        """
        Args:
            audio:      [B, 1, n_mels, T]
            input_ids:  [B, L]
            attn_mask:  [B, L]

        Returns dict with:
            emotion_logits : [B, num_classes]
            vad_scores     : [B, 3]   values in [0,1], scale to [1,5] externally
            audio_attn     : [B, T']  frame-level attention weights (XAI)
            token_attn     : [B, L]   token-level attention weights (XAI)
            fused          : [B, fusion_dim]  joint representation
        """
        audio_feat, audio_attn = self.audio_model(audio)
        text_feat, token_attn = self.text_model(input_ids, attn_mask)
        fused = self.fusion(audio_feat, text_feat)

        emotion_logits = self.emotion_head(fused)
        vad_scores = self.vad_head(fused) * 4.0 + 1.0  # scale sigmoid→[1,5]

        return {
            "emotion_logits": emotion_logits,
            "vad_scores": vad_scores,
            "audio_attn": audio_attn,
            "token_attn": token_attn,
            "fused": fused,
        }


def build_model(config: dict) -> SERModel:
    """Instantiate SERModel from config dict."""
    m = config["model"]
    return SERModel(
        num_classes=len(config["data"]["emotions"]),
        roberta_model=m["roberta_model"],
        lstm_hidden=m["lstm_hidden"],
        lstm_layers=m["lstm_layers"],
        lstm_dropout=m["lstm_dropout"],
        fusion_dim=m["fusion_dim"],
        dropout=m["dropout"],
    )
