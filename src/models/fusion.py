"""
Multimodal fusion module.

Concatenates audio and text features, then applies two FC layers
with LayerNorm, ReLU, and Dropout to produce a joint representation.

Input : audio_feat [B, audio_dim], text_feat [B, text_dim]
Output: fused [B, fusion_dim]
"""

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """Late fusion via feature concatenation + FC projection.

    audio_feat [B, 512] + text_feat [B, 768]
    → concat [B, 1280]
    → FC(1280→512) + LN + ReLU + Dropout
    → FC(512→fusion_dim) + ReLU + Dropout
    → fused [B, fusion_dim]
    """

    def __init__(
        self,
        audio_dim: int = 512,
        text_dim: int = 768,
        fusion_dim: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()
        concat_dim = audio_dim + text_dim  # 1280

        self.fc1 = nn.Linear(concat_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, fusion_dim)
        self.ln2 = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, audio_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_feat: [B, audio_dim]
            text_feat:  [B, text_dim]

        Returns:
            fused: [B, fusion_dim]
        """
        x = torch.cat([audio_feat, text_feat], dim=-1)  # [B, 1280]

        x = self.dropout(self.act(self.ln1(self.fc1(x))))  # [B, 512]
        x = self.dropout(self.act(self.ln2(self.fc2(x))))  # [B, fusion_dim]

        return x
