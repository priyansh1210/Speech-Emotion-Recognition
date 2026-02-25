"""
Text branch: RoBERTa + Token-level Attention.

Input : input_ids [B, L], attention_mask [B, L]
Output: (text_feat [B, 768], token_weights [B, L])

The token-level attention weights enable XAI:
high-weight tokens = emotionally salient words.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TokenAttention(nn.Module):
    """Attention over RoBERTa token embeddings.

    Masked softmax respects padding tokens (attention_mask).

    Input : hidden_states [B, L, H], attention_mask [B, L]
    Output: (context [B, H], weights [B, L])
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple:
        scores = self.attn(hidden_states).squeeze(-1)  # [B, L]

        # Mask padding positions with -inf before softmax
        mask = (attention_mask == 0)
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)             # [B, L]
        # Handle edge case: all masked (shouldn't happen but guard)
        weights = torch.nan_to_num(weights, nan=0.0)

        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # [B, H]
        return context, weights


class TextModel(nn.Module):
    """RoBERTa encoder with token-level attention for SER.

    Uses the last hidden state of RoBERTa and applies a learned
    attention layer to produce a weighted text representation.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.4,
        freeze_layers: int = 6,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            dropout: Dropout rate after attention
            freeze_layers: Number of bottom RoBERTa layers to freeze (0 = train all)
        """
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size  # 768 for roberta-base

        # Optionally freeze bottom layers to save compute
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        self.attention = TokenAttention(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)

        self.out_dim = hidden_size  # 768

    def _freeze_layers(self, n: int) -> None:
        """Freeze the first n transformer layers of RoBERTa."""
        # Always train embeddings? No — freeze for efficiency
        modules_to_freeze = (
            [self.roberta.embeddings]
            + list(self.roberta.encoder.layer[:n])
        )
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> tuple:
        """
        Args:
            input_ids: [B, L]
            attn_mask:  [B, L]

        Returns:
            text_feat:    [B, 768]
            token_weights: [B, L]  — token-level attention (XAI)
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attn_mask,
        )
        hidden_states = outputs.last_hidden_state  # [B, L, 768]
        hidden_states = self.dropout_layer(hidden_states)

        text_feat, token_weights = self.attention(hidden_states, attn_mask)

        return text_feat, token_weights
