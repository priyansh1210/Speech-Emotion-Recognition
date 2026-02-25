"""
Audio branch: CNN + BiLSTM + Temporal Attention.

Input : [B, 1, n_mels, T]
Output: (audio_feat [B, lstm_hidden*2], attn_weights [B, T'])

The attention weights over time frames enable XAI:
high-weight frames = emotionally salient audio segments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool_size: tuple):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TemporalAttention(nn.Module):
    """Additive attention over a sequence of frame features.

    Input : [B, T, H]
    Output: (context [B, H], weights [B, T])
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        # x: [B, T, H]
        scores = self.attn(x).squeeze(-1)          # [B, T]
        weights = F.softmax(scores, dim=-1)         # [B, T]
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, H]
        return context, weights


class AudioModel(nn.Module):
    """CNN + BiLSTM + Temporal Attention for Mel-spectrogram input.

    Architecture:
      CNN extracts local spectro-temporal patterns.
      BiLSTM captures sequential / temporal dynamics.
      Temporal Attention focuses on emotionally salient frames.
    """

    def __init__(
        self,
        n_mels: int = 128,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.dropout = dropout

        # CNN: [B,1,128,T] → [B,128,8,T/16]
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, pool_size=(2, 2)),    # → [B,32,64,T/2]
            ConvBlock(32, 64, pool_size=(2, 2)),   # → [B,64,32,T/4]
            ConvBlock(64, 128, pool_size=(4, 4)),  # → [B,128,8,T/16]
        )

        # After CNN, pool mel dimension → sequence
        # [B,128,8,T'] → mean over mel → [B,T',128]
        self.mel_pool = nn.AdaptiveAvgPool2d((1, None))  # → [B,128,1,T']

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2  # bidirectional
        self.attention = TemporalAttention(lstm_out_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.out_dim = lstm_out_dim  # 512

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [B, 1, n_mels, T]

        Returns:
            audio_feat: [B, lstm_hidden*2]
            attn_weights: [B, T']  — frame-level attention (XAI)
        """
        # CNN feature extraction
        cnn_out = self.cnn(x)                          # [B, 128, 8, T']
        pooled = self.mel_pool(cnn_out)                # [B, 128, 1, T']
        pooled = pooled.squeeze(2)                     # [B, 128, T']
        seq = pooled.permute(0, 2, 1)                  # [B, T', 128]

        # BiLSTM temporal modeling
        lstm_out, _ = self.bilstm(seq)                 # [B, T', 512]
        lstm_out = self.dropout_layer(lstm_out)

        # Temporal attention → context + weights
        audio_feat, attn_weights = self.attention(lstm_out)  # [B,512], [B,T']

        return audio_feat, attn_weights
