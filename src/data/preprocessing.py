"""
Audio preprocessing utilities.

Converts raw .wav files to normalized Mel-spectrograms as PyTorch tensors.
"""

import numpy as np
import torch
import librosa


def load_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load a wav file and resample to target sample_rate."""
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    return waveform


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    fmax: int = 8000,
) -> np.ndarray:
    """Compute log Mel-spectrogram from a waveform array.

    Returns:
        np.ndarray of shape [n_mels, T]
    """
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel  # [n_mels, T]


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Per-sample mean/std normalization."""
    mean = spec.mean()
    std = spec.std() + 1e-8
    return (spec - mean) / std


def pad_or_truncate(spec: np.ndarray, max_frames: int) -> np.ndarray:
    """Pad (with zeros) or truncate spectrogram along the time axis.

    Args:
        spec: [n_mels, T]
        max_frames: target time dimension

    Returns:
        [n_mels, max_frames]
    """
    n_mels, T = spec.shape
    if T >= max_frames:
        return spec[:, :max_frames]
    pad_width = max_frames - T
    return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


def process_audio(
    audio_path: str,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    fmax: int = 8000,
    max_audio_seconds: float = 8.0,
) -> torch.Tensor:
    """Full pipeline: wav → normalized Mel-spectrogram tensor.

    Returns:
        torch.Tensor of shape [1, n_mels, max_frames]
    """
    max_frames = int(max_audio_seconds * sample_rate / hop_length)

    waveform = load_audio(audio_path, sample_rate)
    spec = compute_mel_spectrogram(waveform, sample_rate, n_mels, n_fft, hop_length, fmax)
    spec = normalize_spectrogram(spec)
    spec = pad_or_truncate(spec, max_frames)

    tensor = torch.from_numpy(spec).float().unsqueeze(0)  # [1, n_mels, max_frames]
    return tensor


def get_max_frames(max_audio_seconds: float, sample_rate: int, hop_length: int) -> int:
    """Helper to compute max_frames from config values."""
    return int(max_audio_seconds * sample_rate / hop_length)
