"""
Attention-Based Emotion Localization — XAI Visualization Utilities.

Provides:
  1. plot_audio_attention()   — Mel-spectrogram with temporal attention heatmap overlay
  2. plot_text_attention()    — Word-level attention color-coded HTML / matplotlib
  3. plot_emotion_drift()     — Per-dialog emotion sequence with transition markers
  4. render_text_attention_html() — Returns HTML string for notebook display
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import librosa
import librosa.display
from IPython.display import HTML


EMOTION_COLORS = {
    "ang": "#e74c3c",
    "hap": "#f1c40f",
    "neu": "#95a5a6",
    "sad": "#3498db",
    "fru": "#e67e22",
    "exc": "#2ecc71",
}


# ─────────────────────────────────────────────
# 1. Audio Attention Heatmap
# ─────────────────────────────────────────────

def plot_audio_attention(
    audio_path: str,
    attn_weights: np.ndarray,
    predicted_emotion: str,
    true_emotion: str = None,
    sample_rate: int = 16000,
    hop_length: int = 256,
    n_mels: int = 128,
    n_fft: int = 1024,
    fmax: int = 8000,
    save_path: str = None,
) -> plt.Figure:
    """Plot Mel-spectrogram with temporal attention heatmap overlay.

    The attention weights are upsampled to match the spectrogram time axis
    and overlaid as a semi-transparent heatmap, highlighting emotionally
    salient time regions.

    Args:
        audio_path: Path to .wav file
        attn_weights: 1D array of shape [T'] — frame attention from the model
        predicted_emotion: Predicted emotion label string
        true_emotion: Ground-truth emotion label (optional)
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    waveform, sr = librosa.load(audio_path, sr=sample_rate)
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # [n_mels, T_full]
    T_full = log_mel.shape[1]

    # Upsample attention weights to match full spectrogram time axis
    attn_up = np.interp(
        np.linspace(0, len(attn_weights) - 1, T_full),
        np.arange(len(attn_weights)),
        attn_weights,
    )  # [T_full]

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)

    # ── Top: Mel-spectrogram + attention overlay
    ax_mel = fig.add_subplot(gs[0])
    librosa.display.specshow(
        log_mel, sr=sr, hop_length=hop_length, fmax=fmax,
        x_axis="time", y_axis="mel", ax=ax_mel, cmap="magma",
    )

    # Attention overlay: normalize to [0,1], apply colormap, blend
    attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)
    attn_2d = np.tile(attn_norm, (n_mels, 1))  # broadcast over mel bins
    attn_rgba = plt.cm.hot(attn_norm)           # [T_full, 4]
    attn_rgba_2d = np.tile(attn_rgba, (n_mels, 1, 1))  # [n_mels, T_full, 4]
    attn_rgba_2d[..., 3] = attn_2d * 0.55      # alpha = attention weight × 0.55

    times = librosa.times_like(log_mel, sr=sr, hop_length=hop_length)
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=fmax)
    ax_mel.imshow(
        attn_rgba_2d,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        interpolation="bilinear",
    )

    title = f"Predicted: {predicted_emotion.upper()}"
    if true_emotion:
        title += f"  |  True: {true_emotion.upper()}"
    ax_mel.set_title(title, fontsize=13, fontweight="bold")
    ax_mel.set_xlabel("")

    # ── Bottom: Attention weight bar
    ax_attn = fig.add_subplot(gs[1])
    ax_attn.fill_between(times, attn_norm, alpha=0.8, color="#e74c3c")
    ax_attn.set_xlim(times[0], times[-1])
    ax_attn.set_ylim(0, 1)
    ax_attn.set_xlabel("Time (s)")
    ax_attn.set_ylabel("Attn", fontsize=8)
    ax_attn.set_yticks([])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 2. Text Attention Heatmap
# ─────────────────────────────────────────────

def render_text_attention_html(
    tokens: list,
    attn_weights: np.ndarray,
    predicted_emotion: str,
    colormap: str = "Reds",
) -> str:
    """Generate an HTML string coloring each word by its attention weight.

    Args:
        tokens: List of token strings (from tokenizer.convert_ids_to_tokens)
        attn_weights: 1D array matching len(tokens)
        predicted_emotion: Emotion label for title
        colormap: Matplotlib colormap name (default: Reds)

    Returns:
        HTML string ready for IPython.display.HTML()
    """
    cmap = plt.cm.get_cmap(colormap)
    weights = np.array(attn_weights)
    w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    html_parts = [
        f"<div style='font-family:monospace; font-size:15px; line-height:2.2; padding:12px;'>",
        f"<b>Emotion: {predicted_emotion.upper()}</b><br><br>",
    ]

    for token, w in zip(tokens, w_norm):
        # Skip special tokens
        if token in ("<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"):
            continue
        # RoBERTa Ġ prefix = space, clean for display
        display_token = token.replace("Ġ", " ").replace("▁", " ")

        rgba = cmap(float(w))
        r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
        text_color = "#000" if w < 0.6 else "#fff"
        html_parts.append(
            f"<span style='background-color:rgb({r},{g},{b}); "
            f"color:{text_color}; padding:2px 4px; margin:1px; "
            f"border-radius:3px; title=\"attn={w:.3f}\"'>"
            f"{display_token}</span>"
        )

    html_parts.append("</div>")
    return "".join(html_parts)


def plot_text_attention(
    tokens: list,
    attn_weights: np.ndarray,
    predicted_emotion: str,
    save_path: str = None,
) -> plt.Figure:
    """Matplotlib bar chart of token attention weights."""
    weights = np.array(attn_weights)

    # Filter special tokens for display
    clean = [
        (t.replace("Ġ", "").replace("▁", ""), w)
        for t, w in zip(tokens, weights)
        if t not in ("<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]")
    ]
    if not clean:
        return None
    toks, wts = zip(*clean)

    fig, ax = plt.subplots(figsize=(max(8, len(toks) * 0.5), 4))
    colors = plt.cm.Reds(np.array(wts) / (max(wts) + 1e-8))
    bars = ax.bar(range(len(toks)), wts, color=colors, edgecolor="white")
    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Attention Weight")
    ax.set_title(f"Token Attention — Predicted: {predicted_emotion.upper()}", fontweight="bold")
    ax.set_ylim(0, max(wts) * 1.15)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 3. Emotion Drift Plot
# ─────────────────────────────────────────────

def plot_emotion_drift(
    utterance_ids: list,
    emotions: list,
    transitions: list,
    dialog_name: str,
    save_path: str = None,
) -> plt.Figure:
    """Plot emotion sequence across utterances in a dialog.

    Marks transition points (emotion drift events) with vertical lines.

    Args:
        utterance_ids: List of utterance ID strings
        emotions: List of predicted emotion strings
        transitions: List of dicts with keys: from, to, position
        dialog_name: Dialog identifier for title
        save_path: Save path for figure

    Returns:
        matplotlib Figure
    """
    emotion_list = ["ang", "hap", "neu", "sad", "fru", "exc"]
    emotion_to_y = {e: i for i, e in enumerate(emotion_list)}

    y_vals = [emotion_to_y.get(e, -1) for e in emotions]
    x_vals = list(range(len(emotions)))
    colors = [EMOTION_COLORS.get(e, "#888") for e in emotions]

    fig, ax = plt.subplots(figsize=(max(10, len(emotions) * 0.6), 5))

    # Step line
    ax.step(x_vals, y_vals, where="post", color="#555", linewidth=1.5, zorder=1)

    # Color-coded scatter points
    for xi, yi, ci in zip(x_vals, y_vals, colors):
        ax.scatter(xi, yi, color=ci, s=80, zorder=2)

    # Transition markers
    for t in transitions:
        pos = t["position"]
        ax.axvline(pos, color="red", linestyle="--", alpha=0.5, linewidth=1.2)
        ax.text(
            pos + 0.1, 5.7,
            f"{t['from']}→{t['to']}",
            fontsize=7, color="red", rotation=90, va="top",
        )

    ax.set_yticks(list(emotion_to_y.values()))
    ax.set_yticklabels(list(emotion_to_y.keys()))
    ax.set_xticks(x_vals[::max(1, len(x_vals)//10)])
    short_ids = [uid.split("_")[-1] for uid in utterance_ids]
    ax.set_xticklabels(
        [short_ids[i] for i in x_vals[::max(1, len(x_vals)//10)]],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_xlabel("Utterance")
    ax.set_ylabel("Emotion")
    ax.set_title(
        f"Emotion Drift — {dialog_name}  ({len(transitions)} transitions)",
        fontweight="bold",
    )
    ax.set_ylim(-0.5, len(emotion_list) - 0.3)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
