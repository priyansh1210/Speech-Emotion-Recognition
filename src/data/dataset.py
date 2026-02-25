"""
PyTorch Dataset for IEMOCAP multimodal SER.

Each sample returns:
  audio     : Tensor [1, n_mels, max_frames]
  input_ids : Tensor [max_text_len]
  attn_mask : Tensor [max_text_len]
  emotion   : int (class index)
  vad       : Tensor [3]  (valence, arousal, dominance)
  utterance_id : str
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.data.preprocessing import process_audio, get_max_frames


EMOTION_TO_IDX = {
    "ang": 0,
    "hap": 1,
    "neu": 2,
    "sad": 3,
    "fru": 4,
    "exc": 5,
}

IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}


class IEMOCAPDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        session_ids: list,
        tokenizer_name: str = "roberta-base",
        max_text_len: int = 128,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        fmax: int = 8000,
        max_audio_seconds: float = 8.0,
        tokenizer=None,
    ):
        """
        Args:
            csv_path: Path to iemocap_6class.csv
            session_ids: List of session numbers to include (e.g. [1,2,3,4])
            tokenizer_name: HuggingFace tokenizer identifier
            max_text_len: Max token length for text
            tokenizer: Pre-loaded tokenizer (optional, avoids reloading per worker)
        """
        df = pd.read_csv(csv_path)
        self.df = df[df["session"].isin(session_ids)].reset_index(drop=True)

        self.max_text_len = max_text_len
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmax = fmax
        self.max_audio_seconds = max_audio_seconds
        self.max_frames = get_max_frames(max_audio_seconds, sample_rate, hop_length)

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Audio
        audio = process_audio(
            audio_path=row["audio_path"],
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmax=self.fmax,
            max_audio_seconds=self.max_audio_seconds,
        )

        # Text
        encoding = self.tokenizer(
            str(row["text"]),
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)       # [L]
        attn_mask = encoding["attention_mask"].squeeze(0)  # [L]

        # Labels
        emotion_idx = EMOTION_TO_IDX[row["emotion"]]
        vad = torch.tensor(
            [row["valence"], row["arousal"], row["dominance"]], dtype=torch.float32
        )

        return {
            "audio": audio,
            "input_ids": input_ids,
            "attn_mask": attn_mask,
            "emotion": emotion_idx,
            "vad": vad,
            "utterance_id": row["utterance_id"],
            "dialog": row["dialog"],
            "session": int(row["session"]),
        }


def build_dataloaders(
    csv_path: str,
    test_session: int,
    config: dict,
    tokenizer=None,
):
    """Build train and test DataLoaders using LOSO split.

    Args:
        csv_path: Path to master CSV
        test_session: Session number to hold out for testing (1-5)
        config: Full config dict
        tokenizer: Pre-loaded tokenizer

    Returns:
        (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    train_sessions = [s for s in range(1, 6) if s != test_session]
    test_sessions = [test_session]

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    shared_kwargs = dict(
        csv_path=csv_path,
        tokenizer_name=model_cfg["roberta_model"],
        max_text_len=model_cfg["max_text_len"],
        sample_rate=data_cfg["sample_rate"],
        n_mels=data_cfg["n_mels"],
        n_fft=data_cfg["n_fft"],
        hop_length=data_cfg["hop_length"],
        fmax=data_cfg["fmax"],
        max_audio_seconds=data_cfg["max_audio_seconds"],
        tokenizer=tokenizer,
    )

    train_ds = IEMOCAPDataset(session_ids=train_sessions, **shared_kwargs)
    test_ds = IEMOCAPDataset(session_ids=test_sessions, **shared_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, test_loader
