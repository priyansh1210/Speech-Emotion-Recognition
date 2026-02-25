"""
Parse IEMOCAP dataset into a master CSV.

Reads EmoEvaluation files (emotion labels + VAD scores) and transcription
files, then resolves audio paths to produce a flat CSV ready for training.
"""

import os
import re
import csv
import yaml
import argparse
from pathlib import Path


VALID_EMOTIONS = {"ang", "hap", "neu", "sad", "fru", "exc"}

# Regex patterns
LABEL_RE = re.compile(
    r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\w+)\s+\[(\S+),\s*(\S+),\s*(\S+)\]"
)
TRANSCRIPT_RE = re.compile(r"(\S+)\s+\[\d+\.\d+-\d+\.\d+\]:\s+(.*)")


def parse_emotion_file(filepath: Path) -> dict:
    """Parse an EmoEvaluation .txt file.

    Returns:
        dict mapping utterance_id -> {emotion, valence, arousal, dominance,
                                       start_time, end_time}
    """
    records = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            m = LABEL_RE.match(line)
            if m:
                start_time = float(m.group(1))
                end_time = float(m.group(2))
                utt_id = m.group(3)
                emotion = m.group(4).lower()
                valence = float(m.group(5))
                arousal = float(m.group(6))
                dominance = float(m.group(7))

                if emotion not in VALID_EMOTIONS:
                    continue

                records[utt_id] = {
                    "emotion": emotion,
                    "valence": valence,
                    "arousal": arousal,
                    "dominance": dominance,
                    "start_time": start_time,
                    "end_time": end_time,
                }
    return records


def parse_transcription_file(filepath: Path) -> dict:
    """Parse a transcription .txt file.

    Returns:
        dict mapping utterance_id -> text
    """
    records = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = TRANSCRIPT_RE.match(line)
            if m:
                utt_id = m.group(1)
                text = m.group(2).strip()
                records[utt_id] = text
    return records


def resolve_audio_path(dataset_path: Path, session: int, utt_id: str) -> str:
    """Resolve the .wav file path for a given utterance ID."""
    # utt_id e.g. Ses01F_impro01_F000
    # dialog name is everything up to the last _X000 part
    parts = utt_id.rsplit("_", 1)
    dialog_name = parts[0]  # e.g. Ses01F_impro01
    wav_path = (
        dataset_path
        / f"Session{session}"
        / "sentences"
        / "wav"
        / dialog_name
        / f"{utt_id}.wav"
    )
    return str(wav_path)


def build_dataset(config: dict, output_csv: Path) -> None:
    """Parse all sessions and write the master CSV."""
    dataset_path = Path(config["data"]["dataset_path"])
    emotions = set(config["data"]["emotions"])

    rows = []
    skipped_no_text = 0
    skipped_no_audio = 0
    skipped_emotion = 0

    for session in range(1, 6):
        session_path = dataset_path / f"Session{session}"
        emo_dir = session_path / "dialog" / "EmoEvaluation"
        trans_dir = session_path / "dialog" / "transcriptions"

        if not emo_dir.exists():
            print(f"[WARN] Session {session} EmoEvaluation dir not found: {emo_dir}")
            continue

        for emo_file in sorted(emo_dir.glob("*.txt")):
            dialog_name = emo_file.stem  # e.g. Ses01F_impro01
            trans_file = trans_dir / f"{dialog_name}.txt"

            emo_records = parse_emotion_file(emo_file)
            trans_records = (
                parse_transcription_file(trans_file) if trans_file.exists() else {}
            )

            for utt_id, emo_data in emo_records.items():
                if emo_data["emotion"] not in emotions:
                    skipped_emotion += 1
                    continue

                text = trans_records.get(utt_id, "")
                if not text:
                    skipped_no_text += 1
                    continue

                audio_path = resolve_audio_path(dataset_path, session, utt_id)
                if not Path(audio_path).exists():
                    skipped_no_audio += 1
                    continue

                rows.append(
                    {
                        "utterance_id": utt_id,
                        "session": session,
                        "dialog": dialog_name,
                        "audio_path": audio_path,
                        "text": text,
                        "emotion": emo_data["emotion"],
                        "valence": emo_data["valence"],
                        "arousal": emo_data["arousal"],
                        "dominance": emo_data["dominance"],
                        "start_time": emo_data["start_time"],
                        "end_time": emo_data["end_time"],
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "utterance_id", "session", "dialog", "audio_path", "text",
        "emotion", "valence", "arousal", "dominance", "start_time", "end_time",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDataset built: {len(rows)} utterances → {output_csv}")
    print(f"  Skipped (bad emotion): {skipped_emotion}")
    print(f"  Skipped (no text):     {skipped_no_text}")
    print(f"  Skipped (no audio):    {skipped_no_audio}")

    # Print class distribution
    from collections import Counter
    dist = Counter(r["emotion"] for r in rows)
    print("\nClass distribution:")
    for emo, count in sorted(dist.items()):
        print(f"  {emo}: {count} ({100*count/len(rows):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build IEMOCAP master CSV")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output = Path(args.output) if args.output else Path(config["data"]["processed_path"]) / "iemocap_6class.csv"
    build_dataset(config, output)
