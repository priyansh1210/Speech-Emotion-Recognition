"""
Cross-session evaluation with emotion drift analysis.

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/fold5_best.pt --session_test 5
    python evaluate.py --loso   # Evaluate all 5 folds using saved checkpoints
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.dataset import build_dataloaders, IDX_TO_EMOTION, EMOTION_TO_IDX
from src.models.ser_model import build_model
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.visualization import plot_emotion_drift


def evaluate_fold(
    config: dict,
    checkpoint_path: str,
    test_session: int,
    tokenizer,
    device: torch.device,
    save_drift: bool = True,
) -> dict:
    csv_path = Path(config["data"]["processed_path"]) / "iemocap_6class.csv"

    _, test_loader = build_dataloaders(
        csv_path=str(csv_path),
        test_session=test_session,
        config=config,
        tokenizer=tokenizer,
    )

    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    all_utterance_ids, all_dialogs = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating fold {test_session}"):
            audio = batch["audio"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            emotion_targets = batch["emotion"]

            outputs = model(audio, input_ids, attn_mask)
            preds = outputs["emotion_logits"].argmax(dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(emotion_targets.tolist())
            all_utterance_ids.extend(batch["utterance_id"])
            all_dialogs.extend(batch["dialog"])

    num_classes = len(config["data"]["emotions"])
    metrics = compute_metrics(all_labels, all_preds, num_classes)
    print(f"\n--- Fold {test_session} Results ---")
    print_metrics(metrics)

    # Emotion drift analysis per dialog
    drift_events = compute_emotion_drift(all_dialogs, all_preds, all_utterance_ids)

    if save_drift:
        drift_dir = Path(config["training"]["log_dir"]) / "drift"
        drift_dir.mkdir(parents=True, exist_ok=True)
        for dialog_name, drift_data in list(drift_events.items())[:5]:
            plot_emotion_drift(
                drift_data["utterance_ids"],
                drift_data["emotions"],
                drift_data["transitions"],
                dialog_name,
                save_path=str(drift_dir / f"{dialog_name}_drift.png"),
            )

    return {"fold": test_session, "metrics": metrics, "drift_events": len(drift_events)}


def compute_emotion_drift(
    dialogs: list, preds: list, utterance_ids: list
) -> dict:
    """Group predictions by dialog and detect emotion transitions.

    A drift event is any consecutive pair with a different predicted emotion.

    Returns:
        dict: dialog_name → {utterance_ids, emotions, transitions}
    """
    # Group by dialog preserving order
    dialog_data = defaultdict(lambda: {"utterance_ids": [], "emotions": []})
    for utt_id, dialog, pred in zip(utterance_ids, dialogs, preds):
        dialog_data[dialog]["utterance_ids"].append(utt_id)
        dialog_data[dialog]["emotions"].append(IDX_TO_EMOTION[pred])

    result = {}
    for dialog_name, data in dialog_data.items():
        emotions = data["emotions"]
        transitions = []
        for i in range(1, len(emotions)):
            if emotions[i] != emotions[i - 1]:
                transitions.append(
                    {
                        "from": emotions[i - 1],
                        "to": emotions[i],
                        "at_utterance": data["utterance_ids"][i],
                        "position": i,
                    }
                )
        result[dialog_name] = {
            "utterance_ids": data["utterance_ids"],
            "emotions": emotions,
            "transitions": transitions,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to a single checkpoint")
    parser.add_argument("--session_test", type=int, default=None)
    parser.add_argument("--loso", action="store_true",
                        help="Evaluate all 5 folds using checkpoints/foldX_best.pt")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["roberta_model"])
    ckpt_dir = Path(config["training"]["checkpoint_dir"])

    if args.loso:
        all_results = []
        for s in config["evaluation"]["sessions"]:
            ckpt = str(ckpt_dir / f"fold{s}_best.pt")
            if not Path(ckpt).exists():
                print(f"[SKIP] Checkpoint not found: {ckpt}")
                continue
            r = evaluate_fold(config, ckpt, s, tokenizer, device)
            all_results.append(r)

        if all_results:
            avg_wf1 = sum(r["metrics"]["weighted_f1"] for r in all_results) / len(all_results)
            avg_ua = sum(r["metrics"]["ua"] for r in all_results) / len(all_results)
            print(f"\n{'='*60}")
            print(f" LOSO Average WF1: {avg_wf1:.4f} | UA: {avg_ua:.4f}")
            print(f"{'='*60}")

            log_dir = Path(config["training"]["log_dir"])
            with open(log_dir / "loso_eval.json", "w") as f:
                json.dump({"results": all_results, "avg_wf1": avg_wf1, "avg_ua": avg_ua}, f, indent=2)
    else:
        s = args.session_test or 5
        ckpt = args.checkpoint or str(ckpt_dir / f"fold{s}_best.pt")
        evaluate_fold(config, ckpt, s, tokenizer, device)


if __name__ == "__main__":
    main()
