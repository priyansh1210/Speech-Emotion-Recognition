"""
Main training entry point.

Usage:
    python train.py --config configs/config.yaml --session_test 5
    python train.py --loso   # Run all 5 folds (Leave-One-Session-Out)
"""

import argparse
import json
import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer

from src.data.dataset import build_dataloaders, EMOTION_TO_IDX
from src.models.ser_model import build_model
from src.training.losses import SERLoss, compute_class_weights
from src.training.trainer import Trainer, build_optimizer, build_scheduler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fold(config: dict, test_session: int, tokenizer, device: torch.device) -> dict:
    csv_path = Path(config["data"]["processed_path"]) / "iemocap_6class.csv"

    print(f"\nBuilding dataloaders (test session: {test_session})...")
    train_loader, test_loader = build_dataloaders(
        csv_path=str(csv_path),
        test_session=test_session,
        config=config,
        tokenizer=tokenizer,
    )
    print(f"  Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # Class weights from training split only
    train_sessions = [s for s in range(1, 6) if s != test_session]
    import pandas as pd
    df = pd.read_csv(csv_path)
    train_df = df[df["session"].isin(train_sessions)]
    tmp_csv = Path(config["data"]["processed_path"]) / f"_train_fold{test_session}.csv"
    train_df.to_csv(tmp_csv, index=False)
    class_weights = compute_class_weights(str(tmp_csv), EMOTION_TO_IDX).to(device)
    os.remove(tmp_csv)

    model = build_model(config).to(device)
    criterion = SERLoss(
        vad_weight=config["training"]["vad_loss_weight"],
        class_weights=class_weights,
        device=str(device),
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        fold=test_session,
    )

    history = trainer.fit()

    # Save history
    log_dir = Path(config["training"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"fold{test_session}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    best_val = max(h["val"]["weighted_f1"] for h in history)
    return {"fold": test_session, "best_wf1": best_val, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--session_test", type=int, default=None,
                        help="Single session to test (1-5). If omitted, uses --loso.")
    parser.add_argument("--loso", action="store_true",
                        help="Run full Leave-One-Session-Out evaluation (all 5 folds)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["roberta_model"])

    if args.loso or args.session_test is None:
        sessions = config["evaluation"]["sessions"]
        results = []
        for s in sessions:
            r = train_fold(config, s, tokenizer, device)
            results.append(r)
            print(f"\nFold {s} Best WF1: {r['best_wf1']:.4f}")

        avg_wf1 = sum(r["best_wf1"] for r in results) / len(results)
        print(f"\n{'='*60}")
        print(f" LOSO Average WF1: {avg_wf1:.4f}")
        print(f"{'='*60}")

        log_dir = Path(config["training"]["log_dir"])
        with open(log_dir / "loso_summary.json", "w") as f:
            json.dump({"results": results, "avg_wf1": avg_wf1}, f, indent=2)
    else:
        r = train_fold(config, args.session_test, tokenizer, device)
        print(f"\nFold {args.session_test} Best WF1: {r['best_wf1']:.4f}")


if __name__ == "__main__":
    main()
