"""
evaluate.py — Run safety + trajectory evaluation on saved checkpoint.
Use this instead of re-running train.py when the model is already trained.
"""

import json
import os
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from model import TrajectoryTransformer
from safety import SafetyClassifier, EVENT_NAMES
from train import TrajectoryDataset, min_ade, min_fde, evaluate_safety


def main():
    # ── load data and normalization stats ─────────────────────────────────────
    X_val  = np.load("data/processed/X_val.npy")
    Y_val  = np.load("data/processed/Y_val.npy")
    X_test = np.load("data/processed/X_test.npy")
    Y_test = np.load("data/processed/Y_test.npy")

    X_mean = np.load("outputs/X_mean.npy")
    X_std  = np.load("outputs/X_std.npy")
    Y_mean = np.load("outputs/Y_mean.npy")
    Y_std  = np.load("outputs/Y_std.npy")

    X_val_n  = (X_val  - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std
    Y_val_n  = (Y_val  - Y_mean) / Y_std
    Y_test_n = (Y_test - Y_mean) / Y_std

    val_loader  = DataLoader(TrajectoryDataset(X_val_n,  Y_val_n),  batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(TrajectoryDataset(X_test_n, Y_test_n), batch_size=128, shuffle=False, num_workers=0)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # ── load model ────────────────────────────────────────────────────────────
    model = TrajectoryTransformer(
        input_dim=X_val.shape[2], pred_len=Y_val.shape[1],
        d_model=128, nhead=4, num_layers=3, dim_feedforward=256, num_modes=6,
    ).to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pt", map_location=device))
    model.eval()
    print(f"Model loaded from outputs/checkpoints/best_model.pt  |  device={device}")

    # ── load safety classifier ────────────────────────────────────────────────
    clf = SafetyClassifier.load("outputs/checkpoints/safety_clf.pkl")

    # ── trajectory metrics ────────────────────────────────────────────────────
    Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32)
    Y_std_t  = torch.tensor(Y_std,  dtype=torch.float32)

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        all_preds_k, all_gts = [], []
        with torch.no_grad():
            for X_batch, Y_batch in loader:
                X_batch = X_batch.to(device)
                with autocast("cuda", enabled=use_amp):
                    preds_k, _ = model(X_batch)
                preds_dn = preds_k.float().cpu() * Y_std_t + Y_mean_t
                all_preds_k.append(preds_dn)
                all_gts.append(Y_batch * Y_std_t + Y_mean_t)

        all_preds_k = torch.cat(all_preds_k, dim=0)
        all_gts     = torch.cat(all_gts,     dim=0)
        ade = min_ade(all_preds_k, all_gts)
        fde = min_fde(all_preds_k, all_gts)
        print(f"\n{split_name.upper()} trajectory metrics:  minADE={ade:.4f}m  minFDE={fde:.4f}m")

    # ── safety classification ─────────────────────────────────────────────────
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        print(f"\nSafety classification — {split_name} set:")
        report = evaluate_safety(model, loader, clf, device, Y_mean, Y_std)
        print(report)

    # ── save results ──────────────────────────────────────────────────────────
    existing = {}
    if os.path.exists("outputs/results.json"):
        with open("outputs/results.json") as f:
            existing = json.load(f)

    with open("outputs/results.json", "w") as f:
        json.dump(existing, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
