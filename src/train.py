"""
train.py — Train TrajectoryTransformer (multimodal, K=6 modes).

- Winner-Takes-All (WTA) loss prevents mode collapse.
- minADE / minFDE — standard Argoverse 2 leaderboard metrics.
- Input/output normalization — prevents NaN loss from large coordinate values.
- AMP (torch.amp) — ~1.5x speedup on RTX 4060.
- Fits SafetyClassifier on ground-truth train trajectories.
- Evaluates on both val and test sets.
"""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe in background/headless runs
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

from model import TrajectoryTransformer
from safety import SafetyClassifier, EVENT_NAMES, extract_features, _derive_label

# ── load config (project root / configs/) ─────────────────────────────────────
import importlib.util, pathlib
_cfg_path = pathlib.Path(__file__).parent.parent / "configs" / "train_config.py"
_spec = importlib.util.spec_from_file_location("train_config", _cfg_path)
_cfg  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cfg)

SEED                = _cfg.SEED
D_MODEL             = _cfg.D_MODEL
NHEAD               = _cfg.NHEAD
NUM_LAYERS_CFG      = _cfg.NUM_LAYERS
DIM_FEEDFORWARD     = _cfg.DIM_FEEDFORWARD
DROPOUT             = _cfg.DROPOUT
NUM_MODES           = _cfg.NUM_MODES
BATCH_SIZE          = _cfg.BATCH_SIZE
LEARNING_RATE       = _cfg.LEARNING_RATE
NUM_EPOCHS          = _cfg.NUM_EPOCHS
EARLY_STOP_PATIENCE = _cfg.EARLY_STOP_PATIENCE
LR_FACTOR           = _cfg.LR_FACTOR
LR_PATIENCE_CFG     = _cfg.LR_PATIENCE
MIN_LR              = _cfg.MIN_LR


# ── dataset ───────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ── metrics ───────────────────────────────────────────────────────────────────

def min_ade(preds_k: torch.Tensor, gt: torch.Tensor) -> float:
    gt_exp   = gt.unsqueeze(1).expand_as(preds_k)
    mode_ade = torch.norm(preds_k - gt_exp, dim=-1).mean(dim=-1)
    return mode_ade.min(dim=1).values.mean().item()


def min_fde(preds_k: torch.Tensor, gt: torch.Tensor) -> float:
    gt_exp    = gt.unsqueeze(1).expand_as(preds_k)
    final_err = torch.norm(preds_k[:, :, -1] - gt_exp[:, :, -1], dim=-1)
    return final_err.min(dim=1).values.mean().item()


# ── winner-takes-all loss ─────────────────────────────────────────────────────

def wta_loss(preds_k: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    gt_exp    = gt.unsqueeze(1).expand_as(preds_k)
    mode_ade  = torch.norm(preds_k - gt_exp, dim=-1).mean(dim=-1)
    best_mode = mode_ade.argmin(dim=1)
    B, K, T, C = preds_k.shape
    idx       = best_mode.view(B, 1, 1, 1).expand(B, 1, T, C)
    best_pred = preds_k.gather(1, idx).squeeze(1)
    return nn.functional.mse_loss(best_pred, gt)


# ── attention visualisation ───────────────────────────────────────────────────

def save_attention_plot(attn_weights: torch.Tensor, sample_idx: int, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_weights.cpu().numpy(), aspect="auto", cmap="viridis")
    ax.set_title(f"Encoder Self-Attention — Sample {sample_idx}")
    ax.set_xlabel("Key timestep")
    ax.set_ylabel("Query timestep")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(out_dir, f"attention_{sample_idx}.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Saved: {path}")


# ── safety evaluation ─────────────────────────────────────────────────────────

def evaluate_safety(model, loader, clf, device, Y_mean, Y_std):
    """
    Evaluate safety classification on model-predicted vs GT trajectories.

    BUG1 FIX: GT labels derived via _derive_label(extract_features(gt_traj)),
              NOT clf.predict(gt_traj). GT labels are now classifier-independent.

    BUG2 FIX: Mode selected by conf.argmax() (most confident), NOT mode_ade.argmin()
              (oracle WTA). This matches the real inference pipeline.
    """
    model.eval()
    pred_labels, gt_labels = [], []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            with autocast("cuda", enabled=(device.type == "cuda")):
                preds_k, confs_k = model(X_batch)   # capture confidences

            Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32).to(device)
            Y_std_t  = torch.tensor(Y_std,  dtype=torch.float32).to(device)
            preds_k  = preds_k.float() * Y_std_t + Y_mean_t
            Y_batch  = Y_batch.float() * Y_std_t + Y_mean_t

            # B1-BUG2: confidence-based mode selection (matches demo pipeline)
            best_idx = confs_k.argmax(dim=1)

            for i in range(len(preds_k)):
                best_pred = preds_k[i, best_idx[i]].cpu().numpy()
                gt_traj   = Y_batch[i].cpu().numpy()
                if np.isnan(best_pred).any() or np.isnan(gt_traj).any():
                    continue
                label_p, _, _ = clf.predict(best_pred)
                # B1-BUG1: fixed GT label — independent of classifier
                label_g = _derive_label(extract_features(gt_traj))
                pred_labels.append(label_p)
                gt_labels.append(label_g)

    if not pred_labels:
        return "No valid predictions (all NaN — model may need more training)."
    target_names = [EVENT_NAMES[i] for i in range(5)]
    return classification_report(gt_labels, pred_labels, labels=list(range(5)),
                                  target_names=target_names, zero_division=0)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── fixed seed for reproducibility ────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/figures",     exist_ok=True)
    os.makedirs("outputs/attention",   exist_ok=True)
    os.makedirs("outputs/experiments", exist_ok=True)
    os.makedirs("logs",                exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    X_train = np.load("data/processed/X_train.npy")
    Y_train = np.load("data/processed/Y_train.npy")
    X_val   = np.load("data/processed/X_val.npy")
    Y_val   = np.load("data/processed/Y_val.npy")
    X_test  = np.load("data/processed/X_test.npy")
    Y_test  = np.load("data/processed/Y_test.npy")

    print(f"Train : X={X_train.shape}  Y={Y_train.shape}")
    print(f"Val   : X={X_val.shape}    Y={Y_val.shape}")
    print(f"Test  : X={X_test.shape}   Y={Y_test.shape}")

    # ── normalize using training set statistics ───────────────────────────────
    # Compute per-feature mean/std over (N, T) axes from training data
    X_mean = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)  # [1,1,5]
    X_std  = (X_train.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
    Y_mean = Y_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)  # [1,1,2]
    Y_std  = (Y_train.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)

    X_train_n = (X_train - X_mean) / X_std
    X_val_n   = (X_val   - X_mean) / X_std
    X_test_n  = (X_test  - X_mean) / X_std
    Y_train_n = (Y_train - Y_mean) / Y_std
    Y_val_n   = (Y_val   - Y_mean) / Y_std
    Y_test_n  = (Y_test  - Y_mean) / Y_std

    # Save stats — needed for inference in demo_pipeline.py
    np.save("outputs/X_mean.npy", X_mean)
    np.save("outputs/X_std.npy",  X_std)
    np.save("outputs/Y_mean.npy", Y_mean)
    np.save("outputs/Y_std.npy",  Y_std)
    print("Normalization stats saved.")
    print(f"  X_mean (per feature): {X_mean.squeeze().tolist()}")
    print(f"  X_std  (per feature): {X_std.squeeze().tolist()}")

    feat_dim = X_train_n.shape[2]
    pred_len = Y_train_n.shape[1]

    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(TrajectoryDataset(X_train_n, Y_train_n), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True, generator=g)
    val_loader   = DataLoader(TrajectoryDataset(X_val_n,   Y_val_n),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(TrajectoryDataset(X_test_n,  Y_test_n),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"\nDevice: {device}  |  AMP: {use_amp}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── model ─────────────────────────────────────────────────────────────────
    model = TrajectoryTransformer(
        input_dim=feat_dim,
        pred_len=pred_len,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS_CFG,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_modes=NUM_MODES,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE_CFG, min_lr=MIN_LR
    )
    scaler    = GradScaler("cuda", enabled=use_amp)

    best_val_ade   = float("inf")
    early_stop_ctr = 0
    history        = []

    # ── training loop ─────────────────────────────────────────────────────────
    t_train_start = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        nan_batches = 0
        t_epoch = time.time()

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()

            with autocast("cuda", enabled=use_amp):
                preds_k, _ = model(X_batch)
                loss = wta_loss(preds_k, Y_batch)

            if torch.isnan(loss):
                nan_batches += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        if nan_batches > 0:
            print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | WARNING: {nan_batches} NaN batches skipped")

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds_k, all_gts = [], []

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                with autocast("cuda", enabled=use_amp):
                    preds_k, _ = model(X_batch)
                # Denormalize for metric computation in meter space
                Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32)
                Y_std_t  = torch.tensor(Y_std,  dtype=torch.float32)
                preds_dn = preds_k.float().cpu() * Y_std_t + Y_mean_t
                all_preds_k.append(preds_dn)
                all_gts.append(Y_batch * Y_std_t + Y_mean_t)

        all_preds_k = torch.cat(all_preds_k, dim=0)
        all_gts     = torch.cat(all_gts,     dim=0)
        val_ade     = min_ade(all_preds_k, all_gts)
        val_fde     = min_fde(all_preds_k, all_gts)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "minADE": round(val_ade, 4) if not np.isnan(val_ade) else None,
            "minFDE": round(val_fde, 4) if not np.isnan(val_fde) else None,
        })
        epoch_sec = time.time() - t_epoch
        elapsed   = time.time() - t_train_start
        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | loss={train_loss:.4f} | minADE={val_ade:.4f} | minFDE={val_fde:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | {epoch_sec:.0f}s/ep | total={elapsed/60:.1f}m")

        if not np.isnan(val_ade) and val_ade < best_val_ade:
            best_val_ade  = val_ade
            early_stop_ctr = 0
            torch.save(model.state_dict(), "outputs/checkpoints/best_model.pt")
            print(f"  * Best model saved (minADE={best_val_ade:.4f})")
        else:
            early_stop_ctr += 1
            if early_stop_ctr >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

        # ReduceLROnPlateau uses val_ade as the monitored metric
        scheduler.step(val_ade)

    # ── save training history ─────────────────────────────────────────────────
    with open("outputs/train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── attention visualisation ───────────────────────────────────────────────
    print("\nGenerating attention visualisations...")
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        for i in range(min(4, len(X_val_n))):
            x = torch.tensor(X_val_n[i:i+1], dtype=torch.float32).to(device)
            _, _, attn = model(x, return_attention=True)
            save_attention_plot(attn[0], i, "outputs/attention")

    # ── fit safety classifier on training trajectories ────────────────────────
    print("\nFitting SafetyClassifier on train ground-truth trajectories (original scale)...")
    clf = SafetyClassifier()
    clf.fit(Y_train)   # fit on un-normalized meter-scale trajectories
    clf.save("outputs/checkpoints/safety_clf.pkl")

    # ── safety evaluation ─────────────────────────────────────────────────────
    print("\nSafety classification on val set:")
    val_report = evaluate_safety(model, val_loader, clf, device, Y_mean, Y_std)
    print(val_report)

    print("Safety classification on test set:")
    test_report = evaluate_safety(model, test_loader, clf, device, Y_mean, Y_std)
    print(test_report)

    # ── test set trajectory metrics ───────────────────────────────────────────
    print("Computing test set trajectory metrics...")
    model.eval()
    all_preds_k, all_gts = [], []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            with autocast("cuda", enabled=use_amp):
                preds_k, _ = model(X_batch)
            Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32)
            Y_std_t  = torch.tensor(Y_std,  dtype=torch.float32)
            preds_dn = preds_k.float().cpu() * Y_std_t + Y_mean_t
            all_preds_k.append(preds_dn)
            all_gts.append(Y_batch * Y_std_t + Y_mean_t)

    all_preds_k = torch.cat(all_preds_k, dim=0)
    all_gts     = torch.cat(all_gts,     dim=0)
    test_ade    = min_ade(all_preds_k, all_gts)
    test_fde    = min_fde(all_preds_k, all_gts)
    print(f"Test  minADE={test_ade:.4f}  minFDE={test_fde:.4f}")

    # ── save results ──────────────────────────────────────────────────────────
    results = {
        "best_val_minADE":  best_val_ade,
        "final_val_minFDE": val_fde,
        "test_minADE":      test_ade,
        "test_minFDE":      test_fde,
        "val_safety_report":  val_report,
        "test_safety_report": test_report,
    }
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nTraining complete. Outputs in outputs/")


if __name__ == "__main__":
    main()
