"""
demo_small.py — Full pipeline demo on a tiny data slice.

Designed for demo / screen-recording. Runs end-to-end in ~5-10 minutes:
  1. Slices existing .npy files to a small subset (no re-preprocessing needed)
  2. Trains TrajectoryTransformer for N_DEMO_EPOCHS epochs
  3. Fits SafetyClassifier on the training slice
  4. Evaluates on the test slice  →  minADE / minFDE
  5. Runs inference on DEMO_SAMPLES samples with trajectory visualization
  6. Runs SafetyClassifier  →  event label + features
  7. Generates LLM safety report (Llama-3.2-3B or template fallback)
  8. Saves all outputs to outputs/demo/

Usage:
    cd C:\\Users\\abhis\\projects\\av_safety
    python src/demo_small.py
"""

import os
import sys
import json
import time
import importlib.util
import pathlib

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# ── resolve src/ so local imports work regardless of cwd ─────────────────────
SRC_DIR  = pathlib.Path(__file__).parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from model import TrajectoryTransformer
from safety import SafetyClassifier, extract_features, _derive_label, EVENT_NAMES
from retrieval import DiagnosisPipeline

# ── load config ───────────────────────────────────────────────────────────────
_cfg_path = ROOT_DIR / "configs" / "train_config.py"
_spec = importlib.util.spec_from_file_location("train_config", _cfg_path)
_cfg  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cfg)

# ── demo settings (tweak these to trade speed vs quality) ────────────────────
N_TRAIN        = 5_000    # training samples from existing X_train.npy
N_VAL          = 1_000    # validation samples
N_TEST         = 500      # test samples
N_DEMO_EPOCHS  = 10       # epochs to train  (10 ≈ 3-5 min on RTX 4060)
DEMO_SAMPLES   = 5        # samples to run full pipeline on
# Use the fully-trained model for inference so predictions are realistic (minADE=1.44m).
# Set to False to use the freshly trained demo model instead.
USE_PRETRAINED_FOR_INFERENCE = True
DEMO_BATCH     = 128      # smaller batch for demo slice
OUT_DIR        = ROOT_DIR / "outputs" / "demo"

SEED = _cfg.SEED
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(OUT_DIR, exist_ok=True)


# ── dataset ───────────────────────────────────────────────────────────────────
class TrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ── metrics ───────────────────────────────────────────────────────────────────
def min_ade(preds_k, gt):
    gt_exp   = gt.unsqueeze(1).expand_as(preds_k)
    mode_ade = torch.norm(preds_k - gt_exp, dim=-1).mean(dim=-1)
    return mode_ade.min(dim=1).values.mean().item()

def min_fde(preds_k, gt):
    gt_exp    = gt.unsqueeze(1).expand_as(preds_k)
    final_err = torch.norm(preds_k[:, :, -1] - gt_exp[:, :, -1], dim=-1)
    return final_err.min(dim=1).values.mean().item()


# ── WTA loss ──────────────────────────────────────────────────────────────────
def wta_loss(preds_k, gt):
    gt_exp    = gt.unsqueeze(1).expand_as(preds_k)
    mode_ade  = torch.norm(preds_k - gt_exp, dim=-1).mean(dim=-1)
    best_mode = mode_ade.argmin(dim=1)
    B, K, T, C = preds_k.shape
    idx       = best_mode.view(B, 1, 1, 1).expand(B, 1, T, C)
    best_pred = preds_k.gather(1, idx).squeeze(1)
    return nn.functional.huber_loss(best_pred, gt, delta=1.0)


# ── trajectory visualisation ──────────────────────────────────────────────────
def plot_trajectory(past_xy, pred_modes, best_mode_idx, gt_xy,
                    label_name, sample_idx, features, save_path):
    """
    past_xy    : [50, 2]  — past cumulative positions (relative)
    pred_modes : [6, 60, 2] — predicted displacements
    best_mode_idx : int
    gt_xy      : [60, 2]  — GT displacements
    """
    # convert relative displacements to cumulative positions
    def to_pos(disp):
        return np.cumsum(disp, axis=0)

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#141820")

    past_pos = to_pos(past_xy)

    # past trajectory
    ax.plot(past_pos[:, 0], past_pos[:, 1],
            color="#4a9eff", lw=2, label="Past (50 steps)", zorder=4)
    ax.scatter(past_pos[-1, 0], past_pos[-1, 1],
               color="#4a9eff", s=80, zorder=5)

    # all 6 predicted modes (dim)
    mode_colors = ["#ff6b6b", "#ffd166", "#06d6a0", "#a8dadc", "#e76f51", "#c77dff"]
    for k, mode_disp in enumerate(pred_modes):
        pos = to_pos(mode_disp) + past_pos[-1]
        alpha = 0.85 if k == best_mode_idx else 0.2
        lw    = 2.5  if k == best_mode_idx else 1.0
        ax.plot(pos[:, 0], pos[:, 1],
                color=mode_colors[k], lw=lw, alpha=alpha,
                label=f"Mode {k}{' ← selected' if k==best_mode_idx else ''}", zorder=3)

    # ground truth
    gt_pos = to_pos(gt_xy) + past_pos[-1]
    ax.plot(gt_pos[:, 0], gt_pos[:, 1],
            color="white", lw=2, linestyle="--", alpha=0.7, label="Ground Truth", zorder=4)
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 1],
               color="white", marker="*", s=150, zorder=5)

    # annotation
    speed_str = f"{features['max_speed']:.2f} m/s"
    lat_str   = f"{features['max_lateral_dev']:.2f} m"
    osc_str   = f"{features['oscillation_score']:.3f}"
    ann = (f"Safety Event: {label_name}\n"
           f"Max Speed: {speed_str}   Lateral Dev: {lat_str}   Osc: {osc_str}")
    ax.text(0.02, 0.97, ann, transform=ax.transAxes,
            va="top", ha="left", fontsize=9, color="#e8e8e8",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1530", alpha=0.85))

    ax.set_title(f"Sample {sample_idx} — Trajectory Prediction (K=6 modes)",
                 color="#e8e8e8", fontsize=13, fontweight="bold")
    ax.set_xlabel("X displacement (m)", color="#aaaaaa")
    ax.set_ylabel("Y displacement (m)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values(): sp.set_color("#333333")
    ax.legend(loc="upper right", framealpha=0.3, labelcolor="#e8e8e8",
              facecolor="#111111", edgecolor="#444444", fontsize=8)
    ax.grid(True, color="#222233", lw=0.5)
    ax.set_aspect("equal", "datalim")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved trajectory plot: {save_path}")


# ── training loss curve ───────────────────────────────────────────────────────
def plot_training_curve(history, save_path):
    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    ades   = [h["minADE"] for h in history if h["minADE"] is not None]
    ade_ep = [h["epoch"]  for h in history if h["minADE"] is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#141820")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values(): sp.set_color("#333333")
        ax.grid(True, color="#222233", lw=0.5)

    ax1.plot(epochs, losses, color="#4a9eff", lw=2)
    ax1.set_title("Training Loss (WTA Huber)", color="#e8e8e8", fontweight="bold")
    ax1.set_xlabel("Epoch", color="#aaaaaa")
    ax1.set_ylabel("Loss", color="#aaaaaa")

    ax2.plot(ade_ep, ades, color="#06d6a0", lw=2, marker="o", markersize=4)
    ax2.set_title("Validation minADE (↓ better)", color="#e8e8e8", fontweight="bold")
    ax2.set_xlabel("Epoch", color="#aaaaaa")
    ax2.set_ylabel("minADE (m)", color="#aaaaaa")

    plt.suptitle("Demo Training — Small Slice", color="#e8e8e8", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved training curve: {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AV Safety Forecasting — Demo Pipeline (Small Slice)")
    print("=" * 60)

    # ── load & slice data ─────────────────────────────────────────────────────
    print(f"\n[1/6] Loading data slice  (train={N_TRAIN}, val={N_VAL}, test={N_TEST})")
    data_dir = ROOT_DIR / "data" / "processed"

    X_train_full = np.load(data_dir / "X_train.npy")
    Y_train_full = np.load(data_dir / "Y_train.npy")
    X_val_full   = np.load(data_dir / "X_val.npy")
    Y_val_full   = np.load(data_dir / "Y_val.npy")
    X_test_full  = np.load(data_dir / "X_test.npy")
    Y_test_full  = np.load(data_dir / "Y_test.npy")

    rng = np.random.RandomState(SEED)
    tr_idx = rng.choice(len(X_train_full), N_TRAIN, replace=False)
    va_idx = rng.choice(len(X_val_full),   N_VAL,   replace=False)
    te_idx = rng.choice(len(X_test_full),  N_TEST,  replace=False)

    X_train = X_train_full[tr_idx]; Y_train = Y_train_full[tr_idx]
    X_val   = X_val_full[va_idx];   Y_val   = Y_val_full[va_idx]
    X_test  = X_test_full[te_idx];  Y_test  = Y_test_full[te_idx]

    print(f"  Train : {X_train.shape}  Val : {X_val.shape}  Test : {X_test.shape}")

    # ── normalise ─────────────────────────────────────────────────────────────
    X_mean = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    X_std  = (X_train.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
    Y_mean = Y_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    Y_std  = (Y_train.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)

    X_train_n = (X_train - X_mean) / X_std
    X_val_n   = (X_val   - X_mean) / X_std
    X_test_n  = (X_test  - X_mean) / X_std
    Y_train_n = (Y_train - Y_mean) / Y_std
    Y_val_n   = (Y_val   - Y_mean) / Y_std
    Y_test_n  = (Y_test  - Y_mean) / Y_std

    feat_dim = X_train_n.shape[2]
    pred_len = Y_train_n.shape[1]

    # ── data loaders ──────────────────────────────────────────────────────────
    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(TrajectoryDataset(X_train_n, Y_train_n),
                              batch_size=DEMO_BATCH, shuffle=True, num_workers=0, generator=g)
    val_loader   = DataLoader(TrajectoryDataset(X_val_n, Y_val_n),
                              batch_size=DEMO_BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TrajectoryDataset(X_test_n, Y_test_n),
                              batch_size=DEMO_BATCH, shuffle=False, num_workers=0)

    # ── device ────────────────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"  Device : {device}" + (f"  ({torch.cuda.get_device_name(0)})" if use_amp else ""))

    # ── model (use full config hyperparameters) ───────────────────────────────
    print(f"\n[2/6] Building model  (d_model={_cfg.D_MODEL}, layers={_cfg.NUM_LAYERS}, heads={_cfg.NHEAD})")
    model = TrajectoryTransformer(
        input_dim=feat_dim,
        pred_len=pred_len,
        d_model=_cfg.D_MODEL,
        nhead=_cfg.NHEAD,
        num_layers=_cfg.NUM_LAYERS,
        dim_feedforward=_cfg.DIM_FEEDFORWARD,
        dropout=_cfg.DROPOUT,
        num_modes=_cfg.NUM_MODES,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=_cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=_cfg.LR_FACTOR,
        patience=_cfg.LR_PATIENCE, min_lr=_cfg.MIN_LR
    )
    scaler = GradScaler("cuda", enabled=use_amp)

    # ── training ──────────────────────────────────────────────────────────────
    print(f"\n[3/6] Training for {N_DEMO_EPOCHS} epochs ...")
    best_val_ade = float("inf")
    history = []
    ckpt_path = OUT_DIR / "demo_model.pt"
    t0 = time.time()

    for epoch in range(N_DEMO_EPOCHS):
        model.train()
        train_loss = 0.0
        t_ep = time.time()

        for X_b, Y_b in train_loader:
            X_b = X_b.to(device); Y_b = Y_b.to(device)
            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                preds_k, _ = model(X_b)
                loss = wta_loss(preds_k, Y_b)
            if torch.isnan(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item()

        # validation
        model.eval()
        all_p, all_g = [], []
        Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32)
        Y_std_t  = torch.tensor(Y_std,  dtype=torch.float32)
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b = X_b.to(device)
                with autocast("cuda", enabled=use_amp):
                    preds_k, _ = model(X_b)
                all_p.append(preds_k.float().cpu() * Y_std_t + Y_mean_t)
                all_g.append(Y_b * Y_std_t + Y_mean_t)

        all_p = torch.cat(all_p); all_g = torch.cat(all_g)
        val_ade = min_ade(all_p, all_g)
        val_fde = min_fde(all_p, all_g)
        scheduler.step(val_ade)

        ep_sec = time.time() - t_ep
        print(f"  Epoch {epoch+1:02d}/{N_DEMO_EPOCHS} | loss={train_loss:.4f} | "
              f"minADE={val_ade:.4f} m | minFDE={val_fde:.4f} m | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {ep_sec:.0f}s")

        if not np.isnan(val_ade) and val_ade < best_val_ade:
            best_val_ade = val_ade
            torch.save(model.state_dict(), ckpt_path)

        history.append({"epoch": epoch+1, "train_loss": round(train_loss, 4),
                        "minADE": round(val_ade, 4), "minFDE": round(val_fde, 4)})

    train_min = (time.time() - t0) / 60
    print(f"\n  Training done in {train_min:.1f} min  |  Best val minADE = {best_val_ade:.4f} m")

    plot_training_curve(history, OUT_DIR / "training_curve.png")
    with open(OUT_DIR / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # reload best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ── test set metrics ──────────────────────────────────────────────────────
    print("\n[4/6] Evaluating on test slice ...")
    all_p, all_g = [], []
    with torch.no_grad():
        for X_b, Y_b in test_loader:
            X_b = X_b.to(device)
            with autocast("cuda", enabled=use_amp):
                preds_k, _ = model(X_b)
            all_p.append(preds_k.float().cpu() * Y_std_t + Y_mean_t)
            all_g.append(Y_b * Y_std_t + Y_mean_t)
    all_p = torch.cat(all_p); all_g = torch.cat(all_g)
    test_ade = min_ade(all_p, all_g)
    test_fde = min_fde(all_p, all_g)
    print(f"  Test minADE = {test_ade:.4f} m   minFDE = {test_fde:.4f} m")

    # ── safety classifier ─────────────────────────────────────────────────────
    print("\n[5/6] Fitting SafetyClassifier on training slice ...")
    clf = SafetyClassifier()
    clf.fit(Y_train)
    clf.save(str(OUT_DIR / "demo_safety_clf.pkl"))
    print("  SafetyClassifier fitted and saved.")

    # ── end-to-end inference + visualisation ──────────────────────────────────
    print(f"\n[6/6] Running full pipeline on {DEMO_SAMPLES} test samples ...")

    # Load pre-trained model for inference if available — gives realistic predictions
    pretrained_ckpt = ROOT_DIR / "outputs" / "checkpoints" / "best_model.pt"
    if USE_PRETRAINED_FOR_INFERENCE and pretrained_ckpt.exists():
        print(f"  Using pre-trained model: {pretrained_ckpt}")
        # Use the pre-trained model's normalization stats for correct denormalization
        pt_X_mean = np.load(ROOT_DIR / "outputs" / "X_mean.npy")
        pt_X_std  = np.load(ROOT_DIR / "outputs" / "X_std.npy")
        pt_Y_mean = np.load(ROOT_DIR / "outputs" / "Y_mean.npy")
        pt_Y_std  = np.load(ROOT_DIR / "outputs" / "Y_std.npy")
        # Re-normalize test inputs with pre-trained stats
        X_test_pt = (X_test - pt_X_mean) / pt_X_std
        Y_mean_t  = torch.tensor(pt_Y_mean, dtype=torch.float32)
        Y_std_t   = torch.tensor(pt_Y_std,  dtype=torch.float32)
        infer_model = TrajectoryTransformer(
            input_dim=feat_dim, pred_len=pred_len,
            d_model=_cfg.D_MODEL, nhead=_cfg.NHEAD,
            num_layers=_cfg.NUM_LAYERS, dim_feedforward=_cfg.DIM_FEEDFORWARD,
            dropout=_cfg.DROPOUT, num_modes=_cfg.NUM_MODES,
        ).to(device)
        infer_model.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
        infer_model.eval()
        X_infer = X_test_pt
    else:
        print("  Using demo-trained model for inference.")
        infer_model = model
        X_infer   = X_test_n
        Y_mean_t  = torch.tensor(Y_mean, dtype=torch.float32)
        Y_std_t   = torch.tensor(Y_std,  dtype=torch.float32)

    diag = DiagnosisPipeline()
    all_reports = []

    for i in range(min(DEMO_SAMPLES, len(X_infer))):
        x = torch.tensor(X_infer[i:i+1], dtype=torch.float32).to(device)

        with torch.no_grad():
            preds_k, confs, attn_w = infer_model(x, return_attention=True)

        # denormalise
        preds_m = (preds_k.cpu().float() * Y_std_t + Y_mean_t)[0]   # [K, 60, 2]
        best_k  = confs[0].argmax().item()
        pred_traj = preds_m[best_k].numpy()                           # [60, 2]
        gt_traj   = Y_test[i]                                         # [60, 2]

        # safety
        label, label_name, features = clf.predict(pred_traj)
        class_probs = clf.predict_proba(pred_traj).tolist()
        gt_label    = _derive_label(extract_features(gt_traj))

        # LLM report
        result = diag.run(label_name, features, class_probs)
        report = result["report"]

        # trajectory plot
        plot_trajectory(
            past_xy       = X_test[i, :, :2],      # dx, dy channels
            pred_modes    = preds_m.numpy(),
            best_mode_idx = best_k,
            gt_xy         = gt_traj,
            label_name    = label_name,
            sample_idx    = i,
            features      = features,
            save_path     = OUT_DIR / f"trajectory_sample_{i}.png",
        )

        # attention heatmap
        attn = attn_w[0].mean(dim=0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#141820")
        ax.bar(range(len(attn)), attn, color="#4a9eff", width=0.8)
        ax.set_title(f"Attention over past timesteps — Sample {i}", color="#e8e8e8")
        ax.set_xlabel("Timestep (0=oldest → 49=current)", color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values(): sp.set_color("#333333")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"attention_sample_{i}.png", dpi=130,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

        # console report
        sep = "-" * 60
        print(f"\n{sep}")
        print(f"  Sample {i}")
        print(f"  Predicted event : [{label}] {label_name}")
        print(f"  GT event        : [{gt_label}] {EVENT_NAMES[gt_label]}")
        print(f"  Severity        : {report.get('severity', 'N/A')}")
        print(f"  Confidence      : {report.get('confidence', 'N/A')}")
        print(f"  Max speed       : {features['max_speed']:.4f} m/step")
        print(f"  Lateral dev     : {features['max_lateral_dev']:.4f} m")
        print(f"  Oscillation     : {features['oscillation_score']:.4f}")
        print(f"  Primary         : {report.get('primary_indicator', '')}")
        print(f"  Action          : {report.get('recommended_action', '')}")
        top3 = attn.argsort()[::-1][:3]
        print(f"  Top-3 attended timesteps: {list(top3)}")

        all_reports.append({
            "sample": i,
            "predicted_event": label_name,
            "gt_event": EVENT_NAMES[gt_label],
            "features": {k: round(float(v), 4) for k, v in features.items()},
            "class_probs": class_probs,
            "report": report,
        })

    # ── save all results ──────────────────────────────────────────────────────
    results = {
        "data_slice": {"N_train": N_TRAIN, "N_val": N_VAL, "N_test": N_TEST},
        "epochs_trained": N_DEMO_EPOCHS,
        "best_val_minADE": round(best_val_ade, 4),
        "test_minADE": round(test_ade, 4),
        "test_minFDE": round(test_fde, 4),
        "training_time_min": round(train_min, 1),
        "demo_reports": all_reports,
    }
    with open(OUT_DIR / "demo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  Demo complete!  All outputs saved to: {OUT_DIR}")
    print(f"  Test minADE = {test_ade:.4f} m   minFDE = {test_fde:.4f} m")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
