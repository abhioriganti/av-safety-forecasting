"""
compare_feature_distributions.py — Phase B2 diagnostic.

Compares 18 safety features extracted from:
  (a) GT future trajectories      (Y_val)
  (b) Model-predicted trajectories (model(X_val), most-confident mode)

Outputs (timestamped under outputs/experiments/):
  feature_drift.json        — per-feature stats + drift rank
  feature_distributions.png — side-by-side histograms for top-8 shifted features
  feature_drift_bar.png     — ranked bar chart of drift scores

Run before AND after B3 retraining to measure improvement.
"""

import os, sys, json
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from safety import extract_features, _features_to_vector, FEATURE_KEYS, _derive_label, EVENT_NAMES
from model import TrajectoryTransformer

N_SAMPLES  = 3000   # subset of val to keep this fast (~30 sec)
BATCH_SIZE = 256


def collect_predictions(model, X_norm, Y_mean, Y_std, device):
    """Return most-confident predicted trajectories [N, 60, 2] in meter scale."""
    Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32)
    Y_std_t  = torch.tensor(Y_std,  dtype=torch.float32)
    ds  = TensorDataset(torch.tensor(X_norm, dtype=torch.float32))
    dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    preds_all = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            pk, ck = model(xb)
            best = ck.argmax(dim=1)
            bp = pk[torch.arange(len(pk)), best].float().cpu() * Y_std_t + Y_mean_t
            preds_all.append(bp.numpy())
    return np.concatenate(preds_all, axis=0)


def extract_feature_matrix(trajs):
    """[N, T, 2] → [N, 18] feature matrix + [N] labels."""
    rows, labels = [], []
    for t in trajs:
        f = extract_features(t)
        rows.append(_features_to_vector(f))
        labels.append(_derive_label(f))
    return np.stack(rows), np.array(labels)


def normalized_drift(a, b):
    """Normalized mean absolute difference between two distributions."""
    denom = (np.std(a) + np.std(b)) / 2 + 1e-8
    return float(abs(np.mean(a) - np.mean(b)) / denom)


def main(label="baseline"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ──────────────────────────────────────────────────────────────
    data_dir = os.path.join(ROOT, "data", "processed")
    X_val  = np.load(os.path.join(data_dir, "X_val.npy"))[:N_SAMPLES]
    Y_val  = np.load(os.path.join(data_dir, "Y_val.npy"))[:N_SAMPLES]
    X_mean = np.load(os.path.join(ROOT, "outputs", "X_mean.npy"))
    X_std  = np.load(os.path.join(ROOT, "outputs", "X_std.npy"))
    Y_mean = np.load(os.path.join(ROOT, "outputs", "Y_mean.npy"))
    Y_std  = np.load(os.path.join(ROOT, "outputs", "Y_std.npy"))

    X_val_n = (X_val - X_mean) / X_std
    print(f"Using {len(X_val)} val samples (input_dim={X_val.shape[2]})")

    # ── load model ─────────────────────────────────────────────────────────────
    model = TrajectoryTransformer(
        input_dim=X_val.shape[2], pred_len=Y_val.shape[1],
        d_model=128, nhead=4, num_layers=3, dim_feedforward=256, num_modes=6,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(ROOT, "outputs", "checkpoints", "best_model.pt"),
        map_location=device, weights_only=True,
    ))
    print("Model loaded.")

    # ── collect predicted trajectories ─────────────────────────────────────────
    print("Collecting model predictions …")
    pred_trajs = collect_predictions(model, X_val_n, Y_mean, Y_std, device)

    # ── extract features ────────────────────────────────────────────────────────
    print("Extracting features from GT and predicted trajectories …")
    X_gt,   y_gt   = extract_feature_matrix(Y_val)
    X_pred, y_pred = extract_feature_matrix(pred_trajs)

    # ── compute per-feature drift ───────────────────────────────────────────────
    drift_results = {}
    for i, key in enumerate(FEATURE_KEYS):
        gt_vals   = X_gt[:, i]
        pred_vals = X_pred[:, i]
        drift_results[key] = {
            "gt_mean":    float(gt_vals.mean()),
            "gt_std":     float(gt_vals.std()),
            "pred_mean":  float(pred_vals.mean()),
            "pred_std":   float(pred_vals.std()),
            "gt_p50":     float(np.median(gt_vals)),
            "pred_p50":   float(np.median(pred_vals)),
            "gt_p90":     float(np.percentile(gt_vals, 90)),
            "pred_p90":   float(np.percentile(pred_vals, 90)),
            "drift_score": normalized_drift(gt_vals, pred_vals),
        }

    # ── label distribution shift ────────────────────────────────────────────────
    gt_label_dist   = {EVENT_NAMES[i]: int((y_gt   == i).sum()) for i in range(5)}
    pred_label_dist = {EVENT_NAMES[i]: int((y_pred == i).sum()) for i in range(5)}

    # ── rank by drift ───────────────────────────────────────────────────────────
    ranked = sorted(drift_results.items(), key=lambda x: x[1]["drift_score"], reverse=True)
    print("\nTop-10 most shifted features:")
    print(f"  {'Feature':<30s} {'GT mean':>10s} {'Pred mean':>10s} {'Drift':>8s}")
    for k, v in ranked[:10]:
        print(f"  {k:<30s} {v['gt_mean']:>10.4f} {v['pred_mean']:>10.4f} {v['drift_score']:>8.3f}")

    print("\nLabel distribution — GT vs Predicted:")
    print(f"  {'Class':<25s} {'GT':>8s} {'Pred':>8s}")
    for cls in EVENT_NAMES.values():
        print(f"  {cls:<25s} {gt_label_dist[cls]:>8d} {pred_label_dist[cls]:>8d}")

    # ── save outputs ────────────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, "outputs", "experiments", f"{ts}_{label}_feature_drift")
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "label": label,
        "n_samples": N_SAMPLES,
        "input_dim": int(X_val.shape[2]),
        "gt_label_distribution":   gt_label_dist,
        "pred_label_distribution": pred_label_dist,
        "features": drift_results,
        "drift_ranking": [k for k, _ in ranked],
    }
    with open(os.path.join(out_dir, "feature_drift.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # ── histogram plots for top-8 drifted features ─────────────────────────────
    top8 = [k for k, _ in ranked[:8]]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for ax, key in zip(axes, top8):
        i = list(FEATURE_KEYS).index(key)
        v = drift_results[key]
        ax.hist(X_gt[:, i],   bins=50, alpha=0.6, color="#1976D2", label="GT",        density=True)
        ax.hist(X_pred[:, i], bins=50, alpha=0.6, color="#FF7043", label="Predicted", density=True)
        ax.axvline(v["gt_mean"],   color="#1976D2", lw=2, linestyle="--")
        ax.axvline(v["pred_mean"], color="#FF7043", lw=2, linestyle="--")
        ax.set_title(f"{key}\ndrift={v['drift_score']:.3f}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"GT vs Predicted Feature Distributions  [{label}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_distributions.png"), dpi=150)
    plt.close(fig)

    # ── drift bar chart ─────────────────────────────────────────────────────────
    keys   = [k for k, _ in ranked]
    scores = [v["drift_score"] for _, v in ranked]
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    colors_bar = ["#C62828" if s > 1.0 else "#FB8C00" if s > 0.5 else "#43A047" for s in scores]
    ax2.bar(range(len(keys)), scores, color=colors_bar)
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels(keys, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("Normalized drift score")
    ax2.set_title(f"Per-feature distribution drift: GT vs Predicted  [{label}]\n"
                  f"Red > 1.0 (severe)  |  Orange 0.5–1.0 (moderate)  |  Green < 0.5 (low)")
    ax2.axhline(1.0, color="red",    lw=1, linestyle="--", alpha=0.5)
    ax2.axhline(0.5, color="orange", lw=1, linestyle="--", alpha=0.5)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "feature_drift_bar.png"), dpi=150)
    plt.close(fig2)

    print(f"\nOutputs saved to: {out_dir}/")
    print("  feature_drift.json")
    print("  feature_distributions.png")
    print("  feature_drift_bar.png")
    return out_dir, payload


if __name__ == "__main__":
    import sys
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    main(label)
