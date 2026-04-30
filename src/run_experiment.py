"""
run_experiment.py — Timestamped safety evaluation snapshot.

Runs evaluate_safety (with B1 bug fixes) on val + test, then saves:
  outputs/experiments/YYYYMMDD_HHMMSS_<label>/
    config.json
    metrics.json              — trajectory + safety macro/weighted F1 + per-class
    confusion_matrix_val.png
    confusion_matrix_test.png
    pr_curves_val.png
    pr_curves_test.png

Usage:
  python src/run_experiment.py baseline_5feat
  python src/run_experiment.py after_b3_7feat
"""

import os, sys, json
from datetime import datetime
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model import TrajectoryTransformer
from safety import SafetyClassifier, EVENT_NAMES, extract_features, _derive_label
from train import TrajectoryDataset, min_ade, min_fde

CLASS_NAMES = [EVENT_NAMES[i] for i in range(5)]
COLORS      = ["#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#2196F3"]
BATCH_SIZE  = 256


def run_inference(model, loader, clf, device, Y_mean, Y_std):
    """Return (y_true, y_pred, y_proba, all_preds_k, all_gts) — all in meter scale."""
    Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32)
    Y_std_t  = torch.tensor(Y_std,  dtype=torch.float32)
    all_preds_k, all_gts = [], []
    best_preds, gt_trajs = [], []

    model.eval()
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            with autocast("cuda", enabled=(device.type == "cuda")):
                preds_k, confs_k = model(X_batch)

            preds_k_dn = preds_k.float().cpu() * Y_std_t + Y_mean_t
            Y_batch_dn = Y_batch.float().cpu() * Y_std_t + Y_mean_t
            all_preds_k.append(preds_k_dn)
            all_gts.append(Y_batch_dn)

            best_idx = confs_k.argmax(dim=1)
            for i in range(len(preds_k_dn)):
                bp = preds_k_dn[i, best_idx[i]].numpy()
                gt = Y_batch_dn[i].numpy()
                if np.isnan(bp).any() or np.isnan(gt).any():
                    continue
                best_preds.append(bp)
                gt_trajs.append(gt)

    # Batch RF prediction — ~100x faster than one-sample-at-a-time
    y_true = np.array([_derive_label(extract_features(gt)) for gt in gt_trajs])
    y_pred, y_proba = clf.predict_batch(best_preds)

    all_preds_k = torch.cat(all_preds_k, dim=0)
    all_gts     = torch.cat(all_gts,     dim=0)
    return (y_true, y_pred, y_proba, all_preds_k, all_gts)


def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(5), yticks=range(5),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_pr_curves(y_true, y_proba, title, path):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for i, (ax, name, color) in enumerate(zip(axes, CLASS_NAMES, COLORS)):
        binary = (y_true == i).astype(int)
        if binary.sum() == 0:
            ax.set_title(f"{name}\n(no GT samples)"); continue
        prec, rec, _ = precision_recall_curve(binary, y_proba[:, i])
        ap = average_precision_score(binary, y_proba[:, i])
        ax.plot(rec, prec, color=color, lw=2, label=f"AP={ap:.3f}")
        ax.fill_between(rec, prec, alpha=0.15, color=color)
        ax.axhline(binary.mean(), color="grey", lw=1, linestyle="--")
        ax.set(xlabel="Recall", title=name, xlim=[0,1], ylim=[0,1.05])
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Precision")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def parse_report_to_dict(report_str):
    """Pull per-class and macro metrics from classification_report string."""
    result = {}
    for line in report_str.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 5 and parts[-1].replace(",","").isdigit():
            cls = " ".join(parts[:-4])
            try:
                result[cls] = {
                    "precision": float(parts[-4]),
                    "recall":    float(parts[-3]),
                    "f1":        float(parts[-2]),
                    "support":   int(parts[-1]),
                }
            except ValueError:
                pass
    return result


def main(label="experiment"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(ROOT, "data", "processed")
    out_root = os.path.join(ROOT, "outputs")

    X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val   = np.load(os.path.join(data_dir, "Y_val.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    Y_test  = np.load(os.path.join(data_dir, "Y_test.npy"))
    X_mean  = np.load(os.path.join(out_root, "X_mean.npy"))
    X_std   = np.load(os.path.join(out_root, "X_std.npy"))
    Y_mean  = np.load(os.path.join(out_root, "Y_mean.npy"))
    Y_std   = np.load(os.path.join(out_root, "Y_std.npy"))

    X_val_n  = (X_val  - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std
    Y_val_n  = (Y_val  - Y_mean) / Y_std
    Y_test_n = (Y_test - Y_mean) / Y_std

    val_loader  = DataLoader(TrajectoryDataset(X_val_n,  Y_val_n),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(TrajectoryDataset(X_test_n, Y_test_n), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TrajectoryTransformer(
        input_dim=X_val.shape[2], pred_len=Y_val.shape[1],
        d_model=128, nhead=4, num_layers=3, dim_feedforward=256, num_modes=6,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(out_root, "checkpoints", "best_model.pt"),
        map_location=device, weights_only=True,
    ))
    clf = SafetyClassifier.load(os.path.join(out_root, "checkpoints", "safety_clf.pkl"))
    print(f"Model ({X_val.shape[2]}-feat input) + classifier loaded | device={device}")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(out_root, "experiments", f"{ts}_{label}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment dir: {exp_dir}")

    metrics = {"label": label, "input_dim": int(X_val.shape[2])}

    for split, loader in [("val", val_loader), ("test", test_loader)]:
        print(f"\nRunning {split} …")
        y_true, y_pred, y_proba, preds_k, gts = run_inference(
            model, loader, clf, device, Y_mean, Y_std)

        ade = min_ade(preds_k, gts)
        fde = min_fde(preds_k, gts)
        print(f"  minADE={ade:.4f}  minFDE={fde:.4f}")

        report = classification_report(y_true, y_pred, labels=list(range(5)),
                                        target_names=CLASS_NAMES, zero_division=0)
        print(report)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
        plot_confusion_matrix(cm, f"Confusion Matrix — {split} [{label}]",
                              os.path.join(exp_dir, f"confusion_matrix_{split}.png"))
        plot_pr_curves(y_true, y_proba, f"PR Curves — {split} [{label}]",
                       os.path.join(exp_dir, f"pr_curves_{split}.png"))

        metrics[split] = {
            "minADE": round(ade, 4),
            "minFDE": round(fde, 4),
            "classification_report": report,
            "per_class": parse_report_to_dict(report),
            "label_distribution": {EVENT_NAMES[i]: int((y_true == i).sum()) for i in range(5)},
        }

    # Save config snapshot
    import importlib.util, pathlib
    cfg_path = pathlib.Path(ROOT) / "configs" / "train_config.py"
    spec = importlib.util.spec_from_file_location("tc", cfg_path)
    cfg  = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
    metrics["config"] = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nAll outputs saved to: {exp_dir}/")
    return exp_dir, metrics


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "experiment"
    main(label)
