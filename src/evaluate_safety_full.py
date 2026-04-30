"""
evaluate_safety_full.py — Comprehensive safety classifier evaluation.

Produces:
  outputs/safety_eval/confusion_matrix.png
  outputs/safety_eval/pr_curves.png
  outputs/safety_eval/calibration.png
  outputs/safety_eval/feature_importance.png
  outputs/safety_eval/metrics_baseline.json

Run BEFORE making any changes to safety.py so you have a clean baseline.
Then run again after refitting with the enhanced feature set to compare.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from safety import SafetyClassifier, extract_features, _features_to_vector, _derive_label, EVENT_NAMES, FEATURE_KEYS

OUT_DIR = os.path.join(ROOT, "outputs", "safety_eval")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = [EVENT_NAMES[i] for i in range(5)]
COLORS = ["#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#2196F3"]


# ── helpers ──────────────────────────────────────────────────────────────────

def build_dataset(trajectories: np.ndarray):
    """Extract features + weak-supervision labels from a trajectory array [N, T, 2]."""
    X_rows, y_true = [], []
    for traj in trajectories:
        feats = extract_features(traj)
        X_rows.append(_features_to_vector(feats))
        y_true.append(_derive_label(feats))
    return np.stack(X_rows), np.array(y_true)


def load_data():
    data_dir = os.path.join(ROOT, "data", "processed")
    print("Loading Y_val  …", end=" ", flush=True)
    Y_val  = np.load(os.path.join(data_dir, "Y_val.npy"))
    print(f"{len(Y_val):,} samples")
    print("Loading Y_test …", end=" ", flush=True)
    Y_test = np.load(os.path.join(data_dir, "Y_test.npy"))
    print(f"{len(Y_test):,} samples")
    return Y_val, Y_test


# ── plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(5), yticks=range(5),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted", ylabel="True", title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2.0
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_pr_curves(y_true, y_proba, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for i, (ax, name, color) in enumerate(zip(axes, CLASS_NAMES, COLORS)):
        binary = (y_true == i).astype(int)
        if binary.sum() == 0:
            ax.set_title(f"{name}\n(no samples)")
            continue
        prec, rec, _ = precision_recall_curve(binary, y_proba[:, i])
        ap = average_precision_score(binary, y_proba[:, i])
        ax.plot(rec, prec, color=color, lw=2, label=f"AP={ap:.3f}")
        ax.fill_between(rec, prec, alpha=0.15, color=color)
        baseline = binary.mean()
        ax.axhline(baseline, color="grey", lw=1, linestyle="--", label=f"base={baseline:.3f}")
        ax.set(xlabel="Recall", title=name, xlim=[0, 1], ylim=[0, 1.05])
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Precision")
    fig.suptitle("Precision–Recall Curves (one-vs-rest)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_calibration(y_true, y_proba, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for i, (ax, name, color) in enumerate(zip(axes, CLASS_NAMES, COLORS)):
        binary = (y_true == i).astype(int)
        if binary.sum() < 10:
            ax.set_title(f"{name}\n(too few)")
            continue
        n_bins = min(10, max(3, binary.sum() // 20))
        try:
            frac_pos, mean_pred = calibration_curve(binary, y_proba[:, i], n_bins=n_bins)
            ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, label="classifier")
        except Exception:
            pass
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.set(xlabel="Mean predicted prob", title=name, xlim=[0, 1], ylim=[-0.05, 1.05])
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Fraction positives")
    fig.suptitle("Calibration (reliability diagram)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_feature_importance(clf_obj, feature_keys, save_path):
    importances = clf_obj.clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(importances)), importances[indices], color="#1976D2", alpha=0.85)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_keys[i] for i in indices], rotation=40, ha="right")
    ax.set(ylabel="Mean decrease in impurity", title="Random Forest — Feature Importance")
    for bar, imp in zip(bars, importances[indices]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{imp:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    clf_path = os.path.join(ROOT, "outputs", "checkpoints", "safety_clf.pkl")
    print(f"\nLoading safety classifier from {clf_path} …")
    clf = SafetyClassifier.load(clf_path)
    print("Classifier loaded.")

    Y_val, Y_test = load_data()

    results = {}
    for split_name, Y in [("val", Y_val), ("test", Y_test)]:
        print(f"\n{'='*60}")
        print(f"  Evaluating on {split_name} ({len(Y):,} trajectories) …")

        print("  Extracting features …", flush=True)
        X, y_true = build_dataset(Y)

        print("  Running classifier predictions …", flush=True)
        X_scaled = clf.scaler.transform(X)
        y_pred   = clf.clf.predict(X_scaled)
        y_proba  = clf.clf.predict_proba(X_scaled)  # [N, n_classes]

        # Pad to 5-column proba if classifier saw fewer classes during fit
        n_clf_classes = y_proba.shape[1]
        if n_clf_classes < 5:
            clf_classes = list(clf.clf.classes_)
            full_proba = np.zeros((len(y_proba), 5), dtype=np.float32)
            for col, cls_idx in enumerate(clf_classes):
                full_proba[:, cls_idx] = y_proba[:, col]
            y_proba = full_proba

        print(f"\n  Classification report ({split_name}):")
        report_str = classification_report(
            y_true, y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
        print(report_str)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))

        plot_confusion_matrix(
            cm,
            title=f"Confusion Matrix — {split_name}",
            save_path=os.path.join(OUT_DIR, f"confusion_matrix_{split_name}.png"),
        )
        plot_pr_curves(
            y_true, y_proba,
            save_path=os.path.join(OUT_DIR, f"pr_curves_{split_name}.png"),
        )
        plot_calibration(
            y_true, y_proba,
            save_path=os.path.join(OUT_DIR, f"calibration_{split_name}.png"),
        )

        # Per-class AP and Brier score
        per_class = {}
        for i in range(5):
            binary = (y_true == i).astype(int)
            if binary.sum() > 0:
                ap = float(average_precision_score(binary, y_proba[:, i]))
                bs = float(brier_score_loss(binary, y_proba[:, i]))
            else:
                ap, bs = None, None
            per_class[EVENT_NAMES[i]] = {"average_precision": ap, "brier_score": bs}

        label_dist = {EVENT_NAMES[i]: int((y_true == i).sum()) for i in range(5)}

        results[split_name] = {
            "n_samples": len(Y),
            "label_distribution": label_dist,
            "classification_report": report_str,
            "per_class_metrics": per_class,
        }

    plot_feature_importance(
        clf,
        list(FEATURE_KEYS),
        save_path=os.path.join(OUT_DIR, "feature_importance.png"),
    )

    # Save metrics JSON — filename includes feature count so runs don't overwrite each other
    n_features = len(FEATURE_KEYS)
    metrics_path = os.path.join(OUT_DIR, f"metrics_{n_features}feat.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print(f"\nAll outputs in: {OUT_DIR}/")
    print("  confusion_matrix_val.png / _test.png")
    print("  pr_curves_val.png / _test.png")
    print("  calibration_val.png / _test.png")
    print("  feature_importance.png")
    print("  metrics_baseline.json")


if __name__ == "__main__":
    main()
