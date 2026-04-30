"""
train_clf_on_predictions.py — Phase 2: close the distribution gap.

Problem:  The safety classifier was trained on GT trajectory features but is
          evaluated on model-predicted trajectory features.  Model predictions
          are smooth; GT trajectories are noisy.  The RF learned GT patterns
          and can't transfer → Sharp Turn recall ≈ 0%.

Fix:      Run the trajectory model over all X_train inputs, collect the most-
          confident predicted trajectory per sample, extract features from those
          predictions, derive GT-style labels from Y_train, and refit the RF on
          (predicted_features, GT_labels).  Now train and test distributions match.

Outputs:
  outputs/checkpoints/safety_clf.pkl   — updated classifier (overwrites)
  outputs/results.json                 — updated safety reports
"""

import os, sys, json
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model import TrajectoryTransformer
from safety import SafetyClassifier, extract_features, _features_to_vector, _derive_label, FEATURE_KEYS, EVENT_NAMES
from train import TrajectoryDataset, evaluate_safety


# ── config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 512   # larger batch = faster forward pass; reduce if OOM
DATA_DIR   = os.path.join(ROOT, "data", "processed")
OUT_DIR    = os.path.join(ROOT, "outputs")
CLF_PATH   = os.path.join(OUT_DIR, "checkpoints", "safety_clf.pkl")


def collect_predicted_trajectories(model, X_norm, Y_mean, Y_std, device, batch_size=512):
    """
    Run model over all X_norm inputs and return the most-confident predicted
    trajectory per sample, denormalized to meter scale.

    Returns: np.ndarray [N, pred_len, 2]
    """
    Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32)
    Y_std_t  = torch.tensor(Y_std,  dtype=torch.float32)
    use_amp  = device.type == "cuda"

    dataset = TensorDataset(torch.tensor(X_norm, dtype=torch.float32))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    model.eval()
    with torch.no_grad():
        for i, (X_batch,) in enumerate(loader):
            X_batch = X_batch.to(device)
            with autocast("cuda", enabled=use_amp):
                preds_k, confs_k = model(X_batch)   # [B,K,T,2], [B,K]
            # select most-confident mode
            best_idx = confs_k.argmax(dim=1)         # [B]
            best_pred = preds_k[torch.arange(len(preds_k)), best_idx]  # [B,T,2]
            # denormalize
            best_dn = best_pred.float().cpu() * Y_std_t + Y_mean_t    # [B,T,2]
            all_preds.append(best_dn.numpy())
            if (i + 1) % 50 == 0:
                done = min((i + 1) * batch_size, len(X_norm))
                print(f"  {done:>7,} / {len(X_norm):,} processed", flush=True)

    return np.concatenate(all_preds, axis=0)   # [N, T, 2]


def build_feature_matrix(predicted_trajs, gt_trajs):
    """
    Extract features from predicted trajectories; derive labels from GT.
    Returns X [N, n_feat], y [N].
    """
    X_rows, y_rows = [], []
    for pred, gt in zip(predicted_trajs, gt_trajs):
        feats_pred = extract_features(pred)
        feats_gt   = extract_features(gt)
        X_rows.append(_features_to_vector(feats_pred))
        y_rows.append(_derive_label(feats_gt))          # label from GT
    return np.stack(X_rows), np.array(y_rows)


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")

    # ── load data ──────────────────────────────────────────────────────────────
    print("\nLoading data …")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    Y_val   = np.load(os.path.join(DATA_DIR, "Y_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test  = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

    X_mean = np.load(os.path.join(OUT_DIR, "X_mean.npy"))
    X_std  = np.load(os.path.join(OUT_DIR, "X_std.npy"))
    Y_mean = np.load(os.path.join(OUT_DIR, "Y_mean.npy"))
    Y_std  = np.load(os.path.join(OUT_DIR, "Y_std.npy"))
    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}")

    X_train_n = (X_train - X_mean) / X_std
    X_val_n   = (X_val   - X_mean) / X_std
    X_test_n  = (X_test  - X_mean) / X_std
    Y_val_n   = (Y_val   - Y_mean) / Y_std
    Y_test_n  = (Y_test  - Y_mean) / Y_std

    # ── load trajectory model ─────────────────────────────────────────────────
    print("\nLoading trajectory model …")
    model = TrajectoryTransformer(
        input_dim=X_train.shape[2], pred_len=Y_train.shape[1],
        d_model=128, nhead=4, num_layers=3, dim_feedforward=256, num_modes=6,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(OUT_DIR, "checkpoints", "best_model.pt"),
        map_location=device, weights_only=True,
    ))
    model.eval()
    print(f"  Loaded  |  device={device}")

    # ── forward pass over X_train ─────────────────────────────────────────────
    print(f"\nCollecting model predictions for {len(X_train):,} training samples …")
    print(f"  Batch size: {BATCH_SIZE}  |  Estimated: ~{len(X_train)//BATCH_SIZE*2} sec on GPU")
    pred_train = collect_predicted_trajectories(
        model, X_train_n, Y_mean, Y_std, device, batch_size=BATCH_SIZE
    )
    print(f"  Done. pred_train shape: {pred_train.shape}")

    # ── build feature matrix (predicted features, GT labels) ──────────────────
    print("\nExtracting features from predicted trajectories + deriving GT labels …")
    X_feat, y_labels = build_feature_matrix(pred_train, Y_train)

    counts = {EVENT_NAMES[i]: int((y_labels == i).sum()) for i in range(5)}
    print(f"  GT label distribution (from Y_train): {counts}")
    print(f"  Feature matrix shape: {X_feat.shape}")

    # ── fit classifier on predicted-trajectory features ───────────────────────
    print("\nFitting RandomForest on predicted-trajectory features …")
    clf = SafetyClassifier()
    clf.scaler.fit(X_feat)
    X_scaled = clf.scaler.transform(X_feat)
    clf.clf.fit(X_scaled, y_labels)
    clf._fitted = True

    label_dist_str = {EVENT_NAMES[i]: int((y_labels == i).sum()) for i in range(5)}
    print(f"  [SafetyClassifier] Fitted on {len(y_labels)} samples.")
    clf.save(CLF_PATH)

    # ── evaluate on val and test (model predictions as usual) ─────────────────
    print("\nEvaluating on val and test …")
    from torch.utils.data import DataLoader as DL

    val_loader  = DL(TrajectoryDataset(X_val_n,  Y_val_n),  batch_size=256, shuffle=False, num_workers=0)
    test_loader = DL(TrajectoryDataset(X_test_n, Y_test_n), batch_size=256, shuffle=False, num_workers=0)

    val_report  = evaluate_safety(model, val_loader,  clf, device, Y_mean, Y_std)
    print("\nVAL safety report:")
    print(val_report)

    test_report = evaluate_safety(model, test_loader, clf, device, Y_mean, Y_std)
    print("TEST safety report:")
    print(test_report)

    # ── update results.json ───────────────────────────────────────────────────
    results_path = os.path.join(OUT_DIR, "results.json")
    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
    existing["val_safety_report"]  = val_report
    existing["test_safety_report"] = test_report
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nUpdated {results_path}")
    print("\nDone. Classifier trained on model predictions.")


if __name__ == "__main__":
    main()
