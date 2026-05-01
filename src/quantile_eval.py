"""
quantile_eval.py -- Quantile threshold sensitivity analysis across Q50/80/85/90/95.

Why GT->GT is always 1.0:
    Labels are a deterministic function of the same features fed to the RF.
    The RF trivially memorises the rule. This is the ceiling, not a result.

The meaningful evaluation:
    Train RF on GT train features -> GT train labels  (quantile-derived)
    Evaluate on model-predicted test features vs GT test labels.
    This is the real deployment gap: model predictions drift from GT,
    so the classifier sees a shifted feature distribution.

Output: per-class Sensitivity / Precision / AUC-ROC / AUC-PR / F1
        for each quantile level on both train (OOB) and test (predicted).

Usage:
    cd C:\\Users\\abhis\\projects\\av_safety
    python src/quantile_eval.py
"""

import sys, pathlib, time, importlib.util
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_score, recall_score, f1_score)

SRC_DIR  = pathlib.Path(__file__).parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from safety import FEATURE_KEYS, EVENT_NAMES
from model  import TrajectoryTransformer

_cfg_path = ROOT_DIR / "configs" / "train_config.py"
_spec = importlib.util.spec_from_file_location("train_config", _cfg_path)
_cfg  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cfg)

CLASSES   = list(range(5))
N_CLASSES = 5
QUANTILES = [50, 80, 85, 90, 95]
THR_FEATS = ["max_heading_change", "oscillation_score",
             "heading_variance", "max_speed", "max_lateral_dev"]


# ── vectorised feature extraction ─────────────────────────────────────────────
def extract_features_batch(trajs):
    diffs  = np.diff(trajs, axis=1)
    speeds = np.linalg.norm(diffs, axis=2)
    angles = np.arctan2(diffs[:,:,1], diffs[:,:,0])
    hd     = np.abs(np.diff(angles, axis=1))
    start  = trajs[:, 0, :]; end = trajs[:,-1, :]
    direc  = end - start
    length = np.linalg.norm(direc, axis=1, keepdims=True) + 1e-8
    unit   = direc / length
    perp   = np.stack([-unit[:,1], unit[:,0]], axis=1)
    lat    = np.abs((trajs - start[:,None,:]) @ perp[:,:,None]).squeeze(-1)
    total_dist  = speeds.sum(axis=1)
    final_disp  = np.linalg.norm(trajs[:,-1,:] - trajs[:,0,:], axis=1)
    speed_diffs = np.diff(speeds, axis=1)
    max_accel   = np.maximum(speed_diffs, 0).max(axis=1)
    max_decel   = np.maximum(-speed_diffs, 0).max(axis=1)
    jerk        = np.abs(np.diff(speed_diffs, axis=1))
    max_jerk    = jerk.max(axis=1)
    signed_hd   = np.diff(angles, axis=1)
    sign_flip   = (np.diff(np.sign(signed_hd), axis=1) != 0).sum(axis=1).astype(float)
    tdc         = sign_flip / max(signed_hd.shape[1] - 1, 1)
    lat_accel   = np.abs(np.diff(lat, axis=1)).max(axis=1)
    path_eff    = final_disp / (total_dist + 1e-8)
    speed_end   = speeds[:,-1]
    mean_curv   = (hd / (speeds[:,1:] + 1e-8)).mean(axis=1)
    osc_score   = np.diff(hd, axis=1).std(axis=1) if hd.shape[1] > 1 else np.zeros(len(trajs))
    return np.stack([
        speeds.max(axis=1), speeds.mean(axis=1),
        hd.max(axis=1), hd.mean(axis=1), hd.var(axis=1),
        lat.max(axis=1), lat.mean(axis=1),
        total_dist, final_disp, osc_score,
        max_accel, max_decel, max_jerk, tdc,
        lat_accel, path_eff, speed_end, mean_curv,
    ], axis=1).astype(np.float32)


def label_quantile(F, thr):
    idx = {k: i for i, k in enumerate(FEATURE_KEYS)}
    ms  = F[:, idx["max_speed"]];          mhc = F[:, idx["max_heading_change"]]
    hv  = F[:, idx["heading_variance"]];   mld = F[:, idx["max_lateral_dev"]]
    osc = F[:, idx["oscillation_score"]]
    y   = np.zeros(len(F), dtype=int)
    y[mhc > thr["max_heading_change"]]                                    = 1
    y[(osc > thr["oscillation_score"]) & (hv > thr["heading_variance"])] = 2
    y[ms  > thr["max_speed"]]                                             = 3
    y[(ms > thr["max_speed"]) & (mld > thr["max_lateral_dev"])]          = 4
    return y


def pad_proba(prob, classes):
    if len(classes) == N_CLASSES: return prob
    full = np.zeros((len(prob), N_CLASSES), dtype=np.float32)
    for j, c in enumerate(classes): full[:, c] = prob[:, j]
    return full


def compute_metrics(y_true, y_prob):
    Y_bin  = label_binarize(y_true, classes=CLASSES)
    y_pred = y_prob.argmax(axis=1)
    rows   = []
    for c in CLASSES:
        yb = Y_bin[:, c]; yp = y_prob[:, c]
        yh = (y_pred == c).astype(int)
        sup = int(yb.sum())
        if sup == 0:
            rows.append((EVENT_NAMES[c], None, None, None, None, None, sup)); continue
        rows.append((
            EVENT_NAMES[c],
            round(roc_auc_score(yb, yp), 4),
            round(average_precision_score(yb, yp), 4),
            round(recall_score(yb, yh, zero_division=0), 4),
            round(precision_score(yb, yh, zero_division=0), 4),
            round(f1_score(yb, yh, zero_division=0), 4),
            sup,
        ))
    valid = [r for r in rows if r[1] is not None]
    rows.append(("MACRO AVG",
                 round(np.mean([r[1] for r in valid]), 4),
                 round(np.mean([r[2] for r in valid]), 4),
                 round(np.mean([r[3] for r in valid]), 4),
                 round(np.mean([r[4] for r in valid]), 4),
                 round(np.mean([r[5] for r in valid]), 4), ""))
    return rows


def print_table(rows, title):
    fmt = "  {:<22} {:>8} {:>8} {:>12} {:>10} {:>7} {:>8}"
    print(f"\n  [{title}]")
    print(fmt.format("Class", "AUC-ROC", "AUC-PR", "Sensitivity", "Precision", "F1", "Support"))
    print("  " + "-"*74)
    for r in rows:
        vals = [str(v) if v is not None else "N/A" for v in r[1:6]]
        print(fmt.format(r[0], *vals, str(r[6])))


# ── model inference ───────────────────────────────────────────────────────────
def get_predicted_trajectories(X_test_raw, device):
    Xm = np.load(ROOT_DIR / "outputs" / "X_mean.npy")
    Xs = np.load(ROOT_DIR / "outputs" / "X_std.npy")
    Ym = np.load(ROOT_DIR / "outputs" / "Y_mean.npy")
    Ys = np.load(ROOT_DIR / "outputs" / "Y_std.npy")
    X_norm = (X_test_raw - Xm) / Xs
    model = TrajectoryTransformer(
        input_dim=X_norm.shape[2], pred_len=60,
        d_model=_cfg.D_MODEL, nhead=_cfg.NHEAD,
        num_layers=_cfg.NUM_LAYERS, dim_feedforward=_cfg.DIM_FEEDFORWARD,
        dropout=_cfg.DROPOUT, num_modes=_cfg.NUM_MODES,
    ).to(device)
    model.load_state_dict(torch.load(
        ROOT_DIR / "outputs" / "checkpoints" / "best_model.pt", map_location=device))
    model.eval()
    Ym_t = torch.tensor(Ym, dtype=torch.float32)
    Ys_t = torch.tensor(Ys, dtype=torch.float32)
    dl   = DataLoader(TensorDataset(torch.tensor(X_norm, dtype=torch.float32)),
                      batch_size=512, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for (Xb,) in dl:
            Xb = Xb.to(device)
            with autocast("cuda", enabled=(device.type == "cuda")):
                pk, confs = model(Xb)
            best = pk[torch.arange(len(pk)), confs.argmax(dim=1)]
            preds.append((best.float().cpu() * Ys_t + Ym_t).numpy())
    return np.concatenate(preds, axis=0)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT_DIR / "data" / "processed"

    print("\nLoading data ...")
    Y_train = np.load(data_dir / "Y_train.npy")
    Y_test  = np.load(data_dir / "Y_test.npy")
    X_test  = np.load(data_dir / "X_test.npy")
    print(f"  Train: {len(Y_train):,} GT trajectories")
    print(f"  Test : {len(Y_test):,}  GT trajectories  +  X_test for model inference")

    print("\nExtracting features ...")
    t0 = time.time()
    F_train = extract_features_batch(Y_train)
    F_test  = extract_features_batch(Y_test)
    print(f"  GT features done in {time.time()-t0:.1f}s")

    print("\nRunning model inference on X_test ...")
    t0 = time.time()
    Y_pred  = get_predicted_trajectories(X_test, device)
    F_pred  = extract_features_batch(Y_pred)
    print(f"  Inference + feature extraction done in {time.time()-t0:.1f}s")

    # ── threshold table across all quantiles ──────────────────────────────────
    print(f"\n{'='*76}")
    print("  QUANTILE THRESHOLDS  (computed from training GT trajectories)")
    print(f"{'='*76}")
    hdr = f"  {'Feature':<25}" + "".join(f"  {'Q'+str(q):>8}" for q in QUANTILES)
    print(hdr); print("  " + "-"*74)
    idx = {k: i for i, k in enumerate(FEATURE_KEYS)}
    for feat in THR_FEATS:
        vals = "".join(
            f"  {np.percentile(F_train[:, idx[feat]], q):>8.4f}" for q in QUANTILES)
        print(f"  {feat:<25}{vals}")

    # ── per-quantile evaluation ───────────────────────────────────────────────
    for q in QUANTILES:
        thr    = {feat: float(np.percentile(F_train[:, idx[feat]], q))
                  for feat in THR_FEATS}
        y_tr   = label_quantile(F_train, thr)
        y_te   = label_quantile(F_test,  thr)   # GT test labels

        print(f"\n{'='*76}")
        print(f"  Q{q}  |  Label counts:")
        for split, y in [("Train", y_tr), ("Test (GT)", y_te)]:
            dist = {EVENT_NAMES[c]: int((y == c).sum()) for c in CLASSES}
            print(f"    {split+':':<12} {dist}")
        print(f"{'='*76}")

        # train RF
        scaler  = StandardScaler()
        F_tr_sc = scaler.fit_transform(F_train)
        F_pr_sc = scaler.transform(F_pred)

        clf = RandomForestClassifier(n_estimators=300,
                                     class_weight="balanced_subsample",
                                     oob_score=True, random_state=42, n_jobs=-1)
        t0 = time.time()
        clf.fit(F_tr_sc, y_tr)
        print(f"  RF fitted in {time.time()-t0:.1f}s  |  OOB acc: {clf.oob_score_:.4f}")

        # TRAIN (OOB) -- unbiased estimate on all 199,908 training samples
        oob_prob = pad_proba(clf.oob_decision_function_, clf.classes_)
        print_table(compute_metrics(y_tr, oob_prob),
                    f"TRAIN OOB  (N={len(y_tr):,}, unbiased)")

        # TEST -- model-predicted features vs GT test labels
        pred_prob = pad_proba(clf.predict_proba(F_pr_sc), clf.classes_)
        print_table(compute_metrics(y_te, pred_prob),
                    f"TEST  (N={len(y_te):,}, model-predicted features vs GT labels)")

    print(f"\n{'='*76}")
    print("  Note: TRAIN OOB uses GT features. TEST uses model-predicted features.")
    print("  Gap between them = distribution shift from model trajectory smoothing.")
    print(f"{'='*76}\n")


if __name__ == "__main__":
    main()
