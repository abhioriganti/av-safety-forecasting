"""
demo_terminal.py - Clean terminal demo matching presentation style.

Output mirrors the style shown in demo:
  - Epoch-by-epoch training progress
  - Final ADE / FDE / precision / recall / F1
  - Per-sample: predicted label, retrieved similar cases, diagnosis

Usage:
    cd C:\\Users\\abhis\\projects\\av_safety
    python src/demo_terminal.py
"""

import os, sys, json, time, importlib.util, pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

SRC_DIR  = pathlib.Path(__file__).parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from model import TrajectoryTransformer
from safety import SafetyClassifier, extract_features, _derive_label, EVENT_NAMES, FEATURE_KEYS

# load config
_cfg_path = ROOT_DIR / "configs" / "train_config.py"
_spec = importlib.util.spec_from_file_location("train_config", _cfg_path)
_cfg  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cfg)

# ── settings ──────────────────────────────────────────────────────────────────
N_TRAIN       = 5_000
N_VAL         = 1_000
N_TEST        = 500
N_EPOCHS      = 10
BATCH_SIZE    = 128
DEMO_SAMPLES  = 5
SEED          = _cfg.SEED

torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

OUT_DIR = ROOT_DIR / "outputs" / "demo"
os.makedirs(OUT_DIR, exist_ok=True)


# ── knowledge base ─────────────────────────────────────────────────────────────
# Representative feature vectors (18-d) and descriptions per safety class.
# Values are calibrated to typical Argoverse 2 ranges.
KNOWLEDGE_BASE = [
    # Safe
    {"label": 0, "features": [0.4, 0.3, 0.2, 0.1, 0.02, 0.3, 0.1, 20, 8, 0.05,
                                0.1, 0.05, 0.02, 1, 0.05, 0.9, 0.3, 0.01],
     "text": "Safe trajectory with smooth steering and low lateral deviation throughout."},
    {"label": 0, "features": [0.6, 0.5, 0.3, 0.15, 0.03, 0.5, 0.2, 30, 12, 0.08,
                                0.15, 0.08, 0.03, 1, 0.06, 0.88, 0.5, 0.02],
     "text": "Normal urban driving - moderate speed, minimal heading variance, safe lateral margins."},
    {"label": 0, "features": [0.3, 0.25, 0.15, 0.08, 0.01, 0.2, 0.08, 15, 5, 0.03,
                                0.08, 0.04, 0.01, 0, 0.03, 0.92, 0.2, 0.008],
     "text": "Low-speed safe maneuver with stable heading and negligible oscillation."},
    # Sharp Turn
    {"label": 1, "features": [0.8, 0.6, 1.5, 0.8, 0.4, 1.2, 0.6, 25, 10, 0.15,
                                0.3, 0.2, 0.1, 3, 0.4, 0.7, 0.5, 0.08],
     "text": "Sharp right turn detected - heading change exceeds 1.2 rad, consistent with aggressive cornering."},
    {"label": 1, "features": [0.7, 0.55, 1.8, 1.0, 0.55, 1.5, 0.7, 22, 9, 0.18,
                                0.25, 0.18, 0.08, 4, 0.5, 0.65, 0.4, 0.1],
     "text": "Abrupt lane-change in a short horizon suggests evasive maneuver or sharp turn."},
    {"label": 1, "features": [0.9, 0.7, 2.0, 1.1, 0.6, 1.8, 0.9, 28, 11, 0.2,
                                0.35, 0.25, 0.12, 5, 0.6, 0.6, 0.6, 0.12],
     "text": "Large heading change in a short horizon suggests a sharp turn or evasive maneuver."},
    # Oscillatory Motion
    {"label": 2, "features": [0.6, 0.5, 0.9, 0.5, 0.6, 0.8, 0.4, 28, 6, 0.45,
                                0.2, 0.15, 0.1, 6, 0.3, 0.55, 0.4, 0.05],
     "text": "Predicted trajectory oscillation can indicate uncertain or unsafe motion planning."},
    {"label": 2, "features": [0.5, 0.4, 0.7, 0.4, 0.55, 0.7, 0.35, 22, 5, 0.5,
                                0.18, 0.12, 0.09, 7, 0.28, 0.5, 0.35, 0.04],
     "text": "Lateral sway pattern with high heading variance suggests oscillatory motion behavior."},
    {"label": 2, "features": [0.7, 0.55, 1.0, 0.6, 0.65, 0.9, 0.45, 30, 7, 0.55,
                                0.22, 0.16, 0.11, 8, 0.32, 0.52, 0.45, 0.06],
     "text": "Frequent direction changes with high oscillation score suggest unstable steering pattern."},
    # High-Speed Risk
    {"label": 3, "features": [2.2, 1.8, 0.5, 0.25, 0.08, 0.8, 0.4, 60, 22, 0.12,
                                0.6, 0.3, 0.2, 2, 0.5, 0.85, 1.9, 0.03],
     "text": "High-speed trajectory detected - velocity exceeds safe urban driving threshold."},
    {"label": 3, "features": [2.5, 2.0, 0.6, 0.3, 0.1, 1.0, 0.5, 70, 26, 0.14,
                                0.7, 0.35, 0.22, 2, 0.55, 0.83, 2.1, 0.04],
     "text": "Sustained high-speed motion with moderate heading change poses collision risk."},
    {"label": 3, "features": [2.0, 1.6, 0.4, 0.2, 0.07, 0.7, 0.35, 55, 20, 0.1,
                                0.5, 0.28, 0.18, 1, 0.45, 0.87, 1.7, 0.025],
     "text": "Aggressive acceleration pattern with high terminal speed observed in predicted trajectory."},
    # Near-Collision Risk
    {"label": 4, "features": [2.0, 1.6, 0.8, 0.4, 0.15, 2.8, 1.4, 55, 20, 0.2,
                                0.55, 0.3, 0.2, 3, 0.9, 0.75, 1.6, 0.05],
     "text": "Near-collision risk: high speed combined with large lateral deviation from expected path."},
    {"label": 4, "features": [2.3, 1.8, 0.9, 0.45, 0.18, 3.2, 1.6, 62, 23, 0.22,
                                0.6, 0.32, 0.22, 3, 1.0, 0.72, 1.8, 0.06],
     "text": "Abrupt lateral deviation at high speed may indicate unsafe lane-change or obstacle avoidance."},
    {"label": 4, "features": [1.9, 1.5, 0.7, 0.35, 0.13, 2.6, 1.3, 50, 19, 0.18,
                                0.5, 0.28, 0.19, 2, 0.85, 0.76, 1.5, 0.045],
     "text": "Combined high-speed and large lateral excursion pattern consistent with near-collision scenario."},
]

KB_FEATURES = np.array([e["features"] for e in KNOWLEDGE_BASE], dtype=np.float32)
KB_NORM     = KB_FEATURES / (np.linalg.norm(KB_FEATURES, axis=1, keepdims=True) + 1e-8)


def retrieve_cases(features: dict, top_k: int = 3):
    vec = np.array([features.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    sims = (KB_NORM @ vec_norm).tolist()
    ranked = sorted(zip(sims, KNOWLEDGE_BASE), key=lambda x: -x[0])
    return [(round(s, 4), e["text"]) for s, e in ranked[:top_k]]


def make_diagnosis(event_name: str, retrieved: list) -> str:
    texts = " | ".join(t for _, t in retrieved)
    return (
        f"Predicted safety event detected. Similar prior cases suggest: {texts}."
    )


# ── helpers ───────────────────────────────────────────────────────────────────
class TrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

def min_ade(pk, gt):
    ge = gt.unsqueeze(1).expand_as(pk)
    return torch.norm(pk - ge, dim=-1).mean(dim=-1).min(dim=1).values.mean().item()

def min_fde(pk, gt):
    ge = gt.unsqueeze(1).expand_as(pk)
    return torch.norm(pk[:,:,-1] - ge[:,:,-1], dim=-1).min(dim=1).values.mean().item()

def wta_loss(pk, gt):
    ge = gt.unsqueeze(1).expand_as(pk)
    best = torch.norm(pk - ge, dim=-1).mean(dim=-1).argmin(dim=1)
    B,K,T,C = pk.shape
    idx = best.view(B,1,1,1).expand(B,1,T,C)
    return nn.functional.huber_loss(pk.gather(1,idx).squeeze(1), gt, delta=1.0)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AV Safety Forecasting  - Terminal Demo")
    print("  Argoverse 2  |  TrajectoryTransformer  |  DATA 612")
    print("=" * 60)
    print()

    # -load & slice data ----------------------------------------------------
    data_dir = ROOT_DIR / "data" / "processed"
    X_tr_f = np.load(data_dir / "X_train.npy")
    Y_tr_f = np.load(data_dir / "Y_train.npy")
    X_va_f = np.load(data_dir / "X_val.npy")
    Y_va_f = np.load(data_dir / "Y_val.npy")
    X_te_f = np.load(data_dir / "X_test.npy")
    Y_te_f = np.load(data_dir / "Y_test.npy")

    rng = np.random.RandomState(SEED)
    ti = rng.choice(len(X_tr_f), N_TRAIN, replace=False)
    vi = rng.choice(len(X_va_f), N_VAL,   replace=False)
    ei = rng.choice(len(X_te_f), N_TEST,  replace=False)

    X_tr, Y_tr = X_tr_f[ti], Y_tr_f[ti]
    X_va, Y_va = X_va_f[vi], Y_va_f[vi]
    X_te, Y_te = X_te_f[ei], Y_te_f[ei]

    Xm = X_tr.mean(axis=(0,1), keepdims=True).astype(np.float32)
    Xs = (X_tr.std(axis=(0,1), keepdims=True) + 1e-8).astype(np.float32)
    Ym = Y_tr.mean(axis=(0,1), keepdims=True).astype(np.float32)
    Ys = (Y_tr.std(axis=(0,1), keepdims=True) + 1e-8).astype(np.float32)

    Xtrn = (X_tr - Xm)/Xs; Ytrn = (Y_tr - Ym)/Ys
    Xvan = (X_va - Xm)/Xs; Yvan = (Y_va - Ym)/Ys
    Xten = (X_te - Xm)/Xs; Yten = (Y_te - Ym)/Ys

    feat_dim = Xtrn.shape[2]; pred_len = Ytrn.shape[1]
    g = torch.Generator(); g.manual_seed(SEED)
    tr_ld = DataLoader(TrajectoryDataset(Xtrn, Ytrn), BATCH_SIZE, shuffle=True,  num_workers=0, generator=g)
    va_ld = DataLoader(TrajectoryDataset(Xvan, Yvan), BATCH_SIZE, shuffle=False, num_workers=0)
    te_ld = DataLoader(TrajectoryDataset(Xten, Yten), BATCH_SIZE, shuffle=False, num_workers=0)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = TrajectoryTransformer(
        input_dim=feat_dim, pred_len=pred_len,
        d_model=_cfg.D_MODEL, nhead=_cfg.NHEAD,
        num_layers=_cfg.NUM_LAYERS, dim_feedforward=_cfg.DIM_FEEDFORWARD,
        dropout=_cfg.DROPOUT, num_modes=_cfg.NUM_MODES,
    ).to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=_cfg.LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", _cfg.LR_FACTOR, _cfg.LR_PATIENCE, min_lr=_cfg.MIN_LR)
    scaler = GradScaler("cuda", enabled=use_amp)
    Ym_t = torch.tensor(Ym, dtype=torch.float32)
    Ys_t = torch.tensor(Ys, dtype=torch.float32)

    # -training loop --------------------------------------------------------
    best_ade = float("inf")
    ckpt = OUT_DIR / "demo_model.pt"
    history = []

    for ep in range(N_EPOCHS):
        model.train(); tr_loss = 0.0
        for Xb, Yb in tr_ld:
            Xb, Yb = Xb.to(device), Yb.to(device)
            opt.zero_grad()
            with autocast("cuda", enabled=use_amp):
                pk, _ = model(Xb)
                loss = wta_loss(pk, Yb)
            if torch.isnan(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tr_loss += loss.item()

        model.eval(); ap, ag = [], []
        with torch.no_grad():
            for Xb, Yb in va_ld:
                Xb = Xb.to(device)
                with autocast("cuda", enabled=use_amp):
                    pk, _ = model(Xb)
                ap.append(pk.float().cpu() * Ys_t + Ym_t)
                ag.append(Yb * Ys_t + Ym_t)
        ap = torch.cat(ap); ag = torch.cat(ag)
        va = min_ade(ap, ag); vf = min_fde(ap, ag)
        sched.step(va)

        print(f"Epoch {ep+1:2d}: train_loss={tr_loss:.4f}, val_ADE={va:.4f}, val_FDE={vf:.4f}")
        history.append({"epoch": ep+1, "train_loss": round(tr_loss,4), "val_ADE": round(va,4), "val_FDE": round(vf,4)})

        if not np.isnan(va) and va < best_ade:
            best_ade = va
            torch.save(model.state_dict(), ckpt)

    # -test metrics ---------------------------------------------------------
    model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()

    # use pre-trained model for inference if available
    pt_ckpt = ROOT_DIR / "outputs" / "checkpoints" / "best_model.pt"
    if pt_ckpt.exists():
        pt_Xm = np.load(ROOT_DIR / "outputs" / "X_mean.npy")
        pt_Xs = np.load(ROOT_DIR / "outputs" / "X_std.npy")
        pt_Ym = np.load(ROOT_DIR / "outputs" / "Y_mean.npy")
        pt_Ys = np.load(ROOT_DIR / "outputs" / "Y_std.npy")
        Xte_inf = (X_te - pt_Xm) / pt_Xs
        inf_Ym_t = torch.tensor(pt_Ym, dtype=torch.float32)
        inf_Ys_t = torch.tensor(pt_Ys, dtype=torch.float32)
        inf_model = TrajectoryTransformer(
            input_dim=feat_dim, pred_len=pred_len,
            d_model=_cfg.D_MODEL, nhead=_cfg.NHEAD,
            num_layers=_cfg.NUM_LAYERS, dim_feedforward=_cfg.DIM_FEEDFORWARD,
            dropout=_cfg.DROPOUT, num_modes=_cfg.NUM_MODES,
        ).to(device)
        inf_model.load_state_dict(torch.load(pt_ckpt, map_location=device))
        inf_model.eval()
    else:
        Xte_inf = Xten
        inf_Ym_t = Ym_t; inf_Ys_t = Ys_t
        inf_model = model

    # test ADE/FDE from demo model
    tp, tg = [], []
    with torch.no_grad():
        for Xb, Yb in te_ld:
            Xb = Xb.to(device)
            with autocast("cuda", enabled=use_amp):
                pk, _ = model(Xb)
            tp.append(pk.float().cpu() * Ys_t + Ym_t)
            tg.append(Yb * Ys_t + Ym_t)
    tp = torch.cat(tp); tg = torch.cat(tg)
    test_ade = min_ade(tp, tg); test_fde = min_fde(tp, tg)

    # safety classifier
    clf = SafetyClassifier()
    clf.fit(Y_tr)

    # safety metrics on test set using pre-trained model predictions
    pred_labels, gt_labels = [], []
    te_ld2 = DataLoader(TrajectoryDataset(Xte_inf, Yten), BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        for Xb, Yb in te_ld2:
            Xb = Xb.to(device)
            with autocast("cuda", enabled=use_amp):
                pk, confs = inf_model(Xb)
            pk_m = pk.float().cpu() * inf_Ys_t + inf_Ym_t
            Yb_m = Yb * inf_Ys_t + inf_Ym_t
            best_k = confs.argmax(dim=1)
            for i in range(len(pk_m)):
                pt = pk_m[i, best_k[i]].numpy()
                gt = Yb_m[i].numpy()
                if np.isnan(pt).any() or np.isnan(gt).any(): continue
                lp, _, _ = clf.predict(pt)
                lg = _derive_label(extract_features(gt))
                pred_labels.append(lp); gt_labels.append(lg)

    p, r, f, _ = precision_recall_fscore_support(
        gt_labels, pred_labels, average="macro", zero_division=0
    )

    print(f"\nFinal Results: {{'ADE': {test_ade:.4f}, 'FDE': {test_fde:.4f}, "
          f"'precision': {p:.2f}, 'recall': {r:.2f}, 'f1': {f:.2f}}}")

    # -demo pipeline --------------------------------------------------------
    print()
    print(f"python src/demo_terminal.py")
    print()

    for i in range(min(DEMO_SAMPLES, len(Xte_inf))):
        x = torch.tensor(Xte_inf[i:i+1], dtype=torch.float32).to(device)
        with torch.no_grad():
            pk, confs, _ = inf_model(x, return_attention=True)
        pk_m = (pk.float().cpu() * inf_Ys_t + inf_Ym_t)[0]
        best_k = confs[0].argmax().item()
        pred_traj = pk_m[best_k].numpy()
        gt_traj   = Y_te[i]

        label, label_name, features = clf.predict(pred_traj)
        gt_label = _derive_label(extract_features(gt_traj))

        retrieved = retrieve_cases(features, top_k=3)
        diagnosis = make_diagnosis(label_name, retrieved)

        print(f"Predicted safety label: {label} ({label_name})")
        print(f"Ground truth label    : {gt_label} ({EVENT_NAMES[gt_label]})")
        print()
        print("Retrieved cases:")
        for sim, text in retrieved:
            print(f"  {sim:.4f} -- {text}")
        print()
        print("Diagnosis:")
        # wrap at 80 chars
        words = diagnosis.split()
        line = ""; lines = []
        for w in words:
            if len(line) + len(w) + 1 > 78:
                lines.append(line); line = w
            else:
                line = (line + " " + w).strip()
        if line: lines.append(line)
        for l in lines: print(l)
        print()

    print("=" * 60)
    print(f"  Demo complete. Outputs: {OUT_DIR}")
    print("=" * 60)

    with open(OUT_DIR / "terminal_demo_results.json", "w") as fp:
        json.dump({"history": history, "test_ADE": round(test_ade,4),
                   "test_FDE": round(test_fde,4),
                   "precision": round(p,4), "recall": round(r,4), "f1": round(f,4)}, fp, indent=2)


if __name__ == "__main__":
    main()
