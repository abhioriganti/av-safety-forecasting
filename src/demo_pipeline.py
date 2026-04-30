"""
demo_pipeline.py — End-to-end inference demo.

Full pipeline per val sample:
  1. Load TrajectoryTransformer  →  predict K=6 future trajectories (normalized input)
  2. Denormalize predictions back to meter space
  3. Pick most confident mode
  4. SafetyClassifier            →  event class + feature dict + class probabilities
  5. LLMReportGenerator          →  structured JSON diagnosis (Llama-3.2-3B)
  6. Print full structured report + attention interpretability summary
"""

import json
import numpy as np
import torch

from model import TrajectoryTransformer
from safety import SafetyClassifier
from retrieval import DiagnosisPipeline


def print_report(idx: int, label: int, label_name: str, features: dict,
                 result: dict, attn: np.ndarray):
    sep = "=" * 70
    report = result["report"]

    print(sep)
    print(f"Sample {idx}")
    print(f"  Event class  : [{label}] {label_name}")
    print(f"  Severity     : {report.get('severity', 'N/A')}")
    print(f"  Confidence   : {report.get('confidence', 'N/A')}")
    print()
    print("  Trajectory features:")
    print(f"    Max speed             : {features['max_speed']:.4f} m/s")
    print(f"    Max heading change    : {features['max_heading_change']:.4f} rad")
    print(f"    Max lateral deviation : {features['max_lateral_dev']:.4f} m")
    print(f"    Oscillation score     : {features['oscillation_score']:.4f}")
    print(f"    Final displacement    : {features['final_displacement']:.4f} m")
    print()
    print("  Diagnosis report:")
    print(f"    Primary indicator  : {report.get('primary_indicator', '')}")
    for s in report.get("secondary_indicators", []):
        print(f"    Secondary          : {s}")
    print(f"    Recommended action : {report.get('recommended_action', '')}")
    print()
    top3 = attn.argsort()[::-1][:3]
    print(f"  Interpretability — top-3 attended past timesteps: {list(top3)}")
    print()


def main():
    # ── load data and normalization stats ─────────────────────────────────────
    X_val   = np.load("data/processed/X_val.npy")
    Y_val   = np.load("data/processed/Y_val.npy")
    pred_len = Y_val.shape[1]
    feat_dim = X_val.shape[2]

    X_mean = np.load("outputs/X_mean.npy")
    X_std  = np.load("outputs/X_std.npy")
    Y_mean = np.load("outputs/Y_mean.npy")
    Y_std  = np.load("outputs/Y_std.npy")

    X_val_n = (X_val - X_mean) / X_std   # normalized inputs for the transformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── trajectory model ──────────────────────────────────────────────────────
    traj_model = TrajectoryTransformer(
        input_dim=feat_dim,
        pred_len=pred_len,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        num_modes=6,
    ).to(device)
    traj_model.load_state_dict(
        torch.load("outputs/checkpoints/best_model.pt", map_location=device)
    )
    traj_model.eval()
    print("Trajectory model loaded.")

    # ── safety classifier ─────────────────────────────────────────────────────
    clf = SafetyClassifier.load("outputs/checkpoints/safety_clf.pkl")
    print("Safety classifier loaded.")

    # ── LLM diagnosis pipeline (loads Llama-3.2-3B once) ─────────────────────
    diag = DiagnosisPipeline()
    print("LLM report generator ready.\n")

    # ── run on first 5 val samples ────────────────────────────────────────────
    all_reports = []
    NUM_SAMPLES = 5

    Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32)
    Y_std_t  = torch.tensor(Y_std,  dtype=torch.float32)

    for idx in range(min(NUM_SAMPLES, len(X_val_n))):
        x = torch.tensor(X_val_n[idx:idx+1], dtype=torch.float32).to(device)

        with torch.no_grad():
            preds_k, confidences, attn_weights = traj_model(x, return_attention=True)

        # Denormalize predictions back to meter space for safety classifier
        preds_k_m = preds_k.cpu().float() * Y_std_t + Y_mean_t   # [1, K, 60, 2]

        # Most confident mode
        best_mode = confidences[0].argmax().item()
        pred_traj = preds_k_m[0, best_mode].numpy()              # [pred_len, 2]

        # Safety classification + probabilities
        label, label_name, features = clf.predict(pred_traj)
        class_probs = clf.predict_proba(pred_traj).tolist()

        # LLM report
        result = diag.run(label_name, features, class_probs)

        # Attention summary (mean over query positions)
        attn = attn_weights[0].mean(dim=0).cpu().numpy()          # [obs_len]

        print_report(idx, label, label_name, features, result, attn)
        all_reports.append({
            "sample": idx,
            "event_label": label,
            "event_name": label_name,
            "features": features,
            "class_probs": class_probs,
            "report": result["report"],
        })

    with open("outputs/demo_reports.json", "w") as f:
        json.dump(all_reports, f, indent=2)
    print("All reports saved to outputs/demo_reports.json")


if __name__ == "__main__":
    main()
