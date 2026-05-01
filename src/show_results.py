"""
show_results.py -- Print saved training history + demo reports in terminal style.
Reads existing JSON files only -- no model loading, no GPU, instant output.

Usage:
    cd C:\\Users\\abhis\\projects\\av_safety
    python src/show_results.py
"""

import json, pathlib, textwrap, time
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent

# ── load saved outputs ────────────────────────────────────────────────────────
history     = json.loads((ROOT / "outputs" / "train_history.json").read_text())
results     = json.loads((ROOT / "outputs" / "results.json").read_text())
demo        = json.loads((ROOT / "outputs" / "demo_reports.json").read_text())

# ── knowledge base for retrieval (same as demo_terminal.py) ──────────────────
FEATURE_KEYS = [
    "max_speed", "mean_speed", "max_heading_change", "mean_heading_change",
    "heading_variance", "max_lateral_dev", "mean_lateral_dev",
    "total_distance", "final_displacement", "oscillation_score",
]

KB = [
    {"features": [0.4,0.3,0.2,0.1,0.02,0.3,0.1,20,8,0.05],
     "text": "Safe trajectory with smooth steering and low lateral deviation throughout."},
    {"features": [0.6,0.5,0.3,0.15,0.03,0.5,0.2,30,12,0.08],
     "text": "Normal urban driving at moderate speed with minimal heading variance, safe lateral margins."},
    {"features": [0.8,0.6,1.5,0.8,0.4,1.2,0.6,25,10,0.15],
     "text": "Sharp right turn detected -- heading change exceeds 1.2 rad, consistent with aggressive cornering."},
    {"features": [0.7,0.55,1.8,1.0,0.55,1.5,0.7,22,9,0.18],
     "text": "Abrupt lane-change in a short horizon suggests a sharp turn or evasive maneuver."},
    {"features": [0.9,0.7,2.0,1.1,0.6,1.8,0.9,28,11,0.2],
     "text": "Large heading change in a short horizon suggests a sharp turn or evasive maneuver."},
    {"features": [0.6,0.5,0.9,0.5,0.6,0.8,0.4,28,6,0.45],
     "text": "Predicted trajectory oscillation can indicate uncertain or unsafe motion planning."},
    {"features": [0.5,0.4,0.7,0.4,0.55,0.7,0.35,22,5,0.5],
     "text": "Lateral sway pattern with high heading variance suggests oscillatory motion behavior."},
    {"features": [2.2,1.8,0.5,0.25,0.08,0.8,0.4,60,22,0.12],
     "text": "High-speed trajectory detected -- velocity exceeds safe urban driving threshold."},
    {"features": [2.5,2.0,0.6,0.3,0.1,1.0,0.5,70,26,0.14],
     "text": "Sustained high-speed motion with moderate heading change poses collision risk."},
    {"features": [2.0,1.6,0.8,0.4,0.15,2.8,1.4,55,20,0.2],
     "text": "Near-collision risk: high speed combined with large lateral deviation from expected path."},
    {"features": [2.3,1.8,0.9,0.45,0.18,3.2,1.6,62,23,0.22],
     "text": "Abrupt lateral deviation at high speed may indicate unsafe lane-change or obstacle avoidance."},
]

KB_F = np.array([e["features"] for e in KB], dtype=np.float32)
KB_N = KB_F / (np.linalg.norm(KB_F, axis=1, keepdims=True) + 1e-8)

def retrieve(features, top_k=3):
    vec = np.array([features.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    sims = (KB_N @ vec).tolist()
    ranked = sorted(zip(sims, KB), key=lambda x: -x[0])
    return [(round(s, 4), e["text"]) for s, e in ranked[:top_k]]

# ── print training history (one epoch at a time with delay) ──────────────────
DELAY = 0.0   # seconds between epochs — set to 0.3 for live-replay effect

print()
for h in history:
    ade = h["minADE"] if h["minADE"] is not None else float("nan")
    fde = h["minFDE"] if h["minFDE"] is not None else float("nan")
    print(f"Epoch {h['epoch']:2d}: train_loss={h['train_loss']:.4f}, "
          f"val_ADE={ade:.4f}, val_FDE={fde:.4f}", flush=True)
    time.sleep(DELAY)

# ── final results line ────────────────────────────────────────────────────────
# parse macro precision/recall/f1 from the saved classification report string
import re
test_report = results["test_safety_report"]
macro_line  = [l for l in test_report.splitlines() if "macro avg" in l][0]
nums        = re.findall(r"\d+\.\d+", macro_line)
prec, rec, f1 = float(nums[0]), float(nums[1]), float(nums[2])

print(f"\nFinal Results: {{'ADE': {results['test_minADE']:.4f}, "
      f"'FDE': {results['test_minFDE']:.4f}, "
      f"'precision': {prec:.2f}, 'recall': {rec:.2f}, 'f1': {f1:.2f}}}")

# ── demo pipeline output ──────────────────────────────────────────────────────
print()
print("python src/demo_pipeline.py")
print()

CLASS_NAMES = ["Safe", "Sharp Turn", "Oscillatory Motion",
               "High-Speed Risk", "Near-Collision Risk"]

for entry in demo:
    label     = entry["event_label"]
    name      = entry["event_name"]
    features  = entry["features"]
    report    = entry["report"]
    probs     = entry["class_probs"]

    print(f"Predicted safety label: {label} ({name})")
    print()

    retrieved = retrieve(features)
    print("Retrieved cases:")
    for sim, text in retrieved:
        print(f"  {sim:.4f} -- {text}")
    print()

    # build diagnosis from retrieved cases
    texts    = " | ".join(t for _, t in retrieved)
    severity = report.get("severity", "N/A")
    action   = report.get("recommended_action", "")
    diag = (f"Predicted safety event detected. Similar prior cases suggest: {texts}. "
            f"Severity: {severity}. Recommended action: {action}")

    print("Diagnosis:")
    for line in textwrap.wrap(diag, width=80):
        print(line)
    print()
