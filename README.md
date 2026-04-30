# AV Trajectory Forecasting with Safety Classification

A deep learning pipeline for autonomous vehicle trajectory forecasting on [Argoverse 2](https://www.argoverse.org/av2.html), extended with a rule-based weak-supervision safety event classifier. Built as a capstone project for DATA 612 Deep Learning.

## Overview

The system has two components:

1. **Trajectory Forecaster** - A Transformer encoder that takes 5 seconds of observed motion (x, y, vx, vy, heading at 10 Hz) and predicts 6 candidate future trajectories over the next 6 seconds. The best trajectory is selected by confidence score at inference time.

2. **Safety Classifier** - A Random Forest trained on 18 hand-crafted kinematic features extracted from the predicted trajectory. It assigns one of five safety labels: Safe, Sharp Turn, Oscillatory Motion, High-Speed Risk, or Near-Collision Risk.

```
Observed history [N, 50, 5]
        |
  TrajectoryTransformer
  (d_model=128, 3 layers, 4 heads, K=6 modes)
        |
  Best predicted trajectory [N, 60, 2]
        |
  Feature extraction (18 kinematic features)
        |
  RandomForest (300 trees, balanced_subsample)
        |
  Safety label + class probabilities
```

## Results

All metrics evaluated on a held-out test split (4,998 scenarios from Argoverse 2 val).

**Trajectory prediction**

| Metric | Value |
|--------|-------|
| minADE (m) | 1.61 |
| minFDE (m) | 3.55 |

**Safety classification**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Safe | 0.96 | 0.45 | 0.61 | 3,669 |
| Sharp Turn | 0.07 | 0.23 | 0.10 | 300 |
| Oscillatory Motion | 0.37 | 0.81 | 0.51 | 879 |
| High-Speed Risk | 0.39 | 0.81 | 0.53 | 144 |
| Near-Collision Risk | 0.00 | 0.00 | 0.00 | 6 |
| **Macro avg** | **0.36** | **0.46** | **0.35** | **4,998** |

Confusion matrices, PR curves, and feature importance plots are in `outputs/final_eval/` and `outputs/safety_eval/`.

## Repository Structure

```
av_safety/
  configs/
    train_config.py          # Transformer hyperparameters and training settings
    safety_config.py         # Safety classifier thresholds and RF settings
  src/
    preprocess_full.py       # Argoverse 2 data preprocessing (outputs [N, 50, 5])
    model.py                 # TrajectoryTransformer architecture
    train.py                 # Training loop with WTA loss, AMP, early stopping
    safety.py                # 18-feature extraction + SafetyClassifier
    evaluate.py              # Trajectory-only evaluation (minADE/minFDE)
    evaluate_safety_full.py  # Full safety eval with calibration plots
    run_experiment.py        # Timestamped experiment snapshots
    compare_feature_distributions.py  # GT vs predicted feature drift analysis
    demo_pipeline.py         # End-to-end inference demo
    retrieval.py             # Nearest-neighbor scenario retrieval
    refit_safety.py          # Standalone safety classifier refit utility
  outputs/
    checkpoints/
      best_model.pt               # Trained transformer weights
      best_model_5feat_baseline.pt  # Backup of the verified baseline checkpoint
      safety_clf.pkl              # Trained Random Forest classifier
    X_mean.npy / X_std.npy       # Input normalization statistics
    Y_mean.npy / Y_std.npy       # Output normalization statistics
    final_eval/                  # Final baseline evaluation outputs
    safety_eval/                 # Safety classifier evaluation plots
    attention/                   # Attention map visualizations
    results.json                 # Aggregated results
  requirements.txt
  generate_report.py
```

## Setup

**Requirements:** Python 3.10+, CUDA 12.1 (tested on RTX 4060 8 GB), ~32 GB RAM for full dataset preprocessing.

```bash
# Clone the repo
git clone https://github.com/abhioriganti/av_safety.git
cd av_safety

# Install dependencies
pip install -r requirements.txt

# PyTorch with CUDA 12.1 (run this instead of the torch line in requirements.txt)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Data

Download the [Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html) dataset (train + val splits). The expected directory layout:

```
C:/Users/<you>/Downloads/
  train/train/   # ~200K scenario folders
  val/val/       # ~25K scenario folders
```

Then run preprocessing (outputs 5-feature arrays to `data/processed/`):

```bash
python src/preprocess_full.py \
  --train_dir "C:/path/to/train/train" \
  --val_dir   "C:/path/to/val/val"
```

## Training

```bash
# Train the trajectory forecaster and refit the safety classifier
python src/train.py

# Evaluate on val + test and save a timestamped snapshot
python src/run_experiment.py my_experiment_label
```

Training takes roughly 35-40 minutes on an RTX 4060. The best checkpoint is saved to `outputs/checkpoints/best_model.pt`.

## Inference Demo

```bash
python src/demo_pipeline.py
```

Runs the full pipeline on a random val scenario and prints the predicted trajectory, selected safety class, and confidence scores.

## Evaluation

```bash
# Full safety eval with confusion matrices, PR curves, calibration plots
python src/evaluate_safety_full.py

# Feature drift between GT and predicted trajectories
python src/compare_feature_distributions.py baseline
```

## Design Notes

**Weak supervision:** Safety labels are derived from 18 kinematic features using rule-based thresholds (no manual annotation). This keeps labeling cost at zero while providing enough signal for the classifier.

**Evaluation correctness:** GT labels use `_derive_label(extract_features(gt_traj))` directly, making them independent of the classifier. Mode selection at inference uses `confs.argmax()`, matching production behavior rather than the oracle `mode_ade.argmin()` used during training.

**Safety feature importance:** `turning_direction_changes` is the single most predictive feature (RF importance 0.16), reflecting that noisy step-to-step heading reversals in predicted trajectories are the main driver of false unsafe-event predictions.

## Citation

Dataset: Wilson et al., "Argoverse 2: Next Generation Acceleration of Research in Forecasting," NeurIPS 2021 Datasets and Benchmarks.
