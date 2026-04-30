# An Empirical Study of Transformer-Based Trajectory Prediction for Autonomous Vehicle Safety Forecasting

A deep learning pipeline for autonomous vehicle trajectory forecasting on [Argoverse 2](https://www.argoverse.org/av2.html), extended with a weak-supervision safety event classifier. Built as a capstone project for DATA 612 Deep Learning at the University of Maryland.

---

## Overview

The system has two components:

**1. Trajectory Forecaster**
A Transformer encoder observes 5 seconds of an agent's motion history (50 timesteps at 10 Hz) and predicts 6 candidate future trajectories over the next 6 seconds (60 timesteps). The most confident trajectory is selected at inference time using a learned confidence head.

**2. Safety Classifier**
A Random Forest trained on 18 hand-crafted kinematic features extracted from the predicted trajectory assigns one of five safety labels: Safe, Sharp Turn, Oscillatory Motion, High-Speed Risk, or Near-Collision Risk. Labels are derived via rule-based weak supervision with no manual annotation.

---

## Architecture

### TrajectoryTransformer

```
Input:  [batch, 50, 6]   (dx, dy, vx, vy, sin_heading, cos_heading per timestep)
         |
  Linear projection     input_dim -> d_model (256)
         |
  Sinusoidal positional encoding
         |
  TransformerEncoder
    4 layers
    d_model = 256
    nhead   = 4
    FFN dim = 512
    dropout = 0.1
         |
  Mean-pool across 50 timesteps  -> [batch, 256]
         |
       /       \
  Mode head    Confidence head
  Linear(256, 256)   Linear(256, 6)
  ReLU
  Linear(256, 6 x 60 x 2)
       |                |
  [batch, 6, 60, 2]  [batch, 6]
  (6 candidate         (mode
   trajectories)        logits)
```

**Key design choices:**

- **Relative displacement encoding** Input features use per-step (dx, dy) displacements rather than absolute positions, making the representation translation-invariant. Sin/cos heading replaces raw heading angle to remove the +/-pi discontinuity.
- **Mean pooling** over all 50 encoder hidden states (not just the last token) preserves the full temporal context of the observed history.
- **Multimodal output (K=6)** predicts a distribution over plausible futures. Training uses Winner-Takes-All (WTA) loss: only the mode closest to the ground-truth receives gradient, encouraging mode diversity.
- **Huber loss (delta=1.0)** replaces MSE in the WTA objective, reducing sensitivity to isolated outlier timesteps and lowering predicted-trajectory jerk.
- **Confidence head** learns to rank modes without oracle knowledge at inference. Mode selection uses `confs.argmax()`, not `mode_ade.argmin()`.
- **Attention extraction** A separate `MultiheadAttention` layer on the encoder output provides interpretable attention maps showing which past timesteps drove the prediction.

### SafetyClassifier

```
Predicted trajectory  [60, 2]
         |
  Feature extraction (18 kinematic features)
  - max_speed, mean_speed, speed_at_end
  - max_heading_change, heading_variance, mean_heading_change
  - max_lateral_dev, oscillation_score
  - turning_direction_changes (top RF feature, importance 0.16)
  - max_jerk, max_acceleration, max_deceleration
  - lateral_accel_max, mean_curvature
  - path_efficiency, total_distance, final_displacement
         |
  StandardScaler
         |
  RandomForestClassifier
    n_estimators  = 300
    class_weight  = "balanced_subsample"
    n_jobs        = -1
         |
  Safety label  +  class probabilities [5]
```

**Weak supervision:** GT labels for training are derived via rule-based thresholds on the same 18 features applied to ground-truth trajectories, making annotation cost zero. Classifier labels and GT labels are kept independent to avoid circular evaluation.

---

## Results

Evaluated on a held-out test split (4,998 scenarios from Argoverse 2 val).

### Trajectory Prediction

| Metric | Value |
|--------|-------|
| minADE (m) | 1.44 |
| minFDE (m) | 3.28 |

### Safety Classification (test split)

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| Safe | 0.95 | 0.48 | 0.64 | 3,669 |
| Sharp Turn | 0.05 | 0.19 | 0.09 | 300 |
| Oscillatory Motion | 0.38 | 0.81 | 0.52 | 879 |
| High-Speed Risk | 0.48 | 0.62 | 0.54 | 144 |
| Near-Collision Risk | 0.07 | 0.17 | 0.10 | 6 |
| **Macro avg** | **0.39** | **0.46** | **0.38** | **4,998** |

Confusion matrices, PR curves, calibration plots, and feature importance charts are in `outputs/final_eval/` and `outputs/safety_eval/`.

---

## Repository Structure

```
av-safety-forecasting/
  configs/
    train_config.py          # Transformer hyperparameters and training settings
    safety_config.py         # Safety classifier thresholds and RF settings
  src/
    preprocess_full.py       # Argoverse 2 preprocessing -> [N, 50, 6] arrays
    model.py                 # TrajectoryTransformer architecture
    train.py                 # Training loop: WTA loss, AMP, early stopping
    safety.py                # 18-feature extraction + SafetyClassifier
    evaluate.py              # Trajectory evaluation (minADE / minFDE)
    evaluate_safety_full.py  # Full safety eval with calibration plots
    run_experiment.py        # Timestamped experiment snapshots
    compare_feature_distributions.py  # GT vs predicted feature drift analysis
    demo_pipeline.py         # End-to-end inference demo
    retrieval.py             # Nearest-neighbor scenario retrieval
    refit_safety.py          # Standalone safety classifier refit utility
    train_clf_on_predictions.py  # Experiment: train classifier on predicted trajs
  outputs/
    checkpoints/
      best_model.pt                   # Trained transformer weights
      best_model_5feat_baseline.pt    # Verified baseline checkpoint (backup)
      safety_clf.pkl                  # Trained Random Forest
    X_mean.npy / X_std.npy           # Input normalization stats
    Y_mean.npy / Y_std.npy           # Output normalization stats
    final_eval/                       # Final evaluation outputs
    safety_eval/                      # Safety classifier evaluation plots
    attention/                        # Attention map visualizations
    results.json                      # Aggregated results
  requirements.txt
  generate_report.py
```

---

## Setup

**Requirements:** Python 3.10+, CUDA 12.1 (tested on RTX 4060 8 GB), ~32 GB RAM for preprocessing.

```bash
git clone https://github.com/abhioriganti/av-safety-forecasting.git
cd av-safety-forecasting

pip install -r requirements.txt

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Data

Download the [Argoverse 2 Motion Forecasting](https://www.argoverse.org/av2.html) dataset (train + val splits) and extract to:

```
Downloads/
  train/train/   # ~200K scenario folders
  val/val/       # ~25K scenario folders
```

Run preprocessing (~65 minutes on a modern CPU):

```bash
python src/preprocess_full.py \
  --train_dir "path/to/train/train" \
  --val_dir   "path/to/val/val"
```

Output: `data/processed/` with `X_train.npy` [199908, 50, 6], `Y_train.npy` [199908, 60, 2], and corresponding val/test splits.

---

## Training

```bash
# Train transformer + refit safety classifier (~35-40 min on RTX 4060)
python src/train.py

# Save a timestamped evaluation snapshot
python src/run_experiment.py my_label
```

Best checkpoint saved to `outputs/checkpoints/best_model.pt`.

---

## Evaluation

```bash
# Full safety eval: confusion matrices, PR curves, calibration plots
python src/evaluate_safety_full.py

# Analyze feature drift between GT and predicted trajectories
python src/compare_feature_distributions.py baseline
```

---

## Inference Demo

```bash
python src/demo_pipeline.py
```

Runs the full pipeline on a random val scenario and prints the predicted trajectory, selected safety class, and per-class confidence scores.

---

## Future Developments

### Trajectory Forecasting

**B4 - Autoregressive GRU decoder**
Replace the current mean-pool + linear head with a GRU that unrolls step by step. Each predicted position feeds back into the next step, producing physically consistent rollouts and eliminating the step-to-step oscillations that are the primary source of false unsafe-event predictions.

**B5 - Vectorized map and social context**
The current model uses only the ego agent's kinematic history. Encoding nearby lane centerlines and other agents' motion (e.g., via a vectorized map encoder or transformer cross-attention over neighbors) would give the model the spatial context it needs to predict lane-following and interaction-aware behavior.

**B6 - Scene-conditioned multimodality**
The K=6 modes currently differ only in output space. Adding a learned anchor system (clustering training trajectories into K prototype futures) would encourage each mode to represent a genuinely distinct maneuver rather than small perturbations of the same trajectory.

### Safety Classification

**C1 - Classifier calibration**
Precision is consistently high but recall on Safe is low (0.45), meaning the classifier over-fires on unsafe classes. Applying isotonic regression or Platt scaling after training should align predicted probabilities with true outcome frequencies.

**C2 - Rare-class augmentation**
Near-Collision has only 6 test samples and F1 = 0.00. Synthetic augmentation by perturbing GT near-collision trajectories, or SMOTE on the 18-feature vectors, would give the classifier meaningful signal for this critical class.

**C3 - Uncertainty-aware labeling**
The current weak-supervision rules assign hard labels. Replacing them with soft label distributions (e.g., label = 0.7 Oscillatory + 0.3 Safe near the threshold boundary) would reduce noise in the training signal for the Random Forest.

### System-Level

**LLM safety diagnosis**
`src/retrieval.py` contains a retrieval module that matches predicted scenarios against a library of annotated cases. Connecting this to a locally hosted Llama-3.2-3B-Instruct model would allow the system to generate natural-language explanations of detected safety events, making predictions interpretable for downstream operators.

**Real-time inference pipeline**
The current pipeline processes scenarios in batch. Refactoring to a streaming design with a fixed 100 ms budget per forward pass would make the system suitable for deployment in a live AV stack.

---

## Citation

Wilson et al., "Argoverse 2: Next Generation Acceleration of Research in Forecasting," NeurIPS 2021 Datasets and Benchmarks Track.
