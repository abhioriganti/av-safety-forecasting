"""
train_config.py — Single source of truth for trajectory model hyperparameters.
Import in train.py, refit_safety.py, and any experiment script.
"""

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── model architecture ────────────────────────────────────────────────────────
D_MODEL        = 256
NHEAD          = 4
NUM_LAYERS     = 4
DIM_FEEDFORWARD = 512
DROPOUT        = 0.1
NUM_MODES      = 6

# ── training ──────────────────────────────────────────────────────────────────
BATCH_SIZE          = 256
LEARNING_RATE       = 5e-4
NUM_EPOCHS          = 75
EARLY_STOP_PATIENCE = 15

# ── scheduler ────────────────────────────────────────────────────────────────
LR_FACTOR   = 0.5
LR_PATIENCE = 5
MIN_LR      = 1e-5

# ── data ──────────────────────────────────────────────────────────────────────
OBS_LEN  = 50
PRED_LEN = 60
