"""
safety_config.py — Single source of truth for safety classifier hyperparameters.
"""

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── random forest ─────────────────────────────────────────────────────────────
N_ESTIMATORS  = 300
CLASS_WEIGHT  = "balanced_subsample"
N_JOBS        = -1
MAX_DEPTH     = None    # unlimited

# ── weak supervision thresholds (mirrors _derive_label in safety.py) ──────────
SPEED_HIGH       = 1.8   # m/step ≈ 65 km/h at 10 Hz
SPEED_NC_LATERAL = 2.5   # max_lateral_dev for Near-Collision
OSC_SCORE_THRESH = 0.3
HEADING_VAR_THRESH = 0.5
MHC_SHARP_THRESH = 1.2
