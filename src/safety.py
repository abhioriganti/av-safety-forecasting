"""
safety.py — Multi-class safety event classifier for AV trajectory forecasting.

Replaces the binary rule-based thresholds (Precision=0.24) with:
  1. A richer feature extraction step over predicted trajectories.
  2. A small MLP classifier trained on those features with labelled data
     derived from clear threshold rules (weak supervision bootstrap).

Event classes
-------------
  0 : Safe
  1 : Sharp Turn          (high heading change)
  2 : Oscillatory Motion  (high variance in direction)
  3 : High-Speed Risk     (sustained high speed)
  4 : Near-Collision Risk (high speed + large lateral deviation)

Usage
-----
  # At training time — fit the classifier on ground-truth trajectories
  from safety import SafetyClassifier
  clf = SafetyClassifier()
  clf.fit(Y_train)          # Y_train: [N, pred_len, 2]
  clf.save("outputs/safety_clf.pkl")

  # At inference time
  clf = SafetyClassifier.load("outputs/safety_clf.pkl")
  label, label_name, features = clf.predict(pred_traj)  # pred_traj: [pred_len, 2]
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ── human-readable class names ─────────────────────────────────────────────
EVENT_NAMES = {
    0: "Safe",
    1: "Sharp Turn",
    2: "Oscillatory Motion",
    3: "High-Speed Risk",
    4: "Near-Collision Risk",
}


# ── feature extraction ──────────────────────────────────────────────────────

FEATURE_KEYS = [
    "max_speed", "mean_speed",
    "max_heading_change", "mean_heading_change", "heading_variance",
    "max_lateral_dev", "mean_lateral_dev",
    "total_distance", "final_displacement", "oscillation_score",
    # Phase-1 additions
    "max_acceleration", "max_deceleration", "max_jerk",
    "turning_direction_changes", "lateral_accel_max",
    "path_efficiency", "speed_at_end", "mean_curvature",
]


def extract_features(traj: np.ndarray) -> dict:
    """
    Extract scalar safety-relevant features from a single trajectory.

    Args:
        traj: [pred_len, 2]  (x, y positions, already relative to last obs)

    Returns:
        dict of named float features
    """
    diffs = np.diff(traj, axis=0)                        # [T-1, 2]
    speeds = np.linalg.norm(diffs, axis=1)               # [T-1]
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])        # [T-1]
    heading_delta = np.abs(np.diff(angles))               # [T-2]

    # Lateral deviation: distance from straight line between first and last point
    start, end = traj[0], traj[-1]
    direction = end - start
    length = np.linalg.norm(direction) + 1e-8
    unit = direction / length
    perp = np.array([-unit[1], unit[0]])
    lateral_devs = np.abs((traj - start) @ perp)

    total_dist   = float(speeds.sum())
    final_disp   = float(np.linalg.norm(traj[-1] - traj[0]))

    # ── acceleration / jerk ─────────────────────────────────────────────────
    speed_diffs  = np.diff(speeds)                        # [T-2]
    max_accel    = float(np.maximum(speed_diffs, 0).max()) if len(speed_diffs) else 0.0
    max_decel    = float(np.maximum(-speed_diffs, 0).max()) if len(speed_diffs) else 0.0
    jerk         = np.abs(np.diff(speed_diffs))           # [T-3]
    max_jerk     = float(jerk.max()) if len(jerk) else 0.0

    # ── turning direction changes (oscillatory signature) ───────────────────
    signed_heading = np.diff(angles)                      # [T-2], signed
    sign_flips = (np.diff(np.sign(signed_heading)) != 0).sum() if len(signed_heading) > 1 else 0
    tdc = float(sign_flips / max(len(signed_heading) - 1, 1))

    # ── lateral acceleration (how fast deviation grows) ─────────────────────
    lat_accel_max = float(np.abs(np.diff(lateral_devs)).max()) if len(lateral_devs) > 1 else 0.0

    # ── path efficiency: 1.0 = straight line, 0 = extreme detour ───────────
    path_eff = final_disp / (total_dist + 1e-8)

    # ── speed at end of horizon ─────────────────────────────────────────────
    speed_at_end = float(speeds[-1]) if len(speeds) else 0.0

    # ── mean curvature (heading change per unit speed) ──────────────────────
    mean_curv = float((heading_delta / (speeds[1:] + 1e-8)).mean()) if len(heading_delta) else 0.0

    return {
        "max_speed":               float(speeds.max()),
        "mean_speed":              float(speeds.mean()),
        "max_heading_change":      float(heading_delta.max()) if len(heading_delta) else 0.0,
        "mean_heading_change":     float(heading_delta.mean()) if len(heading_delta) else 0.0,
        "heading_variance":        float(heading_delta.var()) if len(heading_delta) else 0.0,
        "max_lateral_dev":         float(lateral_devs.max()),
        "mean_lateral_dev":        float(lateral_devs.mean()),
        "total_distance":          total_dist,
        "final_displacement":      final_disp,
        "oscillation_score":       float(np.diff(heading_delta).std()) if len(heading_delta) > 1 else 0.0,
        # Phase-1 additions
        "max_acceleration":        max_accel,
        "max_deceleration":        max_decel,
        "max_jerk":                max_jerk,
        "turning_direction_changes": tdc,
        "lateral_accel_max":       lat_accel_max,
        "path_efficiency":         path_eff,
        "speed_at_end":            speed_at_end,
        "mean_curvature":          mean_curv,
    }


def _features_to_vector(features: dict) -> np.ndarray:
    return np.array([features[k] for k in FEATURE_KEYS], dtype=np.float32)


# ── weak supervision labels (bootstrap) ────────────────────────────────────

def _derive_label(features: dict) -> int:
    """
    Assign a multi-class label using interpretable thresholds.

    Priority: Near-Collision > High-Speed > Oscillatory > Sharp Turn > Safe.

    Speed thresholds use 10 Hz AV2 timestep (1 m/step ≈ 10 m/s ≈ 36 km/h):
      - 1.8 m/step ≈ 65 km/h  (arterial/highway threshold)
    Oscillatory is checked BEFORE Sharp Turn so zigzag ≠ single-turn.
    `turning_direction_changes` and other Phase-1 features are available
    to the RF as inputs but are not hard-coded here — the RF learns their
    optimal thresholds from data.
    """
    ms  = features["max_speed"]
    mhc = features["max_heading_change"]
    hv  = features["heading_variance"]
    mld = features["max_lateral_dev"]
    osc = features["oscillation_score"]

    if ms > 1.8 and mld > 2.5:
        return 4  # Near-Collision Risk: fast + large lateral swerve
    if ms > 1.8:
        return 3  # High-Speed Risk: > ~65 km/h
    if osc > 0.3 and hv > 0.5:
        return 2  # Oscillatory Motion: high oscillation AND high heading variance
    if mhc > 1.2:
        return 1  # Sharp Turn: single large heading deviation
    return 0      # Safe


# ── classifier class ────────────────────────────────────────────────────────

class SafetyClassifier:
    """
    Lightweight MLP classifier over trajectory features.
    Falls back gracefully to rule-based labels if not yet fitted.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",  # handles imbalance natively per tree
            random_state=42,
            n_jobs=-1,
        )
        self._fitted = False

    # ── fit ─────────────────────────────────────────────────────────────────
    def fit(self, trajectories: np.ndarray):
        """
        Fit the classifier using weak supervision labels.

        Args:
            trajectories: [N, pred_len, 2]
        """
        feature_list, label_list = [], []
        for traj in trajectories:
            feats = extract_features(traj)
            feature_list.append(_features_to_vector(feats))
            label_list.append(_derive_label(feats))

        X = np.stack(feature_list)
        y = np.array(label_list)

        counts = {EVENT_NAMES[i]: int((y == i).sum()) for i in range(5)}
        print(f"[SafetyClassifier] Label distribution: {counts}")

        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self._fitted = True
        print(f"[SafetyClassifier] Fitted on {len(y)} samples.")

    # ── predict ─────────────────────────────────────────────────────────────
    def predict(self, traj: np.ndarray):
        """
        Predict safety class for a single trajectory.

        Args:
            traj: [pred_len, 2]

        Returns:
            label (int), label_name (str), features (dict)
        """
        feats = extract_features(traj)
        if not self._fitted:
            label = _derive_label(feats)
        else:
            vec = _features_to_vector(feats).reshape(1, -1)
            vec_scaled = self.scaler.transform(vec)
            label = int(self.clf.predict(vec_scaled)[0])
        return label, EVENT_NAMES[label], feats

    def predict_proba(self, traj: np.ndarray) -> np.ndarray:
        """Return class probabilities [5,]."""
        feats = extract_features(traj)
        if not self._fitted:
            label = _derive_label(feats)
            proba = np.zeros(5, dtype=np.float32)
            proba[label] = 1.0
            return proba
        vec = _features_to_vector(feats).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        return self.clf.predict_proba(vec_scaled)[0]

    def predict_batch(self, trajs: list):
        """Batch predict for a list of trajectories. Returns (labels [N], probas [N,5])."""
        feat_vecs = np.stack([_features_to_vector(extract_features(t)) for t in trajs])
        if not self._fitted:
            labels = np.array([_derive_label(extract_features(t)) for t in trajs])
            probas = np.eye(5, dtype=np.float32)[labels]
            return labels, probas
        scaled = self.scaler.transform(feat_vecs)
        labels = self.clf.predict(scaled).astype(int)
        probas = self.clf.predict_proba(scaled).astype(np.float32)
        return labels, probas

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "clf": self.clf, "fitted": self._fitted}, f)
        print(f"[SafetyClassifier] Saved to {path}")

    @classmethod
    def load(cls, path: str):
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.scaler = data["scaler"]
        obj.clf = data["clf"]
        obj._fitted = data["fitted"]
        return obj


# ── legacy shim (keeps demo_pipeline.py working during transition) ──────────

def derive_safety_label(future_traj: np.ndarray) -> int:
    """Binary safe/unsafe label — kept for backward compatibility."""
    feats = extract_features(future_traj)
    return int(_derive_label(feats) > 0)
