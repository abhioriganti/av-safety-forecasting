"""
preprocess_full.py — Preprocess the full Argoverse 2 Motion Forecasting dataset.

Strategy
--------
- Train : all scenarios from the official AV2 train split (~200K scenarios, 48 GB)
- Val   : 80% of AV2 val split (~20K scenarios)
- Test  : 20% of AV2 val split (~5K scenarios)

The AV2 public test split does NOT include future trajectories, so we hold out
a portion of the val split for final evaluation instead.

Usage (from project root):
    python src/preprocess_full.py

Custom paths:
    python src/preprocess_full.py \
        --train_dir "C:/Users/abhis/Downloads/train/train" \
        --val_dir   "C:/Users/abhis/Downloads/val/val" \
        --out_dir   data/processed

Output files (same schema as preprocess_val_split.py):
    data/processed/X_train.npy   [N_train, 50, 5]
    data/processed/Y_train.npy   [N_train, 60, 2]
    data/processed/X_val.npy     [N_val,   50, 5]
    data/processed/Y_val.npy     [N_val,   60, 2]
    data/processed/X_test.npy    [N_test,  50, 5]
    data/processed/Y_test.npy    [N_test,  60, 2]
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

OBS_LEN  = 50
PRED_LEN = 60


def extract_focal_agent(parquet_path: str):
    """Extract observed + future trajectory for the focal agent."""
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return None, None

    focal_id = df["focal_track_id"].iloc[0]
    focal = (
        df[df["track_id"] == focal_id]
        .sort_values("timestep")
        .reset_index(drop=True)
    )

    if len(focal) < OBS_LEN + PRED_LEN:
        return None, None

    focal = focal.iloc[: OBS_LEN + PRED_LEN]

    ox = focal["position_x"].values[0]
    oy = focal["position_y"].values[0]

    x       = focal["position_x"].values.astype(np.float32) - ox
    y       = focal["position_y"].values.astype(np.float32) - oy
    vx      = focal["velocity_x"].values.astype(np.float32)
    vy      = focal["velocity_y"].values.astype(np.float32)
    heading = focal["heading"].values.astype(np.float32)

    obs  = np.stack([x[:OBS_LEN], y[:OBS_LEN], vx[:OBS_LEN], vy[:OBS_LEN],
                     heading[:OBS_LEN]], axis=1)
    pred = np.stack([x[OBS_LEN:], y[OBS_LEN:]], axis=1)
    return obs, pred


def load_split(split_dir: str, split_name: str, max_scenarios: int = None):
    """Load all usable scenarios from a directory. Returns X [N,50,5], Y [N,60,2]."""
    split_path = Path(split_dir)
    scenario_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])

    if max_scenarios:
        scenario_dirs = scenario_dirs[:max_scenarios]

    total = len(scenario_dirs)
    print(f"\nLoading {split_name} split: {split_dir}")
    print(f"  Scenario directories found : {total:,}")

    X_list, Y_list = [], []
    skipped = 0
    t0 = time.time()

    for i, scenario_dir in enumerate(scenario_dirs):
        parquet_files = list(scenario_dir.glob("*.parquet"))
        if not parquet_files:
            skipped += 1
            continue

        obs, pred = extract_focal_agent(str(parquet_files[0]))
        if obs is None:
            skipped += 1
            continue

        X_list.append(obs)
        Y_list.append(pred)

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta_s   = (total - i - 1) / rate
            print(f"  [{i+1:7d}/{total:7d}]  loaded={len(X_list):7,}  skipped={skipped:5,}"
                  f"  elapsed={elapsed/60:.1f}m  ETA={eta_s/60:.1f}m")

    elapsed = time.time() - t0
    print(f"  Done: {len(X_list):,} loaded, {skipped:,} skipped in {elapsed/60:.1f} min")

    if not X_list:
        return None, None

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y


def save_split(X: np.ndarray, Y: np.ndarray, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    xp = os.path.join(out_dir, f"X_{name}.npy")
    yp = os.path.join(out_dir, f"Y_{name}.npy")
    np.save(xp, X)
    np.save(yp, Y)
    mb = (X.nbytes + Y.nbytes) / 1e6
    print(f"  Saved {name:5s}: X={X.shape}  Y={Y.shape}  ({mb:.0f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=r"C:\Users\abhis\Downloads\train\train",
                        help="Path to AV2 train split (UUID scenario folders).")
    parser.add_argument("--val_dir",   default=r"C:\Users\abhis\Downloads\val\val",
                        help="Path to AV2 val split (UUID scenario folders).")
    parser.add_argument("--out_dir",   default="data/processed")
    parser.add_argument("--val_frac",  type=float, default=0.80,
                        help="Fraction of AV2 val used for model val (rest becomes test).")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--max_train", type=int, default=None,
                        help="Cap train scenarios (e.g. --max_train 5000 for a smoke test).")
    parser.add_argument("--max_val",   type=int, default=None,
                        help="Cap val scenarios.")
    args = parser.parse_args()

    print("=" * 65)
    print("Argoverse 2 Full Dataset Preprocessor")
    print("=" * 65)
    print(f"  Train dir : {args.train_dir}")
    print(f"  Val dir   : {args.val_dir}")
    print(f"  Output    : {args.out_dir}")
    print(f"  Val split : {int(args.val_frac*100)}/{int((1-args.val_frac)*100)} (val/test from AV2 val)")
    if args.max_train:
        print(f"  !! Train capped at {args.max_train} scenarios (smoke-test mode)")
    print()

    t_total = time.time()

    # ── 1. Train split ────────────────────────────────────────────────────────
    X_train, Y_train = load_split(args.train_dir, "train", max_scenarios=args.max_train)
    if X_train is None:
        print("\nERROR: No usable scenarios found in train_dir. Is the archive extracted?")
        print(f"  Expected: {args.train_dir}")
        return

    # ── 2. Val split → val + test ─────────────────────────────────────────────
    X_val_all, Y_val_all = load_split(args.val_dir, "val", max_scenarios=args.max_val)
    if X_val_all is None:
        print("\nERROR: No usable scenarios found in val_dir.")
        return

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X_val_all))
    n_val  = int(len(X_val_all) * args.val_frac)
    X_val  = X_val_all[idx[:n_val]]
    Y_val  = Y_val_all[idx[:n_val]]
    X_test = X_val_all[idx[n_val:]]
    Y_test = Y_val_all[idx[n_val:]]

    # ── 3. Save ───────────────────────────────────────────────────────────────
    print(f"\nSaving to {args.out_dir}/")
    save_split(X_train, Y_train, args.out_dir, "train")
    save_split(X_val,   Y_val,   args.out_dir, "val")
    save_split(X_test,  Y_test,  args.out_dir, "test")

    total_min = (time.time() - t_total) / 60
    print(f"\nTotal preprocessing time: {total_min:.1f} min")

    print("\nFile sizes:")
    for f in sorted(os.listdir(args.out_dir)):
        if f.endswith(".npy"):
            mb = os.path.getsize(os.path.join(args.out_dir, f)) / 1e6
            print(f"  {f:<25s} {mb:7.1f} MB")

    print("\nDone. Next step:")
    print("  python src/train.py")


if __name__ == "__main__":
    main()
