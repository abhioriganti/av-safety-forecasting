"""
preprocess_val_split.py — Split Argoverse 2 val folder into train/val/test numpy arrays.

Usage (from project root):
    python src/preprocess_val_split.py

Or with custom paths:
    python src/preprocess_val_split.py \
        --val_dir "C:/Users/abhis/Downloads/val/val" \
        --out_dir data/processed \
        --train_frac 0.70 \
        --val_frac 0.15

Output files:
    data/processed/X_train.npy   [N_train, 50, 5]
    data/processed/Y_train.npy   [N_train, 60, 2]
    data/processed/X_val.npy     [N_val,   50, 5]
    data/processed/Y_val.npy     [N_val,   60, 2]
    data/processed/X_test.npy    [N_test,  50, 5]
    data/processed/Y_test.npy    [N_test,  60, 2]
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

OBS_LEN  = 50
PRED_LEN = 60


def extract_focal_agent(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    focal_id = df["focal_track_id"].iloc[0]
    focal = (
        df[df["track_id"] == focal_id]
        .sort_values("timestep")
        .reset_index(drop=True)
    )

    if len(focal) < OBS_LEN + PRED_LEN:
        return None, None

    focal = focal.iloc[: OBS_LEN + PRED_LEN]

    # Normalize positions relative to the first observed timestep
    ox = focal["position_x"].values[0]
    oy = focal["position_y"].values[0]

    x       = focal["position_x"].values.astype(np.float32) - ox
    y       = focal["position_y"].values.astype(np.float32) - oy
    vx      = focal["velocity_x"].values.astype(np.float32)
    vy      = focal["velocity_y"].values.astype(np.float32)
    heading = focal["heading"].values.astype(np.float32)

    obs  = np.stack([x[:OBS_LEN],  y[:OBS_LEN],  vx[:OBS_LEN],  vy[:OBS_LEN],  heading[:OBS_LEN]],  axis=1)
    pred = np.stack([x[OBS_LEN:],  y[OBS_LEN:]],  axis=1)
    return obs, pred


def load_all_scenarios(val_dir: str, max_scenarios: int = None):
    val_path = Path(val_dir)
    scenario_dirs = sorted([d for d in val_path.iterdir() if d.is_dir()])

    if max_scenarios:
        scenario_dirs = scenario_dirs[:max_scenarios]

    X_list, Y_list = [], []
    skipped = 0
    total = len(scenario_dirs)

    print(f"Loading scenarios from: {val_dir}")
    print(f"Total directories found: {total}")

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

        if (i + 1) % 2000 == 0:
            print(f"  [{i+1:6d}/{total}]  loaded={len(X_list):6d}  skipped={skipped}")

    print(f"\nFinished: {len(X_list)} usable scenarios, {skipped} skipped.")
    X = np.stack(X_list, axis=0)  # [N, 50, 5]
    Y = np.stack(Y_list, axis=0)  # [N, 60, 2]
    return X, Y


def split_and_save(X: np.ndarray, Y: np.ndarray, out_dir: str,
                   train_frac: float, val_frac: float, seed: int = 42):
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val

    splits = {
        "train": idx[:n_train],
        "val":   idx[n_train : n_train + n_val],
        "test":  idx[n_train + n_val :],
    }

    os.makedirs(out_dir, exist_ok=True)

    print(f"\nSplit ({int(train_frac*100)}/{int(val_frac*100)}/{int((1-train_frac-val_frac)*100)}):")
    for split_name, split_idx in splits.items():
        X_s = X[split_idx]
        Y_s = Y[split_idx]
        np.save(os.path.join(out_dir, f"X_{split_name}.npy"), X_s)
        np.save(os.path.join(out_dir, f"Y_{split_name}.npy"), Y_s)
        size_mb = (X_s.nbytes + Y_s.nbytes) / 1e6
        print(f"  {split_name:5s}: {len(split_idx):6d} scenarios  X={X_s.shape}  Y={Y_s.shape}  ({size_mb:.1f} MB)")

    print(f"\nAll files saved to: {os.path.abspath(out_dir)}/")
    for f in sorted(os.listdir(out_dir)):
        mb = os.path.getsize(os.path.join(out_dir, f)) / 1e6
        print(f"  {f:<25s} {mb:6.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Preprocess AV2 val set into train/val/test splits.")
    parser.add_argument("--val_dir",    default=r"C:\Users\abhis\Downloads\val\val",
                        help="Path to the Argoverse 2 val directory containing UUID scenario folders.")
    parser.add_argument("--out_dir",    default="data/processed",
                        help="Output directory for .npy files.")
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac",   type=float, default=0.15)
    parser.add_argument("--max",        type=int,   default=None,
                        help="Cap total scenarios processed (e.g. --max 1000 for a quick test run).")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    assert args.train_frac + args.val_frac < 1.0, \
        f"train_frac ({args.train_frac}) + val_frac ({args.val_frac}) must be < 1.0"

    print("=" * 60)
    print("Argoverse 2 Val-Set Preprocessor")
    print("=" * 60)
    print(f"  Source : {args.val_dir}")
    print(f"  Output : {args.out_dir}")
    print(f"  Split  : {int(args.train_frac*100)}/{int(args.val_frac*100)}"
          f"/{int((1-args.train_frac-args.val_frac)*100)}  (train/val/test)")
    if args.max:
        print(f"  Cap    : {args.max} scenarios (quick-test mode)")
    print()

    X, Y = load_all_scenarios(args.val_dir, max_scenarios=args.max)
    split_and_save(X, Y, args.out_dir, args.train_frac, args.val_frac, seed=args.seed)

    print("\nDone! Next step:")
    print("  python src/train.py")


if __name__ == "__main__":
    main()
