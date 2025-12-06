"""
Format predictions for Kaggle submission.

Usage:
    python scripts/create_submission.py --predictions predictions.csv --output submission.csv
"""
from __future__ import annotations

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Format predictions for Kaggle.")
    parser.add_argument("--predictions", type=str, required=True, help="Predictions CSV")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV")
    parser.add_argument("--test_info", type=str, default="data/raw/test_info.csv", help="Test info CSV")
    args = parser.parse_args()

    # Load predictions
    pred_df = pd.read_csv(args.predictions)
    
    # Load test info to get slice_idx
    test_info = pd.read_csv(args.test_info)
    
    # Merge to get slice_idx
    submission = pred_df.merge(test_info[["tile_id", "slice_idx"]], on="tile_id", how="left")
    
    # Ensure opacity is in [0, 1] range
    submission["opacity"] = submission["opacity"].clip(0.0, 1.0)
    
    # Sort by slice_idx for consistency
    submission = submission.sort_values("slice_idx")
    
    # Select required columns
    submission = submission[["tile_id", "opacity"]]
    
    # Save
    submission.to_csv(args.output, index=False)
    print(f"Submission saved to {args.output}")
    print(f"Shape: {submission.shape}")
    print(f"Opacity range: [{submission['opacity'].min():.4f}, {submission['opacity'].max():.4f}]")


if __name__ == "__main__":
    main()








