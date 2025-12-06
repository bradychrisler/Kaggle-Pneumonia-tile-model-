"""
Train all folds for a given model configuration.

Usage:
    python scripts/train_all_folds.py --config configs/efficientnet_b0.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def main():
    parser = argparse.ArgumentParser(description="Train all folds for a model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds to train")
    args = parser.parse_args()

    config = load_config(args.config)
    n_folds = config["data"]["n_folds"]
    
    print(f"Training {n_folds} folds for {config['model']['name']}...")
    
    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"Training Fold {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Update config for this fold
        config["data"]["fold"] = fold
        
        # Save temporary config
        temp_config = Path("configs") / f"temp_fold_{fold}.yaml"
        with open(temp_config, "w", encoding="utf-8") as fp:
            yaml.dump(config, fp)
        
        # Run training
        cmd = [
            sys.executable,
            "train.py",
            "--config",
            str(temp_config),
        ]
        
        result = subprocess.run(cmd, check=False)
        
        # Clean up temp config
        temp_config.unlink()
        
        if result.returncode != 0:
            print(f"ERROR: Fold {fold} training failed!")
            sys.exit(1)
        
        print(f"Fold {fold + 1} completed successfully!")
    
    print(f"\n{'='*60}")
    print("All folds trained successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()








