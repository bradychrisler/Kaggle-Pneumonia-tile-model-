"""
Generate predictions using ensemble of fold models.

Usage:
    python scripts/inference.py --model_dir outputs/20251124_141839 --test_csv data/processed/test_metadata.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PneumoniaDataset, valid_transforms


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    # Import build_model from train.py
    from train import build_model
    
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def predict(model, loader, device: torch.device) -> np.ndarray:
    """Generate predictions for a dataloader."""
    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def find_fold_models(model_dir: Path) -> List[Path]:
    """Find all fold model directories."""
    fold_dirs = []
    for item in model_dir.iterdir():
        if item.is_dir() and "fold" in item.name.lower():
            best_model = item / "best_model.pth"
            if best_model.exists():
                fold_dirs.append(item)
    return sorted(fold_dirs)


def main():
    parser = argparse.ArgumentParser(description="Generate ensemble predictions.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with fold models")
    parser.add_argument("--test_csv", type=str, required=True, help="Test metadata CSV")
    parser.add_argument("--image_root", type=str, default=".", help="Root directory for images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find all fold models - look for best_model.pth in subdirectories
    best_models = list(model_dir.glob("*/best_model.pth"))
    if not best_models:
        # Try recursive search
        best_models = list(model_dir.glob("**/best_model.pth"))
    
    if not best_models:
        # Single directory with a model
        if (model_dir / "best_model.pth").exists():
            fold_dirs = [model_dir]
        else:
            raise ValueError(f"No best_model.pth files found in {model_dir} or subdirectories")
    else:
        # Get the 5 most recent fold models (assuming they're timestamped)
        fold_dirs = [m.parent for m in sorted(best_models, key=lambda x: x.stat().st_mtime, reverse=True)[:5]]
        print(f"Using {len(fold_dirs)} most recent models:")
        for d in fold_dirs:
            print(f"  - {d.name}")
    
    print(f"Found {len(fold_dirs)} fold models")
    
    # Load test data
    test_df = pd.read_csv(args.test_csv)
    image_root = Path(args.image_root)
    
    # Create dataset and loader
    # For test data, use a dummy label column if opacity doesn't exist
    if "opacity" not in test_df.columns:
        test_df["opacity"] = 0.0  # Dummy labels for test set
    
    test_transforms = valid_transforms(224)
    test_ds = PneumoniaDataset(test_df, image_root, test_transforms, label_column="opacity")
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Ensemble predictions
    all_predictions = []
    for fold_dir in fold_dirs:
        best_model_path = fold_dir / "best_model.pth"
        print(f"Loading model from {best_model_path}")
        model = load_model(best_model_path, device)
        preds = predict(model, test_loader, device)
        all_predictions.append(preds)
    
    # Average predictions
    ensemble_preds = np.mean(all_predictions, axis=0)
    
    # Create submission
    submission_df = pd.DataFrame({
        "tile_id": test_df["tile_id"],
        "opacity": ensemble_preds.flatten(),
    })
    
    submission_df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
    print(f"Prediction range: [{ensemble_preds.min():.4f}, {ensemble_preds.max():.4f}]")


if __name__ == "__main__":
    main()

