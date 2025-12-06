"""
Generate ROC curves from validation sets for all trained folds.

Usage:
    python scripts/generate_roc_curves.py --output_dir outputs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PneumoniaDataset, build_dataloaders, valid_transforms
from train import build_model, load_config

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 10)


def get_predictions_and_labels(model, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions and get true labels from a dataloader."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
    
    return np.concatenate(all_probs, axis=0).flatten(), np.concatenate(all_labels, axis=0)


def plot_roc_curves(roc_data: List[dict], output_path: Path):
    """Plot ROC curves for all folds."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(roc_data)))
    
    for i, data in enumerate(roc_data):
        fpr = data["fpr"]
        tpr = data["tpr"]
        roc_auc = data["auc"]
        fold_name = data["fold_name"]
        
        ax.plot(fpr, tpr, 
                color=colors[i], 
                lw=2.5, 
                label=f'{fold_name} (AUC = {roc_auc:.4f})',
                alpha=0.8)
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 
            color='navy', 
            lw=2, 
            linestyle='--', 
            alpha=0.5,
            label='Random Classifier (AUC = 0.50)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves - Validation Sets (All Folds)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add text box with average AUC
    if roc_data:
        avg_auc = np.mean([d["auc"] for d in roc_data])
        std_auc = np.std([d["auc"] for d in roc_data])
        textstr = f'Average AUC: {avg_auc:.4f} ± {std_auc:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.6, 0.15, textstr, fontsize=12, 
                verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / "roc_curves.png", dpi=300, bbox_inches="tight")
    print(f"Saved ROC curves to {output_path / 'roc_curves.png'}")
    plt.close()


def plot_roc_comparison(roc_data: List[dict], output_path: Path):
    """Plot ROC curves with zoomed inset for better comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(roc_data)))
    
    for i, data in enumerate(roc_data):
        fpr = data["fpr"]
        tpr = data["tpr"]
        roc_auc = data["auc"]
        fold_name = data["fold_name"]
        
        # Full ROC curve
        ax1.plot(fpr, tpr, 
                color=colors[i], 
                lw=2.5, 
                label=f'{fold_name} (AUC = {roc_auc:.4f})',
                alpha=0.8)
        
        # Zoomed view (top-left corner)
        ax2.plot(fpr, tpr, 
                color=colors[i], 
                lw=2.5, 
                label=f'{fold_name} (AUC = {roc_auc:.4f})',
                alpha=0.8)
    
    # Diagonal line
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], 
                color='navy', 
                lw=2, 
                linestyle='--', 
                alpha=0.5,
                label='Random (AUC = 0.50)')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_title('ROC Curves - Full View', fontsize=14, fontweight='bold')
    
    # Zoomed view (focus on top-left where differences matter)
    ax2.set_xlim([0.0, 0.3])
    ax2.set_ylim([0.7, 1.0])
    ax2.set_title('ROC Curves - Zoomed View (High Performance Region)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / "roc_curves_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved ROC comparison to {output_path / 'roc_curves_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate ROC curves from validation sets.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory with training outputs")
    parser.add_argument("--save_dir", type=str, default="visualizations", help="Directory to save plots")
    parser.add_argument("--metadata_csv", type=str, default="data/processed/train_metadata.csv", help="Training metadata")
    parser.add_argument("--image_root", type=str, default=".", help="Root directory for images")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load metadata
    df = pd.read_csv(args.metadata_csv)
    if "fold" not in df.columns:
        print("No fold column found. Please ensure folds are created in metadata.")
        return
    
    # Find all training runs (get the 5 most recent EfficientNet runs)
    all_runs = sorted([d for d in output_dir.glob("*/") if (d / "best_model.pth").exists()], 
                      key=lambda x: x.stat().st_mtime, reverse=True)
    
    if len(all_runs) < 5:
        print(f"Found {len(all_runs)} model runs. Need at least 5 for fold comparison.")
        runs_to_use = all_runs
    else:
        # Use the 5 most recent (assuming these are the EfficientNet folds)
        runs_to_use = all_runs[:5]
    
    print(f"Processing {len(runs_to_use)} model runs...")
    
    roc_data = []
    
    for fold_idx, run_dir in enumerate(runs_to_use):
        best_model_path = run_dir / "best_model.pth"
        if not best_model_path.exists():
            continue
        
        print(f"\nProcessing {run_dir.name} (Fold {fold_idx + 1})...")
        
        # Load model
        checkpoint = torch.load(best_model_path, map_location=device)
        config = checkpoint["config"]
        model = build_model(config)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        
        # Get validation set for this fold
        fold = fold_idx  # Assuming folds are 0-4
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        
        if len(valid_df) == 0:
            print(f"  No validation data for fold {fold}, skipping...")
            continue
        
        # Create validation dataloader
        valid_transforms_fn = valid_transforms(config["data"]["image_size"])
        valid_ds = PneumoniaDataset(
            valid_df, 
            Path(args.image_root), 
            valid_transforms_fn, 
            label_column="opacity"
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )
        
        # Get predictions and labels
        print(f"  Generating predictions on {len(valid_df)} validation samples...")
        y_pred, y_true = get_predictions_and_labels(model, valid_loader, device)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        print(f"  Validation AUC: {roc_auc:.4f}")
        
        roc_data.append({
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc,
            "fold_name": f"Fold {fold_idx + 1}",
            "thresholds": thresholds
        })
    
    if not roc_data:
        print("No ROC data generated. Check that models and validation data are available.")
        return
    
    print(f"\nGenerating ROC curve plots...")
    plot_roc_curves(roc_data, save_dir)
    plot_roc_comparison(roc_data, save_dir)
    
    # Save summary statistics
    summary = {
        "folds": len(roc_data),
        "average_auc": float(np.mean([d["auc"] for d in roc_data])),
        "std_auc": float(np.std([d["auc"] for d in roc_data])),
        "min_auc": float(np.min([d["auc"] for d in roc_data])),
        "max_auc": float(np.max([d["auc"] for d in roc_data])),
        "per_fold_auc": {d["fold_name"]: float(d["auc"]) for d in roc_data}
    }
    
    with open(save_dir / "roc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nROC curves saved to {save_dir}/")
    print(f"Summary statistics:")
    print(f"  Average AUC: {summary['average_auc']:.4f} ± {summary['std_auc']:.4f}")
    print(f"  Range: [{summary['min_auc']:.4f}, {summary['max_auc']:.4f}]")


if __name__ == "__main__":
    main()








