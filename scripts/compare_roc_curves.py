"""
Compare ROC curves between original and fine-tuned models on COVID-19 dataset.

This script:
1. Loads the original pre-trained model
2. Loads the fine-tuned model
3. Evaluates both on COVID-19 validation set
4. Generates comparison ROC curves
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PneumoniaDataset, valid_transforms
from train import build_model

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 10)


@torch.no_grad()
def get_predictions(model, loader, device: torch.device):
    """Get predictions and labels from model."""
    model.eval()
    all_probs = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
    
    probs = np.concatenate(all_probs, axis=0).flatten()
    labels = np.concatenate(all_labels, axis=0)
    return labels, probs


def plot_roc_comparison(roc_data: dict, output_path: Path):
    """Plot ROC curves comparing original vs fine-tuned models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    colors = {
        "original": "#2E86AB",  # Blue
        "finetuned": "#A23B72",  # Purple
    }
    
    for model_name, data in roc_data.items():
        fpr = data["fpr"]
        tpr = data["tpr"]
        roc_auc = data["auc"]
        color = colors.get(model_name, "gray")
        
        # Full ROC curve
        ax1.plot(
            fpr, tpr,
            color=color,
            lw=3,
            label=f'{data["label"]} (AUC = {roc_auc:.4f})',
            alpha=0.8
        )
        
        # Zoomed view
        ax2.plot(
            fpr, tpr,
            color=color,
            lw=3,
            label=f'{data["label"]} (AUC = {roc_auc:.4f})',
            alpha=0.8
        )
    
    # Diagonal line (random classifier)
    for ax in [ax1, ax2]:
        ax.plot(
            [0, 1], [0, 1],
            color='navy',
            lw=2,
            linestyle='--',
            alpha=0.5,
            label='Random Classifier (AUC = 0.50)'
        )
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_title('ROC Curves - Full View', fontsize=14, fontweight='bold', pad=15)
    
    # Zoomed view (high performance region)
    ax2.set_xlim([0.0, 0.3])
    ax2.set_ylim([0.7, 1.0])
    ax2.set_title('ROC Curves - Zoomed View (High Performance Region)', fontsize=14, fontweight='bold', pad=15)
    
    # Add improvement text
    if "original" in roc_data and "finetuned" in roc_data:
        orig_auc = roc_data["original"]["auc"]
        ft_auc = roc_data["finetuned"]["auc"]
        improvement = ft_auc - orig_auc
        improvement_pct = (improvement / orig_auc) * 100 if orig_auc > 0 else 0
        
        textstr = f'AUC Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.6, 0.15, textstr, fontsize=12,
                verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved ROC comparison to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare ROC curves between original and fine-tuned models.")
    parser.add_argument(
        "--original_checkpoint",
        type=str,
        required=True,
        help="Path to original pre-trained model checkpoint"
    )
    parser.add_argument(
        "--finetuned_checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="data/processed/covid19_metadata.csv",
        help="COVID-19 dataset metadata CSV"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/external/covid19_extracted",
        help="Root directory of COVID-19 dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save plots"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load COVID-19 validation set
    metadata_path = Path(args.metadata_csv)
    if not metadata_path.exists():
        print(f"Error: Metadata CSV not found: {metadata_path}")
        print("Please run scripts/prepare_covid19_dataset.py first.")
        return

    df = pd.read_csv(metadata_path)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    print(f"Validation set: {len(val_df)} samples")

    # Create validation dataloader
    dataset_root = Path(args.dataset_root)
    valid_transforms_fn = valid_transforms(args.image_size)

    val_ds = PneumoniaDataset(
        val_df,
        dataset_root,
        valid_transforms_fn,
        label_column="opacity"
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    roc_data = {}

    # Evaluate original model
    print("\nEvaluating original model...")
    orig_checkpoint_path = Path(args.original_checkpoint)
    if not orig_checkpoint_path.exists():
        print(f"Error: Original checkpoint not found: {orig_checkpoint_path}")
        return

    orig_checkpoint = torch.load(orig_checkpoint_path, map_location=device)
    orig_config = orig_checkpoint["config"]
    orig_model = build_model(orig_config).to(device)
    orig_model.load_state_dict(orig_checkpoint["model_state"])
    orig_model.eval()

    orig_labels, orig_probs = get_predictions(orig_model, val_loader, device)
    orig_fpr, orig_tpr, _ = roc_curve(orig_labels, orig_probs)
    orig_auc = auc(orig_fpr, orig_tpr)

    print(f"  Original model AUC: {orig_auc:.4f}")

    roc_data["original"] = {
        "fpr": orig_fpr,
        "tpr": orig_tpr,
        "auc": orig_auc,
        "label": "Original Model"
    }

    # Evaluate fine-tuned model
    print("\nEvaluating fine-tuned model...")
    ft_checkpoint_path = Path(args.finetuned_checkpoint)
    if not ft_checkpoint_path.exists():
        print(f"Error: Fine-tuned checkpoint not found: {ft_checkpoint_path}")
        return

    ft_checkpoint = torch.load(ft_checkpoint_path, map_location=device)
    ft_config = ft_checkpoint["config"]
    ft_model = build_model(ft_config).to(device)
    ft_model.load_state_dict(ft_checkpoint["model_state"])
    ft_model.eval()

    ft_labels, ft_probs = get_predictions(ft_model, val_loader, device)
    ft_fpr, ft_tpr, _ = roc_curve(ft_labels, ft_probs)
    ft_auc = auc(ft_fpr, ft_tpr)

    print(f"  Fine-tuned model AUC: {ft_auc:.4f}")

    roc_data["finetuned"] = {
        "fpr": ft_fpr,
        "tpr": ft_tpr,
        "auc": ft_auc,
        "label": "Fine-tuned Model"
    }

    # Calculate improvement
    improvement = ft_auc - orig_auc
    improvement_pct = (improvement / orig_auc) * 100 if orig_auc > 0 else 0
    print(f"\nAUC Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")

    # Plot comparison
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / "roc_comparison_original_vs_finetuned.png"
    plot_roc_comparison(roc_data, plot_path)

    # Save summary
    summary = {
        "original_auc": float(orig_auc),
        "finetuned_auc": float(ft_auc),
        "improvement": float(improvement),
        "improvement_percent": float(improvement_pct),
        "validation_samples": len(val_df),
    }

    with open(output_dir / "roc_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Log to Weights & Biases
    print("\nLogging to Weights & Biases...")
    wandb.init(
        project="rsna-pneumonia-texture-analysis",
        name="roc_comparison_original_vs_finetuned",
        job_type="evaluation",
        config={
            "original_checkpoint": str(orig_checkpoint_path),
            "finetuned_checkpoint": str(ft_checkpoint_path),
            "validation_samples": len(val_df),
        }
    )
    
    # Log metrics
    wandb.log({
        "original_auc": orig_auc,
        "finetuned_auc": ft_auc,
        "auc_improvement": improvement,
        "auc_improvement_percent": improvement_pct,
    })
    
    # Log the comparison plot
    wandb.log({
        "roc_comparison_plot": wandb.Image(str(plot_path))
    })
    
    # Create a table with ROC data for both models
    roc_table = wandb.Table(columns=["Model", "FPR", "TPR", "AUC"])
    for i in range(len(orig_fpr)):
        roc_table.add_data("Original", orig_fpr[i], orig_tpr[i], orig_auc if i == 0 else None)
    for i in range(len(ft_fpr)):
        roc_table.add_data("Fine-tuned", ft_fpr[i], ft_tpr[i], ft_auc if i == 0 else None)
    
    wandb.log({"roc_curves_table": roc_table})
    
    # Log summary
    wandb.summary.update(summary)
    
    wandb.finish()

    print(f"\n✓ Comparison complete!")
    print(f"  Plot saved to: {plot_path}")
    print(f"  Summary saved to: {output_dir / 'roc_comparison_summary.json'}")
    print(f"  Results logged to Weights & Biases")


if __name__ == "__main__":
    main()

