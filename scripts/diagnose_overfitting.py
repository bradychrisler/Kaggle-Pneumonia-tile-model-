"""
Diagnose potential overfitting by comparing prediction distributions.

This script evaluates the fine-tuned model and shows:
1. Prediction distributions for COVID-19 vs Normal cases
2. Statistical differences
3. Confusion matrix metrics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PneumoniaDataset, valid_transforms
from train import build_model

sns.set_style("whitegrid")


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


def main():
    parser = argparse.ArgumentParser(description="Diagnose overfitting in fine-tuned model.")
    parser.add_argument(
        "--checkpoint",
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
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded: {config['model']['name']}")

    # Load data
    df = pd.read_csv(args.metadata_csv)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    
    dataset_root = Path(args.dataset_root)
    valid_transforms_fn = valid_transforms(224)
    
    val_ds = PneumoniaDataset(val_df, dataset_root, valid_transforms_fn, label_column="opacity")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Get predictions
    print("\nEvaluating model on validation set...")
    labels, probs = get_predictions(model, val_loader, device)

    # Separate by class
    covid_probs = probs[labels == 1.0]
    normal_probs = probs[labels == 0.0]

    print(f"\n{'='*60}")
    print("PREDICTION STATISTICS")
    print(f"{'='*60}")
    print(f"\nCOVID-19 Positive Cases (n={len(covid_probs)}):")
    print(f"  Mean: {covid_probs.mean():.4f}")
    print(f"  Median: {np.median(covid_probs):.4f}")
    print(f"  Std: {covid_probs.std():.4f}")
    print(f"  Min: {covid_probs.min():.4f}")
    print(f"  Max: {covid_probs.max():.4f}")
    print(f"  Range: [{covid_probs.min():.4f}, {covid_probs.max():.4f}]")

    print(f"\nNormal Cases (n={len(normal_probs)}):")
    print(f"  Mean: {normal_probs.mean():.4f}")
    print(f"  Median: {np.median(normal_probs):.4f}")
    print(f"  Std: {normal_probs.std():.4f}")
    print(f"  Min: {normal_probs.min():.4f}")
    print(f"  Max: {normal_probs.max():.4f}")
    print(f"  Range: [{normal_probs.min():.4f}, {normal_probs.max():.4f}]")

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(covid_probs, normal_probs)
    print(f"\nStatistical Test (t-test):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'} (p < 0.05)")

    # Classification metrics
    predictions = (probs >= args.threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION METRICS (threshold={args.threshold})")
    print(f"{'='*60}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  COVID-19")
    print(f"Actual Normal   {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"      COVID-19  {cm[1,0]:4d}     {cm[1,1]:4d}")
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distribution plot
    ax1 = axes[0]
    ax1.hist(normal_probs, bins=50, alpha=0.6, label=f'Normal (n={len(normal_probs)})', color='blue', density=True)
    ax1.hist(covid_probs, bins=50, alpha=0.6, label=f'COVID-19 (n={len(covid_probs)})', color='red', density=True)
    ax1.axvline(normal_probs.mean(), color='blue', linestyle='--', linewidth=2, label=f'Normal mean: {normal_probs.mean():.3f}')
    ax1.axvline(covid_probs.mean(), color='red', linestyle='--', linewidth=2, label=f'COVID-19 mean: {covid_probs.mean():.3f}')
    ax1.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction Distribution: Normal vs COVID-19', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    data_to_plot = [normal_probs, covid_probs]
    bp = ax2.boxplot(data_to_plot, labels=['Normal', 'COVID-19'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Distribution: Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path("visualizations/finetuned_model/prediction_diagnosis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()

    # Overfitting indicators
    print(f"\n{'='*60}")
    print("OVERFITTING INDICATORS")
    print(f"{'='*60}")
    
    mean_diff = abs(covid_probs.mean() - normal_probs.mean())
    std_ratio = min(covid_probs.std(), normal_probs.std()) / max(covid_probs.std(), normal_probs.std()) if max(covid_probs.std(), normal_probs.std()) > 0 else 0
    
    print(f"\n1. Mean Difference: {mean_diff:.4f}")
    if mean_diff < 0.1:
        print("   ⚠️  WARNING: Very small difference - model may not be distinguishing classes")
    elif mean_diff < 0.3:
        print("   ⚠️  CAUTION: Small difference - model may be overfitting")
    else:
        print("   ✓ Good separation between classes")
    
    print(f"\n2. Standard Deviation Ratio: {std_ratio:.4f}")
    if std_ratio < 0.5:
        print("   ⚠️  WARNING: Very different variances - possible overfitting")
    else:
        print("   ✓ Similar variances")
    
    print(f"\n3. Overlap Analysis:")
    overlap = len(np.where((normal_probs >= covid_probs.min()) & (normal_probs <= covid_probs.max()))[0]) / len(normal_probs)
    print(f"   Overlap: {overlap*100:.1f}% of normal predictions fall within COVID-19 range")
    if overlap > 0.8:
        print("   ⚠️  WARNING: High overlap - model not distinguishing well")
    elif overlap > 0.5:
        print("   ⚠️  CAUTION: Moderate overlap")
    else:
        print("   ✓ Good separation")


if __name__ == "__main__":
    main()




