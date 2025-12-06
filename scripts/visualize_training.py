"""
Visualize training progress and model performance.

Usage:
    python scripts/visualize_training.py --output_dir outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)


def load_metrics(output_dir: Path) -> List[dict]:
    """Load metrics from all training runs."""
    metrics_list = []
    for run_dir in sorted(output_dir.glob("*/")):
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                # Add run identifier
                for m in metrics:
                    m["run"] = run_dir.name
                metrics_list.extend(metrics)
    return metrics_list


def plot_training_curves(metrics_list: List[dict], output_path: Path):
    """Plot training and validation curves."""
    df = pd.DataFrame(metrics_list)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for run in df["run"].unique():
        run_data = df[df["run"] == run]
        ax1.plot(run_data["epoch"], run_data["train_loss"], marker="o", label=f"Fold {run[-1] if len(run) > 1 else 'Baseline'}", alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Over Epochs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for run in df["run"].unique():
        run_data = df[df["run"] == run]
        ax2.plot(run_data["epoch"], run_data["val_loss"], marker="s", label=f"Fold {run[-1] if len(run) > 1 else 'Baseline'}", alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Over Epochs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Validation AUC
    ax3 = axes[1, 0]
    for run in df["run"].unique():
        run_data = df[df["run"] == run]
        ax3.plot(run_data["epoch"], run_data["val_auc"], marker="^", label=f"Fold {run[-1] if len(run) > 1 else 'Baseline'}", alpha=0.7, linewidth=2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Validation AUC")
    ax3.set_title("Validation AUC Over Epochs (Higher is Better)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.95, color="r", linestyle="--", alpha=0.5, label="95% Threshold")
    
    # Plot 4: Best AUC per Fold
    ax4 = axes[1, 1]
    best_aucs = df.groupby("run")["val_auc"].max().sort_values(ascending=False)
    bars = ax4.bar(range(len(best_aucs)), best_aucs.values, color=plt.cm.viridis(np.linspace(0, 1, len(best_aucs))))
    ax4.set_xlabel("Fold")
    ax4.set_ylabel("Best Validation AUC")
    ax4.set_title("Best Validation AUC per Fold")
    ax4.set_xticks(range(len(best_aucs)))
    ax4.set_xticklabels([f"Fold {i+1}" for i in range(len(best_aucs))], rotation=45)
    ax4.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(best_aucs.items()):
        ax4.text(i, val + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / "training_curves.png", dpi=300, bbox_inches="tight")
    print(f"Saved training curves to {output_path / 'training_curves.png'}")
    plt.close()


def plot_auc_comparison(metrics_list: List[dict], output_path: Path):
    """Plot AUC comparison across folds."""
    df = pd.DataFrame(metrics_list)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get the 5 most recent folds (assuming they're the EfficientNet folds)
    recent_runs = sorted(df["run"].unique())[-5:] if len(df["run"].unique()) >= 5 else df["run"].unique()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(recent_runs)))
    
    for i, run in enumerate(recent_runs):
        run_data = df[df["run"] == run].sort_values("epoch")
        ax.plot(run_data["epoch"], run_data["val_auc"], 
                marker="o", label=f"Fold {i+1}", 
                color=colors[i], linewidth=2.5, markersize=8, alpha=0.8)
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title("Validation AUC Comparison Across Folds", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="95% Threshold")
    ax.set_ylim([0.90, 1.0])
    
    # Add average line
    if len(recent_runs) >= 5:
        avg_auc = df[df["run"].isin(recent_runs)].groupby("epoch")["val_auc"].mean()
        ax.plot(avg_auc.index, avg_auc.values, 
                color="black", linestyle="--", linewidth=2, 
                label="Average Across Folds", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path / "auc_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved AUC comparison to {output_path / 'auc_comparison.png'}")
    plt.close()


def plot_performance_summary(metrics_list: List[dict], output_path: Path):
    """Create a summary performance plot."""
    df = pd.DataFrame(metrics_list)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Get recent folds
    recent_runs = sorted(df["run"].unique())[-5:] if len(df["run"].unique()) >= 5 else df["run"].unique()
    fold_data = df[df["run"].isin(recent_runs)]
    
    # Plot 1: Final AUC per fold
    ax1 = axes[0]
    final_aucs = fold_data.groupby("run")["val_auc"].last().sort_values(ascending=False)
    bars1 = ax1.bar(range(len(final_aucs)), final_aucs.values, 
                    color=plt.cm.viridis(np.linspace(0, 1, len(final_aucs))))
    ax1.set_xlabel("Fold", fontsize=11)
    ax1.set_ylabel("Final Validation AUC", fontsize=11)
    ax1.set_title("Final AUC per Fold", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(len(final_aucs)))
    ax1.set_xticklabels([f"Fold {i+1}" for i in range(len(final_aucs))])
    ax1.grid(True, alpha=0.3, axis="y")
    for i, val in enumerate(final_aucs.values):
        ax1.text(i, val + 0.003, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    
    # Plot 2: Best AUC per fold
    ax2 = axes[1]
    best_aucs = fold_data.groupby("run")["val_auc"].max().sort_values(ascending=False)
    bars2 = ax2.bar(range(len(best_aucs)), best_aucs.values,
                    color=plt.cm.plasma(np.linspace(0, 1, len(best_aucs))))
    ax2.set_xlabel("Fold", fontsize=11)
    ax2.set_ylabel("Best Validation AUC", fontsize=11)
    ax2.set_title("Best AUC per Fold", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(len(best_aucs)))
    ax2.set_xticklabels([f"Fold {i+1}" for i in range(len(best_aucs))])
    ax2.grid(True, alpha=0.3, axis="y")
    for i, val in enumerate(best_aucs.values):
        ax2.text(i, val + 0.003, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    
    # Plot 3: Average metrics
    ax3 = axes[2]
    avg_final_auc = final_aucs.mean()
    avg_best_auc = best_aucs.mean()
    avg_final_loss = fold_data.groupby("run")["val_loss"].last().mean()
    avg_best_loss = fold_data.groupby("run")["val_loss"].min().mean()
    
    metrics_names = ["Avg Final\nAUC", "Avg Best\nAUC", "Avg Final\nLoss", "Avg Best\nLoss"]
    metrics_values = [avg_final_auc, avg_best_auc, avg_final_loss, avg_best_loss]
    colors3 = ["#2ecc71", "#27ae60", "#e74c3c", "#c0392b"]
    
    bars3 = ax3.bar(metrics_names, metrics_values, color=colors3, alpha=0.7)
    ax3.set_ylabel("Value", fontsize=11)
    ax3.set_title("Average Performance Metrics", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for i, (name, val) in enumerate(zip(metrics_names, metrics_values)):
        if "AUC" in name:
            ax3.text(i, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        else:
            ax3.text(i, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / "performance_summary.png", dpi=300, bbox_inches="tight")
    print(f"Saved performance summary to {output_path / 'performance_summary.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize training progress.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory with training outputs")
    parser.add_argument("--save_dir", type=str, default="visualizations", help="Directory to save plots")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"Loading metrics from {output_dir}...")
    metrics_list = load_metrics(output_dir)
    
    if not metrics_list:
        print(f"No metrics found in {output_dir}")
        return
    
    print(f"Found metrics from {len(set(m['run'] for m in metrics_list))} training runs")
    
    print("Generating visualizations...")
    plot_training_curves(metrics_list, save_dir)
    plot_auc_comparison(metrics_list, save_dir)
    plot_performance_summary(metrics_list, save_dir)
    
    print(f"\nAll visualizations saved to {save_dir}/")
    print("Files created:")
    print("  - training_curves.png (4-panel overview)")
    print("  - auc_comparison.png (AUC across folds)")
    print("  - performance_summary.png (summary statistics)")


if __name__ == "__main__":
    main()








