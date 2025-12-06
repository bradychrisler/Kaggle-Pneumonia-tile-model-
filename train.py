"""
Baseline training script for the RSNA Pneumonia Texture Analysis competition.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import wandb
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch import amp
from torchvision import models

from src.data.dataset import build_dataloaders, create_folds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_model(config: dict) -> nn.Module:
    name = config["model"]["name"]
    pretrained = config["model"].get("pretrained", True)
    
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, 1)
        )
    elif name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, 1)
        )
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    scaler,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        with amp.autocast("cuda"):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model: nn.Module, loader, device: torch.device):
    model.eval()
    losses = []
    all_targets = []
    all_probs = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)

        losses.append(loss.item() * images.size(0))
        all_targets.append(labels.cpu())
        all_probs.append(probs.cpu())

    targets = torch.cat(all_targets).numpy()
    probs = torch.cat(all_probs).numpy()
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = float("nan")
    avg_loss = sum(losses) / len(loader.dataset)
    return avg_loss, auc


def main():
    parser = argparse.ArgumentParser(description="Train RSNA baseline model.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    metadata_path = Path(config["data"]["metadata_csv"])
    df = pd.read_csv(metadata_path)
    if "fold" not in df.columns:
        df = create_folds(df, n_splits=config["data"]["n_folds"], target_col="opacity")
        df.to_csv(metadata_path, index=False)

    train_loader, valid_loader = build_dataloaders(
        df,
        image_root=Path(config["data"]["image_root"]),
        fold=config["data"]["fold"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        use_albumentations=config["data"].get("use_albumentations", True),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    scaler = amp.GradScaler("cuda", enabled=config["training"]["mixed_precision"])

    output_dir = Path(config["training"]["output_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Weights & Biases
    run_name = f"{config['model']['name']}_fold{config['data']['fold']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="rsna-pneumonia-texture-analysis",
        name=run_name,
        config=config,
        dir=str(output_dir),
    )
    
    # Log model architecture
    wandb.config.update({
        "model_architecture": config["model"]["name"],
        "pretrained": config["model"].get("pretrained", True),
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    })

    history = []
    best_auc = -float("inf")
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_auc = validate(model, valid_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
        })

        torch.save({"model_state": model.state_dict(), "config": config}, output_dir / f"model_epoch_{epoch}.pth")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model_state": model.state_dict(), "config": config}, output_dir / "best_model.pth")
            # Log best model artifact
            artifact = wandb.Artifact(f"best_model_{run_name}", type="model")
            artifact.add_file(str(output_dir / "best_model.pth"))
            wandb.log_artifact(artifact)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    
    # Log final metrics summary
    wandb.summary.update({
        "best_val_auc": best_auc,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
    })
    
    wandb.finish()


if __name__ == "__main__":
    main()

