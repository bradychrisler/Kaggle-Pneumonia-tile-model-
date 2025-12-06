"""
Predict pneumonia opacity on full chest X-ray images by tiling.

This script takes a full chest X-ray (any size) and:
1. Tiles it into overlapping patches
2. Runs inference on each patch
3. Aggregates predictions (max, mean, or creates a heatmap)

Usage:
    python scripts/predict_full_xray.py --image path/to/xray.png --model outputs/20251126_122837/best_model.pth
    python scripts/predict_full_xray.py --image path/to/xray.png --model outputs/20251126_122837/best_model.pth --tile_size 64 --stride 32 --output heatmap.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import build_model, load_config
from src.data.dataset import valid_transforms


def tile_image(
    image: np.ndarray,
    tile_size: int = 64,
    stride: int = 32,
    pad_value: int = 0
) -> Tuple[list, list]:
    """
    Tile an image into overlapping patches.
    
    Args:
        image: Input image as numpy array (H, W) or (H, W, C)
        tile_size: Size of each tile (will be resized to 224x224 for model)
        stride: Step size for sliding window (smaller = more overlap)
        pad_value: Value to use for padding
    
    Returns:
        tiles: List of tile arrays
        positions: List of (y, x) top-left corner positions for each tile
    """
    if len(image.shape) == 2:
        h, w = image.shape
        channels = 1
    else:
        h, w, channels = image.shape
    
    tiles = []
    positions = []
    
    # Pad image if needed to ensure we can extract tiles
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h > 0 or pad_w > 0:
        if channels == 1:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), constant_values=pad_value)
        else:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=pad_value)
        h, w = image.shape[:2]
    
    # Extract tiles with sliding window
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            if channels == 1:
                tile = image[y:y+tile_size, x:x+tile_size]
            else:
                tile = image[y:y+tile_size, x:x+tile_size, :]
            tiles.append(tile)
            positions.append((y, x))
    
    # Handle edge cases - ensure we get the bottom-right corner
    if (h - tile_size) % stride != 0:
        y = h - tile_size
        for x in range(0, w - tile_size + 1, stride):
            if channels == 1:
                tile = image[y:y+tile_size, x:x+tile_size]
            else:
                tile = image[y:y+tile_size, x:x+tile_size, :]
            tiles.append(tile)
            positions.append((y, x))
    
    if (w - tile_size) % stride != 0:
        x = w - tile_size
        for y in range(0, h - tile_size + 1, stride):
            if channels == 1:
                tile = image[y:y+tile_size, x:x+tile_size]
            else:
                tile = image[y:y+tile_size, x:x+tile_size, :]
            tiles.append(tile)
            positions.append((y, x))
    
    return tiles, positions


def predict_tiles(
    model: torch.nn.Module,
    tiles: list,
    device: torch.device,
    batch_size: int = 32,
    image_size: int = 224
) -> np.ndarray:
    """
    Run inference on a list of tiles.
    
    Args:
        model: Trained PyTorch model
        tiles: List of tile arrays
        device: Device to run inference on
        batch_size: Batch size for inference
        image_size: Target size for model input (224)
    
    Returns:
        Array of prediction probabilities for each tile
    """
    model.eval()
    transform = valid_transforms(image_size)
    
    all_probs = []
    
    # Process in batches
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i:i+batch_size]
        batch_tensors = []
        
        for tile in batch_tiles:
            # Convert to RGB if grayscale
            if len(tile.shape) == 2:
                tile = np.stack([tile, tile, tile], axis=-1)
            elif tile.shape[2] == 1:
                tile = np.repeat(tile, 3, axis=2)
            
            # Ensure uint8
            if tile.dtype != np.uint8:
                tile = (tile * 255).astype(np.uint8) if tile.max() <= 1.0 else tile.astype(np.uint8)
            
            # Apply transforms
            transformed = transform(image=tile)
            batch_tensors.append(transformed["image"])
        
        # Stack into batch
        batch = torch.stack(batch_tensors).to(device)
        
        # Predict
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    
    return np.concatenate(all_probs, axis=0).flatten()


def create_heatmap(
    image: np.ndarray,
    positions: list,
    predictions: np.ndarray,
    tile_size: int = 64,
    stride: int = 32,
    method: str = "max"
) -> Tuple[np.ndarray, dict]:
    """
    Create a heatmap from tile predictions.
    
    Args:
        image: Original image
        positions: List of (y, x) positions for each tile
        predictions: Prediction probabilities for each tile
        tile_size: Size of tiles
        stride: Stride used for tiling
        method: Aggregation method - 'max' (overlapping regions use max) or 'mean' (average)
    
    Returns:
        heatmap: 2D array with aggregated predictions
        stats: Dictionary with statistics
    """
    if len(image.shape) == 2:
        h, w = image.shape
    else:
        h, w = image.shape[:2]
    
    # Initialize heatmap and count map for averaging
    heatmap = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.int32)
    
    # Aggregate predictions
    for (y, x), pred in zip(positions, predictions):
        if method == "max":
            # Use maximum value for overlapping regions
            heatmap[y:y+tile_size, x:x+tile_size] = np.maximum(
                heatmap[y:y+tile_size, x:x+tile_size],
                pred
            )
        else:  # mean
            # Accumulate for averaging
            heatmap[y:y+tile_size, x:x+tile_size] += pred
            count_map[y:y+tile_size, x:x+tile_size] += 1
    
    # Normalize if using mean
    if method == "mean":
        count_map[count_map == 0] = 1  # Avoid division by zero
        heatmap = heatmap / count_map
    
    stats = {
        "max_prob": float(heatmap.max()),
        "mean_prob": float(heatmap.mean()),
        "min_prob": float(heatmap.min()),
        "num_tiles": len(predictions),
        "tile_size": tile_size,
        "stride": stride,
    }
    
    return heatmap, stats


def visualize_results(
    image: np.ndarray,
    heatmap: np.ndarray,
    stats: dict,
    output_path: Optional[Path] = None,
    tile_size: int = 64
):
    """
    Visualize the original image with prediction heatmap overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax1 = axes[0]
    if len(image.shape) == 2:
        ax1.imshow(image, cmap='gray')
    else:
        ax1.imshow(image)
    ax1.set_title("Original Chest X-Ray", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Heatmap
    ax2 = axes[1]
    im = ax2.imshow(heatmap, cmap='hot', vmin=0, vmax=1, interpolation='bilinear')
    ax2.set_title(f"Pneumonia Probability Heatmap\n(Max: {stats['max_prob']:.3f}, Mean: {stats['mean_prob']:.3f})", 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, label='Probability', fraction=0.046)
    
    # Overlay
    ax3 = axes[2]
    if len(image.shape) == 2:
        ax3.imshow(image, cmap='gray', alpha=0.7)
    else:
        ax3.imshow(image, alpha=0.7)
    overlay = ax3.imshow(heatmap, cmap='hot', alpha=0.5, vmin=0, vmax=1, interpolation='bilinear')
    ax3.set_title("Overlay", fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(overlay, ax=ax3, label='Probability', fraction=0.046)
    
    plt.suptitle(f"Pneumonia Detection Analysis ({stats['num_tiles']} tiles analyzed)", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predict pneumonia on full chest X-ray images.")
    parser.add_argument("--image", type=str, required=True, help="Path to chest X-ray image")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tile_size", type=int, default=64, help="Size of tiles to extract (default: 64)")
    parser.add_argument("--stride", type=int, default=32, help="Stride for sliding window (default: 32, 50% overlap)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--method", type=str, default="max", choices=["max", "mean"], 
                       help="Aggregation method: max or mean (default: max)")
    parser.add_argument("--output", type=str, default=None, help="Output path for heatmap visualization")
    parser.add_argument("--image_size", type=int, default=224, help="Model input size (default: 224)")
    args = parser.parse_args()
    
    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Loading image from {image_path}")
    image = Image.open(image_path)
    if image.mode != 'RGB' and image.mode != 'L':
        image = image.convert('RGB')
    
    image_array = np.array(image)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Convert RGB to grayscale for medical images (use green channel or average)
        image_array = image_array.mean(axis=2).astype(np.uint8)
    
    print(f"Image shape: {image_array.shape}")
    
    # Tile the image
    print(f"Tiling image with tile_size={args.tile_size}, stride={args.stride}...")
    tiles, positions = tile_image(image_array, tile_size=args.tile_size, stride=args.stride)
    print(f"Extracted {len(tiles)} tiles")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print(f"Model loaded: {config['model']['name']}")
    
    # Run inference
    print(f"Running inference on {len(tiles)} tiles...")
    predictions = predict_tiles(model, tiles, device, batch_size=args.batch_size, image_size=args.image_size)
    print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Mean prediction: {predictions.mean():.4f}")
    
    # Create heatmap
    print(f"Creating heatmap using {args.method} aggregation...")
    heatmap, stats = create_heatmap(
        image_array, positions, predictions, 
        tile_size=args.tile_size, stride=args.stride, method=args.method
    )
    
    print(f"\nResults:")
    print(f"  Max probability: {stats['max_prob']:.4f}")
    print(f"  Mean probability: {stats['mean_prob']:.4f}")
    print(f"  Min probability: {stats['min_prob']:.4f}")
    print(f"  Number of tiles: {stats['num_tiles']}")
    
    # Visualize
    output_path = Path(args.output) if args.output else None
    visualize_results(image_array, heatmap, stats, output_path, tile_size=args.tile_size)
    
    # Save heatmap as numpy array if requested
    if args.output:
        np_path = Path(args.output).with_suffix('.npy')
        np.save(np_path, heatmap)
        print(f"Saved heatmap array to {np_path}")


if __name__ == "__main__":
    main()






