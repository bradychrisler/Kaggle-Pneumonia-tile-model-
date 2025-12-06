# RSNA Pneumonia Texture Analysis

Local workspace for the Kaggle RSNA Pneumonia Texture Analysis competition focused on training PyTorch models with GPU acceleration on Windows.

## Quickstart
1. **Create/activate environment**
   ```powershell
   # Option 1: Activate venv (if execution policy allows)
   .\.venv\Scripts\Activate.ps1
   
   # Option 2: Use venv Python directly (if scripts are blocked)
   .\.venv\Scripts\python.exe --version  # Test it works
   
   # Option 3: Bypass execution policy for this session
   Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```
3. **Configure Kaggle + download data**
   ```powershell
   kaggle competitions download -c pneumonia-texture-analysis -p data/raw
   ```
   Keep original `.tif` files in `data/raw` and place derived artifacts in `data/processed`.
4. **Unpack & materialize image tiles**
   ```powershell
   Expand-Archive data/raw/pneumonia-texture-analysis.zip -DestinationPath data/raw -Force
   python scripts/prepare_data.py --subset all
   ```
   Outputs land under `data/processed/train_png` and `data/processed/test_png` with refreshed metadata CSVs.
5. **Validate GPU availability**
   ```powershell
   python scripts/check_cuda.py
   ```
6. **Start experiments**
   - Launch notebooks for EDA: `jupyter lab`
   - Begin training: `python train.py --config configs/baseline.yaml`

See `ENV_SETUP.md` for complete driver, CUDA, and troubleshooting notes.

## Data Layout
- `data/raw/` – untouched Kaggle artifacts (`train.tif`, `test.tif`, metadata CSVs, zip backups).
- `data/processed/train_png` – per-tile PNG exports produced by `prepare_data.py`.
- `data/processed/test_png` – PNG tiles for the test stack.
- `data/processed/train_metadata.csv` / `test_metadata.csv` – original metadata plus `image_path` pointing to the corresponding PNG, ready for dataframe joins in training code.

## Training Pipeline
- `configs/baseline.yaml` – hyperparameters, data paths, and training options (fold, batch size, lr, etc.).
- `src/data/dataset.py` – dataset utilities with stratified folds + dataloaders.
- `train.py` – main entry point. Saves checkpoints/metrics under `outputs/<timestamp>/`.

Run:
```powershell
python train.py --config configs/baseline.yaml
```
After training you'll find:
- `outputs/<timestamp>/best_model.pth`
- Per-epoch checkpoints (`model_epoch_X.pth`)
- `outputs/<timestamp>/metrics.json` capturing losses and AUROC.

## Weights & Biases Integration

This project uses [Weights & Biases](https://wandb.ai) for experiment tracking and visualization.

### Setup
1. **Install wandb** (already in requirements.txt):
   ```powershell
   pip install wandb
   ```

2. **Login to wandb**:
   ```powershell
   wandb login
   ```
   This will prompt you to enter your API key from https://wandb.ai/authorize

3. **Training automatically logs to wandb**:
   - Metrics (train_loss, val_loss, val_auc) are logged each epoch
   - Model checkpoints are saved as artifacts
   - All hyperparameters from config are tracked
   - Project name: `rsna-pneumonia-texture-analysis`

4. **View your runs**:
   Visit https://wandb.ai to see your training runs, compare experiments, and visualize metrics.

## GitHub Setup

### Initial Setup
1. **Initialize git repository** (if not already done):
   ```powershell
   git init
   ```

2. **Add all files**:
   ```powershell
   git add .
   ```

3. **Create initial commit**:
   ```powershell
   git commit -m "Initial commit: RSNA Pneumonia Texture Analysis project"
   ```

### Connect to GitHub
1. **Create a new repository on GitHub**:
   - Go to https://github.com/new
   - Name it (e.g., `rsna-pneumonia-texture-analysis`)
   - Don't initialize with README (we already have one)

2. **Add remote and push**:
   ```powershell
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Future Updates
```powershell
git add .
git commit -m "Description of changes"
git push
```

**Note**: The `.gitignore` file excludes large files like model checkpoints, data files, and wandb logs to keep the repository lightweight.

## Applying Model to Full Chest X-Rays

The model was trained on 64x64 pixel tiles, but you can apply it to full chest X-ray images of any size using the tiling script.

### Usage

```powershell
python scripts/predict_full_xray.py --image path/to/chest_xray.png --model outputs/20251126_122837/best_model.pth
```

### Options

- `--image`: Path to the chest X-ray image (PNG, JPG, etc.)
- `--model`: Path to the trained model checkpoint (`best_model.pth`)
- `--tile_size`: Size of tiles to extract (default: 64, matching training data)
- `--stride`: Step size for sliding window (default: 32, gives 50% overlap)
- `--method`: Aggregation method - `max` (default) or `mean` for overlapping regions
- `--output`: Optional path to save the heatmap visualization
- `--batch_size`: Batch size for inference (default: 32)

### How It Works

1. **Tiling**: The full X-ray is divided into overlapping 64x64 pixel tiles
2. **Inference**: Each tile is resized to 224x224 and run through the model
3. **Aggregation**: Predictions are combined into a heatmap showing pneumonia probability across the image
4. **Visualization**: The script creates a 3-panel visualization:
   - Original X-ray
   - Probability heatmap
   - Overlay of heatmap on original

### Example

```powershell
# Basic usage
python scripts/predict_full_xray.py --image data/external/patient_xray.png --model outputs/20251126_122837/best_model.pth

# With custom tiling and output
python scripts/predict_full_xray.py --image data/external/patient_xray.png --model outputs/20251126_122837/best_model.pth --tile_size 64 --stride 32 --method max --output results/patient_heatmap.png
```

The script outputs:
- Maximum probability across all tiles (most suspicious region)
- Mean probability (overall assessment)
- A heatmap visualization showing where pneumonia is most likely detected

## Downloading External Datasets

### COVID-19 Radiography Database

To download the COVID-19 radiology dataset from Kaggle for testing your model on external data:

1. **Install kagglehub** (if not already installed):
   ```powershell
   pip install kagglehub
   ```

2. **Authenticate** (if needed):
   ```powershell
   kagglehub login
   ```
   Or ensure your `kaggle.json` is in `%USERPROFILE%\.kaggle\kaggle.json`

3. **Download the dataset**:
   ```powershell
   python scripts/download_covid19_dataset.py
   ```

   Or specify a custom output directory:
   ```powershell
   python scripts/download_covid19_dataset.py --output data/external/covid19
   ```

4. **Use the downloaded X-rays**:
   ```powershell
   # Find X-ray images in the downloaded dataset
   # Then run predictions on them
   python scripts/predict_full_xray.py --image <path_to_covid19_xray> --model outputs/.../best_model.pth
   ```

The COVID-19 dataset contains:
- COVID-19 positive chest X-rays
- Normal chest X-rays  
- Other pneumonia cases

This is useful for:
- Testing model generalization on external data
- Validating performance on different X-ray sources
- Creating additional visualizations and analyses

## Fine-Tuning on COVID-19 Dataset

You can fine-tune your pre-trained model on the COVID-19 dataset to improve performance on external data and generate ROC curves for evaluation.

### Quick Start (Complete Pipeline)

Run the complete pipeline (prepares dataset, fine-tunes, and generates ROC curves):

```powershell
python scripts/run_finetuning_pipeline.py --checkpoint outputs/20251126_122837/best_model.pth
```

### Step-by-Step

#### 1. Prepare COVID-19 Dataset Metadata

First, create a metadata CSV from the extracted COVID-19 dataset:

```powershell
python scripts/prepare_covid19_dataset.py --dataset_root data/external/covid19_extracted
```

This will:
- Find all COVID-19 positive and Normal images
- Create train/validation split (80/20 by default)
- Save metadata to `data/processed/covid19_metadata.csv`

#### 2. Fine-Tune the Model

Fine-tune your pre-trained model on the COVID-19 dataset:

```powershell
python scripts/finetune_covid19.py --checkpoint outputs/20251126_122837/best_model.pth --epochs 10 --lr 1e-5
```

Options:
- `--checkpoint`: Path to pre-trained model checkpoint (required)
- `--epochs`: Number of fine-tuning epochs (default: 10)
- `--lr`: Learning rate (default: 1e-5, lower than initial training)
- `--batch_size`: Batch size (default: 32)
- `--freeze_backbone`: Freeze backbone, only train classifier
- `--output_dir`: Output directory (default: `outputs/finetuned_covid19`)

The fine-tuned model will be saved to `outputs/finetuned_covid19/<timestamp>/best_model.pth`.

#### 3. Compare ROC Curves

Generate comparison ROC curves between the original and fine-tuned models:

```powershell
python scripts/compare_roc_curves.py --original_checkpoint outputs/20251126_122837/best_model.pth --finetuned_checkpoint outputs/finetuned_covid19/<timestamp>/best_model.pth
```

This will:
- Evaluate both models on the COVID-19 validation set
- Generate ROC curves showing performance comparison
- Save plots to `visualizations/roc_comparison_original_vs_finetuned.png`
- Save summary statistics to `visualizations/roc_comparison_summary.json`

### Fine-Tuning Tips

1. **Learning Rate**: Use a lower learning rate (1e-5 to 1e-4) than initial training to avoid catastrophic forgetting
2. **Freeze Backbone**: Use `--freeze_backbone` if you want to only adapt the classifier layer
3. **Epochs**: Start with 5-10 epochs and monitor validation AUC
4. **Monitoring**: All metrics are logged to Weights & Biases automatically

### Output Files

After fine-tuning, you'll find:
- `outputs/finetuned_covid19/<timestamp>/best_model.pth` - Best fine-tuned model
- `outputs/finetuned_covid19/<timestamp>/metrics.json` - Training history
- `outputs/finetuned_covid19/<timestamp>/roc_data.json` - ROC curve data
- `visualizations/roc_comparison_original_vs_finetuned.png` - ROC comparison plot
- `visualizations/roc_comparison_summary.json` - Performance summary

