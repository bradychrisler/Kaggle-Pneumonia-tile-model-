"""
Data preparation utilities for the RSNA Pneumonia Texture Analysis dataset.

Usage examples:
    python scripts/prepare_data.py --subset train
    python scripts/prepare_data.py --subset all
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd
import tifffile
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def rescale_to_uint8(array) -> Image.Image:
    """Convert an arbitrary integer array to an 8-bit grayscale PIL image."""
    arr = array
    if arr.dtype != "uint8":
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        span = arr_max - arr_min
        if math.isclose(span, 0.0):
            span = 1.0
        arr = ((arr - arr_min) / span * 255).astype("uint8")
    return Image.fromarray(arr)


def export_stack(tif_path: Path, meta_path: Path, out_dir: Path, meta_out: Path) -> None:
    meta = pd.read_csv(meta_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_count = len(meta)
    with tifffile.TiffFile(tif_path) as tif:
        if len(tif.pages) != page_count:
            raise ValueError(
                f"Metadata rows ({page_count}) do not match TIFF pages ({len(tif.pages)})"
            )

        image_paths: list[str] = []
        for _, row in tqdm(meta.iterrows(), total=page_count, desc=f"Exporting {tif_path.name}"):
            page_index = int(row["slice_idx"])
            page = tif.pages[page_index]
            img = rescale_to_uint8(page.asarray())

            tile_id = row["tile_id"]
            filename = f"{tile_id}.png"
            target = out_dir / filename
            img.save(target)
            image_paths.append(str(target.relative_to(PROJECT_ROOT)))

    meta = meta.assign(image_path=image_paths)
    meta.to_csv(meta_out, index=False)
    print(f"Wrote {page_count} images to {out_dir} and metadata to {meta_out}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare RSNA Pneumonia data.")
    parser.add_argument(
        "--subset",
        choices=("train", "test", "all"),
        default="all",
        help="Which portions to process.",
    )
    parser.add_argument(
        "--train-tif",
        default=str(RAW_DIR / "train.tif"),
        help="Path to train.tif",
    )
    parser.add_argument(
        "--test-tif",
        default=str(RAW_DIR / "test.tif"),
        help="Path to test.tif",
    )
    parser.add_argument(
        "--train-meta",
        default=str(RAW_DIR / "train_all.csv"),
        help="CSV with training metadata",
    )
    parser.add_argument(
        "--test-meta",
        default=str(RAW_DIR / "test_info.csv"),
        help="CSV with test metadata",
    )
    return parser


def subsets_to_run(option: str) -> Iterable[str]:
    if option == "all":
        return ("train", "test")
    return (option,)


def main() -> None:
    args = build_arg_parser().parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for subset in subsets_to_run(args.subset):
        if subset == "train":
            export_stack(
                Path(args.train_tif),
                Path(args.train_meta),
                PROCESSED_DIR / "train_png",
                PROCESSED_DIR / "train_metadata.csv",
            )
        else:
            export_stack(
                Path(args.test_tif),
                Path(args.test_meta),
                PROCESSED_DIR / "test_png",
                PROCESSED_DIR / "test_metadata.csv",
            )


if __name__ == "__main__":
    main()

