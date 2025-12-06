"""
Quick CUDA smoke test for the RSNA Pneumonia Texture Analysis project.

Run with: `python scripts/check_cuda.py`
"""
from __future__ import annotations

import sys
import time


def import_torch() -> "module":
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Activate your environment and run the "
            "install command from ENV_SETUP.md."
        ) from exc
    return torch


def format_bytes(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def main() -> None:
    torch = import_torch()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("⚠️  CUDA is not available. Verify drivers and reinstall PyTorch with CUDA support.")
        sys.exit(1)

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name} | Capability: {props.major}.{props.minor}")
    print(f"Total memory: {format_bytes(props.total_memory)}")

    torch.cuda.empty_cache()

    size = 1024
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    start = time.perf_counter()
    c = a @ b  # noqa: F841  # keep computation for timing
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Matrix multiply ({size}x{size}) completed in {elapsed*1000:.2f} ms")

    print("CUDA smoke test passed [OK]")


if __name__ == "__main__":
    main()

