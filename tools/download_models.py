"""Download vision and voice model weights into models/weights/ (not committed to git).

Usage:
    python tools/download_models.py --check
    python tools/download_models.py --download yolo
    python tools/download_models.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(REPO_ROOT, "models", "weights")

# Official YOLOv8n ONNX release URL
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
YOLO_FILE = "yolov8n.onnx"


def check_models() -> bool:
    """Check whether model weights are downloaded."""
    yolo_path = os.path.join(WEIGHTS_DIR, YOLO_FILE)
    has_yolo = os.path.exists(yolo_path) and os.path.getsize(yolo_path) > 1000
    print(f"YOLOv8n ONNX: {'[FOUND] ' + yolo_path if has_yolo else '[MISSING] ' + yolo_path}")
    return has_yolo


def download_file(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} -> {dest_path}...")

    def progress(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100.0, float(count * block_size) / float(total_size) * 100.0)
            sys.stdout.write(f"\r  {pct:.1f}% ({count * block_size // 1024} KB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=progress)
    print("\nDownload complete.")


def download_yolo() -> None:
    yolo_path = os.path.join(WEIGHTS_DIR, YOLO_FILE)
    if os.path.exists(yolo_path):
        print(f"Already exists: {yolo_path}")
        return
    download_file(YOLO_URL, yolo_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check if models exist")
    parser.add_argument("--download", choices=("yolo", "all"), help="download model")
    parser.add_argument("--all", action="store_true", help="download all models")
    args = parser.parse_args()

    if args.check:
        ok = check_models()
        return 0 if ok else 1

    if args.all or args.download in ("yolo", "all"):
        download_yolo()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
