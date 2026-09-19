"""Train image/video deepfake classifiers.

Expected image layout:
datasets/deepfake/image_dataset/
  real/
  fake/

Expected video layout:
datasets/deepfake/video_dataset/
  real/
    *.mp4
  fake/
    *.mp4

The video trainer samples frames and trains the same EfficientNet-B0
frame classifier. Inference can aggregate frame probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "trained_models"


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_image(data_dir: Path, epochs: int = 5, batch_size: int = 8, max_samples: int = 10_000) -> None:
    if not data_dir.exists():
        raise SystemExit(f"Image dataset not found: {data_dir}")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Build a bounded balanced subset (up to max_samples total) so development\n    # training does not consume the entire deepfake corpus.\n    subset = data_dir.parent / "image_train_subset"\n    for cls in ("real", "fake"):\n        (subset / cls).mkdir(parents=True, exist_ok=True)\n    per_class = max_samples // 2\n    selected = []\n    for cls in ("real", "fake"):\n        files = []\n        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):\n            files.extend(sorted((data_dir / cls).glob(ext)))\n        selected.extend((cls, p) for p in files[:per_class])\n    for cls, src in selected:\n        dst = subset / cls / src.name\n        if not dst.exists():\n            try:\n                dst.hardlink_to(src)\n            except OSError:\n                import shutil\n                shutil.copy2(src, dst)\n    print(f"[image] selected {len(selected):,} images")\n    ds = datasets.ImageFolder(subset, transform=tfm)
    if len(ds.classes) != 2:
        raise SystemExit("Image dataset must contain exactly two folders: real and fake")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
    model.to(device())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device()), y.to(device())
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"[image] epoch {epoch + 1}/{epochs} loss={total / max(1, len(loader)):.4f}")

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "efficientnet_b0",
            "classes": ds.classes,
            "state_dict": model.state_dict(),
        },
        MODELS / "deepfake_image_efficientnet_b0.pth",
    )
    print("[done] deepfake_image_efficientnet_b0.pth")


def train_video(data_dir: Path, epochs: int = 3, batch_size: int = 8, max_samples: int = 10_000) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python is required") from exc

    frame_root = data_dir.parent / "video_frames"
    for cls in ["real", "fake"]:
        (frame_root / cls).mkdir(parents=True, exist_ok=True)

    videos = []\n    per_class = max_samples // 2\n    for cls in ["real", "fake"]:\n        cls_files = []\n        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"):\n            cls_files.extend(sorted((data_dir / cls).glob(ext)))\n        videos.extend((cls, p) for p in cls_files[:per_class])

    if not videos:
        raise SystemExit(f"No videos found under {data_dir}/real and {data_dir}/fake")

    for cls, video in videos:
        stem = video.stem
        out_dir = frame_root / cls / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        if any(out_dir.glob("*.jpg")):
            continue
        cap = cv2.VideoCapture(str(video))
        count = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count % max(1, int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) // 5)) == 0:
                cv2.imwrite(str(out_dir / f"{saved:05d}.jpg"), frame)
                saved += 1
            count += 1
            if saved >= 5:
                break
        cap.release()

    train_image(frame_root, epochs=epochs, batch_size=batch_size, max_samples=max_samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)\n    parser.add_argument("--max-samples", type=int, default=10_000)
    args = parser.parse_args()

    if args.image:
        train_image(args.data or ROOT / "datasets" / "deepfake" / "image_dataset",
                    args.epochs, args.batch_size, args.max_samples)
    elif args.video:
        train_video(args.data or ROOT / "datasets" / "deepfake" / "video_dataset",
                    args.epochs, args.batch_size)
    else:
        parser.error("Use --image or --video")


if __name__ == "__main__":
    main()
