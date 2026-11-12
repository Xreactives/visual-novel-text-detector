import os
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("../dataset/exported_yolo")
TARGET_DIR = Path("../dataset/yolo_dataset")
SPLIT_RATIO = 0.8

for split in ["train", "val"]:
    (TARGET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

images = list((SOURCE_DIR / "images").glob("*.*"))
random.seed(2026)
random.shuffle(images)

split_idx = int(len(images) * SPLIT_RATIO)
splits = {"train": images[:split_idx], "val": images[split_idx:]}

for split, files in splits.items():
    for img_path in files:
        label_path = SOURCE_DIR / "labels" / f"{img_path.stem}.txt"
        shutil.copy(img_path, TARGET_DIR / "images" / split / img_path.name)
        if label_path.exists():
            shutil.copy(label_path, TARGET_DIR / "labels" / split / label_path.name)

print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])}")