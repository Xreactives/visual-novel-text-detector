import os
import torch

# Cap PyTorch CPU threads to prevent 100% CPU usage
torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

from pathlib import Path
import cv2
import easyocr
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_IMAGES_DIR = Path("../dataset/yolo_data/images")
CONSOLIDATED_FILE = Path("../dataset/yolo_data/all_labels.txt")
CONSOLIDATED_FILE.parent.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.35
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Auto-Labeler] Device: {device.upper()}")

reader = easyocr.Reader(["ja", "en"], gpu=(device == "cuda"))


def easyocr_to_yolo(bbox, img_width, img_height):
    pts = np.array(bbox, dtype=np.int32)
    xmin = float(np.max([0, np.min(pts[:, 0])]))
    xmax = float(np.min([img_width, np.max(pts[:, 0])]))
    ymin = float(np.max([0, np.min(pts[:, 1])]))
    ymax = float(np.min([img_height, np.max(pts[:, 1])]))

    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0:
        return None

    cx = (xmin + bw / 2.0) / float(img_width)
    cy = (ymin + bh / 2.0) / float(img_height)
    norm_w = bw / float(img_width)
    norm_h = bh / float(img_height)
    return cx, cy, norm_w, norm_h


def classify_vn_box(cy, norm_w, norm_h, img_height):
    y_pixel = cy * img_height
    if y_pixel > (img_height * 0.65):
        if norm_w < 0.25 and norm_h < 0.08:
            return 0  # character_name
        return 2  # dialogue_box
    if norm_w < 0.6 and norm_h < 0.12:
        return 1  # choice_box
    return 2


def run_autolabel():
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_files = [
        f for f in SOURCE_IMAGES_DIR.iterdir() if f.suffix.lower() in valid_exts
    ]

    if not image_files:
        print(f"[Error] No images found in {SOURCE_IMAGES_DIR.resolve()}")
        return

    print(f"[Auto-Labeler] Processing {len(image_files)} screenshots into single file...")

    with open(CONSOLIDATED_FILE, "w", encoding="utf-8") as out_f, torch.no_grad():
        for img_path in tqdm(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_h, img_w = img.shape[:2]

            detections = reader.readtext(
                img,
                text_threshold=0.3,
                low_text=0.3,
                link_threshold=0.2,
                width_ths=0.7,
                canvas_size=1280,
            )

            for bbox, text, prob in detections:
                if prob < CONF_THRESHOLD or not text.strip():
                    continue

                coords = easyocr_to_yolo(bbox, img_w, img_h)
                if coords is None:
                    continue

                cx, cy, norm_w, norm_h = coords
                cls_id = classify_vn_box(cy, norm_w, norm_h, img_h)

                # Format: image_filename \t class_id x_center y_center width height
                out_f.write(
                    f"{img_path.name}\t{cls_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                )

    print(f"\n[Success] All labels saved to {CONSOLIDATED_FILE.resolve()}")
    print("Use split_labels.py next to format it to YOLO format")

if __name__ == "__main__":
    run_autolabel()