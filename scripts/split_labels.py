from collections import defaultdict
from pathlib import Path

CONSOLIDATED_FILE = Path("../dataset/yolo_data/all_labels.txt")
LABELS_DIR = Path("../dataset/yolo_data/labels")
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def split_consolidated_labels():
    if not CONSOLIDATED_FILE.exists():
        print(f"[Error] File not found: {CONSOLIDATED_FILE.resolve()}")
        return

    image_annotations = defaultdict(list)

    print(f"Reading consolidated annotations from {CONSOLIDATED_FILE}...")
    with open(CONSOLIDATED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) == 2:
                img_name, bbox_str = parts
                image_annotations[img_name].append(bbox_str)

    print(f"Splitting into {len(image_annotations)} individual label files...")
    for img_name, annotations in image_annotations.items():
                stem = Path(img_name).stem
                out_txt = LABELS_DIR / f"{stem}.txt"
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(annotations))

    print(f"[Success] Created individual label files in {LABELS_DIR.resolve()}")


if __name__ == "__main__":
    split_consolidated_labels()