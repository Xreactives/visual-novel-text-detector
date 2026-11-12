import argparse
from pathlib import Path
import cv2
import easyocr
import torch
from ultralytics import YOLO

# Distinct BGR colors for bounding box visualization
BOX_COLORS = {
    "character_name": (75, 75, 255),   # Red
    "choice_box": (245, 139, 75),      # Blue
    "dialogue_box": (151, 201, 32),    # Green
}


def detect_and_extract(
    image_path: str,
    weights_path: str,
    conf_threshold: float,
    languages: list[str],
    save_path: str | None,
) -> list[dict]:
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"Input image not found: {img_file.resolve()}")

    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {weights_file.resolve()}. "
            "Ensure best.pt is copied to weights/best.pt."
        )

    img = cv2.imread(str(img_file))
    if img is None:
        raise ValueError(f"Failed to decode image: {img_file}")

    use_gpu = torch.cuda.is_available()
    detector = YOLO(str(weights_file))
    reader = easyocr.Reader(languages, gpu=use_gpu)

    results = detector.predict(img, conf=conf_threshold, verbose=False)[0]
    annotated = img.copy()
    h, w = img.shape[:2]

    # Sort detections top-to-bottom so character_name prints before dialogue_box
    sorted_boxes = sorted(results.boxes, key=lambda b: float(b.xyxy[0][1]))

    extracted_entries = []
    print(f"\n=== Detections for {img_file.name} ===")

    for box in sorted_boxes:
        cls_id = int(box.cls[0])
        cls_name = detector.names[cls_id]
        conf = float(box.conf[0])

        # Clamp bounding box coordinates within image dimensions
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Run EasyOCR only on the cropped UI box
        text_lines = reader.readtext(crop, detail=0, paragraph=True)
        clean_text = " ".join(text_lines).strip()

        extracted_entries.append({
            "class": cls_name,
            "confidence": round(conf * 100),
            "bbox": [x1, y1, x2, y2],
            "text": clean_text,
        })

        print(f"[{cls_name} | {round(conf * 100)}%]: {clean_text or '(no text detected)'}")

        # Draw bounding box and label on output image
        color = BOX_COLORS.get(cls_name, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {round(conf * 100)}%"
        cv2.putText(
            annotated,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_file), annotated)
        print(f"\nSaved annotated result to: {out_file}")

    return extracted_entries


def main():
    parser = argparse.ArgumentParser(
        description="Detect visual novel UI text boxes with YOLO26 and extract text with EasyOCR."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="assets/demo_input.png",
        help="Path to input screenshot (default: assets/demo_input.png)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to trained YOLO26 weights (default: weights/best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["en", "ja"],
        help="EasyOCR language codes, e.g. --langs en ja (default: en ja)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="assets/demo_output.png",
        help="Path to save annotated output image (default: assets/demo_output.png)",
    )
    args = parser.parse_args()

    detect_and_extract(
        image_path=args.image,
        weights_path=args.weights,
        conf_threshold=args.conf,
        languages=args.langs,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()