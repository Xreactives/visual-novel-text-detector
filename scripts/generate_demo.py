from pathlib import Path
import random
import shutil
import cv2
from ultralytics import YOLO
random.seed(30)
# Paths
VAL_IMAGES_DIR = Path("../dataset/yolo_dataset/images/val")
WEIGHTS_PATH = Path("../runs/detect/vn_yolo26/weights/best.pt")
ASSETS_DIR = Path("../assets")
ASSETS_DIR.mkdir(exist_ok=True)

model = YOLO(str(WEIGHTS_PATH))

# Gather and randomly shuffle all validation screenshots
all_images = [p for p in VAL_IMAGES_DIR.glob("*.*") if p.is_file()]
random.shuffle(all_images)

# Pick the first random image that has both character_name (0) and dialogue_box (2)
selected_img = None
best_result = None

for img_path in all_images:
    res = model.predict(str(img_path), conf=0.6, verbose=False)[0]
    detected_classes = {int(c) for c in res.boxes.cls}
    if {0, 2}.issubset(detected_classes):
        selected_img = img_path
        best_result = res
        break

# Fallback: pick any random image if none matched both classes
if selected_img is None:
    selected_img = random.choice(all_images)
    best_result = model.predict(str(selected_img), conf=0.5, verbose=False)[0]

# 1. Save clean random screenshot as demo_input.png
demo_input = ASSETS_DIR / "demo_input.png"
shutil.copy(selected_img, demo_input)

# 2. Save annotated output as demo_output.png
annotated_img = best_result.plot(line_width=2, font_size=14)
demo_output = ASSETS_DIR / "demo_output.png"
cv2.imwrite(str(demo_output), annotated_img)

print(f"Picked random image: {selected_img.name}")
print(f"Saved {demo_input} and {demo_output}")