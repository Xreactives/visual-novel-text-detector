from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    # 1. Verify CUDA GPU and dataset config
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Verify your PyTorch CUDA installation.")

    data_yaml = Path("dataset/yolo_dataset/data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config missing at: {data_yaml.resolve()}")

    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    # 2. Initialize YOLO26 (use 'yolo26n.pt' for lowest latency or 'yolo26s.pt' for higher accuracy)
    model = YOLO("yolo26n.pt")

    # 3. Start training
    model.train(
        data=str(data_yaml),
        epochs=150,
        imgsz=640,           # Use 1280 if thin character_name boxes lose detail
        batch=16,            # Lower to 8 if you encounter CUDA Out-Of-Memory
        device=0,
        workers=4,           # Keeps Windows multiprocessing stable
        name="vn_yolo26",
        patience=25,         # Stops early if validation accuracy plateaus
        mosaic=0.5,          # Reduced mosaic helps preserve visual novel UI layout structure
        fliplr=0.0,          # Disabled horizontal flip so text boxes stay realistic
    )

    # 4. Run final validation on best weights
    best_weights = Path("runs/detect/vn_yolo26/weights/best.pt")
    best_model = YOLO(str(best_weights))
    metrics = best_model.val()
    print(f"Training complete. Best weights saved to: {best_weights}")
    print(f"Validation mAP50: {round(metrics.box.map50 * 100)}% | mAP50-95: {round(metrics.box.map * 100)}%")


if __name__ == "__main__":
    main()