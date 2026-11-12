from pathlib import Path
import urllib.parse

LABELS_DIR = Path("../dataset/exported_yolo/labels")  # Update to your exported labels folder

renamed_count = 0
for file_path in LABELS_DIR.glob("*.txt"):
    # Decode URL-encoded characters (%20 -> space, %28 -> (, %29 -> ))
    decoded_name = urllib.parse.unquote(file_path.name)

    # Strip Label Studio's random hash prefix if present (e.g., "1a2b3c4d__Screenshot (5).txt")
    if "__" in decoded_name:
        decoded_name = decoded_name.split("__", 1)[1]

    if decoded_name != file_path.name:
        file_path.rename(file_path.with_name(decoded_name))
        renamed_count += 1

print(f"Renamed {renamed_count} label files in {LABELS_DIR}")