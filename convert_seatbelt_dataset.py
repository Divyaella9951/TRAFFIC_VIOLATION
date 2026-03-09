import os
import json
import shutil

json_path = "datasets/archive_seatbelt/data/train/train.jsonl"
image_root = "datasets/archive_seatbelt/data/train/images"  # change if needed

with_dir = "datasets/seatbelt/with_seatbelt"
without_dir = "datasets/seatbelt/without_seatbelt"

os.makedirs(with_dir, exist_ok=True)
os.makedirs(without_dir, exist_ok=True)

with open(json_path, "r") as f:
    for line in f:
        data = json.loads(line)

        img = data["image"]
        label = data["label"]

        src = os.path.join(image_root, img)

        if label == 1:
            dst = with_dir
        else:
            dst = without_dir

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print("Missing:", src)