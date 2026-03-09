import os
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# =========================
# PATHS (UPDATE IF NEEDED)
# =========================

csv_folder = r"C:\Users\DELL\Downloads\TRFFIC PREDICTION\datasets\archivee\helmet_dataset\annotation\annotation"

images_root = r"C:\Users\DELL\Downloads\TRFFIC PREDICTION\datasets\archivee\helmet_dataset\images\part_1"

output_path = r"C:\Users\DELL\Downloads\TRFFIC PREDICTION\datasets\archivee\helmet_dataset"

# =========================
# CREATE YOLO FOLDERS
# =========================

for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(output_path, folder), exist_ok=True)

print("Folders created successfully.")

# =========================
# READ ALL CSV FILES
# =========================

all_data = []

for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(csv_folder, file)
        df = pd.read_csv(file_path)
        df["highway_folder"] = file.replace(".csv", "")
        all_data.append(df)

df = pd.concat(all_data, ignore_index=True)

print("All CSV files loaded successfully.")

# Split 80-20
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# =========================
# PROCESS FUNCTION
# =========================

def process_data(dataframe, split):
    for _, row in dataframe.iterrows():

        frame_id = row["frame_id"]
        x = row["x"]
        y = row["y"]
        w = row["w"]
        h = row["h"]
        label = row["label"]
        highway_folder = row["highway_folder"]

        image_name = f"{int(frame_id):02d}.jpg"

        img_path = os.path.join(images_root, highway_folder, image_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)

        if img is None:
            continue

        img_h, img_w, _ = img.shape

        # Convert to YOLO format
        x_center = (x + w/2) / img_w
        y_center = (y + h/2) / img_h
        width = w / img_w
        height = h / img_h

        class_id = 0 if label == "helmet" else 1

        # Save image
        new_img_path = os.path.join(output_path, f"images/{split}", image_name)
        cv2.imwrite(new_img_path, img)

        # Save label
        label_path = os.path.join(output_path, f"labels/{split}", image_name.replace(".jpg", ".txt"))

        with open(label_path, "a") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"{split} data processed successfully.")


# =========================
# RUN
# =========================

process_data(train_df, "train")
process_data(val_df, "val")

print("Conversion completed successfully!")