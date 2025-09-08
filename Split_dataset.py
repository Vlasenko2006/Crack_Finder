import os
import shutil
import random

# Set paths
original_dir = "Dataset"
output_dir = "data_split"
categories = ["Positive", "Negative"]
train_ratio = 0.8  # 80% train, 20% val

# Create output directories
for split in ["train", "val"]:
    for category in categories:
        os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

# For each category, split and copy images
for category in categories:
    src_folder = os.path.join(original_dir, category)
    images = [f for f in os.listdir(src_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)
    n_train = int(len(images) * train_ratio)
    train_images = images[:n_train]
    val_images = images[n_train:]

    for img_name in train_images:
        src_path = os.path.join(src_folder, img_name)
        dst_path = os.path.join(output_dir, "train", category, img_name)
        shutil.copy2(src_path, dst_path)

    for img_name in val_images:
        src_path = os.path.join(src_folder, img_name)
        dst_path = os.path.join(output_dir, "val", category, img_name)
        shutil.copy2(src_path, dst_path)

print("Dataset split complete!")