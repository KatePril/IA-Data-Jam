import os
import shutil
import random
import textwrap

import albumentations as A
import torch
from ultralytics import RTDETR, settings
from ultralytics.data.augment import Albumentations

BASE_DIR = "out_ds_zip/"
ANNOTATIONS_DIR = BASE_DIR + "labels"
IMAGES_DIR = BASE_DIR + "images"

BASE_OUTPUT_DIR = "output/"
OUTPUT_LABELS_DIR = BASE_OUTPUT_DIR + "labels"
OUTPUT_IMAGES_DIR = BASE_OUTPUT_DIR + "images"

OUTPUT_BASE = "split_dataset"
TRAIN_SPLIT = 0.8


# Function to copy paired image and label
def copy_files(split_set, img_dst, lbl_dst):
    for base in split_set:
        img_src = os.path.join(IMAGES_DIR, base + ".jpg")  # Change extension if needed
        if not os.path.exists(img_src):  # Try .png
            img_src = os.path.join(IMAGES_DIR, base + ".png")
        lbl_src = os.path.join(ANNOTATIONS_DIR, base + ".txt")
        if os.path.exists(img_src):
            shutil.copy(img_src, img_dst)
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)


if __name__ == "__main__":
    settings.update({"wandb": True})

    # Make output directories
    train_img_dir = os.path.join(OUTPUT_BASE, "train", "images")
    train_lbl_dir = os.path.join(OUTPUT_BASE, "train", "labels")
    val_img_dir = os.path.join(OUTPUT_BASE, "val", "images")
    val_lbl_dir = os.path.join(OUTPUT_BASE, "val", "labels")

    # Create output directories
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Collect all CSV annotation files
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    base_names = [os.path.splitext(f)[0] for f in image_files]

    # Shuffle and split
    random.shuffle(base_names)
    split_index = int(len(base_names) * TRAIN_SPLIT)
    train_set = base_names[:split_index]
    val_set = base_names[split_index:]

    # Copy to train and val folders
    copy_files(train_set, train_img_dir, train_lbl_dir)
    copy_files(val_set, val_img_dir, val_lbl_dir)

    yaml_content = textwrap.dedent('''
           path: split_dataset
           train: train/images
           val: val/images

           nc: 3
           names: ["Explosives", "Anti-personnel mine", "Anti-vehicle mine"]
           ''')

    with open("dataset.yaml", "w") as f:
        f.write(yaml_content.strip() + "\n")

    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=8, border_mode=0, p=0.5),
        A.CLAHE(clip_limit=(1.5, 3), tile_grid_size=(4, 8), p=0.3),  # контраст
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo'))  # Assuming YOLO format for bounding boxes
    Albumentations.p = 1.0  # enable albumentations globally in Ultralytics
    Albumentations.transforms = augmentations

    model = RTDETR("rtdetr-l.pt")
    model.info()

    results = model.train(data="dataset.yaml", epochs=25,
                          project="rtdetr-l-ultralytics", name="rtdetr-l", batch=8, device='0',
                          mosaic=False,
                          cache=True, plots=True,
                          augment=True,
                          save_period=3)  # time=3 (limit by training hours), fraction = 0.5 (to train of dataset))

    torch.save(model.model, "rtdetr-l-25.pt")