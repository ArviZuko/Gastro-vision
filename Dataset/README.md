# ?? Esophageal Cancer Dataset - Analysis & Preprocessing Report

The dataset images and their annotation files are stored in this directory under `train/`, `valid/`, and `test/`. Images use **Git LFS**; install Git LFS before cloning this repository and run `git lfs pull` to retrieve their full contents. The original Google Drive download is not required to obtain these checked-in files.

A pre-split, class-balanced dataset of endoscopic images for binary classification, covering two conditions: **Adenocarcinoma (AP)** and **Mucosal Pattern (MP)**.

## ?? Dataset Summary

| Metric | Value |
| :--- | :--- |
| **Total images** | 5,925 annotated endoscopic images |
| **Classes** | 2 (Adenocarcinoma (AP), Mucosal Pattern (MP)) |
| **Split** | Train 70% / Val 15% / Test 15% (stratified) |
| **Random seed** | 42 |
| **Image format** | JPEG |

## ?? Class Distribution

| Class | Total | Train | Val | Test |
| :--- | :--- | :--- | :--- | :--- |
| **AP (Adenocarcinoma)** | 2,962 | 2,073 | 444 | 445 |
| **MP (Mucosal Pattern)**| 2,963 | 2,074 | 444 | 445 |
| **Total** | **5,925** | **4,147** | **888** | **890** |

Split percentages achieved: Train 70%, Val 15%, Test 15%.

## ?? Folder Structure

``text
split_output/
+-- ?? class_distribution.png      # Bar chart of the split
+-- ?? split_manifest.json         # Per-image path + class + split record
+-- ?? split_summary.txt           # Plain-text summary
+-- ?? train/
¦   +-- AP/
¦   +-- MP/
+-- ?? val/
¦   +-- AP/
¦   +-- MP/
+-- ?? test/
    +-- AP/
    +-- MP/
``

## ?? Data Augmentation

Data augmentation techniques such as resizing and normalization were applied to prevent overfitting and improve generalization across diverse patient demographics. The original dataset was balanced prior to experimentation to reduce class imbalance and improve model training.

## ?? Image Properties

The dataset is optimized for deep learning models like YOLOv8, Faster R-CNN, SSD, and RT-DETR.

## ?? Suggested Usage

`python
import torchvision.datasets as datasets

train_ds = datasets.ImageFolder("split_output/train")
val_ds = datasets.ImageFolder("split_output/val")
test_ds = datasets.ImageFolder("split_output/test")
`

## ?? License

Licensed under the MIT License. Developed by SMARTLAB-SU.
