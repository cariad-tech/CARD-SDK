#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample random images with positive and negative labels from the dataset.

This script finds images with positive (class 1) and negative (class 0) labels,
randomly samples the specified number from each category, and copies them to
organized output folders: road_images/positive and road_images/negative
"""

import argparse
import json
import os
import re
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
from tqdm import tqdm

# ---------- Config / constants ----------

IMG_REGEX = re.compile(r"^(cam_\d+)_\d+\.jpg$")
TS_IN_NAME = re.compile(r".*_(\d+)\.(npz|jpg)$")  # cam_0_1234567890.{npz|jpg} → 1234567890
YOLO_LINE = re.compile(r"^\s*([0-9]+)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s+([0-9eE\.\+\-]+)\s*$")

POS_CLASS = 1
NEG_CLASS = 0

TOP_FOLDERS = ["germany_2days", "germany_batch2", "Munich_Inglostadt", "Nardo"]


# ---------- FS helpers ----------

def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------- Data discovery ----------

def is_sequence_dir(seq_dir: Path) -> bool:
    return (seq_dir / "export" / "output.laz.trajectory.json").is_file() and (seq_dir / "img").is_dir()

def find_sequences(root: Path) -> List[Path]:
    seqs = []
    for top in TOP_FOLDERS:
        base = root / top
        if not base.is_dir():
            continue
        for sub in base.rglob("*"):
            if sub.is_dir() and is_sequence_dir(sub):
                seqs.append(sub)
    return sorted(list(dict.fromkeys(seqs)))


# ---------- Image iteration ----------

def iter_image_files(seq_dir: Path) -> Iterable[Path]:
    img_root = seq_dir / "img"
    if not img_root.is_dir():
        return []
    for cam_dir in sorted([d for d in img_root.iterdir() if d.is_dir() and d.name.startswith("cam_")]):
        for f in cam_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".jpg":
                yield f


# ---------- Labels / YOLO parsing ----------

def labels_dir_for_sequence(seq_dir: Path) -> Optional[Path]:
    d = seq_dir / "labels"
    return d if d.is_dir() else None

def label_file_for_image(image_path: Path, labels_root: Path) -> Path:
    return labels_root / (image_path.stem + ".txt")

def parse_yolo_file(file_path: Path) -> List[int]:
    cls_ids: List[int] = []
    if not file_path.is_file():
        return cls_ids
    try:
        with open(file_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                m = YOLO_LINE.match(s)
                if m:
                    cid = int(float(m.group(1)))
                    cls_ids.append(cid)
                else:
                    parts = s.split()
                    try:
                        cid = int(float(parts[0]))
                        cls_ids.append(cid)
                    except Exception:
                        continue
    except Exception:
        pass
    return cls_ids

def count_labels_for_image(image_path: Path, labels_root: Optional[Path]) -> Tuple[int, int, int]:
    if labels_root is None:
        return (0, 0, 0)
    lf = label_file_for_image(image_path, labels_root)
    cls_ids = parse_yolo_file(lf)
    pos = sum(1 for c in cls_ids if c == POS_CLASS)
    neg = sum(1 for c in cls_ids if c == NEG_CLASS)
    other = len(cls_ids) - pos - neg
    return (pos, neg, max(other, 0))


# ---------- Image sampling ----------

def collect_labeled_images(sequences: List[Path]) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    """
    Collect all images with positive and negative labels.
    Returns: (positive_images, negative_images)
    Each item is (image_path, sequence_name)
    """
    positive_images = []
    negative_images = []
    
    for seq_dir in tqdm(sequences, desc="Collecting labeled images"):
        seq_name = f"{seq_dir.parent.name}/{seq_dir.name}"
        labels_root = labels_dir_for_sequence(seq_dir)
        
        if labels_root is None:
            continue
            
        for img_path in iter_image_files(seq_dir):
            pos, neg, other = count_labels_for_image(img_path, labels_root)
            
            if pos > 0:
                positive_images.append((img_path, seq_name))
            if neg > 0:
                negative_images.append((img_path, seq_name))
    
    return positive_images, negative_images

def sample_and_copy_images(
    image_list: List[Tuple[Path, str]], 
    output_dir: Path, 
    num_samples: int, 
    category: str
) -> None:
    """
    Randomly sample images from the list and copy them to output directory.
    """
    if len(image_list) == 0:
        print(f"No {category} images found!")
        return
    
    # Sample random images
    num_to_sample = min(num_samples, len(image_list))
    sampled_images = random.sample(image_list, num_to_sample)
    
    print(f"Sampling {num_to_sample} {category} images out of {len(image_list)} available")
    
    # Create output directory
    ensure_outdir(output_dir)
    
    # Copy images with informative names
    for i, (img_path, seq_name) in enumerate(tqdm(sampled_images, desc=f"Copying {category} images")):
        # Create filename with sequence info
        safe_seq_name = seq_name.replace("/", "_").replace(" ", "_")
        new_filename = f"{safe_seq_name}_{img_path.name}"
        output_path = output_dir / new_filename
        
        # Copy the image
        shutil.copy2(img_path, output_path)
        
        # Also copy the corresponding label file if it exists
        labels_root = labels_dir_for_sequence(img_path.parent.parent.parent)  # Go up to sequence dir
        if labels_root:
            label_file = label_file_for_image(img_path, labels_root)
            if label_file.exists():
                label_output = output_dir / (new_filename.replace('.jpg', '.txt'))
                shutil.copy2(label_file, label_output)
    
    print(f"Copied {num_to_sample} {category} images to {output_dir}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Sample random images with positive and negative labels from the dataset."
    )
    parser.add_argument("root", type=str, help="Root folder, e.g. /media/T7/cariad dataset/")
    parser.add_argument("--out", type=str, default="road_images", 
                       help="Output folder (default: road_images)")
    parser.add_argument("--num-positive", type=int, default=100,
                       help="Number of positive images to sample (default: 100)")
    parser.add_argument("--num-negative", type=int, default=100,
                       help="Number of negative images to sample (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--copy-labels", action="store_true",
                       help="Also copy the corresponding label files")
    
    args = parser.parse_args()

    # Set random seed for reproducible results
    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out).resolve()

    # Find all sequences
    sequences = find_sequences(root)
    if not sequences:
        print(f"No sequences found under: {root}")
        return

    print(f"Found {len(sequences)} sequences under {root}")

    # Collect all labeled images
    positive_images, negative_images = collect_labeled_images(sequences)
    
    print(f"\nFound:")
    print(f"  - {len(positive_images)} images with positive labels")
    print(f"  - {len(negative_images)} images with negative labels")

    # Sample and copy positive images
    if len(positive_images) > 0:
        positive_out_dir = out_root / "positive"
        sample_and_copy_images(positive_images, positive_out_dir, args.num_positive, "positive")
    else:
        print("No positive images found!")

    # Sample and copy negative images
    if len(negative_images) > 0:
        negative_out_dir = out_root / "negative"
        sample_and_copy_images(negative_images, negative_out_dir, args.num_negative, "negative")
    else:
        print("No negative images found!")

    print(f"\n=== Sampling Complete ===")
    print(f"Output directory: {out_root}")
    print(f"Positive images: {out_root / 'positive'}")
    print(f"Negative images: {out_root / 'negative'}")


if __name__ == "__main__":
    main()