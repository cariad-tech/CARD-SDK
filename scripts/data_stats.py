#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset statistics with label & depth analytics.

Key behavior:
- "Average agg_depth points per image" is computed over images with >0 points (nonzero) to avoid bias.
- Also logs "including_zeros" variants for transparency.

Outputs (under --out, default Final_stats/):
  overall/
    overall_summary.json
    hist_agg_depth_overall_nonzero.png
    hist_agg_depth_overall_stats_nonzero.json
    (optional) hist_agg_depth_overall_including_zeros.png
    (optional) hist_agg_depth_overall_stats_including_zeros.json
    (optional) hist_pos_boxes_overall.png
    (optional) hist_neg_boxes_overall.png
    (optional) image_level_all.csv
    (always) per_top_folder_summary.json  [if --save-per-folder-json]
  by_top_folder/<top>/
    folder_summary.json                   [if --save-per-folder-json]
  by_sequence/<top>/<sequence_name>/
    sequence.json
    hist_agg_depth_nonzero.png
    hist_agg_depth_stats_nonzero.json
    (optional) hist_agg_depth_including_zeros.png
    (optional) hist_agg_depth_stats_including_zeros.json
    (optional) hist_pos_boxes.png
    (optional) hist_neg_boxes.png
    (optional) image_level.csv
  dataset_stats.csv                       (per-sequence flat table)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def outdirs_for_sequence(out_root: Path, top_folder: str, seq_name: str) -> Path:
    """
    Returns the per-sequence output dir:
      <out_root>/by_sequence/<top_folder>/<seq_name>
    """
    return ensure_outdir(out_root / "by_sequence" / top_folder / seq_name)

def outdir_overall(out_root: Path) -> Path:
    return ensure_outdir(out_root / "overall")

def outdir_top_folder(out_root: Path, top_folder: str) -> Path:
    return ensure_outdir(out_root / "by_top_folder" / top_folder)


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


# ---------- Trajectory / coverage ----------

def load_trajectory(traj_json: Path) -> Optional[List[Tuple[int, List[float]]]]:
    try:
        with open(traj_json, "r") as f:
            data = json.load(f)
        poses = data.get("trajectory_poses", [])
        return poses if len(poses) > 1 else None
    except Exception:
        return None

def distance_traveled_m(poses: Optional[List[Tuple[int, List[float]]]]) -> float:
    if not poses or len(poses) < 2:
        return 0.0
    p = np.array([[v[1][0], v[1][1], v[1][2]] for v in poses], dtype=np.float64)
    diffs = p[1:] - p[:-1]
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())

def duration_seconds(poses: Optional[List[Tuple[int, List[float]]]]) -> float:
    if not poses:
        return 0.0
    t0 = poses[0][0]; t1 = poses[-1][0]
    return (t1 - t0) / 1e6  # microseconds -> seconds


# ---------- Image & depth iteration ----------

def count_images(seq_dir: Path) -> Tuple[int, Dict[str, int]]:
    """Return (total_images, per_cam_counts)"""
    img_root = seq_dir / "img"
    per_cam = {}
    total = 0
    if not img_root.is_dir():
        return 0, per_cam
    for cam_dir in sorted([d for d in img_root.iterdir() if d.is_dir() and d.name.startswith("cam_")]):
        count = sum(1 for f in cam_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg")
        per_cam[cam_dir.name] = count
        total += count
    return total, per_cam

def iter_image_files(seq_dir: Path) -> Iterable[Path]:
    img_root = seq_dir / "img"
    if not img_root.is_dir():
        return []
    for cam_dir in sorted([d for d in img_root.iterdir() if d.is_dir() and d.name.startswith("cam_")]):
        for f in cam_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".jpg":
                yield f

def iter_agg_depth_npz(seq_dir: Path) -> Iterable[Path]:
    root = seq_dir / "agg_depth"
    if not root.is_dir():
        return []
    for cam_dir in sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("cam_")]):
        for f in cam_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".npz":
                yield f

def count_pts_in_npz(npz_path: Path) -> int:
    try:
        with np.load(npz_path, mmap_mode="r") as data:
            if "pts_cam" in data.files:
                return int(data["pts_cam"].shape[0])
            for key in data.files:
                if key.startswith("pts") and data[key].ndim == 2 and data[key].shape[1] >= 3:
                    return int(data[key].shape[0])
    except Exception:
        pass
    return 0

def extract_cam_and_ts(path: Path) -> Tuple[Optional[str], Optional[str]]:
    cam = path.parent.name if path.parent.name.startswith("cam_") else None
    m = TS_IN_NAME.match(path.name)
    ts = m.group(1) if m else None
    return cam, ts

def build_per_image_depth_map(seq_dir: Path) -> Dict[str, Dict[str, int]]:
    """
    Returns nested dict: {cam_x: {timestamp: num_points, ...}, ...}
    Only includes images that have an agg_depth .npz file (missing npz => absent here).
    """
    out: Dict[str, Dict[str, int]] = {}
    for npz_path in iter_agg_depth_npz(seq_dir):
        cam, ts = extract_cam_and_ts(npz_path)
        if cam is None or ts is None:
            cam = cam or "unknown_cam"
            ts = ts or npz_path.stem
        n = count_pts_in_npz(npz_path)
        out.setdefault(cam, {})[ts] = n
    return out


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


# ---------- Plotting / stats ----------

def save_histogram(values: List[int], title: str, out_png: Path, xlabel: str):
    if len(values) == 0:
        return
    plt.figure(figsize=(10, 6))
    bins = min(60, max(10, int(np.sqrt(len(values)))))
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def array_stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()) if arr.size > 0 else None,
        "median": float(np.median(arr)) if arr.size > 0 else None,
        "min": int(arr.min()) if arr.size > 0 else None,
        "max": int(arr.max()) if arr.size > 0 else None,
        "p10": float(np.percentile(arr, 10)) if arr.size > 0 else None,
        "p90": float(np.percentile(arr, 90)) if arr.size > 0 else None,
    }


# ---------- Per-sequence analysis ----------

def analyze_sequence(
    seq_dir: Path,
    out_root: Path,
    save_per_seq_json: bool,
    emit_image_level_csv: bool,
    global_image_rows: Optional[List[Dict]],
) -> Tuple[Dict, Dict]:
    """Compute stats for one sequence directory; returns (csv_row, payload)."""

    top_folder = seq_dir.parent.name
    seq_name = seq_dir.name
    seq_out = outdirs_for_sequence(out_root, top_folder, seq_name)

    print(f"\n[seq] Analyzing: {seq_dir}")
    print(f"[seq] Output dir: {seq_out}")

    # Trajectory
    poses = load_trajectory(seq_dir / "export" / "output.laz.trajectory.json")
    dist_m = distance_traveled_m(poses)
    duration_s = duration_seconds(poses)

    # Images
    total_images, per_cam_images = count_images(seq_dir)
    print(f"[seq] images total={total_images} per_cam={per_cam_images}")

    # Depth maps (agg_depth)
    per_image_depth_map = build_per_image_depth_map(seq_dir)  # {cam: {ts: n}}
    counts_including_zeros: List[int] = []
    counts_nonzero: List[int] = []

    per_cam_depth_stats_nonzero: Dict[str, Dict[str, float]] = {}
    per_cam_depth_stats_incz: Dict[str, Dict[str, float]] = {}

    for cam, ts_map in per_image_depth_map.items():
        vals = list(ts_map.values())
        if not vals:
            continue
        arr_all = np.array(vals, dtype=np.int64)
        arr_nz = arr_all[arr_all > 0]

        counts_including_zeros.extend(arr_all.tolist())
        counts_nonzero.extend(arr_nz.tolist())

        per_cam_depth_stats_incz[cam] = array_stats(arr_all)
        per_cam_depth_stats_nonzero[cam] = array_stats(arr_nz)

    num_npz = len(counts_including_zeros)          # images that have an agg_depth npz
    num_nonzero = len(counts_nonzero)              # subset with >0 points
    num_zero = num_npz - num_nonzero
    agg_total_points = int(np.sum(counts_including_zeros)) if num_npz > 0 else 0

    # Sequence-level stats (NONZERO primary)
    arr_nz = np.array(counts_nonzero, dtype=np.int64)
    arr_all = np.array(counts_including_zeros, dtype=np.int64)

    stats_nz = array_stats(arr_nz)
    stats_incz = array_stats(arr_all)

    # Histograms (saved inside seq folder)
    if stats_nz["count"] > 0:
        p = seq_out / "hist_agg_depth_nonzero.png"
        save_histogram(counts_nonzero, f"{seq_name}: agg_depth points / image (nonzero only)", p, "agg_depth points / image")
        with open(seq_out / "hist_agg_depth_stats_nonzero.json", "w") as f:
            json.dump(stats_nz, f, indent=2)
        print(f"[seq] saved hist(nonzero): {p}")
    if stats_incz["count"] > 0 and num_zero > 0:
        p = seq_out / "hist_agg_depth_including_zeros.png"
        save_histogram(counts_including_zeros, f"{seq_name}: agg_depth points / image (including zeros)", p, "agg_depth points / image")
        with open(seq_out / "hist_agg_depth_stats_including_zeros.json", "w") as f:
            json.dump(stats_incz, f, indent=2)
        print(f"[seq] saved hist(including zeros): {p}")

    # Labels
    per_img_pos: Dict[str, Dict[str, int]] = {}
    per_img_neg: Dict[str, Dict[str, int]] = {}
    per_img_other: Dict[str, Dict[str, int]] = {}
    imgs_pos: List[str] = []
    imgs_neg: List[str] = []

    labels_root = labels_dir_for_sequence(seq_dir)
    for img_path in iter_image_files(seq_dir):
        cam, ts = extract_cam_and_ts(img_path)
        if cam is None or ts is None:
            cam = cam or img_path.parent.name
            ts = ts or img_path.stem
        pos, neg, other = count_labels_for_image(img_path, labels_root)
        if pos > 0: imgs_pos.append(img_path.name)
        if neg > 0: imgs_neg.append(img_path.name)
        per_img_pos.setdefault(cam, {})[ts] = pos
        per_img_neg.setdefault(cam, {})[ts] = neg
        per_img_other.setdefault(cam, {})[ts] = other

    total_pos_boxes = sum(sum(ts_map.values()) for ts_map in per_img_pos.values())
    total_neg_boxes = sum(sum(ts_map.values()) for ts_map in per_img_neg.values())
    total_other_boxes = sum(sum(ts_map.values()) for ts_map in per_img_other.values())

    images_with_positive = len(imgs_pos)
    images_with_negative = len(imgs_neg)
    images_with_any_label = len({*imgs_pos, *imgs_neg})
    images_with_pos_only = len([b for b in imgs_pos if b not in imgs_neg])
    images_with_neg_only = len([b for b in imgs_neg if b not in imgs_pos])
    images_with_both = len([b for b in imgs_pos if b in imgs_neg])

    # Label histograms (only if >0 entries)
    pos_counts = [c for m in per_img_pos.values() for c in m.values() if c > 0]
    neg_counts = [c for m in per_img_neg.values() for c in m.values() if c > 0]
    if len(pos_counts) > 0:
        p = seq_out / "hist_pos_boxes.png"
        save_histogram(pos_counts, f"{seq_name}: POS boxes per image (>0 only)", p, "positive boxes per image")
        print(f"[seq] saved: {p}")
    if len(neg_counts) > 0:
        p = seq_out / "hist_neg_boxes.png"
        save_histogram(neg_counts, f"{seq_name}: NEG boxes per image (>0 only)", p, "negative boxes per image")
        print(f"[seq] saved: {p}")

    # Coverage & averages (nonzero primary)
    pos_boxes_per_image_mean = float(total_pos_boxes) / float(total_images) if total_images > 0 else 0.0
    neg_boxes_per_image_mean = float(total_neg_boxes) / float(total_images) if total_images > 0 else 0.0
    pos_boxes_per_labeled_image_mean = float(total_pos_boxes) / float(images_with_positive) if images_with_positive > 0 else 0.0
    neg_boxes_per_labeled_image_mean = float(total_neg_boxes) / float(images_with_negative) if images_with_negative > 0 else 0.0

    print(f"[seq] agg_depth: npz_files={num_npz} nonzero={num_nonzero} zero={num_zero} total_pts={agg_total_points}")
    print(f"[seq] avg (nonzero) pts/img={stats_nz['mean']}, avg (incl. zeros) pts/img={stats_incz['mean']}")
    print(f"[seq] labels: +={total_pos_boxes} -={total_neg_boxes} other={total_other_boxes} images_with_any={images_with_any_label}")

    # Optional per-sequence image-level CSV
    if emit_image_level_csv:
        seq_rows: List[Dict] = []
        for img_path in iter_image_files(seq_dir):
            cam, ts = extract_cam_and_ts(img_path)
            if cam is None or ts is None:
                cam = cam or img_path.parent.name
                ts = ts or img_path.stem
            pos, neg, other = count_labels_for_image(img_path, labels_root)
            depth_pts = per_image_depth_map.get(cam, {}).get(ts, 0)
            seq_rows.append({
                "sequence": seq_name,
                "top_folder": top_folder,
                "cam": cam,
                "timestamp": ts,
                "image": img_path.name,
                "pos_boxes": pos,
                "neg_boxes": neg,
                "other_boxes": other,
                "agg_depth_points": depth_pts,
            })
        if seq_rows:
            csv_path = seq_out / "image_level.csv"
            pd.DataFrame(seq_rows).to_csv(csv_path, index=False)
            print(f"[seq] wrote image-level CSV: {csv_path}")
            if global_image_rows is not None:
                global_image_rows.extend(seq_rows)

    # Per-sequence JSON payload (rich)
    seq_payload = {
        "sequence_path": str(seq_dir),
        "sequence_name": seq_name,
        "top_folder": top_folder,
        "distance_m": float(dist_m),
        "distance_km": float(dist_m / 1000.0),
        "duration_s": float(duration_s),
        "duration_h": float(duration_s / 3600.0),
        "num_images_total": int(total_images),
        "num_images_per_cam": per_cam_images,
        # Depth aggregates
        "num_agg_depth_files": int(num_npz),
        "num_images_with_agg_depth_nonzero": int(num_nonzero),
        "num_images_with_agg_depth_zero": int(num_zero),
        "agg_depth_total_points": int(agg_total_points),
        # Averages (NONZERO primary)
        "agg_depth_points_per_image_mean": stats_nz["mean"],  # <-- headline per sequence (nonzero only)
        "agg_depth_points_per_image_median": stats_nz["median"],
        "agg_depth_points_per_image_min": stats_nz["min"],
        "agg_depth_points_per_image_max": stats_nz["max"],
        "agg_depth_points_per_image_p10": stats_nz["p10"],
        "agg_depth_points_per_image_p90": stats_nz["p90"],
        # Also keep including-zeros for auditing
        "agg_depth_points_per_image_mean_including_zeros": stats_incz["mean"],
        "agg_depth_points_per_image_median_including_zeros": stats_incz["median"],
        "agg_depth_points_per_image_min_including_zeros": stats_incz["min"],
        "agg_depth_points_per_image_max_including_zeros": stats_incz["max"],
        "agg_depth_points_per_image_p10_including_zeros": stats_incz["p10"],
        "agg_depth_points_per_image_p90_including_zeros": stats_incz["p90"],
        "per_cam_depth_points_stats_nonzero": per_cam_depth_stats_nonzero,
        "per_cam_depth_points_stats_including_zeros": per_cam_depth_stats_incz,
        # Full per-image maps
        "per_image_depth_points": per_image_depth_map,
        "per_image_positive_boxes": per_img_pos,
        "per_image_negative_boxes": per_img_neg,
        "per_image_other_boxes": per_img_other,
        # Sequence-level label stats
        "total_positive_boxes": int(total_pos_boxes),
        "total_negative_boxes": int(total_neg_boxes),
        "total_other_boxes": int(total_other_boxes),
        "images_with_any_label": int(images_with_any_label),
        "images_with_positive": int(images_with_positive),
        "images_with_negative": int(images_with_negative),
        "images_with_positive_only": int(images_with_pos_only),
        "images_with_negative_only": int(images_with_neg_only),
        "images_with_both_pos_and_neg": int(images_with_both),
        "pos_boxes_per_image_mean": float(pos_boxes_per_image_mean),
        "neg_boxes_per_image_mean": float(neg_boxes_per_image_mean),
        "pos_boxes_per_labeled_image_mean": float(pos_boxes_per_labeled_image_mean),
        "neg_boxes_per_labeled_image_mean": float(neg_boxes_per_labeled_image_mean),
        "list_images_with_positive": sorted(imgs_pos),
        "list_images_with_negative": sorted(imgs_neg),
    }

    # Write per-sequence JSON into its folder
    if save_per_seq_json:
        with open(seq_out / "sequence.json", "w") as f:
            json.dump(seq_payload, f, indent=2)
        print(f"[seq] wrote JSON: {seq_out / 'sequence.json'}")

    # Also return a row for top-level CSV (flat)
    csv_row = {
        k: (json.dumps(v, sort_keys=True) if k in ("num_images_per_cam",) else v)
        for k, v in seq_payload.items()
        if k not in (
            "per_image_depth_points",
            "per_image_positive_boxes",
            "per_image_negative_boxes",
            "per_image_other_boxes",
            "list_images_with_positive",
            "list_images_with_negative",
            "per_cam_depth_points_stats_nonzero",
            "per_cam_depth_points_stats_including_zeros",
        )
    }
    return csv_row, seq_payload


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics for the multi-modal driving dataset.")
    parser.add_argument("root", type=str, help="Root folder, e.g. /media/T7/cariad dataset/")
    parser.add_argument("--out", type=str, default="Final_stats", help="Output folder for CSV, per-sequence folders, histograms & JSON")
    parser.add_argument("--no-per-seq-json", action="store_true",
                        help="Do not save a JSON file per sequence (default is to save).")
    parser.add_argument("--save-per-folder-json", action="store_true",
                        help="Also write per-top-folder rollups to JSON.")
    parser.add_argument("--emit-image-level-csv", action="store_true",
                        help="Write per-sequence image-level CSVs and a global CSV with image rows.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out).resolve()
    ensure_outdir(out_root)

    sequences = find_sequences(root)
    if not sequences:
        print(f"No sequences found under: {root}")
        return

    print(f"Found {len(sequences)} sequences under {root}")

    rows: List[Dict] = []
    per_seq_payloads: List[Dict] = []

    overall_points_nonzero: List[int] = []
    overall_points_incz: List[int] = []
    overall_pos_per_image: List[int] = []
    overall_neg_per_image: List[int] = []
    global_image_rows: List[Dict] = [] if args.emit_image_level_csv else None

    per_folder_agg: Dict[str, Dict[str, float]] = {}  # top_folder → aggregates

    for seq in tqdm(sequences, desc="Analyzing sequences"):
        row, payload = analyze_sequence(
            seq,
            out_root,
            save_per_seq_json=not args.no_per_seq_json,
            emit_image_level_csv=args.emit_image_level_csv,
            global_image_rows=global_image_rows,
        )
        rows.append(row)
        per_seq_payloads.append(payload)

        # Accumulate overall depth distributions
        # Nonzero:
        for cam, ts_map in payload.get("per_image_depth_points", {}).items():
            overall_points_nonzero.extend([v for v in ts_map.values() if v > 0])
            overall_points_incz.extend(list(ts_map.values()))

        # Accumulate label dists for overall hist
        for cam, ts_map in payload.get("per_image_positive_boxes", {}).items():
            overall_pos_per_image.extend([c for c in ts_map.values() if c > 0])
        for cam, ts_map in payload.get("per_image_negative_boxes", {}).items():
            overall_neg_per_image.extend([c for c in ts_map.values() if c > 0])

        # Per-top-folder aggregation (with nonzero awareness)
        tf = payload["top_folder"]
        agg = per_folder_agg.setdefault(tf, {
            "top_folder": tf,
            "num_sequences": 0,
            "distance_m": 0.0,
            "distance_km": 0.0,
            "duration_s": 0.0,
            "num_images_total": 0,
            "num_agg_depth_files": 0,
            "num_agg_depth_files_nonzero": 0,
            "num_agg_depth_files_zero": 0,
            "agg_depth_total_points": 0,
            "total_positive_boxes": 0,
            "total_negative_boxes": 0,
            "total_other_boxes": 0,
            "images_with_any_label": 0,
        })
        agg["num_sequences"] += 1
        agg["distance_m"] += float(payload.get("distance_m", 0.0) or 0.0)
        agg["distance_km"] += float(payload.get("distance_km", 0.0) or 0.0)
        agg["duration_s"] += float(payload.get("duration_s", 0.0) or 0.0)
        agg["num_images_total"] += int(payload.get("num_images_total", 0) or 0)
        agg["num_agg_depth_files"] += int(payload.get("num_agg_depth_files", 0) or 0)
        agg["num_agg_depth_files_nonzero"] += int(payload.get("num_images_with_agg_depth_nonzero", 0) or 0)
        agg["num_agg_depth_files_zero"] += int(payload.get("num_images_with_agg_depth_zero", 0) or 0)
        agg["agg_depth_total_points"] += int(payload.get("agg_depth_total_points", 0) or 0)
        agg["total_positive_boxes"] += int(payload.get("total_positive_boxes", 0) or 0)
        agg["total_negative_boxes"] += int(payload.get("total_negative_boxes", 0) or 0)
        agg["total_other_boxes"] += int(payload.get("total_other_boxes", 0) or 0)
        agg["images_with_any_label"] += int(payload.get("images_with_any_label", 0) or 0)

    # Save per-sequence CSV
    df = pd.DataFrame(rows)
    csv_path = out_root / "dataset_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[ok] Wrote per-sequence CSV: {csv_path}")

    # Per-top-folder JSON(s)
    overall_dir = outdir_overall(out_root)
    if args.save_per_folder_json and per_folder_agg:
        # derive averages (nonzero primary)
        folders_out_list = []
        for tf, folder in per_folder_agg.items():
            denom_nz = max(1, int(folder.get("num_agg_depth_files_nonzero", 0)))
            denom_all = max(1, int(folder.get("num_agg_depth_files", 0)))
            folder["avg_agg_depth_points_per_image"] = float(folder.get("agg_depth_total_points", 0)) / float(denom_nz)
            folder["avg_agg_depth_points_per_image_including_zeros"] = float(folder.get("agg_depth_total_points", 0)) / float(denom_all)
            # Write per-folder file
            tf_dir = outdir_top_folder(out_root, tf)
            with open(tf_dir / "folder_summary.json", "w") as f:
                json.dump(folder, f, indent=2)
            folders_out_list.append(folder)
            print(f"[ok] Wrote folder summary: {tf_dir / 'folder_summary.json'}")
        # Also a single combined file under overall/
        with open(overall_dir / "per_top_folder_summary.json", "w") as f:
            json.dump({"folders": folders_out_list}, f, indent=2)
        print(f"[ok] Wrote combined per_top_folder_summary.json")

    # Overall aggregates (build totals first; add histogram-derived stats after)
    totals = {}
    if not df.empty:
        for col in ["distance_m", "distance_km", "duration_s"]:
            if col in df.columns:
                totals[col] = float(df[col].fillna(0).sum())
        totals["num_sequences"] = int(len(df))
        totals["num_images_total"] = int(df["num_images_total"].fillna(0).sum())
        totals["num_agg_depth_files"] = int(df["num_agg_depth_files"].fillna(0).sum())
        totals["agg_depth_total_points"] = int(df["agg_depth_total_points"].fillna(0).sum())
        # Label totals if present
        for col in ["total_positive_boxes", "total_negative_boxes", "total_other_boxes", "images_with_any_label"]:
            if col in df.columns:
                totals[col] = int(df[col].fillna(0).sum())

    # Overall histograms: NONZERO (primary)
    if len(overall_points_nonzero) > 0:
        p = overall_dir / "hist_agg_depth_overall_nonzero.png"
        save_histogram(overall_points_nonzero, "Overall: agg_depth points per image (nonzero only)", p, "agg_depth points / image")
        arr_nz = np.array(overall_points_nonzero, dtype=np.int64)
        stats_nz = array_stats(arr_nz)
        with open(overall_dir / "hist_agg_depth_overall_stats_nonzero.json", "w") as f:
            json.dump(stats_nz, f, indent=2)
        print(f"[ok] Wrote overall nonzero hist + stats in {overall_dir}")

        totals.update({
            "images_with_agg_depth_nonzero": stats_nz["count"],
            "agg_depth_points_per_image_mean": stats_nz["mean"],  # <-- HEADLINE (nonzero only)
            "agg_depth_points_per_image_median": stats_nz["median"],
            "agg_depth_points_per_image_min": stats_nz["min"],
            "agg_depth_points_per_image_max": stats_nz["max"],
            "agg_depth_points_per_image_p10": stats_nz["p10"],
            "agg_depth_points_per_image_p90": stats_nz["p90"],
        })

    # Overall histograms: INCLUDING ZEROS (optional, only if zeros exist)
    zeros_exist = (len(overall_points_incz) > len(overall_points_nonzero))
    if len(overall_points_incz) > 0 and zeros_exist:
        p = overall_dir / "hist_agg_depth_overall_including_zeros.png"
        save_histogram(overall_points_incz, "Overall: agg_depth points per image (including zeros)", p, "agg_depth points / image")
        arr_all = np.array(overall_points_incz, dtype=np.int64)
        stats_all = array_stats(arr_all)
        with open(overall_dir / "hist_agg_depth_overall_stats_including_zeros.json", "w") as f:
            json.dump(stats_all, f, indent=2)
        print(f"[ok] Wrote overall including-zeros hist + stats in {overall_dir}")

        totals.update({
            "images_with_agg_depth_including_zeros": stats_all["count"],
            "agg_depth_points_per_image_mean_including_zeros": stats_all["mean"],
            "agg_depth_points_per_image_median_including_zeros": stats_all["median"],
            "agg_depth_points_per_image_min_including_zeros": stats_all["min"],
            "agg_depth_points_per_image_max_including_zeros": stats_all["max"],
            "agg_depth_points_per_image_p10_including_zeros": stats_all["p10"],
            "agg_depth_points_per_image_p90_including_zeros": stats_all["p90"],
        })

        totals["images_with_agg_depth_zero"] = int(stats_all["count"] - totals.get("images_with_agg_depth_nonzero", 0))

    # Optional overall histograms for label counts
    if len(overall_pos_per_image) > 0:
        save_histogram(
            overall_pos_per_image,
            title="Overall: POS boxes per image (>0 only)",
            out_png=overall_dir / "hist_pos_boxes_overall.png",
            xlabel="positive boxes per image",
        )
    if len(overall_neg_per_image) > 0:
        save_histogram(
            overall_neg_per_image,
            title="Overall: NEG boxes per image (>0 only)",
            out_png=overall_dir / "hist_neg_boxes_overall.png",
            xlabel="negative boxes per image",
        )

    # Optional: write a global image-level CSV (potentially large!)
    if args.emit_image_level_csv and global_image_rows:
        global_csv = overall_dir / "image_level_all.csv"
        pd.DataFrame(global_image_rows).to_csv(global_csv, index=False)
        print(f"[ok] Wrote GLOBAL image-level CSV: {global_csv} (rows={len(global_image_rows)})")

    # Now write overall summary JSON
    if totals:
        overall_json = overall_dir / "overall_summary.json"
        with open(overall_json, "w") as f:
            json.dump(totals, f, indent=2)
        print(f"[ok] Wrote overall summary: {overall_json}")

    # Console summary (human-readable)
    total_km = totals.get("distance_km", 0.0)
    total_hours = totals.get("duration_s", 0.0) / 3600.0 if "duration_s" in totals else 0.0
    print("\n=== Dataset Summary ===")
    print(f"Sequences:                                  {len(sequences)}")
    print(f"Total distance:                             {total_km:.2f} km")
    print(f"Total duration:                             {total_hours:.2f} h")
    print(f"Total images:                               {totals.get('num_images_total', 0)}")
    print(f"Total agg_depth NPZ:                        {totals.get('num_agg_depth_files', 0)}")
    print(f"Total agg_depth pts:                        {totals.get('agg_depth_total_points', 0)}")
    if "agg_depth_points_per_image_mean" in totals:
        print(f"AVG agg_depth pts/img (nonzero only):       {totals['agg_depth_points_per_image_mean']:.2f}  <-- headline")
    if "agg_depth_points_per_image_mean_including_zeros" in totals:
        print(f"AVG agg_depth pts/img (including zeros):    {totals['agg_depth_points_per_image_mean_including_zeros']:.2f}")
    print(f"Total POS boxes:                            {totals.get('total_positive_boxes', 0)}")
    print(f"Total NEG boxes:                            {totals.get('total_negative_boxes', 0)}")
    print(f"Total OTHER boxes:                          {totals.get('total_other_boxes', 0)}")
    print(f"Images with any label (sum across seqs):    {totals.get('images_with_any_label', 0)}")
    print(f"Outputs saved under:                        {out_root}")

if __name__ == "__main__":
    main()



#python data_stats.py "/media/T7/x\ dataset/" --out Final_stats --save-per-folder-json --emit-image-level-csv
