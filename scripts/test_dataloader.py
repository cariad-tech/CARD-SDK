#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2025 CARIAD SE
"""Check a CARD split or inspect one sample.

Run from the SDK checkout:
python -m scripts.test_dataloader --base_root /path/to/data --split_file "$PWD/splits/val.txt" --test_integrity
"""

import argparse
import sys
import time
import numpy as np
import cv2
import torch
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from card_dataset.dataloader import UnifiedCardDataset

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def chw_to_hwc_uint8(img_t, mean, std):
    if img_t is None: return None
    img = img_t.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = (img * std + mean)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def depth_to_colormap(depth_chw, vmin=0.1, vmax=80.0):
    if depth_chw is None: return None
    d = depth_chw[0].detach().cpu().numpy()
    d = np.nan_to_num(d, nan=0.0)
    if d.size == 0: return None
    d = np.clip(d, vmin, vmax)
    norm = (d - vmin) / (vmax - vmin)
    d_uint8 = (norm * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(d_uint8, cv2.COLORMAP_TURBO)[:, :, ::-1]
    alpha = (d > 0.01).astype(np.float32)[..., None]
    return (cm.astype(np.float32) * alpha).astype(np.uint8)

def overlay(img, cm, a=0.6):
    if cm is None: return img
    mask = (cm.sum(axis=2) > 0).astype(np.float32)[..., None]
    return ((a * cm + (1 - a) * img) * mask + img * (1 - mask)).astype(np.uint8)

def draw_bboxes(img, labels, color=(255, 255, 0), thickness=2):
    if labels is None or len(labels) == 0: return img
    out = img.copy()
    for label in labels:
        x1, y1, x2, y2 = label['bbox']
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(out, str(label['class']), (int(x1), int(y1)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out

def backproject(depth_chw, K_3x3):
    d = depth_chw[0].detach().cpu().numpy()
    K = K_3x3.detach().cpu().numpy()
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    m = d > 0
    if not np.any(m): return np.zeros((0,3)), np.zeros((0,2))
    z = d[m]; u=u[m]; v=v[m]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], 1), np.stack([u, v], 1)

def transform(pts, T4x4):
    if pts.size == 0: return pts
    ph = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (T4x4 @ ph.T).T[:, :3]

def project(pts, K, hwc):
    H, W = hwc[0], hwc[1]
    z = pts[:, 2]
    m = z > 1e-3
    u = (K[0,0] * pts[:,0] / z) + K[0,2]
    v = (K[1,1] * pts[:,1] / z) + K[1,2]
    m = m & (u>=0) & (u<W) & (v>=0) & (v<H)
    return np.stack([u,v], 1), m

def draw_pts(img, uv, m, color=(0,255,0), r=1):
    out = img.copy()
    pts = uv[m].astype(np.int32)
    for x,y in pts:
        cv2.circle(out, (x,y), r, color, -1)
    return out

def run_integrity_check(ds):
    """
    Iterates over the entire dataset to ensure every sample loads without error.
    """
    print(f"\n{'='*60}")
    print("🚀 STARTING DATASET INTEGRITY TEST")
    print(f"   Total Samples: {len(ds)}")
    print(f"{'='*60}\n")

    success_count = 0
    fail_count = 0
    failed_indices = []
    interrupted = False

    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(range(len(ds)), desc="Testing Loading")
    except ImportError:
        iterator = range(len(ds))
        print("(Install 'tqdm' for a progress bar)")

    start_t = time.time()
    
    for i in iterator:
        try:
            sample = ds[i]
            if sample is None:
                fail_count += 1
                failed_indices.append((i, "Loader returned None (missing or filtered sample)"))
                continue
            success_count += 1
        except KeyboardInterrupt:
            print("\n[Stop] Integrity check interrupted by user.")
            interrupted = True
            break
        except Exception as e:
            fail_count += 1
            failed_indices.append((i, str(e)))

    total_time = time.time() - start_t
    
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS")
    print(f"{'='*60}")
    print(f"   Time Elapsed : {total_time:.2f}s")
    checked_count = success_count + fail_count
    print(f"   Avg Time/Item: {total_time/max(1, checked_count):.4f}s")
    print(f"   ✅ SUCCESS   : {success_count}")
    print(f"   ❌ FAILED    : {fail_count}")
    
    if fail_count > 0:
        print("\n   ⚠️  FAILURE REPORT:")
        for idx, error in failed_indices[:20]: # Show first 20 errors
            print(f"      Idx {idx}: {error}")
        if fail_count > 20:
            print(f"      ... and {fail_count - 20} more.")
    elif not interrupted and checked_count > 0:
        print("\n   ✨ PASSED: All samples loaded correctly.")
    else:
        print("\n   NOT PASSED: Integrity check was interrupted or the split is empty.")
    print(f"{'='*60}\n")
    return not interrupted and fail_count == 0 and checked_count > 0


def run_single_visualization(ds, idx, out_dir, skip_projections):
    """
    Detailed visualization for a SINGLE index.
    """
    print(f"\n[Info] visualizing Index: {idx}")
    _ensure_dir(out_dir)

    try:
        sample = ds[idx]
    except Exception as e:
        print(f"[Error] Failed to load index {idx}: {e}")
        return

    if sample is None:
        print("[Error] Loader returned None (Sample might be empty/filtered).")
        return

    meta = sample["meta"]
    center_cam = meta["center_cam"]
    frames = sample[center_cam]["frames"]
    f0 = frames[0]

    # Unpack stats
    mean, std = ds.rgb_mean[0,0], ds.rgb_std[0,0]

    # 1. RGB
    rgb = chw_to_hwc_uint8(f0["rgb"], mean, std)
    cv2.imwrite(str(out_dir / "center_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    # 2. Labels
    if f0.get("labels"):
        rgb_lbl = draw_bboxes(rgb, f0["labels"])
        cv2.imwrite(str(out_dir / "center_labels.png"), cv2.cvtColor(rgb_lbl, cv2.COLOR_RGB2BGR))
        print(f"   -> Saved Labels ({len(f0['labels'])} boxes)")

    # 3. Ground Plane
    if f0.get("ground_plane_info"):
        gp = f0["ground_plane_info"]
        print(f"   -> Ground Plane: Normal={gp['plane_normal_world'].numpy()}, H={gp['camera_height_above_ground']:.2f}m")
        with open(out_dir / "gp_info.txt", "w") as f:
            f.write(str(gp))

    # 4. Sparse/LAZ Depth
    sparse = f0.get("lidar_sparse")
    if sparse is not None:
        cm = depth_to_colormap(sparse)
        over = overlay(rgb, cm)
        cv2.imwrite(str(out_dir / "lidar_sparse_overlay.png"), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
        print(f"   -> Saved Sparse LiDAR overlay (Pts: {torch.count_nonzero(sparse)})")
    
        # 5. Reprojections
        if not skip_projections:
            print("   -> Running Reprojection Check...")
            pts_c, _ = backproject(sparse, f0["K"])
            T_cw = f0["T_cam_world"].numpy()
            pts_w = transform(pts_c, T_cw) # Cam -> World

            for off, nf in frames.items():
                if off == 0: continue
                # Project World -> Neighbor
                T_nw = nf["T_cam_world"].numpy()
                rgb_n = chw_to_hwc_uint8(nf["rgb"], mean, std)
                
                # World -> Neighbor = inv(T_neighbor_world)
                pts_n = transform(pts_w, np.linalg.inv(T_nw))
                uv, valid = project(pts_n, nf["K"].numpy(), rgb_n.shape)
                
                vis = draw_pts(rgb_n, uv, valid)
                cv2.imwrite(str(out_dir / f"reproj_neighbor_{off}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                print(f"      -> Reprojected to neighbor {off}")

    print(f"[Success] Visualization saved to: {out_dir}\n")

def main():
    parser = argparse.ArgumentParser(description="Unified Data Loader Tester")
    
    # Paths
    parser.add_argument("--base_root", required=True, help="Path to dataset root")
    parser.add_argument("--split_file", required=True, help="Path to a file in splits/ (absolute or relative to base_root)")
    parser.add_argument("--out_dir", default="./viz_results", help="Output folder")
    
    # Mode Selection
    parser.add_argument("--test_integrity", action="store_true", help="Run Unit Test on ALL indices")
    parser.add_argument("--idx", type=int, default=0, help="Index to visualize (if not testing all)")
    
    # Dataset Config
    parser.add_argument("--load_ground_plane", action="store_true", help="Compute ground planes")
    parser.add_argument("--no_labels", action="store_true", help="Disable labels")
    parser.add_argument("--no_stereo", action="store_true", help="Disable stereo")
    parser.add_argument("--skip_projections", action="store_true", help="Skip heavy reprojection logic in viz")

    args = parser.parse_args()

    # Smart Split Path Handling
    # If user provides full path to split file, handle it gracefully
    split_path = Path(args.split_file)
    base_path = Path(args.base_root)
    
    if base_path in split_path.parents:
        # User gave absolute path inside base_root, make it relative
        final_split_file = str(split_path.relative_to(base_path))
    else:
        final_split_file = args.split_file

    print(f"[Init] Loading Split: {final_split_file}")
    
    # Init Dataset
    # We use 'for_inference' for checking or 'for_visualization' for viz. 
    # 'for_visualization' is generally safer for inspection.
    ds = UnifiedCardDataset.for_visualization(
        base_root=args.base_root,
        split_file=final_split_file,
        load_ground_plane_info=args.load_ground_plane,
        load_labels=not args.no_labels,
        stereo=not args.no_stereo,
    )

    if args.test_integrity:
        if not run_integrity_check(ds):
            raise SystemExit(1)
    else:
        out_path = Path(args.out_dir) / f"idx_{args.idx}"
        run_single_visualization(ds, args.idx, out_path, args.skip_projections)

if __name__ == "__main__":
    main()