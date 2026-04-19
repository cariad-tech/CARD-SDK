#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 CARIAD SE
"""
Comprehensive CARD Dataset Dataloader Visualization

VISUALIZATION: Uses UnifiedCardDataset.for_visualization() 
for instant initialization and fast single-sample examination.

Features:
- RGB images with bounding box annotations
- Depth maps (sparse LiDAR, single-frame, aggregated)  
- Stereo pairs and rectification
- Temporal neighbor frames
- Ground plane computation from wheels
- Performance timing

Usage Examples:
# Fast visualization
python -m scripts.viz_dataloader --base_root "/path/to/dataset" --split_file "val.txt" --idx 0

# Minimal (fastest)  
python -m scripts.viz_dataloader --base_root "/path/to/dataset" --split_file "val.txt" --idx 0 \
  --no_labels --no_neighbors --skip_projections

# Full features with ground plane
python -m scripts.viz_dataloader --base_root "/path/to/dataset" --split_file "val.txt" --idx 0 \
  --load_ground_plane
"""

import argparse, os, sys, numpy as np, torch, cv2
from pathlib import Path
from card_dataset.dataloader import UnifiedCardDataset
from collections import defaultdict


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def chw_to_hwc_uint8(img_t, mean, std):
    img = img_t.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = (img * std + mean)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def depth_to_colormap(depth_chw, vmin=0.1, vmax=60.0):
    if depth_chw is None:
        return None
    d = depth_chw[0].detach().cpu().numpy()
    d = np.nan_to_num(d, nan=0.0)
    if d.size == 0:
        return None
    d = np.clip(d, vmin, vmax)
    rng = max(1e-6, (vmax - vmin))
    d_uint8 = (np.clip((d - vmin) / rng, 0.0, 1.0) * 255.0).astype(np.uint8)
    cm = cv2.applyColorMap(d_uint8, cv2.COLORMAP_TURBO)[:, :, ::-1]  # BGR->RGB
    alpha = (d > 0).astype(np.float32)[..., None]
    return (cm.astype(np.float32) * alpha + 255 * (1 - alpha)).astype(np.uint8)


def overlay(img, cm, a=0.6):
    return img if cm is None else (a * cm + (1 - a) * img).astype(np.uint8)


def backproject(depth_chw, K_3x3):
    d = depth_chw[0].detach().cpu().numpy()
    K = K_3x3.detach().cpu().numpy()
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    m = d > 0
    if not np.any(m):
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32)
    z = d[m].astype(np.float32)
    u = u[m].astype(np.float32)
    v = v[m].astype(np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], 1), np.stack([u, v], 1)


def project(pts_cam, K, hwc_shape):
    H, W = hwc_shape[0], hwc_shape[1]
    z = pts_cam[:, 2]
    m = z > 1e-6
    z = np.maximum(z, 1e-6)
    u = (K[0, 0] * pts_cam[:, 0] / z) + K[0, 2]
    v = (K[1, 1] * pts_cam[:, 1] / z) + K[1, 2]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    m = m & inb
    return np.stack([u, v], 1), m


def transform(pts, T4x4):
    if pts.size == 0:
        return pts
    N = pts.shape[0]
    ph = np.hstack([pts, np.ones((N, 1), np.float32)])
    return (T4x4 @ ph.T).T[:, :3]


def draw_pts(img, uv, m, color=(0, 255, 0), r=1):
    out = img.copy()
    pts = uv[m].astype(np.int32)
    if pts.size == 0:
        return out
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), r, color, -1)
    return out


def draw_bboxes(img, labels, color=(255, 255, 0), thickness=2):
    """Draw bounding boxes on image.
    
    Args:
        img: RGB image array (H, W, 3)
        labels: List of dicts with 'class' and 'bbox' (x1, y1, x2, y2) keys
        color: RGB color tuple for bounding box
        thickness: Line thickness
    
    Returns:
        Image with bounding boxes drawn
    """
    if labels is None or len(labels) == 0:
        return img
    
    out = img.copy()
    for label in labels:
        class_id = label['class']
        x1, y1, x2, y2 = label['bbox']
        
        # Draw bounding box
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # Draw class label
        text = f"Class {class_id}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(out, (int(x1), int(y1) - text_size[1] - 5), 
                      (int(x1) + text_size[0], int(y1)), color, -1)
        cv2.putText(out, text, (int(x1), int(y1) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return out


def _get_neighbor_frames(frames_dict):
    """Return (minus1, plus1) frame dicts or None if missing."""
    f_m1 = frames_dict.get(-1, None)
    f_p1 = frames_dict.get(+1, None)
    return f_m1, f_p1


def _save_png(path: Path, rgb):
    _ensure_dir(path.parent)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _do_projection_bundle(
    tag: str,
    out_root: Path,
    rgb0, K0, Tcw0, source_depth_chw,
    frames,  # dict: { -1: {...}, 0:{...}, +1:{...} }
    stereo_payload,  # dict or None
    mean_vec, std_vec,
    center_labels=None,  # bounding box labels for center frame
):
    """
    Project a given source (agg or sparse) from center@t into:
      - center@t
      - neighbor@t-1 (if exists)
      - neighbor@t+1 (if exists)
      - stereo mate (if exists)
    Save under: out_root/<tag>/...
    """
    root = out_root / tag
    _ensure_dir(root)

    # # 1) Visualize source on center
    # cm0 = depth_to_colormap(source_depth_chw)
    # over0 = overlay(rgb0, cm0, 0.6)
    
    # # Add bounding boxes to overlay if available
    # if center_labels is not None and len(center_labels) > 0:
    #     over0 = draw_bboxes(over0, center_labels)
    
    # _save_png(root / "t_center_overlay.png", over0)

    # 2) Backproject source@t -> 3D (center camera)
    pts_c, _ = backproject(source_depth_chw, torch.from_numpy(K0))
    if pts_c.shape[0] == 0:
        # Nothing to reproject, still save the plain center
        _save_png(root / "t_center_rgb.png", rgb0)
        return

    # 3) Reproject back to center (sanity)
    Pw = transform(pts_c, Tcw0)            # center@t -> world
    Twc0 = np.linalg.inv(Tcw0)             # world -> center
    Pc0 = transform(Pw, Twc0)
    uv0, m0 = project(Pc0, K0, rgb0.shape)
    rgb0_pts = draw_pts(rgb0, uv0, m0, (0, 255, 0), 1)
    
    # Add bounding boxes to reprojection
    if center_labels is not None and len(center_labels) > 0:
        rgb0_pts = draw_bboxes(rgb0_pts, center_labels)
    
    _save_png(root / "t_center_reproj_from_t.png", rgb0_pts)

    # 4) Reproject to neighbors (if available)
    f_m1, f_p1 = _get_neighbor_frames(frames)
    if f_m1 is not None and f_m1.get("rgb") is not None:
        rgb_m1 = chw_to_hwc_uint8(f_m1["rgb"], mean_vec, std_vec)
        Km1 = f_m1["K"].numpy(); Tcwm1 = f_m1["T_cam_world"].numpy()
        Twcm1 = np.linalg.inv(Tcwm1)
        Pcm1 = transform(Pw, Twcm1)
        uvm1, mm1 = project(Pcm1, Km1, rgb_m1.shape)
        _save_png(root / "t-1_reproj_from_t.png", draw_pts(rgb_m1, uvm1, mm1, (255, 200, 0), 1))

    if f_p1 is not None and f_p1.get("rgb") is not None:
        rgb_p1 = chw_to_hwc_uint8(f_p1["rgb"], mean_vec, std_vec)
        Kp1 = f_p1["K"].numpy(); Tcwp1 = f_p1["T_cam_world"].numpy()
        Twcp1 = np.linalg.inv(Tcwp1)
        Pcp1 = transform(Pw, Twcp1)
        uvp1, mp1 = project(Pcp1, Kp1, rgb_p1.shape)
        _save_png(root / "t+1_reproj_from_t.png", draw_pts(rgb_p1, uvp1, mp1, (255, 0, 200), 1))


def main():
    ap = argparse.ArgumentParser(description="Comprehensive CARD Dataset Dataloader Visualization")
    ap.add_argument("--base_root", required=True, type=str,
                    help="Dataset root folder (sequence roots and export live under here)")
    ap.add_argument("--split_file", required=True, type=str,
                    help="Split file (relative to base_root or absolute)")
    ap.add_argument("--idx", type=int, default=0,
                    help="Sample index to visualize")
    ap.add_argument("--out_dir", type=str, default="./viz_out",
                    help="Output directory for visualizations")
    
    # Image parameters
    ap.add_argument("--h", type=int, default=1216,
                    help="Target image height")
    ap.add_argument("--w", type=int, default=1216,
                    help="Target image width")
    
    # Dataset efficiency flags
    ap.add_argument("--no_stereo", action="store_true", 
                    help="Disable stereo pairing even if available")
    ap.add_argument("--no_labels", action="store_true", 
                    help="Disable loading labels for efficiency")
    ap.add_argument("--no_neighbors", action="store_true", 
                    help="Disable loading neighbor frames for efficiency")
    ap.add_argument("--load_ground_plane", action="store_true", 
                    help="Load ground plane information from wheel contact points")
    
    # Visualization options
    ap.add_argument("--skip_projections", action="store_true",
                    help="Skip depth projection visualizations (faster)")
    ap.add_argument("--depth_range", nargs=2, type=float, default=[0.1, 60.0],
                    help="Depth visualization range [min max] in meters")
    
    args = ap.parse_args()

    split_path = Path(args.split_file)
    if not split_path.is_absolute():
        split_path = Path(args.base_root) / args.split_file

    print("[info] Creating visualization-optimized dataset:")
    print(f"  Image size: {args.w}x{args.h}")
    print(f"  Labels: {'disabled' if args.no_labels else 'enabled'}")
    print(f"  Neighbors: {'disabled' if args.no_neighbors else 'enabled'}")
    print(f"  Stereo: {'disabled' if args.no_stereo else 'enabled'}")
    print(f"  Ground plane: {'enabled' if args.load_ground_plane else 'disabled'}")
    
    import time
    start_time = time.time()
    
    ds = UnifiedCardDataset.for_inference(
        base_root=args.base_root,
        split_file=str(split_path.relative_to(args.base_root)) if str(split_path).startswith(str(Path(args.base_root))) else str(split_path),
        height=args.h, width=args.w,
        load_labels=not args.no_labels,
        load_neighbors=not args.no_neighbors,
        load_ground_plane_info=args.load_ground_plane,
        stereo=not args.no_stereo,
        context_offsets=(-1, 0, +1) if not args.no_neighbors else (0,),
    )
    
    init_time = time.time() - start_time
    print(f"[info] Dataset initialized in {init_time:.2f}s")

    print(f"[info] Loading sample {args.idx}...")
    load_start = time.time()
    item = ds[args.idx]
    load_time = time.time() - load_start
    print(f"[info] Sample loaded in {load_time:.2f}s")
    center_cam = item["meta"]["center_cam"]
    center_frames = item[center_cam]["frames"]

    f0 = center_frames[0]
    mean = ds.rgb_mean[0, 0, :]
    std = ds.rgb_std[0, 0, :]

    # Center tensors
    rgb0 = chw_to_hwc_uint8(f0["rgb"], mean, std)
    K0 = f0["K"].numpy()
    Tcw0 = f0["T_cam_world"].numpy()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    
    # ---------- Basic visualizations ----------
    print("[info] Saving basic visualizations...")
    
    # Save plain center image
    _save_png(out_dir / "center_rgb.png", rgb0)
    
    # ---------- Bounding box visualization ----------
    labels = f0.get("labels", None)
    if labels is not None and len(labels) > 0:
        rgb0_with_bboxes = draw_bboxes(rgb0, labels)
        _save_png(out_dir / "center_with_bboxes.png", rgb0_with_bboxes)
        print(f"[info] Found {len(labels)} bounding boxes in center frame")
        
        # Save bounding box details
        with open(out_dir / "bboxes_info.txt", "w") as f:
            f.write(f"Bounding Boxes Information\n")
            f.write(f"==========================\n")
            f.write(f"Total bounding boxes: {len(labels)}\n")
            f.write(f"Image dimensions: {rgb0.shape[1]}x{rgb0.shape[0]} (WxH)\n\n")
            for i, label in enumerate(labels):
                x1, y1, x2, y2 = label['bbox']
                width = x2 - x1
                height = y2 - y1
                f.write(f"Box {i+1}:\n")
                f.write(f"  Class: {label['class']}\n")
                f.write(f"  Coordinates: ({x1}, {y1}) to ({x2}, {y2})\n")
                f.write(f"  Size: {width}x{height} pixels\n\n")
    else:
        print("[info] No bounding boxes found in center frame")
    
    # ---------- Ground plane information ----------
    if 'ground_plane' in item[center_cam]:
        gp = item[center_cam]['ground_plane']
        print(f"[info] Ground plane information available:")
        print(f"  Plane normal (world coords): {gp['plane_normal_world']}")
        print(f"  Plane point (world coords): {gp['plane_point_world']}")
        print(f"  Camera height above ground: {gp['camera_height_above_ground']:.3f}m")
        print(f"  Number of wheel points: {gp['wheel_points_world'].shape[0]}")
        
        # Save ground plane info to text file
        with open(out_dir / "ground_plane_info.txt", "w") as f:
            f.write(f"Ground Plane Information\n")
            f.write(f"========================\n")
            f.write(f"Computed from wheel ground contact points at timestamp {gp['timestamp_us']}\n\n")
            f.write(f"Plane normal (world coords): {gp['plane_normal_world']}\n")
            f.write(f"Plane point (world coords): {gp['plane_point_world']}\n") 
            f.write(f"Camera height above ground: {gp['camera_height_above_ground']:.6f} meters\n\n")
            f.write(f"Wheel ground contact points (world coords):\n")
            wheel_names = ["FL (Front Left)", "FR (Front Right)", "RL (Rear Left)", "RR (Rear Right)"]
            for i, point in enumerate(gp['wheel_points_world']):
                wheel_name = wheel_names[i] if i < len(wheel_names) else f"Wheel {i+1}"
                f.write(f"  {wheel_name}: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})\n")
    else:
        print("[info] Ground plane information not available (use --load_ground_plane to enable)")
    
    # ---------- Neighbor frames information ----------
    neighbor_count = len([k for k in center_frames.keys() if k != 0])
    print(f"[info] Loaded {neighbor_count} neighbor frames")
    
    if neighbor_count > 0:
        with open(out_dir / "neighbor_frames_info.txt", "w") as f:
            f.write(f"Neighbor Frames Information\n")
            f.write(f"===========================\n")
            f.write(f"Total neighbor frames: {neighbor_count}\n")
            f.write(f"Available frame offsets: {list(center_frames.keys())}\n\n")
            for offset, frame in center_frames.items():
                if offset != 0:
                    f.write(f"Frame at offset {offset:+d}:\n")
                    f.write(f"  Timestamp: {frame['timestamp_us']} microseconds\n")
                    f.write(f"  RGB shape: {frame['rgb'].shape}\n")
                    f.write(f"  Has labels: {frame.get('labels') is not None}\n\n")
    
    # ---------- Stereo information ----------
    if not args.no_stereo and "stereo" in item and item["stereo"] is not None:
        stereo_info = item["stereo"]
        print(f"[info] Stereo pairing available:")
        print(f"  Left camera: {stereo_info['left_cam']}")
        print(f"  Right camera: {stereo_info['right_cam']}")
        print(f"  Baseline: {stereo_info['baseline_m']:.3f}m")
        
        with open(out_dir / "stereo_info.txt", "w") as f:
            f.write(f"Stereo Configuration\n")
            f.write(f"====================\n")
            f.write(f"Left camera: {stereo_info['left_cam']}\n")
            f.write(f"Right camera: {stereo_info['right_cam']}\n")
            f.write(f"Baseline: {stereo_info['baseline_m']:.6f} meters\n\n")
            f.write(f"Relative transformation (right from left):\n")
            f.write(f"Rotation matrix:\n{stereo_info['R_right_from_left']}\n\n")
            f.write(f"Translation vector:\n{stereo_info['t_right_from_left']}\n")
    else:
        print("[info] No stereo pairing available")
    
    # ---------- Depth projection visualizations ----------
    if not args.skip_projections:
        print("[info] Creating depth projection visualizations...")
        
        vmin, vmax = args.depth_range
        
        # A) Projections using AGG depth (if available)
        if f0.get("depth_agg") is not None:
            print("  - Processing aggregated depth...")
            _do_projection_bundle(
                tag="out_agg",
                out_root=out_dir,
                rgb0=rgb0, K0=K0, Tcw0=Tcw0,
                source_depth_chw=f0["depth_agg"],
                frames=center_frames,
                stereo_payload=item.get("stereo", None),
                mean_vec=mean, std_vec=std,
                center_labels=labels,
            )
        else:
            print("  - No aggregated depth available")

        # B) Projections using SPARSE depth (prefer single scan over accumulated)
        sparse_src = None
        src_name = None
        if f0.get("lidar_single_sparse") is not None:
            sparse_src = f0["lidar_single_sparse"]
            src_name = "lidar_single_sparse (single LiDAR scan)"
        elif f0.get("lidar_sparse") is not None:
            sparse_src = f0["lidar_sparse"]
            src_name = "lidar_sparse (accumulated ground truth - may show multiple scans)"
            print("    WARNING: Using accumulated sparse ground truth instead of single scan")
            print("    This may show accumulated points from multiple LiDAR scans rather than a single scan")

        if sparse_src is not None:
            print(f"  - Processing {src_name}...")
            _do_projection_bundle(
                tag="out_sparse",
                out_root=out_dir,
                rgb0=rgb0, K0=K0, Tcw0=Tcw0,
                source_depth_chw=sparse_src,
                frames=center_frames,
                stereo_payload=item.get("stereo", None),
                mean_vec=mean, std_vec=std,
                center_labels=labels,
            )
        else:
            print("  - No sparse depth available")
    else:
        print("[info] Skipping depth projection visualizations (--skip_projections)")
    
    # ---------- Summary ----------
    print(f"\n[summary] Dataset loading summary:")
    print(f"  Sample index: {args.idx}")
    print(f"  Center camera: {center_cam}")
    print(f"  Image dimensions: {args.w}x{args.h}")
    print(f"  Labels: {'enabled' if not args.no_labels else 'disabled'} ({len(labels) if labels else 0} found)")
    print(f"  Neighbors: {'enabled' if not args.no_neighbors else 'disabled'} ({neighbor_count} loaded)")
    print(f"  Ground plane: {'enabled' if args.load_ground_plane else 'disabled'}")
    print(f"  Stereo: {'enabled' if not args.no_stereo else 'disabled'}")
    print(f"  Available depth types: {[k for k in f0.keys() if 'depth' in k or 'sparse' in k]}")

    print(f"\n[done] Wrote results under: {out_dir.resolve()}")

if __name__ == "__main__":
    main()


# python -m scripts.viz_dataloader   --base_root "<path to data root>"   --split_file "val_all_data_clean.txt"   --idx 68 --load_ground_plane

#     # Full comprehensive visualization (all features)
# python -m scripts.viz_dataloader \
#   --base_root "<path to data root>" \
#   --split_file "val.txt" \
#   --idx 6268 --load_ground_plane

# # Fast minimal visualization (center image only)
# python -m scripts.viz_dataloader \
#   --base_root "<path to data root>" \
#   --split_file "val.txt" \
#   --idx 6268 --no_labels --no_neighbors --no_stereo --skip_projections

# # Custom configuration
# python -m scripts.viz_dataloader \
#   --base_root "<path to data root>" \
#   --split_file "val.txt" \
#   --idx 6268 --load_ground_plane --skip_projections --h 480 --w 640