# card_dataset/scripts/viz_sample.py
# Example:
# python -m card_dataset.scripts.viz_sample \
#   --base_root "/media/T7/x dataset" \
#   --split_file "val_all_data_clean.txt" \
#   --idx 8465 \
#   --out_dir "./viz_out"

import argparse, os, sys, numpy as np, torch, cv2
from pathlib import Path
from card_dataset.dataloader import UnifiedCardDataset


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

    # 1) Visualize source on center
    cm0 = depth_to_colormap(source_depth_chw)
    over0 = overlay(rgb0, cm0, 0.6)
    _save_png(root / "t_center_overlay.png", over0)

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
    _save_png(root / "t_center_reproj_from_t.png", rgb0_pts)

    # 4) Reproject to neighbors
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

    # 5) Reproject to stereo mate (if present)
    if stereo_payload is not None:
        left = stereo_payload.get("left_cam")
        right = stereo_payload.get("right_cam")
        # Which one is "other" relative to our current center?
        # We'll just try both keys in the top-level item; only one will exist.
        # The caller passes the partner cam's payload via `item[other_cam]`.
        # Here we infer from frames: they belong to the current cam only,
        # so we need the other cam directly from the item dict (caller provides in args).
        # Instead, the caller will pass `stereo_partner_frame0` here for simplicity.

    # (Handled by caller separately for clarity.)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_root", required=True, type=str,
                    help="Dataset root folder (sequence roots and export live under here)")
    ap.add_argument("--split_file", required=True, type=str,
                    help="Split file (relative to base_root or absolute)")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="./viz_out")
    ap.add_argument("--h", type=int, default=352)
    ap.add_argument("--w", type=int, default=1216)
    ap.add_argument("--no_stereo", action="store_true", help="Disable stereo pairing even if available")
    args = ap.parse_args()

    # Allow absolute or relative split path
    split_path = Path(args.split_file)
    if not split_path.is_absolute():
        split_path = Path(args.base_root) / args.split_file

    ds = UnifiedCardDataset(
        base_root=args.base_root,
        split_file=str(split_path.relative_to(args.base_root)) if str(split_path).startswith(str(Path(args.base_root))) else str(split_path),
        mode="val",
        height=args.h, width=args.w,
        context_offsets=(-1, 0, +1),
        stereo=not args.no_stereo,
        use_color_aug=False, use_geom_aug=False
    )

    item = ds[args.idx]
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

    # ---------- A) Projections using AGG depth (if available) ----------
    if f0.get("depth_agg") is not None:
        _do_projection_bundle(
            tag="out_agg",
            out_root=out_dir,
            rgb0=rgb0, K0=K0, Tcw0=Tcw0,
            source_depth_chw=f0["depth_agg"],
            frames=center_frames,
            stereo_payload=item.get("stereo", None),
            mean_vec=mean, std_vec=std,
        )
    else:
        print("[info] No depth_agg at t — skipping agg projections.")

    # ---------- B) Projections using SPARSE depth ----------
    # Prefer lidar_single_sparse (projected single LiDAR), else lidar_sparse (GT sparse)
    sparse_src = None
    src_name = None
    if f0.get("lidar_single_sparse") is not None:
        sparse_src = f0["lidar_single_sparse"]
        src_name = "lidar_single_sparse"
    elif f0.get("lidar_sparse") is not None:
        sparse_src = f0["lidar_sparse"]
        src_name = "lidar_sparse"

    if sparse_src is not None:
        _do_projection_bundle(
            tag="out_sparse",
            out_root=out_dir,
            rgb0=rgb0, K0=K0, Tcw0=Tcw0,
            source_depth_chw=sparse_src,
            frames=center_frames,
            stereo_payload=item.get("stereo", None),
            mean_vec=mean, std_vec=std,
        )
    else:
        print("[info] No sparse depth at t — skipping sparse projections.")

    # ---------- C) Stereo reprojection to the partner camera ----------
    # If stereo pairing exists, reproject source depth to the partner cam at its t
    stereo = item.get("stereo", None)
    if stereo is not None:
        # identify partner cam payload
        other_cam = None
        if UnifiedCardDataset.CAM_LEFT in item and UnifiedCardDataset.CAM_RIGHT in item:
            # if we have both cams in the item, choose the *other* one
            other_cam = UnifiedCardDataset.CAM_RIGHT if center_cam == UnifiedCardDataset.CAM_LEFT else UnifiedCardDataset.CAM_LEFT
        elif UnifiedCardDataset.CAM_LEFT in item:
            other_cam = UnifiedCardDataset.CAM_LEFT if center_cam != UnifiedCardDataset.CAM_LEFT else None
        elif UnifiedCardDataset.CAM_RIGHT in item:
            other_cam = UnifiedCardDataset.CAM_RIGHT if center_cam != UnifiedCardDataset.CAM_RIGHT else None

        if other_cam is not None and item.get(other_cam) is not None:
            other_frames = item[other_cam]["frames"]
            if 0 in other_frames and other_frames[0] is not None:
                f_other0 = other_frames[0]
                rgb_o = chw_to_hwc_uint8(f_other0["rgb"], mean, std)
                Ko = f_other0["K"].numpy()
                Tcwo = f_other0["T_cam_world"].numpy()
                Twco = np.linalg.inv(Tcwo)

                # If we already backprojected once, reuse AGG if present, else SPARSE if present.
                # Re-run backprojection to keep code clean:
                def reproject_source(tag_name: str, depth_src):
                    root = Path(args.out_dir) / tag_name
                    _ensure_dir(root)
                    pts_c, _ = backproject(depth_src, torch.from_numpy(K0))
                    if pts_c.shape[0] == 0:
                        return
                    Pw = transform(pts_c, Tcw0)   # center@t -> world
                    Pco = transform(Pw, Twco)     # world -> other cam
                    uvo, mo = project(Pco, Ko, rgb_o.shape)
                    _save_png(root / "stereo_partner_reproj_from_t.png",
                              draw_pts(rgb_o, uvo, mo, (0, 180, 255), 1))

                if f0.get("depth_agg") is not None:
                    reproject_source("out_agg", f0["depth_agg"])
                if sparse_src is not None:
                    reproject_source("out_sparse", sparse_src)
        else:
            print("[info] Stereo enabled but no partner camera payload for this sample.")
    else:
        print("[info] Stereo pairing not present for this sample (or disabled).")

    print(f"[done] Wrote results under: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
