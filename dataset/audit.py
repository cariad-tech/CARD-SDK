# audit_neighbors_and_poses.py
import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch

# import YOUR dataset (adjust path/module name if needed)
from dataloader import UnifiedCardDataset  # <-- your class file


# ---------------- Utilities: filenames, timestamps, sequence root ----------------

def parse_cam_and_ts(relpath: str) -> Tuple[Optional[str], Optional[int]]:
    name = Path(relpath).name
    m = re.match(r"(cam_\d+)_(\d+)\.(jpg|jpeg|png)$", name, re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))

def seq_root_from_rel(base_root: Path, rel: str) -> Path:
    # everything up to ".../img/<cam>/<file>" is the sequence root
    parts = rel.replace("\\", "/").split("/img/")
    if len(parts) < 2:
        # fallback: ascend until "img" is a parent
        p = base_root / rel
        for q in p.parents:
            if q.name == "img":
                return q.parent
        # last resort: go up three
        return p.parent.parent.parent
    return (base_root / parts[0])

def list_camera_dir(seq_root: Path, cam: str) -> List[Tuple[int, Path]]:
    cam_dir = seq_root / "img" / cam
    out = []
    if cam_dir.is_dir():
        for p in cam_dir.iterdir():
            if not p.is_file():
                continue
            m = re.match(rf"{cam}_(\d+)\.(jpg|jpeg|png)$", p.name, re.IGNORECASE)
            if m:
                out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out

def find_index_by_ts(lst: List[Tuple[int, Path]], ts: int) -> int:
    # exact match, else -1
    times = [t for t, _ in lst]
    try:
        return times.index(ts)
    except ValueError:
        return -1


# ---------------- Pose math (no SciPy dep): SLERP + linear translation -----------

def quat_slerp_sf(q0_sf, q1_sf, t) -> np.ndarray:
    """
    Slerp for scalar-first quaternions [qw, qx, qy, qz].
    Returns scalar-first.
    """
    q0 = np.array([q0_sf[0], q0_sf[1], q0_sf[2], q0_sf[3]], dtype=np.float64)
    q1 = np.array([q1_sf[0], q1_sf[1], q1_sf[2], q1_sf[3]], dtype=np.float64)

    # ensure shortest path
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # If very close, use nlerp
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        q = q / np.linalg.norm(q)
        return q.astype(np.float64)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    q = (s0 * q0) + (s1 * q1)
    q = q / np.linalg.norm(q)
    return q.astype(np.float64)

def pose_to_matrix(x, y, z, qw, qx, qy, qz) -> np.ndarray:
    # convert scalar-first quaternion to rotation matrix
    # (formula from quaternion to rotation)
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, xq, yq, zq = q
    R = np.array([
        [1 - 2*(yq*yq + zq*zq), 2*(xq*yq - zq*w),     2*(xq*zq + yq*w)],
        [2*(xq*yq + zq*w),     1 - 2*(xq*xq + zq*zq), 2*(yq*zq - xq*w)],
        [2*(xq*zq - yq*w),     2*(yq*zq + xq*w),     1 - 2*(xq*xq + yq*yq)]
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def interpolate_pose_scalar_first(traj: Dict, ts_us: int):
    """
    traj['trajectory_poses'] = [[timestamp_us, [x,y,z, qw,qx,qy,qz]], ...] sorted by time
    Linear translation + quaternion slerp (scalar-first).
    """
    poses = traj["trajectory_poses"]
    i = 0
    n = len(poses)
    while i < n and poses[i][0] <= ts_us:
        i += 1
    if i == 0:
        return poses[0][1]
    if i >= n:
        return poses[-1][1]

    ts0, p0 = poses[i-1]
    ts1, p1 = poses[i]
    # interpolation weight
    t = float(np.clip((ts_us - ts0) / max(1, (ts1 - ts0)), 0.0, 1.0))
    trans0 = np.array(p0[:3], dtype=np.float64)
    trans1 = np.array(p1[:3], dtype=np.float64)
    trans = (1.0 - t) * trans0 + t * trans1

    q0 = np.array(p0[3:], dtype=np.float64)  # [qw,qx,qy,qz]
    q1 = np.array(p1[3:], dtype=np.float64)
    q = quat_slerp_sf(q0, q1, t)             # scalar-first
    return [float(trans[0]), float(trans[1]), float(trans[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]

def T_cam_world_from_traj(traj: Dict, cam: str, ts_img_us: int) -> np.ndarray:
    """
    T_cam_world(ts) = T_rig_world(ts + timestamp_offset) @ T_cam_to_rig
    """
    ci = traj["camera_infos"][cam]
    ts_eff = ts_img_us + ci.get("timestamp_offset", 0)
    xr, yr, zr, qw, qx, qy, qz = interpolate_pose_scalar_first(traj, ts_eff)
    T_rig_world = pose_to_matrix(xr, yr, zr, qw, qx, qy, qz)
    xs, ys, zs, qw2, qx2, qy2, qz2 = traj["sensor_to_trajectory_poses"][cam]
    T_cam_rig = pose_to_matrix(xs, ys, zs, qw2, qx2, qy2, qz2)
    return (T_rig_world @ T_cam_rig).astype(np.float64)  # cam->world


# ---------------- Audit procedure ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_root", required=True, type=str)
    ap.add_argument("--split_file", required=True, type=str)  # can be absolute or relative to base_root
    ap.add_argument("--idx", type=int, default=0)
    args = ap.parse_args()

    base_root = Path(args.base_root)
    split_file = Path(args.split_file)
    if not split_file.is_absolute():
        split_file = base_root / split_file

    # read split
    lines = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    if args.idx < 0 or args.idx >= len(lines):
        raise IndexError(f"idx {args.idx} out of range 0..{len(lines)-1}")

    center_rel = lines[args.idx]
    cam_center, ts_center = parse_cam_and_ts(center_rel)
    assert cam_center and ts_center is not None, f"Cannot parse camera/timestamp from: {center_rel}"

    seq_root = seq_root_from_rel(base_root, center_rel)
    traj_path = seq_root / "export" / "output.laz.trajectory.json"
    traj = json.loads(traj_path.read_text())

    # enumerate camera dir & find neighbors by filename order
    cam_list = list_camera_dir(seq_root, cam_center)
    idx_center_camlist = find_index_by_ts(cam_list, ts_center)
    if idx_center_camlist < 0:
        raise RuntimeError(f"Center timestamp {ts_center} not found in {seq_root}/img/{cam_center}")

    idx_m1 = idx_center_camlist - 1 if idx_center_camlist - 1 >= 0 else None
    idx_p1 = idx_center_camlist + 1 if idx_center_camlist + 1 < len(cam_list) else None

    # ground-truth (by enumeration) neighbors
    gt = {}
    gt[0]  = cam_list[idx_center_camlist]
    if idx_m1 is not None:
        gt[-1] = cam_list[idx_m1]
    if idx_p1 is not None:
        gt[+1] = cam_list[idx_p1]

    print("\n=== FILE-ENUMERATION (ground truth by folder order) ===")
    for off in (-1, 0, +1):
        if off in gt:
            ts, p = gt[off]
            print(f"  t{off:+}: {ts}  ->  {p.relative_to(seq_root)}")
        else:
            print(f"  t{off:+}: N/A (boundary)")

    # poses by direct trajectory interpolation (+ timestamp_offset)
    print("\n=== POSES FROM TRAJECTORY (cam->world) ===")
    T_gt = {}
    for off in (-1, 0, +1):
        if off in gt:
            ts = gt[off][0]
            T = T_cam_world_from_traj(traj, cam_center, ts)
            T_gt[off] = T
            tvec = T[:3, 3]
            # print compact pose info
            print(f"  t{off:+}: ts={ts}  |  t=[{tvec[0]:.3f} {tvec[1]:.3f} {tvec[2]:.3f}]  |  ||R^T R - I||={np.linalg.norm(T[:3,:3].T @ T[:3,:3] - np.eye(3)):.2e}")

    # now pull the same item from YOUR dataset and compare
    ds = UnifiedCardDataset(
        base_root=str(base_root),
        split_file=str(split_file) if split_file.is_absolute() else str(split_file.relative_to(base_root)),
        mode="val",
        height=352, width=1216,
        context_offsets=(-1, 0, +1),
        stereo=False,
        use_color_aug=False, use_geom_aug=False
    )
    item = ds[args.idx]
    cam_key = item["meta"]["center_cam"]
    frames = item[cam_key]["frames"]

    print("\n=== DATASET RETURNED FRAMES ===")
    for off in (-1, 0, +1):
        f = frames.get(off, None)
        if f is None:
            print(f"  t{off:+}: None")
            continue
        ts_ds = int(f["timestamp_us"])
        # reconstruct expected path from timestamp (for visibility)
        # try common extensions
        p_guess = None
        for ext in ("jpg", "jpeg", "png", "JPG", "PNG", "JPEG"):
            cand = seq_root / "img" / cam_key / f"{cam_key}_{ts_ds}.{ext}"
            if cand.is_file():
                p_guess = cand
                break
        print(f"  t{off:+}: ts={ts_ds}  path_guess={p_guess.relative_to(seq_root) if p_guess else '(not found)'}")

    # compare timestamps chosen by dataset vs by enumeration
    print("\n=== TIMESTAMP MATCH CHECK (dataset vs folder order) ===")
    for off in (-1, 0, +1):
        f = frames.get(off, None)
        if f is None and off not in gt:
            print(f"  t{off:+}: ✅ both N/A")
            continue
        if f is None and off in gt:
            print(f"  t{off:+}: ❌ dataset missing but folder has {gt[off][0]}")
            continue
        if f is not None and off not in gt:
            print(f"  t{off:+}: ❌ dataset has {int(f['timestamp_us'])} but folder has no neighbor")
            continue
        ts_ds = int(f["timestamp_us"])
        ts_gt = gt[off][0]
        ok = (ts_ds == ts_gt)
        print(f"  t{off:+}: {'✅' if ok else '❌'}  dataset_ts={ts_ds} vs folder_ts={ts_gt}")

    # compare poses dataset vs recomputed
    print("\n=== POSE MATCH CHECK (||T_ds - T_gt||_F) ===")
    for off in (-1, 0, +1):
        f = frames.get(off, None)
        if f is None or off not in T_gt:
            print(f"  t{off:+}: N/A")
            continue
        T_ds = f["T_cam_world"].numpy().astype(np.float64)
        T_ref = T_gt[off]
        frob = np.linalg.norm(T_ds - T_ref, ord="fro")
        print(f"  t{off:+}: diff Frobenius = {frob:.3e}  -> {'✅ close' if frob < 1e-5 else ('⚠️ small' if frob < 1e-3 else '❌ large')}")

    # also show relative neighbor->center (sanity)
    def inv_SE3(T):
        R = T[:3,:3]; t = T[:3,3]
        Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
        return Ti
    if (0 in T_gt) and ((-1 in T_gt) or (+1 in T_gt)):
        print("\n=== RELATIVE TRANSFORMS (neighbor -> center) ===")
        for off in (-1, +1):
            if off not in T_gt:
                continue
            Trel = inv_SE3(T_gt[0]) @ T_gt[off]
            tvec = Trel[:3,3]
            print(f"  T_{off:+}->0  t=[{tvec[0]:.3f} {tvec[1]:.3f} {tvec[2]:.3f}]  (meters)")

    print("\nDone. If you see any ❌ above, that’s the smoking gun (neighbor selection or pose timing).")


if __name__ == "__main__":
    main()
