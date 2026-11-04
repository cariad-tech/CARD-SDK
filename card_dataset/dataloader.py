# SPDX-License-Identifier: MIT
# Copyright 2025 CARIAD SE
import bisect
import functools
import json
import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset, get_worker_info

from . import torch_augs as augs
from .camera import undistort_bboxes
from .ground_plane import road_plane_at


# ---------- utilities ----------
def _rgb(path: Path) -> np.ndarray:
    import PIL.Image as Image

    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def _normalize(rgb_uint8: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = rgb_uint8.astype(np.float32) / 255.0
    return (x - mean) / std


def _chw(x: Optional[np.ndarray]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32))


def _depth_in_range(z: Optional[np.ndarray], dmin=0.1, dmax=80.0) -> Optional[np.ndarray]:
    # Depths outside [dmin, dmax] become 0 (= no GT) rather than being clamped to the bound
    if z is None:
        return None
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    if z.ndim == 2:
        z = z[..., None]
    z = np.where((z >= dmin) & (z <= dmax), z, 0.0)
    return z.astype(np.float32)


def _project_pts_to_depth(pts_cam: np.ndarray, K: np.ndarray, HW: Tuple[int, int]) -> np.ndarray:
    H, W = HW
    dep = np.zeros((H, W), np.float32)
    if pts_cam is None or len(pts_cam) == 0:
        return dep
    z = pts_cam[:, 2]
    ok = z > 0
    if not np.any(ok):
        return dep
    x, y, z = pts_cam[ok, 0], pts_cam[ok, 1], pts_cam[ok, 2]
    u = (K[0, 0] * x / z) + K[0, 2]
    v = (K[1, 1] * y / z) + K[1, 2]
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(m):
        return dep
    u, v, z = u[m], v[m], z[m]
    # Z-Buffer logic: Sort by depth so closer points overwrite further ones
    order = np.argsort(z)[::-1]
    u, v, z = u[order], v[order], z[order]
    dep[v.astype(int), u.astype(int)] = z
    return dep


def custom_collate_fn(batch: List[Any]) -> Any:
    """Batch valid samples without losing alignment of optional or variable-sized fields."""
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        raise ValueError("Cannot collate an empty batch or a batch containing only None")

    def collate(values):
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            return values
        element = values[0]
        if isinstance(element, torch.Tensor):
            if all(
                isinstance(value, torch.Tensor) and value.shape == element.shape for value in values
            ):
                return torch.stack(values, 0)
            return values
        if isinstance(element, dict):
            keys = dict.fromkeys(key for value in values for key in value)
            result = {}
            for key in keys:
                field_values = [value.get(key) for value in values]
                if key in (
                    "labels",
                    "wheel_points_world",
                    "timestamp_us",
                    "camera_height_above_ground",
                ):
                    result[key] = field_values
                else:
                    result[key] = collate(field_values)
            return result
        if isinstance(element, list):
            if all(isinstance(value, list) and len(value) == len(element) for value in values):
                return [collate(list(items)) for items in zip(*values)]
        return values

    return collate(batch)


# ---------- pose math ----------
def pose_to_matrix(x, y, z, qw, qx, qy, qz):
    if not np.isfinite([x, y, z, qw, qx, qy, qz]).all():
        raise ValueError("Pose components must be finite")
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def _slerp_scalar_first(q0, q1, t):
    q0_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]])
    q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]])
    Rends = Rotation.from_quat(np.stack([q0_xyzw, q1_xyzw], 0))
    s = Slerp([0.0, 1.0], Rends)
    q_xyzw = s([t]).as_quat()[0]
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


def interpolate_pose(traj: Dict, ts_us: int):
    poses = traj["trajectory_poses"]
    if not poses:
        raise ValueError("Trajectory has no poses")
    if not poses[0][0] <= ts_us <= poses[-1][0]:
        raise ValueError(f"Timestamp {ts_us} is outside trajectory coverage")
    i = 0
    while i < len(poses) and poses[i][0] <= ts_us:
        i += 1
    if i == 0:
        return poses[0][1]
    if i >= len(poses):
        return poses[-1][1]
    ts0, p0 = poses[i - 1]
    ts1, p1 = poses[i]
    if ts1 <= ts0:
        raise ValueError("Trajectory timestamps must be strictly increasing")
    t = (ts_us - ts0) / (ts1 - ts0)
    trans = (1.0 - t) * np.array(p0[:3]) + t * np.array(p1[:3])
    quat = _slerp_scalar_first(p0[3:], p1[3:], t)
    return [*trans.astype(float).tolist(), *quat]


def inv_SE3(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rel_pose(Tn_world, Tc_world):
    return inv_SE3(Tc_world) @ Tn_world


# ---------- dataset ----------
class UnifiedCardDataset(Dataset):
    """Comprehensive CARD reference loader prioritizing functionality over speed.

    Samples expose RGB, optional depth/labels, calibration, physical camera
    poses, temporal/stereo context, and optional ground-plane information.
    Applications can reuse these fields without changing the export readers.
    This implementation can be slow and is not an optimized training pipeline.
    Adapt the required functions, preprocessing, and I/O to each application's
    domain and throughput requirements before using it for large-scale training.
    """

    CAM_LEFT = "cam_1"
    CAM_RIGHT = "cam_0"

    def __init__(
        self,
        base_root: str,
        split_file: str,
        mode="train",
        height=1124,
        width=1126,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        context_offsets=(-1, 0, +1),
        stereo=False,
        stereo_time_tolerance_us=2000,
        min_sparse_points_train=200,
        use_color_aug=True,
        use_geom_aug=True,
        bumper_mask_path: Optional[str] = None,
        load_labels=True,
        load_neighbors=True,
        load_ground_plane_info=False,
        seed=0,
        agg_depth_variant: Optional[str] = None,
        min_depth=0.1,
        max_depth=80.0,
    ):
        super().__init__()
        self.base_root = Path(base_root)
        self.mode = mode
        self.H, self.W = height, width
        self.context = tuple(context_offsets)
        if height <= 0 or width <= 0:
            raise ValueError("Image height and width must be positive")
        if 0 not in self.context:
            raise ValueError("context_offsets must include the center frame (0)")
        if not 0 < min_depth < max_depth:
            raise ValueError("Depth bounds must satisfy 0 < min_depth < max_depth")
        if not np.isfinite([min_depth, max_depth]).all():
            raise ValueError("Depth bounds must be finite")
        if mode not in ("train", "val", "test"):
            raise ValueError("mode must be train, val, or test")
        if len(set(self.context)) != len(self.context) or any(not isinstance(offset, int) for offset in self.context):
            raise ValueError("context_offsets must contain unique integer offsets")
        self.stereo = stereo
        self.stereo_tol = stereo_time_tolerance_us
        self.min_sparse = min_sparse_points_train

        self.load_labels = load_labels
        self.load_neighbors = load_neighbors
        self.load_ground_plane_info = load_ground_plane_info
        self.agg_depth_variant = agg_depth_variant
        self.min_depth, self.max_depth = min_depth, max_depth

        self.seed = seed
        self._aug_random = random.Random(seed)
        self._aug_worker = None

        self.rgb_mean = np.array(mean, dtype=np.float32)[None, None, :]
        self.rgb_std = np.array(std, dtype=np.float32)[None, None, :]
        self.aug_rgb = (
            augs.Compose([augs.TorchColorAug()])
            if (mode == "train" and use_color_aug)
            else augs.Compose([])
        )
        if mode == "train" and use_geom_aug:
            self.aug_geo = augs.Compose(
                [
                    # 1. Zoom/Crop (Simulate focal length changes)
                    # scale=(0.5, 1.0) means crop area can be 50% to 100% of original
                    augs.TorchRandomResizedCrop(size=(height, width), scale=(0.5, 1.0)),
                    # 2. Horizontal Flip
                    augs.TorchGeoAug(hflip_p=0.5),
                ]
            )
        else:
            self.aug_geo = augs.Compose([])

        self.bumper_mask = None
        if bumper_mask_path:
            import PIL.Image as Image

            with Image.open(bumper_mask_path) as image:
                self.bumper_mask = np.array(image.convert("L"), np.float32) / 255.0

        with open(self.base_root / split_file, "r") as f:
            self.centers = [line.strip() for line in f if line.strip()]

        # Caches
        self._camdir_cache: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}

    # ----- path parsing -----
    @staticmethod
    def _parse_cam_ts(rel: str) -> Tuple[Optional[str], Optional[int]]:
        name = Path(rel).name
        m = re.match(r"(cam_\d+)_(\d+)\.(?:jpg|jpeg|png)$", name)
        if not m:
            return None, None
        return m.group(1), int(m.group(2))

    def _seq_root_from_rel(self, rel: str) -> Path:
        parts = rel.replace("\\", "/").split("/img/")
        if len(parts) != 2 or not parts[0] or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError(f"Expected a dataset-relative sequence/img/camera/image path: {rel}")
        return self.base_root / parts[0]

    # ----- RAM SAFE LOADING -----
    @staticmethod
    @functools.lru_cache(maxsize=8)  # Only keep 8 JSONs in memory per worker
    def _get_traj_cached(seq_root_str: str) -> Dict:
        traj_path = Path(seq_root_str) / "export" / "output.laz.trajectory.json"
        with open(traj_path, "r") as f:
            traj = json.load(f)
        timestamps = [pose[0] for pose in traj["trajectory_poses"]]
        if not timestamps or any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError(f"Trajectory timestamps must be nonempty and strictly increasing: {traj_path}")
        return traj

    def _get_traj(self, seq_root: Path) -> Dict:
        return self._get_traj_cached(str(seq_root))

    # ----- Geometry Helpers -----
    @staticmethod
    def _K_dist_from_traj(traj: Dict, cam: str):
        ci = traj["camera_infos"][cam]
        K = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1]], np.float32)
        dist_coeffs = ci.get("dist_coeffs")
        if dist_coeffs is None:
            raise ValueError(f"Missing distortion coefficients for camera {cam}")
        dist = np.asarray(dist_coeffs, dtype=np.float32)
        return K, dist

    @staticmethod
    def _T_cam_to_rig_from_traj(traj: Dict, cam: str):
        x, y, z, qw, qx, qy, qz = traj["sensor_to_trajectory_poses"][cam]
        return pose_to_matrix(x, y, z, qw, qx, qy, qz)

    def _T_cam_world(self, traj: Dict, cam: str, ts_img_us: int):
        ci = traj["camera_infos"][cam]
        ts = ts_img_us + ci["timestamp_offset"]
        x, y, z, qw, qx, qy, qz = interpolate_pose(traj, ts)
        T_rig_world = pose_to_matrix(x, y, z, qw, qx, qy, qz)
        T_cam_rig = self._T_cam_to_rig_from_traj(traj, cam)
        return T_rig_world @ T_cam_rig

    @staticmethod
    def _relative_cam_extrinsics(traj: Dict, cam_a: str, cam_b: str):
        T_a_rig = UnifiedCardDataset._T_cam_to_rig_from_traj(traj, cam_a)
        T_b_rig = UnifiedCardDataset._T_cam_to_rig_from_traj(traj, cam_b)
        T_ba = inv_SE3(T_b_rig) @ T_a_rig
        return T_ba[:3, :3].astype(np.float32), T_ba[:3, 3].astype(np.float32)

    @staticmethod
    def _stereo_rectify_params(traj: Dict, cam_left: str, cam_right: str):
        ci_l = traj["camera_infos"][cam_left]
        # Calculate these once per sequence/pair
        K_l, dist_l = UnifiedCardDataset._K_dist_from_traj(traj, cam_left)
        K_r, dist_r = UnifiedCardDataset._K_dist_from_traj(traj, cam_right)
        R_lr, t_lr = UnifiedCardDataset._relative_cam_extrinsics(traj, cam_left, cam_right)
        image_size = (int(ci_l["image_width"]), int(ci_l["image_height"]))
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K_l.astype(np.float64),
            dist_l.astype(np.float64),
            K_r.astype(np.float64),
            dist_r.astype(np.float64),
            image_size,
            R_lr.astype(np.float64),
            t_lr.astype(np.float64).reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
        )
        return {
            "K_left": K_l.astype(np.float32),
            "K_right": K_r.astype(np.float32),
            "dist_left": dist_l.astype(np.float32),
            "dist_right": dist_r.astype(np.float32),
            "R_left": R1.astype(np.float32),
            "R_right": R2.astype(np.float32),
            "P_left": P1.astype(np.float32),
            "P_right": P2.astype(np.float32),
            "Q": Q.astype(np.float32),
            "image_size": np.array(image_size, np.int32),
            "R_lr": R_lr,
            "t_lr": t_lr,
        }

    # ----- cam dir index -----
    def _cam_dir_list(self, seq_root: Path, cam: str) -> List[Tuple[int, str]]:
        key = (str(seq_root), cam)
        if key in self._camdir_cache:
            return self._camdir_cache[key]
        cam_dir = seq_root / "img" / cam
        out = []
        if cam_dir.is_dir():
            for p in cam_dir.iterdir():
                m = re.match(rf"{cam}_(\d+)\.(jpg|jpeg|png)$", p.name)
                if m:
                    out.append((int(m.group(1)), str(p)))
        out.sort(key=lambda x: x[0])
        self._camdir_cache[key] = out
        return out

    # ----- loaders -----
    def _prepare_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[..., None]
        if depth.ndim != 3 or depth.shape[2] != 1:
            raise ValueError(f"Expected depth with shape HxW or HxWx1, got {depth.shape}")
        if self.bumper_mask is not None:
            mask = cv2.resize(
                self.bumper_mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            depth = depth * mask[..., None]
        return _depth_in_range(depth, self.min_depth, self.max_depth)

    def _load_sparse(self, seq_root: Path, cam: str, ts: int):
        p = seq_root / "sparse_ground_truth" / f"{cam}_{ts}.npz"
        if not p.is_file():
            return None
        with np.load(p, allow_pickle=False) as archive:
            key = f"{cam}_{ts}"
            arr = archive[key] if key in archive else archive.get("depth")
        if arr is None:
            raise ValueError(f"Depth archive has no supported depth key: {p}")
        return self._prepare_depth(arr)

    def _load_single(self, seq_root: Path, cam: str, ts: int, K: np.ndarray, H: int, W: int):
        p = seq_root / "single_depth" / cam / f"{cam}_{ts}.npz"
        if not p.is_file():
            return None
        with np.load(p, allow_pickle=False) as archive:
            if "depth" in archive:
                dep = archive["depth"].astype(np.float32)
            elif "pts_cam" in archive:
                dep = _project_pts_to_depth(archive["pts_cam"].astype(np.float32), K, (H, W))
            else:
                raise ValueError(f"Depth archive must contain depth or pts_cam: {p}")
        return self._prepare_depth(dep)

    def _load_agg(
        self,
        seq_root: Path,
        cam: str,
        ts: int,
        K: np.ndarray,
        dist: np.ndarray,
        H: int,
        W: int,
        K_dense: Optional[np.ndarray] = None,
    ):
        agg_dir = self.agg_depth_variant if self.agg_depth_variant else "agg_depth"
        p = seq_root / agg_dir / cam / f"{cam}_{ts}.npz"
        if not p.is_file():
            return None
        with np.load(p, allow_pickle=False) as archive:
            if "pts_cam" in archive:
                dep = _project_pts_to_depth(archive["pts_cam"].astype(np.float32), K, (H, W))
                dense = False
            elif "depth" in archive:
                dep = archive["depth"].astype(np.float32)
                dense = True
            else:
                raise ValueError(f"Depth archive must contain depth or pts_cam: {p}")
        if dense:
            # Dense depth maps live in a pinhole frame with the
            # raw intrinsics K_dense, while the image was undistorted to K, so resample it into
            # K's frame: x_dense = K_dense @ inv(K) @ x. Same camera centre and rotation, so the
            # depth values themselves are unchanged.
            if dep.shape[:2] != (H, W):
                raise ValueError(f"Depth shape {dep.shape} does not match image {(H, W)}: {p}")
            if K_dense is not None and not np.allclose(K_dense, K):
                M = K_dense.astype(np.float64) @ np.linalg.inv(K.astype(np.float64))
                dep = cv2.warpPerspective(
                    dep,
                    M,
                    (W, H),
                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
        return self._prepare_depth(dep)

    def _load_single_lidar_sparse(
        self, seq_root: Path, cam: str, ts: int, K: np.ndarray, H: int, W: int
    ):
        roots = [
            seq_root / "single_lidar" / cam,
            seq_root / "lidar_single" / cam,
            seq_root / "sparse_lidar" / cam,
            seq_root / "lidar_sparse" / cam,
        ]
        candidates = [root / f"{cam}_{ts}.npz" for root in roots if (root / f"{cam}_{ts}.npz").is_file()]
        if not candidates:
            return None, None
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous single-LiDAR depth exports: {candidates}")
        npz = candidates[0]

        with np.load(npz, allow_pickle=False) as archive:
            pts_cam = archive.get("pts_cam")
            if pts_cam is not None:
                pts_cam = pts_cam.astype(np.float32)
            dep = archive.get("depth")
        if dep is None and pts_cam is not None:
            dep = _project_pts_to_depth(pts_cam, K, (H, W))
        elif dep is not None:
            dep = dep.astype(np.float32)
        else:
            raise ValueError(f"Depth archive must contain depth or pts_cam: {npz}")

        return self._prepare_depth(dep), pts_cam

    def _load_labels(self, seq_root: Path, cam: str, ts: int, img_path: str):
        p = seq_root / "labels" / f"{cam}_{ts}.txt"
        if not p.is_file():
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        out = []
        with open(p, "r") as f:
            for line in f:
                a = line.strip().split()
                if not a:
                    continue
                if len(a) != 5:
                    raise ValueError(f"Expected class and four normalized box values: {p}")
                cls, x_c, y_c, bw, bh = map(float, a)
                if not np.isfinite([cls, x_c, y_c, bw, bh]).all() or cls < 0 or cls != int(cls) or not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    raise ValueError(f"Invalid YOLO bounding box: {p}")
                x1, y1 = (x_c - bw / 2) * w, (y_c - bh / 2) * h
                x2, y2 = (x_c + bw / 2) * w, (y_c + bh / 2) * h
                out.append({"class": int(cls), "bbox": (x1, y1, x2, y2)})
        return out

    def _compute_road_plane_from_wheels(self, traj: Dict, timestamp_us: int, seq_root: Path):
        rig_to_world = pose_to_matrix(*interpolate_pose(traj, timestamp_us))
        return road_plane_at(traj, seq_root, timestamp_us, rig_to_world)

    def _compute_camera_height_from_ground_plane(self, traj, cam, ts, norm, pt):
        Tcw = self._T_cam_world(traj, cam, ts)
        cam_pt = Tcw[:3, 3]
        return float(np.dot(cam_pt - pt, norm / np.linalg.norm(norm)))

    # ----- shared augmentation params -----
    @staticmethod
    def _capture_aug_rng():
        # Snapshot both RNGs: color/flip draw from `random`, RRC draws from torch
        return random.getstate(), torch.get_rng_state()

    @staticmethod
    def _restore_aug_rng(state):
        random.setstate(state[0])
        torch.set_rng_state(state[1])

    def _sample_aug_rng(self):
        worker = get_worker_info()
        worker_key = None if worker is None else (worker.id, worker.seed)
        if worker_key != self._aug_worker:
            self._aug_random = random.Random(self.seed + worker.seed)
            self._aug_worker = worker_key
        sample_seed = self._aug_random.randrange(2**63)
        return random.Random(sample_seed).getstate(), torch.Generator().manual_seed(sample_seed).get_state()

    def _apply_augs_and_crop(self, rgb_uint8, depth_like, K, labels=None, lidar_pts_cam=None, rng_state=None):
        depth_keys = list(depth_like)
        for key, depth in depth_like.items():
            if depth.shape[:2] != rgb_uint8.shape[:2]:
                raise ValueError(f"Depth {key} shape {depth.shape} does not match RGB {rgb_uint8.shape}")
        original_rng = self._capture_aug_rng()
        try:
            if rng_state is not None:
                self._restore_aug_rng(rng_state)
            colored = self.aug_rgb(rgb_uint8, K)[0]
            rgb, *transformed = self.aug_geo(colored, *depth_like.values(), lidar_pts_cam, K)
        finally:
            self._restore_aug_rng(original_rng)
        points, intrinsics = transformed[-2:]
        scale_x, scale_y = self.W / rgb.shape[1], self.H / rgb.shape[0]
        depths = {}
        for key, depth in zip(depth_keys, transformed[:-2]):
            resized = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            depths[key] = resized[..., None] if resized.ndim == 2 else resized
        intrinsics = intrinsics.copy()
        intrinsics[0, :] *= scale_x
        intrinsics[1, :] *= scale_y
        final_labels = None if labels is None else []
        if labels is not None and not self.aug_geo.transforms:
            for label in labels:
                bounds = np.asarray(label["bbox"]) * [scale_x, scale_y, scale_x, scale_y]
                bounds = np.clip(bounds, 0, [self.W - 1, self.H - 1, self.W - 1, self.H - 1])
                left, top, right, bottom = bounds
                if right > left and bottom > top:
                    final_labels.append({**label, "bbox": tuple(bounds), "x1": left, "y1": top, "x2": right, "y2": bottom})
        return cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_LINEAR), depths, intrinsics, final_labels, points

    def _find_neighbors(self, idx: int, total: int, offsets: Tuple[int, ...]) -> Dict[int, int]:
        return {offset: idx + offset for offset in offsets if 0 <= idx + offset < total}

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, i: int):
        rel = self.centers[i]
        cam_center, ts_center = self._parse_cam_ts(rel)
        if cam_center is None:
            raise ValueError(f"Invalid camera image path in split at index {i}: {rel}")

        seq_root = self._seq_root_from_rel(rel)
        traj = self._get_traj(seq_root)

        cam_list = self._cam_dir_list(seq_root, cam_center)
        ts_list = [t for t, _ in cam_list]
        if ts_center not in ts_list:
            raise FileNotFoundError(
                f"Center image listed in split was not found: {self.base_root / rel}"
            )
        cidx = ts_list.index(ts_center)

        aug_rng = self._sample_aug_rng()

        def build_cam_payload(cam_name: str, center_idx: int, require_agg=False):
            K_raw, dist_raw = self._K_dist_from_traj(traj, cam_name)
            neigh = self._find_neighbors(
                center_idx, len(self._cam_dir_list(seq_root, cam_name)), self.context
            )

            # Init Rectification Maps ONCE
            ts_c, path_c = self._cam_dir_list(seq_root, cam_name)[neigh[0]]
            # Peek size
            rgb_c = _rgb(Path(path_c))
            H_raw, W_raw = rgb_c.shape[:2]

            newK, _ = cv2.getOptimalNewCameraMatrix(
                K_raw, dist_raw, (W_raw, H_raw), 0, (W_raw, H_raw)
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K_raw, dist_raw, None, newK, (W_raw, H_raw), cv2.CV_32FC1
            )

            K0 = newK.astype(np.float32)
            dist0 = np.zeros(5, np.float32)  # Effectively zeroed by remap

            frames, rels = {}, {}

            # --- CENTER ---
            rgb_c_ud = cv2.remap(rgb_c, mapx, mapy, interpolation=cv2.INTER_LINEAR)

            Tcw_c = self._T_cam_world(traj, cam_name, ts_c)
            sparse_c = self._load_sparse(seq_root, cam_name, ts_c)
            single_c = self._load_single(seq_root, cam_name, ts_c, K0, H_raw, W_raw)
            agg_c = self._load_agg(seq_root, cam_name, ts_c, K0, dist0, H_raw, W_raw, K_dense=K_raw)
            if agg_c is None and require_agg:
                raise FileNotFoundError(f"Requested depth variant {self.agg_depth_variant} is missing for {cam_name}_{ts_c} in {seq_root}")
            lidar_d_c, lidar_p_c = self._load_single_lidar_sparse(
                seq_root, cam_name, ts_c, K0, H_raw, W_raw
            )
            labels_c = (
                self._load_labels(seq_root, cam_name, ts_c, path_c) if self.load_labels else None
            )
            labels_c = undistort_bboxes(labels_c, K_raw, dist_raw, K0)

            depth_like = {}
            if sparse_c is not None:
                depth_like["sparse"] = sparse_c
            if single_c is not None:
                depth_like["depth_single"] = single_c
            if agg_c is not None:
                depth_like["depth_agg"] = agg_c
            if lidar_d_c is not None:
                depth_like["lidar_single_sparse"] = lidar_d_c

            # Augs (Note receiving 5th arg pts_aug)
            rgb_cc, dcts, Kc, labels_transformed, pts_c_aug = self._apply_augs_and_crop(
                rgb_c_ud, depth_like, K0, labels_c, lidar_pts_cam=lidar_p_c, rng_state=aug_rng
            )

            if (
                self.mode == "train"
                and "sparse" in dcts
                and dcts["sparse"] is not None
                and np.count_nonzero(dcts["sparse"]) < self.min_sparse
            ):
                raise ValueError(f"Insufficient sparse supervision for {cam_name}_{ts_c}: minimum {self.min_sparse} valid pixels required")

            # Ground Plane
            gp_info = None
            if self.load_ground_plane_info:
                pose_timestamp = ts_c + traj["camera_infos"][cam_name]["timestamp_offset"]
                norm, pt, wh, plane_source = self._compute_road_plane_from_wheels(traj, pose_timestamp, seq_root)
                h_cam = self._compute_camera_height_from_ground_plane(
                    traj, cam_name, ts_c, norm, pt
                )
                # Compute plane point & normal in camera coordinates
                # Tcw_c is camera->world (numpy) defined above
                T_world_to_cam = inv_SE3(Tcw_c)
                pt_h = np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float32)
                plane_point_cam = (T_world_to_cam @ pt_h)[:3].astype(np.float32)

                R_world_to_cam = T_world_to_cam[:3, :3].astype(np.float32)
                plane_normal_cam = R_world_to_cam @ norm.astype(np.float32)
                nrm = np.linalg.norm(plane_normal_cam)
                if nrm > 0:
                    plane_normal_cam = (plane_normal_cam / nrm).astype(np.float32)
                gp_info = {
                    "plane_normal_world": torch.from_numpy(norm.astype(np.float32)),
                    "plane_point_world": torch.from_numpy(pt.astype(np.float32)),
                    "plane_normal_cam": torch.from_numpy(plane_normal_cam.astype(np.float32)),
                    "plane_point_cam": torch.from_numpy(plane_point_cam.astype(np.float32)),
                    "wheel_points_world": torch.from_numpy(wh.astype(np.float32)),
                    "camera_height_above_ground": h_cam,
                    "source": plane_source,
                    "timestamp_us": pose_timestamp,
                }

            frames[0] = {
                "timestamp_us": ts_c,
                "rgb": _chw(_normalize(rgb_cc, self.rgb_mean, self.rgb_std)),
                "lidar_sparse": _chw(dcts.get("sparse", None)),
                "depth_single": _chw(dcts.get("depth_single", None)),
                "depth_agg": _chw(dcts.get("depth_agg", None)),
                "lidar_single_sparse": _chw(dcts.get("lidar_single_sparse", None)),
                "lidar_single_pts_cam": torch.from_numpy(pts_c_aug.astype(np.float32))
                if pts_c_aug is not None
                else None,
                "K": torch.from_numpy(Kc.astype(np.float32)),
                "dist": torch.from_numpy(dist0),
                "T_cam_world": torch.from_numpy(Tcw_c.astype(np.float32)),
                "labels": labels_transformed,
                "ground_plane_info": gp_info,
            }

            # --- NEIGHBORS ---
            if self.load_neighbors:
                clist_full = self._cam_dir_list(seq_root, cam_name)
                for off, nidx in neigh.items():
                    if off == 0:
                        continue
                    ts_n, path_n = clist_full[nidx]
                    rgb_n = _rgb(Path(path_n))
                    # UNIFIED REMAP (Geometric Consistency!)
                    rgb_n_ud = cv2.remap(rgb_n, mapx, mapy, interpolation=cv2.INTER_LINEAR)

                    Tcw_n = self._T_cam_world(traj, cam_name, ts_n)
                    sparse_n = self._load_sparse(seq_root, cam_name, ts_n)
                    lidar_d_n, _ = self._load_single_lidar_sparse(
                        seq_root, cam_name, ts_n, K0, H_raw, W_raw
                    )

                    dl_n = {}
                    if sparse_n is not None:
                        dl_n["sparse"] = sparse_n
                    if lidar_d_n is not None:
                        dl_n["lidar_single_sparse"] = lidar_d_n

                    rgb_nc, dcts_n, Kn, _, _ = self._apply_augs_and_crop(
                        rgb_n_ud, dl_n, K0, None, None, rng_state=aug_rng
                    )

                    frames[off] = {
                        "timestamp_us": ts_n,
                        "rgb": _chw(_normalize(rgb_nc, self.rgb_mean, self.rgb_std)),
                        "lidar_sparse": _chw(dcts_n.get("sparse", None)),
                        "lidar_single_sparse": _chw(dcts_n.get("lidar_single_sparse", None)),
                        "K": torch.from_numpy(Kn.astype(np.float32)),
                        "T_cam_world": torch.from_numpy(Tcw_n.astype(np.float32)),
                    }
                    T_rel = rel_pose(Tcw_n, Tcw_c)
                    rels[off] = torch.from_numpy(T_rel.astype(np.float32))

            return {
                "frames": frames,
                "rel_poses_to_center": rels,
                "K_ref": torch.from_numpy(K0),
                "dist_ref": torch.from_numpy(dist0),
                "T_cam_to_rig": torch.from_numpy(
                    self._T_cam_to_rig_from_traj(traj, cam_name).astype(np.float32)
                ),
            }

        cam_payload = build_cam_payload(cam_center, cidx, require_agg=bool(self.agg_depth_variant))

        out = {
            "meta": {
                "dataset_root": str(self.base_root),
                "sequence_root": str(seq_root),
                "mode": self.mode,
                "context": list(self.context),
                "stereo": self.stereo,
                "center_relpath": rel,
                "center_cam": cam_center,
                "center_timestamp_us": ts_center,
                "depth_agg_variant": self.agg_depth_variant or "agg_depth",
            },
            cam_center: cam_payload,
        }

        # Stereo
        if self.stereo:
            out["stereo"] = None
            other = self.CAM_RIGHT if cam_center == self.CAM_LEFT else self.CAM_LEFT
            other_l = self._cam_dir_list(seq_root, other)
            # Find nearest
            times = [t for t, _ in other_l]
            k = bisect.bisect_left(times, ts_center)
            cands = [k] + ([k - 1] if k > 0 else []) + ([k + 1] if k + 1 < len(times) else [])
            best, bestd = -1, 1e18
            for ix in cands:
                if 0 <= ix < len(times):
                    d = abs(times[ix] - ts_center)
                    if d < bestd:
                        bestd, best = d, ix

            if best >= 0 and bestd <= self.stereo_tol:
                op = build_cam_payload(other, best)
                if op:
                    out[other] = op
                    # Stereo Extrinsics
                    left_cam, right_cam = self.CAM_LEFT, self.CAM_RIGHT
                    if left_cam in out and right_cam in out:
                        si = self._stereo_rectify_params(traj, left_cam, right_cam)
                        R_lr, t_lr = si["R_lr"], si["t_lr"]
                        out["stereo"] = {
                            "left_cam": left_cam,
                            "right_cam": right_cam,
                            "baseline_m": float(np.linalg.norm(si["t_lr"])),
                            "R_right_from_left": torch.from_numpy(R_lr),
                            "t_right_from_left": torch.from_numpy(t_lr),
                            "rectification": {
                                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                                for k, v in si.items()
                            },
                        }

            if out["stereo"] is None:
                warnings.warn(f"No usable stereo pair within {self.stereo_tol} us for {rel}", RuntimeWarning, stacklevel=2)

        return out

    @classmethod
    def for_training(cls, base_root, split_file, **kwargs):
        """
        Factory for training. Defaults can be overridden by passing arguments.
        Example: UnifiedCardDataset.for_training(..., load_neighbors=False)
        """
        cfg = {
            "mode": "train",
            "load_neighbors": True,
            "load_labels": True,
            "use_color_aug": True,
            "use_geom_aug": True,
        }

        cfg.update(kwargs)

        return cls(base_root, split_file, **cfg)

    @classmethod
    def for_inference(cls, base_root, split_file, **kwargs):
        """
        Factory for inference/val. Defaults can be overridden by passing arguments.
        Example: UnifiedCardDataset.for_inference(..., load_neighbors=True)
        """
        cfg = {
            "mode": "val",
            "load_neighbors": False,
            "load_labels": False,
            "use_color_aug": False,
            "use_geom_aug": False,
            "context_offsets": (0,),  # Default to single frame, can be overridden
        }

        cfg.update(kwargs)

        return cls(base_root, split_file, **cfg)

    @classmethod
    def for_visualization(cls, base_root, split_file, **kwargs):
        """
        Factory for visualization. Minimal overhead.
        """
        cfg = {
            "mode": "val",
            "load_neighbors": True,  # Visualization usually wants context
            "load_labels": True,
            "use_color_aug": False,
            "use_geom_aug": False,
        }
        cfg.update(kwargs)

        return cls(base_root, split_file, **cfg)
