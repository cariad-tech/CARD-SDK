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
from torch.utils.data import Dataset

from . import torch_augs as augs


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
        return {}

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
    i = 0
    while i < len(poses) and poses[i][0] <= ts_us:
        i += 1
    if i == 0:
        return poses[0][1]
    if i >= len(poses):
        return poses[-1][1]
    ts0, p0 = poses[i - 1]
    ts1, p1 = poses[i]
    t = float(np.clip((ts_us - ts0) / max(1, (ts1 - ts0)), 0.0, 1.0))
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
    """Load split-listed CARD images, precomputed depth, poses, and optional context."""

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
        self.stereo = stereo
        self.stereo_tol = stereo_time_tolerance_us
        self.min_sparse = min_sparse_points_train

        self.load_labels = load_labels
        self.load_neighbors = load_neighbors
        self.load_ground_plane_info = load_ground_plane_info
        self.agg_depth_variant = agg_depth_variant
        self.min_depth, self.max_depth = min_depth, max_depth

        random.seed(seed)

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
        if len(parts) < 2:
            return self.base_root  # fallback
        return self.base_root / parts[0]

    # ----- RAM SAFE LOADING -----
    @staticmethod
    @functools.lru_cache(maxsize=8)  # Only keep 8 JSONs in memory per worker
    def _get_traj_cached(seq_root_str: str) -> Dict:
        traj_path = Path(seq_root_str) / "export" / "output.laz.trajectory.json"
        with open(traj_path, "r") as f:
            return json.load(f)

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
        ts = ts_img_us + ci.get("timestamp_offset", 0)
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
            K_l, dist_l, K_r, dist_r, image_size, R_lr, t_lr, flags=cv2.CALIB_ZERO_DISPARITY
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
            return None
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
                return None
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
                return None
        if dense:
            # Dense depth map (e.g. fs_dense_depth_aligned). It lives in a pinhole frame with the
            # raw intrinsics K_dense, while the image was undistorted to K, so resample it into
            # K's frame: x_dense = K_dense @ inv(K) @ x. Same camera centre and rotation, so the
            # depth values themselves are unchanged.
            if dep.shape[:2] != (H, W):
                return None
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
        npz = None
        for r in roots:
            p = r / f"{cam}_{ts}.npz"
            if p.is_file():
                npz = p
                break
        if npz is None:
            return None, None

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
            return None, pts_cam

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
                if len(a) == 5:
                    cls, x_c, y_c, bw, bh = map(float, a)
                    x1, y1 = int((x_c * w) - (bw * w) / 2), int((y_c * h) - (bh * h) / 2)
                    x2, y2 = int((x_c * w) + (bw * w) / 2), int((y_c * h) + (bh * h) / 2)
                    out.append({"class": int(cls), "bbox": (x1, y1, x2, y2)})
        return out

    def _compute_road_plane_from_wheels(self, traj: Dict, timestamp_us: int):
        """Compute road plane from wheel ground contact points at a specific timestamp."""
        wheel_ground_keys = [
            "cariad_wheel_FL_ground",
            "cariad_wheel_FR_ground",
            "cariad_wheel_RL_ground",
            "cariad_wheel_RR_ground",
        ]
        wheel_ground_keys = [key.lower() for key in wheel_ground_keys]

        wheel_points_world = []
        wheel_trajectories = {
            key.lower(): value for key, value in traj.get("wheel_trajectories", {}).items()
        }
        sensor_poses = {
            key.lower(): value for key, value in traj.get("sensor_to_trajectory_poses", {}).items()
        }

        # Check if wheel trajectories exist (time-varying positions)
        if wheel_trajectories:
            # Use wheel_trajectories if available (true time-varying wheel positions)
            for wheel_key in wheel_ground_keys:
                if wheel_key in wheel_trajectories:
                    wheel_traj = wheel_trajectories[wheel_key]
                    wheel_pose = interpolate_pose({"trajectory_poses": wheel_traj}, timestamp_us)
                    wheel_point = np.array([wheel_pose[0], wheel_pose[1], wheel_pose[2]])
                    wheel_points_world.append(wheel_point)

        # If no time-varying wheel trajectories, use static positions transformed to world coordinates
        if len(wheel_points_world) == 0:
            for wheel_key in wheel_ground_keys:
                if wheel_key in sensor_poses:
                    # Static wheel position in vehicle/rig coordinates
                    wheel_pose = sensor_poses[wheel_key]

                    # Transform static wheel position to world coordinates at current timestamp
                    vehicle_pose = interpolate_pose(traj, timestamp_us)
                    T_vehicle_to_world = pose_to_matrix(*vehicle_pose)

                    # Create homogeneous coordinates for the wheel position
                    wheel_vehicle = np.array([wheel_pose[0], wheel_pose[1], wheel_pose[2], 1.0])
                    wheel_world = (T_vehicle_to_world @ wheel_vehicle)[:3]
                    wheel_points_world.append(wheel_world)

        if len(wheel_points_world) < 3:
            warnings.warn(
                "Wheel ground-contact calibration is unavailable; using a horizontal plane "
                "1.8 m below the vehicle. This is an estimate, not wheel-derived ground truth.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Default: horizontal plane at a reasonable height below vehicle
            vehicle_pose = interpolate_pose(traj, timestamp_us)
            vehicle_z = vehicle_pose[2]
            default_ground_z = vehicle_z - 1.8  # 1.8m below vehicle center
            default_point = np.array(
                [vehicle_pose[0], vehicle_pose[1], default_ground_z], dtype=np.float32
            )
            return (
                np.array([0, 0, 1], dtype=np.float32),
                default_point,
                np.array([default_point], dtype=np.float32),
            )

        wheel_points_world = np.array(wheel_points_world)

        # Fit a plane through the wheel contact points in world coordinates
        # Use SVD to find the best-fit plane
        centroid_world = np.mean(wheel_points_world, axis=0)
        points_centered = wheel_points_world - centroid_world

        # SVD to find the plane normal (smallest singular vector)
        _, _, vh = np.linalg.svd(points_centered)
        normal_world = vh[-1]  # Last row of V^T is the normal vector with smallest singular value

        # Ensure normal points "up" (positive z component in world coordinates)
        if normal_world[2] < 0:
            normal_world = -normal_world

        return (
            normal_world.astype(np.float32),
            centroid_world.astype(np.float32),
            wheel_points_world.astype(np.float32),
        )

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

    def _last_aug_flipped(self) -> bool:
        return any(getattr(t, "last_flipped", False) for t in self.aug_geo.transforms)

    def _apply_augs_and_crop(
        self,
        rgb_uint8: np.ndarray,
        depth_like: Dict[str, np.ndarray],
        K: np.ndarray,
        labels=None,
        lidar_pts_cam=None,
        rng_state=None,
    ):
        # Replaying the same RNG state gives every frame of a sample identical crop/flip/color params
        if rng_state is not None:
            self._restore_aug_rng(rng_state)

        # 1) Color aug
        # We pass K just to match the signature, though ColorAug ignores it
        out = self.aug_rgb(rgb_uint8, K)
        rgb1 = out[0]

        # 2) Geometric aug (RRC + Flip)
        # We pass EVERYTHING into the pipeline: RGB, Depths, Points, K
        # The pipeline handles detecting what is what.

        # Prepare list of inputs
        depth_keys = list(depth_like.keys())
        depth_values = [depth_like[k] for k in depth_keys]

        # Inputs: RGB, *Depths, LidarPoints, K
        # Note: We must handle None in lidar_pts_cam
        aug_inputs = [rgb1] + depth_values + [lidar_pts_cam, K]

        # Run Augmentation
        aug_outputs = self.aug_geo(*aug_inputs)

        # Unpack Outputs
        rgb2 = aug_outputs[0]
        # Middle items are depths
        dvals = aug_outputs[1 : 1 + len(depth_values)]
        # Second to last is Points
        pts_aug = aug_outputs[-2]
        # Last is K
        newK = aug_outputs[-1]

        # Reconstruct depth dictionary
        dct_c = {k: v for k, v in zip(depth_keys, dvals)}

        # 3) Labels Transformation
        # Since RRC is non-deterministic (random crop), we cannot easily transform 2D bbox labels
        # unless we pass them into the aug class.
        # For MoGe (Depth training), we usually IGNORE labels or rely on them being approximately correct if only flipping.
        # IF YOU NEED LABELS with RRC: You must update TorchRandomResizedCrop to handle bounding boxes.
        # For now, we assume labels are not critical for depth training or only H-flip is used for labels.

        final_labels = []
        if labels:
            # For inference (no augs), labels should be passed through
            # For training with augs, labels may be inaccurate after random crop
            # Check if augmentations are enabled by checking if aug_geo has transforms
            has_augs = len(self.aug_geo.transforms) > 0 or len(self.aug_rgb.transforms) > 0

            if not has_augs:
                # Inference mode - pass labels through but SCALE them to match resized image
                # Get original and target dimensions
                # rgb2 is after augmentations (or same as rgb1 if no augs)
                H_orig, W_orig = rgb2.shape[:2]
                H_target, W_target = self.H, self.W  # target size

                # Scale bboxes to match target size
                scale_y = H_target / H_orig
                scale_x = W_target / W_orig

                final_labels = []
                for label in labels:
                    if "bbox" in label:
                        x1, y1, x2, y2 = label["bbox"]
                        # Scale coordinates
                        x1_scaled = int(x1 * scale_x)
                        y1_scaled = int(y1 * scale_y)
                        x2_scaled = int(x2 * scale_x)
                        y2_scaled = int(y2 * scale_y)

                        # Clamp to image bounds
                        x1_scaled = max(0, min(x1_scaled, W_target - 1))
                        y1_scaled = max(0, min(y1_scaled, H_target - 1))
                        x2_scaled = max(0, min(x2_scaled, W_target - 1))
                        y2_scaled = max(0, min(y2_scaled, H_target - 1))

                        final_labels.append(
                            {
                                "x1": x1_scaled,
                                "y1": y1_scaled,
                                "x2": x2_scaled,
                                "y2": y2_scaled,
                                "class": label.get("class", "object"),
                                "bbox": (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
                            }
                        )
                    else:
                        # Pass through labels without bbox field
                        final_labels.append(label)
            else:
                # Training mode with augmentations - labels are not transformed
                # You would need to implement bbox transformation here if needed
                final_labels = []

        # 4) Final Resize to Target Resolution
        # Calculate scaling factors from current size to target size
        H_current, W_current = rgb2.shape[:2]
        scale_h = self.H / H_current
        scale_w = self.W / W_current

        rgb_c = cv2.resize(rgb2, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

        # Apply safety resize to depths (Nearest Neighbor)
        for k, d in dct_c.items():
            if d is not None:
                drs = cv2.resize(d, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                if drs.ndim == 2:
                    drs = drs[..., None]
                dct_c[k] = drs

        # CRITICAL: Scale intrinsics to match the resized image
        if newK is not None:
            newK = newK.copy()
            newK[0, 0] *= scale_w  # fx
            newK[0, 2] *= scale_w  # cx
            newK[1, 1] *= scale_h  # fy
            newK[1, 2] *= scale_h  # cy

        return rgb_c, dct_c, newK, final_labels, pts_aug

    def _find_neighbors(self, idx: int, total: int, offsets: Tuple[int, ...]) -> Dict[int, int]:
        out = {}
        for off in offsets:
            j = idx + off
            out[off] = j if 0 <= j < total else (idx if off == 0 else -1)
        return out

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

        # One aug draw per sample, shared by all frames and both stereo cams
        aug_rng = self._capture_aug_rng()
        # Mirror matrix for camera-frame quantities under H-flip (X-Right, Y-Down, Z-Forward)
        S_flip = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32)

        def build_cam_payload(cam_name: str, center_idx: int):
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
            # An explicitly requested GT variant is required: a sample without it would hand the
            # collate a None next to tensors, so drop it here instead.
            if agg_c is None and self.agg_depth_variant:
                return None
            lidar_d_c, lidar_p_c = self._load_single_lidar_sparse(
                seq_root, cam_name, ts_c, K0, H_raw, W_raw
            )
            labels_c = (
                self._load_labels(seq_root, cam_name, ts_c, path_c) if self.load_labels else None
            )

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
            flipped = self._last_aug_flipped()

            if (
                self.mode == "train"
                and "sparse" in dcts
                and dcts["sparse"] is not None
                and np.count_nonzero(dcts["sparse"]) < self.min_sparse
            ):
                return None

            # Ground Plane
            gp_info = None
            if self.load_ground_plane_info:
                norm, pt, wh = self._compute_road_plane_from_wheels(traj, ts_c)
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
                if flipped:
                    # Image (and depth/points) are mirrored, so the camera-frame plane must be too
                    plane_point_cam[0] *= -1.0
                    plane_normal_cam[0] *= -1.0

                gp_info = {
                    "plane_normal_world": torch.from_numpy(norm.astype(np.float32)),
                    "plane_point_world": torch.from_numpy(pt.astype(np.float32)),
                    "plane_normal_cam": torch.from_numpy(plane_normal_cam.astype(np.float32)),
                    "plane_point_cam": torch.from_numpy(plane_point_cam.astype(np.float32)),
                    "wheel_points_world": torch.from_numpy(wh.astype(np.float32)),
                    "camera_height_above_ground": h_cam,
                    "source": "wheel_contacts" if len(wh) >= 3 else "vehicle_height_estimate",
                    "timestamp_us": ts_c,
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
                    if nidx < 0:
                        # Sequence boundary: fall back to the center frame with an identity
                        # relative pose so every sample exposes the SAME neighbor offsets
                        # (keeps custom_collate_fn happy). Photometric loss on this pair is ~0.
                        frames[off] = {
                            "timestamp_us": ts_c,
                            "rgb": frames[0]["rgb"],
                            "lidar_sparse": frames[0]["lidar_sparse"],
                            "lidar_single_sparse": frames[0]["lidar_single_sparse"],
                            "K": frames[0]["K"],
                            "T_cam_world": frames[0]["T_cam_world"],
                        }
                        rels[off] = torch.eye(4, dtype=torch.float32)
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
                    if flipped:
                        # Relative motion seen in mirrored images: S @ T @ S (still a proper SE3)
                        T_rel = S_flip @ T_rel @ S_flip
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

        cam_payload = build_cam_payload(cam_center, cidx)
        if cam_payload is None:
            return None

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
            },
            cam_center: cam_payload,
        }

        # Stereo
        if self.stereo:
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
                        if self._last_aug_flipped():
                            # Both cams share the flip; mirror the extrinsics to match the images
                            S3 = S_flip[:3, :3]
                            R_lr, t_lr = (
                                (S3 @ R_lr @ S3).astype(np.float32),
                                (S3 @ t_lr).astype(np.float32),
                            )
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
