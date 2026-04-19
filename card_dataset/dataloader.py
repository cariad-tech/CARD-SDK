# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 CARIAD SE
import os, re, json, random, functools, bisect, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from . import torch_augs as augs

# Try importing laspy for LAZ support
try:
    import laspy
except ImportError:
    laspy = None

class DatasetMissingAssetError(FileNotFoundError):
    """Raised when expected dataset assets are missing."""
    pass


class DatasetCorruptionError(RuntimeError):
    """Raised when dataset files exist but appear malformed/corrupt/inconsistent."""
    pass


class DatasetConfigError(ValueError):
    """Raised when dataset loader configuration is inconsistent."""
    pass


# ---------- pure-numpy quaternion math (no scipy dependency) ----------

def _quat_xyzw_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert a unit quaternion (x,y,z,w) to a 3x3 rotation matrix."""
    n = x*x + y*y + z*z + w*w
    if n < 1e-10:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    xx, xy, xz = x*x*s, x*y*s, x*z*s
    yy, yz, zz = y*y*s, y*z*s, z*z*s
    return np.array([
        [1.0 - (yy + zz),        xy - wz,          xz + wy],
        [      xy + wz,    1.0 - (xx + zz),         yz - wx],
        [      xz - wy,          yz + wx,     1.0 - (xx + yy)],
    ], dtype=np.float32)


def _slerp_xyzw(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions (x,y,z,w)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


# ---------- utilities ----------
def _rgb(path: Path) -> np.ndarray:
    import PIL.Image as Image
    im = Image.open(str(path)).convert("RGB")
    return np.array(im)


def _normalize(rgb_uint8: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = rgb_uint8.astype(np.float32) / 255.0
    return (x - mean) / std


def _chw(x: Optional[np.ndarray]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.ndim == 2:
        x = x[..., None]
    return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32))


def _clip_min(z: Optional[np.ndarray], dmin=0.1, dmax=80.0) -> Optional[np.ndarray]:
    if z is None:
        return None
    z = np.nan_to_num(z, nan=0.0)
    if z.ndim == 2:
        z = z[..., None]
    m = z > 0
    z[m] = np.maximum(z[m], dmin)
    return np.clip(z, 0.0, dmax).astype(np.float32)


def _project_pts_to_depth(pts_cam: np.ndarray, K: np.ndarray, HW: Tuple[int, int]) -> np.ndarray:
    H, W = HW
    dep = np.zeros((H, W), np.float32)
    if pts_cam is None or len(pts_cam) == 0:
        return dep

    # Filter points behind camera
    z = pts_cam[:, 2]
    ok = z > 0
    if not np.any(ok):
        return dep

    x, y, z = pts_cam[ok, 0], pts_cam[ok, 1], pts_cam[ok, 2]

    # Project
    u = (K[0, 0] * x / z) + K[0, 2]
    v = (K[1, 1] * y / z) + K[1, 2]

    # Check bounds
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(m):
        return dep

    u, v, z = u[m], v[m], z[m]

    # Z-buffer: draw far first so near overwrites
    order = np.argsort(z)[::-1]
    u, v, z = u[order], v[order], z[order]

    dep[v.astype(int), u.astype(int)] = z
    return dep


def custom_collate_fn(batch: List[Any]) -> Any:
    """
    Publishable collate:
    - Drops None samples.
    - Leaves variable-length lists (e.g., labels) as lists-of-items.
    - For dicts: unions keys and fills missing keys with None.
    - For tensors: stacks if shapes match, else returns list of tensors.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        shapes = [tuple(x.shape) for x in batch]
        if all(s == shapes[0] for s in shapes):
            return torch.stack(batch, 0)
        return batch

    if isinstance(elem, np.ndarray):
        shapes = [tuple(x.shape) for x in batch]
        if all(s == shapes[0] for s in shapes):
            return np.stack(batch, 0)
        return batch

    if isinstance(elem, dict):
        keys = set()
        for d in batch:
            keys |= set(d.keys())
        out = {}
        for k in keys:
            out[k] = custom_collate_fn([d.get(k, None) for d in batch])
        return out

    if isinstance(elem, (list, tuple)):
        return batch

    return batch


# ---------- pose math ----------
def pose_to_matrix(x, y, z, qw, qx, qy, qz):
    R = _quat_xyzw_to_matrix(qx, qy, qz, qw)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def _slerp_scalar_first(q0, q1, t):
    # q0, q1 in scalar-first format [qw, qx, qy, qz]; return same format
    q0_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]], dtype=np.float64)
    q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]], dtype=np.float64)
    q_xyzw = _slerp_xyzw(q0_xyzw, q1_xyzw, t)
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
    CAM_LEFT = "cam_1"
    CAM_RIGHT = "cam_0"

    def __init__(
        self,
        base_root: str,
        split_file: str,
        mode="train",
        height=1000,
        width=1216,
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
        # Publishable hardening flags
        strict: bool = True,
        require_laz: bool = True,
        validate_laz_index: bool = True,
        laz_index_cache_size: int = 64,
        # ===== CHANGED: Sparse LAZ association settings =====
        laz_time_base: str = "trajectory",          # "trajectory" or "camera_raw"
        sparse_window_mode: str = "midpoint",       # "midpoint" | "prev_to_curr" | "fixed"
        sparse_fixed_window_us: int = 100_000,      # used if mode=="fixed" or if midpoint degenerates
        sparse_use_all_lasers: bool = True,         # if False, you may pass sparse_laser_names
        sparse_laser_names: Optional[List[str]] = None,
        max_sparse_points: Optional[int] = None,    # optional cap (random subsample) after concatenation
        **kwargs,
    ):
        super().__init__()
        self.base_root = Path(base_root)
        self.mode = mode
        self.H, self.W = height, width
        self.context = tuple(context_offsets)
        self.stereo = stereo
        self.stereo_tol = stereo_time_tolerance_us
        self.min_sparse = min_sparse_points_train

        self.load_labels = load_labels
        self.load_neighbors = load_neighbors
        self.load_ground_plane_info = load_ground_plane_info
        self.agg_depth_variant = agg_depth_variant

        self.strict = bool(strict)
        self.require_laz = bool(require_laz)
        self.validate_laz_index = bool(validate_laz_index)
        self.laz_index_cache_size = int(laz_index_cache_size)

        self.laz_time_base = str(laz_time_base)
        self.sparse_window_mode = str(sparse_window_mode)
        self.sparse_fixed_window_us = int(sparse_fixed_window_us)
        self.sparse_use_all_lasers = bool(sparse_use_all_lasers)
        self.sparse_laser_names = sparse_laser_names[:] if sparse_laser_names is not None else None
        self.max_sparse_points = int(max_sparse_points) if max_sparse_points is not None else None

        if self.laz_time_base not in ("trajectory", "camera_raw"):
            raise DatasetConfigError(f"Invalid laz_time_base='{self.laz_time_base}'. Use 'trajectory' or 'camera_raw'.")
        if self.sparse_window_mode not in ("midpoint", "prev_to_curr", "fixed"):
            raise DatasetConfigError(
                f"Invalid sparse_window_mode='{self.sparse_window_mode}'. Use 'midpoint', 'prev_to_curr', or 'fixed'."
            )
        if (not self.sparse_use_all_lasers) and (not self.sparse_laser_names):
            raise DatasetConfigError("If sparse_use_all_lasers=False, you must provide sparse_laser_names=[...].")

        random.seed(seed)

        self.rgb_mean = np.array(mean, dtype=np.float32)[None, None, :]
        self.rgb_std = np.array(std, dtype=np.float32)[None, None, :]

        self.aug_rgb = (
            augs.Compose([augs.TorchColorAug()])
            if (mode == "train" and use_color_aug)
            else augs.Compose([])
        )
        self.aug_geo = (
            augs.Compose([augs.TorchGeoAug(hflip_p=0.5)])
            if (mode == "train" and use_geom_aug)
            else augs.Compose([])
        )

        self.bumper_mask = None
        if bumper_mask_path and Path(bumper_mask_path).is_file():
            import PIL.Image as Image
            self.bumper_mask = np.array(Image.open(bumper_mask_path).convert("L"), np.float32) / 255.0

        split_path = self.base_root / split_file
        if not split_path.is_file():
            raise DatasetMissingAssetError(f"Split file not found: {split_path}")

        with open(split_path, "r") as f:
            self.centers = [line.strip() for line in f if line.strip()]

        if len(self.centers) == 0:
            raise DatasetCorruptionError(f"Split file is empty: {split_path}")

        # Caches
        self._camdir_cache: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
        self._laz_index_cache: Dict[str, Dict] = {}
        self._warned_once: set = set()

        if self.require_laz and laspy is None:
            raise DatasetMissingAssetError(
                "laspy is not installed, but require_laz=True. Install: pip install 'laspy[lazrs]'"
            )

    def _warn_once(self, key: str, msg: str):
        if key in self._warned_once:
            return
        warnings.warn(msg)
        self._warned_once.add(key)

    def _fail(self, msg: str, exc: Optional[Exception] = None):
        if self.strict:
            if exc is None:
                raise DatasetCorruptionError(msg)
            raise DatasetCorruptionError(f"{msg}\nCaused by: {type(exc).__name__}: {exc}") from exc
        return None

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
            return self.base_root
        return (self.base_root / parts[0])

    # ----- RAM SAFE LOADING -----
    @functools.lru_cache(maxsize=8)
    def _get_traj_cached(self, seq_root_str: str) -> Dict:
        traj_path = Path(seq_root_str) / "export" / "output.laz.trajectory.json"
        with open(traj_path, "r") as f:
            return json.load(f)

    def _get_traj(self, seq_root: Path) -> Dict:
        traj_path = seq_root / "export" / "output.laz.trajectory.json"
        if not traj_path.is_file():
            if self.strict:
                raise DatasetMissingAssetError(f"Trajectory JSON missing: {traj_path}")
            return None
        try:
            return self._get_traj_cached(str(seq_root))
        except Exception as e:
            return self._fail(f"Failed to read trajectory JSON: {traj_path}", e)

    # ----- LAZ INDEX LOADING -----
    def _get_laz_index(self, seq_root: Path) -> Optional[Dict]:
        key = str(seq_root)
        if key in self._laz_index_cache:
            return self._laz_index_cache[key]

        laz_path = seq_root / "export" / "output.laz"
        index_path = seq_root / "export" / "output.laz.index.json"

        if not laz_path.is_file() or not index_path.is_file():
            if self.require_laz:
                raise DatasetMissingAssetError(
                    f"Missing LAZ assets for sequence:\n"
                    f"  seq_root={seq_root}\n"
                    f"  laz_path={laz_path} exists={laz_path.is_file()}\n"
                    f"  index_path={index_path} exists={index_path.is_file()}"
                )
            return None

        try:
            with open(index_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise DatasetCorruptionError(f"Failed to read LAZ index JSON: {index_path}") from e

        entries = data.get("entries", None)
        if not isinstance(entries, list) or len(entries) == 0:
            raise DatasetCorruptionError(f"LAZ index has no valid 'entries' list: {index_path}")

        required_keys = {"end_ts", "point_offset_in_file", "num_pts"}
        for k, e in enumerate(entries[:20]):
            if not isinstance(e, dict) or not required_keys.issubset(e.keys()):
                raise DatasetCorruptionError(
                    f"LAZ index entry {k} missing keys {required_keys} in {index_path}. "
                    f"Got keys={list(e.keys()) if isinstance(e, dict) else type(e)}"
                )

        # Reject multipart unless you implement merging
        if any(int(e.get("part_index", 0)) != 0 for e in entries):
            raise DatasetCorruptionError(
                f"Multipart LAZ not supported (found part_index!=0) in {index_path}."
            )

        try:
            entries.sort(key=lambda x: int(x["end_ts"]))
            end_times = [int(e["end_ts"]) for e in entries]
        except Exception as e:
            raise DatasetCorruptionError(f"Failed to sort/parse LAZ index entries in {index_path}") from e

        point_count = None
        if self.validate_laz_index:
            if laspy is None:
                if self.require_laz:
                    raise DatasetMissingAssetError(
                        "laspy not installed; cannot validate/read output.laz. Install: pip install 'laspy[lazrs]'"
                    )
            else:
                try:
                    with laspy.open(str(laz_path)) as las:
                        hdr = las.header
                        point_count = getattr(hdr, "point_count", None)
                        if point_count is None:
                            point_count = getattr(hdr, "point_records_count", None)
                except Exception as e:
                    raise DatasetCorruptionError(f"Failed to open LAZ file for validation: {laz_path}") from e

                if point_count is None:
                    raise DatasetCorruptionError(f"Could not determine point_count from LAZ header: {laz_path}")

                for k, e in enumerate(entries):
                    off = int(e["point_offset_in_file"])
                    n = int(e["num_pts"])
                    if off < 0 or n < 0:
                        raise DatasetCorruptionError(
                            f"Invalid LAZ index entry {k}: offset={off}, num_pts={n} in {index_path}"
                        )
                    if off >= point_count:
                        raise DatasetCorruptionError(
                            f"LAZ index entry {k} offset out of range: offset={off} >= point_count={point_count}\n"
                            f"index={index_path}"
                        )
                    if off + n > point_count:
                        raise DatasetCorruptionError(
                            f"LAZ index entry {k} range out of bounds: offset+num_pts={off + n} > point_count={point_count}\n"
                            f"index={index_path}"
                        )

        cache_item = {
            "laz_path": str(laz_path),
            "index_path": str(index_path),
            "entries": entries,
            "end_times": end_times,
            "point_count": point_count,
        }

        if len(self._laz_index_cache) >= max(1, self.laz_index_cache_size):
            oldest_key = next(iter(self._laz_index_cache.keys()))
            self._laz_index_cache.pop(oldest_key, None)

        self._laz_index_cache[key] = cache_item
        return cache_item

    # ===== CHANGED: window computation & entry selection =====
    def _cam_ts_to_laz_time(self, traj: Dict, cam: str, ts_cam_raw_us: int) -> int:
        """
        Convert a camera filename timestamp into the time base used by LAZ index.
        - laz_time_base="trajectory": add camera_infos[cam]["timestamp_offset"]
        - laz_time_base="camera_raw": keep as-is
        """
        if self.laz_time_base == "camera_raw":
            return int(ts_cam_raw_us)
        off = int(traj["camera_infos"][cam].get("timestamp_offset", 0))
        return int(ts_cam_raw_us + off)

    def _compute_sparse_window(
        self,
        traj: Dict,
        cam: str,
        cam_ts_prev_raw: int,
        cam_ts_center_raw: int,
        cam_ts_next_raw: int,
    ) -> Tuple[int, int, int]:
        """
        Returns (t0_laz, t1_laz, t_center_laz) in the LAZ time base.
        """
        # raw window
        if self.sparse_window_mode == "midpoint":
            t0_raw = int((cam_ts_prev_raw + cam_ts_center_raw) // 2)
            t1_raw = int((cam_ts_center_raw + cam_ts_next_raw) // 2)
        elif self.sparse_window_mode == "prev_to_curr":
            t0_raw = int(cam_ts_prev_raw)
            t1_raw = int(cam_ts_center_raw)
        else:  # fixed
            half = max(1, int(self.sparse_fixed_window_us // 2))
            t0_raw = int(cam_ts_center_raw - half)
            t1_raw = int(cam_ts_center_raw + half)

        # convert to LAZ base
        t0 = self._cam_ts_to_laz_time(traj, cam, t0_raw)
        t1 = self._cam_ts_to_laz_time(traj, cam, t1_raw)
        tc = self._cam_ts_to_laz_time(traj, cam, int(cam_ts_center_raw))

        # ensure non-degenerate window
        if t1 <= t0:
            half = max(1, int(self.sparse_fixed_window_us // 2))
            t0 = tc - half
            t1 = tc + half

        return int(t0), int(t1), int(tc)

    def _select_laz_entries_in_window(self, idx_data: Dict, t0: int, t1: int) -> List[Dict]:
        """
        Select all entries whose end_ts is in (t0, t1].
        """
        end_times = idx_data["end_times"]
        entries = idx_data["entries"]
        lo = bisect.bisect_right(end_times, int(t0))
        hi = bisect.bisect_right(end_times, int(t1))
        return entries[lo:hi]

    def _read_laz_points_for_entries(self, laz_path: str, entries: List[Dict]) -> Optional[np.ndarray]:
        if len(entries) == 0:
            return None
        if laspy is None:
            if self.require_laz:
                raise DatasetMissingAssetError(
                    "laspy is not installed, cannot read output.laz. Install: pip install 'laspy[lazrs]'"
                )
            return None

        pts_list = []
        try:
            with laspy.open(laz_path) as las:
                for e in entries:
                    if (not self.sparse_use_all_lasers) and (self.sparse_laser_names is not None):
                        ln = e.get("laser_name", None)
                        if ln is not None and ln not in self.sparse_laser_names:
                            continue

                    off = int(e["point_offset_in_file"])
                    n = int(e["num_pts"])
                    las.seek(off)
                    p = las.read_points(n)
                    coords = np.vstack((p.x, p.y, p.z)).T.astype(np.float32)
                    if coords.size > 0:
                        pts_list.append(coords)
        except Exception as ex:
            return self._fail(f"Failed reading LAZ entries from {laz_path}", ex)

        if len(pts_list) == 0:
            return None

        pts = np.concatenate(pts_list, axis=0)

        # optional subsample cap
        if self.max_sparse_points is not None and pts.shape[0] > self.max_sparse_points:
            idx = np.random.choice(pts.shape[0], self.max_sparse_points, replace=False)
            pts = pts[idx]

        return pts

    # ----- Geometry Helpers -----
    @staticmethod
    def _K_dist_from_traj(traj: Dict, cam: str):
        ci = traj["camera_infos"][cam]
        K = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1]], np.float32)
        dist = np.array(ci.get("dist_coeffs", [0, 0, 0, 0, 0]), np.float32)
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
    def _load_sparse_windowed(
        self,
        seq_root: Path,
        cam: str,
        ts_center_raw: int,
        ts_prev_raw: int,
        ts_next_raw: int,
        traj: Dict,
        K: np.ndarray,
        H: int,
        W: int,
    ):
        """
        ===== CHANGED: interval-based LAZ loading =====
        Reads ALL LAZ entries with end_ts in (t0, t1] (in the configured LAZ time base),
        where [t0,t1] is derived from adjacent camera timestamps.

        Returns projected sparse depth (H,W,1) or None.
        """
        idx_data = self._get_laz_index(seq_root)
        if idx_data is None:
            return None

        # Compute window in LAZ time base
        t0, t1, tc = self._compute_sparse_window(traj, cam, ts_prev_raw, ts_center_raw, ts_next_raw)

        # Select entries in (t0, t1]
        entries = self._select_laz_entries_in_window(idx_data, t0, t1)

        # Fallback: if none selected, pick nearest chunk at/after tc
        if len(entries) == 0:
            end_times = idx_data["end_times"]
            all_entries = idx_data["entries"]
            k = bisect.bisect_left(end_times, int(tc))
            if k >= len(all_entries):
                k = len(all_entries) - 1
            if k >= 0:
                entries = [all_entries[k]]

        pts_world = self._read_laz_points_for_entries(idx_data["laz_path"], entries)
        if pts_world is None or pts_world.shape[0] == 0:
            return None

        # World -> Camera (camera pose computed at CENTER frame time)
        try:
            T_cw = self._T_cam_world(traj, cam, ts_center_raw)
            T_wc = inv_SE3(T_cw)
            pts_cam = (T_wc[:3, :3] @ pts_world.T).T + T_wc[:3, 3]
        except Exception as e:
            return self._fail(f"Failed World->Cam transform for sparse LAZ: cam={cam} ts={ts_center_raw}", e)

        dep = _project_pts_to_depth(pts_cam, K, (H, W))

        if self.bumper_mask is not None:
            try:
                mask = cv2.resize(self.bumper_mask, (dep.shape[1], dep.shape[0]), interpolation=cv2.INTER_NEAREST)
                dep = dep * mask
            except Exception as e:
                return self._fail(f"Failed applying bumper mask for sparse LAZ: cam={cam}", e)

        return _clip_min(dep)

    def _load_single(self, seq_root: Path, cam: str, ts: int, K: np.ndarray, H: int, W: int):
        p = seq_root / "single_depth" / cam / f"{cam}_{ts}.npz"
        if not p.is_file():
            return None
        try:
            z = np.load(str(p), allow_pickle=True)
            if "depth" in z:
                dep = z["depth"].astype(np.float32)
            elif "pts_cam" in z:
                dep = _project_pts_to_depth(z["pts_cam"].astype(np.float32), K, (H, W))
            else:
                return None
        except Exception as e:
            return self._fail(f"Failed reading single_depth npz: {p}", e)

        if self.bumper_mask is not None:
            try:
                m = cv2.resize(self.bumper_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                dep = dep * m
            except Exception as e:
                return self._fail(f"Failed applying bumper mask to single_depth: {p}", e)

        return _clip_min(dep)

    def _load_agg(self, seq_root: Path, cam: str, ts: int, K: np.ndarray, dist: np.ndarray, H: int, W: int):
        agg_dir = self.agg_depth_variant if self.agg_depth_variant else "agg_depth"
        p = seq_root / agg_dir / cam / f"{cam}_{ts}.npz"
        if not p.is_file():
            return None
        try:
            z = np.load(str(p), allow_pickle=True)
            if "pts_cam" not in z:
                return None
            dep = _project_pts_to_depth(z["pts_cam"].astype(np.float32), K, (H, W))
        except Exception as e:
            return self._fail(f"Failed reading agg_depth npz: {p}", e)

        if self.bumper_mask is not None:
            try:
                m = cv2.resize(self.bumper_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                dep = dep * m
            except Exception as e:
                return self._fail(f"Failed applying bumper mask to agg_depth: {p}", e)

        return _clip_min(dep)

    def _load_single_lidar_sparse(self, seq_root: Path, cam: str, ts: int, K: np.ndarray, H: int, W: int):
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

        try:
            z = np.load(str(npz), allow_pickle=True)
            pts_cam = z.get("pts_cam", None)
            if pts_cam is not None:
                pts_cam = pts_cam.astype(np.float32)

            dep = z.get("depth", None)
            if dep is None and pts_cam is not None:
                dep = _project_pts_to_depth(pts_cam, K, (H, W))
            elif dep is not None:
                dep = dep.astype(np.float32)
            else:
                return None, pts_cam
        except Exception as e:
            return self._fail(f"Failed reading lidar sparse npz: {npz}", e), None

        if self.bumper_mask is not None:
            try:
                m = cv2.resize(self.bumper_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                dep = dep * m
            except Exception as e:
                return self._fail(f"Failed applying bumper mask to lidar sparse: {npz}", e), pts_cam

        return _clip_min(dep), pts_cam

    def _load_labels(self, seq_root: Path, cam: str, ts: int, img_path: str):
        p = seq_root / "labels" / f"{cam}_{ts}.txt"
        if not p.is_file():
            return None
        img = cv2.imread(img_path)
        if img is None:
            return self._fail(f"Failed to read image for label sizing: {img_path}")
        h, w = img.shape[:2]
        out = []
        try:
            with open(p, "r") as f:
                for line in f:
                    a = line.strip().split()
                    if len(a) == 5:
                        cls, x_c, y_c, bw, bh = map(float, a)
                        x1, y1 = int((x_c * w) - (bw * w) / 2), int((y_c * h) - (bh * h) / 2)
                        x2, y2 = int((x_c * w) + (bw * w) / 2), int((y_c * h) + (bh * h) / 2)
                        out.append({"class": int(cls), "bbox": (x1, y1, x2, y2)})
        except Exception as e:
            return self._fail(f"Failed reading labels: {p}", e)
        return out

    def _compute_road_plane_from_wheels(self, traj: Dict, timestamp_us: int):
        wheel_keys = [
            "cariad_wheel_FL_ground",
            "cariad_wheel_FR_ground",
            "cariad_wheel_RL_ground",
            "cariad_wheel_RR_ground",
        ]
        points = []
        for wk in wheel_keys:
            if wk in traj.get("sensor_to_trajectory_poses", {}):
                w_pose = traj["sensor_to_trajectory_poses"][wk]
                v_pose = interpolate_pose(traj, timestamp_us)
                T_veh_world = pose_to_matrix(*v_pose)
                w_veh = np.array([w_pose[0], w_pose[1], w_pose[2], 1.0])
                points.append((T_veh_world @ w_veh)[:3])

        if len(points) < 3:
            vp = interpolate_pose(traj, timestamp_us)
            p = np.array([vp[0], vp[1], vp[2] - 1.8], dtype=np.float32)
            return (np.array([0, 0, 1], dtype=np.float32), p, np.array([p]))

        pts = np.array(points, dtype=np.float32)
        ctr = np.mean(pts, axis=0)
        _, _, vh = np.linalg.svd(pts - ctr)
        norm = vh[-1]
        if norm[2] < 0:
            norm = -norm
        return (norm, ctr, pts)

    def _compute_camera_height_from_ground_plane(self, traj, cam, ts, norm, pt):
        Tcw = self._T_cam_world(traj, cam, ts)
        cam_pt = Tcw[:3, 3]
        return float(np.dot(cam_pt - pt, norm / np.linalg.norm(norm)))

    # ----- augs & crop -----
    def _apply_augs_and_crop(
        self,
        rgb_uint8: np.ndarray,
        depth_like: Dict[str, np.ndarray],
        K: np.ndarray,
        labels=None,
        lidar_pts_cam=None,
    ):
        out = self.aug_rgb(rgb_uint8, *([v for v in depth_like.values()] or [rgb_uint8]), K)
        rgb1 = out[0]

        out = self.aug_geo(rgb1, *([v for v in depth_like.values()] or []), K)
        rgb2 = out[0]
        newK = out[-1].copy() if out[-1] is not None else (K.copy() if K is not None else None)
        dvals = out[1:-1]

        orig_w = rgb_uint8.shape[1]
        is_flipped = False
        if newK is not None and K is not None:
            if abs(newK[0, 2] - (orig_w - K[0, 2])) < 1.0 and abs(newK[0, 2] - K[0, 2]) > 10.0:
                is_flipped = True

        pts_aug = None
        if lidar_pts_cam is not None:
            pts_aug = lidar_pts_cam.copy()
            if is_flipped:
                pts_aug[:, 0] *= -1.0

        final_labels = []
        if labels:
            for lbl in labels:
                cls, (x1, y1, x2, y2) = lbl["class"], lbl["bbox"]
                if is_flipped:
                    x1, x2 = orig_w - 1 - x2, orig_w - 1 - x1
                final_labels.append({"class": cls, "bbox": (x1, y1, x2, y2)})

        H, W = rgb2.shape[:2]
        scale_h, scale_w = self.H / H, self.W / W

        rgb_c = cv2.resize(rgb2, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

        dct_c = {}
        for (k, _), d in zip(depth_like.items(), dvals):
            if d is None:
                dct_c[k] = None
            else:
                drs = cv2.resize(d, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                if drs.ndim == 2:
                    drs = drs[..., None]
                dct_c[k] = drs

        if newK is not None:
            newK[0, 0] *= scale_w
            newK[0, 2] *= scale_w
            newK[1, 1] *= scale_h
            newK[1, 2] *= scale_h

        out_labels = []
        for lbl in final_labels:
            x1, y1, x2, y2 = lbl["bbox"]
            x1, x2 = x1 * scale_w, x2 * scale_w
            y1, y2 = y1 * scale_h, y2 * scale_h
            if x2 > x1 and y2 > y1:
                out_labels.append({"class": lbl["class"], "bbox": (int(x1), int(y1), int(x2), int(y2))})

        return rgb_c, dct_c, newK, out_labels, pts_aug

    def _find_neighbors(self, idx: int, total: int, offsets: Tuple[int, ...]) -> Dict[int, int]:
        out = {}
        for off in offsets:
            j = idx + off
            out[off] = (j if 0 <= j < total else (idx if off == 0 else -1))
        return out

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, i: int):
        rel = self.centers[i]
        cam_center, ts_center = self._parse_cam_ts(rel)
        if cam_center is None:
            return self._fail(f"Invalid center path format in split file: '{rel}'")

        seq_root = self._seq_root_from_rel(rel)
        traj = self._get_traj(seq_root)
        if traj is None:
            return None

        cam_list = self._cam_dir_list(seq_root, cam_center)
        if len(cam_list) == 0:
            return self._fail(f"No images found for cam={cam_center} in seq_root={seq_root}")

        ts_list = [t for t, _ in cam_list]
        try:
            cidx = ts_list.index(ts_center)
        except Exception as e:
            return self._fail(
                f"Center timestamp not found in cam directory listing: cam={cam_center} ts={ts_center} seq_root={seq_root}",
                e,
            )

        def build_cam_payload(cam_name: str, center_idx: int):
            try:
                K_raw, dist_raw = self._K_dist_from_traj(traj, cam_name)
            except Exception as e:
                return self._fail(f"Missing/inconsistent camera intrinsics for cam={cam_name} seq_root={seq_root}", e)

            clist = self._cam_dir_list(seq_root, cam_name)
            if len(clist) == 0:
                return self._fail(f"No images for cam={cam_name} in seq_root={seq_root}")

            neigh = self._find_neighbors(center_idx, len(clist), self.context)

            ts_c, path_c = clist[neigh[0]]

            # prev/next timestamps for sparse window
            ts_prev = clist[center_idx - 1][0] if center_idx > 0 else ts_c
            ts_next = clist[center_idx + 1][0] if (center_idx + 1) < len(clist) else ts_c

            try:
                _tmp = _rgb(Path(path_c))
            except Exception as e:
                return self._fail(f"Failed reading center RGB: {path_c}", e)

            H_raw, W_raw = _tmp.shape[:2]

            try:
                newK, _ = cv2.getOptimalNewCameraMatrix(K_raw, dist_raw, (W_raw, H_raw), 0, (W_raw, H_raw))
                mapx, mapy = cv2.initUndistortRectifyMap(K_raw, dist_raw, None, newK, (W_raw, H_raw), cv2.CV_32FC1)
            except Exception as e:
                return self._fail(f"Failed computing undistort maps for cam={cam_name} seq_root={seq_root}", e)

            K0 = newK.astype(np.float32)
            dist0 = np.zeros(5, np.float32)

            frames, rels = {}, {}

            # --- CENTER ---
            try:
                rgb_c = _rgb(Path(path_c))
                rgb_c_ud = cv2.remap(rgb_c, mapx, mapy, interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                return self._fail(f"Failed undistorting RGB: {path_c}", e)

            try:
                Tcw_c = self._T_cam_world(traj, cam_name, ts_c)
            except Exception as e:
                return self._fail(f"Failed computing T_cam_world for cam={cam_name} ts={ts_c}", e)

            # ===== CHANGED: interval-based sparse LAZ loading =====
            sparse_c = self._load_sparse_windowed(seq_root, cam_name, ts_c, ts_prev, ts_next, traj, K0, H_raw, W_raw)

            single_c = self._load_single(seq_root, cam_name, ts_c, K0, H_raw, W_raw)
            agg_c = self._load_agg(seq_root, cam_name, ts_c, K0, dist0, H_raw, W_raw)
            lidar_d_c, lidar_p_c = self._load_single_lidar_sparse(seq_root, cam_name, ts_c, K0, H_raw, W_raw)
            labels_c = self._load_labels(seq_root, cam_name, ts_c, path_c) if self.load_labels else None

            depth_like = {}
            if sparse_c is not None:
                depth_like["sparse"] = sparse_c
            if single_c is not None:
                depth_like["depth_single"] = single_c
            if agg_c is not None:
                depth_like["depth_agg"] = agg_c
            if lidar_d_c is not None:
                depth_like["lidar_single_sparse"] = lidar_d_c

            rgb_cc, dcts, Kc, labels_transformed, pts_c_aug = self._apply_augs_and_crop(
                rgb_c_ud, depth_like, K0, labels_c, lidar_pts_cam=lidar_p_c
            )

            if (
                self.mode == "train"
                and "sparse" in dcts
                and dcts["sparse"] is not None
                and np.count_nonzero(dcts["sparse"]) < self.min_sparse
            ):
                return None

            gp_info = None
            if self.load_ground_plane_info:
                try:
                    norm, pt, wh = self._compute_road_plane_from_wheels(traj, ts_c)
                    h_cam = self._compute_camera_height_from_ground_plane(traj, cam_name, ts_c, norm, pt)
                    gp_info = {
                        "plane_normal_world": torch.from_numpy(norm),
                        "plane_point_world": torch.from_numpy(pt),
                        "wheel_points_world": torch.from_numpy(wh),
                        "camera_height_above_ground": h_cam,
                        "timestamp_us": ts_c,
                    }
                except Exception as e:
                    return self._fail(f"Failed computing ground plane info: cam={cam_name} ts={ts_c}", e)

            frames[0] = {
                "timestamp_us": ts_c,
                "rgb": _chw(_normalize(rgb_cc, self.rgb_mean, self.rgb_std)),
                "lidar_sparse": _chw(dcts.get("sparse", None)),
                "depth_single": _chw(dcts.get("depth_single", None)),
                "depth_agg": _chw(dcts.get("depth_agg", None)),
                "lidar_single_sparse": _chw(dcts.get("lidar_single_sparse", None)),
                "lidar_single_pts_cam": torch.from_numpy(pts_c_aug.astype(np.float32)) if pts_c_aug is not None else None,
                "K": torch.from_numpy(Kc.astype(np.float32)) if Kc is not None else None,
                "dist": torch.from_numpy(dist0),
                "T_cam_world": torch.from_numpy(Tcw_c.astype(np.float32)),
                "labels": labels_transformed,
                "ground_plane_info": gp_info,
            }

            # --- NEIGHBORS ---
            if self.load_neighbors:
                clist_full = self._cam_dir_list(seq_root, cam_name)
                for off, nidx in neigh.items():
                    if off == 0 or nidx < 0:
                        continue
                    ts_n, path_n = clist_full[nidx]

                    # window for neighbor
                    ts_prev_n = clist_full[nidx - 1][0] if nidx > 0 else ts_n
                    ts_next_n = clist_full[nidx + 1][0] if (nidx + 1) < len(clist_full) else ts_n

                    try:
                        rgb_n = _rgb(Path(path_n))
                        rgb_n_ud = cv2.remap(rgb_n, mapx, mapy, interpolation=cv2.INTER_LINEAR)
                    except Exception as e:
                        return self._fail(f"Failed undistorting neighbor RGB: {path_n}", e)

                    try:
                        Tcw_n = self._T_cam_world(traj, cam_name, ts_n)
                    except Exception as e:
                        return self._fail(f"Failed computing neighbor pose: cam={cam_name} ts={ts_n}", e)

                    sparse_n = self._load_sparse_windowed(
                        seq_root, cam_name, ts_n, ts_prev_n, ts_next_n, traj, K0, H_raw, W_raw
                    )
                    lidar_d_n, _ = self._load_single_lidar_sparse(seq_root, cam_name, ts_n, K0, H_raw, W_raw)

                    dl_n = {}
                    if sparse_n is not None:
                        dl_n["sparse"] = sparse_n
                    if lidar_d_n is not None:
                        dl_n["lidar_single_sparse"] = lidar_d_n

                    rgb_nc, dcts_n, Kn, _, _ = self._apply_augs_and_crop(rgb_n_ud, dl_n, K0, None, None)

                    frames[off] = {
                        "timestamp_us": ts_n,
                        "rgb": _chw(_normalize(rgb_nc, self.rgb_mean, self.rgb_std)),
                        "lidar_sparse": _chw(dcts_n.get("sparse", None)),
                        "lidar_single_sparse": _chw(dcts_n.get("lidar_single_sparse", None)),
                        "K": torch.from_numpy(Kn.astype(np.float32)) if Kn is not None else None,
                        "T_cam_world": torch.from_numpy(Tcw_n.astype(np.float32)),
                    }
                    rels[off] = torch.from_numpy(rel_pose(Tcw_n, Tcw_c).astype(np.float32))

            return {
                "frames": frames,
                "rel_poses_to_center": rels,
                "K_ref": torch.from_numpy(K0),
                "dist_ref": torch.from_numpy(dist0),
                "T_cam_to_rig": torch.from_numpy(self._T_cam_to_rig_from_traj(traj, cam_name).astype(np.float32)),
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
                "laz_time_base": self.laz_time_base,
                "sparse_window_mode": self.sparse_window_mode,
                "sparse_fixed_window_us": self.sparse_fixed_window_us,
            },
            cam_center: cam_payload,
        }

        # Stereo
        if self.stereo:
            other = self.CAM_RIGHT if cam_center == self.CAM_LEFT else self.CAM_LEFT
            other_l = self._cam_dir_list(seq_root, other)
            if len(other_l) == 0:
                if self.strict:
                    raise DatasetCorruptionError(
                        f"Stereo enabled but other camera has no frames: other={other} seq_root={seq_root}"
                    )
            else:
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
                        l, r = self.CAM_LEFT, self.CAM_RIGHT
                        if l in out and r in out:
                            si = self._stereo_rectify_params(traj, l, r)
                            out["stereo"] = {
                                "left_cam": l,
                                "right_cam": r,
                                "baseline_m": float(np.linalg.norm(si["t_lr"])),
                                "R_right_from_left": torch.from_numpy(si["R_lr"]),
                                "t_right_from_left": torch.from_numpy(si["t_lr"]),
                                "rectification": {
                                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                                    for k, v in si.items()
                                },
                            }

        return out

    @classmethod
    def for_training(cls, base_root, split_file, **kwargs):
        cfg = {
            "mode": "train",
            "load_neighbors": True,
            "load_labels": True,
            "use_color_aug": True,
            "use_geom_aug": True,
            # training typically skips rare bad samples
            "strict": kwargs.pop("strict", False),
            "require_laz": kwargs.pop("require_laz", True),
            "validate_laz_index": kwargs.pop("validate_laz_index", True),
        }
        cfg.update(kwargs)
        return cls(base_root, split_file, **cfg)

    @classmethod
    def for_inference(cls, base_root, split_file, **kwargs):
        cfg = {
            "mode": "val",
            "load_neighbors": False,
            "load_labels": False,
            "use_color_aug": False,
            "use_geom_aug": False,
            "context_offsets": (0,),
            "strict": kwargs.pop("strict", True),
            "require_laz": kwargs.pop("require_laz", True),
            "validate_laz_index": kwargs.pop("validate_laz_index", True),
        }
        cfg.update(kwargs)
        return cls(base_root, split_file, **cfg)

    @classmethod
    def for_visualization(cls, base_root, split_file, **kwargs):
        cfg = {
            "mode": "val",
            "load_neighbors": True,
            "load_labels": True,
            "use_color_aug": False,
            "use_geom_aug": False,
            "strict": kwargs.pop("strict", True),
            "require_laz": kwargs.pop("require_laz", True),
            "validate_laz_index": kwargs.pop("validate_laz_index", True),
        }
        cfg.update(kwargs)
        return cls(base_root, split_file, **cfg)
