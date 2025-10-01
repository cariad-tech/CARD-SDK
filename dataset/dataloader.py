# unified_dataset.py
import os, re, json, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation, Slerp

import torch_augs as augs  # <— use the pytorch-only augs


# ---------- pose utilities ----------
def pose_to_matrix(x, y, z, qw, qx, qy, qz):
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4, dtype=np.float32); T[:3,:3] = R; T[:3,3] = [x,y,z]
    return T

def _slerp_scalar_first(q0, q1, t):
    q0_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]])
    q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]])
    Rends = Rotation.from_quat(np.stack([q0_xyzw, q1_xyzw], 0))
    s = Slerp([0.0, 1.0], Rends)
    q_xyzw = s([t]).as_quat()[0]
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]

def interpolate_pose(traj: Dict, ts_us: int):
    poses = traj["trajectory_poses"]  # [[ts, [x,y,z,qw,qx,qy,qz]], ...]
    i = 0
    while i < len(poses) and poses[i][0] <= ts_us:
        i += 1
    if i == 0: return poses[0][1]
    if i >= len(poses): return poses[-1][1]
    ts0, p0 = poses[i-1]; ts1, p1 = poses[i]
    t = float(np.clip((ts_us - ts0) / max(1, (ts1 - ts0)), 0.0, 1.0))
    trans = (1.0 - t)*np.array(p0[:3]) + t*np.array(p1[:3])
    quat = _slerp_scalar_first(p0[3:], p1[3:], t)
    return [*trans.astype(float).tolist(), *quat]

def inv_SE3(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
    return Ti

def rel_pose(Tn_world, Tc_world):
    return inv_SE3(Tc_world) @ Tn_world


# ---------- depth & image helpers ----------
def _rgb(path: Path) -> np.ndarray:
    import PIL.Image as Image
    im = Image.open(str(path)).convert("RGB")
    return np.array(im)

def _normalize(rgb_uint8: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = rgb_uint8.astype(np.float32) / 255.0
    return (x - mean) / std

def _chw(x: Optional[np.ndarray]) -> Optional[torch.Tensor]:
    if x is None: return None
    return torch.from_numpy(x.transpose(2,0,1).astype(np.float32))

def _clip_min(z: Optional[np.ndarray], dmin=0.1, dmax=80.0) -> Optional[np.ndarray]:
    if z is None: return None
    z = np.nan_to_num(z, nan=0.0)
    if z.ndim == 2: z = z[..., None]
    m = z > 0; z[m] = np.maximum(z[m], dmin)
    return np.clip(z, 0.0, dmax).astype(np.float32)

def _project_pts_to_depth(pts_cam: np.ndarray, K: np.ndarray, HW: Tuple[int,int]) -> np.ndarray:
    H,W = HW; dep = np.zeros((H,W), np.float32)
    if pts_cam is None or len(pts_cam)==0: return dep
    z = pts_cam[:,2]; ok = z > 0
    if not np.any(ok): return dep
    x,y,z = pts_cam[ok,0], pts_cam[ok,1], pts_cam[ok,2]
    u = (K[0,0]*x/z) + K[0,2]; v = (K[1,1]*y/z) + K[1,2]
    m = (u>=0)&(u<W)&(v>=0)&(v<H)
    if not np.any(m): return dep
    u,v,z = u[m], v[m], z[m]
    order = np.argsort(z)[::-1]
    u,v,z = u[order], v[order], z[order]
    dep[v.astype(int), u.astype(int)] = z
    return dep

def _undistort(rgb_bgr: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    try:
        return cv2.undistort(rgb_bgr, K, dist)
    except Exception:
        return rgb_bgr


# ---------- dataset ----------
class UnifiedCardDataset(Dataset):
    CAM_LEFT = "cam_1"
    CAM_RIGHT = "cam_0"

    def __init__(
        self,
        base_root: str,
        split_file: str,              # path relative to base_root
        mode="train",
        height=352, width=1216,
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225),
        context_offsets=(-1,0,+1),
        stereo=False,
        stereo_time_tolerance_us=2000,
        min_sparse_points_train=200,
        use_color_aug=True,
        use_geom_aug=True,
        bumper_mask_path: Optional[str]=None,
        seed=0,
    ):
        super().__init__()
        self.base_root = Path(base_root)
        self.mode = mode
        self.H, self.W = height, width
        self.context = tuple(context_offsets)
        self.stereo = stereo
        self.stereo_tol = stereo_time_tolerance_us
        self.min_sparse = min_sparse_points_train
        random.seed(seed)

        self.rgb_mean = np.array(mean, dtype=np.float32)[None,None,:]
        self.rgb_std  = np.array(std,  dtype=np.float32)[None,None,:]

        self.aug_rgb = augs.Compose([augs.TorchColorAug()]) if (mode=="train" and use_color_aug) else augs.Compose([])
        self.aug_geo = augs.Compose([augs.TorchGeoAug(hflip_p=0.5)]) if (mode=="train" and use_geom_aug) else augs.Compose([])

        self.bumper_mask = None
        if bumper_mask_path and Path(bumper_mask_path).is_file():
            import PIL.Image as Image
            self.bumper_mask = np.array(Image.open(bumper_mask_path).convert("L"), np.float32)/255.0

        # read split (lines relative to base_root)
        with open(self.base_root / split_file, "r") as f:
            self.centers = [line.strip() for line in f if line.strip()]

        # caches per sequence
        self._traj_cache: Dict[str, Dict] = {}
        self._camdir_cache: Dict[Tuple[str,str], List[Tuple[int,str]]] = {}

    # ----- path parsing -----
    @staticmethod
    def _parse_cam_ts(rel: str) -> Tuple[Optional[str], Optional[int]]:
        name = Path(rel).name  # e.g., cam_1_101801867.jpg
        m = re.match(r"(cam_\d+)_(\d+)\.(?:jpg|jpeg|png)$", name)
        if not m: return None, None
        return m.group(1), int(m.group(2))

    def _seq_root_from_rel(self, rel: str) -> Path:
        # split at "/img/" and take the left part
        parts = rel.replace("\\","/").split("/img/")
        if len(parts) < 2:
            # fallback: take parent of .../img
            p = self.base_root / rel
            # search upwards for "img"
            parents = p.parents
            for q in parents:
                if q.name == "img":
                    return q.parent
            return p.parent.parent.parent  # last resort
        return (self.base_root / parts[0])

    # ----- trajectory & intrinsics per sequence -----
    def _get_traj(self, seq_root: Path) -> Dict:
        key = str(seq_root)
        if key in self._traj_cache:
            return self._traj_cache[key]
        traj_path = seq_root / "export" / "output.laz.trajectory.json"
        with open(traj_path, "r") as f:
            d = json.load(f)
        self._traj_cache[key] = d
        return d

    @staticmethod
    def _K_dist_from_traj(traj: Dict, cam: str):
        ci = traj["camera_infos"][cam]
        fx,fy,cx,cy = ci["fx"],ci["fy"],ci["cx"],ci["cy"]
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)
        dist = np.array(ci.get("dist_coeffs", [0,0,0,0,0]), np.float32)
        return K, dist

    @staticmethod
    def _T_cam_to_rig_from_traj(traj: Dict, cam: str):
        x,y,z, qw,qx,qy,qz = traj["sensor_to_trajectory_poses"][cam]
        return pose_to_matrix(x,y,z, qw,qx,qy,qz)

    def _T_cam_world(self, traj: Dict, cam: str, ts_img_us: int):
        ci = traj["camera_infos"][cam]
        ts = ts_img_us + ci.get("timestamp_offset", 0)
        x,y,z, qw,qx,qy,qz = interpolate_pose(traj, ts)
        T_rig_world = pose_to_matrix(x,y,z, qw,qx,qy,qz)
        T_cam_rig = self._T_cam_to_rig_from_traj(traj, cam)
        return T_rig_world @ T_cam_rig

    # ----- cam dir index per sequence -----
    def _cam_dir_list(self, seq_root: Path, cam: str) -> List[Tuple[int,str]]:
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

    # ----- loaders (per sequence) -----
    def _load_sparse(self, seq_root: Path, cam: str, ts: int):
        p = seq_root / "sparse_ground_truth" / f"{cam}_{ts}.npz"
        if not p.is_file(): return None
        z = np.load(str(p), allow_pickle=True)
        key = f"{cam}_{ts}"
        arr = z[key].astype(np.float32) if key in z else z.get("depth", None)
        if arr is None: return None
        if self.bumper_mask is not None:
            mask = cv2.resize(self.bumper_mask, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST)
            arr = arr * mask
        return _clip_min(arr)

    def _load_single(self, seq_root: Path, cam: str, ts: int, K: np.ndarray, H: int, W: int):
        p = seq_root / "single_depth" / cam / f"{cam}_{ts}.npz"
        if not p.is_file(): return None
        z = np.load(str(p), allow_pickle=True)
        if "depth" in z:
            dep = z["depth"].astype(np.float32)
        elif "pts_cam" in z:
            dep = _project_pts_to_depth(z["pts_cam"].astype(np.float32), K, (H, W))
        else:
            return None
        if self.bumper_mask is not None:
            m = cv2.resize(self.bumper_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            dep = dep * m
        return _clip_min(dep)

    def _load_agg(self, seq_root: Path, cam: str, ts: int, K: np.ndarray, H: int, W: int):
        p = seq_root / "agg_depth" / cam / f"{cam}_{ts}.npz"
        if not p.is_file(): return None
        z = np.load(str(p), allow_pickle=True)
        if "pts_cam" not in z: return None
        dep = _project_pts_to_depth(z["pts_cam"].astype(np.float32), K, (H, W))
        if self.bumper_mask is not None:
            m = cv2.resize(self.bumper_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            dep = dep * m
        return _clip_min(dep)

    def _load_labels(self, seq_root: Path, cam: str, ts: int, img_path: str):
        p = seq_root / "labels" / f"{cam}_{ts}.txt"
        if not p.is_file(): return None
        img = cv2.imread(img_path); 
        if img is None: return None
        h, w = img.shape[:2]
        out = []
        with open(p, "r") as f:
            for line in f:
                a = line.strip().split()
                if len(a)==5:
                    cls,x_c,y_c,bw,bh = map(float, a)
                    x_c,y_c,bw,bh = x_c*w, y_c*h, bw*w, bh*h
                    x1,y1 = int(x_c-bw/2), int(y_c-bh/2)
                    x2,y2 = int(x_c+bw/2), int(y_c+bh/2)
                    out.append({"class": int(cls), "bbox": (x1,y1,x2,y2)})
        return out

    # ----- augs & crop -----
    def _apply_augs_and_crop(self, rgb_uint8: np.ndarray, depth_like: Dict[str, np.ndarray], K: np.ndarray):
        # 1) Color aug on RGB only
        out = self.aug_rgb(rgb_uint8, *([v for v in depth_like.values()] or [rgb_uint8]), K)
        rgb1 = out[0]

        # 2) Geometric aug (currently only H-flip) applied to RGB, depths, and K
        out = self.aug_geo(rgb1, *([v for v in depth_like.values()] or []), K)
        rgb2 = out[0]
        newK = out[-1].copy() if out[-1] is not None else (K.copy() if K is not None else None)
        dvals = out[1:-1]  # list aligned with depth_like.values()

        H, W = rgb2.shape[:2]

        # 3) (Optional) pad if smaller than target
        if H < self.H or W < self.W:
            ph, pw = max(0, self.H - H), max(0, self.W - W)
            rgb2 = np.pad(rgb2, ((0, ph), (0, pw), (0, 0)), mode="constant")
            if len(dvals):
                dvals = [np.pad(d, ((0, ph), (0, pw), (0, 0)), mode="constant") if d is not None else None for d in dvals]
            H, W = rgb2.shape[:2]
            # K does not change for zero-padding (image origin unchanged)

        # 4) Choose crop box
        if self.mode == "train":
            top = np.random.randint(0, H - self.H + 1)
            left = np.random.randint(0, W - self.W + 1)
        else:
            top = (H - self.H) // 2
            left = (W - self.W) // 2

        # 5) Apply crop
        rgb_c = rgb2[top:top + self.H, left:left + self.W, :]
        dct_c = {}
        for (k, _), d in zip(depth_like.items(), dvals):
            if d is None:
                dct_c[k] = None
            else:
                dct_c[k] = d[top:top + self.H, left:left + self.W, :]

        # 6) Update intrinsics for crop (shift the principal point)
        if newK is not None:
            newK[0, 2] -= float(left)
            newK[1, 2] -= float(top)

        # 7) (Optional) If you later add a resize step from (self.H, self.W) to some (Hr, Wr),
        #    you must scale K accordingly here. For now, no extra resize is done.

        return rgb_c, dct_c, newK

    # ----- utilities -----
    @staticmethod
    def _find_neighbors(idx: int, total: int, offsets: Tuple[int,...]) -> Dict[int,int]:
        out = {}
        for off in offsets:
            j = idx + off
            out[off] = (j if 0 <= j < total else (idx if off==0 else -1))
        return out

    def __len__(self): return len(self.centers)

    def __getitem__(self, i: int):
        while True:
            rel = self.centers[i]
            cam_center, ts_center = self._parse_cam_ts(rel)
            if cam_center is None: i = random.randint(0, len(self)-1); continue

            seq_root = self._seq_root_from_rel(rel)
            traj = self._get_traj(seq_root)
            cam_list = self._cam_dir_list(seq_root, cam_center)
            # locate center index by ts
            ts_list = [t for t,_ in cam_list]
            try:
                cidx = ts_list.index(ts_center)
            except ValueError:
                i = random.randint(0, len(self)-1); continue

            # intrinsics/dist for the camera
            K_ref, dist_ref = self._K_dist_from_traj(traj, cam_center)

            # build payload for one camera within this sequence
            def build_cam_payload(cam_name: str, center_idx: int):
                K0, dist0 = self._K_dist_from_traj(traj, cam_name)
                frames = {}
                rels = {}
                neigh = self._find_neighbors(center_idx, len(self._cam_dir_list(seq_root, cam_name)), self.context)

                # center
                ts_c, path_c = self._cam_dir_list(seq_root, cam_name)[neigh[0]]
                rgb_c = _rgb(Path(path_c))
                H0, W0 = rgb_c.shape[:2]
                # undistort
                rgb_bgr = cv2.cvtColor(rgb_c, cv2.COLOR_RGB2BGR)
                rgb_bgr = _undistort(rgb_bgr, K0, dist0)
                rgb_c = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

                Tcw_c = self._T_cam_world(traj, cam_name, ts_c)
                sparse_c = self._load_sparse(seq_root, cam_name, ts_c)
                single_c = self._load_single(seq_root, cam_name, ts_c, K0, H0, W0)
                agg_c    = self._load_agg(seq_root, cam_name, ts_c, K0, H0, W0)
                labels_c = self._load_labels(seq_root, cam_name, ts_c, path_c)

                depth_like = {}
                if sparse_c is not None: depth_like["sparse"] = sparse_c
                if single_c is not None: depth_like["depth_single"] = single_c
                if agg_c is not None:    depth_like["depth_agg"] = agg_c
                rgb_cc, dcts, Kc = self._apply_augs_and_crop(rgb_c, depth_like, K0)

                # gate for train on center sparse
                if self.mode=="train" and ("sparse" in dcts) and (dcts["sparse"] is not None) and (np.count_nonzero(dcts["sparse"]) < self.min_sparse):
                    return None

                frames[0] = {
                    "timestamp_us": ts_c,
                    "rgb": _chw(_normalize(rgb_cc, self.rgb_mean, self.rgb_std)),
                    "lidar_sparse": _chw(dcts.get("sparse", None)),
                    "depth_single": _chw(dcts.get("depth_single", None)),
                    "depth_agg":    _chw(dcts.get("depth_agg", None)),
                    "K": torch.from_numpy(Kc.astype(np.float32)) if Kc is not None else None,
                    "dist": torch.from_numpy(dist0.astype(np.float32)),
                    "T_cam_world": torch.from_numpy(Tcw_c.astype(np.float32)),
                    "labels": labels_c,
                }

                # neighbors
                clist = self._cam_dir_list(seq_root, cam_name)
                for off, nidx in neigh.items():
                    if off==0 or nidx < 0: continue
                    ts_n, path_n = clist[nidx]
                    rgb_n = _rgb(Path(path_n))
                    Hn, Wn = rgb_n.shape[:2]
                    rgb_bgr = cv2.cvtColor(rgb_n, cv2.COLOR_RGB2BGR)
                    rgb_bgr = _undistort(rgb_bgr, K0, dist0)
                    rgb_n = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

                    Tcw_n = self._T_cam_world(traj, cam_name, ts_n)
                    sparse_n = self._load_sparse(seq_root, cam_name, ts_n)
                    depth_like_n = {}
                    if sparse_n is not None: depth_like_n["sparse"] = sparse_n
                    rgb_nc, dcts_n, Kn = self._apply_augs_and_crop(rgb_n, depth_like_n, K0)

                    frames[off] = {
                        "timestamp_us": ts_n,
                        "rgb": _chw(_normalize(rgb_nc, self.rgb_mean, self.rgb_std)),
                        "lidar_sparse": _chw(dcts_n.get("sparse", None)),
                        "depth_single": None, "depth_agg": None,
                        "K": torch.from_numpy(Kn.astype(np.float32)) if Kn is not None else None,
                        "dist": torch.from_numpy(dist0.astype(np.float32)),
                        "T_cam_world": torch.from_numpy(Tcw_n.astype(np.float32)),
                        "labels": None,
                    }
                    rels[off] = torch.from_numpy(rel_pose(Tcw_n, Tcw_c).astype(np.float32))

                return {
                    "frames": frames,
                    "rel_poses_to_center": rels,
                    "K_ref": torch.from_numpy(K0.astype(np.float32)),
                    "dist_ref": torch.from_numpy(dist0.astype(np.float32)),
                    "T_cam_to_rig": torch.from_numpy(self._T_cam_to_rig_from_traj(traj, cam_name).astype(np.float32)),
                }

            cam_payload = build_cam_payload(cam_center, cidx)
            if cam_payload is None:
                i = random.randint(0, len(self)-1); continue

            out = {
                "meta": {
                    "dataset_root": str(self.base_root),
                    "sequence_root": str(self._seq_root_from_rel(rel)),
                    "mode": self.mode,
                    "context": list(self.context),
                    "stereo": self.stereo,
                    "center_relpath": rel,
                    "center_cam": cam_center,
                    "center_timestamp_us": ts_center,
                },
                cam_center: cam_payload,
            }

            # stereo pairing within same sequence
            if self.stereo:
                other_cam = self.CAM_RIGHT if cam_center == self.CAM_LEFT else self.CAM_LEFT
                other_list = self._cam_dir_list(self._seq_root_from_rel(rel), other_cam)
                # nearest by ts within tolerance
                times = [t for t,_ in other_list]
                import bisect
                k = bisect.bisect_left(times, ts_center)
                candidates = [k] + ([k-1] if k-1>=0 else []) + ([k+1] if k+1<len(times) else [])
                best, bestd = -1, 1e18
                for ix in candidates:
                    if 0 <= ix < len(times):
                        d = abs(times[ix] - ts_center)
                        if d < bestd:
                            bestd, best = d, ix
                if best >= 0 and bestd <= self.stereo_tol:
                    other_payload = build_cam_payload(other_cam, best)
                    if other_payload is not None:
                        out[other_cam] = other_payload
                    else:
                        out[other_cam] = None
                else:
                    out[other_cam] = None

            return out
