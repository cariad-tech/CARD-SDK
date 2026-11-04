import functools
import json
from pathlib import Path

import numpy as np


def fit_ground_plane(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("At least three wheel ground-contact points are required")
    if not np.isfinite(points).all():
        raise ValueError("Wheel ground-contact points must be finite")
    centroid = points.mean(axis=0)
    _, singular_values, vectors = np.linalg.svd(points - centroid)
    if singular_values[0] == 0 or singular_values[1] <= singular_values[0] * 1e-6:
        raise ValueError("Wheel ground-contact points are collinear or coincident")
    normal = vectors[-1]
    if normal[2] < 0:
        normal = -normal
    return normal.astype(np.float32), centroid.astype(np.float32)


@functools.lru_cache(maxsize=2)
def _contact_trajectories(sequence_root):
    path = Path(sequence_root) / "export" / "wheel_excitement.json"
    if not path.is_file():
        return {}
    with path.open() as stream:
        data = json.load(stream)
    trajectories = {}
    for name in ("FL", "FR", "RL", "RR"):
        result = data.get("wheel_results", {}).get(f"cariad_wheel_{name}", {})
        contact = result.get("contact_point_trajectory", result.get("ground_trajectory"))
        if contact is None:
            continue
        poses = contact["poses"]
        if not poses:
            raise ValueError(f"Empty wheel-contact trajectory for {name}: {path}")
        timestamps = np.asarray([pose[0] for pose in poses], dtype=np.float64)
        positions = np.asarray([pose[1][:3] for pose in poses], dtype=np.float64)
        if positions.shape != (len(timestamps), 3) or not np.isfinite(positions).all():
            raise ValueError(f"Invalid wheel-contact positions for {name}: {path}")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
            raise ValueError(f"Wheel-contact timestamps must be finite and sorted: {path}")
        trajectories[name] = timestamps, positions
    return trajectories


def road_plane_at(traj, sequence_root, timestamp_us, rig_to_world):
    """Fit exported world-frame contacts, or calibrated static rig-frame contacts."""
    trajectories = _contact_trajectories(str(Path(sequence_root).resolve()))
    points = []
    if trajectories:
        for name, (timestamps, positions) in trajectories.items():
            if not timestamps[0] <= timestamp_us <= timestamps[-1]:
                raise ValueError(f"Timestamp {timestamp_us} is outside wheel {name} contact coverage")
            upper = int(np.searchsorted(timestamps, timestamp_us))
            lower = upper if timestamps[upper] == timestamp_us else upper - 1
            for boundary in (timestamps[lower], timestamps[upper]):
                if np.searchsorted(timestamps, boundary, side="right") - np.searchsorted(timestamps, boundary) != 1:
                    raise ValueError(f"Ambiguous duplicate wheel {name} contact timestamp {boundary}")
            points.append([np.interp(timestamp_us, timestamps, positions[:, axis]) for axis in range(3)])
        source = "wheel_contact_trajectories"
    else:
        sensor_poses = {name.lower(): value for name, value in traj["sensor_to_trajectory_poses"].items()}
        for name in ("FL", "FR", "RL", "RR"):
            pose = sensor_poses.get(f"cariad_wheel_{name}_ground".lower())
            if pose is not None:
                points.append((rig_to_world @ np.array([*pose[:3], 1.0]))[:3])
        source = "wheel_contact_calibration"
    normal, point = fit_ground_plane(points)
    return normal, point, np.asarray(points, dtype=np.float32), source