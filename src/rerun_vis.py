#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2025 CARIAD SE
import json
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import laspy
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.archetypes.series_lines import SeriesLines

from card_dataset.camera import undistort_bboxes
from card_dataset.ground_plane import road_plane_at

# ---------- pure-numpy quaternion math (no scipy dependency) ----------


def _quat_xyzw_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert a unit quaternion (x,y,z,w) to a 3x3 rotation matrix."""
    n = x * x + y * y + z * z + w * w
    if not np.isfinite(n) or n < 1e-10:
        raise ValueError("Quaternion must have a finite, non-zero norm")
    s = 2.0 / n
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    xx, xy, xz = x * x * s, x * y * s, x * z * s
    yy, yz, zz = y * y * s, y * z * s, z * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


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


def get_height_colors(
    points,
    road_plane_normal,
    road_plane_point,
    start_height_offset=-0.7,
    end_height_offset=0.0,
):
    """Generate colors based on height relative to road plane using a plasma-like colormap.

    Args:
        points: (N, 3) point cloud in world coordinates
        road_plane_normal: plane normal in world coordinates
        road_plane_point: point on plane in world coordinates
        start_height_offset: lower bound relative to plane (meters)
        end_height_offset: upper bound relative to plane (meters)
    """
    if not np.isfinite([start_height_offset, end_height_offset]).all():
        raise ValueError("Height range must be finite")
    if end_height_offset <= start_height_offset:
        raise ValueError("Height range maximum must be greater than its minimum")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains non-finite coordinates")
    plane_normal_norm = np.linalg.norm(road_plane_normal)
    if not np.isfinite(plane_normal_norm) or plane_normal_norm <= 1e-8:
        raise ValueError("Road-plane normal must be finite and non-zero")
    plane_normal_normalized = road_plane_normal / plane_normal_norm
    point_to_plane_vectors = points - road_plane_point
    distances_to_plane = np.dot(point_to_plane_vectors, plane_normal_normalized)

    start_height = start_height_offset
    end_height = end_height_offset
    height_range = end_height - start_height

    colors = np.zeros((len(points), 3), dtype=np.uint8)

    mask_detailed = (distances_to_plane >= start_height) & (distances_to_plane <= end_height)
    mask_higher = distances_to_plane > end_height
    mask_lower = distances_to_plane < start_height

    if np.any(mask_detailed) and height_range > 0:
        norm_heights = (distances_to_plane[mask_detailed] - start_height) / height_range
        colors[mask_detailed] = np.column_stack(
            [
                np.clip(255 * (0.05 + 0.95 * np.power(norm_heights, 0.5)), 0, 255).astype(np.uint8),
                np.clip(255 * np.power(norm_heights, 1.5), 0, 255).astype(np.uint8),
                np.clip(255 * (0.95 - 0.9 * np.power(norm_heights, 0.8)), 0, 255).astype(np.uint8),
            ]
        )

    if np.any(mask_higher):
        norm_heights = np.clip((distances_to_plane[mask_higher] - end_height) / 3.0, 0.0, 1.0)
        gray_value = (180 + 75 * (1.0 - norm_heights)).astype(np.uint8)
        colors[mask_higher] = np.column_stack([gray_value, gray_value, gray_value]).astype(np.uint8)

    if np.any(mask_lower):
        n = np.sum(mask_lower)
        colors[mask_lower] = np.column_stack(
            [
                np.full(n, 20, dtype=np.uint8),
                np.full(n, 20, dtype=np.uint8),
                np.full(n, 30, dtype=np.uint8),
            ]
        )

    return colors


cam_left = "cam_1"
cam_right = "cam_0"

cc_colors = {
    "FL": rr.components.Color(0xFF7F00FF),
    "FR": rr.components.Color(0xFF4500FF),
    "RL": rr.components.Color(0x836FFFFF),
    "RR": rr.components.Color(0x00FF7FFF),
}


def pose_to_matrix(x, y, z, qw, qx, qy, qz):
    rot = _quat_xyzw_to_matrix(qx, qy, qz, qw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = [x, y, z]
    return T


def interpolate_pose(traj_data, timestamp):
    poses = traj_data["trajectory_poses"]
    if not poses:
        raise ValueError("Trajectory has no poses")
    timestamps = [pose[0] for pose in poses]
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError("Trajectory timestamps must be strictly increasing")
    if timestamp < timestamps[0] or timestamp > timestamps[-1]:
        raise ValueError(
            f"Timestamp {timestamp} is outside trajectory coverage "
            f"[{timestamps[0]}, {timestamps[-1]}]"
        )
    i = 0
    while i < len(poses):
        if poses[i][0] > timestamp:
            break
        i += 1

    if i == 0:
        return poses[0][1]
    elif i >= len(poses):
        return poses[-1][1]

    ts_0, pose_0 = poses[i - 1]
    ts_1, pose_1 = poses[i]
    t = (timestamp - ts_0) / (ts_1 - ts_0)

    trans0, quat0_sf = np.array(pose_0[:3]), pose_0[3:]  # [qw,qx,qy,qz]
    trans1, quat1_sf = np.array(pose_1[:3]), pose_1[3:]

    interp_translation = (1.0 - t) * trans0 + t * trans1

    def sf_to_xyzw(qsf):
        qw, qx, qy, qz = qsf
        return [qx, qy, qz, qw]

    q0_xyzw = sf_to_xyzw(quat0_sf)
    q1_xyzw = sf_to_xyzw(quat1_sf)

    interp_xyzw = _slerp_xyzw(q0_xyzw, q1_xyzw, t)

    qw = float(interp_xyzw[3])
    qx, qy, qz = map(float, interp_xyzw[:3])
    interp_quat_sf = [qw, qx, qy, qz]

    return [*interp_translation.tolist(), *interp_quat_sf]


def relative_transform(pose_a, pose_b):
    T_a = pose_to_matrix(*pose_a)
    T_b = pose_to_matrix(*pose_b)
    T_ab = np.linalg.inv(T_b) @ T_a
    return T_ab[:3, :3], T_ab[:3, 3]


def enumerate_cam_images(folder_path):
    pattern = re.compile(r"cam_\d+_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    files = []
    for filename in os.listdir(folder_path):
        m = pattern.match(filename)
        if m:
            timestamp = int(m.group(1))
            files.append((timestamp, os.path.join(folder_path, filename)))
    files.sort(key=lambda x: x[0])
    return files


class ImageBuffer:
    def __init__(self, all_images, timestamp_offsets=None):
        self.images = all_images
        self.timestamp_offsets = timestamp_offsets or {}

    def get_images_up_to(self, timestamp):
        result = [img for img in self.images if img[0] + self.timestamp_offsets.get(img[1], 0) <= timestamp]
        self.images = [img for img in self.images if img[0] + self.timestamp_offsets.get(img[1], 0) > timestamp]
        return result


def get_dataset_image_buffer(traj_data, dataset_folder, subsample=1):
    if subsample < 1:
        raise ValueError("Camera subsampling interval must be at least one")
    all_images = []
    for cam_name in traj_data["camera_infos"]:
        cam_path = os.path.join(dataset_folder, cam_name)
        if not os.path.isdir(cam_path):
            print(f"WARNING: no camera directory {cam_path}")
            continue
        cam_images = enumerate_cam_images(cam_path)
        poses = traj_data["trajectory_poses"]
        if not poses:
            raise ValueError("Trajectory has no poses")
        offset = traj_data["camera_infos"][cam_name]["timestamp_offset"]
        skipped = 0
        for i, (timestamp, cam_frame_path) in enumerate(cam_images, start=1):
            if i % subsample != 0:
                continue
            if not poses[0][0] <= timestamp + offset <= poses[-1][0]:
                skipped += 1
                continue
            all_images.append((timestamp, cam_name, cam_frame_path))
        if skipped:
            warnings.warn(f"Skipping {skipped} {cam_name} frames outside calibrated pose coverage; poses are not extrapolated", RuntimeWarning, stacklevel=2)
    offsets = {camera: info["timestamp_offset"] for camera, info in traj_data["camera_infos"].items()}
    all_images.sort(key=lambda image: image[0] + offsets[image[1]])
    return ImageBuffer(all_images, offsets)


def rr_send_sensor_extrinsics(traj_data):
    rr.log(
        "world/rig/axis",
        rr.Arrows3D(
            vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
        static=True,
    )
    for frame_name, pose in traj_data["sensor_to_trajectory_poses"].items():
        rr.log(
            f"world/rig/{frame_name}",
            rr.Transform3D(
                translation=[pose[0], pose[1], pose[2]],
                rotation=rr.Quaternion(xyzw=[pose[4], pose[5], pose[6], pose[3]]),
            ),
            static=True,
        )
        rr.log(
            f"world/rig/{frame_name}",
            rr.Arrows3D(
                vectors=[[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
            static=True,
        )


def _in_time_range(timestamp_us, time_start, time_end):
    seconds = timestamp_us / 1e6
    return (time_start is None or seconds >= time_start) and (time_end is None or seconds <= time_end)


def rr_send_rig_trajectory(traj_data, time_start=None, time_end=None):
    timestamps = []
    positions = []
    quaternions_xyzw = []
    for timestamp, pose in traj_data["trajectory_poses"]:
        if not _in_time_range(timestamp, time_start, time_end):
            continue
        x, y, z, qw, qx, qy, qz = pose
        timestamps.append(timestamp / 1e6)
        positions.append([x, y, z])
        quaternions_xyzw.append([qx, qy, qz, qw])
    if not timestamps:
        return
    rr.send_columns(
        "world/rig",
        indexes=[rr.TimeColumn("sensor_time", duration=timestamps)],
        columns=rr.Transform3D.columns(translation=positions, quaternion=quaternions_xyzw),
    )
    rr.log("world/trajectory", rr.LineStrips3D([positions]), static=True)


def read_yolo_bboxes(image_path, labels_dir):
    img = cv2.imread(image_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    label_file = os.path.join(labels_dir, Path(image_path).stem + ".txt")
    bboxes = []
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_c, y_c, bw, bh = map(float, parts)
                    x_c, y_c, bw, bh = x_c * w, y_c * h, bw * w, bh * h
                    x1 = int(x_c - bw / 2)
                    y1 = int(y_c - bh / 2)
                    x2 = int(x_c + bw / 2)
                    y2 = int(y_c + bh / 2)
                    bboxes.append({"class": int(cls), "bbox": (x1, y1, x2, y2)})
    return bboxes


def rr_send_imu_data(sensor_export, time_start=None, time_end=None):
    if "imu" not in sensor_export:
        return
    imu = [sample for sample in sensor_export["imu"] if _in_time_range(sample["timestamp"], time_start, time_end)]
    if not imu:
        return
    ts = np.array([e["timestamp"] / 1e6 for e in imu])
    accel = np.column_stack(
        [
            [e["accel_x"] for e in imu],
            [e["accel_y"] for e in imu],
            [e["accel_z"] for e in imu],
        ]
    )
    gyro = np.column_stack(
        [
            [e["gyro_x"] for e in imu],
            [e["gyro_y"] for e in imu],
            [e["gyro_z"] for e in imu],
        ]
    )
    rr.send_columns(
        "imu/accel",
        indexes=[rr.TimeColumn("sensor_time", duration=ts)],
        columns=rr.Scalars.columns(scalars=accel),
    )
    rr.send_columns(
        "imu/gyro",
        indexes=[rr.TimeColumn("sensor_time", duration=ts)],
        columns=rr.Scalars.columns(scalars=gyro),
    )


def rr_send_gps_data(sensor_export, time_start=None, time_end=None):
    if "gps" not in sensor_export:
        return
    gps = [sample for sample in sensor_export["gps"] if _in_time_range(sample["timestamp"], time_start, time_end)]
    if not gps:
        return
    gps_traj = [[g["lat_rad"] / np.pi * 180.0, g["lon_rad"] / np.pi * 180.0] for g in gps]
    rr.log(
        "gps/trajectory",
        rr.GeoLineStrings(
            lat_lon=gps_traj,
            radii=rr.Radius.ui_points(2.0),
            colors=[0, 0, 255],
        ),
        static=True,
    )
    for g in gps:
        rr.set_time("sensor_time", duration=g["timestamp"] / 1e6)
        rr.set_time("z_gps_time", duration=g["gps_timestamp_us"] / 1e6)
        rr.log(
            "gps/points",
            rr.GeoPoints(
                lat_lon=np.array([g["lat_rad"], g["lon_rad"]]) / np.pi * 180.0,
                radii=rr.Radius.ui_points(4.0),
                colors=[255, 0, 0],
            ),
        )
    rr.reset_time()


def rr_send_wheel_excitement(wheel_excitement, time_start=None, time_end=None):
    for frame_id, wheel_result in wheel_excitement["wheel_results"].items():
        wheel_id = frame_id.split("_")[-1]
        samples = [sample for sample in wheel_result["wheel_excitement"] if _in_time_range(sample[0], time_start, time_end)]
        if samples:
            data = np.array(samples)
            rr.send_columns(
                f"/wheel_excitement/excitement/{wheel_id}",
                indexes=[rr.TimeColumn("sensor_time", duration=data[:, 0] / 1e6)],
                columns=rr.Scalars.columns(scalars=data[:, -1]),
            )

        positions = []
        contact = None
        if "contact_point_trajectory" in wheel_result:
            contact = wheel_result["contact_point_trajectory"]["poses"]
        elif "ground_trajectory" in wheel_result:
            contact = wheel_result["ground_trajectory"]["poses"]
        if contact:
            for timestamp, pose in contact:
                if _in_time_range(timestamp, time_start, time_end):
                    positions.append(pose[:3])
            if positions:
                rr.log(
                    f"world/contact_point_trajectory_{wheel_id}",
                    rr.LineStrips3D([positions], colors=[cc_colors[wheel_id]]),
                    static=True,
                )


def log_agg_depth_points(
    timestamp,
    cam_name,
    cam_info,
    base_dir,
    traj_data,
    color_by_height=False,
    height_range=(-0.5, 0.5),
):
    npz_path = Path(base_dir) / "agg_depth" / cam_name / f"{cam_name}_{timestamp}.npz"
    if not npz_path.is_file():
        warnings.warn(f"Aggregated depth unavailable for {cam_name} under {base_dir}/agg_depth", RuntimeWarning, stacklevel=2)
        return False

    with np.load(npz_path, allow_pickle=False) as data:
        pts_cam = data["pts_cam"].astype(np.float32)
    if pts_cam.ndim != 2 or pts_cam.shape[1] != 3 or not np.isfinite(pts_cam).all():
        raise ValueError(f"Expected finite Nx3 aggregate points: {npz_path}")

    intrinsics = np.array([[cam_info["fx"], 0, cam_info["cx"]], [0, cam_info["fy"], cam_info["cy"]], [0, 0, 1]], dtype=np.float64)
    height, width = int(cam_info["image_height"]), int(cam_info["image_width"])
    depth = np.full((height, width), np.inf, dtype=np.float32)
    visible = pts_cam[(pts_cam[:, 2] >= 0.1) & (pts_cam[:, 2] <= 80.0)]
    if len(visible):
        pixels = cv2.projectPoints(visible, np.zeros(3), np.zeros(3), intrinsics, None)[0][:, 0]
        inside = np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        columns, rows = pixels[inside].astype(np.int32).T
        np.minimum.at(depth, (rows, columns), visible[inside, 2])
    depth[~np.isfinite(depth)] = 0

    pose_cam_to_rig = traj_data["sensor_to_trajectory_poses"][cam_name]
    T_cam_to_rig = pose_to_matrix(*pose_cam_to_rig)

    pose_rig_world = interpolate_pose(traj_data, timestamp + cam_info["timestamp_offset"])
    T_rig_to_world = pose_to_matrix(*pose_rig_world)
    T_cam_to_world = T_rig_to_world @ T_cam_to_rig

    pts_world = (T_cam_to_world @ np.c_[pts_cam, np.ones(len(pts_cam))].T).T[:, :3]

    rr.set_time("sensor_time", duration=(timestamp + cam_info["timestamp_offset"]) / 1e6)
    depth_entity = f"world/rig/{cam_name}/agg_depth"
    rr.log(depth_entity, rr.Pinhole(image_from_camera=intrinsics, width=width, height=height, image_plane_distance=0.1))
    rr.log(depth_entity, rr.DepthImage(depth, meter=1.0))

    colors = None
    if color_by_height:
        normal, point, _, _ = road_plane_at(traj_data, base_dir, timestamp + cam_info["timestamp_offset"], T_rig_to_world)
        colors = get_height_colors(
            pts_world,
            normal,
            point,
            start_height_offset=height_range[0],
            end_height_offset=height_range[1],
        )

    rr.log(
        f"world/agg_depth/{cam_name}",
        rr.Points3D(
            pts_world,
            colors=colors if colors is not None else [255, 255, 0],
            radii=-1,
        ),
    )
    return True


def rr_send_cam_frame(
    timestamp,
    cam_name,
    camera_info,
    cam_frame_path,
    labels_root=None,
    send_agg_depth_flag=False,
    base_dir=None,
    traj_data=None,
    color_by_height=False,
    height_range=(-0.5, 0.5),
):
    image = cv2.imread(cam_frame_path)
    if image is None:
        raise ValueError(f"Image cannot be loaded: {cam_frame_path}")

    fx, fy, cx, cy = (
        camera_info["fx"],
        camera_info["fy"],
        camera_info["cx"],
        camera_info["cy"],
    )
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.array(camera_info["dist_coeffs"])

    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

    rr.set_time("sensor_time", duration=timestamp / 1e6)
    rr.log(
        f"world/rig/{cam_name}/undistorted_image",
        rr.Pinhole(
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            width=camera_info["image_width"],
            height=camera_info["image_height"],
            image_plane_distance=0.1,
        ),
    )
    rr.log(
        f"world/rig/{cam_name}/undistorted_image",
        rr.Image(undistorted, color_model="BGR").compress(jpeg_quality=75),
    )

    if labels_root:
        labels_dir = os.path.join(labels_root, "labels")
        bboxes = read_yolo_bboxes(cam_frame_path, labels_dir)
        bboxes = undistort_bboxes(bboxes, camera_matrix, dist_coeffs, camera_matrix)
        centers = []
        half_sizes = []
        class_ids = []
        for b in bboxes:
            x1, y1, x2, y2 = b["bbox"]
            cxp = (x1 + x2) / 2.0
            cyp = (y1 + y2) / 2.0
            w = abs(x2 - x1) / 2.0
            h = abs(y2 - y1) / 2.0
            centers.append([cxp, cyp])
            half_sizes.append([w, h])
            class_ids.append(b["class"])
        rr.log(
            f"world/rig/{cam_name}/undistorted_image/boxes",
            rr.Boxes2D(
                centers=centers,
                half_sizes=half_sizes,
                class_ids=class_ids,
            ),
        )

    if send_agg_depth_flag and base_dir and traj_data is not None:
        raw_ts = int(Path(cam_frame_path).stem.split("_")[-1])
        log_agg_depth_points(
            raw_ts,
            cam_name,
            camera_info,
            base_dir,
            traj_data,
            color_by_height=color_by_height,
            height_range=height_range,
        )


def send_dense_depth(timestamp_us, cam_name, cam_info, traj_data, depth_root="depth"):
    npz_path = Path(depth_root) / cam_name / f"{cam_name}_{timestamp_us}.npz"
    if not npz_path.is_file():
        warnings.warn(f"Dense depth unavailable for {cam_name} under {depth_root}", RuntimeWarning, stacklevel=2)
        return False
    with np.load(npz_path, allow_pickle=False) as data:
        depth = data["depth"].astype(np.float32)
    expected_shape = (cam_info["image_height"], cam_info["image_width"])
    if depth.shape != expected_shape:
        raise ValueError(f"Expected dense depth shape {expected_shape}, got {depth.shape}: {npz_path}")
    rows, columns = np.nonzero(np.isfinite(depth) & (depth >= 0.1) & (depth <= 80.0))
    depths = depth[rows, columns]
    pts_cam = np.column_stack(((columns - cam_info["cx"]) * depths / cam_info["fx"], (rows - cam_info["cy"]) * depths / cam_info["fy"], depths))
    pose_cam_to_rig = traj_data["sensor_to_trajectory_poses"][cam_name]
    T_cam_to_rig = pose_to_matrix(*pose_cam_to_rig)
    pose_rig_world = interpolate_pose(traj_data, timestamp_us + cam_info["timestamp_offset"])
    T_rig_to_world = pose_to_matrix(*pose_rig_world)
    T_cam_to_world = T_rig_to_world @ T_cam_to_rig
    pts_world = (T_cam_to_world @ np.c_[pts_cam, np.ones(len(pts_cam))].T).T[:, :3]
    rr.set_time("sensor_time", duration=(timestamp_us + cam_info["timestamp_offset"]) / 1e6)
    rr.log(
        f"world/dense_depth/{cam_name}",
        rr.Points3D(pts_world, colors=[0, 255, 255], radii=-1),
    )
    return True


def stereo_rectification(traj_data, left, right):
    ci_l = traj_data["camera_infos"][cam_left]
    ci_r = traj_data["camera_infos"][cam_right]

    fx_l, fy_l, cx_l, cy_l = (
        ci_l["fx"],
        ci_l["fy"],
        ci_l["cx"],
        ci_l["cy"],
    )
    fx_r, fy_r, cx_r, cy_r = (
        ci_r["fx"],
        ci_r["fy"],
        ci_r["cx"],
        ci_r["cy"],
    )

    dist_l = np.array(ci_l["dist_coeffs"])
    dist_r = np.array(ci_r["dist_coeffs"])
    K_l = np.array([[fx_l, 0, cx_l], [0, fy_l, cy_l], [0, 0, 1]])
    K_r = np.array([[fx_r, 0, cx_r], [0, fy_r, cy_r], [0, 0, 1]])

    R, t = relative_transform(
        traj_data["sensor_to_trajectory_poses"][cam_left],
        traj_data["sensor_to_trajectory_poses"][cam_right],
    )

    image_size = (ci_l["image_width"], ci_l["image_height"])
    max_disparity = 16 * 12
    block_size = 11
    sgbm = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=max_disparity,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=0,
        preFilterCap=0,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    timestamp = left[0]
    image_left = cv2.imread(left[2])
    image_right = cv2.imread(right[2])
    if image_left is None or image_right is None:
        raise ValueError(f"Stereo images cannot be loaded: {left[2]}, {right[2]}")
    timestamp = timestamp + ci_l["timestamp_offset"]

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l.astype(np.float64),
        dist_l.astype(np.float64),
        K_r.astype(np.float64),
        dist_r.astype(np.float64),
        image_size,
        R.astype(np.float64),
        t.astype(np.float64).reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
    )
    camera_to_rig = pose_to_matrix(*traj_data["sensor_to_trajectory_poses"][cam_left])
    rr.log(
        "world/rig/cam_stereo",
        rr.Transform3D(translation=camera_to_rig[:3, 3], mat3x3=camera_to_rig[:3, :3] @ R1.T),
        static=True,
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(K_l, dist_l, R1, P1, image_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K_r, dist_r, R2, P2, image_size, cv2.CV_16SC2)
    rect_l = cv2.remap(image_left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(image_right, map1_r, map2_r, cv2.INTER_LINEAR)

    scale = 0.5
    if scale != 1.0:
        Q[0, 3] *= scale
        Q[1, 3] *= scale
        Q[2, 3] *= scale
        P1[:2, :] *= scale
        h, w = rect_l.shape[:2]
        rect_l = cv2.resize(rect_l, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
        h, w = rect_r.shape[:2]
        rect_r = cv2.resize(rect_r, (int(w * scale), int(h * scale)), cv2.INTER_AREA)

    disp = sgbm.compute(rect_l, rect_r).astype(np.float32) / 16.0
    disp[: int(disp.shape[0] / 4), :] = 0
    disp[int(3 * disp.shape[0] / 4) :, :] = 0
    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    pts3d = cv2.reprojectImageTo3D(disp, Q).reshape(-1, 3)
    mask = disp.flatten() > disp.min()
    pts3d = pts3d[mask]
    pts3d = pts3d[(pts3d[:, 2] > 2.5) & (pts3d[:, 2] < 50.0)]
    pts3d = (R1.T @ pts3d.T).T

    T_cam_to_rig = pose_to_matrix(*traj_data["sensor_to_trajectory_poses"][cam_left])
    T_rig_to_world = pose_to_matrix(*interpolate_pose(traj_data, timestamp))
    T_cam_to_world = T_rig_to_world @ T_cam_to_rig

    pts_world = (T_cam_to_world @ np.hstack([pts3d, np.ones((pts3d.shape[0], 1))]).T).T[:, :3]

    rect_pair = np.concatenate((rect_l, rect_r), axis=1)
    for h in range(0, rect_pair.shape[0], 50):
        cv2.line(
            rect_pair,
            (0, h),
            (rect_pair.shape[1], h),
            color=(0, 0, 255),
            thickness=3 if h % 100 == 0 else 1,
        )

    rr.set_time("sensor_time", duration=timestamp / 1e6)
    rr.log(
        "world/stereo_points",
        rr.Points3D(pts_world, radii=-1, colors=[255, 0, 255]),
    )
    rr.log(
        "world/rig/cam_stereo/rectified",
        rr.Image(rect_pair, color_model="BGR").compress(jpeg_quality=75),
    )
    rr.log(
        "world/rig/cam_stereo/disparity",
        rr.Pinhole(
            focal_length=[P1[0][0], P1[1][1]],
            principal_point=[P1[0][2], P1[1][2]],
            width=disp.shape[1],
            height=disp.shape[0],
            image_plane_distance=0.1,
        ),
    )
    rr.log(
        "world/rig/cam_stereo/disparity",
        rr.Image(disp_vis, color_model="L").compress(jpeg_quality=75),
    )


def visualize_laz(
    file_path,
    traj_data,
    image_buffer,
    send_laser_intensities,
    perform_stereo_matching=False,
    labels_root=None,
    load_dense_depth=False,
    send_agg_depth_flag=False,
    base_dir=None,
    accumulate_points=False,
    color_by_height=False,
    time_start=None,
    time_end=None,
    height_range=(-0.5, 0.5),
    dense_depth_dir="depth",
):
    with open(file_path + ".index.json") as index_file:
        index = json.load(index_file)["entries"]
    index.sort(key=lambda e: e["end_ts"])
    if any(e["part_index"] != 0 for e in index):
        raise RuntimeError("Multipart LAZ not supported yet.")

    # Optional time filtering (seconds)
    if time_start is not None or time_end is not None:

        def in_range(e):
            t_sec = e["end_ts"] / 1e6
            if time_start is not None and t_sec < time_start:
                return False
            if time_end is not None and t_sec > time_end:
                return False
            return True

        filtered_index = [e for e in index if in_range(e)]
    else:
        filtered_index = index

    print(f"Processing {len(filtered_index)} entries (of {len(index)} total)")
    if not filtered_index:
        raise ValueError("No LAZ entries in selected time range")
    if time_start is not None:
        image_buffer.get_images_up_to(int(time_start * 1e6) - 1)

    # Point accumulation per laser
    accumulated_points = {}
    accumulated_colors = {}
    accumulated_timestamps = {}
    ACCUMULATION_WINDOW = 8.0  # seconds

    with laspy.open(file_path) as las:
        las.seek(0)
        for entry in filtered_index:
            las.seek(entry["point_offset_in_file"])
            points = las.read_points(entry["num_pts"])
            coords = np.vstack((points.x, points.y, points.z)).T
            colors = None

            end_ts = entry["end_ts"]

            if color_by_height:
                rig_to_world = pose_to_matrix(*interpolate_pose(traj_data, end_ts))
                road_plane_normal, road_plane_point, _, _ = road_plane_at(traj_data, base_dir, end_ts, rig_to_world)
                colors = get_height_colors(
                    coords,
                    road_plane_normal,
                    road_plane_point,
                    start_height_offset=height_range[0],
                    end_height_offset=height_range[1],
                )
            elif send_laser_intensities and hasattr(points, "intensity"):
                intens = points.intensity / 65535.0
                colors = np.vstack((intens, intens, intens)).T

            images = image_buffer.get_images_up_to(end_ts)

            for timestamp, cam_name, cam_frame_path in images:
                camera_info = traj_data["camera_infos"][cam_name]
                rr_send_cam_frame(
                    timestamp + camera_info["timestamp_offset"],
                    cam_name,
                    camera_info,
                    cam_frame_path,
                    labels_root=base_dir,
                    send_agg_depth_flag=send_agg_depth_flag,
                    base_dir=base_dir,
                    traj_data=traj_data,
                    color_by_height=color_by_height,
                    height_range=height_range,
                )
                if load_dense_depth:
                    send_dense_depth(
                        int(Path(cam_frame_path).stem.split("_")[-1]),
                        cam_name,
                        camera_info,
                        traj_data,
                        depth_root=os.path.join(base_dir, dense_depth_dir),
                    )

            if perform_stereo_matching:
                grouped = defaultdict(dict)
                for t_img, cam_name, cam_frame_path in images:
                    grouped[t_img][cam_name] = (t_img, cam_name, cam_frame_path)
                for cams in grouped.values():
                    if cam_left in cams and cam_right in cams:
                        stereo_rectification(traj_data, cams[cam_left], cams[cam_right])

            rr.set_time("sensor_time", duration=end_ts / 1e6)

            laser_name = entry["laser_name"]
            current_time = end_ts / 1e6

            if accumulate_points:
                if laser_name not in accumulated_points:
                    accumulated_points[laser_name] = coords
                    accumulated_colors[laser_name] = colors.copy() if colors is not None else None
                    accumulated_timestamps[laser_name] = np.full(len(coords), current_time)
                else:
                    # Remove older than window
                    mask = (
                        current_time - accumulated_timestamps[laser_name]
                    ) <= ACCUMULATION_WINDOW
                    accumulated_points[laser_name] = accumulated_points[laser_name][mask]
                    if accumulated_colors[laser_name] is not None:
                        accumulated_colors[laser_name] = accumulated_colors[laser_name][mask]
                    accumulated_timestamps[laser_name] = accumulated_timestamps[laser_name][mask]

                    # Append new
                    accumulated_points[laser_name] = np.vstack(
                        (accumulated_points[laser_name], coords)
                    )
                    if colors is not None:
                        if accumulated_colors[laser_name] is None:
                            accumulated_colors[laser_name] = colors.copy()
                        else:
                            accumulated_colors[laser_name] = np.vstack(
                                (
                                    accumulated_colors[laser_name],
                                    colors,
                                )
                            )
                    accumulated_timestamps[laser_name] = np.append(
                        accumulated_timestamps[laser_name],
                        np.full(len(coords), current_time),
                    )

                rr.log(
                    f"world/{laser_name}",
                    rr.Points3D(
                        accumulated_points[laser_name],
                        colors=accumulated_colors[laser_name],
                        radii=-1,
                    ),
                )
            else:
                rr.log(
                    f"world/{laser_name}",
                    rr.Points3D(coords, colors=colors, radii=-1),
                )


def run_viz(
    base_dir: str,
    subsample_cam: int = 1,
    send_laser_intensities: bool = False,
    stereo_matching: bool = False,
    load_dense_depth: bool = False,
    send_agg_depth_flag: bool = False,
    accumulate_points: bool = False,
    color_by_height: bool = False,
    time_start: float | None = None,
    time_end: float | None = None,
    height_range: tuple = (-0.5, 0.5),
    save_path: str | None = None,
    dense_depth_dir: str = "depth",
):
    if subsample_cam < 1:
        raise ValueError("Camera subsampling interval must be at least one")
    if not np.isfinite(height_range).all() or height_range[0] >= height_range[1]:
        raise ValueError("Height range must be finite and increasing")
    if time_start is not None and not np.isfinite(time_start) or time_end is not None and not np.isfinite(time_end):
        raise ValueError("Time bounds must be finite")
    if time_start is not None and time_end is not None and time_start > time_end:
        raise ValueError("Start time must not exceed end time")
    rr.init("Data Vis", spawn=save_path is None, strict=True)
    if save_path is not None:
        rr.save(save_path)

    wheel_excitement_path = os.path.join(base_dir, "export/wheel_excitement.json")
    has_wheel_excitement = os.path.isfile(wheel_excitement_path)

    optional_views = []
    if send_agg_depth_flag:
        optional_views.append(
            rrb.Spatial2DView(
                name="Aggregated depth (cam_1)",
                origin="/world/rig/cam_1/agg_depth",
                contents=["+ /world/rig/cam_1/agg_depth/**"],
            )
        )
    if stereo_matching:
        optional_views.append(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    name="cam_stereo_rectified",
                    origin="/world/rig/cam_stereo/rectified",
                    contents=["+ /world/rig/cam_stereo/rectified/**"],
                ),
                rrb.Spatial2DView(
                    name="disparity (cam left)",
                    origin="/world/rig/cam_stereo/disparity",
                    contents=["+ /world/rig/cam_stereo/disparity/**"],
                ),
            )
        )

    if has_wheel_excitement:
        optional_views.append(
            rrb.TimeSeriesView(origin="/wheel_excitement/excitement", name="Wheel excitement")
        )
        rr.log(
            "/wheel_excitement/excitement/FL",
            SeriesLines(colors=cc_colors["FL"]),
            static=True,
        )
        rr.log(
            "/wheel_excitement/excitement/FR",
            SeriesLines(colors=cc_colors["FR"]),
            static=True,
        )
        rr.log(
            "/wheel_excitement/excitement/RL",
            SeriesLines(colors=cc_colors["RL"]),
            static=True,
        )
        rr.log(
            "/wheel_excitement/excitement/RR",
            SeriesLines(colors=cc_colors["RR"]),
            static=True,
        )

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world/rig",
                name="3D Laser View",
                contents=["+ /world/rig/**", "+ /world/trajectory/**", "+ /world/**"],
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="cam_1",
                        origin="/world/rig/cam_1/undistorted_image",
                        contents=["+ /world/rig/cam_1/**"],
                    ),
                    rrb.Spatial2DView(
                        name="cam_0",
                        origin="/world/rig/cam_0/undistorted_image",
                        contents=["+ /world/rig/cam_0/**"],
                    ),
                ),
                rrb.Tabs(
                    rrb.MapView(
                        origin="gps",
                        name="GPS Map View",
                        zoom=16.0,
                        background=rrb.MapProvider.OpenStreetMap,
                    ),
                    rrb.TimeSeriesView(
                        origin="/imu/accel",
                        name="Accel",
                        overrides={
                            "/imu/accel/0": [SeriesLines(colors=[255, 0, 0])],
                            "/imu/accel/1": [SeriesLines(colors=[0, 255, 0])],
                            "/imu/accel/2": [SeriesLines(colors=[0, 0, 255])],
                        },
                    ),
                    rrb.TimeSeriesView(
                        origin="/imu/gyro",
                        name="Gyro",
                        overrides={
                            "/imu/gyro/0": [SeriesLines(colors=[255, 0, 0])],
                            "/imu/gyro/1": [SeriesLines(colors=[0, 255, 0])],
                            "/imu/gyro/2": [SeriesLines(colors=[0, 0, 255])],
                        },
                    ),
                ),
                *optional_views,
            ),
            column_shares=[60, 40],
        )
    )

    if save_path is None:
        rr.connect_grpc(default_blueprint=blueprint)
    rr.send_blueprint(blueprint, make_active=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world",
        rr.Transform3D(translation=[0, 0, 0], quaternion=[0, 0, 0, 1]),
        static=True,
    )

    with open(os.path.join(base_dir, "export/output.laz.sensor_export.json"), "r") as f:
        sensor_export = json.load(f)
    rr_send_gps_data(sensor_export, time_start, time_end)
    rr_send_imu_data(sensor_export, time_start, time_end)

    if has_wheel_excitement:
        with open(wheel_excitement_path, "r") as f:
            rr_send_wheel_excitement(json.load(f), time_start, time_end)

    with open(os.path.join(base_dir, "export/output.laz.trajectory.json"), "r") as f:
        traj_data = json.load(f)

    rr_send_sensor_extrinsics(traj_data)
    rr_send_rig_trajectory(traj_data, time_start, time_end)

    dir_img = os.path.join(base_dir, "img")
    image_buffer = get_dataset_image_buffer(traj_data, dir_img, subsample=subsample_cam)

    visualize_laz(
        os.path.join(base_dir, "export/output.laz"),
        traj_data,
        image_buffer,
        send_laser_intensities,
        perform_stereo_matching=stereo_matching,
        labels_root=base_dir,
        load_dense_depth=load_dense_depth,
        send_agg_depth_flag=send_agg_depth_flag,
        base_dir=base_dir,
        accumulate_points=accumulate_points,
        color_by_height=color_by_height,
        time_start=time_start,
        time_end=time_end,
        height_range=height_range,
        dense_depth_dir=dense_depth_dir,
    )
    if save_path is not None:
        rr.get_global_data_recording().flush()


def cli_main():
    import argparse

    p = argparse.ArgumentParser("card-viz")
    p.add_argument("dataset_dir", type=str)
    p.add_argument("--save", type=str, help="Write a local .rrd recording instead of opening the GUI")
    p.add_argument("--subsample-cam", type=int, default=1)
    p.add_argument("--send-laser-intensities", action="store_true")
    p.add_argument("--stereo-matching", action="store_true")
    p.add_argument("--load-dense-depth", action="store_true")
    p.add_argument("--dense-depth-dir", default="depth", help="Optional custom depth-map folder, relative to the sequence or absolute")
    p.add_argument("--send-agg-depth-flag", action="store_true", help="Add aggregate 3D points and a metric depth-image panel; raw LiDAR remains enabled")
    p.add_argument(
        "--accumulate-points",
        action="store_true",
        help="Accumulate lidar points over time",
    )
    p.add_argument(
        "--color-by-height",
        action="store_true",
        help="Color points and agg_depth by height relative to ground plane",
    )
    p.add_argument(
        "--time-start",
        type=float,
        help="Start time in seconds for processing (optional)",
    )
    p.add_argument(
        "--time-end",
        type=float,
        help="End time in seconds for processing (optional)",
    )
    p.add_argument(
        "--height-range",
        nargs=2,
        type=float,
        default=[0.0, 0.5],
        help="Height range relative to ground plane [min max] in meters (default: 0.0 0.5)",
    )

    args = p.parse_args()

    run_viz(
        base_dir=args.dataset_dir,
        subsample_cam=args.subsample_cam,
        send_laser_intensities=args.send_laser_intensities,
        stereo_matching=args.stereo_matching,
        load_dense_depth=args.load_dense_depth,
        send_agg_depth_flag=args.send_agg_depth_flag,
        accumulate_points=args.accumulate_points,
        color_by_height=args.color_by_height,
        time_start=args.time_start,
        time_end=args.time_end,
        height_range=tuple(args.height_range),
        save_path=args.save,
        dense_depth_dir=args.dense_depth_dir,
    )


if __name__ == "__main__":
    cli_main()
