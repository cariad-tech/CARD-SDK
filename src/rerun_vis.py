


import os, re, json, cv2, laspy, rerun as rr, rerun.blueprint as rrb
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from rerun.archetypes.series_lines import SeriesLines
class _Timeline:
    def __init__(self):
        self.t0_us = None
        self.i = -1  # frame index

    def advance(self, t_us: int):
        t_us = int(t_us)
        if self.t0_us is None:
            self.t0_us = t_us
        self.i += 1
        # Primary deterministic playback axis (sequence index)
        rr.set_time("frame", sequence=int(self.i))
        # Normalized sensor time in seconds since first sample
        rr.set_time("sensor_time", duration=(t_us - self.t0_us) / 1e6)



def compute_road_plane_from_wheels(traj_data, vehicle_pose):
    """Compute road plane from wheel ground contact points.
    
    The wheel ground contact points in sensor_to_trajectory_poses are static vehicle-relative positions
    that define where the wheels touch the ground relative to the vehicle body.
    
    Returns:
        plane_normal: Normal vector to the ground plane in vehicle local coordinates
        plane_point: A point on the ground plane in vehicle local coordinates
    """
    # Get wheel ground contact positions (these are vehicle-relative, not world positions)
    wheel_ground_keys = [
        "cariad_wheel_FL_Ground",
        "cariad_wheel_FR_Ground", 
        "cariad_wheel_RL_Ground",
        "cariad_wheel_RR_Ground"
    ]
    
    wheel_points_vehicle = []
    for wheel_key in wheel_ground_keys:
        if wheel_key in traj_data["sensor_to_trajectory_poses"]:
            wheel_pose = traj_data["sensor_to_trajectory_poses"][wheel_key]
            # These are already in vehicle coordinates - they represent fixed offsets from vehicle center
            wheel_point = np.array([wheel_pose[0], wheel_pose[1], wheel_pose[2]])
            wheel_points_vehicle.append(wheel_point)
            print(f"  {wheel_key} (vehicle coords): {wheel_point}")
    
    if len(wheel_points_vehicle) < 3:
        print(f"Warning: Only {len(wheel_points_vehicle)} wheel contact points found, using default plane")
        return np.array([0, 0, 1]), np.array([0, 0, -1.8])  # Default: horizontal plane 1.8m below vehicle
    
    wheel_points_vehicle = np.array(wheel_points_vehicle)
    
    # Fit a plane through the wheel contact points in vehicle coordinates
    # Use SVD to find the best-fit plane
    centroid_vehicle = np.mean(wheel_points_vehicle, axis=0)
    points_centered = wheel_points_vehicle - centroid_vehicle
    
    # SVD to find the plane normal (smallest singular vector)
    _, _, vh = np.linalg.svd(points_centered)
    normal_vehicle = vh[-1]  # Last row of V^T is the normal vector with smallest singular value
    
    # Ensure normal points "up" (positive z component in vehicle coordinates)
    if normal_vehicle[2] < 0:
        normal_vehicle = -normal_vehicle
    
    print(f"Ground plane in vehicle coords: normal={normal_vehicle}, centroid={centroid_vehicle}")
    print(f"Ground plane centroid z={centroid_vehicle[2]:.3f}m (distance below vehicle center)")
    
    return normal_vehicle, centroid_vehicle

def get_height_colors(points, vehicle_pose, road_plane_normal, road_plane_point, start_height_offset=-0.7, end_height_offset=0.0):
    """Generate colors based on height relative to road plane using plasma colormap.
    
    Args:
        points: Point cloud data in world coordinates
        vehicle_pose: Reference vehicle pose [x, y, z, qw, qx, qy, qz] in world coordinates
        road_plane_normal: Normal vector to the road plane in vehicle local coordinates
        road_plane_point: A point on the road plane in vehicle local coordinates
        start_height_offset: Lower bound offset from road plane (default: -0.7m below road)
        end_height_offset: Upper bound offset from road plane (default: 0.0m at road level)
    """
    # Transform points to vehicle's local coordinate system
    T_world_to_vehicle = np.linalg.inv(pose_to_matrix(*vehicle_pose))
    points_homogeneous = np.c_[points, np.ones(len(points))]  # Add homogeneous coordinate
    points_vehicle_local = (T_world_to_vehicle @ points_homogeneous.T).T[:, :3]
    
    # Calculate distance from each point to the road plane
    # Distance = |normal · (point - plane_point)| / |normal|
    # Since we want signed distance (positive above plane), we don't use absolute value
    plane_normal_normalized = road_plane_normal / np.linalg.norm(road_plane_normal)
    
    # Vector from plane point to each point
    point_to_plane_vectors = points_vehicle_local - road_plane_point
    
    # Signed distance to plane (positive = above road, negative = below road)
    distances_to_plane = np.dot(point_to_plane_vectors, plane_normal_normalized)
    
    # Calculate absolute range based on road plane
    start_height = start_height_offset
    end_height = end_height_offset
    
    # Initialize color array
    colors = np.zeros((len(points), 3), dtype=np.uint8)

    # Calculate range for normalization
    height_range = end_height - start_height
    
    # Dynamic masks based on distance from road plane
    mask_detailed = (distances_to_plane >= start_height) & (distances_to_plane <= end_height)
    mask_higher = distances_to_plane > end_height
    mask_lower = distances_to_plane < start_height

    # Colors for points in the specified range (using true plasma colormap)
    if np.any(mask_detailed):
        # Normalize to 0-1 range for the specified window
        norm_heights = (distances_to_plane[mask_detailed] - start_height) / height_range
        
        # True plasma colormap implementation
        # Based on matplotlib's plasma colormap values
        colors[mask_detailed] = np.column_stack([
            # Red channel: starts very low, increases dramatically after 0.5
            np.clip(255 * (0.05 + 0.95 * np.power(norm_heights, 0.5)), 0, 255).astype(np.uint8),
            # Green channel: S-curve, peaks around 0.8-0.9
            np.clip(255 * np.power(norm_heights, 1.5), 0, 255).astype(np.uint8),
            # Blue channel: starts high (purple), decreases to create the plasma effect
            np.clip(255 * (0.95 - 0.9 * np.power(norm_heights, 0.8)), 0, 255).astype(np.uint8)
        ])

    # Colors for points above the range (simple gray gradient to de-emphasize)
    if np.any(mask_higher):
        # Normalize for points above the range (up to 3m above the road plane)
        norm_heights = np.clip((distances_to_plane[mask_higher] - end_height) / 3.0, 0, 1)
        gray_value = (180 + 75 * (1 - norm_heights)).astype(np.uint8)  # Light gray gradient
        colors[mask_higher] = np.column_stack([
            gray_value,
            gray_value, 
            gray_value
        ])
    
    # Colors for points below the range (very dark to make them almost invisible)
    if np.any(mask_lower):
        colors[mask_lower] = np.column_stack([
            np.full(np.sum(mask_lower), 20, dtype=np.uint8),  # Very dark
            np.full(np.sum(mask_lower), 20, dtype=np.uint8),
            np.full(np.sum(mask_lower), 30, dtype=np.uint8)   # Slightly blue tint
        ])
    
    return colors


# --- constants you had ---
cam_left = "cam_1"
cam_right = "cam_0"

cc_colors = {
    "FL": rr.components.Color(0xff7f00ff),
    "FR": rr.components.Color(0xff4500ff),
    "RL": rr.components.Color(0x836fffff),
    "RR": rr.components.Color(0x00ff7fff),
}

# ----------------- your helpers (unchanged in spirit) -----------------
def pose_to_matrix(x, y, z, qw, qx, qy, qz):
    rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4); T[:3, :3] = rot; T[:3, 3] = [x, y, z]
    return T

def interpolate_pose(traj_data, timestamp):
    poses = traj_data["trajectory_poses"]
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
    trans1, quat1_sf = np.array(pose_1[:3]), pose_1[3:]  # [qw,qx,qy,qz]

    # linear translation
    interp_translation = (1 - t) * trans0 + t * trans1

    # ---- quaternion slerp ----
    # convert scalar-first [qw,qx,qy,qz] -> xyzw for SciPy
    def sf_to_xyzw(qsf):
        qw, qx, qy, qz = qsf
        return [qx, qy, qz, qw]

    q0_xyzw = sf_to_xyzw(quat0_sf)
    q1_xyzw = sf_to_xyzw(quat1_sf)

    rotations = Rotation.from_quat(np.array([q0_xyzw, q1_xyzw]))  # expects xyzw
    slerp = Slerp([0.0, 1.0], rotations)
    interp_xyzw = slerp([t]).as_quat()[0]  # xyzw

    # convert back to scalar-first [qw,qx,qy,qz] to match the rest of your code
    qw = float(interp_xyzw[3])
    qx, qy, qz = map(float, interp_xyzw[:3])
    interp_quat_sf = [qw, qx, qy, qz]

    return [*interp_translation.tolist(), *interp_quat_sf]

def relative_transform(pose_a, pose_b):
    T_a = pose_to_matrix(*pose_a); T_b = pose_to_matrix(*pose_b)
    T_ab = np.linalg.inv(T_b) @ T_a
    return T_ab[:3,:3], T_ab[:3,3]

def enumerate_cam_images(folder_path):
    pattern = re.compile(r'cam_.*_(\d+)\.jpg')
    files = []
    for filename in os.listdir(folder_path):
        m = pattern.match(filename)
        if m:
            timestamp = int(m.group(1))
            files.append((timestamp, folder_path + "/" + filename))
    files.sort(key=lambda x: x[0])
    return files

class ImageBuffer:
    def __init__(self, all_images): self.images = all_images
    def get_images_up_to(self, timestamp):
        result = [img for img in self.images if img[0] <= timestamp]
        self.images = [img for img in self.images if img[0] > timestamp]
        return result

def get_dataset_image_buffer(traj_data, dataset_folder, subsample=1):
    all_images = []
    for cam_name in traj_data["camera_infos"]:
        cam_path = os.path.join(dataset_folder, cam_name)
        if not os.path.isdir(cam_path):
            print(f"WARNING: no camera directory {cam_path}")
            continue
        cam_images = enumerate_cam_images(cam_path)
        for i, (timestamp, cam_frame_path) in enumerate(cam_images, start=1):
            if i % max(1, subsample) != 0: continue
            all_images.append((timestamp, cam_name, cam_frame_path))
    all_images.sort(key=lambda x: x[0])
    return ImageBuffer(all_images)

def rr_send_sensor_extrinsics(traj_data):
    rr.log("world/rig/axis", rr.Arrows3D(
        vectors=[[0.1,0,0],[0,0.1,0],[0,0,0.1]],
        colors=[[255,0,0],[0,255,0],[0,0,255]]), static=True)
    for frame_name, pose in traj_data["sensor_to_trajectory_poses"].items():
        rr.log(f"world/rig/{frame_name}", rr.Transform3D(
            translation=[pose[0], pose[1], pose[2]],
            rotation=rr.Quaternion(xyzw=[pose[4], pose[5], pose[6], pose[3]])), static=True)
        rr.log(f"world/rig/{frame_name}", rr.Arrows3D(
            vectors=[[0.03,0,0],[0,0.03,0],[0,0,0.03]],
            colors=[[255,0,0],[0,255,0],[0,0,255]]), static=True)

def rr_send_rig_trajectory(traj_data):
    timestamps=[]; positions=[]; quaternions_xyzw=[]
    for timestamp, pose in traj_data["trajectory_poses"]:
        x,y,z, qw,qx,qy,qz = pose
        timestamps.append(timestamp / 1e6)
        positions.append([x,y,z])
        quaternions_xyzw.append([qx,qy,qz,qw])
    rr.send_columns("world/rig",
        indexes=[rr.TimeColumn("sensor_time", duration=timestamps)],
        columns=rr.Transform3D.columns(translation=positions, quaternion=quaternions_xyzw))
    rr.log("world/trajectory", rr.LineStrips3D([positions]), static=True)

def read_yolo_bboxes(image_path, labels_dir):
    img = cv2.imread(image_path); h, w = img.shape[:2]
    label_file = os.path.join(labels_dir, os.path.basename(image_path).replace('.jpg','.txt'))
    bboxes=[]
    if os.path.exists(label_file):
        with open(label_file,'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts)==5:
                    cls,x_c,y_c,bw,bh = map(float, parts)
                    x_c,y_c,bw,bh = x_c*w, y_c*h, bw*w, bh*h
                    x1=int(x_c-bw/2); y1=int(y_c-bh/2)
                    x2=int(x_c+bw/2); y2=int(y_c+bh/2)
                    bboxes.append({'class':int(cls), 'bbox':(x1,y1,x2,y2)})
    return bboxes

def rr_send_imu_data(sensor_export):
    if "imu" not in sensor_export: return
    imu = sensor_export["imu"]
    ts = np.array([e["timestamp"]/1e6 for e in imu])
    accel = np.column_stack([[e["accel_x"] for e in imu],
                             [e["accel_y"] for e in imu],
                             [e["accel_z"] for e in imu]])
    gyro  = np.column_stack([[e["gyro_x"] for e in imu],
                             [e["gyro_y"] for e in imu],
                             [e["gyro_z"] for e in imu]])
    rr.send_columns("imu/accel", indexes=[rr.TimeColumn("sensor_time", duration=ts)],
                    columns=rr.Scalars.columns(scalars=accel))
    rr.send_columns("imu/gyro", indexes=[rr.TimeColumn("sensor_time", duration=ts)],
                    columns=rr.Scalars.columns(scalars=gyro))

def rr_send_gps_data(sensor_export):
    if "gps" not in sensor_export: return
    gps = sensor_export["gps"]
    gps_traj = [[g['lat_rad']/np.pi*180.0, g['lon_rad']/np.pi*180.0] for g in gps]
    rr.log("gps/trajectory", rr.GeoLineStrings(lat_lon=gps_traj,
                                               radii=rr.Radius.ui_points(2.0),
                                               colors=[0,0,255]), static=True)
    for g in gps:
        rr.set_time("sensor_time", duration=g['timestamp']/1e6)
        rr.set_time("z_gps_time", duration=g['gps_timestamp_us']/1e6)
        rr.log("gps/points", rr.GeoPoints(lat_lon=np.array([g['lat_rad'], g['lon_rad']])/np.pi*180.0,
                                          radii=rr.Radius.ui_points(4.0), colors=[255,0,0]))
    rr.reset_time()

def rr_send_wheel_excitement(wheel_excitement):
    for frame_id, wheel_result in wheel_excitement["wheel_results"].items():
        wheel_id = frame_id.split("_")[-1]
        data = np.array(wheel_result["wheel_excitement"])
        ts = data[:,0]/1e6
        rr.send_columns(f"/wheel_excitement/excitement/{wheel_id}",
                        indexes=[rr.TimeColumn("sensor_time", duration=ts)],
                        columns=rr.Scalars.columns(scalars=data[:,-1]))
        positions=[]; quats=[]
        contact = None
        if "contact_point_trajectory" in wheel_result:
            contact = wheel_result["contact_point_trajectory"]["poses"]
        elif "ground_trajectory" in wheel_result:
            contact = wheel_result["ground_trajectory"]["poses"]
        if contact:
            for timestamp, pose in contact:
                x,y,z, qw,qx,qy,qz = pose
                positions.append([x,y,z]); quats.append([qx,qy,qz,qw])
            rr.log(f"world/contact_point_trajectory_{wheel_id}",
                   rr.LineStrips3D([positions], colors=[cc_colors[wheel_id]]), static=True)

def rr_send_cam_frame(timestamp, cam_name, camera_info, cam_frame_path,
                      labels_root=None, send_agg_depth_flag=False, base_dir=None, traj_data=None):
    image = cv2.imread(cam_frame_path)
    if image is None:
        raise ValueError(f"Image cannot be loaded: {cam_frame_path}")
    fx,fy,cx,cy = camera_info['fx'],camera_info['fy'],camera_info['cx'],camera_info['cy']
    camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    dist_coeffs = np.array(camera_info['dist_coeffs'])
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    rr.set_time("sensor_time", duration=timestamp/1e6)
    rr.log(f"world/rig/{cam_name}/undistorted_image",
           rr.Pinhole(focal_length=[fx,fy], principal_point=[cx,cy],
                      width=camera_info["image_width"], height=camera_info["image_height"],
                      image_plane_distance=0.1))
    rr.log(f"world/rig/{cam_name}/undistorted_image",
           rr.Image(undistorted, color_model="BGR").compress(jpeg_quality=75))
    if labels_root:
        labels_dir = os.path.join(labels_root, "labels")
        bboxes = read_yolo_bboxes(cam_frame_path, labels_dir)
        centers=[]; half_sizes=[]; class_ids=[]
        for b in bboxes:
            x1,y1,x2,y2 = b['bbox']
            cxp = (x1+x2)/2; cyp=(y1+y2)/2; w=abs(x2-x1)/2; h=abs(y2-y1)/2
            centers.append([cxp,cyp]); half_sizes.append([w,h]); class_ids.append(b['class'])
        rr.log(f"world/rig/{cam_name}/undistorted_image/boxes",
               rr.Boxes2D(centers=centers, half_sizes=half_sizes, class_ids=class_ids))

    if send_agg_depth_flag and base_dir and traj_data is not None:
        raw_ts = int(Path(cam_frame_path).stem.split('_')[-1])
        log_agg_depth_points(raw_ts, cam_name, camera_info, base_dir, traj_data)

def send_dense_depth(timestamp_us, cam_name, cam_info, traj_data, depth_root="depth"):
    npz_path = Path(depth_root) / cam_name / f"{cam_name}_{timestamp_us}.npz"
    if not npz_path.is_file():
        return
    data = np.load(npz_path)
    pts_cam = data["pts_cam"].astype(np.float32)
    pose_cam_to_rig = traj_data["sensor_to_trajectory_poses"][cam_name]
    T_cam_to_rig = pose_to_matrix(*pose_cam_to_rig)
    pose_rig_world = interpolate_pose(traj_data, timestamp_us + cam_info.get("timestamp_offset",0))
    T_rig_to_world = pose_to_matrix(*pose_rig_world)
    T_cam_to_world = T_rig_to_world @ T_cam_to_rig
    pts_world = (T_cam_to_world @ np.c_[pts_cam, np.ones(len(pts_cam))].T).T[:, :3]
    rr.set_time("sensor_time", duration=timestamp_us/1e6)
    rr.log(f"world/dense_depth/{cam_name}", rr.Points3D(pts_world, colors=[0,255,255], radii=-1))

def log_agg_depth_points(timestamp, cam_name, cam_info, base_dir, traj_data):
    npz_path = Path(base_dir) / "agg_depth" / cam_name / f"{cam_name}_{timestamp}.npz"
    if not npz_path.is_file(): return
    data = np.load(npz_path); pts_cam = data["pts_cam"].astype(np.float32)
    pose_cam_to_rig = traj_data["sensor_to_trajectory_poses"][cam_name]
    T_cam_to_rig = pose_to_matrix(*pose_cam_to_rig)
    pose_rig_world = interpolate_pose(traj_data, timestamp + cam_info.get("timestamp_offset",0))
    T_rig_to_world = pose_to_matrix(*pose_rig_world)
    T_cam_to_world = T_rig_to_world @ T_cam_to_rig
    pts_world = (T_cam_to_world @ np.c_[pts_cam, np.ones(len(pts_cam))].T).T[:, :3]
    rr.set_time("sensor_time", duration=timestamp/1e6)
    rr.log(f"world/agg_depth/{cam_name}", rr.Points3D(pts_world, colors=[255,255,0], radii=-1))

def check_epipolar_alignment(img_left, img_right, num_matches_to_check=200):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img_left,None)
    kp2, des2 = orb.detectAndCompute(img_right,None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = [m for m in matches if abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) < 40]
    y_diffs = [abs(kp1[m.queryIdx].pt[1]-kp2[m.trainIdx].pt[1]) for m in matches[:num_matches_to_check]]
    y_diffs = np.array(y_diffs)
    print(f"vertical diff mean {np.mean(y_diffs):.3f}, min {np.min(y_diffs):.3f}, max {np.max(y_diffs):.3f}")

def stereo_rectification(traj_data, left, right):
    ci_l = traj_data['camera_infos'][cam_left]
    ci_r = traj_data['camera_infos'][cam_right]
    fx_l,fy_l,cx_l,cy_l = ci_l['fx'],ci_l['fy'],ci_l['cx'],ci_l['cy']
    fx_r,fy_r,cx_r,cy_r = ci_r['fx'],ci_r['fy'],ci_r['cx'],ci_r['cy']
    dist_l = np.array(ci_l['dist_coeffs']); dist_r = np.array(ci_r['dist_coeffs'])
    K_l = np.array([[fx_l,0,cx_l],[0,fy_l,cy_l],[0,0,1]])
    K_r = np.array([[fx_r,0,cx_r],[0,fy_r,cy_r],[0,0,1]])

    cam_pose = traj_data["sensor_to_trajectory_poses"][cam_left]
    rr.log(f"world/rig/cam_stereo", rr.Transform3D(
        translation=cam_pose[:3],
        rotation=rr.Quaternion(xyzw=[cam_pose[4], cam_pose[5], cam_pose[6], cam_pose[3]])), static=True)

    R, t = relative_transform(traj_data["sensor_to_trajectory_poses"][cam_left],
                              traj_data["sensor_to_trajectory_poses"][cam_right])

    image_size = (ci_l['image_width'], ci_l['image_height'])
    max_disparity = 16*12; block_size=11
    sgbm = cv2.StereoSGBM.create(
        minDisparity=0, numDisparities=max_disparity, blockSize=block_size,
        P1=8*3*block_size**2, P2=32*3*block_size**2, disp12MaxDiff=0,
        preFilterCap=0, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    timestamp = left[0]
    image_left = cv2.imread(left[2]); image_right = cv2.imread(right[2])
    timestamp = timestamp + ci_l["timestamp_offset"]

    R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(K_l, dist_l, K_r, dist_r, image_size, R, t, flags=cv2.CALIB_ZERO_DISPARITY)
    map1_l, map2_l = cv2.initUndistortRectifyMap(K_l, dist_l, R1, P1, image_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K_r, dist_r, R2, P2, image_size, cv2.CV_16SC2)
    rect_l = cv2.remap(image_left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(image_right, map1_r, map2_r, cv2.INTER_LINEAR)

    scale = 0.5
    if scale != 1.0:
        Q[0,3]*=scale; Q[1,3]*=scale; Q[2,3]*=scale; P1[:2,:]*=scale
        h,w = rect_l.shape[:2]; rect_l = cv2.resize(rect_l, (int(w*scale), int(h*scale)), cv2.INTER_AREA)
        h,w = rect_r.shape[:2]; rect_r = cv2.resize(rect_r, (int(w*scale), int(h*scale)), cv2.INTER_AREA)

    disp = sgbm.compute(rect_l, rect_r).astype(np.float32)/16.0
    disp[:int(disp.shape[0]/4),:] = 0; disp[int(3*disp.shape[0]/4):,:]=0
    disp_vis = cv2.normalize(disp, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)

    pts3d = cv2.reprojectImageTo3D(disp, Q).reshape(-1,3)
    mask = disp.flatten() > disp.min()
    pts3d = pts3d[mask]; pts3d = pts3d[(pts3d[:,2] > 2.5) & (pts3d[:,2] < 50)]

    T_cam_to_rig = pose_to_matrix(*traj_data["sensor_to_trajectory_poses"][cam_left])
    T_rig_to_world = pose_to_matrix(*interpolate_pose(traj_data, timestamp))
    T_cam_to_world = T_rig_to_world @ T_cam_to_rig
    pts_world = (T_cam_to_world @ np.hstack([pts3d, np.ones((pts3d.shape[0],1))]).T).T[:, :3]

    rect_pair = np.concatenate((rect_l, rect_r), axis=1)
    for h in range(0, rect_pair.shape[0], 50):
        cv2.line(rect_pair, (0,h), (rect_pair.shape[1],h), color=(0,0,255), thickness=3 if h%100==0 else 1)

    rr.set_time("sensor_time", duration=timestamp/1e6)
    rr.log("world/stereo_points", rr.Points3D(pts_world, radii=-1, colors=[255,0,255]))
    rr.log("world/rig/cam_stereo/rectified", rr.Image(rect_pair, color_model="BGR").compress(jpeg_quality=75))
    rr.log("world/rig/cam_stereo/disparity",
           rr.Pinhole(focal_length=[P1[0][0], P1[1][1]], principal_point=[P1[0][2], P1[1][2]],
                      width=disp.shape[1], height=disp.shape[0], image_plane_distance=0.1))
    rr.log("world/rig/cam_stereo/disparity", rr.Image(disp_vis, color_model="L").compress(jpeg_quality=75))

def visualize_laz(file_path, traj_data, image_buffer, send_laser_intensities,
                  perform_stereo_matching=False, labels_root=None,
                  load_dense_depth=False, send_agg_depth_flag=False, base_dir=None,
                  accumulate_points=False, color_by_height=False, time_start=None, time_end=None,
                  height_range=(-0.5, 0.5), timeline=None):
    index = json.load(open(file_path + ".index.json"))["entries"]
    index.sort(key=lambda e: e["end_ts"])
    if any(e["part_index"] != 0 for e in index):
        raise RuntimeError("Multipart LAZ not supported yet.")
    
    # Process all entries without time filtering
    filtered_index = index
    print(f"Processing all {len(filtered_index)} entries")

    # Dictionaries to store accumulated points and their timestamps for each lidar
    accumulated_points = {}
    accumulated_colors = {}
    accumulated_timestamps = {}
    ACCUMULATION_WINDOW = 8.0  # 8 seconds window

    # Get the first vehicle pose as fixed reference for height coloring
    reference_vehicle_pose = None
    road_plane_normal = None
    road_plane_point = None
    if color_by_height and filtered_index:
        first_entry_ts = filtered_index[0]["end_ts"]
        reference_vehicle_pose = interpolate_pose(traj_data, first_entry_ts)
        
        # Compute road plane from wheel contact points at reference position
        road_plane_normal, road_plane_point = compute_road_plane_from_wheels(traj_data, reference_vehicle_pose)
        
        print(f"Using fixed reference vehicle pose from timestamp {first_entry_ts/1e6:.2f}s")
        print(f"Reference vehicle pose: {reference_vehicle_pose}")
        print(f"Final road plane normal (vehicle coords): {road_plane_normal}")
        print(f"Final road plane point (vehicle coords): {road_plane_point}")

    with laspy.open(file_path) as las:
        las.seek(0)
        for entry in filtered_index:  # Use filtered_index instead of index
            las.seek(entry["point_offset_in_file"])
            points = las.read_points(entry["num_pts"])
            coords = np.vstack((points.x, points.y, points.z)).T
            colors = None
            
            end_ts = entry["end_ts"]
            if color_by_height and reference_vehicle_pose is not None and road_plane_normal is not None:
                # Use fixed reference vehicle pose and road plane for stable ground reference
                # Color points relative to the computed GROUND PLANE from wheel contact points
                # Height offsets are relative to the ground plane:
                # -0.5m = 50cm below ground (underground) = dark purple
                # +0.5m = 50cm above ground (speed bumps, obstacles) = bright yellow
                colors = get_height_colors(coords, reference_vehicle_pose, road_plane_normal, road_plane_point,
                                         start_height_offset=height_range[0], end_height_offset=height_range[1])
            elif send_laser_intensities and hasattr(points, 'intensity'):
                intens = points.intensity / 65535.0
                colors = np.vstack((intens, intens, intens)).T

            images = image_buffer.get_images_up_to(end_ts)

            for timestamp, cam_name, cam_frame_path in images:
                camera_info = traj_data['camera_infos'][cam_name]
                rr_send_cam_frame(timestamp + camera_info['timestamp_offset'],
                                  cam_name, camera_info, cam_frame_path,
                                  labels_root=base_dir,
                                  send_agg_depth_flag=send_agg_depth_flag,
                                  base_dir=base_dir, traj_data=traj_data)
                if load_dense_depth:
                    send_dense_depth(int(os.path.basename(cam_frame_path).split('_')[-1]),
                                     cam_name, camera_info, traj_data,
                                     depth_root=os.path.join(base_dir, "depth"))

            if perform_stereo_matching:
                grouped = defaultdict(dict)
                for timestamp, cam_name, cam_frame_path in images:
                    grouped[timestamp][cam_name] = (timestamp, cam_name, cam_frame_path)
                for ts, cams in grouped.items():
                    if cam_left in cams and cam_right in cams:
                        stereo_rectification(traj_data, cams[cam_left], cams[cam_right])

            rr.set_time("sensor_time", duration=end_ts/1e6)
            
            # Handle point accumulation
            laser_name = entry["laser_name"]
            current_time = end_ts / 1e6  # Convert to seconds
            
            if accumulate_points:
                if laser_name not in accumulated_points:
                    accumulated_points[laser_name] = coords
                    accumulated_colors[laser_name] = colors if colors is not None else None
                    accumulated_timestamps[laser_name] = np.full(len(coords), current_time)
                else:
                    # Remove points older than the window
                    mask = (current_time - accumulated_timestamps[laser_name]) <= ACCUMULATION_WINDOW
                    accumulated_points[laser_name] = accumulated_points[laser_name][mask]
                    if accumulated_colors[laser_name] is not None:
                        accumulated_colors[laser_name] = accumulated_colors[laser_name][mask]
                    accumulated_timestamps[laser_name] = accumulated_timestamps[laser_name][mask]
                    
                    # Add new points
                    accumulated_points[laser_name] = np.vstack((accumulated_points[laser_name], coords))
                    if colors is not None:
                        accumulated_colors[laser_name] = np.vstack((accumulated_colors[laser_name], colors))
                    accumulated_timestamps[laser_name] = np.append(accumulated_timestamps[laser_name], 
                                                                  np.full(len(coords), current_time))
                
                # Visualize accumulated points
                rr.log("world/" + laser_name, rr.Points3D(
                    accumulated_points[laser_name],
                    colors=accumulated_colors[laser_name],
                    radii=-1
                ))
            else:
                # Visualize only current points
                rr.log("world/" + laser_name, rr.Points3D(coords, colors=colors, radii=-1))

# ----------------- PUBLIC ENTRYPOINT -----------------
def run_viz(
    base_dir: str,
    subsample_cam: int = 1,
    send_laser_intensities: bool = False,
    stereo_matching: bool = False,
    load_dense_depth: bool = False,
    send_agg_depth_flag: bool = False,
    accumulate_points: bool = False,
    color_by_height: bool = False,
    time_start: float = None,
    time_end: float = None,
    height_range: tuple = (-0.5, 0.5),
):
    rr.init("Data Vis", spawn=True)
    #timeline = _Timeline()
    wheel_excitement_path = os.path.join(base_dir, "export/wheel_excitement.json")
    has_wheel_excitement = os.path.isfile(wheel_excitement_path)

    optional_views = []
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
        optional_views.append(rrb.TimeSeriesView(origin="/wheel_excitement/excitement", name="Wheel excitement"))
        rr.log("/wheel_excitement/excitement/FL", SeriesLines(colors=cc_colors["FL"]), static=True)
        rr.log("/wheel_excitement/excitement/FR", SeriesLines(colors=cc_colors["FR"]), static=True)
        rr.log("/wheel_excitement/excitement/RL", SeriesLines(colors=cc_colors["RL"]), static=True)
        rr.log("/wheel_excitement/excitement/RR", SeriesLines(colors=cc_colors["RR"]), static=True)

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
                    rrb.MapView(origin="gps", name="GPS Map View", zoom=16.0, background=rrb.MapProvider.OpenStreetMap),
                    rrb.TimeSeriesView(
                        origin="/imu/accel", name="Accel",
                        overrides={"/imu/accel/0":[rr.components.Color([255,0,0])],
                                   "/imu/accel/1":[rr.components.Color([0,255,0])],
                                   "/imu/accel/2":[rr.components.Color([0,0,255])]}
                    ),
                    rrb.TimeSeriesView(
                        origin="/imu/gyro", name="Gyro",
                        overrides={"/imu/gyro/0":[rr.components.Color([255,0,0])],
                                   "/imu/gyro/1":[rr.components.Color([0,255,0])],
                                   "/imu/gyro/2":[rr.components.Color([0,0,255])]}
                    ),
                ),
                *optional_views,
            ),
            column_shares=[60, 40],
        )
    )

    rr.connect_grpc(default_blueprint=blueprint)
    rr.send_blueprint(blueprint, make_active=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world", rr.Transform3D(translation=[0,0,0], quaternion=[0,0,0,1]), static=True)

    with open(os.path.join(base_dir, "export/output.laz.sensor_export.json"), "r") as f:
        sensor_export = json.load(f)
    rr_send_gps_data(sensor_export)
    rr_send_imu_data(sensor_export)

    if has_wheel_excitement:
        with open(wheel_excitement_path, "r") as f:
            rr_send_wheel_excitement(json.load(f))

    with open(os.path.join(base_dir, "export/output.laz.trajectory.json"), "r") as f:
        traj_data = json.load(f)
        rr_send_sensor_extrinsics(traj_data)
        rr_send_rig_trajectory(traj_data)

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
        )



# --- CLI wrapper (keep it tiny) ---
def cli_main():
    import argparse
    from rerun_vis import run_viz

    p = argparse.ArgumentParser("card-viz")
    p.add_argument("dataset_dir", type=str)
    p.add_argument("--subsample-cam", type=int, default=1)
    p.add_argument("--send-laser-intensities", action="store_true")
    p.add_argument("--stereo-matching", action="store_true")
    p.add_argument("--load-dense-depth", action="store_true")
    p.add_argument("--send-agg-depth-flag", action="store_true")
    p.add_argument("--accumulate-points", action="store_true", help="Accumulate lidar points over time")
    p.add_argument("--color-by-height", action="store_true", help="Color points by height relative to ground plane from wheel contact points")
    p.add_argument("--time-start", type=float, help="Start time in seconds for processing (optional)")
    p.add_argument("--time-end", type=float, help="End time in seconds for processing (optional)")
    p.add_argument("--height-range", nargs=2, type=float, default=[0.0, 0.5], 
                   help="Height range relative to ground plane [min_height max_height] in meters (default: 0.0 0.5)")
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
    )

if __name__ == "__main__":
    cli_main()