# 👁️ CARD Visualization Tool (card-viz)

The card-viz tool allows you to visualize the CARD dataset in 3D using the Rerun SDK. It supports replaying LiDAR data, camera images, IMU/GPS data, and computed disparities.

## 🎥 Demo
[Insert a GIF or Video link of the Rerun output here showing the car moving through the point cloud]

## 🚀 Usage
The basic command syntax is:
```bash
card-viz <path_to_dataset_folder> [OPTIONS]
```

### Common Examples

#### 1. Standard View (Lightweight)
View the trajectory, images, and basic point cloud.
```bash
card-viz <path-to-sequence>
```

#### 2. High-Fidelity Analysis
Accumulate LiDAR points to see the map build up, and color points by their height relative to the road plane.
```bash
card-viz <path-to-sequence> --accumulate-points --color-by-height
```

#### 3. Stereo Depth & Intensity
Enable stereo matching (disparity map calculation) and use laser intensity for coloring.
```bash
card-viz <path-to-sequence> --stereo-matching --send-laser-intensities
```

## ⚙️ CLI Options Reference
You can view the help menu at any time by running:
```bash
card-viz -h
```

### Core Options
| Flag                | Type     | Description                                                                 |
|---------------------|----------|-----------------------------------------------------------------------------|
| dataset_dir         | Required | Path to the specific dataset folder (e.g., <path-to-sequence>).          |
| --subsample-camint  | int      | Reduce camera framerate to improve performance. Default is 1 (load every frame). Set to 2 or 4 for smoother playback on slower machines. |
| --time-start        | float    | Start the visualization at a specific timestamp (in seconds).               |
| --time-end          | float    | Stop the visualization at a specific timestamp (in seconds).                |

### LiDAR & Point Cloud Options
| Flag                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| --send-laser-intensities| If set, LiDAR points are colored based on return intensity instead of solid colors. |
| --accumulate-points    | Highly Recommended. Instead of showing only the current scan, points are kept in the viewer for a few seconds, creating a dense map view of the environment. |
| --color-by-height      | Colors the point cloud based on height relative to the computed road plane (Ground Contact). |
| --height-range         | Usage: --height-range MIN MAX. Defines the meters above/below the road plane for the color gradient. Default: 0.0 0.5. |

## 🛣️ Ground Plane Estimation
When `--color-by-height` is used, the script dynamically calculates the road plane based on the vehicle's wheel contact points.

- **Red/Blue Gradients:** Indicate obstacles or terrain features above the road.
- **Grey:** Indicates points significantly above the vehicle (e.g., buildings/trees).

## 🛞 Wheel Excitement
A time-series graph will automatically appear in the Rerun dashboard showing the vertical excitement of all 4 wheels.


### Camera & Depth Options
| Flag                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| --stereo-matching      | Performs real-time stereo rectification and disparity calculation between Left/Right cameras. Note: This is computationally expensive. |
| --load-dense-depth     | Loads pre-computed dense depth maps (.npz files) if they exist in the dataset folder. |
| --send-agg-depth-flag  | Visualizes aggregated depth points projected into the 3D world.               |

---

## 4. Basic Usage

With the environment activated and the project installed, you can launch the viewer with

```bash
card-viz <path-to-sequence>
```

This will

- start the Rerun viewer application, and  
- by default visualize:
	- the vehicle rig and its 3D trajectory,
	- LiDAR scans in 3D (`/world/<laser_name>`),
	- undistorted stereo images from `cam_0` and `cam_1`,
	- the GPS track and IMU signals

The default visualization is deliberately kept simple so that additional options can be enabled via command-line flags as described below.

---

## 5. Command-Line Options

This section summarises the most relevant `card-viz` options and their effects.

### 5.1 LiDAR intensities

```bash
card-viz <path-to-sequence> --send-laser-intensities
```

- Colors the LiDAR points by **intensity** instead of a uniform color.  
- Useful for checking sensor returns and reflectivity differences.

### 5.2 Online stereo matching and epipolar check

```bash
card-viz <path-to-sequence> --stereo-matching
```

- Runs **online stereo matching** using Semi-Global Matching (SGM) on the rectified stereo pair.  
- Visualizes:
	- the resulting 3D point cloud from stereo disparity, and  
	- a stacked rectified stereo image with horizontal epipolar lines.

This mode is primarily intended for checking the **accuracy of the stereo calibration**:

- In the rectified view, corresponding scene points should lie close to the same horizontal epipolar line.  
- The generated stereo point cloud can be compared visually with LiDAR scans and aggregated depth.

### 5.3 Aggregated / dense depth visualization

```bash
card-viz <path-to-sequence> --send-agg-depth-flag
```

- Enables visualization of the **per-image aggregated depth point clouds** stored under `agg_depth/`.  
- For each image, the precomputed dense or quasi-dense depth map is reprojected into 3D and logged under

	```text
	/world/agg_depth/<cam_name>
	```

- This is the key flag to inspect the **dense depth representation** used in our method and to verify how well it captures the speed bump geometry compared to raw LiDAR.

> If the sample sequence includes only a subset of aggregated depth files, only those timestamps will display `agg_depth` points.

### 5.4 Height-based colorization

```bash
card-viz <path-to-sequence> --color-by-height
```

- Colors LiDAR and aggregated depth points by **height relative to the road surface**.  
- The road surface is approximated by a **ground plane** fitted through the four wheel ground-contact points:
	- `wheel_FL_Ground`  
	- `wheel_FR_Ground`  
	- `wheel_RL_Ground`  
	- `wheel_RR_Ground`  

This provides an intuitive visualization of subtle elevation changes:

- Points close to the road plane are colored with a plasma-like colormap.  
- Points above the road (for example, the crest of the speed bump) appear in warmer colors.  
- Points well below the plane are strongly de-emphasized (dark colors).

#### Custom height range

```bash
card-viz <path-to-sequence>   --color-by-height   --height-range 0.0 0.5
```

- `--height-range min max` defines the vertical window (in meters) above the ground plane that is mapped to the colormap.  
- By default, the tool uses

	```text
	--height-range 0.0 0.5
	```

	which is particularly suitable to highlight a **0–50 cm** elevation band, ideal for visualizing the speed bump relative to the surrounding road.

### 5.5 Point accumulation (optional)

you may also use (to aggregate the lidar points. However, given that we only deliver short sequence, it is not important):

```bash
card-viz <path-to-sequence> --accumulate-points
```

- Accumulates LiDAR points over a short sliding time window (e.g. 8 seconds) instead of showing only the current scan.  
- This can be useful to see a more complete local map around the speed bump.

---

## 6. Example Commands

A few concrete example invocations that may be useful:

1. **Default visualization (no additional options)**

	```bash
	card-viz <path-to-sequence>
	```

2. **LiDAR intensity plus aggregated depth**

	```bash
	card-viz <path-to-sequence>      --send-laser-intensities      --send-agg-depth-flag
	```

3. **Height-colored LiDAR and aggregated depth around the speed bump**

	```bash
	card-viz <path-to-sequence>      --send-agg-depth-flag      --color-by-height      --height-range 0.0 0.5
	```

4. **Inspect stereo calibration and compare stereo depth to LiDAR**

	```bash
	card-viz <path-to-sequence>      --stereo-matching      --send-agg-depth-flag
	```

---

## 7. Notes and known limitations

- The included sample sequence is intentionally **short** (a few steps around a speed bump) to conform to submission size limits and the restriction on linking to external dataset hosts.  
- The full CARD dataset has longer sequences (up to 10 minutes), but follow the same directory and file structure.  
