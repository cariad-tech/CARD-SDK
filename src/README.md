# Rerun Visualizer

Install the `viz` extra as described in the [SDK README](../README.md), then
open a downloaded sequence:

```bash
card-viz /path/to/sequence --send-agg-depth-flag
```

The viewer displays camera images, LiDAR, trajectory, GPS, and IMU data.
Wheel-excitement plots appear when the sequence contains the corresponding
metadata.

Raw LiDAR is always logged. To add both aggregate depth and stereo results:

```bash
card-viz /path/to/sequence --send-agg-depth-flag --stereo-matching
```

These are separate layers: raw LiDAR, aggregate 3D points and a depth-image
panel, plus stereo points and disparity. `--load-dense-depth` is only for
an additional custom depth folder; it does not enable aggregate depth.
Images outside calibrated pose coverage are skipped with a warning rather
than assigned invented poses. Time-window options apply to displayed telemetry
as well as the camera and LiDAR streams.

## Options

| Option | Effect |
| --- | --- |
| `--send-agg-depth-flag` | Add aggregate 3D points and a metric depth-image panel. |
| `--load-dense-depth` | Load metric depth from `depth/<camera>/*.npz`; warn when absent. |
| `--dense-depth-dir FOLDER` | Select a custom depth-map folder, relative to the sequence or absolute. |
| `--stereo-matching` | Compute stereo disparity and show rectified images. |
| `--send-laser-intensities` | Color LiDAR points by return intensity. |
| `--accumulate-points` | Accumulate LiDAR over an eight-second window. |
| `--color-by-height` | Color points relative to an estimated road plane. |
| `--height-range MIN MAX` | Set the height-color range in meters; default: `0.0 0.5`. |
| `--subsample-cam N` | Display every Nth camera frame; default: `1`. |
| `--time-start SECONDS` | Start at this dataset timestamp. |
| `--time-end SECONDS` | Stop at this dataset timestamp. |
| `--save FILE.rrd` | Write a local recording without spawning a GUI or connecting to a server. |

Use `card-viz --help` for the command reference. Dataset files are downloaded
separately.

## Road-Plane Sources

The dataloader and Rerun fit road planes from wheel ground-contact
trajectories, or fixed ground-contact extrinsics when trajectories are absent.
Germany Batch 1 lacks the fixed ground-contact extrinsics; its road plane
uses the exported contact trajectories instead.

## Custom Depth

Custom depth archives must contain a `depth` array with the original image
dimensions in the camera's pinhole frame and depths in meters, we added this function in case you
have some depth predictions that you want to visualize in our rerun visualizer