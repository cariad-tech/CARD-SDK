# Rerun Visualizer

Install the `viz` extra as described in the [SDK README](../README.md), then
open a downloaded sequence:

```bash
card-viz /path/to/sequence --send-agg-depth-flag
```

The viewer displays camera images, LiDAR, trajectory, GPS, and IMU data.
Wheel-excitement plots appear when the sequence contains the corresponding
metadata.

## Options

| Option | Effect |
| --- | --- |
| `--send-agg-depth-flag` | Show precomputed aggregated depth. |
| `--load-dense-depth` | Load precomputed dense depth when available. |
| `--stereo-matching` | Compute stereo disparity and show rectified images. |
| `--send-laser-intensities` | Color LiDAR points by return intensity. |
| `--accumulate-points` | Accumulate LiDAR over an eight-second window. |
| `--color-by-height` | Color points relative to an estimated road plane. |
| `--height-range MIN MAX` | Set the height-color range in meters; default: `0.0 0.5`. |
| `--subsample-cam N` | Display every Nth camera frame; default: `1`. |
| `--time-start SECONDS` | Start at this dataset timestamp. |
| `--time-end SECONDS` | Stop at this dataset timestamp. |

Use `card-viz --help` for the command reference. Dataset files are downloaded
separately. Height coloring is an estimate and depends on available
wheel-contact calibration; it is not a measured road surface.