#!/usr/bin/env python3
"""
Thin launcher for the CARD visualizer.

Example:
  python scripts/vis.py "xx" --send_agg_depth
"""

import os
import sys
import argparse

def _ensure_src_on_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # <root>
    src = os.path.join(repo_root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

def main():
    _ensure_src_on_path()
    try:
        # expects <root>/src/rerun_vis.py to expose run_viz(...)
        from rerun_vis import run_viz
    except Exception as e:
        print("[ERROR] Could not import run_viz from src/rerun_vis.py")
        print("Reason:", repr(e))
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Run CARD visualizer (Rerun).")
    ap.add_argument("dir", help="Path to the dataset base directory")
    ap.add_argument("--subsample_cam", type=int, default=1, help="Visualize every n-th image only")
    ap.add_argument("--send_laser_intensities", action="store_true", help="Send laser intensities if available")
    ap.add_argument("--stereo_matching", action="store_true", help="Compute stereo disparity + 3D points")
    ap.add_argument("--load_dense_depth", action="store_true", help="Stream dense depth .npz point clouds if present")
    ap.add_argument("--send_agg_depth", action="store_true", help="Stream agg_depth .npz point clouds if present")
    ap.add_argument("--accumulate_points", action="store_true", help="Accumulate points over time for better visualization.")
    ap.add_argument("--accumulate-points", action="store_true", help="Accumulate lidar points over time")
    ap.add_argument("--color-by-height", action="store_true", help="Color points by height relative to ground plane from wheel contact points")
    ap.add_argument("--time-start", type=float, help="Start time in seconds for processing (optional)")
    ap.add_argument("--time-end", type=float, help="End time in seconds for processing (optional)")
    ap.add_argument("--height-range", nargs=2, type=float, default=[0.0, 1.0],
                   help="Height range relative to ground plane [min_height max_height] in meters (default: 0.0 0.5)")
    args = ap.parse_args()

    run_viz(
        base_dir=args.dir,
        subsample_cam=args.subsample_cam,
        send_laser_intensities=args.send_laser_intensities,
        stereo_matching=args.stereo_matching,
        load_dense_depth=args.load_dense_depth,
        send_agg_depth_flag=args.send_agg_depth,
        accumulate_points=args.accumulate_points,
        color_by_height=args.color_by_height,
        time_start=args.time_start,
        time_end=args.time_end,
        height_range=tuple(args.height_range),
    )

if __name__ == "__main__":
    main()
