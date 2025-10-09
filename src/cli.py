# src/card_sdk/cli.py
import argparse
from .rerun_vis import run_viz

def _cmd_viz(args):
    run_viz(
        base_dir=args.dir,
        subsample_cam=args.subsample_cam,
        send_laser_intensities=args.send_laser_intensities,
        stereo_matching=args.stereo_matching,
        load_dense_depth=args.load_dense_depth,
        send_agg_depth=args.send_agg_depth,
        accumulate_points=args.accumulate_points,
    )

def main():
    p = argparse.ArgumentParser("card")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_viz = sub.add_parser("viz", help="Open a Rerun session and visualize a CARD sequence.")
    p_viz.add_argument("--dir", required=True, help="Path to the dataset base directory")
    p_viz.add_argument("--subsample-cam", type=int, default=1)
    p_viz.add_argument("--send-laser-intensities", action="store_true")
    p_viz.add_argument("--stereo-matching", action="store_true")
    p_viz.add_argument("--load-dense-depth", action="store_true")
    p_viz.add_argument("--send-agg-depth", action="store_true")
    p_viz.add_argument("--accumulate-points", action="store_true", help="Accumulate points over time for better visualization.")
    p.add_argument("--color-by-height", action="store_true", help="Color points by height relative to ground plane from wheel contact points")
    p.add_argument("--time-start", type=float, help="Start time in seconds for processing (optional)")
    p.add_argument("--time-end", type=float, help="End time in seconds for processing (optional)")
    p.add_argument("--height-range", nargs=2, type=float, default=[0.0, 0.5], 
                   help="Height range relative to ground plane [min_height max_height] in meters (default: 0.0 0.5)")
    p_viz.set_defaults(func=_cmd_viz)

    args = p.parse_args()
    args.func(args)
if __name__ == "__main__":
    main()