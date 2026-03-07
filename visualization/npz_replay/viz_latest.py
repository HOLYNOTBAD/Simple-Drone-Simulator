# visualization/npz_replay/viz_latest.py
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from visualization.npz_replay.plot_traj import PlotConfig, plot_trajectory
    from visualization.npz_replay.animate_3d import AnimConfig, animate_3d
except ModuleNotFoundError:
    from plot_traj import PlotConfig, plot_trajectory
    from animate_3d import AnimConfig, animate_3d


def find_latest_npz(run_dir: str | Path = "runs") -> Path:
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No .npz files found in {run_dir.resolve()}")
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default=None, help="path to a specific .npz file")
    ap.add_argument("--runs", type=str, default="runs", help="runs directory (default: runs)")
    ap.add_argument("--no-anim", action="store_true", help="disable animation")
    ap.add_argument("--no-plot", action="store_true", help="disable trajectory plot")
    ap.add_argument("--stride", type=int, default=5, help="animation stride")
    ap.add_argument("--no-realtime", action="store_true", help="play animation at fixed frame interval")
    ap.add_argument("--speed", type=float, default=1.0, help="animation speed multiplier in realtime mode")
    args = ap.parse_args()

    if args.path is None:
        npz_path = find_latest_npz(args.runs)
    else:
        npz_path = Path(args.path)

    print(f"Visualizing: {npz_path.resolve()}")

    if not args.no_plot:
        plot_trajectory(npz_path, PlotConfig(show=True))

    if not args.no_anim:
        animate_3d(
            npz_path,
            AnimConfig(
                stride=args.stride,
                show=True,
            ),
        )


if __name__ == "__main__":
    main()

