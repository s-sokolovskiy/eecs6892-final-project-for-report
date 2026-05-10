import argparse
import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.road_network import RoadNetwork

REPO_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default='midtown')
    parser.add_argument("--lat", type=float, default=40.7580)
    parser.add_argument("--lon", type=float, default=-73.9855)
    parser.add_argument("--dist", type=int, default=1200)
    args = parser.parse_args()

    processed_dir = REPO_ROOT / "data" / "processed"
    counts_path = processed_dir / "traffic_counts.pkl"

    print(f"Loading {args.zone} RoadNetwork...")
    sub_rn = RoadNetwork(zone=args.zone, lat=args.lat, lon=args.lon, dist=args.dist)

    if not counts_path.exists():
        raise FileNotFoundError(f"{counts_path} not found. Run build_routes on full Manhattan first.")

    with open(counts_path, "rb") as f:
        manhattan_counts = pickle.load(f)

    print("Loading Manhattan RoadNetwork for edge lookup...")
    manhattan_rn = RoadNetwork(zone="manhattan")

    sub_counts = np.zeros((sub_rn.E, 168), dtype=manhattan_counts.dtype)
    missing = 0
    for edge, idx in sub_rn.edge_to_idx.items():
        man_idx = manhattan_rn.edge_to_idx.get(edge)
        if man_idx is not None:
            sub_counts[idx] = manhattan_counts[man_idx]
        else:
            missing += 1

    if missing:
        print(f"Warning: {missing}/{sub_rn.E} edges had no match in Manhattan counts (set to zero)")

    out_path = processed_dir / f"traffic_counts_{args.zone}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(sub_counts, f)
    print(f"Saved {out_path} with shape {sub_counts.shape}")


if __name__ == "__main__":
    main()
