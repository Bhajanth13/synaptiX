"""Record keypoint clips for one ISL sign from the webcam.

    python training/sign/record.py LAND --clips 20

Saves training/sign/recordings/<GLOSS>/<n>.npy (keypoints only, no video: privacy by default).
"""

import argparse
from pathlib import Path

import numpy as np

from synaptix.inputs.sign.landmarks import LandmarkExtractor

OUT = Path(__file__).parent / "recordings"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gloss")
    p.add_argument("--clips", type=int, default=20)
    p.add_argument("--frames", type=int, default=45, help="frames per clip (~1.5 s)")
    args = p.parse_args()
    out = OUT / args.gloss.upper()
    out.mkdir(parents=True, exist_ok=True)
    ext = LandmarkExtractor()
    start = len(list(out.glob("*.npy")))
    for i in range(args.clips):
        input(f"[{i + 1}/{args.clips}] Press Enter, then sign {args.gloss.upper()}...")
        np.save(out / f"{start + i}.npy", ext.video(0, max_frames=args.frames))
    print("Saved to", out)


if __name__ == "__main__":
    main()
