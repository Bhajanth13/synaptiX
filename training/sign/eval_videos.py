"""End-to-end accuracy on raw videos: MediaPipe extraction -> ISL model, exactly as at runtime.

    python training/sign/eval_videos.py videos/            # files named <Word>__<anything>.mov
    python training/sign/eval_videos.py videos/ --backend hands
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from synaptix.inputs.sign.isl_model import ISLModel, clean_label, from_synaptix  # noqa: E402
from synaptix.inputs.sign.landmarks import LandmarkExtractor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--backend", default="holistic")
    ap.add_argument("--weights", type=Path)
    args = ap.parse_args()
    m = ISLModel(args.weights) if args.weights else ISLModel()
    vids = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in {".mov", ".mp4", ".webm", ".avi"})
    t1 = t3 = 0
    for v in vids:
        want = clean_label(v.name.split("__")[0].split("_")[0])
        seq = LandmarkExtractor(backend=args.backend).video(str(v), 3000)
        top = m.predict_kps75(from_synaptix(seq), 3)
        t1 += top[0][0] == want
        t3 += want in [g for g, _ in top]
        print(f"{want:14} -> {top[0][0]:14} {top[0][1]:.2f}  {'ok' if top[0][0] == want else 'MISS ' + str([g for g, _ in top])}",
              flush=True)
    print(f"{len(vids)} videos: top-1 {t1}/{len(vids)} ({t1 / len(vids):.0%}), top-3 {t3}/{len(vids)}")


if __name__ == "__main__":
    main()
