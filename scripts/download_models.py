"""Download the default small language model into models/slm.gguf.

    python scripts/download_models.py            # Qwen2.5 1.5B Instruct, 4-bit (~1 GB, Apache-2.0)
    python scripts/download_models.py --url URL  # any other GGUF

Any 1-2B instruct GGUF works (e.g. Sarvam-1, Gemma 3 1B). Pick one that fits your RAM.
"""

import argparse
import urllib.request
from pathlib import Path

DEFAULT = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
OUT = Path(__file__).resolve().parents[1] / "models" / "slm.gguf"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT)
    args = p.parse_args()
    if OUT.exists():
        print("Already present:", OUT)
        return
    tmp = OUT.with_suffix(".part")
    print("Downloading", args.url)
    urllib.request.urlretrieve(args.url, tmp)
    tmp.rename(OUT)
    print("Saved", OUT)


if __name__ == "__main__":
    main()
