"""Download a few INCLUDE *test-split* videos without fetching the multi-GB zips (HTTP range requests).

    python training/sign/fetch_include_videos.py --splits Train_Test_Split --out videos --per-zip 4 Places_1of4 Jobs_1of2

Videos: INCLUDE dataset, https://zenodo.org/records/4010759 (CC BY 4.0). Used for end-to-end tests.
"""

import argparse
import csv
import http.client
import struct
import urllib.request
import zlib
from pathlib import Path

BASE = "https://zenodo.org/records/4010759/files/{}.zip"


def _get(url, start, end, tries=5):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            data = urllib.request.urlopen(req, timeout=120).read()
            if len(data) == end - start + 1:
                return data
        except (OSError, http.client.HTTPException):
            if attempt == tries - 1:
                raise
    raise OSError(f"short read for bytes {start}-{end}")


def _size(url):
    return int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])


def entries(url):
    """name -> (local header offset, compressed size, method), read from the remote central directory."""
    n = _size(url)
    tail = _get(url, n - 65536, n - 1)
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    j = tail.rfind(b"PK\x06\x06")
    if j != -1:  # zip64
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = _get(url, cd_off, cd_off + cd_size - 1)
    p, out = 0, {}
    while cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nl].decode("utf-8", "replace")
        extra, q = cd[p + 46 + nl:p + 46 + nl + el], 0
        while q < len(extra):
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals = list(struct.unpack("<" + "Q" * (hl // 8), extra[q + 4:q + 4 + hl]))
                if usize == 0xFFFFFFFF:
                    usize = vals.pop(0)
                if csize == 0xFFFFFFFF:
                    csize = vals.pop(0)
                if off == 0xFFFFFFFF:
                    off = vals.pop(0)
            q += 4 + hl
        out[name] = (off, csize, method)
        p += 46 + nl + el + cl
    return out


def extract(url, ent, dest: Path):
    off, csize, method = ent
    nl, el = struct.unpack("<HH", _get(url, off, off + 29)[26:30])
    data = _get(url, off + 30 + nl + el, off + 30 + nl + el + csize - 1)
    dest.write_bytes(zlib.decompress(data, -15) if method == 8 else data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-zip", type=int, default=4)
    ap.add_argument("zips", nargs="+")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    test = {r["FilePath"] for r in csv.DictReader(open(args.splits / "test_include.csv"))}
    for z in args.zips:
        url = BASE.format(z)
        ents = entries(url)
        picked, words = [], set()
        for name in sorted(ents):
            if name in test and name.split("/")[1] not in words:
                words.add(name.split("/")[1])
                picked.append(name)
        step = max(1, len(picked) // args.per_zip)
        for name in picked[::step][: args.per_zip]:
            # file name: <Word>__<video>, e.g. "Court__MVI_3370.MOV"
            dest = args.out / f"{name.split('/')[1].split('. ', 1)[-1].replace(' ', '')}__{Path(name).name}"
            if not dest.exists():
                extract(url, ents[name], dest)
            print(dest, flush=True)


if __name__ == "__main__":
    main()
