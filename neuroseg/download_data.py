"""Download the XSoma3D dataset from Zenodo and unpack it (version used in the manuscript,
https://doi.org/10.5281/zenodo.20797635; all versions: https://doi.org/10.5281/zenodo.19812777).

    python -m neuroseg.download_data --out data                      # all six species-modality sets (~8.4 GB)
    python -m neuroseg.download_data --out data --datasets rat_oblique

Result: data/<dataset>/{images/, masks/, split.json, preproc_meta.json}; split.json holds the train/val/test split
of the manuscript.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
import zipfile

from .registry import DATASETS, ZENODO_RECORD, resolve_dataset

API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
HEADERS = {"User-Agent": "neuroseg-download", "Accept": "application/json"}


def _files():
    with urllib.request.urlopen(urllib.request.Request(API, headers=HEADERS)) as r:
        return {f["key"]: f for f in json.load(r)["files"]}


def download(out: str, datasets=DATASETS, keep_zip: bool = False):
    os.makedirs(out, exist_ok=True)
    files = _files()
    for ds in datasets:
        target = os.path.join(out, ds)
        if os.path.exists(os.path.join(target, "split.json")):
            print(f"{ds}: present, skipped"); continue
        f = files[f"{ds}.zip"]
        zpath = os.path.join(out, f"{ds}.zip")
        if not (os.path.exists(zpath) and os.path.getsize(zpath) == f["size"]):
            print(f"{ds}: downloading {f['size'] / 2**30:.2f} GiB ...", flush=True)
            req = urllib.request.Request(f["links"]["self"], headers={"User-Agent": HEADERS["User-Agent"]})
            with urllib.request.urlopen(req) as r, open(zpath + ".part", "wb") as fh:
                shutil.copyfileobj(r, fh, length=16 * 2**20)
            os.replace(zpath + ".part", zpath)
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            # archives may or may not contain a top-level folder named after the dataset
            prefix = f"{ds}/" if all(n.startswith(f"{ds}/") for n in names) else ""
            z.extractall(out if prefix else target)
        if not keep_zip:
            os.remove(zpath)
        sp = json.load(open(os.path.join(target, "split.json")))
        print(f"{ds}: {len(os.listdir(os.path.join(target, 'images')))} stacks "
              f"(train {len(sp['train'])}, val {len(sp['val'])}, test {len(sp['test'])})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--datasets", nargs="*", default=DATASETS, help="dataset names or aliases (default: all)")
    ap.add_argument("--keep-zip", action="store_true")
    a = ap.parse_args()
    download(a.out, [resolve_dataset(d) for d in a.datasets], a.keep_zip)


if __name__ == "__main__":
    main()
