"""
predict_cpsam.py - 3D inference with a fine-tuned (or the pretrained) Cellpose-SAM model.

Default mode = protocol of the Cellpose 3 baseline in the manuscript: the 2D model is run
slice by slice and masks are stitched across z (do_3D=False, stitch_threshold=0.5, min_size=15).
Optional --do-3d: cellpose 3D mode (2D network on XY, XZ and YZ planes, flows combined, 3D
flow dynamics); --anisotropy = z step / lateral pixel size.

Usage:
    python predict_cpsam.py --model <path or "cpsam"> --input <tif | folder> --output <folder> [--files a.tif,b.tif]
                            [--do-3d --anisotropy 8.7] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import tifffile
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--files", default="", help="comma-separated file names inside --input")
    ap.add_argument("--do-3d", action="store_true")
    ap.add_argument("--anisotropy", type=float, default=None)
    ap.add_argument("--stitch-threshold", type=float, default=0.5)
    ap.add_argument("--min-size", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    from cellpose import models
    if os.path.isdir(a.input):
        files = [os.path.join(a.input, f) for f in a.files.split(",")] if a.files else \
            sorted(glob.glob(os.path.join(a.input, "*.tif*")))
    else:
        files = [a.input]
    os.makedirs(a.output, exist_ok=True)
    model = models.CellposeModel(gpu=True, pretrained_model=a.model)
    timing = []
    for f in files:
        vol = tifffile.imread(f)
        if vol.ndim == 4:
            vol = vol[..., 0]
        torch.cuda.synchronize(); t0 = time.time()
        if a.do_3d:
            masks = model.eval(vol, z_axis=0, do_3D=True, anisotropy=a.anisotropy,
                               min_size=a.min_size, batch_size=a.batch_size)[0]
        else:
            masks = model.eval(vol, z_axis=0, do_3D=False, stitch_threshold=a.stitch_threshold,
                               min_size=a.min_size, batch_size=a.batch_size)[0]
        torch.cuda.synchronize(); dt = time.time() - t0
        name = os.path.basename(f)
        tifffile.imwrite(os.path.join(a.output, f"pred_{name}"), np.asarray(masks).astype(np.uint16), compression="zlib")
        timing.append(dict(file=name, shape=list(vol.shape), sec=round(dt, 2), n_inst=int(np.asarray(masks).max()),
                           gpu=torch.cuda.get_device_name(0), do_3d=a.do_3d))
        print(f"{name}: {int(np.asarray(masks).max())} instances, {dt:.1f} s", flush=True)
    with open(os.path.join(a.output, "timing.jsonl"), "a") as fh:
        for t in timing:
            fh.write(json.dumps(t) + "\n")


if __name__ == "__main__":
    main()
