"""Segment your own 3D stacks with a released model (weights are downloaded from the Hugging Face Hub).

    python -m neuroseg.predict --method nnunet --model rat_dgc --input stack.tif --output out/
    python -m neuroseg.predict --method cellpose_sam --model human_oblique --input folder_with_tifs/ --output out/

--model: the training data of the model: a dataset (human_dgc, mouse_dgc, rat_dgc, human_oblique, rat_oblique,
mouse_oblique, or the Zenodo names) or, for nnU-Net, a pooled set (rat_mouse_dgc, rat_mouse_oblique, all_dgc,
all_oblique, all_dgc_oblique, human_dgc_oblique). Choose a model trained on the modality of your images; the
models expect (Z, Y, X) stacks at the voxel size of that modality (DGC 1 x 0.115 x 0.115 um, oblique
1 x 0.171 x 0.171 um). Output: <output>/<input name>.tif with one integer label per soma.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from .io import read_volume, write_labels
from .methods import load_model
from .registry import METHODS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--model", required=True, help="training data of the model (see above)")
    ap.add_argument("--variant", default=None,
                    help="unet3d: no_aug (default) | light_aug; stardist3d: noaug (default) | lightaug; "
                         "microsam: pretrained for the zero-shot vit_b_lm model")
    ap.add_argument("--input", required=True, help="a TIFF stack or a folder of TIFF stacks")
    ap.add_argument("--output", required=True)
    ap.add_argument("--low-memory", action="store_true",
                    help="nnU-Net: combine the sliding-window predictions on the CPU (less GPU memory, same result)")
    ap.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: cuda if available)")
    ap.add_argument("--no-tta", action="store_true", help="nnU-Net: switch off mirroring (faster, slightly different)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.input, "*.tif*"))) if os.path.isdir(a.input) else [a.input]
    kw = {"tta": not a.no_tta, "low_memory": a.low_memory} if a.method == "nnunet" else {}
    model = load_model(a.method, a.model, a.variant, a.device, **kw)
    os.makedirs(a.output, exist_ok=True)
    log = []
    for f in files:
        t0 = time.time()
        lab = model.predict(read_volume(f))
        name = os.path.basename(f)
        write_labels(os.path.join(a.output, name), lab)
        log.append(dict(file=name, n_instances=int(len(np.unique(lab)) - 1), seconds=round(time.time() - t0, 1)))
        print(f"{name}: {lab.max()} labels, {time.time() - t0:.1f} s", flush=True)
    json.dump(dict(method=a.method, model=model.spec["name"], variant=a.variant, files=log),
              open(os.path.join(a.output, "neuroseg_predict.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
