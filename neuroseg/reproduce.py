"""Reproduce the test-set results of the manuscript from the Zenodo data and the released weights.

    python -m neuroseg.download_data --out data
    python -m neuroseg.reproduce --data data --method nnunet                        # within-domain, six test sets
    python -m neuroseg.reproduce --data data --method nnunet --experiment transfer  # Tables 3 and 4 (nnU-Net v2)
    python -m neuroseg.reproduce --data data --method cellpose_sam --experiment transfer  # Supplementary Table S12
    python -m neuroseg.reproduce --data data --method stardist3d --variant lightaug --datasets rat_dgc

For every test volume of split.json the released model predicts the instances (same pipeline as the manuscript),
the metrics of the manuscript are computed (neuroseg/metrics.py) and compared with the published values
(neuroseg/reference_results.json). Outputs in --out: predictions (pred_<file>.tif), per-volume metrics (CSV),
and summary.md (mean ± SD over the 4 test volumes next to the manuscript values).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from .io import read_volume, write_labels
from .methods import load_model
from .metrics import METRIC_COLUMNS, evaluate_volume, summarize
from .registry import DATASETS, DEFAULT_VARIANT, METHODS, POOLED_METHODS, resolve_dataset

REF = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_results.json")))
DGC = ["human_neurons_dodt", "rat_neurons_dodt", "mice_neurons_dodt"]
OBL = ["human_neurons_oblique", "rat_neurons_oblique", "mice_neurons_oblique"]
# transfer and pooling experiments (nnU-Net v2: Tables 3 and 4; Cellpose-SAM: Supplementary Table S12):
# experiment -> (training data of the model, test datasets)
TRANSFER = {
    "Dataset107_XM_Oblique2Dodt": ("all_oblique", DGC), "Dataset106_XM_Dodt2Oblique": ("all_dgc", OBL),
    "Dataset101_XD_RatMice_Dodt": ("rat_mouse_dgc", ["human_neurons_dodt"]),
    "Dataset102_XD_RatMice_Oblique": ("rat_mouse_oblique", ["human_neurons_oblique"]),
    "Dataset103_XD_AllSpecies_Dodt": ("all_dgc", DGC), "Dataset104_XD_AllSpecies_Oblique": ("all_oblique", OBL),
    "Dataset105_XD_AllSpecies_Mixed": ("all_dgc_oblique", DGC + OBL),
    "Dataset108_XM_Human_DodtOblique": ("human_dgc_oblique", ["human_neurons_dodt", "human_neurons_oblique"]),
}
SHOW = ["AJI", "Obj_F1@0.5", "mAP_0.5_0.95", "Pixel_Dice"]


def ref_key(method, variant):
    if method in ("unet3d", "stardist3d"):
        return f"{method}:{variant or DEFAULT_VARIANT[method]}"
    if method == "microsam" and variant == "pretrained":
        return "microsam:pretrained"
    return method


def run(model, data, ds, out_dir, method, skip_existing=True):
    split = json.load(open(os.path.join(data, ds, "split.json")))
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for fn in split["test"]:
        gt = read_volume(os.path.join(data, ds, "masks", fn))
        p = os.path.join(out_dir, f"pred_{fn}")
        if skip_existing and os.path.exists(p):
            pred = read_volume(p)
        else:
            t0 = time.time()
            kw = {"annotation": gt} if method == "stardist3d" else {}  # manuscript procedure (methods.StarDist3DModel)
            pred = model.predict(read_volume(os.path.join(data, ds, "images", fn)), **kw)
            write_labels(p, pred)
            print(f"  {ds}/{fn}: {len(np.unique(pred)) - 1} instances, {time.time() - t0:.0f} s", flush=True)
        r = evaluate_volume(gt, pred)
        r["file"] = fn
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with the unpacked Zenodo archives")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--experiment", choices=["within", "transfer"], default="within",
                    help="transfer: transfer and pooling experiments (nnU-Net v2: Tables 3 and 4; Cellpose-SAM: "
                         "Supplementary Table S12)")
    ap.add_argument("--datasets", nargs="*", default=None, help="within-domain: subset of datasets (default: all)")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help=f"transfer: subset of the experiments (default: all): {', '.join(TRANSFER)}")
    ap.add_argument("--out", default="results")
    ap.add_argument("--low-memory", action="store_true",
                    help="nnU-Net: combine the sliding-window predictions on the CPU (less GPU memory, same result)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    tag = a.method + (f"_{a.variant}" if a.variant else "")
    out = os.path.join(a.out, tag, a.experiment)
    os.makedirs(out, exist_ok=True)

    jobs = []  # (label, training data, test dataset, reference values)
    if a.experiment == "within":
        for ds in [resolve_dataset(d) for d in (a.datasets or DATASETS)]:
            jobs.append((ds, ds, ds, REF["within"].get(ref_key(a.method, a.variant), {}).get(ds, {})))
    else:
        if a.method not in POOLED_METHODS:
            raise SystemExit(f"transfer experiments: released models exist for {', '.join(POOLED_METHODS)} only")
        unknown = set(a.experiments or []) - set(TRANSFER)
        if unknown:
            raise SystemExit(f"unknown experiments {sorted(unknown)}; use {list(TRANSFER)}")
        for exp, (train, tests) in TRANSFER.items():
            if a.experiments and exp not in a.experiments:
                continue
            for ds in tests:
                jobs.append((exp, train, ds, REF[f"transfer_{a.method}"].get(exp, {}).get(ds, {})))

    lines = [f"# {tag}, {a.experiment}", "",
             "| Experiment | Test dataset | " + " | ".join(f"{m} (reproduced) | {m} (manuscript)" for m in SHOW) + " |",
             "|---|---|" + "---|---|" * len(SHOW)]
    allrows, model, loaded = [], None, None
    for label, train, ds, ref in jobs:
        if loaded != train:
            model = load_model(a.method, train, a.variant, a.device,
                               **({"low_memory": a.low_memory} if a.method == "nnunet" else {}))
            loaded = train
        print(f"{label}: model {model.spec['name']} on {ds}", flush=True)
        rows = run(model, a.data, ds, os.path.join(out, label, ds), a.method)
        s = summarize(rows)
        allrows += [dict(experiment=label, dataset=ds, **r) for r in rows]
        lines.append(f"| {label} | {ds} | " + " | ".join(
            f"{s[m][0]:.2f} ± {s[m][1]:.2f} | {ref[m]:.2f}" if m in ref else f"{s[m][0]:.2f} ± {s[m][1]:.2f} | –"
            for m in SHOW) + " |")
        print(lines[-1], flush=True)
    with open(os.path.join(out, "metrics_per_volume.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["experiment", "dataset", "file"] + METRIC_COLUMNS, extrasaction="ignore")
        w.writeheader(); w.writerows(allrows)
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out}/summary.md, metrics_per_volume.csv")


if __name__ == "__main__":
    main()
