"""
train_cpsam.py - fine-tune Cellpose-SAM (cellpose 4, "cpsam") on 2D slices of the 3D stacks.

Data protocol = Cellpose 3 baseline of the manuscript (cellpose/train_chunked.py):
  - training slices: z-slices of the train split with >= 3 annotated instances;
  - validation slices: same rule on the val split (used for the logged validation loss only);
  - the final model (last epoch) is used, as in the Cellpose 3 baseline.
Optimisation = defaults of the cellpose 4 CLI (`python -m cellpose --train`):
  AdamW, learning rate 1e-5, weight decay 0.1, 100 epochs, batch size 1, 256x256 crops,
  cellpose LR schedule and augmentation (random_rotate_and_resize).

The training loop is the one of cellpose.train.train_seg (v4.0.5) with two changes:
  - slices and flows are read from a memory-mapped cache (uint16 images, float16 flows)
    instead of holding float32 copies of all slices in RAM;
  - precision (--precision, default "amp"): float32 weights and optimizer state, forward pass
    under bfloat16 autocast. cellpose 4.0.5 trains the bfloat16 network directly; with 8-bit
    mantissas an AdamW step of ~1e-5 is rounded away for 87.5 % of the cpsam weights, so most
    of the network would not change. cellpose 4.0.9 reverted training to float32 for this.
    "bf16" reproduces the 4.0.5 behaviour, "fp32" is full float32 (slowest).

Usage:
    python train_cpsam.py prep  --dataset <dir with images/ masks/ split.json> --cache <dir>
    python train_cpsam.py train --cache <dir>[,<dir>...] --out <dir> [--epochs 100] [--lr 1e-5] [--precision amp]
    (several caches: training and validation slices of all of them, e.g. for pooled-species models)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import tifffile
import torch

MIN_INSTANCES = 3


def read_stack(path):
    v = tifffile.imread(path)
    if v.ndim == 4:
        v = v[..., 0]
    return v


def prep(dataset: str, cache: str):
    import fastremap
    from cellpose import dynamics, utils
    os.makedirs(cache, exist_ok=True)
    split = json.load(open(os.path.join(dataset, "split.json")))
    dev = torch.device("cuda")
    for subset in ("train", "val"):
        if os.path.exists(os.path.join(cache, f"{subset}_meta.json")):
            print(f"{subset}: cache exists, skipping")
            continue
        keep = []  # (file, z)
        shape = None
        files = []
        for fn in split[subset]:
            try:
                tifffile.TiffFile(os.path.join(dataset, "images", fn)).close()
                files.append(fn)
            except Exception as ex:  # noqa: BLE001
                print(f"WARNING: skipping unreadable image {fn}: {ex}", flush=True)
        for fn in files:
            m = read_stack(os.path.join(dataset, "masks", fn))
            shape = shape or m.shape[1:]
            assert m.shape[1:] == shape, f"{fn}: slice shape {m.shape[1:]} != {shape}"
            for z in range(m.shape[0]):
                u = np.unique(m[z])
                if (u != 0).sum() >= MIN_INSTANCES:
                    keep.append((fn, z))
        n = len(keep)
        img_dtype = read_stack(os.path.join(dataset, "images", files[0])).dtype
        imgs = np.lib.format.open_memmap(os.path.join(cache, f"{subset}_img.npy"), "w+", img_dtype, (n, *shape))
        flows = np.lib.format.open_memmap(os.path.join(cache, f"{subset}_flows.npy"), "w+", np.float16, (n, 3, *shape))
        diams = np.zeros(n, np.float32)
        i = 0
        for fn in files:
            zs = [z for f, z in keep if f == fn]
            if not zs:
                continue
            im = read_stack(os.path.join(dataset, "images", fn))
            m = read_stack(os.path.join(dataset, "masks", fn)).astype(np.int32)
            labs = [fastremap.renumber(m[z], in_place=False)[0].astype(np.int32) for z in zs]
            fl = dynamics.labels_to_flows(labs, device=dev)
            for z, lab, f in zip(zs, labs, fl):
                imgs[i] = im[z]
                flows[i] = f[1:].astype(np.float16)  # cellprob, flowY, flowX
                diams[i] = max(5.0, utils.diameters(lab)[0])
                i += 1
            print(f"{subset} {fn}: {len(zs)} slices", flush=True)
        imgs.flush(); flows.flush()
        np.save(os.path.join(cache, f"{subset}_diam.npy"), diams)
        json.dump({"n": n, "shape": list(shape), "slices": keep, "min_instances": MIN_INSTANCES},
                  open(os.path.join(cache, f"{subset}_meta.json"), "w"))
        print(f"{subset}: {n} slices cached", flush=True)


class _Concat:
    """Index several memory-mapped arrays as one (slices of different datasets may differ in shape)."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.offsets = np.cumsum([0] + [len(a) for a in arrays])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, i):
        k = int(np.searchsorted(self.offsets, i, side="right")) - 1
        return self.arrays[k][int(i - self.offsets[k])]


def train(cache: str, out: str, n_epochs: int = 100, learning_rate: float = 1e-5,
          weight_decay: float = 0.1, batch_size: int = 1, bsize: int = 256, scale_range: float = 0.5,
          precision: str = "amp", nimg_per_epoch: int | None = None, prefetch_workers: int = 0):
    import contextlib
    from cellpose import models
    from cellpose.train import _loss_fn_seg, _reshape_norm
    from cellpose.transforms import random_rotate_and_resize

    os.makedirs(out, exist_ok=True)
    log = logging.getLogger("cpsam")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(out, "train_log.txt")); fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(fh); log.addHandler(logging.StreamHandler())

    caches = cache.split(",")  # several caches = pooled training data (transfer experiments), read without copying
    tr_img = _Concat([np.load(os.path.join(c, "train_img.npy"), mmap_mode="r") for c in caches])
    tr_fl = _Concat([np.load(os.path.join(c, "train_flows.npy"), mmap_mode="r") for c in caches])
    va_img = _Concat([np.load(os.path.join(c, "val_img.npy"), mmap_mode="r") for c in caches])
    va_fl = _Concat([np.load(os.path.join(c, "val_flows.npy"), mmap_mode="r") for c in caches])
    diam_train = np.concatenate([np.load(os.path.join(c, "train_diam.npy")) for c in caches])
    nimg, nimg_test = len(tr_img), len(va_img)
    nimg_per_epoch = nimg if not nimg_per_epoch else nimg_per_epoch
    log.info(f"caches: {caches}; nimg_per_epoch={nimg_per_epoch}; prefetch_workers={prefetch_workers}")

    model = models.CellposeModel(gpu=True, pretrained_model="cpsam", use_bfloat16=(precision == "bf16"))
    net, device = model.net, model.device
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if precision == "amp" else contextlib.nullcontext
    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)
    normalize_params = dict(models.normalize_default); normalize_params["normalize"] = True

    def batch(img, fl, inds):
        imgs = _reshape_norm([np.asarray(img[i]) for i in inds], normalize_params=normalize_params)
        lbls = [np.asarray(fl[i], dtype=np.float32) for i in inds]
        return imgs, lbls

    # learning-rate schedule of cellpose.train.train_seg
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 99:
        LR = LR[:-50]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    log.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_val={nimg_test}, AdamW lr={learning_rate} wd={weight_decay} "
             f"batch={batch_size} bsize={bsize} weights={net.dtype} precision={precision}")
    filename = os.path.join(out, "cpsam_finetuned")
    t0 = time.time()
    hist = []
    lavg, nsum = 0., 0
    ones = lambda k: np.ones(k, "float32")  # noqa: E731  (rescale=False in cellpose 4 training)
    def prep_train(inds):  # CPU part of an iteration: read, normalise, random rotation/scale/crop
        imgs, lbls = batch(tr_img, tr_fl, inds)
        return random_rotate_and_resize(imgs, Y=lbls, rescale=ones(len(inds)), scale_range=scale_range,
                                        xy=(bsize, bsize))[:2]

    pool = None
    if prefetch_workers > 0:  # overlap the CPU part with the GPU step (same augmentation, threads)
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=prefetch_workers)
    it_total, t_it = 0, time.time()
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        if nimg_per_epoch != nimg:  # cellpose.train.train_seg: random images for the epoch (uniform probabilities)
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,))
        else:
            rperm = np.random.permutation(np.arange(0, nimg))
        for g in optimizer.param_groups:
            g["lr"] = LR[iepoch]
        net.train()
        chunks = [rperm[k:k + batch_size] for k in range(0, nimg_per_epoch, batch_size)]
        if pool is not None:
            ahead = [pool.submit(prep_train, c) for c in chunks[:2 * prefetch_workers]]
        for j, inds in enumerate(chunks):
            if pool is not None:
                imgi, lbl = ahead.pop(0).result()
                if j + len(ahead) + 1 < len(chunks):
                    ahead.append(pool.submit(prep_train, chunks[j + len(ahead) + 1]))
            else:
                imgi, lbl = prep_train(inds)
            it_total += 1
            if it_total % 500 == 0:
                log.info(f"iteration {it_total}, {(time.time() - t_it) / 500:.3f} s/iteration")
                t_it = time.time()
            X = torch.from_numpy(imgi).to(device).to(net.dtype)
            lbl = torch.from_numpy(lbl).to(device).to(net.dtype)
            with autocast():
                y = net(X)[0]
                loss = _loss_fn_seg(lbl, y.to(lbl.dtype), device)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            lavg += loss.item() * len(imgi); nsum += len(imgi)
        if iepoch == 5 or iepoch % 10 == 0 or iepoch == n_epochs - 1:
            lavgt = 0.
            np.random.seed(42)
            rp = np.random.permutation(np.arange(0, nimg_test))
            with torch.no_grad():
                net.eval()
                for i in range(0, len(rp), batch_size):
                    inds = rp[i:i + batch_size]
                    imgs, lbls = batch(va_img, va_fl, inds)
                    imgi, lbl = random_rotate_and_resize(imgs, Y=lbls, rescale=ones(len(inds)),
                                                         scale_range=scale_range, xy=(bsize, bsize))[:2]
                    X = torch.from_numpy(imgi).to(device).to(net.dtype)
                    lbl = torch.from_numpy(lbl).to(device).to(net.dtype)
                    with autocast():
                        lavgt += _loss_fn_seg(lbl, net(X)[0].to(lbl.dtype), device).item() * len(imgi)
            lavgt /= max(len(rp), 1)
            lavg /= max(nsum, 1)
            hist.append(dict(epoch=iepoch, train_loss=lavg, val_loss=lavgt, lr=float(LR[iepoch]),
                             time_s=round(time.time() - t0, 1)))
            log.info(f"{iepoch}, train_loss={lavg:.4f}, val_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.1f}s")
            lavg, nsum = 0., 0
            json.dump(hist, open(os.path.join(out, "history.json"), "w"), indent=1)
    net.save_model(filename)
    json.dump(dict(n_train=int(nimg), n_val=int(nimg_test), nimg_per_epoch=int(nimg_per_epoch), epochs=n_epochs, lr=learning_rate,
                   weight_decay=weight_decay, batch_size=batch_size, bsize=bsize, precision=precision,
                   train_time_s=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0)),
              open(os.path.join(out, "train_info.json"), "w"), indent=1)
    log.info(f"saved {filename}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["prep", "train"])
    ap.add_argument("--dataset")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--precision", choices=["amp", "fp32", "bf16"], default="amp")
    ap.add_argument("--nimg-per-epoch", type=int, default=None,
                    help="images per epoch, sampled at random as in cellpose train_seg (default: all training slices)")
    ap.add_argument("--prefetch-workers", type=int, default=0, help="threads preparing the next batches (0 = none)")
    a = ap.parse_args()
    if a.mode == "prep":
        prep(a.dataset, a.cache)
    else:
        train(a.cache, a.out, a.epochs, a.lr, precision=a.precision, nimg_per_epoch=a.nimg_per_epoch,
              prefetch_workers=a.prefetch_workers)
