"""
predict.py - nnU-Net v2 inference for 3D neuron instance segmentation.

Uses the official nnU-Net predictor (nnUNetPredictor), so preprocessing is exactly
the one used during training: per-volume z-score normalization (ZScoreNormalization
in plans.json), sliding window with Gaussian weighting and mirroring TTA.
Instances are obtained from the foreground probability by thresholding and
26-connected-component labelling with a minimum-volume filter.

Usage:
    python predict.py --model <trained model folder> --input <file.tif | folder> --output <folder>
                      [--checkpoint checkpoint_best.pth] [--no-tta] [--save-prob]
                      [--threshold 0.5] [--min-volume 15] [--device cuda:0] [--zoom-xy 1.0]

<trained model folder> is the folder that holds dataset.json, plans.json and fold_0/,
e.g. nnUNet_results/Dataset001_HumanNeuronsDodt/nnUNetTrainer_Comet__nnUNetPlans__3d_fullres.

Input volumes are TIFF stacks ordered (Z, Y, X). The released models were trained
with unit voxel spacing (1, 1, 1), so no resampling is done inside nnU-Net.
--zoom-xy rescales the lateral axes before prediction (e.g. to match the pixel size
of the training modality); probabilities are resampled back to the input shape.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

DEFAULT_CHECKPOINT = "checkpoint_best.pth"


def load_predictor(model_dir: str, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cuda:0",
                   tta: bool = True, tile_step_size: float = 0.5, verbose: bool = False):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=tta,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=verbose,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(model_dir, use_folds=(0,), checkpoint_name=checkpoint)
    return predictor


def _zoom_xy(vol: torch.Tensor, shape_yx, mode: str) -> torch.Tensor:
    """Resize the lateral axes of a (Z, Y, X) tensor; z is left unchanged."""
    x = vol[:, None]  # (Z, 1, Y, X): each slice is resized in 2D
    if mode == "area":
        x = F.interpolate(x, size=tuple(shape_yx), mode="area")
    else:
        x = F.interpolate(x, size=tuple(shape_yx), mode="bilinear", align_corners=False)
    return x[:, 0]


def predict_foreground_probability(predictor, vol: np.ndarray, zoom_xy: float = 1.0) -> np.ndarray:
    """Foreground probability (float32, same shape as vol) for one (Z, Y, X) volume."""
    vol = np.asarray(vol, dtype=np.float32)
    orig_shape = vol.shape
    if zoom_xy != 1.0:
        t = torch.from_numpy(vol)
        new_yx = [max(1, int(round(s * zoom_xy))) for s in orig_shape[1:]]
        vol = _zoom_xy(t, new_yx, "area" if zoom_xy < 1 else "bilinear").numpy()
    spacing = list(predictor.configuration_manager.spacing)
    _, probs = predictor.predict_single_npy_array(vol[None], {"spacing": spacing}, None, None, True)
    fg = probs[1].astype(np.float32)
    if zoom_xy != 1.0:
        fg = _zoom_xy(torch.from_numpy(fg), orig_shape[1:], "bilinear").numpy()
    return fg


def probability_to_instances(prob: np.ndarray, threshold: float = 0.5, min_volume: int = 15,
                             connectivity: int = 26) -> np.ndarray:
    """Threshold -> connected components -> remove components smaller than min_volume -> relabel.

    Same procedure as binary_to_instances() in eval.py, used for the manuscript.
    """
    import cc3d
    binary = (prob > threshold).astype(np.uint8)
    inst = cc3d.connected_components(binary, connectivity=connectivity)
    if min_volume > 0:
        ids, counts = np.unique(inst, return_counts=True)
        small = ids[(counts < min_volume) & (ids != 0)]
        if len(small):
            inst[np.isin(inst, small)] = 0
        inst = cc3d.connected_components((inst > 0).astype(np.uint8), connectivity=connectivity)
    return inst.astype(np.int32)


def read_volume(path: str) -> np.ndarray:
    v = tifffile.imread(path)
    if v.ndim == 4:
        v = v[..., 0]
    if v.ndim != 3:
        raise ValueError(f"{path}: expected a 3D (Z, Y, X) stack, got shape {v.shape}")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="trained model folder (holds plans.json and fold_0/)")
    ap.add_argument("--input", required=True, help="TIFF stack or folder of TIFF stacks")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--no-tta", action="store_true", help="disable mirroring test-time augmentation")
    ap.add_argument("--save-prob", action="store_true", help="also save foreground probability (uint8, 0-255)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-volume", type=int, default=15, help="minimum instance volume in voxels")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--zoom-xy", type=float, default=1.0, help="lateral rescaling factor before prediction")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "*.tif*"))) if os.path.isdir(args.input) else [args.input]
    os.makedirs(args.output, exist_ok=True)
    predictor = load_predictor(args.model, args.checkpoint, args.device, tta=not args.no_tta)
    for f in files:
        t0 = time.time()
        prob = predict_foreground_probability(predictor, read_volume(f), args.zoom_xy)
        inst = probability_to_instances(prob, args.threshold, args.min_volume)
        name = os.path.basename(f)
        tifffile.imwrite(os.path.join(args.output, f"pred_{name}"), inst.astype(np.uint16), compression="zlib")
        if args.save_prob:
            tifffile.imwrite(os.path.join(args.output, f"prob_{name}"),
                             np.round(prob * 255).astype(np.uint8), compression="zlib")
        print(f"{name}: {inst.max()} instances, {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
