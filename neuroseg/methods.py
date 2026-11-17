"""Inference pipelines of the six methods, with the settings used for the manuscript.

    model = load_model(method, trained_on, variant=None, device="cuda")
    labels = model.predict(volume)             # volume: (Z, Y, X) array, labels: int32 instances

Each pipeline runs from the raw stack to instance labels (normalisation, network, post-processing):
  nnunet        official nnU-Net v2 predictor (z-score normalisation of the plans, sliding window with Gaussian
                weighting, mirroring test-time augmentation, checkpoint_best), probability -> post-processing
                selected on validation data (26-connected components)
  unet3d        percentile normalisation (1, 99.8), non-overlapping 32 x 256 x 256 tiles, post-processing selected
                on validation data (6-connected components)
  stardist3d    rescaling to the standard cell size of training, percentile normalisation (1, 99.8),
                predict_instances with the thresholds optimised on validation data, labels mapped back
  cellpose3     2D prediction per slice, stitch_threshold 0.5, min_size 15, diameter of the trained model
  cellpose_sam  as cellpose3 with the fine-tuned Cellpose-SAM model (cellpose 4)
  microsam      automatic instance segmentation (AIS) with the fine-tuned vit_b_lm model, percentile
                normalisation (1, 99.8) to uint8, 3D merging with gap_closing 2 and min_z_extent 2
The frameworks are imported only when a model of that method is loaded (one environment per framework).
"""
from __future__ import annotations

import os

import numpy as np

from . import hub
from .postprocess import probability_to_instances
from .registry import STARDIST_PADSIZE, STARDIST_STD_CELL_SIZE, model_spec


def _percentile_norm(v, lo=1.0, hi=99.8):
    a, b = np.percentile(v, (lo, hi))
    return (v.astype(np.float32) - a) / max(b - a, 1e-6)


class Model:
    def __init__(self, spec, path, device):
        self.spec, self.path, self.device = spec, path, device

    def predict(self, volume: np.ndarray, **kw) -> np.ndarray:
        raise NotImplementedError


class NnUNet(Model):
    def __init__(self, spec, path, device, tta=True, low_memory=False):
        super().__init__(spec, path, device)
        import torch
        import nnunetv2.inference.predict_from_raw_data as P
        orig = P.recursive_find_python_class

        def find(folder, name, current_module):  # the released checkpoints name nnUNetTrainer_Comet, which only adds
            cls = orig(folder, name, current_module)  # experiment logging to nnUNetTrainer (same network)
            if cls is None and name == "nnUNetTrainer_Comet":
                from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
                cls = nnUNetTrainer
            return cls
        P.recursive_find_python_class = find
        self.predictor = P.nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=tta,
                                           perform_everything_on_device=device.startswith("cuda") and not low_memory,
                                           device=torch.device(device), verbose=False, verbose_preprocessing=False,
                                           allow_tqdm=False)
        self.predictor.initialize_from_trained_model_folder(path, use_folds=(0,), checkpoint_name="checkpoint_best.pth")

    def probability(self, volume):
        spacing = list(self.predictor.configuration_manager.spacing)
        _, probs = self.predictor.predict_single_npy_array(np.asarray(volume, np.float32)[None], {"spacing": spacing},
                                                           None, None, True)
        return probs[1].astype(np.float32)

    def predict(self, volume, **kw):
        t, mv, mz = self.spec["postproc"]
        return probability_to_instances(self.probability(volume), t, mv, mz, self.spec["connectivity"])


class UNet3D(Model):
    PATCH = (32, 256, 256)

    def __init__(self, spec, path, device):
        super().__init__(spec, path, device)
        import tensorflow as tf
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unet"))
        from model import unet_3d  # network definition of the training code (unet/model.py)
        self.net = unet_3d(input_dim=self.PATCH + (1,), filters=16, number_output_classes=1, use_upsampling=False,
                           dropout_rate=0.20)
        self.net.load_weights(os.path.join(path, "best.weights.h5"))

    def probability(self, volume):
        x = _percentile_norm(volume)
        pz, py, px = self.PATCH
        orig = x.shape
        padded = np.pad(x, [(0, (p - s % p) % p) for s, p in zip(orig, self.PATCH)], mode="reflect")
        gz, gy, gx = (s // p for s, p in zip(padded.shape, self.PATCH))
        idx = [(iz, iy, ix) for iz in range(gz) for iy in range(gy) for ix in range(gx)]
        tiles = np.stack([padded[iz*pz:(iz+1)*pz, iy*py:(iy+1)*py, ix*px:(ix+1)*px] for iz, iy, ix in idx])[..., None]
        out = self.net.predict(tiles, batch_size=1, verbose=0)
        prob = np.zeros_like(padded)
        for k, (iz, iy, ix) in enumerate(idx):
            prob[iz*pz:(iz+1)*pz, iy*py:(iy+1)*py, ix*px:(ix+1)*px] = out[k, ..., 0]
        return prob[:orig[0], :orig[1], :orig[2]]

    def predict(self, volume, **kw):
        t, mv, mz = self.spec["postproc"]
        return probability_to_instances(self.probability(volume), t, mv, mz, self.spec["connectivity"])


class StarDist3DModel(Model):
    def __init__(self, spec, path, device):
        super().__init__(spec, path, device)
        import tensorflow as tf
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
        from stardist.models import StarDist3D
        self.model = StarDist3D(None, name=os.path.basename(path), basedir=os.path.dirname(path))

    @staticmethod
    def _cell_size(mask):
        from scipy.ndimage import find_objects
        ext = [[s.stop - s.start for s in sl] for sl in find_objects(mask) if sl is not None]
        return np.mean(ext, axis=0)

    def predict(self, volume, annotation=None, **kw):
        """New images: whole volume, scaled with the mean cell size of the training stacks.
        annotation (manuscript evaluation, stardist/pred.ipynb): the stack is cropped to the annotated region plus
        a 64-voxel margin and scaled with the mean annotated cell size of the stack."""
        from csbdeep.utils import normalize
        from skimage.transform import rescale
        std = np.array(STARDIST_STD_CELL_SIZE)
        if annotation is None:
            sl = tuple(slice(0, s) for s in volume.shape)
            factor = std / np.array(self.spec["train_mean_cell_size"])
        else:
            sl = []
            for ax in range(3):
                nz = np.where(np.any(annotation > 0, axis=tuple(a for a in range(3) if a != ax)))[0]
                sl.append(slice(max(0, nz[0] - STARDIST_PADSIZE), min(volume.shape[ax], nz[-1] + 1 + STARDIST_PADSIZE)))
            sl = tuple(sl)
            factor = std / self._cell_size(annotation)
        crop = volume[sl]
        small = rescale(crop, factor, order=1, anti_aliasing=False, preserve_range=True).astype(np.uint16)
        x = normalize(small.astype(np.float32), 1, 99.8, axis=(0, 1, 2))
        lab, _ = self.model.predict_instances(x, axes="ZYX", n_tiles=self.model._guess_n_tiles(x), show_tile_progress=False)
        back = rescale(lab.astype(float), np.array(crop.shape) / np.array(lab.shape), order=0, anti_aliasing=False,
                       preserve_range=True).astype(np.int32)
        if back.shape != crop.shape:
            tmp = np.zeros(crop.shape, np.int32)
            s2 = tuple(slice(0, min(a, b)) for a, b in zip(back.shape, crop.shape))
            tmp[s2] = back[s2]
            back = tmp
        full = np.zeros(volume.shape, np.int32)
        full[sl] = back
        return full


class CellposeModel(Model):
    def __init__(self, spec, path, device):
        super().__init__(spec, path, device)
        from cellpose import models
        gpu = device.startswith("cuda")
        if spec["method"] == "cellpose3":
            self.model = models.CellposeModel(gpu=gpu, pretrained_model=os.path.join(path, spec["name"]))
            self.diameter = None
            for attr in ("diam_labels", "diam_mean"):
                if getattr(self.model, attr, None) is not None:
                    self.diameter = float(getattr(self.model, attr)); break
        else:
            self.model = models.CellposeModel(gpu=gpu, pretrained_model=os.path.join(path, "cpsam_finetuned"))

    def predict(self, volume, **kw):
        if self.spec["method"] == "cellpose3":
            r = self.model.eval(volume, batch_size=32, channels=[0, 0], diameter=self.diameter, do_3D=False, z_axis=0,
                                stitch_threshold=0.5, min_size=15)
        else:
            r = self.model.eval(volume, z_axis=0, do_3D=False, stitch_threshold=0.5, min_size=15, batch_size=8)
        return np.asarray(r[0]).astype(np.int32)


class MicroSAM(Model):
    def __init__(self, spec, path, device):
        super().__init__(spec, path, device)
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter
        ckpt = os.path.join(path, "best.pt") if path else None
        self.predictor, self.segmenter = get_predictor_and_segmenter(model_type="vit_b_lm", checkpoint=ckpt,
                                                                     device=device, segmentation_mode="ais",
                                                                     is_tiled=False)

    def predict(self, volume, **kw):
        from micro_sam.automatic_segmentation import automatic_instance_segmentation
        x = np.clip(_percentile_norm(volume) * 255, 0, 255).astype(np.uint8)
        return np.asarray(automatic_instance_segmentation(predictor=self.predictor, segmenter=self.segmenter,
                                                          input_path=x, ndim=3, batch_size=1, gap_closing=2,
                                                          min_z_extent=2)).astype(np.int32)


CLASSES = {"nnunet": NnUNet, "unet3d": UNet3D, "stardist3d": StarDist3DModel, "cellpose3": CellposeModel,
           "cellpose_sam": CellposeModel, "microsam": MicroSAM}


def default_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cuda"


def load_model(method: str, trained_on: str, variant: str | None = None, device: str | None = None,
               cache_dir: str | None = None, **kw) -> Model:
    """Released model of `method` trained on `trained_on` (dataset name, alias or pooled set; see registry)."""
    spec = model_spec(method, trained_on, variant)
    path = hub.fetch(spec, cache_dir)
    device = device or default_device()
    if device == "cuda":
        device = "cuda:0"
    return CLASSES[method](spec, path, device, **kw)
