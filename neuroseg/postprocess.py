"""Instances from a foreground probability: threshold, connected components, removal of small or thin components.

This is the post-processing of nnU-Net v2 and 3D U-Net in the manuscript (settings selected on validation data,
registry.NNUNET_POSTPROC / UNET3D_POSTPROC). The probability is quantised to 8 bits first, as in the evaluated
pipeline, so that the instances are identical to the published predictions.
"""
from __future__ import annotations

import numpy as np


def probability_to_instances(prob: np.ndarray, threshold: float, min_volume: int = 1, min_z: int = 1,
                             connectivity: int = 26) -> np.ndarray:
    import cc3d
    p8 = np.round(np.clip(prob, 0, 1) * 255).astype(np.uint8)
    lab = cc3d.connected_components((p8.astype(np.float32) / 255.0 > threshold).astype(np.uint8),
                                    connectivity=connectivity)
    stats = cc3d.statistics(lab)
    vols = np.asarray(stats["voxel_counts"])
    zext = np.zeros(len(vols), dtype=np.int64)
    for i, bb in enumerate(stats["bounding_boxes"]):
        if i and bb is not None:
            zext[i] = bb[0].stop - bb[0].start
    keep = np.where((vols >= min_volume) & (zext >= min_z))[0]
    keep = keep[keep != 0]
    lab = np.where(np.isin(lab, keep), lab, 0)
    return cc3d.connected_components((lab > 0).astype(np.uint8), connectivity=connectivity).astype(np.int32)
