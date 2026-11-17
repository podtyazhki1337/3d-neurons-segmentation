"""
metrics.py - 3D instance segmentation metrics used in the benchmark.

The functions are copied unchanged from benchmark_evaluation.ipynb (cells 6-11),
so scripts can import the exact metric definitions used for the manuscript:

  - AJI (Kumar et al., 2017)
  - object-level F1 / precision / recall / mean matched IoU at IoU = 0.5
    (Hungarian matching)
  - mAP averaged over IoU thresholds 0.50:0.05:0.95 (101-point interpolation)
  - pixel-level Dice / IoU / precision / recall

Mean and SD across volumes are reported with np.std (ddof = 0), as in the notebook.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_THRESH = 0.5
MAP_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

METRIC_COLUMNS = [
    "AJI", f"Obj_F1@{IOU_THRESH}", f"Obj_Prec@{IOU_THRESH}", f"Obj_Rec@{IOU_THRESH}",
    f"Obj_IoU@{IOU_THRESH}", "mAP_0.5_0.95",
    "Pixel_Dice", "Pixel_IoU", "Pixel_Prec", "Pixel_Rec",
]


def iou_matrix(gt, pred):
    gt_ids = np.unique(gt);     gt_ids = gt_ids[gt_ids != 0]
    pred_ids = np.unique(pred); pred_ids = pred_ids[pred_ids != 0]
    ng, np_ = len(gt_ids), len(pred_ids)
    if ng == 0 or np_ == 0:
        return (gt_ids, pred_ids,
                np.zeros((ng, np_), dtype=np.float64),
                np.zeros((ng, np_), dtype=np.int64),
                np.zeros(ng, dtype=np.int64),
                np.zeros(np_, dtype=np.int64))
    gt_rm = np.zeros(int(gt.max()) + 1, dtype=np.int64)
    pr_rm = np.zeros(int(pred.max()) + 1, dtype=np.int64)
    for i, g in enumerate(gt_ids):   gt_rm[g] = i + 1
    for j, p in enumerate(pred_ids): pr_rm[p] = j + 1
    np1 = np.int64(np_ + 1)
    codes = gt_rm[gt.ravel()] * np1 + pr_rm[pred.ravel()]
    tbl = np.bincount(codes, minlength=int((ng + 1) * np1)).reshape(ng + 1, int(np1))
    ov = tbl[1:, 1:].copy()
    gv = tbl[1:, :].sum(1)
    pv = tbl[:, 1:].sum(0)
    un = gv[:, None] + pv[None, :] - ov
    iou = np.zeros_like(ov, dtype=np.float64)
    m = un > 0
    iou[m] = ov[m].astype(np.float64) / un[m].astype(np.float64)
    return gt_ids, pred_ids, iou, ov, gv, pv


def hungarian_match(iou_mat, thresh):
    if iou_mat.size == 0:
        return []
    gt_has_overlap = np.any(iou_mat > 0, axis=1)
    pr_has_overlap = np.any(iou_mat > 0, axis=0)
    gt_idx = np.where(gt_has_overlap)[0]
    pr_idx = np.where(pr_has_overlap)[0]
    if len(gt_idx) == 0 or len(pr_idx) == 0:
        return []
    sub = iou_mat[np.ix_(gt_idx, pr_idx)]
    row_ind, col_ind = linear_sum_assignment(-sub)
    matches = []
    for r, c in zip(row_ind, col_ind):
        iou_val = sub[r, c]
        if iou_val >= thresh:
            matches.append((int(gt_idx[r]), int(pr_idx[c]), float(iou_val)))
    return matches


def compute_aji(gt_ids, pred_ids, iou_mat, overlap, gt_vols, pred_vols):
    ng, np_ = len(gt_ids), len(pred_ids)
    if ng == 0 and np_ == 0: return 1.
    if ng == 0 or np_ == 0:  return 0.
    used = set()
    si, su = 0, 0
    for i in range(ng):
        bj, bv = -1, 0.
        for j in range(np_):
            if j in used: continue
            if iou_mat[i, j] > bv: bv = iou_mat[i, j]; bj = j
        if bj >= 0 and bv > 0:
            inter = int(overlap[i, bj])
            union = int(gt_vols[i]) + int(pred_vols[bj]) - inter
            si += inter; su += union; used.add(bj)
        else:
            su += int(gt_vols[i])
    for j in range(np_):
        if j not in used: su += int(pred_vols[j])
    return si / su if su > 0 else 0.


def compute_ap_single_thresh(iou_mat, n_gt, n_pred, thresh):
    if n_gt == 0 and n_pred == 0: return 1.
    if n_gt == 0: return 0.
    if n_pred == 0: return 0.
    pred_scores = iou_mat.max(axis=0)
    sorted_j = np.argsort(-pred_scores)
    matched_gt = np.zeros(n_gt, dtype=bool)
    tps = np.zeros(n_pred, dtype=np.float64)
    fps = np.zeros(n_pred, dtype=np.float64)
    for rank, j in enumerate(sorted_j):
        best_i = -1
        best_iou = 0.
        for i in range(n_gt):
            if matched_gt[i]:
                continue
            if iou_mat[i, j] > best_iou:
                best_iou = iou_mat[i, j]
                best_i = i
        if best_i >= 0 and best_iou >= thresh:
            tps[rank] = 1
            matched_gt[best_i] = True
        else:
            fps[rank] = 1
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recalls = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum)
    ap = 0.
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max()
    return ap / 101.


def compute_mAP(iou_mat, n_gt, n_pred):
    return float(np.mean([compute_ap_single_thresh(iou_mat, n_gt, n_pred, t)
                          for t in MAP_THRESHOLDS]))


def pixel_metrics(gt, pred):
    g, p = (gt > 0), (pred > 0)
    tp = int(np.sum(g & p)); fp = int(np.sum(~g & p)); fn = int(np.sum(g & ~p))
    pr = tp / (tp + fp) if tp + fp > 0 else 1.
    rc = tp / (tp + fn) if tp + fn > 0 else 1.
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.
    return dict(pixel_dice=dice, pixel_iou=iou, pixel_prec=pr, pixel_rec=rc)


def evaluate_volume(gt, pred):
    """All benchmark metrics for one volume (same as evaluate_volume_cached in the notebook)."""
    gt = np.asarray(gt).astype(np.int32)
    pred = np.asarray(pred).astype(np.int32)
    gt_ids, pred_ids, iou_mat, overlap, gt_vols, pred_vols = iou_matrix(gt, pred)
    ng, np_ = len(gt_ids), len(pred_ids)
    matches = hungarian_match(iou_mat, IOU_THRESH)
    tp = len(matches)
    fp = np_ - tp
    fn = ng - tp
    pr = tp / (tp + fp) if tp + fp > 0 else 0.
    rc = tp / (tp + fn) if tp + fn > 0 else 0.
    f1 = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.
    mi = sum(m[2] for m in matches) / ng if ng > 0 else 0.
    if ng == 0 and np_ == 0:
        pr, rc, f1, mi = 1., 1., 1., 1.
    aji = compute_aji(gt_ids, pred_ids, iou_mat, overlap, gt_vols, pred_vols)
    mAP = compute_mAP(iou_mat, ng, np_)
    pix = pixel_metrics(gt, pred)
    return {
        "n_gt": ng, "n_pred": np_,
        "AJI": aji,
        f"Obj_F1@{IOU_THRESH}": f1,
        f"Obj_Prec@{IOU_THRESH}": pr,
        f"Obj_Rec@{IOU_THRESH}": rc,
        f"Obj_IoU@{IOU_THRESH}": mi,
        "mAP_0.5_0.95": mAP,
        "Pixel_Dice": pix["pixel_dice"],
        "Pixel_IoU": pix["pixel_iou"],
        "Pixel_Prec": pix["pixel_prec"],
        "Pixel_Rec": pix["pixel_rec"],
        "TP_obj": tp, "FP_obj": fp, "FN_obj": fn,
    }


def summarize(rows, columns=METRIC_COLUMNS):
    """Mean and SD (ddof = 0) in percent, as reported in the manuscript."""
    out = {}
    for c in columns:
        v = np.array([r[c] for r in rows], dtype=np.float64) * 100
        out[c] = (float(v.mean()), float(v.std()))
    return out
