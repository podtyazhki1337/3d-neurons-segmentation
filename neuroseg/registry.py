"""Released models: where their files are on the Hugging Face Hub and the inference settings used in the manuscript.

Hub repository: https://huggingface.co/Podtyazhki1337/3d-neurons-segmentation
XSoma3D dataset (Zenodo): https://doi.org/10.5281/zenodo.19812777

Dataset names are the archive names on Zenodo. A model is selected by its method and the dataset (or pooled
dataset) it was trained on, e.g. ("nnunet", "rat_neurons_dodt") or ("nnunet", "all_dgc"). Pooled models exist for
nnU-Net v2 and Cellpose-SAM.
"""
from __future__ import annotations

HF_REPO = "Podtyazhki1337/3d-neurons-segmentation"
ZENODO_DOI = "10.5281/zenodo.19812777"  # concept DOI: always resolves to the latest version of the record
ZENODO_RECORD = ZENODO_DOI.rsplit(".", 1)[1]

DATASETS = ["human_neurons_dodt", "mice_neurons_dodt", "rat_neurons_dodt",
            "human_neurons_oblique", "rat_neurons_oblique", "mice_neurons_oblique"]
ALIASES = {  # short names accepted on the command line
    "human_dgc": "human_neurons_dodt", "mouse_dgc": "mice_neurons_dodt", "rat_dgc": "rat_neurons_dodt",
    "human_oblique": "human_neurons_oblique", "rat_oblique": "rat_neurons_oblique", "mouse_oblique": "mice_neurons_oblique",
}
# voxel size (z, y, x) in micrometres; the models were trained on the native grid of each modality
VOXEL_SIZE_UM = {"dodt": (1.0, 0.115, 0.115), "oblique": (1.0, 0.171, 0.171)}

METHODS = ["nnunet", "unet3d", "stardist3d", "cellpose3", "cellpose_sam", "microsam"]

# ---------------------------------------------------------------- nnU-Net v2
NNUNET_ID = {"human_neurons_dodt": "Dataset001_HumanNeuronsDodt", "mice_neurons_dodt": "Dataset002_MiceNeuronsDodt",
             "rat_neurons_dodt": "Dataset003_RatNeuronsDodt", "human_neurons_oblique": "Dataset004_HumanNeuronsOblique",
             "rat_neurons_oblique": "Dataset005_RatNeuronsOblique", "mice_neurons_oblique": "Dataset006_MiceNeuronsOblique",
             # pooled training data (manuscript Tables 3 and 4)
             "rat_mouse_dgc": "Dataset101_XD_RatMice_Dodt", "rat_mouse_oblique": "Dataset102_XD_RatMice_Oblique",
             "all_dgc": "Dataset103_XD_AllSpecies_Dodt", "all_oblique": "Dataset104_XD_AllSpecies_Oblique",
             "all_dgc_oblique": "Dataset105_XD_AllSpecies_Mixed", "human_dgc_oblique": "Dataset108_XM_Human_DodtOblique"}
NNUNET_TRAINER_DIR = "nnUNetTrainer_Comet__nnUNetPlans__3d_fullres"
# post-processing selected on the validation stacks of the training data: threshold, min. volume (voxels),
# min. z-extent (slices); 26-connected components
NNUNET_POSTPROC = {
    "Dataset001_HumanNeuronsDodt": (0.7, 20000, 1), "Dataset002_MiceNeuronsDodt": (0.7, 20000, 1),
    "Dataset003_RatNeuronsDodt": (0.7, 40000, 1), "Dataset004_HumanNeuronsOblique": (0.3, 20000, 1),
    "Dataset005_RatNeuronsOblique": (0.7, 5000, 1), "Dataset006_MiceNeuronsOblique": (0.4, 20000, 1),
    "Dataset101_XD_RatMice_Dodt": (0.6, 20000, 1), "Dataset102_XD_RatMice_Oblique": (0.7, 10000, 1),
    "Dataset103_XD_AllSpecies_Dodt": (0.5, 10000, 4), "Dataset104_XD_AllSpecies_Oblique": (0.3, 10000, 1),
    "Dataset105_XD_AllSpecies_Mixed": (0.6, 20000, 4), "Dataset108_XM_Human_DodtOblique": (0.3, 30000, 1),
}

POOLED = [k for k in NNUNET_ID if k not in DATASETS]  # rat_mouse_dgc, ..., human_dgc_oblique
POOLED_METHODS = ["nnunet", "cellpose_sam"]  # methods with released models of the pooled training sets

# ---------------------------------------------------------------- 3D U-Net (variant: no_aug | light_aug)
# post-processing selected on validation data: threshold, min. volume, min. z-extent; 6-connected components
UNET3D_POSTPROC = {
    "unet3d_human_neurons_dodt_light_aug": (0.7, 30000, 1), "unet3d_human_neurons_dodt_no_aug": (0.5, 40000, 1),
    "unet3d_human_neurons_oblique_light_aug": (0.5, 30000, 1), "unet3d_human_neurons_oblique_no_aug": (0.5, 40000, 1),
    "unet3d_mice_neurons_dodt_light_aug": (0.7, 1, 4), "unet3d_mice_neurons_dodt_no_aug": (0.7, 10000, 1),
    "unet3d_mice_neurons_oblique_light_aug": (0.7, 20000, 1), "unet3d_mice_neurons_oblique_no_aug": (0.4, 20000, 1),
    "unet3d_rat_neurons_dodt_light_aug": (0.7, 500, 6), "unet3d_rat_neurons_dodt_no_aug": (0.6, 2500, 6),
    "unet3d_rat_neurons_oblique_light_aug": (0.7, 2500, 1), "unet3d_rat_neurons_oblique_no_aug": (0.5, 5000, 1),
}

# ---------------------------------------------------------------- StarDist3D (variant: noaug | lightaug)
# Volumes are rescaled to a standard cell size (z, y, x in voxels) before prediction. For new images the mean
# cell size of the training stacks of the model is used.
STARDIST_STD_CELL_SIZE = (4.0, 29.0, 29.0)
STARDIST_PADSIZE = 64
STARDIST_TRAIN_MEAN_CELL_SIZE = {
    "human_neurons_dodt": (14.934, 157.755, 152.328), "mice_neurons_dodt": (9.767, 145.057, 120.235),
    "rat_neurons_dodt": (11.574, 169.38, 132.226), "human_neurons_oblique": (12.561, 119.399, 103.057),
    "rat_neurons_oblique": (9.682, 121.257, 92.29), "mice_neurons_oblique": (12.359, 107.805, 81.738),
}

# ---------------------------------------------------------------- micro-SAM
MICROSAM_SHORT = {d: d.replace("_neurons", "") for d in DATASETS}  # checkpoint folder names: human_dodt, ...

DEFAULT_VARIANT = {"unet3d": "no_aug", "stardist3d": "noaug"}


def resolve_dataset(name: str) -> str:
    name = ALIASES.get(name, name)
    if name in DATASETS or name in NNUNET_ID or name.startswith("Dataset"):
        return name
    raise ValueError(f"unknown dataset/model name {name!r}; use one of {DATASETS + list(ALIASES)} "
                     f"or, for nnU-Net v2 and Cellpose-SAM, {POOLED}")


def model_spec(method: str, trained_on: str, variant: str | None = None) -> dict:
    """Hub sub-folder, files to download and settings of one released model."""
    trained_on = resolve_dataset(trained_on)
    variant = variant or DEFAULT_VARIANT.get(method)
    if method == "nnunet":
        did = trained_on if trained_on.startswith("Dataset") else NNUNET_ID[trained_on]
        sub = f"nnunet/{did}/{NNUNET_TRAINER_DIR}"
        return dict(method=method, name=did, subfolder=sub,
                    files=[f"{sub}/plans.json", f"{sub}/dataset.json", f"{sub}/dataset_fingerprint.json",
                           f"{sub}/fold_0/checkpoint_best.pth"],
                    postproc=NNUNET_POSTPROC[did], connectivity=26)
    if method == "cellpose_sam" and trained_on not in DATASETS:  # pooled training set, hub folder as for nnU-Net v2
        did = trained_on if trained_on.startswith("Dataset") else NNUNET_ID[trained_on]
        return dict(method=method, name=f"cellpose_sam_{did}", subfolder=f"cellpose_sam/{did}",
                    files=[f"cellpose_sam/{did}/cpsam_finetuned"])
    if trained_on not in DATASETS:
        raise ValueError(f"{method}: models exist for the six datasets only, not for {trained_on!r}")
    if method == "unet3d":
        name = f"unet3d_{trained_on}_{variant}"
        return dict(method=method, name=name, subfolder=f"unet3d/{name}", files=[f"unet3d/{name}/best.weights.h5"],
                    postproc=UNET3D_POSTPROC[name], connectivity=6)
    if method == "stardist3d":
        name = f"stardist_{trained_on}_{variant}_final"
        sub = f"stardist3d/{name}"
        return dict(method=method, name=name, subfolder=sub,
                    files=[f"{sub}/config.json", f"{sub}/thresholds.json", f"{sub}/weights_best.h5"],
                    train_mean_cell_size=STARDIST_TRAIN_MEAN_CELL_SIZE[trained_on])
    if method == "cellpose3":
        name = f"{trained_on}_chunked"
        return dict(method=method, name=name, subfolder=f"cellpose/{name}", files=[f"cellpose/{name}/{name}"])
    if method == "cellpose_sam":
        return dict(method=method, name=f"cellpose_sam_{trained_on}", subfolder=f"cellpose_sam/{trained_on}",
                    files=[f"cellpose_sam/{trained_on}/cpsam_finetuned"])
    if method == "microsam":
        if variant == "pretrained":  # zero-shot vit_b_lm, downloaded by micro-sam itself
            return dict(method=method, name="vit_b_lm", subfolder=None, files=[])
        name = f"microsam_vit_b_lm_finetuned_{MICROSAM_SHORT[trained_on]}"
        return dict(method=method, name=name, subfolder=f"microsam/checkpoints/{name}",
                    files=[f"microsam/checkpoints/{name}/best.pt"])
    raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
