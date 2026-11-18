# XSoma3D a large scale neuronal dataset revealing cross species generalization in label free microscopy

Code for **XSoma3D**, the dataset and benchmark of 3D neuronal soma segmentation in label-free microscopy (oblique illumination and Dodt
gradient contrast, DGC) of rat, mouse and human brain slices.

| Resource | Link |
|---|---|
| XSoma3D dataset (240 annotated 3D stacks, train/val/test split) | Zenodo, https://doi.org/10.5281/zenodo.19812777 |
| Trained models (all methods and training sets) | Hugging Face, https://huggingface.co/Podtyazhki1337/3d-neurons-segmentation |
| Code | this repository; the release used in the article (v1.0.0) is archived on Zenodo, https://doi.org/10.5281/zenodo.24320772 |

**New to Python or conda? Follow the step-by-step guide [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**
(installation, a first segmentation, viewing the result, your own data, common problems).

The `neuroseg` package connects the three: it downloads the data from Zenodo, loads the released weights from
Hugging Face and runs the inference pipelines of the manuscript (normalisation, network, post-processing), so
that the published test results can be reproduced and the models applied to new stacks without editing
configuration files.

## Methods

| Method | `--method` | Environment file | Training code |
|---|---|---|---|
| nnU-Net v2 | `nnunet` | `environments/nnunet.yml` | `nnUnet/` |
| 3D U-Net | `unet3d` | `environments/tensorflow.yml` | `unet/` |
| StarDist3D | `stardist3d` | `environments/tensorflow.yml` | `stardist/` |
| Cellpose 3 | `cellpose3` | `environments/cellpose3.yml` | `cellpose/` |
| Cellpose-SAM | `cellpose_sam` | `environments/cellpose4.yml` | `cellpose_sam/` |
| micro-SAM | `microsam` | `environments/microsam.yml` | `microsam/` |

The frameworks have conflicting dependencies, so each method has its own conda environment. The `neuroseg`
package itself needs only numpy and tifffile and imports a framework only when one of its models is loaded.

## Installation

```bash
git clone https://github.com/podtyazhki1337/3d-neurons-segmentation
cd 3d-neurons-segmentation
conda env create -f environments/nnunet.yml      # or the environment of another method (table above)
conda activate neuroseg-nnunet
```

Run the commands below from the repository root. All environments were used with CUDA 12 on NVIDIA RTX A5000
GPUs (24 GB); every method also runs on the CPU (`--device cpu`), more slowly.

## Segment your own stacks

```bash
python -m neuroseg.predict --method nnunet --model rat_dgc --input my_stack.tif --output segmented/
python -m neuroseg.predict --method cellpose_sam --model human_oblique --input folder_with_tifs/ --output segmented/
```

or from Python:

```python
from neuroseg import load_model, read_volume, write_labels
model = load_model("nnunet", "rat_dgc")            # downloads the weights from Hugging Face on first use
labels = model.predict(read_volume("my_stack.tif"))  # (Z, Y, X) -> one integer label per soma
write_labels("my_stack_labels.tif", labels)
```

* `--model` is the training data of the model: `human_dgc`, `mouse_dgc`, `rat_dgc`, `human_oblique`,
  `rat_oblique`, `mouse_oblique` (or the Zenodo names, e.g. `rat_neurons_dodt`); for nnU-Net v2 and Cellpose-SAM also the pooled
  models `rat_mouse_dgc`, `rat_mouse_oblique`, `all_dgc`, `all_oblique`, `all_dgc_oblique`, `human_dgc_oblique`.
* Use a model of the same imaging modality as your images. The models were trained on the native voxel grid:
  DGC 1 × 0.115 × 0.115 µm, oblique 1 × 0.171 × 0.171 µm (z, y, x). In the manuscript, the pooled models
  `all_dgc` and `all_oblique` (nnU-Net v2) were the most robust choice within a modality.
* `--variant` selects the training variant where several exist: `unet3d` `no_aug` (default) | `light_aug`,
  `stardist3d` `noaug` (default) | `lightaug`, `microsam` `pretrained` (zero-shot vit_b_lm).
* Input: TIFF stacks ordered (Z, Y, X); output: TIFF label stacks of the same shape and a log
  (`neuroseg_predict.json`).
* `--device cpu` runs without a GPU; `--low-memory` lowers the GPU memory of nnU-Net v2 (same result).

Example with a stack of the dataset (≈1.3 GB download):

```bash
python -m neuroseg.download_data --out data --datasets rat_oblique
python -m neuroseg.predict --method nnunet --model rat_oblique \
    --input data/rat_neurons_oblique/images/20250811_140233_2.tif --output example_output/
```

## Reproduce the results of the manuscript

```bash
python -m neuroseg.download_data --out data                                        # all six sets, ~8.4 GB
python -m neuroseg.reproduce --data data --method nnunet                           # within-domain, 6 test sets
python -m neuroseg.reproduce --data data --method nnunet --experiment transfer     # Tables 3 and 4
python -m neuroseg.reproduce --data data --method cellpose_sam --experiment transfer   # Supplementary Table S12
python -m neuroseg.reproduce --data data --method stardist3d --variant lightaug
```

`reproduce` predicts every test volume of `split.json` with the released model trained on the corresponding
training set, computes the metrics of the manuscript (`neuroseg/metrics.py`: AJI, object F1/precision/recall
at IoU 0.5, mAP over IoU 0.5–0.95, pixel Dice/IoU/precision/recall) and writes `results/<method>/<experiment>/`:

* `<experiment>/<dataset>/pred_<file>.tif` — predicted instances,
* `metrics_per_volume.csv` — all metrics per test volume,
* `summary.md` — mean ± SD over the four test volumes next to the published values
  (`neuroseg/reference_results.json`).

Methods and variants: `nnunet`, `unet3d` (`no_aug`, `light_aug`), `stardist3d` (`noaug`, `lightaug`), `cellpose3`,
`cellpose_sam`, `microsam` (fine-tuned; `pretrained` for the zero-shot model). Inference times on one RTX A5000
are 3–75 s per test volume (Table 2 of the manuscript).

## Released models (Hugging Face)

| Folder on the Hub | Models |
|---|---|
| `nnunet/Dataset00{1-6}_*/` | nnU-Net v2, one per species–modality set |
| `nnunet/Dataset10{1-5,8}_*/` | nnU-Net v2 trained on pooled sets (rodent, all species per modality, all data, human DGC + oblique) |
| `unet3d/unet3d_<dataset>_{no_aug,light_aug}/` | 3D U-Net |
| `stardist3d/stardist_<dataset>_{noaug,lightaug}_final/` | StarDist3D (with the thresholds optimised on validation data) |
| `cellpose/<dataset>_chunked/` | Cellpose 3 |
| `cellpose_sam/<dataset>/` | Cellpose-SAM (fine-tuned) |
| `cellpose_sam/Dataset10{1-5,8}_*/` | Cellpose-SAM trained on the pooled sets of nnU-Net v2 (Supplementary Table S12) |
| `microsam/checkpoints/microsam_vit_b_lm_finetuned_<dataset>/` | micro-SAM (fine-tuned vit_b_lm) |

The inference settings of every model (post-processing thresholds selected on validation data, StarDist3D
scaling) are listed in `neuroseg/registry.py`. To work offline, download the Hub repository and set
`NEUROSEG_MODEL_DIR` to its local copy.

## Dataset

Each Zenodo archive (`<species>_neurons_<modality>.zip`) contains `images/` (raw stacks), `masks/` (instance
annotations), `split.json` (train/validation/test split used for all methods) and `preproc_meta.json`.
`python -m neuroseg.download_data` downloads and unpacks them into `data/<dataset>/`.

## Training

The training code of each method is in its folder; set the dataset path in the configuration to
`data/<dataset>/` (the unpacked Zenodo archive) and run the training script in the environment of the method.

| Method | Configuration | Training |
|---|---|---|
| nnU-Net v2 | `nnUnet/config.py` | `nnUnet/convert_dataset.py` (conversion, unit voxel spacing, the split of `split.json` as fold 0), `nnUnet/merge_datasets.py` (pooled sets), `nnUnet/train.py`; see `nnUnet/README.md` |
| 3D U-Net | `unet/config.py` | `python unet/train.py` |
| StarDist3D | `stardist/config.py` | `python stardist/train.py` |
| Cellpose 3 | parameters at the top of `cellpose/train_chunked.py` | `python cellpose/train_chunked.py` |
| Cellpose-SAM | command-line options | `python cellpose_sam/train_cpsam.py prep --dataset data/<dataset> --cache cache/<dataset>`, then `python cellpose_sam/train_cpsam.py train --cache cache/<dataset> --out models/<dataset>`; pooled sets: several caches separated by commas and `--nimg-per-epoch 1618` (fixed iteration budget of the manuscript) |
| micro-SAM | `microsam/config_microsam.py` | `python microsam/train_microsam.py` |

The training settings of the released models are listed in Supplementary Table S1 of the manuscript.

## Evaluation and analyses of the manuscript

`neuroseg/metrics.py` contains the metric definitions used for all results. The notebooks and scripts in
`eval_scripts/` produced the additional analyses (z-profile and boundary analysis, failure modes):
`benchmark_evaluation.ipynb` (per-dataset evaluation of prediction folders), `aggregate_eval.py`
(aggregation across datasets), `stats_aggregated.py` (failure-mode tables), `collect_mask_stats.ipynb`
(annotation statistics).

## Citation

If you use the data, code or models, please cite the associated publication (reference to be added on
publication), the XSoma3D dataset (https://doi.org/10.5281/zenodo.19812777) and, for the code, its Zenodo archive
(https://doi.org/10.5281/zenodo.24320772).

## License

See `LICENSE`.
