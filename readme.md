# 3D Neuron Segmentation Benchmark

Code for training, inference, and evaluation of deep learning models for 3D neuron segmentation in volumetric microscopy data.

This repository accompanies a research project on benchmarking 3D neuron instance segmentation models across label-free microscopy modalities and biological datasets.

## Description

This repository provides model-specific training and inference pipelines for 3D neuron segmentation in volumetric microscopy images.

Supported model families include:

- 3D U-Net
- nnU-Net v2
- Cellpose
- StarDist3D
- micro-SAM

The repository is organized as a set of self-contained modules. Each model has its own dependencies, configuration files or script-level parameters, and training/inference scripts.

## Data and pretrained weights

This repository contains **code only**.

Datasets and pretrained prediction results are **not included** in this repository. Dataset access and usage will be described in the associated publication.

Pretrained model weights are available on HuggingFace:

https://huggingface.co/Podtyazhki1337/3d-neurons-segmentation

Input data should be prepared separately as 3D microscopy volumes with corresponding instance masks.

The dataset is provided on Zenodo:
https://doi.org/10.5281/zenodo.19812778

## Repository structure

| Module | Purpose | Configuration | Main entry point |
|---|---|---|---|
| `cellpose/` | Cellpose-based chunked training and slice-wise 2D/3D stitching baseline | script-level parameters | `train_chunked.py` |
| `stardist/` | StarDist3D training and inference | `config.py` | `train.py` |
| `micro-sam/` | micro-SAM fine-tuning and inference | `config_microsam.py` | `train_microsam.py` |
| `unet/` | Custom 3D U-Net baseline | `config.py` | `train.py` |
| `nnunet/` | nnU-Net v2 experiments | separate README | see `nnunet/README.md` |
| `eval_scripts/` | Unified evaluation pipeline for 3D instance segmentation, Z-depth error analysis, boundary relaxation tests, and failure-mode aggregation | notebook/script parameters | `benchmark_evaluation.ipynb`, `aggregate_eval.py`, `stats_aggregated.py` |

Each model subfolder contains:

- a `requirements.txt` file;
- a configuration file or script-level parameters;
- scripts for training and/or inference;
- method-specific notes where required.

## Quickstart pattern

Most methods follow the same workflow:

1. Enter the corresponding model folder.

2. Create an environment and install method-specific dependencies:

   `pip install -r requirements.txt`

3. Edit the configuration file or script parameters.

4. Run the training or inference script.

nnU-Net v2 requires additional dataset conversion and environment setup. See `nnunet/README.md` for details.

## Data format

Input data should be prepared as 3D microscopy volumes with corresponding annotation masks.

The expected structure depends on the model-specific pipeline, but in general the code assumes:

- raw volumetric microscopy images;
- corresponding 3D instance masks;
- model predictions saved as volumetric mask files;
- dataset and result paths manually specified in configuration files, scripts, or notebooks.

Dataset paths must be set manually before running training, inference, or evaluation.

## Model training and inference

Each model family has its own folder and environment because the dependencies and execution workflows differ between methods.

### Cellpose

The `cellpose/` folder contains the Cellpose-based baseline.

The main entry point is:

`train_chunked.py`

This script runs the chunked Cellpose training workflow. Parameters are defined directly in the script.

### StarDist3D

The `stardist/` folder contains the StarDist3D training and inference pipeline.

Main files:

- `config.py`
- `train.py`

Edit `config.py` before running training or inference.

### micro-SAM

The `micro-sam/` folder contains the micro-SAM fine-tuning and inference pipeline.

Main files:

- `config_microsam.py`
- `train_microsam.py`

Edit `config_microsam.py` before running the pipeline.

### 3D U-Net

The `unet/` folder contains the custom 3D U-Net baseline.

Main files:

- `config.py`
- `train.py`

Edit `config.py` before running training or inference.

### nnU-Net v2

The `nnunet/` folder contains the nnU-Net v2 setup and experiment scripts.

nnU-Net requires additional dataset conversion, environment variables, and folder structure setup. See:

`nnunet/README.md`

## Prediction generation

After training, each model folder contains model-specific inference/evaluation scripts that generate prediction masks for the corresponding dataset.

Depending on the model, prediction generation is implemented either as Python scripts or Jupyter notebooks. These scripts are specific to each model family because preprocessing, inference, postprocessing, and dependency requirements differ between methods.

The expected workflow is:

1. Train a model using the corresponding model-specific training script.

2. Run the model-specific inference/evaluation script located inside the same model folder.

3. Save generated prediction masks into the corresponding dataset result folder.

4. Once predictions from different models are saved under the dataset result directory, they can be jointly evaluated using the unified scripts in `eval_scripts/`.

In general, the evaluation pipeline expects predictions to be organized under a dataset-specific `results/` directory, for example:

```text
<dataset>/
├── images/
├── masks/
└── results/
    ├── nnunet_model_predictions/
    ├── stardist_model_predictions/
    ├── cellpose_model_predictions/
    ├── microsam_model_predictions/
    └── unet_model_predictions/
```

## Evaluation

The `eval_scripts/` folder contains the unified evaluation pipeline used to compare all model predictions across datasets.

Main files:

- `benchmark_evaluation.ipynb` — runs per-dataset evaluation across all model prediction folders and computes 3D instance-level, pixel-level, Z-slice, boundary, and failure-mode metrics.
- `aggregate_eval.py` — aggregates per-dataset evaluation CSV files into unified summary tables for the Results section.
- `stats_aggregated.py` — aggregates GT-perspective and prediction-perspective failure modes into compact tables.
- `collect_mask_stats.ipynb` — collects dataset-level and object-level statistics from annotation masks.

The evaluation includes:

- object-level F1, precision, and recall at IoU = 0.5;
- mean Average Precision across multiple IoU thresholds;
- Aggregated Jaccard Index;
- pixel-level Dice, IoU, precision, and recall;
- per-volume metrics;
- Z-slice error profiles;
- edge-vs-interior statistical tests;
- boundary relaxation analysis;
- within-object start/middle/end Z-zone analysis;
- GT-side and prediction-side failure-mode classification.

## Evaluation workflow


The expected evaluation workflow is:

1. Train each model using its model-specific training pipeline.

2. Generate prediction masks using the model-specific inference/evaluation scripts located inside each model folder.

3. Save the generated predictions into the corresponding dataset `results/` directory.

4. Open:

   `eval_scripts/benchmark_evaluation.ipynb`

5. Set the dataset, ground-truth, prediction, and output paths inside the notebook.

6. Run the notebook to evaluate all available prediction folders for the selected dataset.

7. Aggregate all dataset-level evaluation files:

   `python eval_scripts/aggregate_eval.py`

8. Generate aggregated failure-mode summary tables:

   `python eval_scripts/stats_aggregated.py

## Output files

The evaluation and aggregation scripts generate CSV files with per-model, per-dataset, and aggregated metrics.

Typical aggregated outputs include:

- `metrics_summary_all.csv`
- `per_volume_metrics_all.csv`
- `zone_test_all.csv`
- `boundary_relaxation_all.csv`
- `edge_interior_mw_all.csv`
- `depth_spearman_all.csv`
- `z_profile_all.csv`
- `failure_modes_gt_all.csv`
- `failure_modes_pred_all.csv`
- `table5_gt_failure_modes.csv`
- `table6_pred_failure_modes.csv`

These files are used for downstream statistical analysis, figure generation, and manuscript tables.

## Notes

- This repository is intended for research reproducibility.
- Dataset paths must be set manually in configuration files, scripts, or notebooks.
- Environment setup may differ between model families because of dependency conflicts.
- Datasets and pretrained prediction results are not included in this repository.
- Dataset access and usage will be described in the associated publication.
- The repository will be updated together with the associated manuscript.

## Status

This repository is under active development and accompanies a manuscript currently in preparation.

## Citation

If you use this code in your work, please cite the associated publication once available.

## Contact

For questions regarding the code, please contact the repository author.
