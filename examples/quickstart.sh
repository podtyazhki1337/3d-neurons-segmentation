#!/usr/bin/env bash
# Quick start: download one species-modality set from Zenodo (~1.3 GB), segment one of its test stacks with the
# released nnU-Net v2 model (weights from Hugging Face) and reproduce the published test results of that set.
# Run from the repository root in the nnU-Net environment: conda activate neuroseg-nnunet
set -e
python -m neuroseg.download_data --out data --datasets rat_oblique
python -m neuroseg.predict --method nnunet --model rat_oblique \
    --input data/rat_neurons_oblique/images/20250811_140233_2.tif --output example_output/
python -m neuroseg.reproduce --data data --method nnunet --datasets rat_oblique --out results
cat results/nnunet/within/summary.md
