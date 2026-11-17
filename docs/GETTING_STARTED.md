# Getting started, step by step

This guide assumes no experience with Python or deep learning. It takes you from an empty computer to a
segmented 3D stack in about 30 minutes (most of it is downloading). Every command is typed into a terminal
(Linux/macOS: the "Terminal" application; Windows: see the note in step 1).

What the tools do: you give them a 3D image stack of a brain slice (label-free microscopy, oblique illumination
or Dodt gradient contrast), and they return a **label image** of the same size in which every neuronal cell body
(soma) has its own number (1, 2, 3, …) and the background is 0.

## What you need

* A computer with Linux (tested: Ubuntu 22.04). macOS and Windows (through WSL, see step 1) should work, but
  were not tested.
* Optional but recommended: an NVIDIA GPU with at least 8 GB memory and a recent driver. Without a GPU
  everything runs on the CPU, about 10–50 times slower (minutes instead of seconds per stack).
* About 15 GB free disk space for one method (software ~10 GB, weights, one example dataset).

## Step 1. Install conda (once)

conda installs Python and all libraries into separate "environments", so the methods do not interfere with each
other or with other software on your computer. We recommend Miniforge:

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
bash Miniforge3-$(uname)-$(uname -m).sh      # answer "yes" to the questions, then close and reopen the terminal
conda --version                               # should print a version number
```

Windows: install "Ubuntu" from the Microsoft Store (Windows Subsystem for Linux, WSL) and run all commands of this
guide in the Ubuntu terminal.

## Step 2. Download the code (once)

```bash
git clone https://github.com/podtyazhki1337/3d-neurons-segmentation
cd 3d-neurons-segmentation
```

No `git`? Download the ZIP from the GitHub page (green "Code" button → "Download ZIP"), unpack it and `cd` into the
folder. **All following commands are run inside this folder.**

## Step 3. Create the environment of a method (once per method)

Start with **nnU-Net v2**: it was the most accurate method on average in the manuscript.

```bash
conda env create -f environments/nnunet.yml     # 5–15 minutes
conda activate neuroseg-nnunet                  # needed in every new terminal
```

The prompt now starts with `(neuroseg-nnunet)`. Other methods: replace the file and the name according to the
table in the [README](../readme.md#methods), e.g. `environments/cellpose4.yml` / `neuroseg-cellpose4` for
Cellpose-SAM.

## Step 4. Try it on a stack from our dataset

```bash
python -m neuroseg.download_data --out data --datasets rat_oblique
python -m neuroseg.predict --method nnunet --model rat_oblique \
    --input data/rat_neurons_oblique/images/20250811_140233_2.tif --output my_first_result/
```

The first command downloads one part of the dataset from Zenodo (1.3 GB) into `data/`. The second downloads the
trained model from Hugging Face (only the first time) and segments one stack. You should see a line like
`20250811_140233_2.tif: 28 labels, 40 s`. The result is `my_first_result/20250811_140233_2.tif`.

Messages such as `nnUNet_raw is not defined` are harmless: they concern training, not prediction.

## Step 5. Look at the result

**Fiji/ImageJ** (https://fiji.sc): open the raw stack and the result stack (File → Open). Select the result window,
then Image → Lookup Tables → `glasbey_on_dark` to give every soma its own colour; Image → Color → Merge Channels
overlays both.

**napari** (Python viewer), in a separate environment:

```bash
conda create -n napari -c conda-forge napari pyqt tifffile -y
conda activate napari
napari data/rat_neurons_oblique/images/20250811_140233_2.tif my_first_result/20250811_140233_2.tif
```

In napari, right-click the second layer → "Convert to labels" if it is shown as an image.

## Step 6. Segment your own stacks

1. **Format.** Save each stack as a TIFF file with the axis order (Z, Y, X), i.e. a stack of 2D slices (in Fiji:
   File → Save As → Tiff). A multi-channel stack stored as (Z, Y, X, C) is reduced to its first channel.
2. **Choose the model** (`--model`) by imaging modality and, if possible, species:

   | Your images | `--model` |
   |---|---|
   | Dodt gradient contrast (DGC), rat / mouse / human | `rat_dgc` / `mouse_dgc` / `human_dgc`, or `all_dgc` (nnU-Net, all species) |
   | Oblique illumination, rat / mouse / human | `rat_oblique` / `mouse_oblique` / `human_oblique`, or `all_oblique` (nnU-Net) |
   | Unknown species or other tissue | `all_dgc` or `all_oblique` (nnU-Net) |

3. **Pixel size.** The models were trained on 0.115 µm (DGC) and 0.171 µm (oblique) pixels with 1 µm between
   slices. If your pixel size differs a lot (e.g. another objective), rescale the stack to these values first
   (in Fiji: Image → Scale).
4. **Run** on one file or on a whole folder:

   ```bash
   python -m neuroseg.predict --method nnunet --model all_dgc --input my_stacks/ --output my_results/
   ```

## Step 7 (optional). Reproduce the results of the manuscript

```bash
python -m neuroseg.download_data --out data                  # all six parts of the dataset, 8.4 GB
python -m neuroseg.reproduce --data data --method nnunet     # about 30 min on a GPU
cat results/nnunet/within/summary.md
```

`summary.md` lists the metrics on the 24 test stacks next to the values published in the manuscript. The same
command works for every method (`--method unet3d`, `stardist3d`, `cellpose3`, `cellpose_sam`, `microsam`; each in
its own environment).

## Problems and solutions

| Problem | Solution |
|---|---|
| `conda: command not found` | Close and reopen the terminal after step 1, or run `~/miniforge3/bin/conda init`. |
| `No module named neuroseg` | Run the commands inside the downloaded `3d-neurons-segmentation` folder (step 2). |
| `No module named torch` / `tensorflow` / `cellpose` | Activate the environment of the method (step 3); each method has its own. |
| `CUDA out of memory` | Close other GPU programs, add `--low-memory` (nnU-Net), or use the CPU with `--device cpu`. |
| Very slow | No GPU was found: check `nvidia-smi`; on the CPU a stack takes minutes. |
| Almost nothing or everything is segmented | Check the axis order (slices must be the first axis), the modality of the model and the pixel size (step 6). |
| No internet on the computer | Download the Hugging Face repository elsewhere and set `NEUROSEG_MODEL_DIR` to its folder. |

## Words used here

* **Instance segmentation / label image**: every object gets its own integer; touching cells get different
  numbers.
* **DGC** (Dodt gradient contrast) and **oblique illumination**: the two label-free microscopy modalities of the
  dataset.
* **AJI, F1, mAP**: accuracy measures computed by `reproduce`; higher is better (100 = perfect).
* **Environment**: a separate installation of Python and libraries created by conda.
