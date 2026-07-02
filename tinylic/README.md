# Lightweight Learned Image Compression

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20ready-ee4c2c.svg)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-Learned%20Image%20Compression-6f42c1.svg)](#)
[![Metrics](https://img.shields.io/badge/Metrics-PSNR%20%7C%20BPP%20%7C%20MS--SSIM-2ea44f.svg)](#)

This repository provides lightweight learned image compression models, pretrained checkpoints, and reproducible evaluation utilities for rate-distortion analysis. The released evaluation pipeline reports PSNR, bits per pixel (BPP), and MS-SSIM, and includes scripts for generating RD curves and VTM-anchor BD-rate results.

## Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Single-Checkpoint Evaluation](#single-checkpoint-evaluation)
- [Full RD Evaluation](#full-rd-evaluation)
- [RD Curves and BD-Rate](#rd-curves-and-bd-rate)
- [Complexity Analysis](#complexity-analysis)
- [Implementation Notes](#implementation-notes)
- [Runtime Note](#runtime-note)

## Repository Structure

```text
ckpts/                 Pretrained checkpoints, four rate points per model
data/                  Dataset loading helpers
datasets/              Example evaluation image sets
eval/                  Optional FID/KID metric utilities
models/                Model definitions
scripts/               Runnable evaluation entry points
utils/                 Plotting and complexity-analysis utilities
```

## Installation

Install PyTorch with the CUDA version appropriate for your system, then install the pinned dependencies:

```bash
pip install -r requirements.txt
```

If the default PyPI PyTorch wheel does not match your CUDA setup, install the appropriate PyTorch and torchvision wheels from the official PyTorch instructions first, then run the same command for the remaining packages.

## Checkpoints

Each architecture provides four rate points:

```text
ckpts/tinylic_10k/{1,2,3,4}.pth.tar
ckpts/tinylic_20k/{1,2,3,4}.pth.tar
ckpts/tinylic_50k/{1,2,3,4}.pth.tar
```

Rate point `1` is the lowest-rate checkpoint, and rate point `4` is the highest-rate checkpoint.

## Single-Checkpoint Evaluation

Evaluate a single checkpoint on an image directory:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/test.py \
  --architecture tinylic_50k \
  --dataset /path/to/images \
  --path ckpts/tinylic_50k/4.pth.tar \
  --fuse-repconv
```

Use `--save` to write reconstructed images to `rec/`.

## Full RD Evaluation

Evaluate all three models and all four rate points with fused RepConv branches:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/eval_fuse_all.py
```

The default dataset entries are configured in `scripts/eval_fuse_all.py`. To evaluate selected datasets only, pass their names explicitly:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/eval_fuse_all.py \
  --datasets kodak clic2020
```

The result file is written to:

```text
utils/draw/rd_results.json
```

## RD Curves and BD-Rate

Generate RD plots and VTM-anchor BD-rate summaries:

```bash
PYTHONPATH=. python3 utils/draw_rd.py
```

Outputs are written to `utils/draw/`.

## Complexity Analysis

Complexity is measured after RepConv branch fusion, matching the deployed inference structure:

```bash
PYTHONPATH=. python3 utils/compute_kmacs.py
```

Useful options:

```bash
python3 utils/compute_kmacs.py --models tinylic_50k --input-size 256
python3 utils/compute_kmacs.py --cpu
```

## Runtime Note

The reported runtime is measured with the reference PyTorch implementation and has not been aggressively optimized. Further speedups are expected with engineering optimizations such as custom CUDA kernels, operator fusion, optimized entropy-coding routines, or deployment backends such as TensorRT. Similar acceleration strategies have been explored in some codecs, including DCVC-RT and HPCM.
