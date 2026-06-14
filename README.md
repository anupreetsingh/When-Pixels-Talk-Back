# When Pixels Talk Back

> *How watermarks disrupt medical-image-analysis AI.* As standards like [C2PA](https://c2pa.org/) push to embed provenance directly into media — including medical scans — this project asks a downstream question prior work largely ignored: **if you watermark a scan to prove it's authentic, do you also break the AI that reads it?**

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Framework" src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Task" src="https://img.shields.io/badge/Task-Covid--19%20CT%20classification-orange">
  <img alt="Models" src="https://img.shields.io/badge/CNNs-ResNet50%20%C2%B7%20DenseNet169%20%C2%B7%20MobileNetV2-blueviolet">
  <img alt="Tooling" src="https://img.shields.io/badge/Env-uv-success">
</p>

*Course project — CMSC 652, University of Maryland Baltimore County.*
*Authors: **Anupreet Singh**, Sean Moulton, Omkar Kulkarni — Dept. of Computer Science & Electrical Engineering, UMBC.*

---

## Table of Contents

- [Overview](#overview)
- [Why this is interesting](#why-this-is-interesting)
- [How the study works](#how-the-study-works)
- [Models under test](#models-under-test)
- [Watermarking algorithms under test](#watermarking-algorithms-under-test)
- [Key findings](#key-findings)
- [Contribution highlight — RONI blind fragile watermarking](#contribution-highlight--roni-blind-fragile-watermarking-anupreet-singh)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Tech stack](#tech-stack)
- [Limitations & future work](#limitations--future-work)
- [Contributors](#contributors)
- [Paper, slides & poster](#paper-slides--poster)

---

## Overview

Digital watermarking embeds hidden data into an image to prove authorship or detect tampering. That's increasingly being proposed for medical scans — but those same scans are also fed to deep-learning classifiers that diagnose disease. This project measures the collision between the two.

We take a **Covid-19 lung CT scan classification** task, embed **three different watermarking algorithms** into the images, and measure both the **visual fidelity** of each watermark (SSIM / PSNR) and its **effect on classifier performance** (accuracy, precision, recall, F1, confusion matrices) across **three CNNs**. The gap a watermark opens between clean and watermarked accuracy is the "attack."

## Why this is interesting

- **A downstream question prior work skipped.** Watermarking research optimizes for imperceptibility and robustness — not for whether the watermark changes what a diagnostic model *decides*. This project measures exactly that.
- **The failure mode is clinically dangerous.** Degradation isn't uniform noise — it skews toward **false negatives** (missed Covid cases), the worst kind of error in a screening setting.
- **A constructive result, not just an attack.** A region-confined fragile scheme (**RONI**) authenticates the image at **SSIM ≈ 0.9999** with essentially no accuracy loss — showing authentication and reliable AI *can* coexist.
- **Three watermark families, three CNNs, one controlled comparison.** Robust vs. reversible vs. fragile, on architectures from 25M down to 2.2M parameters.

## How the study works

Every result flows from one train-once / evaluate-many loop:

1. **Train** the CNNs (ResNet50 / DenseNet169 / MobileNetV2) on the **original, unaltered** CT scans.
2. **Baseline** — run inference on the clean test set to get reference accuracy / precision / recall / F1.
3. **Watermark** — generate watermarked copies of the test set, one folder per algorithm (Krawtchouk · Guo-Zhuang · RONI) and embedding strength.
4. **Re-evaluate** — run the *same* trained models on each watermarked test set and compare against the baseline.

The gap between step 2 and step 4 is the **"attack"**: the watermark acts as an adversarial perturbation, and the degradation it causes is the headline measurement.

## Models under test

| Model | Family | ~Params | Role |
|-------|--------|---------|------|
| ResNet50 | Residual CNN | 25M | Architecturally distinct baseline |
| DenseNet169 | Densely-connected CNN | 20M | Primary reported model |
| MobileNetV2 | Lightweight inverted-residual CNN | 2.2M | Most resource-constrained / most vulnerable |

All three share a common training/eval harness in [shared_util.py](watermarking-attack/shared_util.py) (224×224 inputs, ImageNet normalization, light augmentation, and a `ModelPerformanceReport` for metrics & confusion matrices).

## Watermarking algorithms under test

| Algorithm | Type | Where it embeds | Behavior |
|-----------|------|-----------------|----------|
| **Krawtchouk moments** | Robust, moment-domain | Whole image, dithered moments | Survives processing but **significantly degrades** classification |
| **Guo-Zhuang** | Reversible, difference-expansion | Smooth regions outside the ROI | Promising — far lower accuracy hit |
| **RONI fragile** | Blind, fragile, spatial-domain | Region-of-Non-Interest (lung-free area) | **Near-lossless**, no meaningful degradation |

## Key findings

- **Fragile watermarking does not break the AI; robust watermarking does.** Robust Krawtchouk-moment embedding drives classification accuracy down (worst case ~48% of positive Covid cases correctly classified), while the fragile RONI scheme leaves performance essentially untouched.
- **Degradation is not random — it skews toward false negatives.** Under watermarking the models increasingly miss *positive* Covid cases. Clinically, that means missed diagnoses, delayed treatment, and higher risk of spread — a far more dangerous failure mode than uniform noise.
- **Accuracy tracks visual fidelity (SSIM).** The more a watermark perturbs the image, the more the classifier suffers — so watermarks for medical AI must be evaluated not only for imperceptibility and robustness, but for their **downstream effect on model decisions**.

## Contribution highlight — RONI blind fragile watermarking (Anupreet Singh)

> The standout contribution of this project: a **1024-bit blind fragile** watermark that authenticates a scan from the watermarked image *alone* — no original required ("blind") — where any single-pixel tampering invalidates it ("fragile") — and that does so **without disturbing the diagnostic region**, so it leaves classifier accuracy intact.

The flow runs start to finish — segment ROI/RONI → compute a SHA-512 fingerprint → scramble the RONI → embed in the LSBs → unscramble — and verification reverses it to confirm authenticity:

1. **Segment ROI vs. RONI.** The lung parenchyma (the diagnostically important *Region of Interest*) is isolated via Otsu thresholding + a 4-quadrant, 4-connected region-growing (BFS) flood fill. Everything outside the lungs becomes the **RONI** — the only place the watermark is allowed to live, so the diagnostic region stays untouched.
2. **Compute the watermark.** Zero the LSBs of every pixel, then take a **SHA-512** hash of the image. The 128-character hex digest expands to its `128 × 8 = 1024`-bit binary representation — a content-derived fingerprint of the scan.
3. **Scramble the RONI pixels.** RONI intensities are shuffled with a secret key, so the watermark's true pixel locations are unknowable to an adversary who lacks the key — even one who can reproduce the segmentation.
4. **Embed in the LSBs.** One watermark bit is written into the LSB of each of the first 1024 scrambled RONI pixels.
5. **Unscramble.** Re-seeding the same key and applying the inverse permutation returns the watermarked pixels to their true spatial positions, producing the final image.

**Verification** re-segments, re-scrambles, reads the 1024 LSBs, recomputes the SHA-512 hash of the (LSB-zeroed) image, and compares: a match means authentic, a mismatch means tampering.

**Result:** near-lossless embedding at **SSIM ≈ 0.9999**, with **no meaningful impact** on DenseNet169 performance (**F1 ≈ 0.952**, matching the unaltered baseline) — demonstrating that content-authentication watermarking *can* coexist with reliable medical-image AI when confined to the RONI.

Implementation: [RONI Watermarking.ipynb](watermarking-attack/RONI%20Watermarking.ipynb) (annotated walkthrough) and [RONI Mass WM Script.ipynb](watermarking-attack/RONI%20Mass%20WM%20Script.ipynb) (batch generation over the dataset).

## Repository layout

```
.
├── Final Paper / Slides / Poster / Proposal (PDFs)
└── watermarking-attack/
    ├── RONI Watermarking.ipynb         # RONI fragile watermark — annotated walkthrough  [Anupreet]
    ├── RONI Mass WM Script.ipynb       # RONI watermark — batch over full dataset         [Anupreet]
    ├── krawtchouk_watermark.py         # Krawtchouk-moment watermark core
    ├── krawtchouk_watermark_all.py     # Apply Krawtchouk over a dataset (CLI)
    ├── krawtchouk_watermark_automated.py  # Sweep multiple embedding strengths
    ├── guo_zhuang/                     # Guo-Zhuang difference-expansion watermark
    │   ├── watermarker.py
    │   ├── gz_watermark_CT.py
    │   └── guo_zhuang.ipynb
    ├── watermark.ipynb                 # Watermark exploration / comparison
    ├── ctscan_resnet50.ipynb           # Train + evaluate ResNet50
    ├── ctscan_densenet169.ipynb        # Train + evaluate DenseNet169
    ├── ctscan_mobilenetv2.ipynb        # Train + evaluate MobileNetV2
    ├── shared_util.py                  # Dataset, transforms, metrics, ModelPerformanceReport
    ├── download_datasets.py            # Fetch datasets from Kaggle
    ├── pyproject.toml / uv.lock        # uv environment
    └── datasets/                       # Downloaded data + train/val/test splits
```

## Getting started

This project uses [**uv**](https://docs.astral.sh/uv/) for dependency management ([install instructions](https://docs.astral.sh/uv/getting-started/installation/)). You may need to restart your terminal after installing; verify with the `uv` command.

**1. Datasets.** The reported results use the Covid-19 lung CT scan dataset; the X-ray and MRI sets were used in exploration.

- **CT scan (primary):** <https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset>
- X-ray: <https://www.kaggle.com/datasets/sachinkumar413/cxr-2-classes>
- MRI: <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

Download from inside `watermarking-attack/` (each run fetches a fresh copy):

```bash
uv run download_datasets.py
```

**2. Run.** From inside `watermarking-attack/`, launch Jupyter Lab:

```bash
uv run --with jupyter jupyter lab
```

Your browser opens Jupyter Lab, where you can run the model notebooks (`ctscan_*.ipynb`), the watermarking scripts/notebooks, and reproduce the comparative analysis.

## Tech stack

| Component | Choice |
| --- | --- |
| Classifiers | ResNet50, DenseNet169, MobileNetV2 (PyTorch / torchvision) |
| Watermarks | Krawtchouk moments, Guo-Zhuang difference-expansion, RONI blind fragile |
| Fidelity metrics | SSIM, PSNR (`scikit-image`) |
| Classifier metrics | accuracy, precision, recall, F1, confusion matrices (`scikit-learn`, `mlxtend`) |
| Crypto / authentication | SHA-512 (RONI fingerprint) |
| Environment | `uv` (Python 3.10+) |

## Limitations & future work

- **Single task & dataset** — results are reported on one Covid-19 CT dataset; broader modalities (X-ray, MRI) were only explored.
- **Train-clean / test-watermarked** — models are never trained on watermarked data, so robustness from watermark-aware training is untested.
- **Key management for RONI** — the fragile scheme depends on a secret scramble key; production use needs a real key-distribution story.
- **Three algorithms** — the watermark space is large; more robust/reversible schemes could refine the fidelity-vs-degradation curve.
- **Ideas:** watermark-aware fine-tuning, per-pixel sensitivity maps, and extending the RONI scheme to other imaging modalities.

## Contributors

- **Anupreet Singh** — designed and implemented the **RONI blind fragile watermarking** scheme end to end (ROI/RONI segmentation, SHA-512 fingerprinting, key-based pixel scrambling, LSB embedding, and verification), including the annotated walkthrough and the batch dataset-generation script; contributed to the comparative classifier evaluation and analysis.
- **Sean Moulton** — contributed to the watermarking algorithms, model training/evaluation pipeline, and analysis.
- **Omkar Kulkarni** — contributed to the watermarking algorithms, model training/evaluation pipeline, and analysis.

## Paper, slides & poster

The full write-up, slides, and poster are in the repository root:

- [Final Paper](Final%20Paper%20-%20Watermarking%20Attack.pdf)
- [Final Presentation Slides](Final%20Presentation%20Slides%20-%20Watermarking%20Attack.pdf)
- [Final Poster](Final%20Poster%20-%20Watermarking%20Attack.pdf)
- [Project Proposal](CMSC_652_Project%20Proposal.pdf)
