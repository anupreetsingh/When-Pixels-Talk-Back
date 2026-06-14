# When Pixels Talk Back: How Watermarks Disrupt Medical Image Analysis AI

A research project (UMBC CMSC-652) investigating how **digital watermarking** of medical images affects the **deep-learning models that analyze them**. As initiatives like [C2PA](https://c2pa.org/) push to embed provenance and authorship data directly into media — including medical scans — we ask a downstream question that prior work largely ignored: *if you watermark a scan to prove it is authentic, do you also break the AI that reads it?*

To answer this we take a Covid-19 lung CT scan classification task, embed three different watermarking algorithms into the images, and measure both the **visual fidelity** of the watermark (SSIM / PSNR) and its **effect on classifier performance** (accuracy, precision, recall, F1, confusion matrices) across three state-of-the-art CNNs.

> **Authors:** Anupreet Singh, Sean Moulton, Omkar Kulkarni — Department of Computer Science and Electrical Engineering, University of Maryland Baltimore County.

The full write-up, slides, and poster are included in the repository root:

- [Final Paper](Final%20Paper%20-%20Watermarking%20Attack.pdf)
- [Final Presentation Slides](Final%20Presentation%20Slides%20-%20Watermarking%20Attack.pdf)
- [Final Poster](Final%20Poster%20-%20Watermarking%20Attack.pdf)
- [Project Proposal](CMSC_652_Project%20Proposal.pdf)

---

## How the study works

The pipeline runs end-to-end in two phases, and every result flows from this loop:

1. **Train** the computer-vision models on the original, unaltered CT scan dataset.
2. **Baseline** — run inference on the unaltered test set to establish reference metrics.
3. **Watermark** — generate watermarked copies of the dataset, one folder per algorithm and embedding strength.
4. **Re-evaluate** — run the *same* trained models on each watermarked test set and compare against the baseline.

The gap between step 2 and step 4 is the "attack": the watermark acts as an adversarial perturbation, and the degradation it causes is the headline measurement.

### Models under test

| Model | Family | ~Params | Role |
|-------|--------|---------|------|
| ResNet50 | Residual CNN | 25M | Architecturally distinct baseline |
| DenseNet169 | Densely-connected CNN | 20M | Primary reported model |
| MobileNetV2 | Lightweight inverted-residual CNN | 2.2M | Most resource-constrained / most vulnerable |

### Watermarking algorithms under test

| Algorithm | Type | Where it embeds | Behavior |
|-----------|------|-----------------|----------|
| **Krawtchouk moments** | Robust, moment-domain | Whole image, dithered moments | Survives processing but **significantly degrades** classification |
| **Guo-Zhuang** | Reversible, difference-expansion | Smooth regions outside the ROI | Promising — far lower accuracy hit |
| **RONI fragile** | Blind, fragile, spatial-domain | Region-of-Non-Interest (lung-free area) | **Near-lossless**, no meaningful degradation |

---

## Key findings

- **Fragile watermarking does not break the AI; robust watermarking does.** Robust Krawtchouk-moment embedding drives classification accuracy down (worst case ~48% of positive Covid cases correctly classified), while the fragile RONI scheme leaves performance essentially untouched.
- **Degradation is not random — it skews toward false negatives.** Under watermarking the models increasingly miss *positive* Covid cases. Clinically, that means missed diagnoses, delayed treatment, and higher risk of spread — a more dangerous failure mode than uniform noise.
- **Accuracy tracks visual fidelity (SSIM).** The more a watermark perturbs the image, the more the classifier suffers — so watermarks for medical AI must be evaluated not only for imperceptibility and robustness, but for their **downstream effect on model decisions**.

---

## Contribution highlight — RONI blind fragile watermarking (Anupreet Singh)

The Region-of-Non-Interest (RONI) watermark is a **1024-bit blind fragile** scheme designed so that authenticity can be verified from the watermarked image alone — no access to the original is required ("blind"), and any single-pixel tampering invalidates it ("fragile").

The flow, start to finish:

1. **Segment ROI vs. RONI.** The lung parenchyma (the diagnostically important *Region of Interest*) is isolated via Otsu thresholding + a 4-quadrant, 4-connected region-growing (BFS) flood fill. Everything outside the lungs becomes the **RONI**, the only place the watermark is allowed to live — so the diagnostic region stays untouched.
2. **Compute the watermark.** Zero the LSBs of every pixel, then take a **SHA-512** hash of the image. The 128-character hex digest is expanded to its `128 × 8 = 1024`-bit binary representation — a content-derived fingerprint of the scan.
3. **Scramble the RONI pixels.** RONI intensities are shuffled with a secret key. This makes the watermark's true pixel locations unknowable to an adversary who lacks the key, even if they can reproduce the segmentation.
4. **Embed in the LSBs.** One watermark bit is written into the LSB of each of the first 1024 scrambled RONI pixels.
5. **Unscramble.** Re-seeding the same key and applying the inverse permutation returns the watermarked pixels to their true spatial positions, producing the final image.

**Verification** re-segments, re-scrambles, reads the 1024 LSBs, recomputes the SHA-512 hash of the (LSB-zeroed) image, and compares: a match means authentic, a mismatch means tampering.

**Result:** near-lossless embedding at **SSIM ≈ 0.9999**, with **no meaningful impact** on DenseNet169 performance (**F1 ≈ 0.952**, matching the unaltered baseline) — demonstrating that content-authentication watermarking *can* coexist with reliable medical-image AI when it is confined to the RONI.

Implementation: [RONI Watermarking.ipynb](watermarking-attack/RONI%20Watermarking.ipynb) (annotated walkthrough) and [RONI Mass WM Script.ipynb](watermarking-attack/RONI%20Mass%20WM%20Script.ipynb) (batch generation over the dataset).

---

## Repository layout

```
.
├── Final Paper / Slides / Poster / Proposal (PDFs)
└── watermarking-attack/
    ├── RONI Watermarking.ipynb        # RONI fragile watermark — annotated walkthrough
    ├── RONI Mass WM Script.ipynb      # RONI watermark — batch over full dataset
    ├── krawtchouk_watermark.py        # Krawtchouk-moment watermark core
    ├── krawtchouk_watermark_all.py    # Apply Krawtchouk over a dataset (CLI)
    ├── krawtchouk_watermark_automated.py  # Sweep multiple embedding strengths
    ├── guo_zhuang/                    # Guo-Zhuang difference-expansion watermark
    │   ├── watermarker.py
    │   └── gz_watermark_CT.py
    ├── ctscan_resnet50.ipynb          # Train + evaluate ResNet50
    ├── ctscan_densenet169.ipynb       # Train + evaluate DenseNet169
    ├── ctscan_mobilenetv2.ipynb       # Train + evaluate MobileNetV2
    ├── shared_util.py                 # Dataset, transforms, metrics, ModelPerformanceReport
    ├── download_datasets.py           # Fetch datasets from Kaggle
    └── datasets/                      # Downloaded data + train/val/test splits
```

---

## Setup

### Installing UV

This project uses [UV](https://docs.astral.sh/uv/) for dependency management. See the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

You may need to restart your terminal after installing. Verify it works by running the `uv` command.

### Datasets

The reported results use the Covid-19 lung CT scan dataset; the X-ray and MRI sets were used in exploration.

- **CT scan (primary):** <https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset>
- X-ray: <https://www.kaggle.com/datasets/sachinkumar413/cxr-2-classes>
- MRI: <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

Download into the project from inside `watermarking-attack/`:

```
uv run download_datasets.py
```

Every run downloads a fresh copy of the dataset.

## Running

From inside `watermarking-attack/`, launch Jupyter Lab:

```
uv run --with jupyter jupyter lab
```

Your browser opens Jupyter Lab where you can run the model notebooks (`ctscan_*.ipynb`), the watermarking scripts/notebooks, and reproduce the comparative analysis.
