# Palette-aware StyleGAN for Few-Shot Flat-Colour Synthesis of Jiangzhou Woodblock Prints

## 1. Title

Palette-aware StyleGAN for Few-Shot Flat-Colour Synthesis of Jiangzhou Woodblock Prints

## 2. Overview

This repository implements the manuscript's unconditional few-shot image-synthesis method for artwork characterised by a limited palette, broad flat-colour regions, and coherent line structure. The implementation is deliberately compact and keeps paper-derived parameters in config.yaml.

## 3. Method components

- A differentiable structural prior with continuous line art, line and flat-fill masks, and distance weights.
- A finite CIELAB(D65) palette prior learned with weighted sampling and weighted K-means.
- An unaugmented U-port for Palette-aware regularisation and an augmented A-port for adversarial learning.
- Independent, parallel appearance and structural discriminator branches with weighted score fusion.
- Reconstruction, palette-proportion, and entropy terms in the Palette-aware objective.
- Coordinated adaptive data augmentation (ADA) and FreezeD capacity control.

## 4. Installation

    python -m pip install -r requirements.txt

Python 3.9 or newer is required. The dependency versions include the PyTorch version reported in the manuscript.

## 5. Private dataset

The dataset is not publicly distributed.

No source images, acquisition records, calibration profiles, private palettes, or trained weights are included.

## 6. Data organisation

Prepare your authorised, standardised 512 x 512 patches as follows:

    private_dataset/
    |-- train/
    |-- val/
    `-- test/

PNG, JPEG, and TIFF are supported. Split source images before extracting patches so that patches from one source image never cross subsets. preprocessing/preprocess.py exposes the source-level split and patch-building interfaces.

## 7. Preprocessing

The preprocessing module provides image loading, optional ICC and white-balance hooks, geometric normalisation, source-level splitting, sliding-window patch extraction, sRGB-to-CIELAB(D65) conversion, and configurable grey-level and saturation screening. Dataset-specific calibration files remain caller supplied.

## 8. Training

    python train.py --config config.yaml --data-root /path/to/private_dataset

The training entry point learns the palette codebook only from the training split, constructs the U-port and A-port, and implements discriminator-first updates, Palette-aware feedback, lazy R1 and path-length regularisation, ADA, FreezeD, EMA validation, checkpointing, and resume. It does not download data or pretrained weights.

## 9. Generation

    python generate.py --checkpoint /path/to/model.pt --num-images 16 --seed 42 --output outputs/

Trained checkpoints from the study are not distributed with this code.

## 10. Evaluation

    python -m evaluation.evaluate --generated outputs/ --reference /path/to/private_dataset/test --checkpoint /path/to/model.pt

The evaluator reports FID and KID through torchmetrics, Palette-KL, flatness violation, closure rate, and ground-truth-normalised line continuity. Human subjective evaluation is described in the manuscript and is not reproduced automatically by this repository.

## 11. Data availability

The Jiangzhou woodblock New Year print dataset used in the study is not publicly distributed due to cultural-heritage and authorization considerations. Data access may be requested from the corresponding author and is subject to the relevant permissions.

## 12. Reproducibility limitation

The code implements the methodology described in the manuscript. The private dataset, dataset-specific calibration files, and trained checkpoints used to obtain the reported numerical results are not distributed.

Reported numerical results require the authorised study data and corresponding training resources.
