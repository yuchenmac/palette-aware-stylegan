# Palette-aware StyleGAN: Craft-Informed Adversarial Synthesis of Jiangzhou Woodblock New Year Prints

This repository implements unconditional synthesis of Jiangzhou woodblock New Year prints under limited-data conditions. Model, training and evaluation settings are defined in `config.yaml`.

## Method

- A differentiable structural prior produces continuous line art, line/flat-fill masks and distance weights.
- A finite CIELAB(D65) palette is learned from the training split using weighted sampling and weighted K-means.
- The unaugmented U-port supplies Palette-aware regularisation; the augmented A-port supplies adversarial learning.
- Parallel appearance and structural discriminator branches use weighted score fusion.
- Palette-aware regularisation combines reconstruction, palette-proportion and entropy terms.
- ADA and FreezeD coordinate augmentation and discriminator capacity.

## Installation

```sh
python -m pip install -r requirements.txt
```

Use Python 3.9 or newer with compatible dependency wheels. The requirements pin PyTorch 2.3.1, as reported in the manuscript. Generator and discriminator weights are initialised randomly. FID/KID validation and evaluation use pretrained Inception features; the feature weights are downloaded on first use if absent from the local cache.

## Data preparation

Prepare authorised source images with calibration, sRGB conversion, orientation and central-panel cropping already applied, then run:

```sh
python -m preprocessing.prepare --source-root /path/to/standardised_sources --output /path/to/patch_dataset --config config.yaml
```

The command splits source images before extracting patches and writes `train/`, `val/`, `test/`, CIELAB arrays under `lab/`, and `manifest.json`. The manifest preserves source identities and patch coordinates; the dataset loader checks that a recorded source does not appear in different splits. Existing directories of prepared patches can also be used. See [data/README.md](data/README.md) for input formats, calibration hooks and manifest handling.

The file-loading path uses 8-bit sRGB working images. The array interface accepts uint8/uint16 RGB, normalised by its dtype range, or finite floating-point RGB in `[0, 1]`. Patch size, stride, source-split ratios and screening thresholds are configurable. The retained sample counts depend on the supplied sources and screening settings.

## Training and loss curves

```sh
python train.py --config config.yaml --data-root /path/to/patch_dataset --checkpoint-dir checkpoints/base
```

The trainer learns the palette from the training split and performs discriminator-first updates, lazy R1/path-length regularisation, adaptive control, generator EMA, validation and checkpointing. Resume with the same configuration and `--resume /path/to/checkpoint.pt`.

Every completed main step records the total losses actually used for optimisation, including active regularisers, in `checkpoints/base/logs/loss-steps-*.csv`. A separate `loss-summary-*.csv` records interval means every 2,000 main steps, including all regulariser-triggering steps within that interval. The final shorter interval is labelled. Progress is recorded as `kimg = step × batch_size × gradient_accumulation / 1000`.

The scalar loss EMA is updated every main step. Its half-life is set explicitly by `training.loss_ema_half_life_steps` (2,000 steps in this configuration), with the first observed loss as its initial value. Generator-weight EMA uses the separate `training.ema_half_life_kimg` setting. Checkpoints preserve loss EMA and pending interval statistics; older checkpoints without this state begin a new loss history at the next update.

Plot recorded interval means and loss EMAs with:

```sh
python plot_training.py checkpoints/base/logs/loss-summary-*.csv --output outputs/training-curves.pdf
```

Use `loss-steps-*.csv` to plot per-step totals instead. Each restart creates a new log segment. Select segments from one continuation; overlapping steps are rejected. The plots use the recorded observations and saved EMA values.

Optional manual feedback for visible banding or block breakup can be supplied through:

```sh
python train.py --config config.yaml --data-root /path/to/patch_dataset --checkpoint-dir checkpoints/base --prior-feedback-file prior-feedback.json
```

The feedback file accepts `{"request_id": "review-001", "prior_too_strong": true}`. A request is applied once at a validation event when valid Palette-KL is decreasing. Use a new `request_id` for another review.

## Generation and objective evaluation

Generate the 1,000 EMA-generator outputs used by the final evaluation protocol:

```sh
python generate.py --checkpoint checkpoints/base/final.pt --num-images 1000 --seed 42 --output outputs/base
python -m evaluation.evaluate --generated outputs/base --reference /path/to/patch_dataset/test --checkpoint checkpoints/base/final.pt --seed 42 --output results/base.json
```

Generation uses the checkpoint's EMA generator when available, otherwise its generator weights. Choose a fresh output directory. `--overwrite` replaces existing `sample-NNNNN.png` outputs and removes surplus samples from a previous generation, while refusing directories containing other images. For a quick visual preview, generate 16 images in a separate directory with `--num-images 16`.

The evaluator uses the checkpoint's stored configuration when present, otherwise `--config`. The supplied configuration requires 1,000 generated images and all 2,400 test reference patches. Generated images are selected in deterministic filename order; all reference images are used. Incomplete counts fail by default. `--allow-partial` enables exploratory evaluation, marks the result accordingly, and excludes it from formal seed aggregation.

The JSON contains FID, `kid_mean`, `kid_subset_std`, Palette-KL, flatness violation, closure rate and reference-normalised line continuity, together with image counts, seeds, configuration and file identities. `--seed` controls evaluation randomness, including KID subset sampling. `kid_subset_std` describes variation among KID subsets within one evaluation. It is distinct from the standard deviation across independent training runs. Undefined metrics are saved as JSON `null`.

All variants use `palette_loss.temperature_end` for final Palette-KL evaluation (0.05 in this configuration). P3 disables entropy and fixes the **training** temperature at `temperature_start` (0.25). Structure metrics retain the two-sweep main-branch calculation; it gives the geodesic diameter for a tree and an estimate for a component containing cycles.

## Repeated runs and ablations

Create configurations for component ablations A–E, Palette-aware sub-term ablations P1–P3 and the configured K sensitivity values:

```sh
python prepare_experiments.py --config config.yaml --output-dir experiments --suite all --seeds 42 43 44
python train.py --config experiments/configs/Base-seed-42.yaml --data-root /path/to/patch_dataset --checkpoint-dir checkpoints/Base-seed-42
```

The three seed values above are examples for new runs. Supply the training seeds for your experiment. The tool preserves the palette-sampling seed and changes only the training seed and each variant's specified factor; it writes configurations and a run manifest without starting training. Select `--suite components`, `palette` or `k` for one study.

Generate and evaluate each trained seed separately using the same reference set and evaluation seed. Then aggregate three results for the same variant:

```sh
python -m evaluation.aggregate_results results/base-seed42.json results/base-seed43.json results/base-seed44.json --output results/base-summary.json
```

The aggregator verifies distinct training seeds/checkpoints and matching configurations and evaluation protocols. It computes the mean and sample standard deviation (`ddof=1`) across per-seed metric values, including `kid_mean`. KID values remain in their original units; multiply both the mean and standard deviation by 1,000 for a table labelled KID × 10⁻³. Human ratings, MOS and Fleiss' κ require the separate rating records described in the manuscript.

## Data and model availability

The Jiangzhou woodblock New Year print dataset is not publicly available due to cultural-heritage and authorization considerations. Data access may be requested from the corresponding author, subject to the relevant permissions. Source images, acquisition calibration files and trained study checkpoints are not included in this repository.

## Tests

Run from the repository root:

```sh
python sanity_check.py
python -m unittest discover -s tests -v
```

The smoke and regression tests use synthetic inputs to check preprocessing, model updates, adaptive control, loss logging and evaluation bookkeeping.
