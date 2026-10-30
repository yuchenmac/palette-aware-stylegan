# Dataset preparation and source identities

Prepare authorised source images with acquisition calibration, sRGB conversion,
orientation and central-panel/content cropping already applied. Then run from the
repository root:

```sh
python -m preprocessing.prepare --source-root /path/to/standardised_sources --output /path/to/patch_dataset --config config.yaml
```

This command splits source images before patch extraction and creates `train/`,
`val/`, `test/`, `lab/` and `manifest.json`. The manifest records each patch's
original `source_id`, split, top/left coordinates and RGB/Lab file paths. Counts
printed by the command describe the supplied sources and retained patches; they
are not fixed to the manuscript's dataset sizes. The output must be new or empty
and outside the source directory. The source identifier is the original file's
path relative to `--source-root`.

The optional YAML configuration supplies the `patch` settings. Command-line
overrides are available for patch size, stride, split ratios, quality thresholds
and seed. Set a quality threshold to `null` in YAML to disable that filter.
Threshold defaults are configurable implementation settings.

PNG, JPEG and TIFF files are supported through Pillow's **8-bit RGB working
representation**. This path does not preserve a high-bit-depth acquisition/ICC
pipeline. High-bit-depth originals must first be converted to standardised sRGB
working images; integer/floating-point grayscale Pillow modes are rejected to
avoid implicit saturation. The lower-level `extract_patches` array interface
accepts uint8 or uint16 RGB and scales by the dtype's full range; floating-point
RGB must already be finite and in `[0, 1]`. Lab files contain float32 CIELAB(D65)
values in their original units.

`WoodblockPatchDataset` automatically loads a root-level `manifest.json` when
present. It rejects repeated source identities across splits and unrecorded
patch files. It returns the original `source_id` and a separate `patch_id`.
Existing `train/`, `val/`, `test/` directories remain supported without a
manifest; in that case `source_id` is an empty string and source separation must
be verified by the caller. Manifest validation checks the recorded identities;
it does not detect distinct source filenames that depict the same print.

The separate ICC and white-balance helper functions accept caller-supplied
calibration data and report when a step is skipped. The preparation command
operates on the resulting standardised source images.
