# Changelog

## 0.1.1 (2026-10-13)

- Fix: the run script and tests imported a non-existent package name and failed
  with `ModuleNotFoundError`; they now import `stipsd`.
- Renamed the run script to `run.py`, the configuration to `configs/default.json`,
  the sample data to `examples/sample_data.npz` and the reference output folder
  to `results/`. The default output directory is `outputs/default/`, and the
  trained weights are saved as `checkpoint.pt`.
- Removed comparison-figure generation; Pillow is no longer a direct dependency.
- `results/` regenerated with this version; `extended/` contains the ADR proxy
  completion run.
- Added the MIT `LICENSE`; the sample data follow the license of
  https://doi.org/10.5281/zenodo.21498704.
- Version number unified to 0.1.1.

## 0.1.0 (2026-10-13)

- Initial package.
