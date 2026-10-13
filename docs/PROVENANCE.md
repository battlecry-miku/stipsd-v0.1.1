# Provenance

Package: `stipsd`, version 0.1.1. First assembled on 2026-10-13; see `../CHANGELOG.md`.

| Component | Source | Treatment |
| --- | --- | --- |
| Residual network | `scripts/imputation_ablation_v2/resnet_model.py` | Byte-identical copy; SHA256 in `source_manifest.json` |
| Interpolation/inpainting | `stipsd_repro/case_s_result2_lib.py` | Adapted, with explicit axis and input checks |
| Fixed-rank SVD | `stipsd_repro/train_and_evaluate.py` | Adapted; named fixed-rank rather than full DINEOF |
| ADR proxy | `stipsd_repro/case_s_stipsd.py` | Simplified; weak-curvature term removed |
| Data | `stipsd_repro/data/tp_s1/TP_S1_ALL` | 64 unique simulated image arrays with unchanged first-three-label values |
| Runner, tests and documentation | New for this package | — |

The components come from different earlier workflows. File hashes in
`source_manifest.json` support traceability.

The TP_S1_ALL data archive is https://doi.org/10.5281/zenodo.21498704. This DOI
identifies the data archive, not this software.

The code is released under the MIT License (see `../LICENSE`); the sample data
remain subject to the license of the data record above. Dependencies are installed
from their distributions, not bundled into this archive, and keep their upstream
licenses.
