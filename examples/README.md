# Sample data

`sample_data.npz` contains 64 simulated, complete 40 x 40 concentration matrices
selected from the TP_S1_ALL collection (https://doi.org/10.5281/zenodo.21498704).
The data remain subject to the license of that record.

| NPZ key | Shape/type | Meaning |
| --- | --- | --- |
| `images` | (64,40,40), float32 | Archived concentrations; interpreted axes: space,time |
| `targets` | (64,3), float32 | First three archived labels, ordered T,L,M_raw |
| `sample_ids` | (64,), Unicode | Original matrix ID, prefixed by collection |

No mass conversion, per-image contrast normalization or augmentation is applied
when packaging the sample. The axis convention follows the ADR residual in
`stipsd/physics.py` and must be verified before using a different archive. Units
are not inferred from raw magnitudes.

`sample_manifest.csv` records original filenames and SHA256 hashes, without local
absolute paths. `data_metadata.json` records selection settings and source scope.
Selection uses seed 42 and excludes exact duplicate float32 image arrays and
samples containing negative concentrations. Rejection counts are saved in the
metadata; selected matrices are not clipped during packaging.

The runner makes a 70/15/15 split with seed 42 (44/9/11 samples after integer
rounding). Reference outputs for this subset are in `../results/`.
