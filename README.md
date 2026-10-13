# STI-PSD: Spatiotemporal Water-Quality Image Imputation and Pollution Source Detection

This warehouse is used to complete incomplete spatio-temporal images of water quality and to regression the parameters of pollution sources using residual networks.
The warehouse contains 64 simulation samples and five basic completion methods (zero-fill, linear interpolation, fixed rank, Telea, N-S).
Optional ResNet training and ADR agent correction, as well as testing and result export.

"Run"

Open the terminal in this directory. It is recommended to use an independent Python virtual environment. For specific creation methods, please refer to the English homepage.

```bash
python -m pip install -r requirements.txt
python run.py
` ` `

After the completion of the check ` outputs/default/summary. CSV `.
When running again, specify another output directory, such as' python run.py --out outputs/run_2 ', to avoid overwriting the existing results.

When it is necessary to run source parameter prediction and ADR agent correction:

```bash
python -m pip install -r requirements-ml.txt
python run.py --with-adr --train-epochs 2 --out outputs/extended
python -m unittest discover -s tests -v
` ` `

'--train-epochs 2' is used for a quick check process. For formal training, please increase the number of rounds. The model learns with complete training images
Then freeze and predict each completion result of the same test sample (Strategy B).


## Citation and License

Reference information can be found in 'CITATION.cff', and version changes can be found in 'CHANGELOG.md'.
The code is released under the MIT LICENSE (see 'License'). The sample data in 'examples/' is taken from the TP_S1_ALL dataset
(https://doi.org/10.5281/zenodo.21498704), which used the data record.
