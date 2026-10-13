# Reference results

Outputs from the 64-sample simulated subset in `examples/`. `core/` contains the
five-method completion run; `extended/` adds the ADR proxy correction.

Reproduce with:

```bash
python run.py --out outputs/check_core
python run.py --with-adr --out outputs/check_extended
```

Only CSV/JSON summaries are archived here. Masks and completed arrays are produced
by the runner.

The Windows/Python environment used is recorded in each `run_manifest.json`.
Different library or hardware versions can produce slightly different numbers.
Tests executed locally include the optional PyTorch tests.
