# Methods

## Completion methods

1. `no_completion`: zero-fill missing pixels after training-only normalization.
2. `linear`: spatial interpolation followed by temporal filling; endpoints use
   nearest observed values. Future missing columns are not observations.
3. `dineof_fixed_rank`: iterative rank-6 SVD with observed-pixel restoration.
   No cross-validated rank selection, EOF centering or uncertainty estimation;
   this is a DINEOF-like baseline, not a complete DINEOF implementation.
4. `telea`: OpenCV Telea using a documented internal 8-bit encoding, followed by
   conversion back to [0,1] and exact restoration of observed float values.
5. `ns_only`: OpenCV Navier-Stokes-based image inpainting (an image-processing
   method; it does not model river hydrodynamics).
6. Optional `ns_adr_proxy`: N-S initialization plus a source-free finite-difference
   ADR residual and an observed/missing interface penalty, with illustrative
   pixel-grid coefficients, fixed iterations and no source input.

All methods use the same event, mask and numeric scale. Known pixels are restored
exactly. Both OpenCV methods round normalized intensities to 0-255 internally;
their missing-region outputs therefore have 8-bit quantization. This avoids the
intensity-scale-sensitive Telea updates observed on [0,1] float inputs.
The ADR proxy applies the residual
`dC/dt + u*dC/dx - d*d2C/dx2 + k*C`, with the diffusion sign consistent with the
forward equation. Image scaling has zero offset; a nonzero concentration offset
would require additional reaction/source terms. No held-out source labels enter
completion. Applying a source-free constraint where a source exists can be wrong.

## Regression

`resnet_model.py` is an unmodified copy of the earlier v2 implementation:
BasicBlock [2,2,2], default channels [16,32,64], three outputs. Model input is
`[N,1,space,time]`; output is standardized `[T,L,M_raw]`.

Training uses CPU, batch size 16, Adam at 0.001, Huber loss, and a fixed seed.
The lowest validation-loss epoch is restored. No test sample is used to fit
preprocessing, train the model, or select its checkpoint. All completion methods
share this frozen clean-input model (Strategy B).

## Metrics

Completion SSIM is the scikit-image implementation with `data_range=1.0`, default
window/settings, compared with the same simulated complete field. RMSE is saved
both over the full field and over missing pixels only. The complete reference is
used for scoring, never as an imputation input.

For each source parameter, `APE_pct = 100*abs(pred-true)/abs(true)`; exactly zero
true values give an undefined (blank) APE. Near-zero values may remain unstable.
`range_NAE_pct = 100*abs(pred-true)/(train_max-train_min)` is a different metric;
a zero training range gives an undefined value. Per-event means require all three
components to be defined. Source predictions and true values are retained for audit.
