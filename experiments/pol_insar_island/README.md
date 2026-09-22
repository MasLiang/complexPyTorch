# Pol-InSAR-Island FP1/L T3

The first Pol-InSAR benchmark uses the audited FP1/L principal coherency block:
`[T11, T22, T33, T12, T13, T23]`, reconstructed as `complex64 [6, 3616, 2502]`.
Diagonal `Tii` channels have genuine zero imaginary parts by Hermitian
construction; the off-diagonals use their stored real and imaginary planes.

Official train/test label maps are retained. Validation is spatially sampled
from official train blocks only. A 5-pixel guard band excludes a center pixel
whenever its 11x11 receptive field touches an explicitly labelled pixel from
the opposite official split.

The FP protocol sweep uses spatially distributed `16x16` validation tiles,
then holds the resulting split fixed for every later FP/BiReal/LUT comparison.
It compares these phase-preserving representations:

- `raw_rms`: train-only per-channel RMS scaling;
- `trace`: `T_ij / (T11 + T22 + T33 + eps)` at every pixel;
- `log_coherence`: real `log(Tii)` diagonal channels and normalized complex
  off-diagonals `Tij / sqrt(Tii*Tjj)`.

```bash
GPU_ID=5 PYTHON_BIN=/home/jliangbr/miniconda3/envs/lut_net/bin/python \
  ./experiments/pol_insar_island/run_full_flow.sh
```

Run the FP-only representation sweep before continuing the quantized flow:

```bash
GPU_ID=5 PYTHON_BIN=/home/jliangbr/miniconda3/envs/lut_net/bin/python \
  ./experiments/pol_insar_island/run_fp_representation_sweep.sh
```

The initial paper protocol is T3 only. Full upper-triangular T6 (21 complex
channels) remains an explicitly deferred follow-up after this seed-0 flow is
complete and stable.
