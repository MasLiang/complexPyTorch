# Complex Dataset Acquisition and Audit

Audit date: 2026-09-04. This is an acquisition and format audit only. No FP, BiReal, LUT4, or LUT6 training was launched.

## Decision Summary

The two immediately usable native-complex additions are **Pol-InSAR-Island V2** and **RadioML 2016.10a**. Pol-InSAR-Island is the first recommended 2-D benchmark because it is a public, calibrated complex coherency product with official pixel labels and a natural `[6,H,W] complex64` C3/T3-like input. RadioML 2016.10a is the recommended first cross-domain 1-D sanity check.

The high-value satellite ship candidates need stricter treatment. The downloaded FUSAR-Ship release is verified `uint8` single-band TIFF and is therefore rejected as a primary native-complex benchmark. SARFish does retain SLC, but its full release is 3.293 TB compressed and was deliberately skipped at the current project stage. OpenSARShip's original public archive has now passed file-level audit: its SLC patches preserve two complex polarizations as four float32 planes.

## Table 1 - Dataset Summary

| Dataset | Domain / task | Complex source | Size | Classes | Public? | Recommended? |
|---|---|---|---:|---:|---|---|
| Pol-InSAR-Island V2 | Pol-InSAR land-cover segmentation | DLR F-SAR 6x6 Hermitian T6 | 5.309 GB archive | 12 + unlabeled | Yes, CC BY-NC-SA 4.0 | Yes, first 2-D experiment |
| RadioML 2016.10a | I/Q modulation classification | Raw I/Q frames | 212.7 MB archive; 225.3 MB pickle | 11 | Yes, CC BY-NC-SA 4.0 | Yes, first 1-D experiment |
| RadioML 2018.01A | I/Q modulation classification | Raw I/Q HDF5 | 21.45 GB mirror; official download currently gated | 24 | Source public, acquisition blocked here | Later |
| Sentinel-1 SLC ship-type release | Ship-type classification | Claimed dual-pol SLC chips | Zenodo record contains only 161 KB paper | claimed 3 / optional 4 | Paper public; data absent | Not reproducible now |
| SARFish | Maritime detection / vessel attributes | Sentinel-1 SLC ComplexInt16 | 3.293 TB compressed; 6.468 TB extracted | vessel / fishing / background labels | Yes but gated / very large | Deferred by request |
| OpenSARShip original SLC | Ship-type chip classification | Sentinel-1 dual-pol SLC, four float32 planes | 1.826 GB archive | 16 discovered chip labels; 2,643 SLC chips | Yes | Yes, third 2-D benchmark |
| OpenSARShip HF detection mirror | Ship detection | Sentinel-1 GRD amplitude patches | 72.3 GB | detection | Yes | Rejected: GRD/amplitude |
| FUSAR-Ship 1.0 (GaoFen-3 related) | Ship recognition | Downloaded uint8 TIFF chips | 484 MB archive | 15 top-level categories | Downloadable; no explicit license found | Rejected: no complex samples |
| CSRSDD / related GF-3 SLC | Ship detection | Claimed GF-3 SLC in papers | no verified raw public release | n/a | Not verified | **NOT PUBLICLY REPRODUCIBLE** |

## Table 2 - Actual Downloaded Files and Format Evidence

| Dataset | Local raw / metadata path | Actual format and shape | Imaginary information | Labels |
|---|---|---|---|---|
| Pol-InSAR-Island | `datasets_external/pol_insar_island/raw/pol_insar_island_v2.tar` | Bag archive MD5 `9ba66a216b1ae441a5d4ef91e30a4aa8`; inner ZIP; FP1/L `3616x2502`, FP2 `3616x2540`; ENVI float32 planes | Genuine separate real/imag upper-triangle planes. Diagonal Tii is mathematically real, not synthetic zero-imag data. | `label/{FP1,FP2}/label_{train,test}.bin`, uint8 ENVI maps |
| RadioML 2016.10a | `datasets_external/radioml_2016_10a/raw/RML2016.10a.tar.bz2`; `raw/extracted/RML2016.10a_dict_optimized.pkl` | pickle dict, 220 keys `(mod,SNR)`, each `[1000,2,128] float32` | Actual channel-axis I/Q. Canonical `I + jQ`; sampled imag nonzero fraction 1.0. | modulation and SNR are dictionary keys |
| RadioML 2018.01A | `datasets_external/radioml_2018_01a/metadata/{kaggle_api.json,official_access_response.html}` | No HDF5 obtained. Expected `X:[2555904,1024,2]`, `Y:[2555904,24]`, `Z:[2555904,1]`. | Claimed raw I/Q; not locally verified. | one-hot modulation + SNR |
| Sentinel-1 SLC ship-type | `datasets_external/sentinel1_slc_ship/metadata/zenodo_record.json` | Actual release only has the paper PDF, no chips or labels. | Cannot verify. | Cannot audit class balance or split. |
| SARFish | `datasets_external/sarfish/metadata/{SLC_train.csv,SLC_validation.csv}` | Full raw SLC intentionally not downloaded. Partial sample ZIP is incomplete and must not be treated as usable. | Release documentation specifies Sentinel-1 IW SLC ComplexInt16, VV/VH. | train 64,286 rows / 553 scenes; val 19,186 rows / 50 scenes; labels audited below |
| FUSAR-Ship | `datasets_external/gaofen3_ship/raw/FUSAR_Ship1.0.zip` | Archive SHA-256 `0e985e3cd74c858eca41410699a769c1e9b2e81b16a50442338bdc62f23e412e`; 5,360 files; sample `512x512` TIFF is Pillow mode `L`, `uint8`, range `[0,255]` | **No**: verified single real-valued image, not SLC. | `meta.csv`: 6,358 chips; 3,899 MMSI values; category/path supplied |
| OpenSARShip SLC | `datasets_external/opensarship/raw/OpenSARShip_original_zip.zip`; `audit/sample/` | 85 scene archives: 35 SLC / 50 GRD. 19 SLC scene archives expose 2,643 float32 four-plane TIFF chips. | Verified: `H x W x 4 float32`, paired into two complex polarizations. | `Ship.xml` includes MMSI, AIS type, coordinates, and detailed type. |

### Pol-InSAR-Island details

The selected reduced first input is FP1/L's principal T3 block:

```text
[T11, T22, T33, T12, T13, T23] -> [6, 3616, 2502] complex64
```

`T11`, `T22`, and `T33` use their stored real float32 values with an imaginary part of zero because Hermitian diagonals are real by definition. `T12`, `T13`, and `T23` are reconstructed from paired `_real.bin` and `_imag.bin` planes. The complete T6 is directly available as six diagonal plus fifteen complex upper-triangle entries: `[21,H,W] complex64`.

Official FP1 labels contain the following pixel counts (0 is unlabeled and is excluded from supervised losses):

| Class ID | train | test |
|---:|---:|---:|
| 1 Tidal flat | 73,227 | 105,965 |
| 2 Water | 247,828 | 235,948 |
| 3 Coastal shrub | 70,595 | 52,584 |
| 4 Dense high vegetation | 38,403 | 30,195 |
| 5 White dune | 49,262 | 49,764 |
| 6 Peat bog | 11,620 | 15,821 |
| 7 Grey dune | 173,131 | 206,442 |
| 8 Couch grass | 28,694 | 32,443 |
| 9 Upper saltmarsh | 132,345 | 144,341 |
| 10 Lower saltmarsh | 113,467 | 94,135 |
| 11 Sand | 137,757 | 126,210 |
| 12 Settlement | 133,546 | 153,455 |

The corresponding unlabeled counts are 7,837,357 train and 7,799,929 test. The official maps should be honored first. Because patch neighborhoods can cross spatial boundaries, validation extraction must not create overlapping train/test windows; use a central-label patch protocol and discard any patch whose receptive field intersects an opposite-split labeled pixel.

### RadioML 2016.10a details

The downloaded pickle has exactly 220,000 frames: 11 modulation classes, 20 SNR values `-20,-18,...,18`, and 1,000 frames per `(modulation,SNR)` condition. Each class is exactly balanced at 20,000 frames. The classes are `8PSK`, `AM-DSB`, `AM-SSB`, `BPSK`, `CPFSK`, `GFSK`, `PAM4`, `QAM16`, `QAM64`, `QPSK`, and `WBFM`.

```python
# raw.shape == [N, 2, 128], float32
x = raw[:, 0:1, :] + 1j * raw[:, 1:2, :]   # [N, 1, 128], complex64
```

Use the raw sequence, not a spectrogram. There is no official fixed split in the pickle. Use a deterministic stratified 70/10/20 or 80/10/10 split inside each `(modulation,SNR)` cell, record the seed, and report both all-SNR and per-SNR accuracy. The release has no burst/source identifier, so signal-group-disjointness cannot be proven; this is a residual leakage limitation.

### SARFish label audit

SARFish remains a scientifically strong SLC candidate but is deferred due to size. Its provided train/validation split is scene-disjoint. For the first classification extraction, use a fixed physical-size crop centered on each annotation in VV/VH SLC, retaining complex samples and avoiding arbitrary complex interpolation. Do not independently normalize real and imaginary parts per chip.

| split | rows | scenes | vessel true | vessel false | vessel blank |
|---|---:|---:|---:|---:|---:|
| train | 64,286 | 553 | 36,482 | 16,712 | 11,092 |
| validation | 19,186 | 50 | 11,941 | 7,127 | 118 |

The labels support vessel/not-vessel and fishing/not-fishing, rather than a Cargo/Tanker/Fishing multiclass ship-type task. Keep train/val scene-disjoint; ship identity is not available sufficiently to prove vessel-disjointness.


### OpenSARShip SLC file audit

The 1.826 GB official archive contains 85 nested scene archives: 35 Sentinel-1
SLC and 50 GRD. It is the 19 OpenSARShip1 SLC scenes with `Patch/*.tif`
that are immediately usable, yielding 2,643 native-complex labeled chips. A
representative `Cargo_x10004_y2630.tif` is `85x85x4 float32`; its TIFF IFD
records 4 samples/pixel and IEEE float sample format. Pixel payload starts at
byte 8. The canonical reconstruction is:

```python
# raw is H x W x 4 float32
x = stack((raw[..., 0] + 1j * raw[..., 1],
           raw[..., 2] + 1j * raw[..., 3]))  # [2,H,W] complex64
```

The example's first pair has 93.94% nonzero imaginary samples and the second
97.77%, ruling out magnitude-only storage. Metadata says the two channels are
VH and VV; the exact plane order should be documented in the loader and
checked against a second scene before publication. `Ship.xml` supplies MMSI,
AIS `Ship_Type`, detailed type, chip coordinates, and scene acquisition context.

The usable SLC chip class distribution is highly imbalanced: Cargo 1,750;
Tanker 489; Other Type 267; Fishing 52; Tug 26; Dredging 23; Passenger 13;
Pilot Vessel 6; Law Enforcement 6; Wing in ground 2; Port Tender 2; Search 2;
Towing 2; High speed craft 1; Diving ops 1; Pleasure Craft 1. The 2,643 chips
have native dimensions from `9x9` to `445x445` (mostly square) and the 19
scenes' XML files expose 2,255 nonzero MMSI values. The first protocol should
use a defensible 3-class subset (Cargo/Tanker/Other Type) or a coarser AIS
hierarchy, apply a fixed physical-size crop/padding policy rather than arbitrary
complex interpolation, and make both MMSI and scene disjoint.

## Table 3 - Preprocessing and Current-Flow Compatibility

| Dataset | Canonical tensor | Normalization | Patch/window and split | LUT6 interpretation / code work |
|---|---|---|---|---|
| Pol-InSAR T3 | `[6,H,W] complex64` | Train-split per-channel real/imag standardization; preserve diagonal imag=0 | Central-label spatial patches; official map with receptive-field exclusion | Each activation is a coherency feature; phase bit measures local real/imag dominance. Current 2-D flow reusable after dataset loader/protocol only. |
| Pol-InSAR T6 | `[21,H,W] complex64` | Same, channelwise | Same as T3; do not train yet | Same 2-D operator; first stem must accept 21 channels. Dataset-only + input config. |
| RadioML 2016 | `[1,128] complex64` | Global training-set RMS or global real/imag scale; never phase-destroying per-frame component normalization | Frame-level stratified split by `(mod,SNR)` | One activation is an IQ sample; phase bit is I/Q dominance. ComplexConv1d and PairLUT1d required. |
| RadioML 2018 | `[1,1024] complex64` | Same as 2016 | Stratify modulation/SNR, document seed | ComplexConv1d/LUT1d required; loader only after HDF5 acquisition. |
| SARFish | `[2,H,W] complex64` for VV/VH | Compare global per-pol real/imag standardization with per-chip common complex scale | crop native SLC around annotations; retain scene split | Complex pixel, phase physically meaningful. Current 2-D flow after chip extractor. |
| OpenSARShip SLC | `[2,H,W] complex64` from `(band0+j*band1, band2+j*band3)` | Train-only per-polarization real/imag standardization; optionally common per-chip complex scale | native variable chips, pad/crop to a fixed physical window; MMSI- and scene-disjoint split | Valid current 2-D input; loader extracts inner ZIP TIFF and labels from `Ship.xml`. |

The current PairLUT operator groups two complex activations. This maps directly to two logical spatial positions in 2-D and two neighboring/grouped samples in 1-D. The phase-refinement bit is meaningful for T3/T6 coherence off-diagonals, raw I/Q, and SLC real/imag pixels; it is not meaningful for rejected magnitude-only products.

## Table 4 - Integration Cost

| Dataset | Current 2-D flow reusable? | 1-D flow required? | Loader work | Model work | Difficulty |
|---|---|---|---|---|---|
| Pol-InSAR T3 | Yes | No | ENVI complex reconstruction + patch protocol | input channels = 6 | Low |
| Pol-InSAR T6 | Yes | No | Same | input channels = 21 | Low-medium |
| RadioML 2016 | No | Yes | pickle to complex frame loader | ComplexConv1d/BiReal/LUT1d | Medium |
| RadioML 2018 | No | Yes | HDF5 loader | same 1-D stack | Medium |
| SARFish | Yes | No | High: SAFE/SLC crop extraction | input channels = 2 | High |
| OpenSARShip SLC | Yes | No | nested ZIP/TIFF/XML decoder and grouped split | input channels = 2 | Medium |

## Leakage Audit

- **Pol-InSAR-Island:** official train/test maps are preferable to a random pixel split, but overlapping neighborhood patches would still leak spatial context. Enforce a guard band at least equal to patch radius.
- **RadioML:** stratification prevents class/SNR imbalance. The source has no transmitter/burst ID, so random frame splits cannot rule out latent generation-source overlap.
- **SARFish:** preserve the supplied scene-disjoint train/validation split; do not split crops randomly across scenes. Annotations alone do not give a robust vessel identity split.
- **OpenSARShip/FUSAR-Ship:** random chip splits are unsafe. OpenSARShip's `Ship.xml` contains MMSI and scene identity, so require both MMSI-disjoint and scene-disjoint splits. FUSAR has MMSI and should be grouped by it for any real-valued reference experiment.

## Table 5 - Ranking (1 = poor, 5 = strong)

Scores are ordered as: genuine complex information, phase relevance, public reproducibility, labels, current-flow compatibility, implementation effort, size/practicality, leakage risk, scientific value.

| Rank | Dataset | Scores | Verdict |
|---:|---|---|---|
| 1 | Pol-InSAR-Island T3 | 5, 5, 5, 5, 5, 5, 4, 4, 5 | Start next |
| 2 | RadioML 2016.10a | 5, 5, 5, 5, 2, 3, 5, 3, 5 | First 1-D extension |
| 3 | OpenSARShip original SLC | 5, 5, 5, 4, 5, 4, 4, 5, 5 | Use after Pol-InSAR; keep rare labels out of first protocol |
| 4 | RadioML 2018.01A | 5, 5, 2, 5, 2, 3, 3, 3, 5 | Formal benchmark after acquisition |
| 5 | SARFish | 5, 5, 2, 4, 4, 2, 1, 4, 5 | Defer: source footprint too large |
| 6 | Sentinel-1 SLC ship-type release | unverified, unverified, 1, 1, 3, 3, 4, unverified, 4 | Paper-only release; do not use |
| 7 | FUSAR-Ship | 1, 1, 3, 4, 4, 4, 5, 2, 3 | Reject for complex claim; useful only as a real-valued reference |
| 8 | CSRSDD | unverified, unverified, 1, unverified, 3, 1, unverified, unverified, 4 | NOT PUBLICLY REPRODUCIBLE |

## Sources, Acquisition Record, and Next Commands

All raw downloads and audits are under `datasets_external/<dataset>/{raw,metadata,processed,audit}`. The reusable inspector is `tools/audit_complex_dataset.py`.
At audit completion, the external dataset directory occupies approximately 17 GB locally, including the 2.8 GB intentionally incomplete SARFish sample and the Pol-InSAR inner archive extracted solely for file-level inspection. The directory is Git-ignored.

- Pol-InSAR-Island V2: [KIT/DLR dataset record](https://publikationen.bibliothek.kit.edu/1000161445), [RADAR direct record](https://radar.kit.edu/radar/de/dataset/IAzBEMXnbTndvIZG). Downloaded 2026-09-04; archive expected and observed MD5 above; CC BY-NC-SA 4.0.
- RadioML 2016.10a: [Zenodo reproducible record](https://zenodo.org/records/18397070), [DeepSig dataset page](https://www.deepsig.ai/datasets/). Downloaded 2026-09-04; archive MD5 `ffedb2c24e33b826d3d0df4aa8f32c1b`; CC BY-NC-SA 4.0.
- RadioML 2018.01A: [DeepSig dataset page](https://www.deepsig.ai/datasets/), [Kaggle metadata mirror](https://www.kaggle.com/datasets/pinxau1000/radioml2018). Official URL returned a contact page and the mirror endpoint was unavailable without external access.
- Sentinel-1 ship-type: [Zenodo record](https://zenodo.org/records/20492437). Current record package contains only the paper.
- SARFish: [official repository](https://github.com/DIUx-xView/SARFish), [sample release](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFishSample). Full raw acquisition intentionally skipped.
- OpenSARShip: [official data/code page](https://opensar.sjtu.edu.cn/DataAndCodes.html). Downloaded 2026-09-04; SHA-256 `c55e5596788bff619723b94ed7b717277d9e7883e3366c400a53503cf4eaa276`. Actual SLC audit is described below.
- FUSAR-Ship: [Fudan resource page](https://emwlab.fudan.edu.cn/resources/list.htm). No explicit license text was found in the downloaded release.

After implementing the loaders (not part of this audit), the proposed order is:

```bash
# 1. Pol-InSAR T3: first reusable 2-D benchmark
./experiments/run_flow.sh --dataset pol_insar_island_t3 --protocol official --seed 0

# 2. RadioML 2016: after ComplexConv1d / PairLUT1d are implemented
./experiments/run_flow.sh --dataset radioml_2016_10a --protocol stratified_snr --seed 0
```

These commands are deliberately not runnable yet: no dataset loader or 1-D model has been added in this audit-only task.
