# Running AreTomo on SHREC Dataset

## 1. Prerequisites

Run through the download and preprocessing steps described in `how-to-run.md`.

## 2. Create Output Directories

```bash
cd /path/to/shrec_benchmark/
mkdir -p at3_alignment at2_alignment
```

## 3. Run AreTomo3

To run only alignment and reconstruction on a folder with `.st` and `.rawtlt` files (CTF correction disabled):

```bash
AreTomo3 \
  -InPrefix iter0/ \
  -InSuffix .st \
  -OutDir at3_alignment/ \
  -TiltAxis 0.0001 -1 \
  -TiltCor -1 \
  -VolZ 180 \
  -AlignZ 180 \
  -Cmd 1 \
  -Serial 1 \
  -CorrCTF 0 \
  -DarkTol 0 \
  -Gpu 3 \
  -FlipVol 1 \
  -Wbp 1
```

### Notes on Parameter Selection

- TiltAxis optimization was disabled as the data has the axis angle in this simulated set is exactly at 0.0 ; leaving it free to optimize results in values up to 5°
- TiltCor is disabled because the samples lay exactly flat in the volume
- AlignZ and VolZ are 180 because this was the thickness of the simulated sample
- DarkTol is disabled because none of the tilts are too dark

### Notes on TiltAxis

- Setting `-TiltAxis 0.0 -1` was attempted to disable tilt axis estimation for the SHREC dataset, but this did not work
- The value must be set to `0.0001` instead of `0.0` (which appears to be a sentinel value)
- **Important**: Even with this setting, alignments are very poor with AreTomo3 on SHREC data

## 4. Run AreTomo2 (Recommended)

AreTomo2 produces decent alignments on the SHREC dataset with the following command:

```bash
for x in {0..9}; do
  AreTomo2 \
    -InMrc iter0/model_"$x".st \
    -OutMrc at2_alignment/model_"$x"_Vol.mrc \
    -AngFile iter0/model_"$x".rawtlt \
    -TiltAxis 0.0001 -1 \
    -AlignZ 180 \
    -VolZ 180 \
    -OutBin 1 \
    -TiltCor -1 \
    -FlipVol 1 \
    -Wbp 1 \
    -DarkTol 0
done
```
