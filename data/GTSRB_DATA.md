# GTSRB Data Files

Place the following files under `Defense/data/gtsrb/` before running the GTSRB experiments.

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `GTSRB-Training_fixed.zip` | 187490228 | `df4144942083645bd60b594de348aa6930126c3e0e5de09e39611630abf8455a` |
| `GTSRB_Final_Test_Images.zip` | 88978620 | `48ba6fab7e877eb64eaf8de99035b0aaecfbc279bee23e35deca4ac1d0a837fa` |
| `GTSRB_Final_Test_GT.zip` | 99620 | `f94e5a7614d75845c74c04ddb26b8796b9e483f43541dd95dd5b726504e16d6d` |
| `GT-final_test.csv` | 377382 | `58a660fab5465cce66d7334eb9422ae09ed16186ad4500766b59b5a077a1c12f` |

The large archives are provided as GitHub Release assets instead of normal repository files. This avoids GitHub's per-file repository blob limit while keeping the review artifact accessible.

Expected layout:

```text
Defense/data/gtsrb/
  GTSRB-Training_fixed.zip
  GTSRB_Final_Test_Images.zip
  GTSRB_Final_Test_GT.zip
  GT-final_test.csv
```

After placement, the training code can also use `torchvision.datasets.GTSRB` to prepare the expected train/test split under `Defense/data/`.
