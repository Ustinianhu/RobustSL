# RobustSL GTSRB Artifact

This repository contains the GTSRB-focused artifact for RobustSL, a split-learning defense against backdoor attacks under heterogeneous client data.

## Contents

- `Defense/run_RobustSL.py`: paper-aligned RobustSL implementation.
- `Defense/run_gtsrb.py`: GTSRB split-learning training and defense runner.
- `Defense/models/`: GTSRB model backbones used in the experiments.
- `Defense/sl_core/`: split-learning client/server utilities.
- `Defense/attacks/`: GTSRB backdoor attack implementation.
- `Defense/eval/`: clean accuracy, unseen accuracy, and backdoor accuracy evaluation.
- `Defense/configs/gtsrb.yaml`: default GTSRB experiment configuration.
- `Defense/results/`: summarized GTSRB experiment outputs.
- Root-level `run_*` and `plot_*` scripts: ablation/scalability experiment drivers and plotting scripts.

Only the GTSRB-related artifact is included. MNIST, FMNIST, CIFAR-10, CIFAR-100, runtime logs, caches, and model checkpoints are intentionally excluded.

## Setup

```bash
conda create -n robustsl python=3.9 -y
conda activate robustsl
pip install -r requirements.txt
```

## Data

The code expects GTSRB data under `Defense/data/gtsrb/`. See `data/GTSRB_DATA.md` for checksums and placement. Large GTSRB archives are provided through the repository release assets because the training archive exceeds GitHub's normal per-file limit for repository blobs.

## Example Run

```bash
cd Defense
python run_RobustSL.py \
  --config configs/gtsrb.yaml \
  --model resnet18 \
  --epochs 30 \
  --num-clients 30 \
  --num-new-clients 6 \
  --unseen 0.70 \
  --num-malicious 3 \
  --attack-type trigger \
  --poison-rate 0.80 \
  --enable-defense \
  --steps-per-client 20 \
  --batch-size 64 \
  --root-size 1000 \
  --root-pretrain-steps 500 \
  --phase2-probe-batches 1 \
  --phase2-gram-orders 1 2 \
  --phase2-mad-k 1.5 \
  --phase2-risk-quantile 0.90 \
  --aggregation-delta 1.0
```

The runner prints MA/Clean Accuracy, Unseen Accuracy, and ASR/Backdoor Accuracy after each training round.
