# Eta Ablation on RobustSL

Generated: 2026-08-25T23:41:30

Only the global aggregation step size eta is changed.

## Fixed Configuration

- Dataset: GTSRB with China-GTSRB/TT100K unseen data.
- Model: ResNet18 split learning architecture.
- Total clients: 30; unseen clients: 6; unseen data ratio: 70%.
- Malicious clients: 3; PMR: 10%; poison rate: 80%; trigger attack.
- Training: 30 rounds; 20 local steps; batch size 64.
- Defense: current RobustSL Phase I and Phase II implementation.
- Phase II: DCT, L2, and Gram views with equal weights.
- Root pretraining: 1000 samples and 500 clean steps.

## Final Results

| eta | MA | UA | ASR | Status | Log |
|---:|---:|---:|---:|---|---|
| 0.25 | 0.9074 | 0.8083 | 0.1030 | completed | [stdout.log](eta_ablation_runs/eta_0_25/stdout.log) |
| 0.50 | 0.9335 | 0.9013 | 0.1644 | completed | [stdout.log](eta_ablation_runs/eta_0_5/stdout.log) |
| 0.75 | 0.9438 | 0.8994 | 0.2683 | completed | [stdout.log](eta_ablation_runs/eta_0_75/stdout.log) |
| 1.00 | 0.9428 | 0.9298 | 0.0001 | completed | [stdout.log](eta_ablation_runs/eta_1/stdout.log) |

## Target Check

Qualified eta values (MA >= 0.90, UA >= 0.90, ASR < 0.01): `1.00`
Best qualified eta by mean(MA, UA): `1.00` (MA=0.9428, UA=0.9298, ASR=0.0001).
