# GTSRB Client-Count Experiment

Generated: 2026-08-24T19:37:26

Controlled setting: GTSRB, ResNet18, 30 rounds, iid_rate=0.8, current tri-view Phase-II defense.
Unseen client ratio = 20%; unseen data ratio within an unseen client = 70%; PMR = 10%; poison rate = 80%.
The exact unseen sampling flag is enabled so that the requested unseen fraction remains valid for 500/1000 clients.

## Results

| Clients | Unseen clients | Malicious clients | MA | UA | ASR | Status | Log |
|---:|---:|---:|---:|---:|---:|---|---|
| 5 | 1 | 1 | 0.9358 | 0.9355 | 0.0001 | completed | [stdout.log](client_count_experiment_runs/clients_0005/stdout.log) |
| 10 | 2 | 1 | 0.9453 | 0.9488 | 0.0002 | completed | [stdout.log](client_count_experiment_runs/clients_0010/stdout.log) |
| 15 | 3 | 2 | 0.9298 | 0.9469 | 0.0000 | completed | [stdout.log](client_count_experiment_runs/clients_0015/stdout.log) |
| 20 | 4 | 2 | 0.9397 | 0.9374 | 0.0003 | completed | [stdout.log](client_count_experiment_runs/clients_0020/stdout.log) |
| 25 | 5 | 2 | 0.9241 | 0.9146 | 0.0019 | completed | [stdout.log](client_count_experiment_runs/clients_0025/stdout.log) |
| 30 | 6 | 3 | 0.9230 | 0.9507 | 0.0000 | completed | [stdout.log](client_count_experiment_runs/clients_0030/stdout.log) |
| 40 | 8 | 4 | 0.9046 | 0.5901 | 0.0002 | completed | [stdout.log](client_count_experiment_runs/clients_0040/stdout.log) |

## Configuration

- Dataset: GTSRB with TT100K/China-GTSRB unseen data.
- Model: ResNet18 split into client-side head, server-side backbone, and client-side tail.
- Total rounds: 30; local step cap: 20; batch size: 64; root size: 1000.
- Phase II: DCT + L2 + Gram, equal weights, 15-version history queue, risk quantile 0.90.
- Aggregation: `aggregation_delta=1.0`, benign trust floor `0.65`.

Files: `client_count_experiment_results.csv`, `client_count_results.png`, and `client_count_results.pdf`.
