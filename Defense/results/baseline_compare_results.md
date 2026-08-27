# Baseline Comparison on GTSRB / GoogLeNet

Setting: 30 clients, 4 malicious clients, 4 unseen clients, 30 rounds, trigger attack, `poison-rate=0.8`, `unseen=0.7`.

DP baseline uses client update clipping with `clip_norm=2.0` and Gaussian noise multiplier `0.05`. k-means baseline clusters update statistics and keeps the majority cluster.

| Method | Status | Clean Accuracy / MA | Unseen Accuracy | ASR | Best MA | Best Unseen | Min ASR | Log |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Differential Privacy | completed | 0.9351 | 0.7970 | 0.6000 | 0.9394 | 0.9203 | 0.0001 | `Defense/baseline_compare_runs/dp/stdout.log` |
| k-means | completed | 0.9562 | 0.1556 | 0.0000 | 0.9562 | 0.1575 | 0.0000 | `Defense/baseline_compare_runs/kmeans/stdout.log` |
| ZORRO | completed | 0.9108 | 0.4118 | 0.0000 | 0.9108 | 0.4326 | 0.0000 | `Defense/baseline_compare_runs/zorro/stdout.log` |

Per-round metrics are stored in `baseline_compare_per_round.csv`.
