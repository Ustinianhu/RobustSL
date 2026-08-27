# GoogLeNet Unseen Ratio Benchmark

Generated: 2026-08-18T22:55:47

All runs use attack + defense, 30 clients, 30 federated rounds, and GoogLeNet on GTSRB.
Defense flags: `--enable-defense --phase2-history-range --phase2-history-size 15 --phase2-history-min-versions 3`.

| Unseen Ratio | Status | Rounds | MA | Clean Accuracy | Unseen Accuracy | ASR | Log |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.4 | completed | 30 | 0.9428 | 0.9428 | 0.7343 | 0.0000 | [stdout.log](Defense/googlenet_unseen_ratio_runs/googlenet_unseen_0.4_attack_defense/stdout.log) |
| 0.6 | completed | 30 | 0.9430 | 0.9430 | 0.9374 | 0.0000 | [stdout.log](Defense/googlenet_unseen_ratio_runs/googlenet_unseen_0.6_attack_defense/stdout.log) |
| 0.8 | completed | 30 | 0.9314 | 0.9314 | 0.0892 | 0.0000 | [stdout.log](Defense/googlenet_unseen_ratio_runs/googlenet_unseen_0.8_attack_defense/stdout.log) |
| 1.0 | completed | 30 | 0.9247 | 0.9247 | 0.0949 | 0.0000 | [stdout.log](Defense/googlenet_unseen_ratio_runs/googlenet_unseen_1.0_attack_defense/stdout.log) |

Files:
- Full logs: `Defense/googlenet_unseen_ratio_runs`
- Summary CSV: `Defense/googlenet_unseen_ratio_results.csv`
- Per-round CSV: `Defense/googlenet_unseen_ratio_per_round.csv`
