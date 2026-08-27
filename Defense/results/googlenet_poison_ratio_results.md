# GoogLeNet Poisoning Ratio Benchmark

Generated: 2026-08-19T02:10:31

All runs use attack + defense, 30 clients, 30 federated rounds, 4 malicious clients, GoogLeNet on GTSRB, and unseen ratio fixed to 0.7.
Defense flags: `--enable-defense --phase2-history-range --phase2-history-size 15 --phase2-history-min-versions 3`.

| Poison Rate | Status | Rounds | MA | Clean Accuracy | Unseen Accuracy | ASR | Log |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.4 | completed | 30 | 0.9336 | 0.9336 | 0.1879 | 0.0000 | [stdout.log](Defense/googlenet_poison_ratio_runs/googlenet_poison_0.4_attack_defense/stdout.log) |
| 0.6 | completed | 30 | 0.9451 | 0.9451 | 0.9545 | 0.0000 | [stdout.log](Defense/googlenet_poison_ratio_runs/googlenet_poison_0.6_attack_defense/stdout.log) |
| 0.8 | completed | 30 | 0.9262 | 0.9262 | 0.0892 | 0.0000 | [stdout.log](Defense/googlenet_poison_ratio_runs/googlenet_poison_0.8_attack_defense/stdout.log) |
| 1.0 | completed | 30 | 0.9285 | 0.9285 | 0.1689 | 0.0001 | [stdout.log](Defense/googlenet_poison_ratio_runs/googlenet_poison_1.0_attack_defense/stdout.log) |

Files:
- Full logs: `Defense/googlenet_poison_ratio_runs`
- Summary CSV: `Defense/googlenet_poison_ratio_results.csv`
- Per-round CSV: `Defense/googlenet_poison_ratio_per_round.csv`
