# GoogLeNet PMR Benchmark

Generated: 2026-08-20T02:58:09

Fixed settings: GoogLeNet, GTSRB, 30 clients, 30 rounds, `--unseen 0.7`, `--poison-rate 0.8`, attack + defense, `--phase2-history-size 15`.

| PMR | Malicious Clients | Status | Rounds | MA | Clean Accuracy | Unseen Accuracy | ASR | Log |
|---|---:|---|---:|---:|---:|---:|---:|---|
| 10% | 3 | completed | 30 | 0.9381 | 0.9381 | 0.9583 | 0.0000 | [stdout.log](Defense/googlenet_pmr_runs/googlenet_pmr_10pct_m3/stdout.log) |
| 20% | 6 | completed | 30 | 0.9379 | 0.9379 | 0.9602 | 0.0000 | [stdout.log](Defense/googlenet_pmr_runs/googlenet_pmr_20pct_m6/stdout.log) |
| 30% | 9 | completed | 30 | 0.9471 | 0.9471 | 0.9412 | 0.1591 | [stdout.log](Defense/googlenet_pmr_runs/googlenet_pmr_30pct_m9/stdout.log) |
| 40% | 12 | completed | 30 | 0.9411 | 0.9411 | 0.9602 | 0.0006 | [stdout.log](Defense/googlenet_pmr_runs/googlenet_pmr_40pct_m12/stdout.log) |
