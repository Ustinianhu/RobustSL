# GTSRB Model Architecture Benchmark

Generated: 2026-08-16T13:31:52

Each model was configured for 30 federated rounds and 30 clients.
`MA` and `Clean Accuracy` are identical in the current implementation because both use `eval_clean_accuracy`.
For the no-attack condition, ASR is `N/A` because no malicious trigger is injected.
The defense condition enables the 15-version historical model-range Phase-2 logic.

## Final Results

| Model | Condition | Status | Rounds | MA | Clean Accuracy | Unseen Accuracy | ASR | Log |
|---|---|---:|---:|---:|---:|---:|---:|---|
| googlenet | attack_defense | completed | 30 | 0.9429 | 0.9429 | 0.9583 | 0.0000 | [stdout.log](Defense/architecture_benchmark_runs/googlenet__attack_defense/stdout.log) |
| googlenet | attack_no_defense | completed | 30 | 0.9624 | 0.9624 | 0.9127 | 0.6399 | [stdout.log](Defense/architecture_benchmark_runs/googlenet__attack_no_defense/stdout.log) |
| googlenet | no_attack_no_defense | completed | 30 | 0.9647 | 0.9647 | 0.9374 | N/A | [stdout.log](Defense/architecture_benchmark_runs/googlenet__no_attack_no_defense/stdout.log) |
| micronnet | attack_defense | completed | 30 | 0.6587 | 0.6587 | 0.8843 | 0.0601 | [stdout.log](Defense/architecture_benchmark_runs/micronnet__attack_defense/stdout.log) |
| micronnet | attack_no_defense | completed | 30 | 0.6804 | 0.6804 | 0.9089 | 0.9813 | [stdout.log](Defense/architecture_benchmark_runs/micronnet__attack_no_defense/stdout.log) |
| micronnet | no_attack_no_defense | completed | 30 | 0.6845 | 0.6845 | 0.8994 | N/A | [stdout.log](Defense/architecture_benchmark_runs/micronnet__no_attack_no_defense/stdout.log) |
| resnet18 | attack_defense | completed | 30 | 0.9201 | 0.9201 | 0.4459 | 0.0000 | [stdout.log](Defense/architecture_benchmark_runs/resnet18__attack_defense/stdout.log) |
| resnet18 | attack_no_defense | completed | 30 | 0.9477 | 0.9477 | 0.8956 | 0.7574 | [stdout.log](Defense/architecture_benchmark_runs/resnet18__attack_no_defense/stdout.log) |
| resnet18 | no_attack_no_defense | completed | 30 | 0.9473 | 0.9473 | 0.9127 | N/A | [stdout.log](Defense/architecture_benchmark_runs/resnet18__no_attack_no_defense/stdout.log) |
| resnet34 | attack_defense | completed | 30 | 0.9264 | 0.9264 | 0.2998 | 0.0000 | [stdout.log](Defense/architecture_benchmark_runs/resnet34__attack_defense/stdout.log) |
| resnet34 | attack_no_defense | completed | 30 | 0.9423 | 0.9423 | 0.9279 | 0.7756 | [stdout.log](Defense/architecture_benchmark_runs/resnet34__attack_no_defense/stdout.log) |
| resnet34 | no_attack_no_defense | completed | 30 | 0.9504 | 0.9504 | 0.9241 | N/A | [stdout.log](Defense/architecture_benchmark_runs/resnet34__no_attack_no_defense/stdout.log) |
| vgg11 | attack_defense | completed | 30 | 0.0071 | 0.0071 | 0.0000 | 0.0000 | [stdout.log](Defense/architecture_benchmark_runs/vgg11__attack_defense/stdout.log) |
| vgg11 | attack_no_defense | completed | 30 | 0.0048 | 0.0048 | 0.0000 | 1.0000 | [stdout.log](Defense/architecture_benchmark_runs/vgg11__attack_no_defense/stdout.log) |
| vgg11 | no_attack_no_defense | completed | 30 | 0.0499 | 0.0499 | 0.2865 | N/A | [stdout.log](Defense/architecture_benchmark_runs/vgg11__no_attack_no_defense/stdout.log) |
| wide_resnet50 | attack_defense | completed | 30 | 0.8961 | 0.8961 | 0.8880 | 0.0021 | [stdout.log](Defense/architecture_benchmark_runs/wide_resnet50__attack_defense/stdout.log) |
| wide_resnet50 | attack_no_defense | completed | 30 | 0.9241 | 0.9241 | 0.8880 | 0.7119 | [stdout.log](Defense/architecture_benchmark_runs/wide_resnet50__attack_no_defense/stdout.log) |
| wide_resnet50 | no_attack_no_defense | completed | 30 | 0.9184 | 0.9184 | 0.8861 | N/A | [stdout.log](Defense/architecture_benchmark_runs/wide_resnet50__no_attack_no_defense/stdout.log) |

## Conditions

- `no_attack_no_defense`: `--disable-attack`, no defense flag.
- `attack_no_defense`: default trigger attack, no defense flag.
- `attack_defense`: trigger attack plus `--enable-defense --phase2-history-range --phase2-history-size 15`.

## Files

- Full logs: `Defense/architecture_benchmark_runs`
- Per-round metrics: `Defense/architecture_benchmark_per_round.csv`
- Final summary CSV: `Defense/architecture_benchmark_results.csv`
