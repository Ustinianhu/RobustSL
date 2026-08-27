# Different Client-Count Experiment Configuration


| Item | Setting |
|---|---|

| Dataset | GTSRB + China-GTSRB/TT100K unseen data |

| Model | ResNet18 split learning architecture |

| Total clients | 5, 10, 15, 20, 25, 30, 40 |

| Unseen client ratio | 20% (`max(1, round(0.2N))`) |

| Unseen data ratio | 70% of each unseen client's local data |

| Malicious client ratio (PMR) | 10% (`max(1, round(0.1N))`) |

| Poison rate | 80% of malicious-client local batches |

| Attack | Trigger backdoor, target label 0, trigger size 6 |

| Defense | Current Phase I + tri-view Phase II |

| Phase II views | DCT low frequency + update L2 + Gram representation |

| Phase II queue | Latest 15 global model versions; minimum 3 versions |

| Fusion | DCT/L2/Gram weights = 1/1/1 |

| Threshold | Historical risk quantile = 0.90; MAD-k = 1.5 during bootstrap |

| Training | 30 rounds, max 20 local steps, batch size 64 |

| Clean reference | Root size 1000, root pretraining 500 steps |

| Data partition | `iid_rate=0.8`, fixed seed from `configs/gtsrb.yaml` |


The experiment enables `--exact-unseen-sampling` only for this scalability study. This preserves the requested unseen-data fraction when each client has very few samples, for small client partitions, while leaving the default behavior of existing runs unchanged.
