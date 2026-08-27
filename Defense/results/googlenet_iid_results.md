# GoogLeNet GTSRB IID-rate Sensitivity

实验设置：30 个客户端，30 轮，4 个恶意客户端（PMR约13.3%），`unseen=0.7`，`poison-rate=0.8`，开启当前 Phase I + Phase II 防御；仅改变 `iid_rate`。

当前划分实现中，`iid_rate` 越高表示客户端中按全类别随机抽样的数据比例越高，整体越接近 IID。

| IID Rate | Status | MA | Clean Accuracy | Unseen Accuracy | ASR | Best MA | Best Unseen | Min ASR | Log |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0.4 | completed | 0.9472 | 0.9472 | 0.8937 | 0.0900 | 0.9472 | 0.9696 | 0.0000 | `Defense/googlenet_iid_runs/googlenet_iid_0p4/stdout.log` |
| 0.6 | completed | 0.9077 | 0.9077 | 0.9583 | 0.0000 | 0.9189 | 0.9583 | 0.0000 | `Defense/googlenet_iid_runs/googlenet_iid_0p6/stdout.log` |
| 0.8 | completed | 0.9414 | 0.9414 | 0.9602 | 0.0000 | 0.9446 | 0.9602 | 0.0000 | `Defense/googlenet_iid_runs/googlenet_iid_0p8/stdout.log` |
| 1.0 | completed | 0.9445 | 0.9445 | 0.2163 | 0.0000 | 0.9478 | 0.4099 | 0.0000 | `Defense/googlenet_iid_runs/googlenet_iid_1p0/stdout.log` |

逐轮数据：`googlenet_iid_per_round.csv`。
