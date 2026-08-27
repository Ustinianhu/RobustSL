import torch

# GTSRB 数据集的近似均值和标准差 (RGB)
GTSRB_MEAN = (0.3337, 0.3064, 0.3171)
GTSRB_STD  = (0.2672, 0.2564, 0.2629)

def apply_red_rectangle(x, size=6):
    """
    在左上角注入一个 size×size 的红色方块
    """
    x = x.clone()
    s = min(size, x.shape[-1], x.shape[-2])
    
    red_r = (1.0 - GTSRB_MEAN[0]) / GTSRB_STD[0]
    red_g = (0.0 - GTSRB_MEAN[1]) / GTSRB_STD[1]
    red_b = (0.0 - GTSRB_MEAN[2]) / GTSRB_STD[2]
    
    x[:, 0, 0:s, 0:s] = red_r
    x[:, 1, 0:s, 0:s] = red_g
    x[:, 2, 0:s, 0:s] = red_b
    return x

def make_backdoor_hook(malicious_client_ids, p=0.1, target_label=0, size=6, dirty_label=True):
    mal_ids = set(int(i) for i in malicious_client_ids)

    def hook(x, y, client_id):
        if client_id not in mal_ids or p <= 0.0:
            return x, y

        N = x.size(0)
        mask = torch.rand(N, device=x.device) < p

        if not mask.any():
            return x, y

        x2 = x.clone()
        x2[mask] = apply_red_rectangle(x2[mask], size=size)

        y2 = y.clone()
        if dirty_label:
            y2[mask] = int(target_label)

        return x2, y2

    return hook

def make_label_flip_hook(malicious_client_ids, target_label=0, p=1.0):
    mal_ids = set(int(i) for i in malicious_client_ids)

    def hook(x, y, client_id):
        if client_id not in mal_ids or p <= 0.0:
            return x, y

        N = x.size(0)
        mask = torch.rand(N, device=x.device) < p

        y2 = y.clone()
        y2[mask] = int(target_label)

        return x, y2

    return hook