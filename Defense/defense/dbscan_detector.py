# import torch
# import numpy as np
# from sklearn.cluster import DBSCAN


# class ModelBuffer:
#     def __init__(self, max_size=15):
#         self.max_size = max_size
#         self.buffer = []

#     def add(self, vec):
#         self.buffer.append(vec.clone().detach())
#         if len(self.buffer) > self.max_size:
#             self.buffer.pop(0)

#     def get(self):
#         return self.buffer


# def cosine_sim(a, b):
#     return torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)


# def compute_weighted_score(current, history):
#     n = len(history)
#     weights = np.arange(n, 0, -1)
#     weights = weights / weights.sum()

#     score = 0
#     for i, h in enumerate(history):
#         score += weights[i] * cosine_sim(current, h).item()
#     return score


# def dbscan_detect(scores, eps=0.05, min_samples=2):
#     X = np.array(scores).reshape(-1, 1)
#     labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
#     return labels

import torch
import numpy as np
from sklearn.cluster import DBSCAN

# --- 你提供的新增核心逻辑 ---
class ModelBuffer:
    def __init__(self, max_size=15):
        self.max_size = max_size
        self.buffer = []

    def add(self, vec):
        # 存入 CPU 节省显存
        self.buffer.append(vec.clone().detach().cpu())
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get(self):
        return self.buffer

def cosine_sim(a, b):
    # 确保是一维向量计算
    return torch.dot(a.view(-1), b.view(-1)) / (torch.norm(a) * torch.norm(b) + 1e-8)

def compute_weighted_score(current, history):
    if not history:
        return 0.0 # 第一轮没有历史，返回0
    n = len(history)
    weights = np.arange(n, 0, -1)
    weights = weights / weights.sum()

    score = 0
    for i, h in enumerate(history):
        score += weights[i] * cosine_sim(current, h).item()
    return score

def dbscan_detect(scores, eps=0.05, min_samples=2):
    X = np.array(scores).reshape(-1, 1)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return labels