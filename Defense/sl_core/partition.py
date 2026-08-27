#partition.py

import numpy as np

from collections import defaultdict



def main_label_partition(labels, num_clients=10, iid_rate=0.8, num_classes=43, seed=1337, root_size=150):

    rng = np.random.default_rng(seed)

    labels = np.array(labels)

    

    # 获取所有的索引

    all_indices = np.arange(len(labels))

    rng.shuffle(all_indices)

    

    # 为 FLTrust 抽离 Root Dataset。尽量按类别均衡抽样，避免 clean warmup 偏向头部类别。

    root_indices = []

    indices_by_class_all = {c: all_indices[labels[all_indices] == c].tolist() for c in range(num_classes)}

    for c in indices_by_class_all:

        rng.shuffle(indices_by_class_all[c])

    per_class_root = root_size // num_classes

    extra_root = root_size % num_classes

    for c in range(num_classes):

        take = per_class_root + (1 if c < extra_root else 0)

        take = min(take, len(indices_by_class_all[c]))

        root_indices.extend(indices_by_class_all[c][:take])

    if len(root_indices) < root_size:

        selected = set(root_indices)

        fill_pool = [idx for idx in all_indices.tolist() if idx not in selected]

        root_indices.extend(fill_pool[:root_size - len(root_indices)])

    root_set = set(root_indices)

    

    # 剩下的给客户端

    remaining_indices = np.array([idx for idx in all_indices.tolist() if idx not in root_set])

    idx_by_class = {c: [] for c in range(num_classes)}

    for idx in remaining_indices:

        idx_by_class[labels[idx]].append(idx)



    for c in idx_by_class:

        rng.shuffle(idx_by_class[c])



    mains = rng.integers(low=0, high=num_classes, size=num_clients)

    client_indices = [[] for _ in range(num_clients)]

    

    total_remaining = len(remaining_indices)

    per_client = total_remaining // num_clients



    for i in range(num_clients):

        need = per_client

        k_iid = int(round(iid_rate * need))

        k_main = need - k_iid



        for _ in range(k_iid):

            c = int(rng.integers(0, num_classes))

            if not idx_by_class[c]:

                for cc in range(num_classes):

                    if idx_by_class[cc]:

                        c = cc; break

            client_indices[i].append(idx_by_class[c].pop())



        main = int(mains[i])

        for _ in range(k_main):

            if not idx_by_class[main]:

                for cc in range(num_classes):

                    if idx_by_class[cc]:

                        main = cc; break

            client_indices[i].append(idx_by_class[main].pop())



    remainder = []

    for c in range(num_classes):

        remainder.extend(idx_by_class[c])

    rng.shuffle(remainder)

    for j, idx in enumerate(remainder):

        client_indices[j % num_clients].append(idx)



    return client_indices, mains.tolist(), root_indices