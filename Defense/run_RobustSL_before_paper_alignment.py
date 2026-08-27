# run_gtsrb.py
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import defaultdict
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

# 引入绘图库
import matplotlib.pyplot as plt

from attacks.backdoor_gtsrb import make_backdoor_hook, make_label_flip_hook
from sl_core.server import ServerBackbone
from sl_core.client import Client
from sl_core.partition import main_label_partition
from eval.metrics import eval_clean_accuracy, eval_asr_strict, eval_unseen_accuracy

try:
    import scipy.fft as scipy_fft
except ImportError:
    scipy_fft = None

#0604
def plot_channel_ddifs(ddifs_tensor, unseen_ids, malicious_ids, epoch, save_dir="channel_plots"):
    os.makedirs(save_dir, exist_ok=True)
    num_clients, num_channels = ddifs_tensor.shape
    
    # 1. 获取各类客户端的 ID 列表
    benign_ids = [i for i in range(num_clients) if i not in unseen_ids and i not in malicious_ids]
    
    # 2. 计算各类客户端在所有通道上的平均值
    benign_mean = ddifs_tensor[benign_ids].mean(dim=0).numpy() if benign_ids else np.zeros(num_channels)
    unseen_mean = ddifs_tensor[unseen_ids].mean(dim=0).numpy() if unseen_ids else np.zeros(num_channels)
    malicious_mean = ddifs_tensor[malicious_ids].mean(dim=0).numpy() if malicious_ids else np.zeros(num_channels)
    
    plt.figure(figsize=(15, 5))
    x = np.arange(num_channels)
    width = 0.25
    
    # 3. 绘制群体平均柱状图
    if benign_ids:
        plt.bar(x - width, benign_mean, width, label=f'Benign (Avg of {len(benign_ids)} clients)', color='green', alpha=0.7)
    if unseen_ids:
        plt.bar(x, unseen_mean, width, label=f'Unseen (Avg of {len(unseen_ids)} clients)', color='blue', alpha=0.7)
    if malicious_ids:
        plt.bar(x + width, malicious_mean, width, label=f'Backdoor (Avg of {len(malicious_ids)} clients)', color='red', alpha=0.8)
        
    # ---------------- 【核心修改区：动态 Y 轴】 ----------------
    # 找出当前所有画出来的柱子里的最大值
    all_vals = []
    if benign_ids: all_vals.extend(benign_mean)
    if unseen_ids: all_vals.extend(unseen_mean)
    if malicious_ids: all_vals.extend(malicious_mean)
    max_val = max(all_vals) if all_vals else 1.0
    
    # 设定 Y 轴最高显示到当前最大值的 1.2 倍（留出顶部空间）
    # 但保底显示到 2.0，以免早期波动太小导致基准线 (1.0) 贴在半空
    y_upper_limit = max(2.0, max_val * 1.2)
    # --------------------------------------------------------

    # 画出参考线
    plt.axhline(y=10.0, color='black', linestyle='--', label='Mask Threshold (10.0)')
    plt.axhline(y=1.0, color='gray', linestyle=':', label='Global Baseline (1.0)')
    
    # 【强制限制 Y 轴范围】：把视角锁定在有数据的区域，无视 10.0 的线带来的拉伸
    plt.ylim(0, y_upper_limit)
    
    plt.title(f'Feature-DDifs Channel Ratios at Epoch {epoch} (Group Averages)', fontsize=16, fontweight='bold')
    plt.xlabel('Channel Index (0-31)', fontsize=14)
    plt.ylabel('Average Activation Ratio', fontsize=14)
    plt.xticks(x)
    
    # 调整图例
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'channels_epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot] 通道群体平均比率对比图已保存至: {save_path}")


def plot_phase1_dbscan(metrics_np, clusters, unseen_ids, malicious_ids, epoch, save_dir="phase1_plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 7))
    
    num_clients = len(metrics_np)
    
    # 找出真实的良性客户端 ID
    benign_ids = [i for i in range(num_clients) if i not in unseen_ids and i not in malicious_ids]
    
    # 定义三类真实身份的样式: (图例名称, ID集合, 颜色, 形状, 大小)
    categories = [
        ('Benign (Ground Truth)', benign_ids, 'black', 'o', 100),
        ('Unseen (Ground Truth)', unseen_ids, 'blue', 'o', 100),
        ('Backdoor (Ground Truth)', malicious_ids, 'red', 'x', 120)
    ]
    
    # 1. 按真实身份绘制散点
    for label, ids, col, marker, size in categories:
        if len(ids) > 0:
            xy = metrics_np[ids]
            # 只有圆点加白边，叉号(x)不加白边
            edge_color = 'white' if marker == 'o' else 'none'
            plt.scatter(xy[:, 0], xy[:, 1], c=col, marker=marker, s=size, label=label, edgecolors=edge_color)
            
    # 2. 按真实身份为文本标注上色
    for i, (x, y) in enumerate(metrics_np):
        if i in malicious_ids:
            text_color = 'red'
        elif i in unseen_ids:
            text_color = 'blue'
        else:
            text_color = 'black'
            
        # 文本标注附加当前点被分配的 DBSCAN 簇 ID (例如: C5(c-1) 表示客户端5被判定为-1离群点)
        cluster_id = clusters[i]
        label_text = f"C{i}(c{cluster_id})"
        
        plt.annotate(label_text, (x, y), textcoords="offset points", xytext=(0, 10), 
                     ha='center', fontsize=9, fontweight='bold', color=text_color)

    plt.title(f'Phase 1 Clustering Space at Epoch {epoch}\n(Colors indicate True Labels, text "c" indicates DBSCAN Cluster)', fontsize=14, fontweight='bold')
    plt.xlabel('Average Channel Shift (MAE) [Standardized]', fontsize=12)
    plt.ylabel('Gradient Norm [Standardized]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    
    save_path = os.path.join(save_dir, f'phase1_dbscan_epoch_{epoch:03d}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  [Plot] Phase 1 DBSCAN 二维聚类图 (真实标签上色) 已保存至: {save_path}")


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_model_classes(model_name):
    name = model_name.lower().strip().replace('-', '_')
    model_map = {
        'default': 'models.resnet_gtsrb',
        'resnet': 'models.resnet_gtsrb',
        'resnet_gtsrb': 'models.resnet_gtsrb',
        'custom_resnet': 'models.resnet_gtsrb',
        'simple_cnn': 'models.simple_cnn_gtsrb',
        'simplecnn': 'models.simple_cnn_gtsrb',
        'cnn': 'models.simple_cnn_gtsrb',
        'resnet18': 'models.resnet18_gtsrb',
        'resnet34': 'models.resnet34_gtsrb',
        'googlenet': 'models.googlenet_gtsrb',
        'vgg11': 'models.vgg11_gtsrb',
        'wide_resnet50': 'models.wide_resnet50_gtsrb',
        'wideresnet50': 'models.wide_resnet50_gtsrb',
        'wide_resnet50_2': 'models.wide_resnet50_gtsrb',
        'micronnet': 'models.micronnet_gtsrb',
    }
    if name not in model_map:
        raise ValueError(
            f"Unsupported model '{model_name}'. Available: "
            "resnet_gtsrb, simple_cnn, resnet18, resnet34, googlenet, "
            "vgg11, wide_resnet50, micronnet"
        )

    module = __import__(model_map[name], fromlist=['Head', 'Backbone', 'Tail'])
    return module.Head, module.Backbone, module.Tail

class ChinaGTSRBDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        if os.path.exists(root):
            for class_dir in os.listdir(root):
                dir_path = os.path.join(root, class_dir)
                if os.path.isdir(dir_path):
                    label = int(class_dir) 
                    for img_name in os.listdir(dir_path):
                        self.samples.append((os.path.join(dir_path, img_name), label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

class MixAlignedDataset(Dataset):
    def __init__(self, gtsrb_ds, china_ds, gtsrb_indices, china_indices):
        self.gtsrb_ds = gtsrb_ds
        self.china_ds = china_ds
        self.items = [('gtsrb', i) for i in gtsrb_indices] + [('china', i) for i in china_indices]
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        source, orig_idx = self.items[idx]
        return self.gtsrb_ds[orig_idx] if source == 'gtsrb' else self.china_ds[orig_idx]

def build_dataloaders(cfg, num_new_clients=1, unseen=0.3, root_size=150,
                      exact_unseen_sampling=False):
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])
    
    train = datasets.GTSRB(root='data', split='train', download=True, transform=tfm)
    test  = datasets.GTSRB(root='data', split='test', download=True, transform=tfm)
    targets = train._labels if hasattr(train, '_labels') else [label for _, label in train]

    idxs, mains, root_indices = main_label_partition(
        targets, num_clients=cfg['num_clients'], iid_rate=cfg['iid_rate'], num_classes=43, seed=cfg['seed'], root_size=root_size
    )
    
    root_dataset = Subset(train, root_indices)
    root_loader = DataLoader(root_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    
    china_train_ds = ChinaGTSRBDataset(root='data/china_gtsrb/train', transform=tfm)
    china_test_ds = ChinaGTSRBDataset(root='data/china_gtsrb/test', transform=tfm)

    china_train_by_class = defaultdict(list)
    for i, (_, label) in enumerate(china_train_ds.samples):
        china_train_by_class[label].append(i)
        
    available_china_classes = list(china_train_by_class.keys())
    
    #***
    # new_data_clients = np.random.choice(cfg['num_clients'], num_new_clients, replace=False).tolist()
    # print(f"\n[Data] 被选为新数据(TT100K)的客户端: {new_data_clients}, 比例 unseen={unseen}")

    # 从总客户端数 (cfg['num_clients']) 中，随机挑选 num_new_clients 个，且不允许重复 (replace=False)
    new_data_clients = np.random.choice(cfg['num_clients'], num_new_clients, replace=False).tolist()
    
    print(f"\n[Data] 【随机生成】新数据(TT100K)的客户端: {new_data_clients}, 比例 unseen={unseen}")

    loaders = []
    for i in range(cfg['num_clients']):
        if i in new_data_clients and len(available_china_classes) > 0:
            num_total = len(idxs[i])
            num_china_target = int(num_total * unseen)
            
            china_indices = []
            if num_china_target > 0:
                if exact_unseen_sampling:
                    # Preserve the requested unseen fraction even when each client
                    # has only a small local partition (e.g., 500/1000 clients).
                    all_china_indices = np.arange(len(china_train_ds), dtype=np.int64)
                    sampled = np.random.choice(
                        all_china_indices,
                        size=num_china_target,
                        replace=(num_china_target > len(all_china_indices)),
                    )
                    china_indices.extend(sampled.tolist())
                else:
                    per_class_target = num_china_target // len(available_china_classes)
                    for cls in available_china_classes:
                        pool = china_train_by_class[cls]
                        if len(pool) > 0:
                            sampled = np.random.choice(pool, per_class_target, replace=(per_class_target > len(pool)))
                            china_indices.extend(sampled.tolist())
            
            actual_china_count = len(china_indices)
            keep_gtsrb_count = max(0, num_total - actual_china_count)
            
            np.random.shuffle(idxs[i])
            gtsrb_indices = idxs[i][:keep_gtsrb_count]
            
            mixed_ds = MixAlignedDataset(train, china_train_ds, gtsrb_indices, china_indices)
            loaders.append(DataLoader(mixed_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=0))
        else:
            loaders.append(DataLoader(Subset(train, idxs[i]), batch_size=cfg['batch_size'], shuffle=True, num_workers=0))

    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=0)
    china_test_loader = DataLoader(china_test_ds, batch_size=512, shuffle=False, num_workers=0) if len(china_test_ds) > 0 else test_loader
    
    return loaders, test_loader, china_test_loader, root_loader, new_data_clients

def compute_phase1_baseline(head0, tail0, server, root_loader, device):
    """
    计算正常梯度基准：使用真实的 Tail 和 Loss 函数进行完整前向/反向传播
    """
    head0.eval()
    tail0.eval()
    server.model.eval()
    
    criterion = torch.nn.CrossEntropyLoss()
    norms = []
    
    for x, y in root_loader:
        x, y = x.to(device), y.to(device)
        
        a = head0(x)
        a_server = a.detach().clone().to(server.device).requires_grad_(True)
        
        b_server = server.model(a_server)
        b_tail = b_server.detach().to(device).requires_grad_(True)
        
        logits = tail0(b_tail)
        loss = criterion(logits, y)
        
        head0.zero_grad(set_to_none=True)
        tail0.zero_grad(set_to_none=True)
        server.opt.zero_grad(set_to_none=True)
        
        loss.backward() 
        
        g_b = b_tail.grad.detach()
        b_server.backward(g_b.to(server.device))
        
        true_grad_norm = torch.norm(a_server.grad.detach(), p=2).item()
        norms.append(true_grad_norm)
        
    return np.mean(norms), np.std(norms)

def get_flat_weights(head, backbone, tail):
    tensors = []
    for param in head.parameters(): tensors.append(param.data.view(-1))
    for param in backbone.parameters(): tensors.append(param.data.view(-1))
    for param in tail.parameters(): tensors.append(param.data.view(-1))
    return torch.cat(tensors)

def set_flat_weights(head, backbone, tail, flat_weights):
    offset = 0
    for model in [head, backbone, tail]:
        for param in model.parameters():
            numel = param.numel()
            param.data.copy_(flat_weights[offset:offset+numel].view_as(param))
            offset += numel

def reset_optimizer_state(*optimizers):
    for opt in optimizers:
        if opt is not None:
            opt.state.clear()

def get_model_buffers(model):
    return {name: buffer.detach().clone().cpu() for name, buffer in model.named_buffers()}

def set_model_buffers(model, buffers):
    if buffers is None:
        return
    named_buffers = dict(model.named_buffers())
    for name, value in buffers.items():
        if name not in named_buffers:
            continue
        target = named_buffers[name]
        target.copy_(value.to(device=target.device, dtype=target.dtype))

def average_model_buffers(buffer_list, weights=None, eps=1e-8):
    if not buffer_list:
        return None
    if weights is None:
        weights = [1.0] * len(buffer_list)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    total_weight = max(float(weights_t.sum().item()), eps)
    max_weight_idx = int(torch.argmax(weights_t).item())

    averaged = {}
    for name in buffer_list[0]:
        first = buffer_list[0][name]
        if torch.is_floating_point(first):
            acc = torch.zeros_like(first, dtype=torch.float32)
            for buffers, weight in zip(buffer_list, weights_t):
                acc += buffers[name].float() * float(weight.item())
            averaged[name] = (acc / total_weight).to(dtype=first.dtype)
        else:
            averaged[name] = buffer_list[max_weight_idx][name].clone()
    return averaged

def blend_model_buffers(local_buffers, reference_buffers, local_coeff):
    if local_buffers is None:
        return reference_buffers
    if reference_buffers is None:
        return {name: value.clone() for name, value in local_buffers.items()}

    blended = {}
    for name, local_value in local_buffers.items():
        reference_value = reference_buffers.get(name)
        if reference_value is None:
            blended[name] = local_value.clone()
        elif torch.is_floating_point(local_value):
            blended[name] = (
                float(local_coeff) * local_value.float()
                + (1.0 - float(local_coeff)) * reference_value.float()
            ).to(dtype=local_value.dtype)
        else:
            blended[name] = local_value.clone()
    return blended

def prepare_client_from_global(client, server, global_flat, server_buffers=None):
    set_flat_weights(client.head, server.model, client.tail, global_flat)
    set_model_buffers(server.model, server_buffers)
    reset_optimizer_state(server.opt, client.opt_head, client.opt_tail)

def root_calibration_step(head, tail, server, root_loader, device, head_opt, tail_opt, server_opt, max_steps=1):
    head.train()
    tail.train()
    server.model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    steps = 0
    root_iter = iter(root_loader)

    for _ in range(max_steps):
        try:
            x, y = next(root_iter)
        except StopIteration:
            root_iter = iter(root_loader)
            x, y = next(root_iter)

        x, y = x.to(device), y.to(device)

        head_opt.zero_grad(set_to_none=True)
        tail_opt.zero_grad(set_to_none=True)
        server_opt.zero_grad(set_to_none=True)

        a = head(x)
        a_server = a.detach().clone().to(server.device).requires_grad_(True)
        b_server = server.model(a_server)
        b_tail = b_server.detach().to(device).requires_grad_(True)
        logits = tail(b_tail)
        loss = criterion(logits, y)

        loss.backward()
        g_b = b_tail.grad.detach()
        b_server.backward(g_b.to(server.device))
        g_a = a_server.grad.detach()
        torch.autograd.backward(a, g_a.to(device))

        server_opt.step()
        tail_opt.step()
        head_opt.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


def compute_cosine_score(gi_flat, history_updates):
    N = len(history_updates)
    if N == 0:
        return 0.0
    total_weight = sum(range(1, N + 1))
    weights = [k / total_weight for k in range(1, N + 1)]
    score = 0.0
    
    gi_flat_cpu = gi_flat.cpu()
    
    for k, h_update in enumerate(history_updates):
        sim = F.cosine_similarity(gi_flat_cpu.unsqueeze(0), h_update.unsqueeze(0)).item()
        score += weights[k] * sim
    return score

def compute_sparsity(gi_flat, threshold=1e-4):
    sparsity = (gi_flat.abs() < threshold).float().mean().item()
    return sparsity

def torch_dct_1d(x):
    n = x.shape[-1]
    if n == 0:
        return x

    y = torch.cat([x, torch.flip(x, dims=[-1])], dim=-1)
    fft_y = torch.fft.fft(y, dim=-1)[..., :n]
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    factor = torch.exp((-1j * np.pi / (2 * n)) * k.to(torch.complex64))
    result = (fft_y * factor).real

    result[..., 0] *= 1.0 / np.sqrt(4 * n)
    if n > 1:
        result[..., 1:] *= 1.0 / np.sqrt(2 * n)
    return result

def torch_dct_2d(x):
    x = torch_dct_1d(x)
    x = torch_dct_1d(x.transpose(-1, -2)).transpose(-1, -2)
    return x

def compute_dct_low_frequency_score(flat_update, low_freq_ratio=0.5):
    vec_t = flat_update.detach().float().cpu().view(-1)
    if vec_t.numel() == 0:
        return 0.0

    side = int(np.ceil(np.sqrt(vec_t.numel())))
    padded_t = torch.zeros(side * side, dtype=torch.float32)
    padded_t[:vec_t.numel()] = vec_t
    matrix_2d_t = padded_t.reshape(side, side)

    low_freq_size = int(np.ceil(side * low_freq_ratio))
    low_freq_size = max(1, min(side, low_freq_size))

    if scipy_fft is not None:
        dct_matrix = scipy_fft.dctn(matrix_2d_t.numpy(), norm='ortho')
        low_freq_components = dct_matrix[:low_freq_size, :low_freq_size]
        return float(np.sum(np.abs(low_freq_components)))

    dct_matrix_t = torch_dct_2d(matrix_2d_t)
    low_freq_components_t = dct_matrix_t[:low_freq_size, :low_freq_size]
    return float(torch.sum(torch.abs(low_freq_components_t)).item())


def build_fixed_probe_batches(loader, num_batches):
    """Cache a fixed trusted probe subset so all clients are compared on identical inputs."""
    probe_batches = []
    if num_batches <= 0:
        return probe_batches

    iterator = iter(loader)
    for _ in range(num_batches):
        try:
            x, y = next(iterator)
        except StopIteration:
            break
        probe_batches.append((x.detach().cpu(), y.detach().cpu()))
    return probe_batches

@torch.no_grad()
def compute_probe_gram_representation(head, probe_batches, device, orders=(1, 2), eps=1e-12):
    """Extract Beatrix-style high-order Gram features at the split-layer output."""
    if not probe_batches:
        return np.empty(0, dtype=float)

    was_training = head.training
    head.eval()
    batch_representations = []
    for x_cpu, _ in probe_batches:
        features = head(x_cpu.to(device))
        features = torch.clamp(features, min=0.0)
        batch_size, channels, _, _ = features.shape
        flattened = features.flatten(start_dim=2)
        spatial_size = max(1, flattened.shape[-1])
        upper_idx = torch.triu_indices(channels, channels, device=features.device)
        order_features = []

        for order in orders:
            powered = flattened.pow(int(order))
            gram = torch.bmm(powered, powered.transpose(1, 2)) / float(spatial_size)
            if int(order) > 1:
                gram = torch.clamp(gram, min=0.0).pow(1.0 / float(order))
            order_features.append(gram[:, upper_idx[0], upper_idx[1]])

        batch_representations.append(torch.cat(order_features, dim=1).mean(dim=0).cpu())

    if was_training:
        head.train()
    return torch.stack(batch_representations).mean(dim=0).numpy()

def robust_mad_deviation(value, references, mad_k=3.0, eps=1e-8):
    """Average relative distance outside a coordinate-wise median/MAD interval."""
    ref = np.asarray(references, dtype=float)
    value = np.asarray(value, dtype=float).reshape(-1)
    if ref.size == 0:
        return 0.0
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)

    median = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - median), axis=0)
    fallback_std = np.std(ref, axis=0)
    spread = np.where(mad > eps, mad, fallback_std)
    spread = np.maximum(spread, eps)
    margin = float(mad_k) * spread
    outside_distance = np.maximum(np.abs(value - median) - margin, 0.0)
    return float(np.mean(outside_distance / (margin + eps)))

def bootstrap_benign_deviations(references, mad_k=3.0, rounds=100, rng=None):
    """Estimate the benign deviation distribution with held-out bootstrap sampling."""
    ref = np.asarray(references, dtype=float)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    num_references = ref.shape[0] if ref.ndim == 2 else 0
    if num_references < 3:
        return np.zeros(max(1, int(rounds)), dtype=float)

    rng = np.random.default_rng() if rng is None else rng
    scores = np.zeros(max(1, int(rounds)), dtype=float)
    all_indices = np.arange(num_references)
    for b in range(len(scores)):
        holdout_idx = int(rng.integers(0, num_references))
        pool = ref[all_indices != holdout_idx]
        sampled_idx = rng.integers(0, len(pool), size=len(pool))
        bootstrap_reference = pool[sampled_idx]
        scores[b] = robust_mad_deviation(ref[holdout_idx], bootstrap_reference, mad_k=mad_k)
    return scores

def build_risk_calibrator(benign_deviations, upper_quantile=0.95, eps=1e-8):
    benign_deviations = np.asarray(benign_deviations, dtype=float)
    center = float(np.quantile(benign_deviations, 0.50))
    upper = float(np.quantile(benign_deviations, upper_quantile))
    return center, max(upper, center + eps)

def deviation_to_risk(deviation, calibrator):
    center, upper = calibrator
    return float(np.clip((float(deviation) - center) / max(upper - center, 1e-8), 0.0, 1.0))


def compute_fused_phase2_risks(feature_views, group_labels, weights, mad_k=3.0,
                                bootstrap_rounds=100, risk_quantile=0.95, rng=None):
    """Return per-client tri-view risks and group-specific bootstrap thresholds."""
    names = tuple(feature_views.keys())
    num_clients = len(group_labels)
    weights = np.asarray([weights[name] for name in names], dtype=float)
    weights = weights / max(float(weights.sum()), 1e-8)
    risks = np.zeros(num_clients, dtype=float)
    deviations = {name: np.zeros(num_clients, dtype=float) for name in names}
    component_risks = {name: np.zeros(num_clients, dtype=float) for name in names}
    thresholds = {}
    rng = np.random.default_rng() if rng is None else rng

    unique_groups = sorted(set(group_labels))
    all_ids = np.arange(num_clients)
    for group in unique_groups:
        group_ids = np.asarray([idx for idx, label in enumerate(group_labels) if label == group], dtype=int)
        reference_ids = group_ids if len(group_ids) >= 3 else all_ids
        position = {int(cid): idx for idx, cid in enumerate(reference_ids.tolist())}
        calibrators = {}
        bootstrap_deviations = {}

        for name in names:
            references = np.asarray(feature_views[name], dtype=float)[reference_ids]
            bootstrap_deviations[name] = bootstrap_benign_deviations(
                references, mad_k=mad_k, rounds=bootstrap_rounds, rng=rng,
            )
            calibrators[name] = build_risk_calibrator(
                bootstrap_deviations[name], upper_quantile=risk_quantile,
            )

        for cid in group_ids:
            for name in names:
                references = np.asarray(feature_views[name], dtype=float)[reference_ids]
                local_pos = position.get(int(cid))
                if local_pos is not None and len(references) > 2:
                    references = np.delete(references, local_pos, axis=0)
                deviation = robust_mad_deviation(
                    feature_views[name][cid], references, mad_k=mad_k,
                )
                deviations[name][cid] = deviation
                component_risks[name][cid] = deviation_to_risk(deviation, calibrators[name])

            risks[cid] = float(np.dot(
                weights,
                np.asarray([component_risks[name][cid] for name in names], dtype=float),
            ))

        bootstrap_component_risks = np.stack([
            np.asarray([
                deviation_to_risk(value, calibrators[name])
                for value in bootstrap_deviations[name]
            ], dtype=float)
            for name in names
        ], axis=1)
        fused_bootstrap = bootstrap_component_risks @ weights
        thresholds[group] = float(np.quantile(fused_bootstrap, risk_quantile))

    return risks, thresholds, component_risks, deviations

def get_kmeans_threshold(scores):
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return float('inf')

    mu1, mu2 = float(np.min(scores)), float(np.max(scores))
    if np.isclose(mu1, mu2):
        return mu2 + 1.0

    for _ in range(10):
        c1 = scores[np.abs(scores - mu1) <= np.abs(scores - mu2)]
        c2 = scores[np.abs(scores - mu1) > np.abs(scores - mu2)]
        new_mu1 = float(np.mean(c1)) if len(c1) > 0 else mu1
        new_mu2 = float(np.mean(c2)) if len(c2) > 0 else mu2
        if np.isclose(mu1, new_mu1) and np.isclose(mu2, new_mu2):
            break
        mu1, mu2 = new_mu1, new_mu2

    return (mu1 + mu2) / 2.0

def compute_phase2_threshold(scores, mode='mad', mad_beta=2.5, eps=1e-8):
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return float('inf'), {'median': 0.0, 'mad': 0.0, 'scale': 1.0}

    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    std = float(np.std(scores))
    scale = max(1.4826 * mad, std if mad < eps else 0.0, eps)

    if mode == 'kmeans':
        threshold = get_kmeans_threshold(scores)
    else:
        threshold = med + mad_beta * scale

    return float(threshold), {'median': med, 'mad': mad, 'scale': scale}

def compute_robust_confidence(values, eps=1e-8):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.asarray([], dtype=float), 0.0, 0.0

    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = max(1.4826 * mad, float(np.std(values)) if mad < eps else 0.0, eps)
    z = np.maximum(0.0, (values - med) / scale)
    alphas = np.exp(-0.5 * z ** 2)
    return alphas.astype(float), med, mad

def append_limited(buffer, item, max_size):
    buffer.append(item)
    if len(buffer) > max_size:
        buffer.pop(0)


def append_model_version(queue, flat_weights, feature_views, max_size):
    """Keep the latest global model snapshots and their Phase-2 feature views."""
    queue.append({
        "flat": flat_weights.detach().cpu().clone(),
        "views": {
            name: np.asarray(value, dtype=float).reshape(-1).copy()
            for name, value in feature_views.items()
        },
    })
    while len(queue) > max_size:
        queue.pop(0)


def range_outside_deviation(value, references, eps=1e-8):
    """Mean normalized distance of a vector outside a reference min/max box."""
    value = np.asarray(value, dtype=float).reshape(-1)
    references = np.asarray(references, dtype=float)
    if references.size == 0:
        return 0.0
    if references.ndim == 1:
        references = references.reshape(-1, 1)

    lower = np.min(references, axis=0)
    upper = np.max(references, axis=0)
    span = np.maximum(upper - lower, eps)
    outside = np.maximum(lower - value, 0.0) + np.maximum(value - upper, 0.0)
    return float(np.mean(outside / span))


def leave_one_out_range_deviations(values, eps=1e-8):
    """Estimate normal range excursions of historical versions with leave-one-out ranges."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] < 2:
        return np.zeros(values.shape[0], dtype=float)

    deviations = np.zeros(values.shape[0], dtype=float)
    for index in range(values.shape[0]):
        references = np.delete(values, index, axis=0)
        deviations[index] = range_outside_deviation(values[index], references, eps=eps)
    return deviations


def compute_history_range_phase2_risks(feature_views, model_history_queue, weights,
                                       risk_quantile=0.90, eps=1e-8):
    """Use the latest global versions as min/max references for Phase-2 risk scoring."""
    names = tuple(feature_views.keys())
    num_clients = len(next(iter(feature_views.values())))
    history_views = {
        name: np.stack([entry["views"][name] for entry in model_history_queue], axis=0)
        for name in names
    }

    normalized_weights = np.asarray([weights[name] for name in names], dtype=float)
    normalized_weights = normalized_weights / max(float(normalized_weights.sum()), eps)

    current_deviations = {
        name: np.zeros(num_clients, dtype=float) for name in names
    }
    current_component_risks = {
        name: np.zeros(num_clients, dtype=float) for name in names
    }
    historical_component_risks = []

    for name in names:
        history_values = history_views[name]
        historical_deviations = leave_one_out_range_deviations(history_values, eps=eps)
        center = float(np.quantile(historical_deviations, 0.50))
        upper = float(np.quantile(historical_deviations, risk_quantile))
        scale = max(upper - center, eps)

        historical_component_risks.append(
            np.clip((historical_deviations - center) / scale, 0.0, 1.0)
        )

        current_values = np.asarray(feature_views[name], dtype=float)
        for cid in range(num_clients):
            deviation = range_outside_deviation(
                current_values[cid], history_values, eps=eps,
            )
            current_deviations[name][cid] = deviation
            current_component_risks[name][cid] = float(
                np.clip((deviation - center) / scale, 0.0, 1.0)
            )

    historical_component_risks = np.stack(historical_component_risks, axis=1)
    historical_fused_risks = historical_component_risks @ normalized_weights
    threshold = float(np.quantile(historical_fused_risks, risk_quantile))
    current_risks = np.stack(
        [current_component_risks[name] for name in names], axis=1
    ) @ normalized_weights

    thresholds = {"others": threshold, "unseen": threshold}
    return current_risks, thresholds, current_component_risks, current_deviations

def custom_dbscan(X, eps, min_samples):
    UNCLASSIFIED = -2
    NOISE = -1
    n_samples = X.shape[0]
    labels = np.full(n_samples, UNCLASSIFIED)
    cluster_id = 0

    dist = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=-1)

    for i in range(n_samples):
        if labels[i] != UNCLASSIFIED:
            continue

        neighbors = np.where(dist[i] <= eps)[0]

        if len(neighbors) < min_samples:
            labels[i] = NOISE
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        i_seed = 0
        while i_seed < len(seed_set):
            q = seed_set[i_seed]
            if labels[q] == NOISE:
                labels[q] = cluster_id
            elif labels[q] == UNCLASSIFIED:
                labels[q] = cluster_id
                q_neighbors = np.where(dist[q] <= eps)[0]
                if len(q_neighbors) >= min_samples:
                    for n in q_neighbors:
                        if n not in seed_set:
                            seed_set.append(n)
            i_seed += 1
            
        cluster_id += 1

    return labels


def get_flat_model_weights(model):
    tensors = [param.data.detach().view(-1).cpu() for param in model.parameters()]
    return torch.cat(tensors) if tensors else torch.empty(0)

def split_flat_update(head, backbone, tail, flat_update):
    sizes = [sum(param.numel() for param in model.parameters()) for model in [head, backbone, tail]]
    head_end = sizes[0]
    backbone_end = sizes[0] + sizes[1]
    return (
        flat_update[:head_end],
        flat_update[head_end:backbone_end],
        flat_update[backbone_end:backbone_end + sizes[2]],
    )

def compute_peer_cosine_scores(updates):
    if len(updates) <= 1:
        return np.zeros(len(updates), dtype=float)
    update_tensor = torch.stack([u.float().view(-1) for u in updates])
    normed = F.normalize(update_tensor, p=2, dim=1, eps=1e-12)
    sim = torch.matmul(normed, normed.t())
    diag = torch.diag(sim)
    scores = (sim.sum(dim=1) - diag) / max(1, len(updates) - 1)
    return scores.cpu().numpy()

def compute_gradient_spa(update, eps=1e-12):
    vec = update.float().view(-1)
    if vec.numel() == 0:
        return 0.0
    l1 = torch.norm(vec, p=1).item()
    l2 = torch.norm(vec, p=2).item()
    return float(l1 / (np.sqrt(vec.numel()) * l2 + eps))

def haar_frequency_distribution(feature_map, eps=1e-12):
    x = feature_map.detach().float().cpu()
    if x.dim() == 4:
        x = x.mean(dim=0)
    if x.dim() == 2:
        x = x.unsqueeze(0)

    h = (x.shape[-2] // 2) * 2
    w = (x.shape[-1] // 2) * 2
    if h < 2 or w < 2:
        return np.ones(4, dtype=float) / 4.0

    x = x[..., :h, :w]
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5

    energies = torch.stack([band.pow(2).mean() for band in [ll, lh, hl, hh]]).numpy()
    energies = np.maximum(energies, eps)
    return energies / np.sum(energies)

def compute_lfd_scores(feature_distributions, eps=1e-12):
    distributions = np.asarray(feature_distributions, dtype=float)
    ref = np.median(distributions, axis=0)
    ref = np.maximum(ref, eps)
    ref = ref / np.sum(ref)

    scores = []
    for p in distributions:
        p = np.maximum(p, eps)
        p = p / np.sum(p)
        scores.append(float(np.sum(p * (np.log(p) - np.log(ref)))))
    return np.asarray(scores, dtype=float)

def robust_scale_metrics(metrics_np, eps=1e-8):
    metrics_np = np.asarray(metrics_np, dtype=float)
    med = np.median(metrics_np, axis=0)
    mad = np.median(np.abs(metrics_np - med), axis=0)
    std = np.std(metrics_np, axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale < eps, std, scale)
    scale = np.where(scale < eps, 1.0, scale)
    return (metrics_np - med) / scale


def robust_z_1d(values, eps=1e-8):
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    scale = max(1.4826 * mad, np.std(values), eps)
    return (values - med) / scale

def rank01(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(1, len(values) - 1)

def compute_channel_distribution_scores(features_tensor, eps=1e-12):
    features = torch.clamp(features_tensor.detach().float().cpu(), min=0.0) + eps
    distributions = features / features.sum(dim=1, keepdim=True)
    ref = torch.median(distributions, dim=0).values
    ref = ref / ref.sum()
    scores = (distributions * (torch.log(distributions) - torch.log(ref + eps))).sum(dim=1)
    return scores.numpy()

def build_phase1_unseen_scores(deep_cosine_scores, shallow_spa_scores, lfd_scores,
                               channel_shift_scores, channel_dist_scores, spike_scores):
    deep_anom = np.maximum(0.0, robust_z_1d(-deep_cosine_scores))
    spa_anom = np.abs(robust_z_1d(shallow_spa_scores))
    lfd_anom = np.maximum(0.0, robust_z_1d(lfd_scores))
    shift_anom = np.maximum(0.0, robust_z_1d(channel_shift_scores))
    dist_anom = np.maximum(0.0, robust_z_1d(channel_dist_scores))
    spike_anom = np.maximum(0.0, robust_z_1d(spike_scores))

    domain_score = 0.55 * rank01(dist_anom) + 0.45 * rank01(shift_anom)
    lfd_rank = rank01(lfd_anom)
    cosine_rank = rank01(deep_cosine_scores)
    spa_rank = rank01(spa_anom)

    # 适度提高域偏移权重，避免真 unseen 被 LFD 主导的排序压下去。
    score = 0.45 * lfd_rank + 0.35 * domain_score + 0.12 * cosine_rank + 0.08 * spa_rank
    return score, {
        'deep_anom': deep_anom,
        'spa_anom': spa_anom,
        'lfd_anom': lfd_anom,
        'shift_anom': shift_anom,
        'dist_anom': dist_anom,
        'spike_anom': spike_anom,
        'domain_score': domain_score,
        'lfd_rank': lfd_rank,
        'cosine_rank': cosine_rank,
        'spa_rank': spa_rank,
    }

def domain_gated_unseen_clusters(unseen_scores, domain_scores, expected_unseen, candidate_multiplier=1.5):
    num_clients = len(unseen_scores)
    k = int(max(0, min(expected_unseen, num_clients)))
    clusters = np.zeros(num_clients, dtype=int)
    if k == 0:
        return [], clusters, None, []

    candidate_count = int(np.ceil(k * max(1.0, candidate_multiplier)))
    candidate_count = int(max(k, min(num_clients, candidate_count)))
    domain_order = np.argsort(-np.asarray(domain_scores, dtype=float))
    candidate_ids = sorted(domain_order[:candidate_count].tolist())

    candidate_scores = np.asarray(unseen_scores, dtype=float)[candidate_ids]
    selected_local = np.argsort(-candidate_scores)[:k]
    predicted_unseen = sorted([candidate_ids[j] for j in selected_local])
    clusters[predicted_unseen] = 1
    threshold = float(np.min(np.asarray(unseen_scores)[predicted_unseen]))
    return predicted_unseen, clusters, threshold, candidate_ids

def confidence_from_unseen_scores(unseen_scores):
    z = np.maximum(0.0, robust_z_1d(unseen_scores))
    return np.exp(-0.5 * z ** 2)

def get_majority_cluster(clusters):
    valid = clusters[clusters >= 0]
    if len(valid) == 0:
        return None
    unique, counts = np.unique(valid, return_counts=True)
    return int(unique[np.argmax(counts)])

def compute_cluster_confidence_alphas(metrics_scaled, clusters, majority_cluster, eps=1e-8):
    if majority_cluster is None:
        return np.ones(len(clusters), dtype=float)

    main_mask = clusters == majority_cluster
    if not np.any(main_mask):
        return np.ones(len(clusters), dtype=float)

    centroid = metrics_scaled[main_mask].mean(axis=0)
    dists = np.linalg.norm(metrics_scaled - centroid, axis=1)
    ref_dists = dists[main_mask]
    med = np.median(ref_dists)
    mad = np.median(np.abs(ref_dists - med))
    spread = max(1.4826 * mad, np.std(ref_dists), eps)
    z = np.maximum(0.0, (dists - med) / spread)
    return np.exp(-0.5 * z ** 2)

def plot_phase1_three_metrics(metrics_np, clusters, alphas, unseen_ids, malicious_ids, epoch, save_dir="phase1_plots"):
    os.makedirs(save_dir, exist_ok=True)
    metrics_np = np.asarray(metrics_np, dtype=float)
    spa_vals = metrics_np[:, 1]
    spa_min, spa_max = float(np.min(spa_vals)), float(np.max(spa_vals))
    sizes = 80.0 + 180.0 * (spa_vals - spa_min) / (spa_max - spa_min + 1e-8)

    num_clients = len(metrics_np)
    benign_ids = [i for i in range(num_clients) if i not in unseen_ids and i not in malicious_ids]
    categories = [
        ('Benign (GT)', benign_ids, 'black', 'o'),
        ('Unseen (GT)', unseen_ids, 'blue', 'o'),
        ('Backdoor (GT)', malicious_ids, 'red', 'x'),
    ]

    plt.figure(figsize=(10, 7))
    scatter_ref = None
    for label, ids, edge_color, marker in categories:
        ids = [i for i in ids if 0 <= i < num_clients]
        if not ids:
            continue
        plot_edgecolors = 'none' if marker == 'x' else edge_color
        scatter_ref = plt.scatter(
            metrics_np[ids, 0], metrics_np[ids, 2],
            c=spa_vals[ids], cmap='viridis', vmin=spa_min, vmax=spa_max,
            s=sizes[ids], marker=marker, label=label,
            edgecolors=plot_edgecolors, linewidths=1.4, alpha=0.88,
        )

    for i, (cos_score, _, lfd_score) in enumerate(metrics_np):
        if i in malicious_ids:
            text_color = 'red'
        elif i in unseen_ids:
            text_color = 'blue'
        else:
            text_color = 'black'
        plt.annotate(
            f"C{i}(c{clusters[i]},a{alphas[i]:.2f})",
            (cos_score, lfd_score), textcoords="offset points", xytext=(0, 9),
            ha='center', fontsize=8, fontweight='bold', color=text_color,
        )

    if scatter_ref is not None:
        cbar = plt.colorbar(scatter_ref)
        cbar.set_label('Shallow Gradient Sparsity f_spa')

    plt.title(f'Phase 1 Three-Metric Clustering at Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.xlabel('Deep Gradient Peer Cosine Similarity', fontsize=12)
    plt.ylabel('Cut-layer Frequency Discrepancy (LFD)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    save_path = os.path.join(save_dir, f'phase1_three_metric_epoch_{epoch:03d}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  [Plot] Phase 1 三指标聚类图已保存至: {save_path}")



def plot_phase1_z_distributions(metrics_scaled, epoch, save_dir="phase1_z_plots"):
    """Plot robust Z1/Z2/Z3 empirical distributions against N(0, 1)."""
    os.makedirs(save_dir, exist_ok=True)
    z_values = np.asarray(metrics_scaled, dtype=float)
    if z_values.ndim != 2 or z_values.shape[1] != 3:
        raise ValueError('metrics_scaled must have shape [num_clients, 3]')

    metric_names = [
        'Z1: Deep cosine similarity',
        'Z2: Shallow gradient sparsity',
        'Z3: Cut-layer LFD',
    ]
    colors = ['#2563eb', '#059669', '#dc2626']
    finite_values = z_values[np.isfinite(z_values)]
    max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 3.5
    x_limit = max(3.5, min(8.0, max_abs + 0.6))
    x_grid = np.linspace(-x_limit, x_limit, 500)
    normal_pdf = np.exp(-0.5 * x_grid ** 2) / np.sqrt(2.0 * np.pi)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for idx, ax in enumerate(axes):
        values = z_values[:, idx]
        values = values[np.isfinite(values)]
        bins = max(6, min(10, int(np.ceil(np.sqrt(max(1, len(values)))) + 2)))

        ax.hist(
            values, bins=bins, density=True, color=colors[idx], alpha=0.30,
            edgecolor=colors[idx], linewidth=1.0, label='Empirical histogram',
        )

        if len(values) >= 2:
            sample_std = float(np.std(values, ddof=1))
            iqr = float(np.subtract(*np.percentile(values, [75, 25])))
            robust_sigma = iqr / 1.349 if iqr > 0 else sample_std
            bandwidth = 0.9 * min(sample_std, robust_sigma) * (len(values) ** -0.2)
            bandwidth = max(0.20, min(1.0, bandwidth if np.isfinite(bandwidth) else 0.35))
            kernel = np.exp(-0.5 * ((x_grid[:, None] - values[None, :]) / bandwidth) ** 2)
            kde = kernel.mean(axis=1) / (bandwidth * np.sqrt(2.0 * np.pi))
            ax.plot(x_grid, kde, color=colors[idx], linewidth=2.2, label='Gaussian KDE')
        else:
            sample_std = 0.0

        ax.plot(x_grid, normal_pdf, color='#111827', linestyle='--', linewidth=1.8, label='Standard normal N(0,1)')
        ax.axvline(0.0, color='#6b7280', linestyle=':', linewidth=1.2)
        ax.axvline(-1.96, color='#9ca3af', linestyle=':', linewidth=1.0)
        ax.axvline(1.96, color='#9ca3af', linestyle=':', linewidth=1.0)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_title(metric_names[idx], fontsize=11)
        ax.set_xlabel('Robust Z value')
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.text(
            0.03, 0.95,
            f'n={len(values)}\nmean={np.mean(values):.2f}\nstd={sample_std:.2f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='#d1d5db', alpha=0.88),
        )

    axes[0].set_ylabel('Density')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f'Phase 1 Robust Z Distributions at Epoch {epoch}', fontsize=14, fontweight='bold', y=1.08)
    fig.tight_layout()
    save_path = os.path.join(save_dir, f'z123_distribution_epoch_{epoch:03d}.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=180)
    plt.close(fig)
    print(f"  [Plot] Z1/Z2/Z3 distribution saved to: {save_path}")


def plot_dbscan_results(metrics_np, clusters, epoch, num_clients=10, save_dir="dbscan_plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 7))
    
    unique_clusters = set(clusters)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_clusters))]
    
    for k, col in zip(unique_clusters, colors):
        if k == -1:
            col = [0, 0, 0, 1]
            marker = 'x'
            label = 'Outliers (-1)'
            size = 120
        else:
            marker = 'o'
            label = f'Cluster {k} (Benign)'
            size = 100
        
        class_member_mask = (clusters == k)
        xy = metrics_np[class_member_mask]
        
        if len(xy) > 0:
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, s=size, label=label, edgecolors='k' if k != -1 else None)
    
    # === 核心修改：区分历史点与当前点，并还原真实 Client ID ===
    total_points = len(metrics_np)
    for i, (x, y) in enumerate(metrics_np):
        real_client_id = i % num_clients
        # 如果是最后 num_clients 个点，说明是当前轮次的最新点
        if i >= total_points - num_clients:
            text_label = f"C{real_client_id}"
            font_weight = 'bold'
            font_color = 'red' if clusters[i] == -1 else 'blue'  # 当前轮次加醒目颜色
            font_size = 12
        else:
            # 历史点标记为灰色和小写，避免视觉干扰
            text_label = f"h{real_client_id}"
            font_weight = 'normal'
            font_color = 'gray'
            font_size = 8

        plt.annotate(text_label, (x, y), textcoords="offset points", xytext=(0, 10), 
                     ha='center', fontsize=font_size, fontweight=font_weight, color=font_color)

    plt.title(f'DBSCAN Clustering at Epoch {epoch}\n(Score vs Sparsity |15 Sliding Window)', fontsize=14)
    plt.xlabel('Cosine Similarity Score (Weighted)', fontsize=12)
    plt.ylabel('Gradient Sparsity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    
    save_path = os.path.join(save_dir, f'dbscan_epoch_{epoch:03d}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  [Plot] DBSCAN 二维可视化已保存至: {save_path}")
    
def main(config_path, args):
    cfg = load_config(config_path)
    if args.epochs is not None: cfg['epochs'] = args.epochs
    if args.steps_per_client is not None: cfg['steps_per_client'] = args.steps_per_client
    if args.batch_size is not None: cfg['batch_size'] = args.batch_size
    
    # 强制将每轮客户端修改为10，或者通过传参传入10
    #0604
    #cfg['num_clients'] = args.num_clients if args.num_clients is not None else 10
    cfg['num_clients'] = args.num_clients if args.num_clients is not None else 30

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    loaders, test_loader, china_test_loader, root_loader, new_data_clients = build_dataloaders(
        cfg, num_new_clients=args.num_new_clients, unseen=args.unseen, root_size=args.root_size,
        exact_unseen_sampling=args.exact_unseen_sampling,
    )

    device_str = cfg['server'].get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    HeadCls, BackboneCls, TailCls = get_model_classes(args.model)

    head0, back, tail0 = HeadCls().to(device), BackboneCls().to(device), TailCls().to(device)
    server = ServerBackbone(backbone=back, lr=cfg['server']['lr'], device=device)
    root_head_opt = torch.optim.SGD(head0.parameters(), lr=args.root_calib_lr, momentum=0.0)
    root_tail_opt = torch.optim.SGD(tail0.parameters(), lr=args.root_calib_lr, momentum=0.0)
    root_server_opt = torch.optim.SGD(server.model.parameters(), lr=args.root_calib_lr, momentum=0.0)

    clients = []

    for i in range(cfg['num_clients']):
        clients.append(
            Client(
                i,
                HeadCls(),
                TailCls(),
                loaders[i],
                lr_head=cfg['clients']['lr_head'],
                lr_tail=cfg['clients']['lr_tail'],
                device=device,
            )
        )

    malicious_clients = []
    attack_hook = None
    if args.enable_attack:
        # malicious_clients = np.random.choice(cfg['num_clients'], args.num_malicious, replace=False).tolist()
        # print(f"[Attack] 恶意客户端 IDs: {malicious_clients}, 攻击类型: {args.attack_type}")
        
        # if args.attack_type == 'trigger':
        #     attack_hook = make_backdoor_hook(malicious_clients, p=args.poison_rate, target_label=args.target_label, size=args.trigger_size)
        # elif args.attack_type == 'labelflip':
        #     attack_hook = make_label_flip_hook(malicious_clients, target_label=args.target_label, p=args.poison_rate)
        # 从非 unseen 客户端中随机挑选恶意节点，避免身份重叠污染 Phase 1 评估。
        candidate_clients = [i for i in range(cfg['num_clients']) if i not in set(new_data_clients)]
        if args.num_malicious > len(candidate_clients):
            raise ValueError('num_malicious 不能超过非 unseen 客户端数量')
        malicious_clients = np.random.choice(candidate_clients, args.num_malicious, replace=False).tolist()
        
        print(f"[Attack] 【随机生成】恶意客户端 IDs: {malicious_clients}, 攻击类型: {args.attack_type}")
        
        if args.attack_type == 'trigger':
            attack_hook = make_backdoor_hook(malicious_clients, p=args.poison_rate, target_label=args.target_label, size=args.trigger_size)
        elif args.attack_type == 'labelflip':
            attack_hook = make_label_flip_hook(malicious_clients, target_label=args.target_label, p=args.poison_rate)

    global_flat = get_flat_weights(head0, server.model, tail0)
    global_server_buffers = get_model_buffers(server.model)

    if args.root_pretrain_steps > 0:
        pre_head_opt = torch.optim.SGD(head0.parameters(), lr=args.root_pretrain_lr, momentum=0.9)
        pre_tail_opt = torch.optim.SGD(tail0.parameters(), lr=args.root_pretrain_lr, momentum=0.9)
        pre_server_opt = torch.optim.SGD(server.model.parameters(), lr=args.root_pretrain_lr, momentum=0.9)
        pretrain_loss = root_calibration_step(
            head0, tail0, server, root_loader, device,
            pre_head_opt, pre_tail_opt, pre_server_opt,
            max_steps=args.root_pretrain_steps,
        )
        global_flat = get_flat_weights(head0, server.model, tail0).detach()
        global_server_buffers = get_model_buffers(server.model)
        print(f"[RootPretrain] steps={args.root_pretrain_steps} lr={args.root_pretrain_lr} loss={pretrain_loss:.4f}")

    model_history_queue = []
    history_model_size = max(1, int(args.phase2_history_size))
    history_min_versions = max(
        2, min(int(args.phase2_history_min_versions), history_model_size)
    )
    history_last_flat = global_flat.detach().clone()
    if args.phase2_history_range:
        print(
            f"[Phase2] Historical model-range mode enabled: "
            f"queue_size={history_model_size}, min_versions={history_min_versions}"
        )
    
    # ================= 核心修改区域：初始化全局滑动窗口 =================
    HISTORY_WINDOW_SIZE = 15
    history_updates = []
    global_metrics_buffer = [] # 存储最近若干个 DCT 分数，便于后续诊断扩展
    
    phase2_ema_scores = {i: None for i in range(cfg['num_clients'])}
    phase2_ema_alpha = min(max(args.phase2_ema_alpha, 0.0), 0.99)
    phase1_update_mix = min(max(args.phase1_update_mix, 0.0), 1.0)
    phase2_rng = np.random.default_rng(cfg['seed'] + 20260813)
    phase2_probe_batches = build_fixed_probe_batches(root_loader, args.phase2_probe_batches)
    if phase2_probe_batches:
        print(f"[Phase2] Fixed trusted probe batches: {len(phase2_probe_batches)}")
    else:
        print("[Phase2] Warning: no trusted probe batch is available; Gram risk will be zero.")

    for ep in range(cfg['epochs']):
        client_losses = []
        print(f"\nEpoch {ep+1}/{cfg['epochs']}")

        # # ================= Phase 1: 获取高维特征并计算 Feature-DDifs =================
        # client_features = []
        # for i in range(cfg['num_clients']):
        #     set_flat_weights(clients[i].head, server.model, clients[i].tail, global_flat)
        #     # 在前向传播中，自动收集当前客户端的平均通道特征
        #     stats = clients[i].run_batches(server, max_steps=cfg['steps_per_client'], attack_hook=attack_hook)
        #     client_features.append(stats['phase1_features'])

        # # 将所有客户端的特征堆叠，形状为 [num_clients, Channels]
        # features_tensor = torch.stack(client_features)

        # # 1. 动态中位数锚点 (Dynamic Median Anchor)
        # # 通过当前轮次的多数派共识提取健康的全局基准，无需额外依赖预设的 Root Dataset
        # pseudo_global_anchor = torch.median(features_tensor, dim=0).values

        # # 2. 计算除法差异 (Feature-DDifs)
        # # 异常数据(如TT100K)唤醒的休眠通道将在这里产生极端的数值爆炸
        # ddifs_tensor = features_tensor / (pseudo_global_anchor + 1e-8)


        # # # ------------------- 【核心修改区：改变特征评估维度】 -------------------
        # # # 旧逻辑：只看最高点 (会被后门的刺带偏)
        # # # max_ddifs_ratios = ddifs_tensor.max(dim=1).values.numpy()

        # # # 新逻辑：计算所有 32 个通道偏离 1.0 (健康基准线) 的平均绝对偏移量 (MAE)
        # # # 这种方式对“大面积轻微偏移”的 Unseen 数据极其敏感！
        # # shift_scores = torch.abs(ddifs_tensor - 1.0).mean(dim=1).numpy()

        # # # 4. 使用绝对中位差 (MAD) 对整体偏移量进行稳健评估
        # # median_shift = np.median(shift_scores)
        # # mad_shift = np.median(np.abs(shift_scores - median_shift))
        # # mad_shift = max(mad_shift, 1e-4) # 防止除零

        # # phase1_alphas = []
        # # for i in range(cfg['num_clients']):
        # #     score_i = shift_scores[i]
            
        # #     # 只有当该客户端的整体偏移量，大于场上的中位数时，才开始计算惩罚
        # #     if score_i > median_shift:
        # #         # 放大 Z-score 的敏感度，使得 Unseen 数据被迅速锁定
        # #         z_score = (score_i - median_shift) / (1.4826 * mad_shift)
        # #     else:
        # #         z_score = 0.0

        # #     # 映射为 Alpha 得分
        # #     alpha_i = np.exp(-0.5 * (max(0, z_score)**2))
        # #     phase1_alphas.append(alpha_i)
            
        # #     print(f"  Client {i:02d} | Avg Channel Shift: {score_i:6.3f} (Z: {z_score:5.2f}) -> Alpha: {alpha_i:.4f}")

        # #0604
        # # 【新增】：调用我们刚刚写的画图函数，直观展示三者的区别
        # plot_channel_ddifs(ddifs_tensor, new_data_clients, malicious_clients, ep+1)

        # # 3. 提取各个客户端在所有通道中的“最大异常倍数”
        # max_ddifs_ratios = ddifs_tensor.max(dim=1).values.numpy()

        # 4. 使用绝对中位差 (MAD) 计算稳健的置信度得分 Alpha
        # median_ratio = np.median(max_ddifs_ratios)
        # mad_ratio = np.median(np.abs(max_ddifs_ratios - median_ratio))
        # mad_ratio = max(mad_ratio, 1e-4)

        # phase1_alphas = []
        # for i in range(cfg['num_clients']):
        #     ratio_i = max_ddifs_ratios[i]
        #     # 仅对激活倍率远超中位数的客户端进行惩罚
        #     if ratio_i > median_ratio:
        #         z_score = (ratio_i - median_ratio) / (1.4826 * mad_ratio)
        #     else:
        #         z_score = 0.0

        #     alpha_i = np.exp(-0.5 * (max(0, z_score)**2))
        #     phase1_alphas.append(alpha_i)
        #     print(f"  Client {i:02d} | Max DDifs Ratio: {ratio_i:6.2f}x (Z: {z_score:5.2f}) -> Alpha: {alpha_i:.4f}")

        # # Phase 1 评估报告逻辑保持不变
        # THRESHOLD_ALPHA = 0.1 
        # predicted_unseen = []
        # for cid, alpha in enumerate(phase1_alphas):
        #     if alpha < THRESHOLD_ALPHA:
        #         predicted_unseen.append(cid)
                
        # true_unseen_set = set(new_data_clients)
        # pred_unseen_set = set(predicted_unseen)
        # true_positives = len(true_unseen_set.intersection(pred_unseen_set))
        # false_positives = len(pred_unseen_set - true_unseen_set)
        # recall = true_positives / max(1, len(true_unseen_set))
        # fpr = false_positives / max(1, cfg['num_clients'] - len(true_unseen_set))
        
        # print("\n  >>> [Phase 1 评估报告 (Feature-DDifs)] <<<")
        # print(f"  真实的 Unseen 客户端: {new_data_clients}")
        # print(f"  Phase 1 预测出的 Unseen: {predicted_unseen}")
        # print(f"  召回率 (Recall): {recall * 100:.1f}% | 误报率 (FPR): {fpr * 100:.1f}%")
        # print("  " + "="*30 + "\n")
        
        # # base_mu, base_std = compute_phase1_baseline(head0, tail0, server, root_loader, device)
        # # base_std = max(base_std, 1e-6) 
        # # print(f"  [Phase 1] 正常梯度基准 -> Mu: {base_mu:.4f}, Std: {base_std:.4f}")
        
        # client_updates = []
        # # phase1_alphas = [] 

        # # for i in range(cfg['num_clients']):
        # #     set_flat_weights(clients[i].head, server.model, clients[i].tail, global_flat)
        # #     stats = clients[i].run_batches(server, max_steps=cfg['steps_per_client'], attack_hook=attack_hook)
            
        # #     norm_i = stats['phase1_norm']
        # #     z_score = (norm_i - base_mu) / base_std
            
        # #     alpha_i = np.exp(-0.5 * (max(0, z_score)**2)) 
        # #     phase1_alphas.append(alpha_i)
        # #     print(f"  Client {i:02d} | Phase1 Norm: {norm_i:.4f} (Z: {z_score:.2f}) -> Alpha: {alpha_i:.4f}")

        # # THRESHOLD_ALPHA = 0.1 
        # # predicted_unseen = []
        # # for cid, alpha in enumerate(phase1_alphas):
        # #     if alpha < THRESHOLD_ALPHA:
        # #         predicted_unseen.append(cid)
                
        # # true_unseen_set = set(new_data_clients)
        # # pred_unseen_set = set(predicted_unseen)
        # # true_positives = len(true_unseen_set.intersection(pred_unseen_set))
        # # false_positives = len(pred_unseen_set - true_unseen_set)
        # # recall = true_positives / max(1, len(true_unseen_set))
        # # fpr = false_positives / max(1, cfg['num_clients'] - len(true_unseen_set))
        
        # # print("\n  >>> [Phase 1 评估报告] <<<")
        # # print(f"  真实的 Unseen 客户端: {new_data_clients}")
        # # print(f"  Phase 1 预测出的 Unseen: {predicted_unseen}")
        # # print(f"  召回率 (Recall): {recall * 100:.1f}% | 误报率 (FPR): {fpr * 100:.1f}%")
        # # print("  " + "="*30 + "\n")

        # ================= Phase 1: 三指标聚类，分离 unseen / others =================
        client_features = []
        client_feature_maps = []
        phase1_head_updates = []
        phase1_deep_updates = []
        phase1_full_updates = []
        global_flat_cpu = global_flat.detach().cpu()

        for i in range(cfg['num_clients']):
            prepare_client_from_global(clients[i], server, global_flat, global_server_buffers)
            stats = clients[i].run_batches(server, max_steps=cfg['steps_per_client'], attack_hook=attack_hook)

            client_features.append(stats['phase1_features'])
            client_feature_maps.append(stats['phase1_feature_map'])

            gi_flat = get_flat_weights(clients[i].head, server.model, clients[i].tail).detach().cpu() - global_flat_cpu
            phase1_full_updates.append(gi_flat.clone())
            head_delta, backbone_delta, tail_delta = split_flat_update(clients[i].head, server.model, clients[i].tail, gi_flat)
            phase1_head_updates.append(head_delta)
            phase1_deep_updates.append(torch.cat([backbone_delta, tail_delta]))

        # 继续维护原有的通道锚点，用于后续 Active Masking 和诊断图。
        features_tensor = torch.stack(client_features)
        pseudo_global_anchor = torch.median(features_tensor, dim=0).values
        server.global_anchor = pseudo_global_anchor.to(device)
        ddifs_tensor = features_tensor / (pseudo_global_anchor + 1e-8)
        plot_channel_ddifs(ddifs_tensor, new_data_clients, malicious_clients, ep+1)

        deep_cosine_scores = compute_peer_cosine_scores(phase1_deep_updates)
        shallow_spa_scores = np.array([compute_gradient_spa(update) for update in phase1_head_updates], dtype=float)
        lfd_distributions = [haar_frequency_distribution(feature_map) for feature_map in client_feature_maps]
        lfd_scores = compute_lfd_scores(lfd_distributions)
        channel_shift_scores = torch.abs(ddifs_tensor - 1.0).mean(dim=1).numpy()
        channel_dist_scores = compute_channel_distribution_scores(features_tensor)
        spike_scores = ddifs_tensor.max(dim=1).values.numpy()

        phase1_metrics_np = np.column_stack((deep_cosine_scores, shallow_spa_scores, lfd_scores))
        phase1_metrics_scaled = robust_scale_metrics(phase1_metrics_np)
        plot_phase1_z_distributions(phase1_metrics_scaled, ep + 1)
        dbscan_clusters = custom_dbscan(
            phase1_metrics_scaled,
            eps=args.phase1_eps,
            min_samples=args.phase1_min_samples,
        )

        unseen_scores, unseen_components = build_phase1_unseen_scores(
            deep_cosine_scores,
            shallow_spa_scores,
            lfd_scores,
            channel_shift_scores,
            channel_dist_scores,
            spike_scores,
        )
        predicted_unseen, phase1_clusters, unseen_threshold, phase1_candidates = domain_gated_unseen_clusters(
            unseen_scores,
            unseen_components['domain_score'],
            expected_unseen=args.num_new_clients,
            candidate_multiplier=args.phase1_candidate_multiplier,
        )
        others_ids = [cid for cid in range(cfg['num_clients']) if cid not in predicted_unseen]
        phase1_alphas = confidence_from_unseen_scores(unseen_scores)

        print("\n  [Phase 1] --- 三指标融合聚类审查: deep-cosine / shallow-fspa / cut-layer-LFD ---")
        print(f"  [Phase 1] DBSCAN诊断簇: {dbscan_clusters.tolist()}")
        print(f"  [Phase 1] Domain-gated candidates: {phase1_candidates}")
        print(f"  [Phase 1] Score-assisted unseen threshold: {unseen_threshold if unseen_threshold is not None else 'None'} | Predicted Unseen IDs: {predicted_unseen}")
        for i in range(cfg['num_clients']):
            pred_label = "Unseen" if i in predicted_unseen else "Others"
            print(
                f"    Client {i:02d} | Score: {unseen_scores[i]:6.3f} | Cos: {deep_cosine_scores[i]:7.4f} | "
                f"F_spa: {shallow_spa_scores[i]:7.4f} | LFD: {lfd_scores[i]:8.5f} | "
                f"Shift: {channel_shift_scores[i]:7.4f} | ChKL: {channel_dist_scores[i]:8.5f} | "
                f"Spike: {spike_scores[i]:6.2f} | Cluster {phase1_clusters[i]:2d} ({pred_label}) -> Alpha={phase1_alphas[i]:.4f}"
            )

        plot_phase1_three_metrics(
            phase1_metrics_np,
            phase1_clusters,
            phase1_alphas,
            new_data_clients,
            malicious_clients,
            ep+1,
        )

        # ================= Phase 1 效果自动评估 =================
        true_unseen_set = set(new_data_clients)
        pred_unseen_set = set(predicted_unseen)
        true_positives = len(true_unseen_set.intersection(pred_unseen_set))
        false_positives = len(pred_unseen_set - true_unseen_set)
        recall = true_positives / max(1, len(true_unseen_set))
        fpr = false_positives / max(1, cfg['num_clients'] - len(true_unseen_set))

        print(f"  >>> [Phase 1 评估结果] Unseen 召回率: {recall * 100:.1f}% | 误报率: {fpr * 100:.1f}%")
        print("  " + "="*40 + "\n")

        # ================= Phase 2: DCT low-frequency update audit =================
        current_round_updates = []
        current_round_dct_scores = []
        current_round_dct_raw_scores = []
        current_round_update_norms = []
        current_round_server_buffers = []
        current_round_dct_features = []
        current_round_history_dct_features = []
        current_round_l2_features = []
        current_round_gram_features = []

        for i in range(cfg['num_clients']):
            prepare_client_from_global(clients[i], server, global_flat, global_server_buffers)
            stats = clients[i].run_batches(server, max_steps=cfg['steps_per_client'], attack_hook=attack_hook)

            gi_flat = get_flat_weights(clients[i].head, server.model, clients[i].tail) - global_flat

            if args.enable_attack and i in malicious_clients:
                gi_flat = gi_flat * args.attack_scale

            update_cpu = gi_flat.clone().detach().cpu()
            dct_raw = compute_dct_low_frequency_score(update_cpu, low_freq_ratio=args.phase2_dct_ratio)
            if phase2_ema_scores[i] is None:
                phase2_ema_scores[i] = dct_raw
            else:
                phase2_ema_scores[i] = phase2_ema_alpha * phase2_ema_scores[i] + (1 - phase2_ema_alpha) * dct_raw
            dct_score = float(phase2_ema_scores[i])
            update_norm = float(torch.norm(update_cpu.float(), p=2).item())
            dct_feature = np.asarray([np.log1p(dct_score)], dtype=float)
            l2_feature = np.asarray([np.log1p(update_norm)], dtype=float)
            gram_feature = compute_probe_gram_representation(
                clients[i].head, phase2_probe_batches, device,
                orders=tuple(args.phase2_gram_orders),
            )
            if gram_feature.size == 0:
                gram_feature = np.zeros(1, dtype=float)

            client_losses.append(stats['loss'])
            current_round_updates.append(update_cpu)
            current_round_dct_raw_scores.append(dct_raw)
            current_round_dct_scores.append(dct_score)
            current_round_update_norms.append(update_norm)
            current_round_server_buffers.append(get_model_buffers(server.model))
            current_round_dct_features.append(dct_feature)
            current_round_history_dct_features.append(
                np.asarray([np.log1p(dct_raw)], dtype=float)
            )
            current_round_l2_features.append(l2_feature)
            current_round_gram_features.append(gram_feature)

            print(
                f"  Client {i:02d} | steps={stats['steps']} loss={stats['loss']:.4f} | "
                f"DCT-low(raw/ema)={dct_raw:.4f}/{dct_score:.4f} | ||g||={update_norm:.4f}"
            )

        # ================= PHASE2_LEGACY_DCT_ONLY_BEGIN =================
        # The previous DCT-only Phase2 decision is retained below for manual rollback.
        # To restore it, replace the active tri-view block with these lines after removing
        # the leading "# " from each line.
#         # ================= Final decision: Phase1 x Phase2 =================
#         if args.enable_defense:
#             phase2_reference_ids = others_ids if len(others_ids) >= 2 else list(range(cfg['num_clients']))
#             reference_scores = [current_round_dct_scores[cid] for cid in phase2_reference_ids]
#             phase2_threshold, threshold_stats = compute_phase2_threshold(
#                 reference_scores,
#                 mode=args.phase2_threshold_mode,
#                 mad_beta=args.phase2_mad_beta,
#             )
#             phase2_backdoor_ids = [
#                 cid for cid in range(cfg['num_clients'])
#                 if current_round_dct_scores[cid] > phase2_threshold
#             ]
#             phase2_benign_ids = [cid for cid in range(cfg['num_clients']) if cid not in phase2_backdoor_ids]
#
#             phase2_confidence, phase2_score_median, phase2_score_mad = compute_robust_confidence(current_round_dct_scores)
#             norm_alphas, norm_median, norm_mad = compute_robust_confidence(current_round_update_norms)
#
#             def get_reference_weights(candidate_ids, dtype=torch.float32):
#                 return torch.tensor(
#                     [max(1e-4, float(phase2_confidence[cid]) * float(norm_alphas[cid])) for cid in candidate_ids],
#                     dtype=dtype,
#                 )
#
#             def build_weighted_reference(candidate_ids):
#                 if not candidate_ids:
#                     return None
#                 ref_stack = torch.stack([current_round_updates[cid] for cid in candidate_ids])
#                 ref_weights = get_reference_weights(candidate_ids, dtype=ref_stack.dtype)
#                 ref_delta = (ref_stack * ref_weights.view(-1, *([1] * (ref_stack.dim() - 1)))).sum(dim=0)
#                 return ref_delta / max(float(ref_weights.sum().item()), 1e-8)
#
#             def build_weighted_reference_buffers(candidate_ids):
#                 if not candidate_ids:
#                     return None
#                 ref_weights = get_reference_weights(candidate_ids)
#                 ref_buffers = [current_round_server_buffers[cid] for cid in candidate_ids]
#                 return average_model_buffers(ref_buffers, ref_weights.tolist())
#
#             clean_reference_ids = [
#                 cid for cid in range(cfg['num_clients'])
#                 if cid not in predicted_unseen and cid not in phase2_backdoor_ids
#             ]
#             reference_source_ids = clean_reference_ids
#             reference_update = build_weighted_reference(reference_source_ids)
#             reference_server_buffers = build_weighted_reference_buffers(reference_source_ids)
#             if reference_update is None:
#                 reference_source_ids = phase2_benign_ids
#                 reference_update = build_weighted_reference(reference_source_ids)
#                 reference_server_buffers = build_weighted_reference_buffers(reference_source_ids)
#             if reference_update is None:
#                 reference_source_ids = list(range(cfg['num_clients']))
#                 reference_update = build_weighted_reference(reference_source_ids)
#                 reference_server_buffers = build_weighted_reference_buffers(reference_source_ids)
#
#             accepted_updates = []
#             accepted_weights = []
#             accepted_server_buffers = []
#             filtered_backdoor_ids = []
#             suspicious_unseen_ids = []
#             benign_kept_ids = []
#             unseen_benign_ids = []
#
#             print(
#                 "\n  [Defense] --- Phase 2 DCT 低频参数变化判定 ---"
#                 f"\n  [Defense] Threshold({args.phase2_threshold_mode})={phase2_threshold:.4f} | "
#                 f"Median={threshold_stats['median']:.4f} | MAD={threshold_stats['mad']:.4f} | "
#                 f"RefIDs={phase2_reference_ids}"
#             )
#             print(f"  [Defense] Phase2 Backdoor IDs: {phase2_backdoor_ids} | Phase2 Benign IDs: {phase2_benign_ids}")
#             print(f"  [Defense] Phase2 confidence baseline: Median={phase2_score_median:.4f} | MAD={phase2_score_mad:.4f}")
#             print(f"  [Defense] Norm alpha baseline: Median={norm_median:.4f} | MAD={norm_mad:.4f}")
#             print(f"  [Defense] Weighted clean reference IDs: {reference_source_ids} | Phase1Mix={phase1_update_mix:.2f}")
#
#             for cid in range(cfg['num_clients']):
#                 is_unseen = cid in predicted_unseen
#                 is_backdoor = cid in phase2_backdoor_ids
#                 alpha_i = float(norm_alphas[cid])
#                 phase2_trust = float(phase2_confidence[cid])
#                 base_update = (1.0 - phase1_update_mix) * current_round_updates[cid] + phase1_update_mix * phase1_full_updates[cid]
#
#                 display_weight = 0.0
#                 display_blend = 0.0
#                 if (not is_unseen) and is_backdoor:
#                     final_action = "FILTER: others+backdoor"
#                     filtered_backdoor_ids.append(cid)
#                 elif is_unseen and is_backdoor:
#                     blend_coeff = alpha_i * phase2_trust
#                     update_star = (1.0 - blend_coeff) * reference_update + blend_coeff * base_update
#                     buffer_star = blend_model_buffers(current_round_server_buffers[cid], reference_server_buffers, blend_coeff)
#                     client_weight = max(0.05, 0.10 + 0.20 * phase2_trust)
#                     accepted_updates.append(update_star)
#                     accepted_weights.append(client_weight)
#                     accepted_server_buffers.append(buffer_star)
#                     display_weight = client_weight
#                     display_blend = blend_coeff
#                     suspicious_unseen_ids.append(cid)
#                     final_action = f"BLEND: unseen+backdoor alpha={alpha_i:.4f}"
#                     append_limited(history_updates, update_star, HISTORY_WINDOW_SIZE)
#                     append_limited(global_metrics_buffer, [current_round_dct_scores[cid]], HISTORY_WINDOW_SIZE)
#                 elif (not is_unseen) and (not is_backdoor):
#                     blend_coeff = phase2_trust
#                     update_star = blend_coeff * base_update + (1.0 - blend_coeff) * reference_update
#                     buffer_star = blend_model_buffers(current_round_server_buffers[cid], reference_server_buffers, blend_coeff)
#                     client_weight = 1.0
#                     accepted_updates.append(update_star)
#                     accepted_weights.append(client_weight)
#                     accepted_server_buffers.append(buffer_star)
#                     display_weight = client_weight
#                     display_blend = blend_coeff
#                     benign_kept_ids.append(cid)
#                     final_action = "KEEP: others+benign"
#                     append_limited(history_updates, update_star, HISTORY_WINDOW_SIZE)
#                     append_limited(global_metrics_buffer, [current_round_dct_scores[cid]], HISTORY_WINDOW_SIZE)
#                 else:
#                     blend_coeff = max(0.65, phase2_trust)
#                     update_star = blend_coeff * current_round_updates[cid] + (1.0 - blend_coeff) * reference_update
#                     buffer_star = blend_model_buffers(current_round_server_buffers[cid], reference_server_buffers, blend_coeff)
#                     client_weight = 1.0 + 0.25 * (1.0 - phase1_alphas[cid])
#                     accepted_updates.append(update_star)
#                     accepted_weights.append(client_weight)
#                     accepted_server_buffers.append(buffer_star)
#                     display_weight = client_weight
#                     display_blend = blend_coeff
#                     unseen_benign_ids.append(cid)
#                     final_action = "KEEP: unseen+benign"
#                     append_limited(history_updates, update_star, HISTORY_WINDOW_SIZE)
#                     append_limited(global_metrics_buffer, [current_round_dct_scores[cid]], HISTORY_WINDOW_SIZE)
#
#                 phase1_label = "unseen" if is_unseen else "others"
#                 phase2_label = "backdoor" if is_backdoor else "benign"
#                 print(
#                     f"    Client {cid:02d} | Phase1={phase1_label:<6} | Phase2={phase2_label:<8} | "
#                     f"DCT={current_round_dct_scores[cid]:10.4f} | ||g||={current_round_update_norms[cid]:9.4f} | "
#                     f"alpha={alpha_i:.4f} | trust={phase2_trust:.4f} | blend={display_blend:.3f} | w={display_weight:.3f} | {final_action}"
#                 )
#
#             print(f"  [Defense] Filtered others+backdoor IDs: {filtered_backdoor_ids}")
#             print(f"  [Defense] Suspicious unseen blended IDs: {suspicious_unseen_ids}")
#             print(f"  [Defense] Kept benign IDs: {benign_kept_ids}")
#             print(f"  [Defense] Kept unseen-benign IDs: {unseen_benign_ids}")
#
#             if accepted_updates:
#                 weights_t = torch.tensor(accepted_weights, dtype=accepted_updates[0].dtype)
#                 stacked_updates = torch.stack(accepted_updates)
#                 agg_delta = (stacked_updates * weights_t.view(-1, *([1] * (stacked_updates.dim() - 1)))).sum(dim=0)
#                 agg_delta = (agg_delta / max(float(weights_t.sum().item()), 1e-8)).to(device)
#                 aggregated_server_buffers = average_model_buffers(accepted_server_buffers, accepted_weights)
#             else:
#                 print("  [Warning] No accepted updates, skip this round.")
#                 agg_delta = torch.zeros_like(global_flat).to(device)
#                 aggregated_server_buffers = global_server_buffers
#         else:
#             agg_delta = torch.stack(current_round_updates).mean(dim=0).to(device)
#             aggregated_server_buffers = average_model_buffers(current_round_server_buffers)
#             for cid in range(cfg['num_clients']):
#                 append_limited(history_updates, current_round_updates[cid], HISTORY_WINDOW_SIZE)
#                 append_limited(global_metrics_buffer, [current_round_dct_scores[cid]], HISTORY_WINDOW_SIZE)
#
        # ================= PHASE2_LEGACY_DCT_ONLY_END =================

        # ================= Active Phase 2: DCT + L2 + Gram risk fusion =================
        if args.enable_defense:
            group_labels = ["unseen" if cid in predicted_unseen else "others" for cid in range(cfg['num_clients'])]
            feature_views = {
                "dct": np.stack(current_round_dct_features, axis=0),
                "l2": np.stack(current_round_l2_features, axis=0),
                "gram": np.stack(current_round_gram_features, axis=0),
            }
            risk_weights = {
                "dct": args.phase2_dct_weight,
                "l2": args.phase2_l2_weight,
                "gram": args.phase2_gram_weight,
            }
            use_history_range = (
                args.phase2_history_range
                and len(model_history_queue) >= history_min_versions
            )
            if use_history_range:
                history_feature_views = {
                    "dct": np.stack(current_round_history_dct_features, axis=0),
                    "l2": np.stack(current_round_l2_features, axis=0),
                    "gram": np.stack(current_round_gram_features, axis=0),
                }
                phase2_risks, phase2_thresholds, phase2_component_risks, phase2_deviations = compute_history_range_phase2_risks(
                    history_feature_views,
                    model_history_queue,
                    risk_weights,
                    risk_quantile=args.phase2_risk_quantile,
                )
            else:
                phase2_risks, phase2_thresholds, phase2_component_risks, phase2_deviations = compute_fused_phase2_risks(
                    feature_views,
                    group_labels,
                    risk_weights,
                    mad_k=args.phase2_mad_k,
                    bootstrap_rounds=args.phase2_bootstrap_rounds,
                    risk_quantile=args.phase2_risk_quantile,
                    rng=phase2_rng,
                )
            phase2_backdoor_ids = [
                cid for cid in range(cfg['num_clients'])
                if phase2_risks[cid] > phase2_thresholds[group_labels[cid]]
            ]
            phase2_benign_ids = [cid for cid in range(cfg['num_clients']) if cid not in phase2_backdoor_ids]
            norm_alphas, norm_median, norm_mad = compute_robust_confidence(current_round_update_norms)

            def get_reference_weights(candidate_ids, dtype=torch.float32):
                return torch.tensor(
                    [max(1e-4, 1.0 - float(phase2_risks[cid])) for cid in candidate_ids],
                    dtype=dtype,
                )

            def build_weighted_reference(candidate_ids):
                if not candidate_ids:
                    return None
                ref_stack = torch.stack([current_round_updates[cid] for cid in candidate_ids])
                ref_weights = get_reference_weights(candidate_ids, dtype=ref_stack.dtype)
                ref_delta = (ref_stack * ref_weights.view(-1, *([1] * (ref_stack.dim() - 1)))).sum(dim=0)
                return ref_delta / max(float(ref_weights.sum().item()), 1e-8)

            def build_weighted_reference_buffers(candidate_ids):
                if not candidate_ids:
                    return None
                ref_weights = get_reference_weights(candidate_ids)
                ref_buffers = [current_round_server_buffers[cid] for cid in candidate_ids]
                return average_model_buffers(ref_buffers, ref_weights.tolist())

            clean_reference_ids = [
                cid for cid in range(cfg['num_clients'])
                if cid not in predicted_unseen and cid not in phase2_backdoor_ids
            ]
            reference_source_ids = clean_reference_ids
            reference_update = build_weighted_reference(reference_source_ids)
            reference_server_buffers = build_weighted_reference_buffers(reference_source_ids)
            if reference_update is None:
                reference_source_ids = phase2_benign_ids
                reference_update = build_weighted_reference(reference_source_ids)
                reference_server_buffers = build_weighted_reference_buffers(reference_source_ids)
            if reference_update is None:
                reference_source_ids = list(range(cfg['num_clients']))
                reference_update = build_weighted_reference(reference_source_ids)
                reference_server_buffers = build_weighted_reference_buffers(reference_source_ids)

            accepted_updates = []
            accepted_weights = []
            accepted_server_buffers = []
            filtered_backdoor_ids = []
            suspicious_unseen_ids = []
            benign_kept_ids = []
            unseen_benign_ids = []

            normalized_weights = np.asarray(list(risk_weights.values()), dtype=float)
            normalized_weights = normalized_weights / max(float(normalized_weights.sum()), 1e-8)
            phase2_threshold_source = (
                "historical min/max range"
                if use_history_range else "current-round bootstrap"
            )
            print(
                f"\n  [Defense] --- Phase 2 tri-view robust risk fusion "
                f"({phase2_threshold_source}) ---"
            )
            print(
                f"  [Defense] Risk weights DCT/L2/Gram="
                f"{normalized_weights[0]:.2f}/{normalized_weights[1]:.2f}/{normalized_weights[2]:.2f} | "
                f"MAD-k={args.phase2_mad_k:.2f} | bootstrap={args.phase2_bootstrap_rounds} | "
                f"history={len(model_history_queue)}/{history_model_size}"
            )
            print(
                f"  [Defense] Group thresholds: others={phase2_thresholds.get('others', 0.0):.4f} | "
                f"unseen={phase2_thresholds.get('unseen', 0.0):.4f}"
            )
            print(f"  [Defense] Phase2 Backdoor IDs: {phase2_backdoor_ids} | Phase2 Benign IDs: {phase2_benign_ids}")
            print(f"  [Defense] Norm alpha baseline: Median={norm_median:.4f} | MAD={norm_mad:.4f}")
            print(f"  [Defense] Weighted clean reference IDs: {reference_source_ids} | Phase1Mix={phase1_update_mix:.2f}")

            for cid in range(cfg['num_clients']):
                is_unseen = cid in predicted_unseen
                is_backdoor = cid in phase2_backdoor_ids
                alpha_i = float(norm_alphas[cid])
                risk_i = float(phase2_risks[cid])
                threshold_i = float(phase2_thresholds[group_labels[cid]])
                suspicious_trust = max(0.0, 1.0 - risk_i)
                benign_trust = max(args.phase2_benign_trust_floor, suspicious_trust)
                base_update = (
                    (1.0 - phase1_update_mix) * current_round_updates[cid]
                    + phase1_update_mix * phase1_full_updates[cid]
                )

                display_weight = 0.0
                display_blend = 0.0
                if (not is_unseen) and is_backdoor:
                    final_action = "FILTER: others+backdoor"
                    filtered_backdoor_ids.append(cid)
                elif is_unseen and is_backdoor:
                    blend_coeff = alpha_i * suspicious_trust
                    update_star = (1.0 - blend_coeff) * reference_update + blend_coeff * base_update
                    buffer_star = blend_model_buffers(current_round_server_buffers[cid], reference_server_buffers, blend_coeff)
                    client_weight = max(0.05, 0.10 + 0.20 * suspicious_trust)
                    accepted_updates.append(update_star)
                    accepted_weights.append(client_weight)
                    accepted_server_buffers.append(buffer_star)
                    display_weight = client_weight
                    display_blend = blend_coeff
                    suspicious_unseen_ids.append(cid)
                    final_action = "BLEND: unseen+backdoor"
                    append_limited(history_updates, update_star, HISTORY_WINDOW_SIZE)
                    append_limited(global_metrics_buffer, [risk_i], HISTORY_WINDOW_SIZE)
                elif (not is_unseen) and (not is_backdoor):
                    blend_coeff = benign_trust
                    update_star = blend_coeff * base_update + (1.0 - blend_coeff) * reference_update
                    buffer_star = blend_model_buffers(current_round_server_buffers[cid], reference_server_buffers, blend_coeff)
                    client_weight = 1.0
                    accepted_updates.append(update_star)
                    accepted_weights.append(client_weight)
                    accepted_server_buffers.append(buffer_star)
                    display_weight = client_weight
                    display_blend = blend_coeff
                    benign_kept_ids.append(cid)
                    final_action = "KEEP: others+benign"
                    append_limited(history_updates, update_star, HISTORY_WINDOW_SIZE)
                    append_limited(global_metrics_buffer, [risk_i], HISTORY_WINDOW_SIZE)
                else:
                    blend_coeff = max(0.65, benign_trust)
                    update_star = blend_coeff * current_round_updates[cid] + (1.0 - blend_coeff) * reference_update
                    buffer_star = blend_model_buffers(current_round_server_buffers[cid], reference_server_buffers, blend_coeff)
                    client_weight = 1.0 + 0.25 * (1.0 - phase1_alphas[cid])
                    accepted_updates.append(update_star)
                    accepted_weights.append(client_weight)
                    accepted_server_buffers.append(buffer_star)
                    display_weight = client_weight
                    display_blend = blend_coeff
                    unseen_benign_ids.append(cid)
                    final_action = "KEEP: unseen+benign"
                    append_limited(history_updates, update_star, HISTORY_WINDOW_SIZE)
                    append_limited(global_metrics_buffer, [risk_i], HISTORY_WINDOW_SIZE)

                phase1_label = "unseen" if is_unseen else "others"
                phase2_label = "backdoor" if is_backdoor else "benign"
                print(
                    f"    Client {cid:02d} | Phase1={phase1_label:<6} | Phase2={phase2_label:<8} | "
                    f"DCT/L2/Gram={phase2_component_risks['dct'][cid]:.3f}/"
                    f"{phase2_component_risks['l2'][cid]:.3f}/"
                    f"{phase2_component_risks['gram'][cid]:.3f} | "
                    f"Risk={risk_i:.4f} > Th={threshold_i:.4f} | "
                    f"blend={display_blend:.3f} | w={display_weight:.3f} | {final_action}"
                )

            print(f"  [Defense] Filtered others+backdoor IDs: {filtered_backdoor_ids}")
            print(f"  [Defense] Suspicious unseen blended IDs: {suspicious_unseen_ids}")
            print(f"  [Defense] Kept benign IDs: {benign_kept_ids}")
            print(f"  [Defense] Kept unseen-benign IDs: {unseen_benign_ids}")

            if accepted_updates:
                weights_t = torch.tensor(accepted_weights, dtype=accepted_updates[0].dtype)
                stacked_updates = torch.stack(accepted_updates)
                agg_delta = (stacked_updates * weights_t.view(-1, *([1] * (stacked_updates.dim() - 1)))).sum(dim=0)
                agg_delta = (agg_delta / max(float(weights_t.sum().item()), 1e-8)).to(device)
                aggregated_server_buffers = average_model_buffers(accepted_server_buffers, accepted_weights)
            else:
                print("  [Warning] No accepted updates, skip this round.")
                agg_delta = torch.zeros_like(global_flat).to(device)
                aggregated_server_buffers = global_server_buffers
        else:
            agg_delta = torch.stack(current_round_updates).mean(dim=0).to(device)
            aggregated_server_buffers = average_model_buffers(current_round_server_buffers)
            for cid in range(cfg['num_clients']):
                append_limited(history_updates, current_round_updates[cid], HISTORY_WINDOW_SIZE)
                append_limited(global_metrics_buffer, [current_round_dct_scores[cid]], HISTORY_WINDOW_SIZE)

        global_flat += args.aggregation_delta * agg_delta
        set_flat_weights(head0, server.model, tail0, global_flat)
        set_model_buffers(server.model, aggregated_server_buffers)
        global_server_buffers = get_model_buffers(server.model)

        if args.root_calib_steps > 0:
            root_calib_loss = root_calibration_step(
                head0, tail0, server, root_loader, device,
                root_head_opt, root_tail_opt, root_server_opt,
                max_steps=args.root_calib_steps,
            )
            global_flat = get_flat_weights(head0, server.model, tail0).detach()
            global_server_buffers = get_model_buffers(server.model)
            print(f"  [RootCalib] steps={args.root_calib_steps} loss={root_calib_loss:.4f}")

        if args.phase2_history_range:
            history_delta = (
                global_flat.detach().cpu() - history_last_flat.detach().cpu()
            )
            history_dct = np.asarray([
                np.log1p(
                    compute_dct_low_frequency_score(
                        history_delta,
                        low_freq_ratio=args.phase2_dct_ratio,
                    )
                )
            ], dtype=float)
            history_l2 = np.asarray([
                np.log1p(torch.norm(history_delta.float(), p=2).item())
            ], dtype=float)
            history_gram = compute_probe_gram_representation(
                head0,
                phase2_probe_batches,
                device,
                orders=tuple(args.phase2_gram_orders),
            )
            if history_gram.size == 0:
                history_gram = np.zeros(1, dtype=float)

            append_model_version(
                model_history_queue,
                global_flat,
                {
                    "dct": history_dct,
                    "l2": history_l2,
                    "gram": history_gram,
                },
                history_model_size,
            )
            history_last_flat = global_flat.detach().clone()
            print(
                f"  [Phase2] Saved global model version to history queue: "
                f"{len(model_history_queue)}/{history_model_size}"
            )


        # ---------------- 3. Eval ----------------
        acc = eval_clean_accuracy(head0, tail0, server, test_loader, device=device)
        china_acc = eval_unseen_accuracy(head0, tail0, server, china_test_loader, device=device)

        if args.enable_attack and args.attack_type == 'trigger':
            asr_strict = eval_asr_strict(head0, tail0, server, test_loader, target_label=args.target_label, trigger_size=args.trigger_size, device=device)
            asr_msg = f"{asr_strict:.4f}"
        else:
            asr_msg = "N/A"
        print(f"\n  [Eval] MA: {acc:.4f} | Unseen Accuracy: {china_acc:.4f} | ASR: {asr_msg} | Defense={'ON' if args.enable_defense else 'OFF'}")

     
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/gtsrb.yaml') 
    ap.add_argument('--model', type=str, default='resnet_gtsrb', choices=['resnet_gtsrb', 'simple_cnn', 'resnet18', 'resnet34', 'googlenet', 'vgg11', 'wide_resnet50', 'micronnet'], help='GTSRB backbone architecture')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--steps-per-client', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    
    # 攻击相关：当前实验默认开启攻击，可用 --disable-attack 关闭。
    ap.add_argument('--enable-attack', dest='enable_attack', action='store_true', default=True)
    ap.add_argument('--disable-attack', dest='enable_attack', action='store_false')
    ap.add_argument('--num-malicious', type=int, default=4, help='恶意客户端的数量')
    ap.add_argument('--attack-type', type=str, default='trigger', choices=['trigger', 'labelflip'])
    ap.add_argument('--poison-rate', type=float, default=0.8)
    ap.add_argument('--target-label', type=int, default=0)
    ap.add_argument('--trigger-size', type=int, default=6)
    ap.add_argument('--attack-scale', type=float, default=1.0, help='恶意梯度的放大倍数(建议设为5.0以上抵抗FedAvg稀释)')

    # 领域偏移 (新数据) 相关
    ap.add_argument('--num-new-clients', type=int, default=4, help='多少个客户端含有新数据')
    ap.add_argument('--unseen', type=float, default=0.7, help='新数据(TT100K)在这些客户端中的占比')
    ap.add_argument('--exact-unseen-sampling', action='store_true',
                    help='按请求数量精确采样 unseen 数据，适用于大客户端数实验')

    # Phase 1 三指标聚类相关
    ap.add_argument('--phase1-eps', type=float, default=1.2, help='Phase 1 三指标DBSCAN半径')
    ap.add_argument('--phase1-min-samples', type=int, default=2, help='Phase 1 三指标DBSCAN最小样本数')
    ap.add_argument('--phase1-candidate-multiplier', type=float, default=1.5, help='Phase 1 领域偏移候选集合大小 = unseen数量 * 该倍率')

    # 防御相关
    ap.add_argument('--enable-defense', action='store_true', help='开启 Phase1 x Phase2 DCT 低频过滤防御')
    ap.add_argument('--dbscan-eps', type=float, default=0.5, help='保留旧DBSCAN绘图/实验参数，当前DCT流程不使用')
    ap.add_argument('--sparsity-threshold', type=float, default=1e-4, help='保留旧稀疏率实验参数，当前DCT流程不使用')
    ap.add_argument('--phase2-dct-ratio', type=float, default=0.5, help='Phase 2 DCT 左上低频块占边长比例')
    ap.add_argument('--phase2-threshold-mode', type=str, default='mad', choices=['mad', 'kmeans'], help='Phase 2 DCT 分数阈值模式')
    ap.add_argument('--phase2-mad-beta', type=float, default=2.5, help='Phase 2 MAD 阈值系数')
    ap.add_argument('--phase2-ema-alpha', type=float, default=0.0, help='保留旧DCT流程的EMA参数')
    ap.add_argument('--phase2-probe-batches', type=int, default=1, help='用于Gram表征的固定可信root probe batch数量')
    ap.add_argument('--phase2-gram-orders', type=int, nargs='+', default=[1, 2], help='Beatrix式Gram表征阶数，例如: --phase2-gram-orders 1 2')
    ap.add_argument('--phase2-history-range', action='store_true', help='使用最近全局模型版本队列的min/max范围计算Phase 2偏离度和阈值')
    ap.add_argument('--phase2-history-size', type=int, default=15, help='Phase 2全局模型版本队列大小')
    ap.add_argument('--phase2-history-min-versions', type=int, default=3, help='启用历史min/max阈值所需的最少历史版本数')
    ap.add_argument('--phase2-mad-k', type=float, default=1.5, help='三视角偏离度的MAD正常区间系数')
    ap.add_argument('--phase2-bootstrap-rounds', type=int, default=100, help='正常风险分布的bootstrap次数')
    ap.add_argument('--phase2-risk-quantile', type=float, default=0.90, help='融合风险阈值的bootstrap分位点')
    ap.add_argument('--phase2-dct-weight', type=float, default=1.0, help='DCT风险融合权重')
    ap.add_argument('--phase2-l2-weight', type=float, default=1.0, help='L2风险融合权重')
    ap.add_argument('--phase2-gram-weight', type=float, default=1.0, help='Gram风险融合权重')
    ap.add_argument('--phase2-benign-trust-floor', type=float, default=0.65, help='通过Phase2的客户端最小保留系数，保护MA')
    ap.add_argument('--phase1-update-mix', type=float, default=1.0, help='最终聚合中混入 Phase1 探测更新的比例')
    ap.add_argument('--aggregation-delta', type=float, default=1.0, help='全局更新步长 delta')
    ap.add_argument('--root-size', type=int, default=1000, help='可信root数据集大小，用于干净warmup/calibration')
    ap.add_argument('--root-pretrain-steps', type=int, default=500, help='联邦训练前使用root_loader做的干净预训练步数')
    ap.add_argument('--root-pretrain-lr', type=float, default=0.01, help='root clean warmup 学习率')
    ap.add_argument('--root-calib-steps', type=int, default=0, help='每轮使用 root_loader 做的干净校准步数')
    ap.add_argument('--root-calib-lr', type=float, default=0.003, help='root calibration 学习率')

    ap.add_argument('--num-clients', type=int, default=None, help='总客户端数量（覆盖yaml里的设置）')

    args = ap.parse_args()
    main(args.config, args)