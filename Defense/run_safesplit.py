# run_safesplit.py
"""SafeSplit-style baseline for GTSRB split learning.

This runner is intentionally independent from run_gtsrb.py's current defense.
It implements the SafeSplit idea: sequential U-shaped split learning with a
server-side rollback check over recent model checkpoints. The check combines a
static DCT-frequency analysis and a dynamic rotational-distance analysis on the
server backbone parameters.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from attacks.backdoor_gtsrb import make_backdoor_hook, make_label_flip_hook
from eval.metrics import eval_asr_strict, eval_clean_accuracy, eval_unseen_accuracy
from models.resnet_gtsrb import Backbone, Head, Tail
from run_gtsrb import build_dataloaders, load_config
from sl_core.client import Client
from sl_core.server import ServerBackbone


@dataclass
class ModelCheckpoint:
    index: int
    epoch: int
    client_id: Optional[int]
    head_state: Dict[str, torch.Tensor]
    backbone_state: Dict[str, torch.Tensor]
    tail_state: Dict[str, torch.Tensor]
    backbone_vec: torch.Tensor
    backbone_delta: torch.Tensor


@dataclass
class SafeSplitDecision:
    selected_history_pos: int
    selected_checkpoint: ModelCheckpoint
    skipped_client_ids: List[int]
    frequency_scores: np.ndarray
    rotation_scores: np.ndarray
    frequency_majority: List[int]
    rotation_majority: List[int]
    benign_majority: List[int]


def clone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_state_dict_cpu(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def reset_optimizer_state(*optimizers: Optional[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        if optimizer is not None:
            optimizer.state.clear()


def backbone_parameter_vector(backbone: torch.nn.Module) -> torch.Tensor:
    params = [param.detach().float().cpu().view(-1) for param in backbone.parameters()]
    return torch.cat(params) if params else torch.empty(0, dtype=torch.float32)


def make_checkpoint(index: int, epoch: int, client_id: Optional[int], head: torch.nn.Module,
                    server: ServerBackbone, tail: torch.nn.Module,
                    previous_backbone_vec: Optional[torch.Tensor]) -> ModelCheckpoint:
    backbone_vec = backbone_parameter_vector(server.model)
    if previous_backbone_vec is None or previous_backbone_vec.numel() != backbone_vec.numel():
        backbone_delta = torch.zeros_like(backbone_vec)
    else:
        backbone_delta = backbone_vec - previous_backbone_vec
    return ModelCheckpoint(
        index=index,
        epoch=epoch,
        client_id=client_id,
        head_state=clone_state_dict(head),
        backbone_state=clone_state_dict(server.model),
        tail_state=clone_state_dict(tail),
        backbone_vec=backbone_vec,
        backbone_delta=backbone_delta,
    )


def load_checkpoint_to_client(checkpoint: ModelCheckpoint, client: Client, server: ServerBackbone) -> None:
    load_state_dict_cpu(client.head, checkpoint.head_state)
    load_state_dict_cpu(server.model, checkpoint.backbone_state)
    load_state_dict_cpu(client.tail, checkpoint.tail_state)
    reset_optimizer_state(client.opt_head, client.opt_tail, server.opt)


def load_checkpoint_to_models(checkpoint: ModelCheckpoint, head: torch.nn.Module,
                              server: ServerBackbone, tail: torch.nn.Module) -> None:
    load_state_dict_cpu(head, checkpoint.head_state)
    load_state_dict_cpu(server.model, checkpoint.backbone_state)
    load_state_dict_cpu(tail, checkpoint.tail_state)


def torch_dct_1d(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n == 0:
        return x
    y = torch.cat([x, torch.flip(x, dims=[-1])], dim=-1)
    fft_y = torch.fft.fft(y, dim=-1)[..., :n]
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    factor = torch.exp((-1j * math.pi / (2 * n)) * k.to(torch.complex64))
    result = (fft_y * factor).real
    result[..., 0] *= 1.0 / math.sqrt(4 * n)
    if n > 1:
        result[..., 1:] *= 1.0 / math.sqrt(2 * n)
    return result


def torch_dct_2d(x: torch.Tensor) -> torch.Tensor:
    x = torch_dct_1d(x)
    x = torch_dct_1d(x.transpose(-1, -2)).transpose(-1, -2)
    return x


def dct_low_frequency_vector(update: torch.Tensor, ratio: float) -> torch.Tensor:
    vec = update.detach().float().cpu().view(-1)
    if vec.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    side = int(math.ceil(math.sqrt(vec.numel())))
    padded = torch.zeros(side * side, dtype=torch.float32)
    padded[:vec.numel()] = vec
    matrix = padded.view(side, side)
    low_size = int(math.ceil(side * float(ratio)))
    low_size = max(1, min(side, low_size))
    dct_matrix = torch_dct_2d(matrix)
    return dct_matrix[:low_size, :low_size].contiguous().view(-1)


def pairwise_euclidean(features: Sequence[torch.Tensor]) -> np.ndarray:
    if not features:
        return np.zeros((0, 0), dtype=float)
    stacked = torch.stack([f.float().view(-1) for f in features])
    return torch.cdist(stacked, stacked, p=2).cpu().numpy()


def smallest_majority_sum(distances: np.ndarray) -> np.ndarray:
    n = distances.shape[0]
    if n == 0:
        return np.asarray([], dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float)
    k = min(n - 1, n // 2 + 1)
    work = np.asarray(distances, dtype=float).copy()
    np.fill_diagonal(work, np.inf)
    sorted_distances = np.sort(work, axis=1)
    return sorted_distances[:, :k].sum(axis=1)


def compute_frequency_scores(window: Sequence[ModelCheckpoint], dct_ratio: float) -> np.ndarray:
    dct_features = [dct_low_frequency_vector(checkpoint.backbone_delta, dct_ratio) for checkpoint in window]
    return smallest_majority_sum(pairwise_euclidean(dct_features))


def compute_rotation_scores(window: Sequence[ModelCheckpoint], eps: float = 1e-15) -> np.ndarray:
    if not window:
        return np.asarray([], dtype=float)
    vectors = torch.stack([checkpoint.backbone_vec.double().view(-1) for checkpoint in window])
    norms = torch.norm(vectors, p=2, dim=1, keepdim=True).clamp_min(eps)
    normalized = vectors / norms
    cosine = torch.matmul(normalized, normalized.t()).clamp(-1.0, 1.0)
    angles = torch.arccos(cosine).cpu().numpy()
    angular_neighborhood = smallest_majority_sum(angles)

    rotation = np.zeros_like(angular_neighborhood, dtype=float)
    if len(rotation) > 1:
        rotation[1:] = np.abs(np.diff(angular_neighborhood)) / (2.0 * math.pi)
        rotation[0] = float(np.median(rotation[1:]))
    return rotation


def analyze_safesplit(history: Sequence[ModelCheckpoint], history_size: int,
                      min_history: int, dct_ratio: float) -> SafeSplitDecision:
    if len(history) < max(2, min_history):
        current_pos = len(history) - 1
        return SafeSplitDecision(
            selected_history_pos=current_pos,
            selected_checkpoint=history[current_pos],
            skipped_client_ids=[],
            frequency_scores=np.zeros(len(history), dtype=float),
            rotation_scores=np.zeros(len(history), dtype=float),
            frequency_majority=list(range(len(history))),
            rotation_majority=list(range(len(history))),
            benign_majority=list(range(len(history))),
        )

    start = max(0, len(history) - history_size)
    window = list(history[start:])
    frequency_scores = compute_frequency_scores(window, dct_ratio=dct_ratio)
    rotation_scores = compute_rotation_scores(window)
    majority_size = len(window) // 2 + 1

    frequency_majority_local = np.argsort(frequency_scores)[:majority_size].tolist()
    rotation_majority_local = np.argsort(rotation_scores)[:majority_size].tolist()
    benign_majority_local = sorted(set(frequency_majority_local).intersection(rotation_majority_local))

    if not benign_majority_local:
        combined_rank = np.argsort(np.argsort(frequency_scores)) + np.argsort(np.argsort(rotation_scores))
        selected_local = int(np.argmin(combined_rank))
        benign_majority_local = [selected_local]
    else:
        selected_local = max(benign_majority_local)

    selected_history_pos = start + selected_local
    selected_checkpoint = history[selected_history_pos]
    skipped_client_ids = [
        checkpoint.client_id for checkpoint in history[selected_history_pos + 1:]
        if checkpoint.client_id is not None
    ]

    return SafeSplitDecision(
        selected_history_pos=selected_history_pos,
        selected_checkpoint=selected_checkpoint,
        skipped_client_ids=skipped_client_ids,
        frequency_scores=frequency_scores,
        rotation_scores=rotation_scores,
        frequency_majority=[start + idx for idx in frequency_majority_local],
        rotation_majority=[start + idx for idx in rotation_majority_local],
        benign_majority=[start + idx for idx in benign_majority_local],
    )


def clean_train_steps(head: torch.nn.Module, tail: torch.nn.Module, server: ServerBackbone,
                      loader, device: torch.device, lr: float, steps: int) -> float:
    if steps <= 0:
        return 0.0
    head.train()
    tail.train()
    server.model.train()
    criterion = nn.CrossEntropyLoss()
    head_opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)
    tail_opt = torch.optim.SGD(tail.parameters(), lr=lr, momentum=0.9)
    server_opt = torch.optim.SGD(server.model.parameters(), lr=lr, momentum=0.9)
    iterator = iter(loader)
    total_loss = 0.0
    actual_steps = 0

    for _ in range(steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
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

        b_server.backward(b_tail.grad.detach().to(server.device))
        torch.autograd.backward(a, a_server.grad.detach().to(device))

        server_opt.step()
        tail_opt.step()
        head_opt.step()
        total_loss += float(loss.item())
        actual_steps += 1

    return total_loss / max(1, actual_steps)


def client_identity(client_id: int, unseen_clients: Sequence[int], malicious_clients: Sequence[int]) -> str:
    if client_id in set(malicious_clients):
        return "Backdoor"
    if client_id in set(unseen_clients):
        return "Unseen"
    return "Benign"


def run_safesplit(config_path: str, args: argparse.Namespace) -> None:
    cfg = load_config(config_path)
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.steps_per_client is not None:
        cfg['steps_per_client'] = args.steps_per_client
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.num_clients is not None:
        cfg['num_clients'] = args.num_clients
    if args.iid_rate is not None:
        cfg['iid_rate'] = args.iid_rate
    if args.seed is not None:
        cfg['seed'] = args.seed

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    loaders, test_loader, china_test_loader, root_loader, unseen_clients = build_dataloaders(
        cfg,
        num_new_clients=args.num_new_clients,
        unseen=args.unseen,
        root_size=args.root_size,
    )
    unseen_clients = [int(cid) for cid in unseen_clients]

    device_str = cfg['server'].get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    eval_head = Head().to(device)
    eval_tail = Tail().to(device)
    server = ServerBackbone(backbone=Backbone().to(device), lr=cfg['server']['lr'], device=device)

    if args.root_pretrain_steps > 0:
        warmup_loss = clean_train_steps(
            eval_head, eval_tail, server, root_loader, device,
            lr=args.root_pretrain_lr,
            steps=args.root_pretrain_steps,
        )
        print(f"[Warmup] clean root steps={args.root_pretrain_steps} lr={args.root_pretrain_lr} loss={warmup_loss:.4f}")

    clients = [
        Client(
            cid,
            Head(),
            Tail(),
            loaders[cid],
            lr_head=cfg['clients']['lr_head'],
            lr_tail=cfg['clients']['lr_tail'],
            device=device,
        )
        for cid in range(cfg['num_clients'])
    ]

    malicious_clients: List[int] = []
    attack_hook = None
    if args.enable_attack:
        if args.allow_malicious_unseen_overlap:
            candidates = list(range(cfg['num_clients']))
        else:
            candidates = [cid for cid in range(cfg['num_clients']) if cid not in set(unseen_clients)]
        if args.num_malicious > len(candidates):
            raise ValueError("num_malicious cannot exceed available candidate clients")
        malicious_clients = np.random.choice(candidates, args.num_malicious, replace=False).astype(int).tolist()
        if args.attack_type == 'trigger':
            attack_hook = make_backdoor_hook(
                malicious_clients,
                p=args.poison_rate,
                target_label=args.target_label,
                size=args.trigger_size,
                dirty_label=True,
            )
        elif args.attack_type == 'labelflip':
            attack_hook = make_label_flip_hook(
                malicious_clients,
                target_label=args.target_label,
                p=args.poison_rate,
            )

    print(f"\n[Setup] SafeSplit sequential U-shaped SL")
    print(f"[Setup] Clients={cfg['num_clients']} | Epochs={cfg['epochs']} | Steps/client={cfg['steps_per_client']} | IID-rate={cfg['iid_rate']}")
    print(f"[Setup] True unseen clients: {unseen_clients}")
    print(f"[Setup] True backdoor clients: {malicious_clients}")
    history_limit = args.history_size or cfg['num_clients']
    min_history = args.min_history if args.min_history is not None else history_limit
    print(
        f"[Setup] Defense={'ON' if args.enable_defense else 'OFF'} | "
        f"history={history_limit} | min_history={min_history} | dct_ratio={args.dct_ratio}"
    )

    initial = make_checkpoint(
        index=0,
        epoch=0,
        client_id=None,
        head=eval_head,
        server=server,
        tail=eval_tail,
        previous_backbone_vec=None,
    )
    history: List[ModelCheckpoint] = [initial]
    active_checkpoint = initial
    next_checkpoint_index = 1

    skipped_counts = {cid: 0 for cid in range(cfg['num_clients'])}
    rng = np.random.default_rng(cfg['seed'] + 241698)

    for epoch in range(1, cfg['epochs'] + 1):
        if args.shuffle_clients:
            client_order = rng.permutation(cfg['num_clients']).astype(int).tolist()
        else:
            client_order = list(range(cfg['num_clients']))

        epoch_losses: List[float] = []
        epoch_skipped: List[int] = []
        print(f"\nEpoch {epoch}/{cfg['epochs']} | order={client_order}")

        for cid in client_order:
            client = clients[cid]
            load_checkpoint_to_client(active_checkpoint, client, server)
            previous_vec = active_checkpoint.backbone_vec
            stats = client.run_batches(server, max_steps=cfg['steps_per_client'], attack_hook=attack_hook)
            epoch_losses.append(float(stats['loss']))

            candidate = make_checkpoint(
                index=next_checkpoint_index,
                epoch=epoch,
                client_id=cid,
                head=client.head,
                server=server,
                tail=client.tail,
                previous_backbone_vec=previous_vec,
            )
            next_checkpoint_index += 1
            history.append(candidate)

            if args.enable_defense:
                decision = analyze_safesplit(
                    history,
                    history_size=max(2, int(history_limit)),
                    min_history=max(2, int(min_history)),
                    dct_ratio=args.dct_ratio,
                )
                current_history_pos = len(history) - 1
                kept_current = decision.selected_history_pos == current_history_pos
                skipped_now = [sid for sid in decision.skipped_client_ids if sid is not None]

                if not kept_current:
                    for skipped_id in skipped_now:
                        skipped_counts[int(skipped_id)] += 1
                    epoch_skipped.extend(int(sid) for sid in skipped_now)
                    active_checkpoint = decision.selected_checkpoint
                    history = history[:decision.selected_history_pos + 1]
                    if len(history) > history_limit:
                        history = history[-history_limit:]
                    load_checkpoint_to_models(active_checkpoint, eval_head, server, eval_tail)
                    action = f"ROLLBACK->ckpt{active_checkpoint.index}"
                else:
                    active_checkpoint = candidate
                    if len(history) > history_limit:
                        history = history[-history_limit:]
                    load_checkpoint_to_models(active_checkpoint, eval_head, server, eval_tail)
                    action = "KEEP"

                local_current_pos = len(decision.frequency_scores) - 1
                freq_current = float(decision.frequency_scores[local_current_pos]) if len(decision.frequency_scores) else 0.0
                rot_current = float(decision.rotation_scores[local_current_pos]) if len(decision.rotation_scores) else 0.0
                in_freq = current_history_pos in set(decision.frequency_majority)
                in_rot = current_history_pos in set(decision.rotation_majority)
                print(
                    f"  Client {cid:02d} [{client_identity(cid, unseen_clients, malicious_clients):8}] "
                    f"loss={stats['loss']:.4f} | E={freq_current:.4f} | R={rot_current:.6e} | "
                    f"freqMaj={int(in_freq)} rotMaj={int(in_rot)} | {action}"
                )
            else:
                active_checkpoint = candidate
                load_checkpoint_to_models(active_checkpoint, eval_head, server, eval_tail)
                print(
                    f"  Client {cid:02d} [{client_identity(cid, unseen_clients, malicious_clients):8}] "
                    f"loss={stats['loss']:.4f} | KEEP(no defense)"
                )

        load_checkpoint_to_models(active_checkpoint, eval_head, server, eval_tail)
        if args.root_calib_steps > 0:
            calib_loss = clean_train_steps(
                eval_head, eval_tail, server, root_loader, device,
                lr=args.root_pretrain_lr,
                steps=args.root_calib_steps,
            )
            calibrated_checkpoint = make_checkpoint(
                index=active_checkpoint.index,
                epoch=epoch,
                client_id=active_checkpoint.client_id,
                head=eval_head,
                server=server,
                tail=eval_tail,
                previous_backbone_vec=active_checkpoint.backbone_vec,
            )
            active_checkpoint = calibrated_checkpoint
            if history:
                history[-1] = calibrated_checkpoint
            print(f"  [RootCalib] steps={args.root_calib_steps} loss={calib_loss:.4f}")

        skip_ratio = len(set(epoch_skipped)) / max(1, cfg['num_clients'])
        if args.root_rescue_steps > 0 and skip_ratio >= args.skip_rescue_fraction:
            rescue_loss = clean_train_steps(
                eval_head, eval_tail, server, root_loader, device,
                lr=args.root_pretrain_lr,
                steps=args.root_rescue_steps,
            )
            rescued_checkpoint = make_checkpoint(
                index=active_checkpoint.index,
                epoch=epoch,
                client_id=active_checkpoint.client_id,
                head=eval_head,
                server=server,
                tail=eval_tail,
                previous_backbone_vec=active_checkpoint.backbone_vec,
            )
            active_checkpoint = rescued_checkpoint
            if history:
                history[-1] = rescued_checkpoint
            print(
                f"  [RootRescue] skip_ratio={skip_ratio:.2f} >= {args.skip_rescue_fraction:.2f} | "
                f"steps={args.root_rescue_steps} loss={rescue_loss:.4f}"
            )

        ma = eval_clean_accuracy(eval_head, eval_tail, server, test_loader, device=device)
        unseen_acc = eval_unseen_accuracy(eval_head, eval_tail, server, china_test_loader, device=device)
        if args.enable_attack and args.attack_type == 'trigger':
            asr = eval_asr_strict(
                eval_head,
                eval_tail,
                server,
                test_loader,
                target_label=args.target_label,
                trigger_size=args.trigger_size,
                device=device,
            )
            asr_msg = f"{asr:.4f}"
        else:
            asr_msg = "N/A"

        skipped_backdoor = sorted(set(epoch_skipped).intersection(malicious_clients))
        skipped_unseen = sorted(set(epoch_skipped).intersection(unseen_clients))
        skipped_benign = sorted(
            set(epoch_skipped)
            - set(malicious_clients)
            - set(unseen_clients)
        )
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(
            f"  [SafeSplit] skipped={sorted(set(epoch_skipped))} | "
            f"backdoor={skipped_backdoor} unseen={skipped_unseen} benign={skipped_benign} | "
            f"mean_loss={mean_loss:.4f}"
        )
        print(f"  [Eval] MA: {ma:.4f} | Unseen Accuracy: {unseen_acc:.4f} | ASR: {asr_msg} | Defense={'ON' if args.enable_defense else 'OFF'}")

    total_skipped = [cid for cid, count in skipped_counts.items() if count > 0]
    print("\n[Summary] SafeSplit skipped counts:")
    for cid in total_skipped:
        print(f"  Client {cid:02d} [{client_identity(cid, unseen_clients, malicious_clients):8}] skipped {skipped_counts[cid]} time(s)")
    if not total_skipped:
        print("  No client update was skipped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeSplit baseline runner for GTSRB split learning")
    parser.add_argument('--config', default='configs/gtsrb.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--steps-per-client', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-clients', type=int, default=30)
    parser.add_argument('--iid-rate', type=float, default=1.0, help='SafeSplit paper uses IID-rate=1.0 for GTSRB to preserve MA')
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--enable-defense', dest='enable_defense', action='store_true', default=True)
    parser.add_argument('--disable-defense', dest='enable_defense', action='store_false')
    parser.add_argument('--history-size', type=int, default=None, help='SafeSplit checkpoint window; default equals num_clients')
    parser.add_argument('--min-history', type=int, default=None, help='Minimum checkpoints before rollback analysis starts; default equals history-size')
    parser.add_argument('--dct-ratio', type=float, default=0.25, help='Low-frequency DCT side ratio for backbone updates')
    parser.add_argument('--shuffle-clients', action='store_true', help='Shuffle the sequential client order every epoch')

    parser.add_argument('--enable-attack', dest='enable_attack', action='store_true', default=True)
    parser.add_argument('--disable-attack', dest='enable_attack', action='store_false')
    parser.add_argument('--num-malicious', type=int, default=4)
    parser.add_argument('--attack-type', type=str, default='trigger', choices=['trigger', 'labelflip'])
    parser.add_argument('--poison-rate', type=float, default=0.8)
    parser.add_argument('--target-label', type=int, default=0)
    parser.add_argument('--trigger-size', type=int, default=6)
    parser.add_argument('--allow-malicious-unseen-overlap', action='store_true')

    parser.add_argument('--num-new-clients', type=int, default=4)
    parser.add_argument('--unseen', type=float, default=0.7)
    parser.add_argument('--root-size', type=int, default=1000)
    parser.add_argument('--root-pretrain-steps', type=int, default=800)
    parser.add_argument('--root-pretrain-lr', type=float, default=0.01)
    parser.add_argument('--root-calib-steps', type=int, default=2)
    parser.add_argument('--root-rescue-steps', type=int, default=6, help='Extra clean steps when SafeSplit skips too many clients in one epoch')
    parser.add_argument('--skip-rescue-fraction', type=float, default=0.5, help='Trigger root rescue when this fraction of clients is skipped')
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_args()
    run_safesplit(cli_args.config, cli_args)
