import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from attacks.backdoor_gtsrb import make_backdoor_hook, make_label_flip_hook
from eval.metrics import eval_asr_strict, eval_clean_accuracy, eval_unseen_accuracy
from run_gtsrb import (
    average_model_buffers,
    build_dataloaders,
    compute_dct_low_frequency_score,
    get_flat_weights,
    get_model_buffers,
    get_model_classes,
    load_config,
    prepare_client_from_global,
    root_calibration_step,
    set_flat_weights,
    set_model_buffers,
)
from sl_core.client import Client
from sl_core.server import ServerBackbone


def robust_standardize(features, eps=1e-8):
    features = np.asarray(features, dtype=np.float64)
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    scale = 1.4826 * np.maximum(mad, eps)
    return (features - median) / scale


def kmeans_two_clusters(features, max_iter=50):
    x = np.asarray(features, dtype=np.float64)
    n = x.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.int64)

    dists = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    first, second = np.unravel_index(np.argmax(dists), dists.shape)
    centroids = np.stack([x[first], x[second]], axis=0)
    labels = np.zeros(n, dtype=np.int64)

    for _ in range(max_iter):
        new_labels = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in (0, 1):
            mask = labels == cluster_id
            if mask.any():
                centroids[cluster_id] = x[mask].mean(axis=0)

    return labels


def build_update_features(updates):
    mean_update = None
    for update in updates:
        update_f = update.detach().float().cpu()
        mean_update = update_f.clone() if mean_update is None else mean_update + update_f
    mean_update = mean_update / max(1, len(updates))
    mean_norm = torch.norm(mean_update, p=2).item()

    rows = []
    for update in updates:
        update_f = update.detach().float().cpu()
        norm = torch.norm(update_f, p=2).item()
        cosine = F.cosine_similarity(
            update_f.view(1, -1),
            mean_update.view(1, -1),
            dim=1,
            eps=1e-12,
        ).item()
        distance = torch.norm(update_f - mean_update, p=2).item()
        relative_norm = norm / max(mean_norm, 1e-12)
        rows.append([np.log1p(norm), cosine, np.log1p(distance), relative_norm])

    return np.asarray(rows, dtype=np.float64)


def aggregate_dp(updates, buffers, clip_norm, noise_multiplier, device, seed):
    clipped = []
    original_norms = []
    scales = []
    clip_norm = float(clip_norm)

    for update in updates:
        update_f = update.detach().float().cpu()
        norm = torch.norm(update_f, p=2).item()
        scale = min(1.0, clip_norm / max(norm, 1e-12))
        clipped.append(update_f * scale)
        original_norms.append(norm)
        scales.append(scale)

    agg_delta = torch.stack(clipped, dim=0).mean(dim=0)
    noise_std = float(noise_multiplier) * clip_norm / max(1, len(updates))
    if noise_std > 0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        noise = torch.normal(
            mean=0.0,
            std=noise_std,
            size=agg_delta.shape,
            generator=generator,
            dtype=agg_delta.dtype,
        )
        agg_delta = agg_delta + noise

    stats = {
        "mean_norm": float(np.mean(original_norms)),
        "max_norm": float(np.max(original_norms)),
        "clipped": int(sum(scale < 0.999999 for scale in scales)),
        "noise_std": noise_std,
    }
    return agg_delta.to(device), average_model_buffers(buffers), stats


def aggregate_kmeans(updates, buffers, device):
    features = build_update_features(updates)
    scaled = robust_standardize(features)
    labels = kmeans_two_clusters(scaled)
    cluster_ids = [np.where(labels == cluster_id)[0].tolist() for cluster_id in (0, 1)]

    if len(cluster_ids[0]) == len(cluster_ids[1]):
        median_norms = []
        for ids in cluster_ids:
            median_norms.append(float(np.median(features[ids, 0])) if ids else float("inf"))
        keep_cluster = int(np.argmin(median_norms))
    else:
        keep_cluster = 0 if len(cluster_ids[0]) > len(cluster_ids[1]) else 1

    kept_ids = cluster_ids[keep_cluster]
    filtered_ids = [cid for cid in range(len(updates)) if cid not in kept_ids]
    if not kept_ids:
        kept_ids = list(range(len(updates)))
        filtered_ids = []

    agg_delta = torch.stack([updates[cid].detach().float().cpu() for cid in kept_ids], dim=0).mean(dim=0)
    kept_buffers = [buffers[cid] for cid in kept_ids]
    stats = {
        "kept_ids": kept_ids,
        "filtered_ids": filtered_ids,
        "cluster0": cluster_ids[0],
        "cluster1": cluster_ids[1],
    }
    return agg_delta.to(device), average_model_buffers(kept_buffers), stats


def rank_zorro_candidates(candidates, beta=0.7, low_freq_ratio=0.5):
    if len(candidates) <= 1:
        return {
            "raw_scores": [0.0 for _ in candidates],
            "adjusted_scores": [0.0 for _ in candidates],
            "remove_idx": None,
            "remaining": list(candidates),
            "bm": candidates[0] if candidates else None,
        }

    raw_scores = [
        float(compute_dct_low_frequency_score(entry["update"], low_freq_ratio=low_freq_ratio))
        for entry in candidates
    ]
    adjusted_scores = np.asarray(raw_scores, dtype=float)
    adjusted_scores[0] = adjusted_scores[0] / max(float(beta), 1e-8)
    adjusted_scores[-1] = adjusted_scores[-1] * float(beta)

    remove_idx = int(np.argmax(adjusted_scores))
    remaining = [entry for idx, entry in enumerate(candidates) if idx != remove_idx]
    remaining_adjusted = np.delete(adjusted_scores, remove_idx)
    bm_idx = int(np.argmin(remaining_adjusted))
    bm = remaining[bm_idx]

    return {
        "raw_scores": raw_scores,
        "adjusted_scores": adjusted_scores.tolist(),
        "remove_idx": remove_idx,
        "remaining": remaining,
        "bm": bm,
    }


def train_zorro(config_path, args):
    cfg = load_config(config_path)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.steps_per_client is not None:
        cfg["steps_per_client"] = args.steps_per_client
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    cfg["num_clients"] = args.num_clients if args.num_clients is not None else 30

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    loaders, test_loader, unseen_test_loader, root_loader, new_data_clients = build_dataloaders(
        cfg,
        num_new_clients=args.num_new_clients,
        unseen=args.unseen,
        root_size=args.root_size,
    )

    device_str = cfg["server"].get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    head_cls, backbone_cls, tail_cls = get_model_classes(args.model)

    head0 = head_cls().to(device)
    back = backbone_cls().to(device)
    tail0 = tail_cls().to(device)
    server = ServerBackbone(backbone=back, lr=cfg["server"]["lr"], device=device)

    root_head_opt = torch.optim.SGD(head0.parameters(), lr=args.root_calib_lr, momentum=0.0)
    root_tail_opt = torch.optim.SGD(tail0.parameters(), lr=args.root_calib_lr, momentum=0.0)
    root_server_opt = torch.optim.SGD(server.model.parameters(), lr=args.root_calib_lr, momentum=0.0)

    clients = [
        Client(
            cid,
            head_cls(),
            tail_cls(),
            loaders[cid],
            lr_head=cfg["clients"]["lr_head"],
            lr_tail=cfg["clients"]["lr_tail"],
            device=device,
        )
        for cid in range(cfg["num_clients"])
    ]

    malicious_clients = []
    attack_hook = None
    if args.enable_attack:
        candidate_clients = [cid for cid in range(cfg["num_clients"]) if cid not in set(new_data_clients)]
        if args.num_malicious > len(candidate_clients):
            raise ValueError("num_malicious cannot exceed non-unseen clients.")
        malicious_clients = np.random.choice(candidate_clients, args.num_malicious, replace=False).tolist()
        print(f"[Attack] Malicious client IDs: {malicious_clients}, attack={args.attack_type}")
        if args.attack_type == "trigger":
            attack_hook = make_backdoor_hook(
                malicious_clients,
                p=args.poison_rate,
                target_label=args.target_label,
                size=args.trigger_size,
            )
        elif args.attack_type == "labelflip":
            attack_hook = make_label_flip_hook(
                malicious_clients,
                target_label=args.target_label,
                p=args.poison_rate,
            )

    print(f"[Data] Unseen client IDs: {new_data_clients}")
    print(
        f"[Baseline] method={args.baseline} model={args.model} clients={cfg['num_clients']} "
        f"queue={args.zorro_queue_size} beta={args.zorro_beta:.2f} dct_ratio={args.zorro_dct_ratio:.2f}"
    )

    current_flat = get_flat_weights(head0, server.model, tail0).detach().cpu().clone()
    current_server_buffers = get_model_buffers(server.model)

    if args.root_pretrain_steps > 0:
        pre_head_opt = torch.optim.SGD(head0.parameters(), lr=args.root_pretrain_lr, momentum=0.9)
        pre_tail_opt = torch.optim.SGD(tail0.parameters(), lr=args.root_pretrain_lr, momentum=0.9)
        pre_server_opt = torch.optim.SGD(server.model.parameters(), lr=args.root_pretrain_lr, momentum=0.9)
        pretrain_loss = root_calibration_step(
            head0,
            tail0,
            server,
            root_loader,
            device,
            pre_head_opt,
            pre_tail_opt,
            pre_server_opt,
            max_steps=args.root_pretrain_steps,
        )
        current_flat = get_flat_weights(head0, server.model, tail0).detach().cpu().clone()
        current_server_buffers = get_model_buffers(server.model)
        print(f"[RootPretrain] steps={args.root_pretrain_steps} lr={args.root_pretrain_lr} loss={pretrain_loss:.4f}")

    queue = []
    bootstrap_clients = max(0, min(int(args.zorro_bootstrap_clients), cfg["num_clients"]))
    queue_size = max(1, int(args.zorro_queue_size))
    beta = min(max(float(args.zorro_beta), 1e-3), 1.0)
    dct_ratio = float(args.zorro_dct_ratio)

    print(
        f"[ZORRO] secure bootstrap uses first {bootstrap_clients} client(s) without attack "
        f"to warm up the queue."
    )

    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        for cid in range(cfg["num_clients"]):
            use_attack = args.enable_attack and not (epoch == 0 and cid < bootstrap_clients)

            prepare_client_from_global(clients[cid], server, current_flat, current_server_buffers)
            stats = clients[cid].run_batches(
                server,
                max_steps=cfg["steps_per_client"],
                attack_hook=attack_hook if use_attack else None,
            )

            new_flat = get_flat_weights(clients[cid].head, server.model, clients[cid].tail).detach().cpu().clone()
            new_buffers = get_model_buffers(server.model)
            update = new_flat - current_flat.detach().cpu()
            candidate = {
                "cid": cid,
                "flat": new_flat,
                "update": update,
                "buffers": new_buffers,
                "loss": stats["loss"],
                "steps": stats["steps"],
            }

            if len(queue) < queue_size:
                queue.append(candidate)
                current_flat = new_flat.clone()
                current_server_buffers = new_buffers
                print(
                    f"  [ZORRO] Client {cid:02d} | steps={stats['steps']} loss={stats['loss']:.4f} | "
                    f"bootstrap queue={len(queue)}/{queue_size}"
                )
            else:
                candidate_pool = queue + [candidate]
                ranked = rank_zorro_candidates(candidate_pool, beta=beta, low_freq_ratio=dct_ratio)
                queue = ranked["remaining"]
                bm = ranked["bm"]
                remove_id = (
                    candidate_pool[ranked["remove_idx"]]["cid"]
                    if ranked["remove_idx"] is not None
                    else cid
                )
                current_flat = bm["flat"].detach().cpu().clone()
                current_server_buffers = bm["buffers"]
                print(
                    f"  [ZORRO] Client {cid:02d} | steps={stats['steps']} loss={stats['loss']:.4f} | "
                    f"raw={np.round(ranked['raw_scores'], 4).tolist()} | "
                    f"adj={np.round(np.asarray(ranked['adjusted_scores']), 4).tolist()} | "
                    f"remove=C{remove_id:02d} | bm=C{bm['cid']:02d} | queue={len(queue)}/{queue_size}"
                )

            set_flat_weights(head0, server.model, tail0, current_flat)
            set_model_buffers(server.model, current_server_buffers)

        if args.root_calib_steps > 0:
            calib_loss = root_calibration_step(
                head0,
                tail0,
                server,
                root_loader,
                device,
                root_head_opt,
                root_tail_opt,
                root_server_opt,
                max_steps=args.root_calib_steps,
            )
            current_flat = get_flat_weights(head0, server.model, tail0).detach().cpu().clone()
            current_server_buffers = get_model_buffers(server.model)
            print(f"  [RootCalib] steps={args.root_calib_steps} loss={calib_loss:.4f}")

        clean_acc = eval_clean_accuracy(head0, tail0, server, test_loader, device=device)
        unseen_acc = eval_unseen_accuracy(head0, tail0, server, unseen_test_loader, device=device)
        if args.enable_attack and args.attack_type == "trigger":
            asr = eval_asr_strict(
                head0,
                tail0,
                server,
                test_loader,
                target_label=args.target_label,
                trigger_size=args.trigger_size,
                device=device,
            )
            asr_msg = f"{asr:.4f}"
        else:
            asr_msg = "N/A"
        print(
            f"\n  [Eval] MA: {clean_acc:.4f} | Unseen Accuracy: {unseen_acc:.4f} | "
            f"ASR: {asr_msg} | Baseline={args.baseline}"
        )


def train_baseline(config_path, args):
    if args.baseline == "zorro":
        return train_zorro(config_path, args)

    cfg = load_config(config_path)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.steps_per_client is not None:
        cfg["steps_per_client"] = args.steps_per_client
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    cfg["num_clients"] = args.num_clients if args.num_clients is not None else 30

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    loaders, test_loader, unseen_test_loader, root_loader, new_data_clients = build_dataloaders(
        cfg,
        num_new_clients=args.num_new_clients,
        unseen=args.unseen,
        root_size=args.root_size,
    )

    device_str = cfg["server"].get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    head_cls, backbone_cls, tail_cls = get_model_classes(args.model)

    head0 = head_cls().to(device)
    back = backbone_cls().to(device)
    tail0 = tail_cls().to(device)
    server = ServerBackbone(backbone=back, lr=cfg["server"]["lr"], device=device)

    root_head_opt = torch.optim.SGD(head0.parameters(), lr=args.root_calib_lr, momentum=0.0)
    root_tail_opt = torch.optim.SGD(tail0.parameters(), lr=args.root_calib_lr, momentum=0.0)
    root_server_opt = torch.optim.SGD(server.model.parameters(), lr=args.root_calib_lr, momentum=0.0)

    clients = [
        Client(
            cid,
            head_cls(),
            tail_cls(),
            loaders[cid],
            lr_head=cfg["clients"]["lr_head"],
            lr_tail=cfg["clients"]["lr_tail"],
            device=device,
        )
        for cid in range(cfg["num_clients"])
    ]

    malicious_clients = []
    attack_hook = None
    if args.enable_attack:
        candidate_clients = [cid for cid in range(cfg["num_clients"]) if cid not in set(new_data_clients)]
        if args.num_malicious > len(candidate_clients):
            raise ValueError("num_malicious cannot exceed non-unseen clients.")
        malicious_clients = np.random.choice(candidate_clients, args.num_malicious, replace=False).tolist()
        print(f"[Attack] Malicious client IDs: {malicious_clients}, attack={args.attack_type}")
        if args.attack_type == "trigger":
            attack_hook = make_backdoor_hook(
                malicious_clients,
                p=args.poison_rate,
                target_label=args.target_label,
                size=args.trigger_size,
            )
        elif args.attack_type == "labelflip":
            attack_hook = make_label_flip_hook(
                malicious_clients,
                target_label=args.target_label,
                p=args.poison_rate,
            )

    print(f"[Data] Unseen client IDs: {new_data_clients}")
    print(f"[Baseline] method={args.baseline} model={args.model} clients={cfg['num_clients']}")

    global_flat = get_flat_weights(head0, server.model, tail0).detach()
    global_server_buffers = get_model_buffers(server.model)

    if args.root_pretrain_steps > 0:
        pre_head_opt = torch.optim.SGD(head0.parameters(), lr=args.root_pretrain_lr, momentum=0.9)
        pre_tail_opt = torch.optim.SGD(tail0.parameters(), lr=args.root_pretrain_lr, momentum=0.9)
        pre_server_opt = torch.optim.SGD(server.model.parameters(), lr=args.root_pretrain_lr, momentum=0.9)
        pretrain_loss = root_calibration_step(
            head0,
            tail0,
            server,
            root_loader,
            device,
            pre_head_opt,
            pre_tail_opt,
            pre_server_opt,
            max_steps=args.root_pretrain_steps,
        )
        global_flat = get_flat_weights(head0, server.model, tail0).detach()
        global_server_buffers = get_model_buffers(server.model)
        print(f"[RootPretrain] steps={args.root_pretrain_steps} lr={args.root_pretrain_lr} loss={pretrain_loss:.4f}")

    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        client_updates = []
        client_buffers = []
        client_losses = []
        global_flat_cpu = global_flat.detach().cpu()

        for cid in range(cfg["num_clients"]):
            prepare_client_from_global(clients[cid], server, global_flat, global_server_buffers)
            stats = clients[cid].run_batches(
                server,
                max_steps=cfg["steps_per_client"],
                attack_hook=attack_hook,
            )
            update = get_flat_weights(clients[cid].head, server.model, clients[cid].tail).detach().cpu() - global_flat_cpu
            if args.enable_attack and cid in malicious_clients:
                update = update * args.attack_scale

            client_updates.append(update)
            client_buffers.append(get_model_buffers(server.model))
            client_losses.append(stats["loss"])
            print(
                f"  Client {cid:02d} | steps={stats['steps']} loss={stats['loss']:.4f} | "
                f"||update||={torch.norm(update.float(), p=2).item():.4f}"
            )

        if args.baseline == "dp":
            agg_delta, aggregated_buffers, stats = aggregate_dp(
                client_updates,
                client_buffers,
                args.dp_clip_norm,
                args.dp_noise_multiplier,
                device,
                seed=cfg["seed"] + 1729 + epoch,
            )
            print(
                "  [DP] "
                f"clip_norm={args.dp_clip_norm:.4f} noise_multiplier={args.dp_noise_multiplier:.4f} "
                f"noise_std={stats['noise_std']:.6f} clipped={stats['clipped']}/{cfg['num_clients']} "
                f"mean_norm={stats['mean_norm']:.4f} max_norm={stats['max_norm']:.4f}"
            )
        elif args.baseline == "kmeans":
            agg_delta, aggregated_buffers, stats = aggregate_kmeans(client_updates, client_buffers, device)
            print(
                "  [KMeans] "
                f"cluster0={stats['cluster0']} cluster1={stats['cluster1']} | "
                f"kept={stats['kept_ids']} filtered={stats['filtered_ids']}"
            )
        else:
            raise ValueError(f"Unsupported baseline: {args.baseline}")

        global_flat = global_flat + args.aggregation_delta * agg_delta
        set_flat_weights(head0, server.model, tail0, global_flat)
        set_model_buffers(server.model, aggregated_buffers)
        global_server_buffers = get_model_buffers(server.model)

        if args.root_calib_steps > 0:
            calib_loss = root_calibration_step(
                head0,
                tail0,
                server,
                root_loader,
                device,
                root_head_opt,
                root_tail_opt,
                root_server_opt,
                max_steps=args.root_calib_steps,
            )
            global_flat = get_flat_weights(head0, server.model, tail0).detach()
            global_server_buffers = get_model_buffers(server.model)
            print(f"  [RootCalib] steps={args.root_calib_steps} loss={calib_loss:.4f}")

        clean_acc = eval_clean_accuracy(head0, tail0, server, test_loader, device=device)
        unseen_acc = eval_unseen_accuracy(head0, tail0, server, unseen_test_loader, device=device)
        if args.enable_attack and args.attack_type == "trigger":
            asr = eval_asr_strict(
                head0,
                tail0,
                server,
                test_loader,
                target_label=args.target_label,
                trigger_size=args.trigger_size,
                device=device,
            )
            asr_msg = f"{asr:.4f}"
        else:
            asr_msg = "N/A"
        print(
            f"\n  [Eval] MA: {clean_acc:.4f} | Unseen Accuracy: {unseen_acc:.4f} | "
            f"ASR: {asr_msg} | Baseline={args.baseline}"
        )


def build_parser():
    parser = argparse.ArgumentParser(description="GTSRB split-learning baseline defenses.")
    parser.add_argument("--config", default="configs/gtsrb.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="googlenet",
        choices=[
            "resnet_gtsrb",
            "simple_cnn",
            "resnet18",
            "resnet34",
            "googlenet",
            "vgg11",
            "wide_resnet50",
            "micronnet",
        ],
    )
    parser.add_argument("--baseline", type=str, required=True, choices=["dp", "kmeans", "zorro"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-client", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)

    parser.add_argument("--enable-attack", dest="enable_attack", action="store_true", default=True)
    parser.add_argument("--disable-attack", dest="enable_attack", action="store_false")
    parser.add_argument("--num-malicious", type=int, default=4)
    parser.add_argument("--attack-type", type=str, default="trigger", choices=["trigger", "labelflip"])
    parser.add_argument("--poison-rate", type=float, default=0.8)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--trigger-size", type=int, default=6)
    parser.add_argument("--attack-scale", type=float, default=1.0)

    parser.add_argument("--num-new-clients", type=int, default=4)
    parser.add_argument("--unseen", type=float, default=0.7)

    parser.add_argument("--aggregation-delta", type=float, default=1.0)
    parser.add_argument("--root-size", type=int, default=1000)
    parser.add_argument("--root-pretrain-steps", type=int, default=500)
    parser.add_argument("--root-pretrain-lr", type=float, default=0.01)
    parser.add_argument("--root-calib-steps", type=int, default=0)
    parser.add_argument("--root-calib-lr", type=float, default=0.003)

    parser.add_argument("--dp-clip-norm", type=float, default=5.0)
    parser.add_argument("--dp-noise-multiplier", type=float, default=0.01)
    parser.add_argument("--zorro-queue-size", type=int, default=3)
    parser.add_argument("--zorro-beta", type=float, default=0.7)
    parser.add_argument("--zorro-dct-ratio", type=float, default=0.5)
    parser.add_argument("--zorro-bootstrap-clients", type=int, default=3)
    return parser


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    parsed_args = build_parser().parse_args()
    train_baseline(parsed_args.config, parsed_args)
