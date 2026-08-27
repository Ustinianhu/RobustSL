#metrics.py
import torch
from attacks.backdoor_gtsrb import apply_red_rectangle

@torch.no_grad()
def eval_clean_accuracy(model_head, model_tail, server, loader, device):
    model_head.eval()
    model_tail.eval()

    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        a = model_head(x)
        b = server.forward_only(a.detach().cpu())
        b = b.to(device)

        logits = model_tail(b)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / max(1, total)


@torch.no_grad()
def eval_asr_strict(model_head, model_tail, server, data_loader,
                    target_label=0, trigger_size=6, device='cuda'):

    model_head.eval()
    model_tail.eval()

    total, hit = 0, 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        keep = (y != int(target_label))
        if not keep.any():
            continue

        x, y = x[keep], y[keep]

        x_trig = apply_red_rectangle(x, size=int(trigger_size)).to(device)

        a = model_head(x_trig)
        b = server.forward_only(a.detach().cpu()).to(device)

        logits = model_tail(b)
        pred = logits.argmax(dim=1)

        hit += (pred == int(target_label)).sum().item()
        total += y.size(0)

    return hit / max(1, total)


@torch.no_grad()
def eval_asr_all(model_head, model_tail, server, data_loader,
                 target_label=0, trigger_size=6, device='cuda'):

    model_head.eval()
    model_tail.eval()

    total, success = 0, 0

    for x, _ in data_loader:
        x = x.to(device)

        x_trig = apply_red_rectangle(x, size=int(trigger_size)).to(device)

        a = model_head(x_trig)
        b = server.forward_only(a.detach().cpu()).to(device)

        logits = model_tail(b)
        pred = logits.argmax(dim=1)

        success += (pred == int(target_label)).sum().item()
        total += pred.numel()

    return success / max(1, total)


@torch.no_grad()
def eval_unseen_accuracy(model_head, model_tail, server, china_loader, device):

    model_head.eval()
    model_tail.eval()

    correct, total = 0, 0

    for x, y in china_loader:
        x, y = x.to(device), y.to(device)

        a = model_head(x)
        b = server.forward_only(a.detach().cpu()).to(device)

        logits = model_tail(b)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / max(1, total)