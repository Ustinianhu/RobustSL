#!/usr/bin/env python3
"""Run no-defense GTSRB experiments matching the RobustSL client-count setup."""
from __future__ import annotations

import csv
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFENSE = ROOT / 'Defense'
PYTHON = os.environ.get('CLIENT_COUNT_PYTHON') or __import__('sys').executable
CONFIG = DEFENSE / 'configs' / 'gtsrb.yaml'
RUN_ROOT = ROOT / 'client_count_no_defense_runs'
SUMMARY_CSV = ROOT / 'client_count_no_defense_results.csv'
SUMMARY_MD = ROOT / 'client_count_no_defense_results.md'
CLIENT_COUNTS = [5, 10, 15, 20, 25, 30, 40]
EPOCH_RE = re.compile(r'Epoch\s+(\d+)/(\d+)')
EVAL_RE = re.compile(
    r'\[Eval\]\s+MA:\s*([0-9eE+\-.]+)\s*\|\s*'
    r'Unseen Accuracy:\s*([0-9eE+\-.]+)\s*\|\s*'
    r'ASR:\s*([0-9eE+\-.]+|N/A)'
)


def ratio_count(total: int, ratio: float) -> int:
    return max(1, min(total - 1, int(round(total * ratio))))


def command_for(total: int) -> list[str]:
    return [
        PYTHON, '-u', str(DEFENSE / 'run_gtsrb.py'),
        '--config', str(CONFIG),
        '--model', 'resnet18',
        '--epochs', '30',
        '--num-clients', str(total),
        '--num-new-clients', str(ratio_count(total, 0.20)),
        '--unseen', '0.70',
        '--num-malicious', str(ratio_count(total, 0.10)),
        '--attack-type', 'trigger',
        '--poison-rate', '0.80',
        '--attack-scale', '1.0',
        '--exact-unseen-sampling',
        '--steps-per-client', '20',
        '--batch-size', '64',
        '--root-size', '1000',
        '--root-pretrain-steps', '500',
        '--phase2-history-range',
        '--phase2-history-size', '15',
        '--phase2-history-min-versions', '3',
        '--phase2-probe-batches', '1',
        '--phase2-gram-orders', '1', '2',
        '--phase2-mad-k', '1.5',
        '--phase2-risk-quantile', '0.90',
        '--phase2-dct-weight', '1.0',
        '--phase2-l2-weight', '1.0',
        '--phase2-gram-weight', '1.0',
        '--phase2-benign-trust-floor', '0.65',
        '--aggregation-delta', '1.0',
    ]


def parse_log(path: Path) -> list[dict]:
    records, epoch = [], None
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epoch = int(m.group(1))
        m = EVAL_RE.search(line)
        if not m:
            continue
        records.append({
            'epoch': epoch if epoch is not None else len(records) + 1,
            'ma': float(m.group(1)),
            'clean_accuracy': float(m.group(1)),
            'unseen_accuracy': float(m.group(2)),
            'asr': None if m.group(3) == 'N/A' else float(m.group(3)),
        })
    return records


def write_results(rows: list[dict]) -> None:
    fields = [
        'num_clients', 'unseen_clients', 'unseen_client_ratio',
        'unseen_data_ratio', 'malicious_clients', 'pmr', 'poison_rate',
        'dataset', 'model', 'epochs', 'status', 'epochs_completed',
        'final_ma', 'final_clean_accuracy', 'final_unseen_accuracy',
        'final_asr', 'log_file',
    ]
    with SUMMARY_CSV.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value):
        return 'N/A' if value is None else f'{float(value):.4f}'

    lines = [
        '# GTSRB Client-Count No-Defense Experiment', '',
        f'Generated: {datetime.now().isoformat(timespec="seconds")}', '',
        'Same configuration as the RobustSL client-count experiment, without `--enable-defense`.', '',
        '| Clients | Unseen clients | Malicious clients | MA | UA | BA | Status | Log |',
        '|---:|---:|---:|---:|---:|---:|---|---|',
    ]
    for row in sorted(rows, key=lambda item: item['num_clients']):
        lines.append(
            f'| {row["num_clients"]} | {row["unseen_clients"]} | {row["malicious_clients"]} | '
            f'{fmt(row["final_ma"])} | {fmt(row["final_unseen_accuracy"])} | '
            f'{fmt(row["final_asr"])} | {row["status"]} | '
            f'[{Path(row["log_file"]).name}]({row["log_file"]}) |'
        )
    SUMMARY_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_one(total: int) -> dict:
    unseen_clients = ratio_count(total, 0.20)
    malicious_clients = ratio_count(total, 0.10)
    run_dir = RUN_ROOT / f'clients_{total:04d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'stdout.log'
    command = command_for(total)
    print(f'\n[START NO DEFENSE] N={total} | unseen={unseen_clients} | malicious={malicious_clients}', flush=True)
    print('[CMD] ' + ' '.join(command), flush=True)
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command, cwd=DEFENSE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith('Epoch ') or '[Eval]' in line or '[Data]' in line or '[Attack]' in line:
                print(f'[N={total} NO DEFENSE] {line.rstrip()}', flush=True)
        return_code = process.wait()
    records = parse_log(log_path)
    final = records[-1] if records else {'ma': None, 'clean_accuracy': None, 'unseen_accuracy': None, 'asr': None}
    status = 'completed' if return_code == 0 and records else 'failed'
    print(
        f'[DONE NO DEFENSE] N={total} | status={status} | rounds={len(records)} | '
        f'MA={final["ma"]} | UA={final["unseen_accuracy"]} | BA={final["asr"]}',
        flush=True,
    )
    return {
        'num_clients': total,
        'unseen_clients': unseen_clients,
        'unseen_client_ratio': unseen_clients / total,
        'unseen_data_ratio': 0.70,
        'malicious_clients': malicious_clients,
        'pmr': malicious_clients / total,
        'poison_rate': 0.80,
        'dataset': 'GTSRB', 'model': 'ResNet18', 'epochs': 30,
        'status': status, 'epochs_completed': len(records),
        'final_ma': final['ma'], 'final_clean_accuracy': final['clean_accuracy'],
        'final_unseen_accuracy': final['unseen_accuracy'], 'final_asr': final['asr'],
        'log_file': str(log_path),
    }


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for total in CLIENT_COUNTS:
        rows.append(run_one(total))
        write_results(rows)
    write_results(rows)
    return 0 if all(row['status'] == 'completed' for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
