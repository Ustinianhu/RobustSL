#!/usr/bin/env python3
"""Run the controlled GTSRB client-count scalability experiment."""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFENSE = ROOT / 'Defense'
PYTHON = os.environ.get('CLIENT_COUNT_PYTHON') or __import__('sys').executable
CONFIG = DEFENSE / 'configs' / 'gtsrb.yaml'
RUN_ROOT = ROOT / 'client_count_experiment_runs'
SUMMARY_CSV = ROOT / 'client_count_experiment_results.csv'
SUMMARY_MD = ROOT / 'client_count_experiment_results.md'
PLOT_SCRIPT = ROOT / 'plot_client_count_results.py'
CLIENT_COUNTS = [5, 10, 15, 20, 25, 30, 40]

EPOCH_RE = re.compile(r'Epoch\s+(\d+)/(\d+)')
EVAL_RE = re.compile(
    r'\[Eval\]\s+MA:\s*([0-9eE+\-.]+)\s*\|\s*'
    r'Unseen Accuracy:\s*([0-9eE+\-.]+)\s*\|\s*'
    r'ASR:\s*([0-9eE+\-.]+|N/A)'
)
CLIENT_RE = re.compile(r'Client\s+(\d+)\s+\|')


def ratio_count(total: int, ratio: float) -> int:
    return max(1, min(total - 1, int(round(total * ratio))))


def command_for(total: int) -> list[str]:
    unseen_clients = ratio_count(total, 0.20)
    malicious_clients = ratio_count(total, 0.10)
    return [
        PYTHON, '-u', str(DEFENSE / 'run_gtsrb.py'),
        '--config', str(CONFIG),
        '--model', 'resnet18',
        '--epochs', '30',
        '--num-clients', str(total),
        '--num-new-clients', str(unseen_clients),
        '--unseen', '0.70',
        '--num-malicious', str(malicious_clients),
        '--attack-type', 'trigger',
        '--poison-rate', '0.80',
        '--attack-scale', '1.0',
        '--enable-defense',
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


def parse_log(log_path: Path) -> list[dict]:
    records = []
    current_epoch = None
    for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
        match = EPOCH_RE.search(line)
        if match:
            current_epoch = int(match.group(1))
        match = EVAL_RE.search(line)
        if not match:
            continue
        asr = None if match.group(3) == 'N/A' else float(match.group(3))
        records.append({
            'epoch': current_epoch if current_epoch is not None else len(records) + 1,
            'ma': float(match.group(1)),
            'clean_accuracy': float(match.group(1)),
            'unseen_accuracy': float(match.group(2)),
            'asr': asr,
        })
    return records


def write_summary(rows: list[dict]) -> None:
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
        '# GTSRB Client-Count Experiment',
        '',
        f'Generated: {datetime.now().isoformat(timespec="seconds")}',
        '',
        'Controlled setting: GTSRB, ResNet18, 30 rounds, iid_rate=0.8, current tri-view Phase-II defense.',
        'Unseen client ratio = 20%; unseen data ratio within an unseen client = 70%; PMR = 10%; poison rate = 80%.',
        'The exact unseen sampling flag is enabled so that the requested unseen fraction remains valid for 500/1000 clients.',
        '',
        '## Results',
        '',
        '| Clients | Unseen clients | Malicious clients | MA | UA | ASR | Status | Log |',
        '|---:|---:|---:|---:|---:|---:|---|---|',
    ]
    for row in sorted(rows, key=lambda item: item['num_clients']):
        log_link = f'[{Path(row["log_file"]).name}]({row["log_file"]})'
        lines.append(
            f'| {row["num_clients"]} | {row["unseen_clients"]} | {row["malicious_clients"]} | '
            f'{fmt(row["final_ma"])} | {fmt(row["final_unseen_accuracy"])} | '
            f'{fmt(row["final_asr"])} | {row["status"]} | {log_link} |'
        )
    lines.extend([
        '',
        '## Configuration',
        '',
        '- Dataset: GTSRB with TT100K/China-GTSRB unseen data.',
        '- Model: ResNet18 split into client-side head, server-side backbone, and client-side tail.',
        '- Total rounds: 30; local step cap: 20; batch size: 64; root size: 1000.',
        '- Phase II: DCT + L2 + Gram, equal weights, 15-version history queue, risk quantile 0.90.',
        '- Aggregation: `aggregation_delta=1.0`, benign trust floor `0.65`.',
        '',
        'Files: `client_count_experiment_results.csv`, `client_count_results.png`, and `client_count_results.pdf`.',
    ])
    SUMMARY_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def print_progress(total: int, line: str) -> None:
    if '[Eval]' in line or line.startswith('Epoch ') or '[Data]' in line or '[Attack]' in line:
        print(f'[N={total}] {line}', flush=True)
        return
    match = CLIENT_RE.search(line)
    if match:
        cid = int(match.group(1))
        if cid in {0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999}:
            print(f'[N={total}] {line}', flush=True)


def run_one(total: int) -> dict:
    unseen_clients = ratio_count(total, 0.20)
    malicious_clients = ratio_count(total, 0.10)
    run_dir = RUN_ROOT / f'clients_{total:04d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'stdout.log'
    command = command_for(total)
    print(f'\n[START] N={total} | unseen={unseen_clients} | malicious={malicious_clients}', flush=True)
    print('[CMD] ' + ' '.join(command), flush=True)

    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command,
            cwd=DEFENSE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print_progress(total, line.rstrip())
        return_code = process.wait()

    records = parse_log(log_path)
    final = records[-1] if records else {
        'ma': None, 'clean_accuracy': None, 'unseen_accuracy': None, 'asr': None,
    }
    status = 'completed' if return_code == 0 and records else 'failed'
    print(
        f'[DONE] N={total} | status={status} | rounds={len(records)} | '
        f'MA={final["ma"]} | UA={final["unseen_accuracy"]} | ASR={final["asr"]}',
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
        'dataset': 'GTSRB',
        'model': 'ResNet18',
        'epochs': 30,
        'status': status,
        'epochs_completed': len(records),
        'final_ma': final['ma'],
        'final_clean_accuracy': final['clean_accuracy'],
        'final_unseen_accuracy': final['unseen_accuracy'],
        'final_asr': final['asr'],
        'log_file': str(log_path),
    }


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for total in CLIENT_COUNTS:
        rows.append(run_one(total))
        write_summary(rows)
        subprocess.run([PYTHON, str(PLOT_SCRIPT)], check=False)
    write_summary(rows)
    subprocess.run([PYTHON, str(PLOT_SCRIPT)], check=False)
    return 0 if all(row['status'] == 'completed' for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
