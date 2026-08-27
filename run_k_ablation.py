#!/usr/bin/env python3
"""Run a Phase-II MAD-k ablation on the controlled GTSRB setting."""
from __future__ import annotations

import csv
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFENSE = ROOT / "Defense"
PYTHON = os.environ.get("K_ABLATION_PYTHON") or __import__("sys").executable
CONFIG = DEFENSE / "configs" / "gtsrb.yaml"
RUN_ROOT = ROOT / "k_ablation_runs"
SUMMARY_CSV = ROOT / "k_ablation_results.csv"
SUMMARY_MD = ROOT / "k_ablation_results.md"

K_VALUES = [0.5, 1.0, 1.5, 2.0]
NUM_CLIENTS = 30
NUM_NEW_CLIENTS = 6
NUM_MALICIOUS = 3
EPOCHS = 30

EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")
EVAL_RE = re.compile(
    r"\[Eval\]\s+MA:\s*([0-9eE+\-.]+)\s*\|\s*"
    r"Unseen Accuracy:\s*([0-9eE+\-.]+)\s*\|\s*"
    r"ASR:\s*([0-9eE+\-.]+|N/A)"
)


def command_for(k_value: float) -> list[str]:
    return [
        PYTHON,
        "-u",
        str(DEFENSE / "run_gtsrb.py"),
        "--config",
        str(CONFIG),
        "--model",
        "resnet18",
        "--epochs",
        str(EPOCHS),
        "--num-clients",
        str(NUM_CLIENTS),
        "--num-new-clients",
        str(NUM_NEW_CLIENTS),
        "--unseen",
        "0.70",
        "--num-malicious",
        str(NUM_MALICIOUS),
        "--attack-type",
        "trigger",
        "--poison-rate",
        "0.80",
        "--attack-scale",
        "1.0",
        "--enable-defense",
        "--exact-unseen-sampling",
        "--steps-per-client",
        "20",
        "--batch-size",
        "64",
        "--root-size",
        "1000",
        "--root-pretrain-steps",
        "500",
        "--phase2-history-range",
        "--phase2-history-size",
        "15",
        "--phase2-history-min-versions",
        "3",
        "--phase2-probe-batches",
        "1",
        "--phase2-gram-orders",
        "1",
        "2",
        "--phase2-mad-k",
        str(k_value),
        "--phase2-risk-quantile",
        "0.90",
        "--phase2-dct-weight",
        "1.0",
        "--phase2-l2-weight",
        "1.0",
        "--phase2-gram-weight",
        "1.0",
        "--phase2-benign-trust-floor",
        "0.65",
        "--aggregation-delta",
        "1.0",
    ]


def parse_log(log_path: Path) -> list[dict]:
    records = []
    current_epoch = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        eval_match = EVAL_RE.search(line)
        if not eval_match:
            continue
        asr = None if eval_match.group(3) == "N/A" else float(eval_match.group(3))
        records.append(
            {
                "epoch": current_epoch if current_epoch is not None else len(records) + 1,
                "ma": float(eval_match.group(1)),
                "ua": float(eval_match.group(2)),
                "asr": asr,
            }
        )
    return records


def run_one(k_value: float) -> dict:
    label = f"k_{k_value:g}".replace(".", "_")
    run_dir = RUN_ROOT / label
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"
    command = command_for(k_value)

    print(f"\n[START] k={k_value:g}", flush=True)
    print("[CMD] " + " ".join(command), flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log:
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
            stripped = line.rstrip()
            if "[Eval]" in stripped or stripped.startswith("Epoch "):
                print(f"[k={k_value:g}] {stripped}", flush=True)
        return_code = process.wait()

    records = parse_log(log_path)
    final = records[-1] if records else {"ma": None, "ua": None, "asr": None}
    status = "completed" if return_code == 0 and records else "failed"
    print(
        f"[DONE] k={k_value:g} | status={status} | rounds={len(records)} | "
        f"MA={final['ma']} | UA={final['ua']} | ASR={final['asr']}",
        flush=True,
    )
    return {
        "k": k_value,
        "dataset": "GTSRB",
        "model": "ResNet18",
        "num_clients": NUM_CLIENTS,
        "unseen_clients": NUM_NEW_CLIENTS,
        "unseen_data_ratio": 0.70,
        "malicious_clients": NUM_MALICIOUS,
        "pmr": NUM_MALICIOUS / NUM_CLIENTS,
        "poison_rate": 0.80,
        "epochs": EPOCHS,
        "status": status,
        "epochs_completed": len(records),
        "final_ma": final["ma"],
        "final_ua": final["ua"],
        "final_asr": final["asr"],
        "log_file": str(log_path),
    }


def write_results(rows: list[dict]) -> None:
    fields = [
        "k",
        "dataset",
        "model",
        "num_clients",
        "unseen_clients",
        "unseen_data_ratio",
        "malicious_clients",
        "pmr",
        "poison_rate",
        "epochs",
        "status",
        "epochs_completed",
        "final_ma",
        "final_ua",
        "final_asr",
        "log_file",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value):
        return "N/A" if value is None else f"{float(value):.4f}"

    lines = [
        "# Phase-II MAD-k Ablation",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Fixed setting: GTSRB, ResNet18, 30 clients, 30 rounds, 6 unseen clients, "
        "3 malicious clients, unseen data ratio 70%, PMR 10%, poison rate 80%.",
        "Only the Phase-II MAD interval coefficient k is changed.",
        "",
        "| k | MA | UA | ASR | Status | Log |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda item: float(item["k"])):
        lines.append(
            f"| {row['k']:g} | {fmt(row['final_ma'])} | {fmt(row['final_ua'])} | "
            f"{fmt(row['final_asr'])} | {row['status']} | "
            f"[stdout.log]({row['log_file']}) |"
        )
    lines.extend(
        [
            "",
            "Phase II settings: DCT + L2 + Gram with equal weights, 15-version "
            "history queue, 90% risk quantile, and benign trust floor 0.65.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [run_one(k_value) for k_value in K_VALUES]
    write_results(rows)
    print(f"\nSaved {SUMMARY_CSV}")
    print(f"Saved {SUMMARY_MD}")
    return 0 if all(row["status"] == "completed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
