#!/usr/bin/env python3
"""Run a four-value eta ablation for the current RobustSL implementation."""
from __future__ import annotations

import csv
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFENSE = ROOT / "Defense"
PYTHON = os.environ.get("ETA_ABLATION_PYTHON") or __import__("sys").executable
CONFIG = DEFENSE / "configs" / "gtsrb.yaml"
RUN_ROOT = ROOT / "eta_ablation_runs"
SUMMARY_CSV = ROOT / "eta_ablation_results.csv"
SUMMARY_MD = ROOT / "eta_ablation_results.md"

ETA_VALUES = [0.25, 0.50, 0.75, 1.00]
GPU_IDS = [0, 1, 2, 3]

NUM_CLIENTS = 30
NUM_NEW_CLIENTS = 6
NUM_MALICIOUS = 3
UNSEEN_RATIO = 0.70
POISON_RATE = 0.80
EPOCHS = 30

EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")
EVAL_RE = re.compile(
    r"\[Eval\]\s+MA:\s*([0-9eE+\-.]+)\s*\|\s*"
    r"Unseen Accuracy:\s*([0-9eE+\-.]+)\s*\|\s*"
    r"ASR:\s*([0-9eE+\-.]+|N/A)"
)


def command_for(eta: float) -> list[str]:
    return [
        PYTHON,
        "-u",
        str(DEFENSE / "run_RobustSL.py"),
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
        str(UNSEEN_RATIO),
        "--num-malicious",
        str(NUM_MALICIOUS),
        "--attack-type",
        "trigger",
        "--poison-rate",
        str(POISON_RATE),
        "--attack-scale",
        "1.0",
        "--enable-defense",
        "--steps-per-client",
        "20",
        "--batch-size",
        "64",
        "--root-size",
        "1000",
        "--root-pretrain-steps",
        "500",
        "--phase2-probe-batches",
        "1",
        "--phase2-gram-orders",
        "1",
        "2",
        "--phase2-mad-k",
        "1.5",
        "--phase2-bootstrap-rounds",
        "100",
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
        str(eta),
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


def print_progress(eta: float, line: str) -> None:
    if (
        line.startswith("Epoch ")
        or "[Eval]" in line
        or "[Data]" in line
        or "[Attack]" in line
        or "[RootPretrain]" in line
        or "[Phase2]" in line
    ):
        print(f"[eta={eta:g}] {line}", flush=True)


def run_one(task: tuple[float, int]) -> dict:
    eta, gpu_id = task
    label = f"eta_{eta:g}".replace(".", "_")
    run_dir = RUN_ROOT / label
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"
    command = command_for(eta)

    print(f"\n[START] eta={eta:g} | GPU={gpu_id}", flush=True)
    print("[CMD] " + " ".join(command), flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

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
            print_progress(eta, line.rstrip())
        return_code = process.wait()

    records = parse_log(log_path)
    final = records[-1] if records else {
        "ma": None,
        "ua": None,
        "asr": None,
    }
    status = "completed" if return_code == 0 and len(records) == EPOCHS else "failed"
    print(
        f"[DONE] eta={eta:g} | status={status} | rounds={len(records)} | "
        f"MA={final['ma']} | UA={final['ua']} | ASR={final['asr']}",
        flush=True,
    )

    return {
        "eta": eta,
        "gpu": gpu_id,
        "dataset": "GTSRB",
        "model": "ResNet18",
        "num_clients": NUM_CLIENTS,
        "unseen_clients": NUM_NEW_CLIENTS,
        "unseen_data_ratio": UNSEEN_RATIO,
        "malicious_clients": NUM_MALICIOUS,
        "pmr": NUM_MALICIOUS / NUM_CLIENTS,
        "poison_rate": POISON_RATE,
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
        "eta",
        "gpu",
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
        writer.writerows(sorted(rows, key=lambda row: float(row["eta"])))

    def fmt(value):
        return "N/A" if value is None else f"{float(value):.4f}"

    lines = [
        "# Eta Ablation on RobustSL",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Only the global aggregation step size eta is changed.",
        "",
        "## Fixed Configuration",
        "",
        "- Dataset: GTSRB with China-GTSRB/TT100K unseen data.",
        "- Model: ResNet18 split learning architecture.",
        "- Total clients: 30; unseen clients: 6; unseen data ratio: 70%.",
        "- Malicious clients: 3; PMR: 10%; poison rate: 80%; trigger attack.",
        "- Training: 30 rounds; 20 local steps; batch size 64.",
        "- Defense: current RobustSL Phase I and Phase II implementation.",
        "- Phase II: DCT, L2, and Gram views with equal weights.",
        "- Root pretraining: 1000 samples and 500 clean steps.",
        "",
        "## Final Results",
        "",
        "| eta | MA | UA | ASR | Status | Log |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda item: float(item["eta"])):
        lines.append(
            f"| {float(row['eta']):.2f} | {fmt(row['final_ma'])} | "
            f"{fmt(row['final_ua'])} | {fmt(row['final_asr'])} | "
            f"{row['status']} | [stdout.log]({row['log_file']}) |"
        )

    qualified = [
        row for row in rows
        if row["status"] == "completed"
        and row["final_ma"] is not None
        and row["final_ua"] is not None
        and row["final_asr"] is not None
        and float(row["final_ma"]) >= 0.90
        and float(row["final_ua"]) >= 0.90
        and float(row["final_asr"]) < 0.01
    ]
    lines.extend(["", "## Target Check", ""])
    if qualified:
        lines.append(
            "Qualified eta values (MA >= 0.90, UA >= 0.90, ASR < 0.01): "
            + ", ".join(f"`{float(row['eta']):.2f}`" for row in sorted(qualified, key=lambda item: float(item["eta"])))
        )
        best = max(
            qualified,
            key=lambda row: (
                (float(row["final_ma"]) + float(row["final_ua"])) / 2.0,
                -float(row["final_asr"]),
            ),
        )
        lines.append(
            f"Best qualified eta by mean(MA, UA): `{float(best['eta']):.2f}` "
            f"(MA={fmt(best['final_ma'])}, UA={fmt(best['final_ua'])}, "
            f"ASR={fmt(best['final_asr'])})."
        )
    else:
        lines.append(
            "No eta value simultaneously satisfies MA >= 0.90, UA >= 0.90, "
            "and ASR < 0.01 in the completed runs."
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = list(zip(ETA_VALUES, GPU_IDS))
    rows = []

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            write_results(rows)
            print(
                f"[SUMMARY] completed {len(rows)}/{len(tasks)}; "
                f"saved {SUMMARY_CSV.name} and {SUMMARY_MD.name}",
                flush=True,
            )

    write_results(rows)
    return 0 if all(row["status"] == "completed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
