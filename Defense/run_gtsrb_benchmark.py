#!/usr/bin/env python3
import csv
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parent
PYTHON = os.environ.get("BENCHMARK_PYTHON") or sys.executable
CONFIG = REPO / "configs" / "gtsrb.yaml"
RUN_ROOT = REPO / "architecture_benchmark_runs"
SUMMARY_MD = REPO / "architecture_benchmark_results.md"
SUMMARY_CSV = REPO / "architecture_benchmark_results.csv"
PER_ROUND_CSV = REPO / "architecture_benchmark_per_round.csv"

MODELS = [
    "resnet18",
    "resnet34",
    "googlenet",
    "vgg11",
    "wide_resnet50",
    "micronnet",
]
CONDITIONS = [
    ("no_attack_no_defense", "No attack + no defense"),
    ("attack_no_defense", "Attack + no defense"),
    ("attack_defense", "Attack + defense"),
]
GPUS = [int(x) for x in os.environ.get("BENCHMARK_GPUS", "0,1,2,3").split(",") if x.strip()]
MAX_WORKERS = max(1, min(len(GPUS), int(os.environ.get("BENCHMARK_WORKERS", str(len(GPUS))))))
EPOCHS = 30

EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")
EVAL_RE = re.compile(
    r"\[Eval\]\s+MA:\s*([0-9eE+\-.]+)\s*\|\s*"
    r"Unseen Accuracy:\s*([0-9eE+\-.]+)\s*\|\s*"
    r"ASR:\s*(N/A|[0-9eE+\-.]+)"
)

print_lock = threading.Lock()
summary_lock = threading.Lock()


def safe_print(message):
    with print_lock:
        print(message, flush=True)


def run_id(model, condition):
    return f"{model}__{condition}"


def make_run_dir(run_name):
    run_dir = RUN_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data_link = run_dir / "data"
    if not data_link.exists():
        data_link.symlink_to(REPO / "data", target_is_directory=True)
    return run_dir


def command_for(model, condition):
    command = [
        PYTHON,
        str(REPO / "run_gtsrb.py"),
        "--config", str(CONFIG),
        "--model", model,
        "--epochs", str(EPOCHS),
        "--num-clients", "30",
    ]
    if condition == "no_attack_no_defense":
        command.append("--disable-attack")
    elif condition == "attack_defense":
        command.extend([
            "--enable-defense",
            "--phase2-history-range",
            "--phase2-history-size", "15",
            "--phase2-history-min-versions", "3",
        ])
    return command


def parse_metrics(lines):
    current_epoch = None
    evaluations = []
    for line in lines:
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        eval_match = EVAL_RE.search(line)
        if not eval_match:
            continue
        asr_text = eval_match.group(3)
        evaluations.append({
            "epoch": current_epoch if current_epoch is not None else len(evaluations) + 1,
            "ma": float(eval_match.group(1)),
            "clean_accuracy": float(eval_match.group(1)),
            "unseen_accuracy": float(eval_match.group(2)),
            "asr": None if asr_text == "N/A" else float(asr_text),
        })
    return evaluations


def write_per_round_csv(records):
    fieldnames = [
        "model", "condition", "epoch", "ma", "clean_accuracy",
        "unseen_accuracy", "asr",
    ]
    with PER_ROUND_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (item["model"], item["condition"], item["epoch"])):
            writer.writerow(record)


def write_summary(records):
    ordered = sorted(records, key=lambda item: (item["model"], item["condition"]))
    fieldnames = [
        "model", "condition", "status", "return_code", "epochs_completed",
        "final_ma", "final_clean_accuracy", "final_unseen_accuracy", "final_asr",
        "best_ma", "best_unseen_accuracy", "min_asr", "log_file",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in ordered:
            writer.writerow(item)

    lines = [
        "# GTSRB Model Architecture Benchmark",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Each model was configured for 30 federated rounds and 30 clients.",
        "`MA` and `Clean Accuracy` are identical in the current implementation because both use `eval_clean_accuracy`.",
        "For the no-attack condition, ASR is `N/A` because no malicious trigger is injected.",
        "The defense condition enables the 15-version historical model-range Phase-2 logic.",
        "",
        "## Final Results",
        "",
        "| Model | Condition | Status | Rounds | MA | Clean Accuracy | Unseen Accuracy | ASR | Log |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in ordered:
        def fmt(value):
            return "N/A" if value is None else f"{value:.4f}"
        log_link = f"[{Path(item['log_file']).name}]({item['log_file']})"
        lines.append(
            f"| {item['model']} | {item['condition']} | {item['status']} | "
            f"{item['epochs_completed']} | {fmt(item['final_ma'])} | "
            f"{fmt(item['final_clean_accuracy'])} | {fmt(item['final_unseen_accuracy'])} | "
            f"{fmt(item['final_asr'])} | {log_link} |"
        )

    lines.extend([
        "",
        "## Conditions",
        "",
        "- `no_attack_no_defense`: `--disable-attack`, no defense flag.",
        "- `attack_no_defense`: default trigger attack, no defense flag.",
        "- `attack_defense`: trigger attack plus `--enable-defense --phase2-history-range --phase2-history-size 15`.",
        "",
        "## Files",
        "",
        f"- Full logs: `{RUN_ROOT}`",
        f"- Per-round metrics: `{PER_ROUND_CSV}`",
        f"- Final summary CSV: `{SUMMARY_CSV}`",
    ])
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(model, condition, gpu):
    condition_label = dict(CONDITIONS)[condition]
    name = run_id(model, condition)
    run_dir = make_run_dir(name)
    log_path = run_dir / "stdout.log"
    command = command_for(model, condition)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"

    safe_print(f"[START] {name} | GPU {gpu} | {condition_label}")
    lines = []
    return_code = -1
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND: " + " ".join(command) + "\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            lines.append(line)
            log_handle.write(raw_line)
            log_handle.flush()
            if EPOCH_RE.search(line) or "[Eval]" in line:
                safe_print(f"[{name}] {line.strip()}")
        return_code = process.wait()

    evaluations = parse_metrics(lines)
    final = evaluations[-1] if evaluations else {
        "epoch": 0,
        "ma": None,
        "clean_accuracy": None,
        "unseen_accuracy": None,
        "asr": None,
    }
    completed = len(evaluations)
    result = {
        "model": model,
        "condition": condition,
        "status": "completed" if return_code == 0 and completed > 0 else "failed",
        "return_code": return_code,
        "epochs_completed": completed,
        "final_ma": final["ma"],
        "final_clean_accuracy": final["clean_accuracy"],
        "final_unseen_accuracy": final["unseen_accuracy"],
        "final_asr": final["asr"],
        "best_ma": max((item["ma"] for item in evaluations), default=None),
        "best_unseen_accuracy": max((item["unseen_accuracy"] for item in evaluations), default=None),
        "min_asr": min((item["asr"] for item in evaluations if item["asr"] is not None), default=None),
        "log_file": str(log_path),
    }
    round_records = [
        {
            "model": model,
            "condition": condition,
            **item,
        }
        for item in evaluations
    ]

    safe_print(
        f"[DONE] {name} | status={result['status']} | rounds={completed}/{EPOCHS} | "
        f"MA={result['final_ma']} | Unseen={result['final_unseen_accuracy']} | ASR={result['final_asr']}"
    )
    return result, round_records


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    total = len(MODELS) * len(CONDITIONS)
    safe_print(f"[BENCHMARK] {total} runs, {EPOCHS} rounds each, GPUs={GPUS}, workers={MAX_WORKERS}")
    safe_print(f"[BENCHMARK] Python={PYTHON}")
    safe_print(f"[BENCHMARK] Summary will be updated in {SUMMARY_MD}")

    all_results = []
    all_round_records = []
    jobs = []
    job_index = 0
    for model in MODELS:
        for condition, _ in CONDITIONS:
            jobs.append((model, condition, GPUS[job_index % len(GPUS)]))
            job_index += 1

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_one, model, condition, gpu) for model, condition, gpu in jobs]
        for future in as_completed(futures):
            result, round_records = future.result()
            with summary_lock:
                all_results.append(result)
                all_round_records.extend(round_records)
                write_summary(all_results)
                write_per_round_csv(all_round_records)
                safe_print(f"[PROGRESS] completed={len(all_results)}/{total}")

    write_summary(all_results)
    write_per_round_csv(all_round_records)
    failed = [item for item in all_results if item["status"] != "completed"]
    safe_print(f"[BENCHMARK] Finished. completed={total - len(failed)}/{total}, failed={len(failed)}")
    safe_print(f"[BENCHMARK] Markdown: {SUMMARY_MD}")
    safe_print(f"[BENCHMARK] CSV: {SUMMARY_CSV}")
    if failed:
        safe_print("[BENCHMARK] Failed runs: " + ", ".join(run_id(item["model"], item["condition"]) for item in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
