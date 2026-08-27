#!/usr/bin/env python3
"""Draw reference-style bar charts for eta ablation metrics."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "eta_ablation_results.csv"

OUTPUTS = {
    "CA": {
        "column": "final_ma",
        "color": "#000000",
        "ylabel": "Clean Accuracy",
        "caption": "(a) CA",
        "png": ROOT / "eta_ablation_CA_bar.png",
        "pdf": ROOT / "eta_ablation_CA_bar.pdf",
    },
    "UA": {
        "column": "final_ua",
        "color": "#1f5aa6",
        "ylabel": "Unseen Accuracy",
        "caption": "(b) UA",
        "png": ROOT / "eta_ablation_UA_bar.png",
        "pdf": ROOT / "eta_ablation_UA_bar.pdf",
    },
    "BA": {
        "column": "final_asr",
        "color": "#d1110d",
        "ylabel": "Backdoor Accuracy",
        "caption": "(c) BA",
        "png": ROOT / "eta_ablation_BA_bar.png",
        "pdf": ROOT / "eta_ablation_BA_bar.pdf",
    },
}

COMBINED_PNG = ROOT / "eta_ablation_three_bars.png"

COMBINED_TITLES = {
    "CA": "(a) Main Task Accuracy (MA)",
    "UA": "(b) Unseen Accuracy (UA)",
    "BA": "(c) Backdoor Accuracy (BA)",
}


def read_results() -> list[dict]:
    with INPUT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: float(row["eta"]))


def percent_formatter(value: float, _pos: int) -> str:
    return f"{value:.0f}%"


def set_reference_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax, etas: list[str], ylabel: str, rotate: bool = True, xlabel: str | None = None) -> np.ndarray:
    x = np.arange(len(etas))
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=5)
    ax.set_xticks(x)
    if rotate:
        ax.set_xticklabels(etas, rotation=35, ha="right")
    else:
        ax.set_xticklabels(etas, rotation=0, ha="center")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#111111")
    ax.tick_params(axis="both", width=1.0, length=4)
    return x


def draw_bars(ax, x: np.ndarray, values: np.ndarray, color: str) -> None:
    ax.bar(
        x,
        values,
        width=0.40,
        color=color,
        edgecolor=color,
        linewidth=0,
    )


def draw_metric(etas: list[str], values: np.ndarray, cfg: dict) -> None:
    set_reference_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.45))
    x = style_axis(ax, etas, cfg["ylabel"], rotate=True)
    draw_bars(ax, x, values, cfg["color"])

    fig.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.36)
    fig.text(
        0.5,
        0.045,
        cfg["caption"],
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(cfg["png"], dpi=300, bbox_inches="tight")
    fig.savefig(cfg["pdf"], bbox_inches="tight")
    plt.close(fig)


def draw_combined(etas: list[str], metric_values: dict[str, np.ndarray]) -> None:
    set_reference_style()
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 9.6))
    for ax, metric in zip(axes, ["CA", "UA", "BA"]):
        cfg = OUTPUTS[metric]
        x = style_axis(ax, etas, cfg["ylabel"], rotate=False, xlabel=r"Aggregation Step Size ($\eta$)")
        draw_bars(ax, x, metric_values[metric], cfg["color"])
        ax.text(
            0.5,
            -0.43,
            COMBINED_TITLES[metric],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.08, hspace=0.90)
    fig.savefig(COMBINED_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = read_results()
    etas = [f"{float(row['eta']):.2f}" for row in rows]
    metric_values = {}

    for metric, cfg in OUTPUTS.items():
        values = np.asarray([float(row[cfg["column"]]) * 100.0 for row in rows])
        metric_values[metric] = values
        draw_metric(etas, values, cfg)
        print(f"saved {cfg['png']}")
        print(f"saved {cfg['pdf']}")

    draw_combined(etas, metric_values)
    print(f"saved {COMBINED_PNG}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
