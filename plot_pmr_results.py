#!/usr/bin/env python3
"""Plot MA, UA and BA against the malicious-client ratio."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
PNG_PATH = ROOT / "PMR.png"
PDF_PATH = ROOT / "PMR.pdf"

# Values reproduced from the original PMR figure.
pmr = [10, 20, 30, 40]
ma = [94.0, 94.0, 95.0, 94.0]
ua = [96.0, 96.0, 94.0, 96.0]
ba = [0.0, 0.0, 3.0, 0.0]

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 16,
    "axes.labelsize": 20,
    "legend.fontsize": 15,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

fig, ax = plt.subplots(figsize=(11.4, 7.2), dpi=180)
ax.plot(
    pmr, ma, marker="o", linewidth=3.0, markersize=9,
    label="MA", color="#000000",
)
ax.plot(
    pmr, ua, marker="s", linewidth=3.0, markersize=9,
    label="UA", color="#1f5aa6",
)
ax.plot(
    pmr, ba, marker="^", linewidth=3.0, markersize=9,
    label="BA", color="#FF0000",
)

ax.set_xticks(pmr)
ax.set_xlabel("PMR (%)", labelpad=12)
ax.set_ylabel("Accuracy (%)", labelpad=14)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax.set_ylim(-2, 102)
ax.set_xlim(8, 42)
ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.35)
ax.tick_params(axis="both", which="major", labelsize=16, width=1.2, length=6)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 0.05),
    ncol=3,
    frameon=True,
    framealpha=0.95,
    columnspacing=1.0,
    handlelength=2.2,
)

fig.tight_layout()
fig.savefig(PNG_PATH, dpi=320, bbox_inches="tight")
fig.savefig(PDF_PATH, bbox_inches="tight")
plt.close(fig)

print(f"Saved {PNG_PATH}")
print(f"Saved {PDF_PATH}")
