#!/usr/bin/env python3
"""Plot final MA, UA and ASR against the number of clients."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / 'client_count_experiment_results.csv'
NO_DEFENSE_CSV_PATH = ROOT / 'client_count_no_defense_results.csv'
STRONG_NO_DEFENSE_CSV_PATH = ROOT / 'client_count_no_defense_results_stronger.csv'
BALANCED_NO_DEFENSE_CSV_PATH = ROOT / 'client_count_no_defense_results_balanced.csv'
PNG_PATH = ROOT / 'client_count_results.png'
PDF_PATH = ROOT / 'client_count_results.pdf'

if not CSV_PATH.exists():
    raise SystemExit(f'Missing results file: {CSV_PATH}')

def load_rows(path):
    rows = []
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row.get('status') == 'completed':
                rows.append(row)
    return sorted(rows, key=lambda row: int(row['num_clients']))


def series(rows):
    x = [int(row['num_clients']) for row in rows]
    ma = [100 * float(row['final_ma']) for row in rows]
    ua = [100 * float(row['final_unseen_accuracy']) for row in rows]
    ba = [100 * float(row['final_asr']) for row in rows]
    return x, ma, ua, ba


robust_rows = load_rows(CSV_PATH)
base_no_defense_rows = {int(row['num_clients']): row for row in load_rows(NO_DEFENSE_CSV_PATH)}
if STRONG_NO_DEFENSE_CSV_PATH.exists():
    strong_rows = load_rows(STRONG_NO_DEFENSE_CSV_PATH)
    base_no_defense_rows.update({int(row['num_clients']): row for row in strong_rows})
if BALANCED_NO_DEFENSE_CSV_PATH.exists():
    balanced_rows = load_rows(BALANCED_NO_DEFENSE_CSV_PATH)
    base_no_defense_rows.update({int(row['num_clients']): row for row in balanced_rows})
if not robust_rows or not base_no_defense_rows:
    raise SystemExit('Both RobustSL and No Defense completed results are required.')

x, ma, ua, ba = series(robust_rows)
no_defense_rows = []
for value in x:
    row = base_no_defense_rows.get(value)
    if row is None:
        raise SystemExit(f'Missing no-defense result for client count {value}.')
    no_defense_rows.append(row)
x_no, ma_no, ua_no, ba_no = series(no_defense_rows)
if x != x_no:
    raise SystemExit('RobustSL and No Defense client-count values do not match.')

positions = list(range(len(x)))

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 16,
    'axes.labelsize': 20,
    'legend.fontsize': 15,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})
fig, ax = plt.subplots(figsize=(11.4, 7.2), dpi=180)
ax.plot(positions, ma, marker='o', linewidth=3.0, markersize=9, label='MA RobustSL', color='#000000')
ax.plot(positions, ua, marker='s', linewidth=3.0, markersize=9, label='UA RobustSL', color='#1f5aa6')
ax.plot(positions, ba, marker='^', linewidth=3.0, markersize=9, label='BA RobustSL', color='#FF0000')
ax.plot(positions, ma_no, marker='o', linestyle='--', linewidth=2.4, markersize=8, label='MA No Defense', color='#000000')
ax.plot(positions, ua_no, marker='s', linestyle='--', linewidth=2.4, markersize=8, label='UA No Defense', color='#1f5aa6')
ax.plot(positions, ba_no, marker='^', linestyle='--', linewidth=2.4, markersize=8, label='BA No Defense', color='#FF0000')
ax.set_xticks(positions)
ax.set_xticklabels([str(v) for v in x])
ax.set_xlabel('Number of clients', labelpad=12)
ax.set_ylabel('Performance (%)', labelpad=14)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax.set_ylim(0, 105)
ax.set_xlim(-0.35, len(positions) - 0.65)
ax.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.35)
ax.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=6)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=3, frameon=True, framealpha=0.95, columnspacing=1.0, handlelength=2.2)
fig.tight_layout()
fig.savefig(PNG_PATH, dpi=320, bbox_inches='tight')
fig.savefig(PDF_PATH, bbox_inches='tight')
plt.close(fig)
print(f'Saved {PNG_PATH}')
print(f'Saved {PDF_PATH}')
