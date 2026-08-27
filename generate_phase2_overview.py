from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle


OUT = Path(__file__).with_name("phase2_overview_reference")

C = {
    "bg": "F8FAFC",
    "navy": "102B5C",
    "text": "1F2937",
    "muted": "667085",
    "border": "C7D2E0",
    "green": "8FC97A",
    "green_soft": "F2FAEE",
    "green_dark": "2E6B1F",
    "blue": "8DBAF1",
    "blue_soft": "F3F8FF",
    "blue_dark": "1F5AA6",
    "red": "F0A19B",
    "red_soft": "FFF5F4",
    "red_dark": "B73B34",
    "purple": "C7A6E6",
    "purple_soft": "FAF5FF",
    "purple_dark": "4C2C78",
    "teal": "93D1C8",
    "teal_soft": "F1FBF9",
}


def _color(value):
    if isinstance(value, str) and len(value) == 6 and all(ch in "0123456789ABCDEFabcdef" for ch in value):
        return f"#{value}"
    return value


def rounded_box(ax, x, y, w, h, fill="FFFFFF", edge=None, lw=1.1, radius=0.015):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            linewidth=lw,
            edgecolor=_color(edge or C["border"]),
            facecolor=_color(fill),
        )
    )


def txt(ax, x, y, s, size=10, color=None, weight="normal", ha="center", va="center", style="normal"):
    ax.text(
        x, y, s,
        fontsize=size,
        color=_color(color or C["text"]),
        fontweight=weight,
        ha=ha,
        va=va,
        style=style,
        family="DejaVu Sans",
    )


def arrow(ax, x1, y1, x2, y2, color=C["navy"], lw=1.5, style="-|>", ms=14):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=_color(color),
            shrinkA=0,
            shrinkB=0,
        )
    )


def mini_dct(ax, x, y, w, h):
    cols, rows = 6, 6
    pad = 0.03
    cell_w = (w - 2 * pad) / cols
    cell_h = (h - 2 * pad) / rows
    for r in range(rows):
        for c in range(cols):
            if r < 2 and c < 2:
                fill = C["blue_dark"]
            elif r < 3 and c < 3:
                fill = C["blue"]
            else:
                fill = "EEF3FB"
            ax.add_patch(Rectangle((x + pad + c * cell_w, y + pad + (rows - 1 - r) * cell_h), cell_w - 0.004, cell_h - 0.004, facecolor=_color(fill), edgecolor=_color("FFFFFF"), linewidth=0.4))


def mini_l2(ax, x, y, w, h):
    for i, height in enumerate([0.25, 0.42, 0.63, 0.9, 0.55]):
        bx = x + 0.08 + i * 0.13
        bw = 0.05
        bh = h * height
        ax.add_patch(Rectangle((bx, y + 0.08), bw, bh, facecolor=_color(C["green_dark"] if i % 2 == 0 else C["green"]), edgecolor="none"))
    ax.plot([x + 0.05, x + w - 0.05], [y + h * 0.15, y + h * 0.15], color=_color(C["muted"]), lw=0.8)
    txt(ax, x + w * 0.58, y + h * 0.72, r'$\|\Delta g_i\|_2$', size=12, color=C["green_dark"], weight="bold")


def mini_gram(ax, x, y, w, h):
    n = 5
    pad = 0.03
    cw = (w - 2 * pad) / n
    ch = (h - 2 * pad) / n
    for r in range(n):
        for c in range(n):
            if c >= r:
                fill = C["purple_dark"] if (r + c) < 3 else C["purple"]
            else:
                fill = "F0EDF7"
            ax.add_patch(Rectangle((x + pad + c * cw, y + pad + (n - 1 - r) * ch), cw - 0.004, ch - 0.004, facecolor=_color(fill), edgecolor=_color("FFFFFF"), linewidth=0.4))


def add_card(ax, x, y, w, h, title, fill, edge, title_color, body_lines, body_size=9.2):
    rounded_box(ax, x, y, w, h, fill=fill, edge=edge, lw=1.0, radius=0.012)
    ax.add_patch(Rectangle((x, y + h - 0.085), w, 0.085, facecolor=_color(edge), edgecolor=_color(edge), linewidth=0))
    txt(ax, x + 0.02, y + h - 0.045, title, size=10.5, color=_color("FFFFFF"), weight="bold", ha="left")
    body = "\n".join(body_lines)
    txt(ax, x + 0.03, y + h * 0.38, body, size=body_size, color=C["text"], ha="left", va="center")


def draw_figure():
    fig = plt.figure(figsize=(18, 9.4), facecolor=_color(C["bg"]))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    txt(
        ax, 0.5, 0.962,
        "Overview of Phase 2: Multi-View Backdoor Risk Fusion and Calibrated Treatment",
        size=20, color=C["navy"], weight="bold"
    )
    txt(
        ax, 0.5, 0.929,
        "Parameter updates are transformed into comparable risk scores using DCT low-frequency energy, gradient L2 norm, and Gram-matrix deviation.",
        size=10.8, color=C["muted"], style="italic"
    )

    # Legend bar
    rounded_box(ax, 0.02, 0.86, 0.96, 0.055, fill="FFFFFF", edge=C["border"], lw=1.0, radius=0.010)
    legend = [
        (0.055, C["green"], C["green_dark"], "Predicted others"),
        (0.235, C["blue"], C["blue_dark"], "Predicted unseen"),
        (0.425, C["red"], C["red_dark"], "Malicious / backdoor"),
        (0.62, C["teal"], C["purple_dark"], "Trusted probes / history"),
    ]
    for x, fill, edge, label in legend:
        ax.add_patch(Circle((x, 0.888), 0.0085, facecolor=_color(fill), edgecolor=_color(edge), linewidth=0.9))
        txt(ax, x + 0.02, 0.888, label, size=10.2, color=C["text"], ha="left")

    # Input / gate box
    rounded_box(ax, 0.03, 0.49, 0.18, 0.32, fill="FFFFFF", edge=C["border"], lw=1.1, radius=0.012)
    txt(ax, 0.12, 0.78, "Phase 1 output", size=11.5, color=C["navy"], weight="bold")
    rounded_box(ax, 0.045, 0.66, 0.15, 0.085, fill=C["green_soft"], edge=C["green"], lw=0.9, radius=0.010)
    txt(ax, 0.12, 0.703, "Others", size=11, color=C["green_dark"], weight="bold")
    rounded_box(ax, 0.045, 0.56, 0.15, 0.085, fill=C["blue_soft"], edge=C["blue"], lw=0.9, radius=0.010)
    txt(ax, 0.12, 0.603, "Unseen", size=11, color=C["blue_dark"], weight="bold")
    txt(ax, 0.12, 0.515, r"Local client update $\Delta g_i$", size=10.2, color=C["text"], weight="bold")

    arrow(ax, 0.21, 0.68, 0.25, 0.68, color=C["navy"], lw=1.6)

    # Feature extraction panel
    rounded_box(ax, 0.25, 0.49, 0.33, 0.36, fill=C["green_soft"], edge=C["green"], lw=1.1, radius=0.014)
    txt(ax, 0.415, 0.83, "Feature View Construction", size=12.6, color=C["green_dark"], weight="bold")

    # DCT card
    add_card(
        ax, 0.266, 0.71, 0.298, 0.11,
        "(1) DCT low-frequency score",
        fill="FFFFFF", edge=C["green"], title_color=C["green_dark"],
        body_lines=[r"$\Delta g_i \rightarrow \mathrm{reshape} \rightarrow \mathrm{DCT}$", r"$\rightarrow$ low-frequency block $\rightarrow s_i^{\mathrm{DCT}} = \|\cdot\|_1$"],
        body_size=9.0,
    )
    mini_dct(ax, 0.278, 0.724, 0.070, 0.070)

    # L2 card
    add_card(
        ax, 0.266, 0.59, 0.298, 0.11,
        "(2) Gradient magnitude",
        fill="FFFFFF", edge=C["green"], title_color=C["green_dark"],
        body_lines=[r"$s_i^{\mathrm{L2}} = \|\Delta g_i\|_2$", "Large deviation indicates an atypical update scale."],
        body_size=9.1,
    )
    mini_l2(ax, 0.276, 0.604, 0.078, 0.066)

    # Gram card
    add_card(
        ax, 0.266, 0.51, 0.298, 0.074,
        "(3) Gram deviation on trusted probes",
        fill="FFFFFF", edge=C["green"], title_color=C["green_dark"],
        body_lines=[r"$G_i = \phi(X_{\mathrm{probe}})\,\phi(X_{\mathrm{probe}})^\top$"],
        body_size=9.0,
    )
    mini_gram(ax, 0.446, 0.516, 0.088, 0.054)
    txt(ax, 0.296, 0.524, r"Deviation from clean Gram statistics", size=8.8, color=C["text"], ha="left")

    arrow(ax, 0.58, 0.67, 0.62, 0.67, color=C["navy"], lw=1.6)

    # Calibration panel
    rounded_box(ax, 0.63, 0.49, 0.34, 0.36, fill=C["purple_soft"], edge=C["purple"], lw=1.1, radius=0.014)
    txt(ax, 0.80, 0.83, "Robust Calibration and Thresholding", size=12.6, color=C["purple_dark"], weight="bold")

    # History queue strip
    rounded_box(ax, 0.646, 0.77, 0.308, 0.06, fill="FFFFFF", edge=C["purple"], lw=0.9, radius=0.010)
    txt(ax, 0.66, 0.80, "History queue of latest global versions", size=8.9, color=C["purple_dark"], weight="bold", ha="left")
    for i, lab in enumerate([r"$M_{t-k}$", r"$\cdots$", r"$M_t$"]):
        bx = 0.80 + i * 0.06
        rounded_box(ax, bx, 0.786, 0.045, 0.022, fill=C["blue_soft"] if i < 2 else C["purple_soft"], edge=C["border"], lw=0.7, radius=0.007)
        txt(ax, bx + 0.0225, 0.797, lab, size=8.4, color=C["navy"], weight="bold")
    txt(ax, 0.66, 0.778, "Used for stable min/max or bootstrap calibration", size=8.2, color=C["muted"], ha="left")

    # Normalization box
    rounded_box(ax, 0.646, 0.66, 0.308, 0.095, fill="FFFFFF", edge=C["purple"], lw=0.9, radius=0.010)
    txt(ax, 0.66, 0.735, "Normalize each metric", size=9.4, color=C["purple_dark"], weight="bold", ha="left")
    txt(ax, 0.66, 0.697, r"$\tilde{s}_i^m = \mathrm{Normalize}(s_i^m;\, \mathrm{MAD}$ or history range$)$", size=9.7, color=C["text"], ha="left")

    # Fusion box
    rounded_box(ax, 0.646, 0.55, 0.308, 0.095, fill="FFFFFF", edge=C["purple"], lw=0.9, radius=0.010)
    txt(ax, 0.66, 0.625, "Weighted risk fusion", size=9.4, color=C["purple_dark"], weight="bold", ha="left")
    txt(ax, 0.66, 0.587, r"$r_i = w_d\tilde{s}_i^{\mathrm{DCT}} + w_l\tilde{s}_i^{\mathrm{L2}} + w_g\tilde{s}_i^{\mathrm{Gram}}$", size=9.6, color=C["text"], ha="left")

    # Threshold box
    rounded_box(ax, 0.646, 0.50, 0.308, 0.04, fill=C["purple_soft"], edge=C["purple"], lw=0.8, radius=0.009)
    txt(ax, 0.80, 0.520, r"Group threshold:  $r_i > \tau_{\mathrm{group}}$  $\Rightarrow$ backdoor", size=9.0, color=C["purple_dark"], weight="bold")

    arrow(ax, 0.97, 0.67, 0.98, 0.67, color=C["navy"], lw=1.6)

    # Treatment policy panel
    rounded_box(ax, 0.11, 0.13, 0.78, 0.26, fill="FFFFFF", edge=C["border"], lw=1.1, radius=0.014)
    txt(ax, 0.50, 0.372, "Client Treatment Policy", size=12.2, color=C["navy"], weight="bold")
    txt(ax, 0.50, 0.347, "Combining phase-1 group prediction and phase-2 risk score", size=9.5, color=C["muted"], style="italic")

    cards = [
        (0.13, 0.165, 0.18, 0.165, C["green_soft"], C["green"], C["green_dark"], "Others + Benign", "Retain update", "✓"),
        (0.325, 0.165, 0.18, 0.165, C["blue_soft"], C["blue"], C["blue_dark"], "Unseen + Benign", ["Retain with", "benign-preserving weighting"], "✓"),
        (0.52, 0.165, 0.18, 0.165, C["red_soft"], C["red"], C["red_dark"], "Others + Backdoor", "Discard update", "✕"),
        (0.715, 0.165, 0.18, 0.165, C["purple_soft"], C["purple"], C["purple_dark"], "Unseen + Backdoor", ["Soft calibrate", "toward clean reference"], "⚙"),
    ]
    for x, y, w, h, fill, edge, title_c, title, body, symbol in cards:
        add_card(ax, x, y, w, h, title, fill, edge, title_c, body if isinstance(body, list) else [body], body_size=8.8)
        txt(ax, x + w - 0.028, y + 0.037, symbol, size=18, color=title_c, weight="bold")

    # Connectors
    arrow(ax, 0.42, 0.49, 0.42, 0.40, color=C["muted"], lw=1.3)
    arrow(ax, 0.80, 0.49, 0.80, 0.40, color=C["muted"], lw=1.3)
    arrow(ax, 0.80, 0.55, 0.80, 0.43, color=C["muted"], lw=1.0)

    # Bottom note
    txt(
        ax, 0.5, 0.05,
        "Phase 2 is designed to be stable under non-IID client updates by comparing each client against a clean reference distribution rather than a single threshold.",
        size=8.8, color=C["muted"], style="italic"
    )

    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw_figure()
