import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================
# FONT CONFIGURATION
# ============================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans"],

    # Mantiene il testo modificabile/vettoriale nel PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# PARAMETRI MODIFICABILI
# ============================================================

# Dimensioni figura
figure_width = 10
figure_height = 4.5

# Dimensioni font
title_fontsize = 22
axis_label_fontsize = 18
tick_fontsize = 16
bar_value_fontsize = 16
legend_fontsize = 15

# Spessore barre
bar_width = 0.28

# Offset verticale dei numeri sopra le barre
bar_value_offset = 2

# Limiti asse Y
y_min = 0
y_max = 110

# Griglia orizzontale
grid_linewidth = 0.8
grid_alpha = 0.35

# Colori
proposed_color = "#006D7A"
baseline_color = "#7A7A7A"

# Testi
title = "Scenario Reliability (%)"
y_label = "Success Rate (%)"

# Nome file di output
output_filename = "scenario_reliability.pdf"

# ============================================================
# DATI
# ============================================================

scenarios = [
    "BoxesAside",
    "StackedBoxes",
    "CupsOnBox",
    "TennisBall"
]

proposed = [100, 100, 68, 100]
baseline = [76, 88, 52, 72]

# ============================================================
# PLOT
# ============================================================

x = np.arange(len(scenarios))

fig, ax = plt.subplots(figsize=(figure_width, figure_height))

bars_proposed = ax.bar(
    x - bar_width / 2,
    proposed,
    bar_width,
    label="Proposed",
    color=proposed_color
)

bars_baseline = ax.bar(
    x + bar_width / 2,
    baseline,
    bar_width,
    label="Baseline",
    color=baseline_color
)

# Titolo e assi
ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
ax.set_ylabel(y_label, fontsize=axis_label_fontsize)

ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=tick_fontsize)
ax.tick_params(axis="y", labelsize=tick_fontsize)

ax.set_ylim(y_min, y_max)

# Righe orizzontali
ax.grid(
    axis="y",
    linestyle="-",
    linewidth=grid_linewidth,
    alpha=grid_alpha
)
ax.set_axisbelow(True)

# Valori sopra le barre
for bars in [bars_proposed, bars_baseline]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + bar_value_offset,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=bar_value_fontsize,
            fontweight="bold"
        )

# Legenda
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    fontsize=legend_fontsize,
    frameon=False
)

# Pulizia estetica
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# Salvataggio in PDF
plt.savefig(output_filename, format="pdf", bbox_inches="tight")

plt.show()