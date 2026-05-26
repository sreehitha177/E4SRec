import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

CONFIGS    = ["Only", "+ Audio/\nLyric/Meta", "+ Audio/Lyric\n/Meta + CR"]
SEQ_MODELS = ["SASRec", "BERT4Rec", "GRU4Rec"]
LLMS       = ["Qwen2.5-7B", "Llama-2-13B", "Llama-3-70B"]

data = {
    "Table": {
        "Qwen2.5-7B": {
            "SASRec":   [[0.019, 0.012, 0.041, 0.019, 0.083, 0.030],
                         [0.024, 0.014, 0.045, 0.021, 0.097, 0.034],
                         [0.016, 0.010, 0.039, 0.017, 0.088, 0.029]],
            "BERT4Rec": [[0.022, 0.015, 0.050, 0.024, 0.107, 0.038],
                         [0.031, 0.017, 0.063, 0.027, 0.121, 0.041],
                         [0.031, 0.019, 0.055, 0.026, 0.102, 0.038]],
            "GRU4Rec":  [[0.036, 0.022, 0.069, 0.033, 0.127, 0.047],
                         [0.022, 0.013, 0.044, 0.020, 0.097, 0.033],
                         [0.017, 0.010, 0.034, 0.015, 0.083, 0.027]],
        },
        "Llama-2-13B": {
            "SASRec":   [[0.019, 0.011, 0.045, 0.019, 0.083, 0.029],
                         [0.038, 0.022, 0.078, 0.035, 0.116, 0.045],
                         [0.013, 0.007, 0.036, 0.014, 0.088, 0.027]],
            "BERT4Rec": [[0.027, 0.013, 0.064, 0.025, 0.108, 0.036],
                         [0.024, 0.015, 0.045, 0.023, 0.091, 0.034],
                         [0.033, 0.019, 0.058, 0.028, 0.102, 0.039]],
            "GRU4Rec":  [[0.034, 0.023, 0.058, 0.030, 0.121, 0.045],
                         [0.031, 0.016, 0.049, 0.022, 0.099, 0.034],
                         [0.031, 0.019, 0.063, 0.030, 0.107, 0.041]],
        },
        "Llama-3-70B": {
            "SASRec":   [[0.020, 0.011, 0.053, 0.021, 0.103, 0.034],
                         [0.024, 0.016, 0.053, 0.025, 0.119, 0.042],
                         [0.020, 0.015, 0.049, 0.024, 0.091, 0.034]],
            "BERT4Rec": [[0.019, 0.011, 0.038, 0.017, 0.086, 0.029],
                         [0.022, 0.013, 0.047, 0.021, 0.107, 0.036],
                         [0.039, 0.025, 0.063, 0.033, 0.124, 0.048]],
            "GRU4Rec":  [[0.020, 0.014, 0.047, 0.022, 0.103, 0.036],
                         [0.033, 0.020, 0.056, 0.028, 0.103, 0.039],
                         [0.025, 0.018, 0.049, 0.025, 0.114, 0.042]],
        },
    },
}

COLORS    = ["#4C72B0", "#DD8452", "#55A868"]
MARKERS   = ["o", "s", "^"]
LINESTYLE = ["-", "--", ":"]

os.makedirs("figures", exist_ok=True)

x = np.arange(len(CONFIGS))

PANELS = [
    (0, 0, 0, "Recall@5"),
    (0, 1, 2, "Recall@10"),
    (0, 2, 4, "Recall@20"),
    (1, 0, 1, "NDCG@5"),
    (1, 1, 3, "NDCG@10"),
    (1, 2, 5, "NDCG@20"),
]

def make_line_chart(table_name, llm):
    fig, axes = plt.subplots(2, 3, figsize=(12.87, 7.49), sharey=False)
    fig.suptitle(f"{llm}", fontsize=18, fontweight="bold", y=1.01)

    for row, col, mi, metric_label in PANELS:
        ax = axes[row, col]

        all_vals = []
        for seq_idx, (seq, color, marker, ls) in enumerate(zip(SEQ_MODELS, COLORS, MARKERS, LINESTYLE)):
            vals = [data[table_name][llm][seq][ci][mi] for ci in range(len(CONFIGS))]
            all_vals.append((vals, seq_idx, color))
            ax.plot(x, vals, color=color, marker=marker, linestyle=ls,
                    linewidth=2.5, markersize=8,
                    label=seq if (row, col) == (0, 0) else "")

        # Find the single global peak across all series and configs in this panel
        peak_val, peak_xi, peak_si = max(
            (v, xi, si)
            for vals, si, _ in all_vals
            for xi, v in enumerate(vals)
        )
        peak_color = COLORS[peak_si]
        # Place the peak label above with a star marker highlight
        ax.annotate(f"{peak_val:.3f}★",
                    (peak_xi, peak_val),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold",
                    color=peak_color)

        ax.set_xticks(x)
        ax.set_xticklabels(CONFIGS, fontsize=13)
        ax.set_title(metric_label, fontsize=15, fontweight="bold")
        ax.set_ylabel("Score", fontsize=13)
        ax.set_xlim(-0.3, len(CONFIGS) - 0.7)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("black")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.yaxis.set_tick_params(labelsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    slug = f"line_{table_name.replace(' ', '')}_{llm.replace('.', '').replace('-', '')}_peak"
    fig.savefig(f"figures/{slug}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"figures/{slug}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{slug}.{{pdf,png}}")


for table_name in data:
    print(f"\n{table_name}")
    for llm in LLMS:
        make_line_chart(table_name, llm)

print("\nDone.")
