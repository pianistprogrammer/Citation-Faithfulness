from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "paper" / "figures"

MODEL_ORDER = [
    ("meta-llama--llama-3.1-8b-instruct", "Llama"),
    ("qwen--qwen2.5-7b-instruct", "Qwen"),
    ("google--gemma-3-12b-it", "Gemma"),
]

LABEL_ORDER = ["PARAMETRIC_MATCH", "AMBIGUOUS", "CONTEXT_MATCH"]
LABEL_NAMES = {
    "PARAMETRIC_MATCH": "Parametric match",
    "AMBIGUOUS": "Ambiguous",
    "CONTEXT_MATCH": "Context match",
}
LABEL_COLORS = {
    "PARAMETRIC_MATCH": "#c94c4c",
    "AMBIGUOUS": "#b8bec9",
    "CONTEXT_MATCH": "#2f78b7",
}


def save_current(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        plt.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight", dpi=220)
    plt.close()


def plot_conflictbank_context_shift() -> None:
    rows: list[dict[str, object]] = []
    for slug, model_name in MODEL_ORDER:
        frame = pd.read_parquet(ARTIFACTS / "conflictbank" / f"{slug}.parquet")
        for setting, column in (("No context", "no_context_class"), ("Conflict context", "conflict_context_class")):
            counts = frame[column].value_counts().reindex(LABEL_ORDER, fill_value=0)
            total = counts.sum()
            for label, count in counts.items():
                rows.append(
                    {
                        "model": model_name,
                        "setting": setting,
                        "label": label,
                        "percent": 100 * count / total,
                    }
                )

    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    x_positions = []
    x_labels = []
    x = 0.0
    width = 0.62
    for model_name in [name for _, name in MODEL_ORDER]:
        for setting in ("No context", "Conflict context"):
            bottom = 0.0
            for label in LABEL_ORDER:
                value = float(data[(data.model == model_name) & (data.setting == setting) & (data.label == label)].percent.iloc[0])
                ax.bar(x, value, width, bottom=bottom, color=LABEL_COLORS[label], edgecolor="white", linewidth=0.8)
                if value >= 7:
                    ax.text(x, bottom + value / 2, f"{value:.0f}%", ha="center", va="center", fontsize=8, color="#1d2636")
                bottom += value
            x_positions.append(x)
            x_labels.append(f"{model_name}\n{setting}")
            x += 0.82
        x += 0.35

    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of examples (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.grid(axis="y", color="#d7dce5", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LABEL_COLORS[label]) for label in LABEL_ORDER]
    ax.legend(handles, [LABEL_NAMES[label] for label in LABEL_ORDER], ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    save_current("conflictbank_context_shift")


def write_summary() -> None:
    summary = {
        "included_figures": [
            {
                "file": "paper/figures/conflictbank_context_shift.pdf",
                "reason": "Shows how outputs shift from no-context answers to conflict-context answers. The paper uses this figure instead of repeating the same ConflictBank result in a table.",
            },
        ],
        "not_included": [
            {
                "candidate": "probe-layer AUROC curves",
                "reason": "They add visual complexity without improving the main explanation. The selected probe results are clearer as a compact table.",
            },
            {
                "candidate": "mechanistic patching heatmaps",
                "reason": "They are dense and harder to read. The main finding is already simple: selected early components did not produce successful generation-level edits.",
            }
        ],
    }
    (FIGURES / "figure_selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    plot_conflictbank_context_shift()
    write_summary()


if __name__ == "__main__":
    main()
