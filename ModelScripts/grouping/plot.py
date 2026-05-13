"""
Plot cross-validation results across feature-set experiments.

Produces a multi-panel figure:
  Panel A: Overall metrics (macro-F1 and balanced accuracy) per experiment,
           sorted by macro-F1, with std error bars from the 5 CV folds.
  Panel B: Per-class recall heatmap    (experiments x cognitive stages).
  Panel C: Per-class precision heatmap (experiments x cognitive stages).

Two scenarios are supported:
  drop  : 5 stages including 'Early MCI'         (FL_UDSD == 2 was dropped)
  merge : 5 stages including 'Early Impaired'    (FL_UDSD == 2 merged into EMCI)

The scenario is auto-detected from the CSV column names but can be forced
with --scenario.

The model name shown in the figure title is inferred from the CSV filename
prefix (xgb_ → XGBoost, rf_ → Random Forest, lr_ → Logistic Regression) but
can be overridden with --model.

Usage:
    python plot.py xgb_results.csv
    python plot.py rf_results.csv
    python plot.py lr_results.csv --scenario merge
    python plot.py xgb_results.csv --top 20 --out fig.png
    python plot.py xgb_results.csv --split    # writes _top and _bottom
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


# Each scenario defines its class labels (full + short) in clinical
# progression order. The full labels must match the column suffixes
# emitted by run_group_experiments — i.e. precision_<label>_mean.
SCENARIOS = {
    "drop": {
        "stages": [
            "Normal cognition",
            "Subjective Cognitive Decline",
            "Early MCI",
            "Late MCI",
            "Dementia",
        ],
        "stages_short": ["NC", "SCD", "EMCI", "LMCI", "Dem"],
        "title_suffix": "[NC, SCD, EMCI, LMCI, DEM]",
    },
    "merge": {
        "stages": [
            "Normal cognition",
            "SCD",
            "Early Impaired",
            "Late MCI",
            "Dementia",
        ],
        "stages_short": ["NC", "SCD", "EImp", "LMCI", "Dem"],
        "title_suffix": "[NC, SCD, EImp, LMCI, DEM]",
    },
}

# Maps CSV filename prefixes to human-readable model names for figure titles.
MODEL_NAMES = {
    "xgb": "XGBoost",
    "rf":  "Random Forest",
    "lr":  "Logistic Regression",
}

TEXT_COLOR = "black"


def infer_model_name(csv_path: Path) -> str:
    stem = csv_path.stem.lower()
    for prefix, name in MODEL_NAMES.items():
        if stem.startswith(prefix + "_") or stem == prefix:
            return name
    return csv_path.stem  # fall back to raw stem


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csv", type=Path, help="Path to results CSV")
    p.add_argument("--scenario", choices=["drop", "merge", "auto"], default="auto",
                   help="Which class scheme the CSV uses. "
                        "'auto' (default) infers from the column names.")
    p.add_argument("--top", type=int, default=None,
                   help="Plot only the top-N experiments by macro-F1 (default: all)")
    p.add_argument("--model", default=None,
                   help="Model name shown in figure title. "
                        "Auto-detected from the CSV filename prefix "
                        "(xgb_ → XGBoost, rf_ → Random Forest, lr_ → Logistic Regression) "
                        "when not provided.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output figure path "
                        "(default: <csv_stem>_figure.png)")
    p.add_argument("--split", action="store_true",
                   help="Split into two figures: top half and bottom half by "
                        "macro-F1 rank. Output paths get '_top' and '_bottom' "
                        "suffixes inserted before the extension.")
    return p.parse_args()


def detect_scenario(df):
    """
    Inspect column names to figure out which class scheme is in use.
    The columns precision_<label>_mean reveal the labels.
    """
    cols = set(df.columns)
    for name, cfg in SCENARIOS.items():
        expected = {f"precision_{s}_mean" for s in cfg["stages"]}
        if expected.issubset(cols):
            return name
    raise ValueError(
        "Could not auto-detect scenario from CSV columns. "
        f"Expected to find one of: {list(SCENARIOS.keys())}. "
        "Pass --scenario explicitly."
    )


def load_and_sort(csv_path, top_n=None):
    df = pd.read_csv(csv_path)
    df = df.sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
    if top_n is not None:
        df = df.head(top_n).reset_index(drop=True)
    return df


def plot_overall(ax, df, n_classes):
    """Panel A: macro-F1 and balanced accuracy bars with std error bars."""
    y = np.arange(len(df))
    h = 0.4

    ax.barh(y - h/2, df["f1_macro_mean"], height=h,
            xerr=df["f1_macro_std"], label="Macro-F1",
            color="#2E86AB", ecolor="#1a4d63", capsize=2, alpha=0.9)
    ax.barh(y + h/2, df["balanced_accuracy_mean"], height=h,
            xerr=df["balanced_accuracy_std"], label="Balanced accuracy",
            color="#E07A5F", ecolor="#8a3f2a", capsize=2, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{e}  (k={n})"
                        for e, n in zip(df["experiment"], df["n_features"])],
                       fontsize=8, color=TEXT_COLOR)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score (mean ± std over 5 CV folds)", color=TEXT_COLOR)
    ax.set_title("A. Overall performance per experiment", loc="left",
                 fontweight="bold", color=TEXT_COLOR)
    chance = 1.0 / n_classes
    ax.axvline(chance, color="grey", lw=0.5, ls=":", alpha=0.6)
    ax.text(chance, -0.6, "chance", fontsize=7, color=TEXT_COLOR, ha="center")
    legend = ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    for txt in legend.get_texts():
        txt.set_color(TEXT_COLOR)

    # plain-text values at the right end of each bar pair
    for i, row in df.iterrows():
        ax.text(1.01, i - h/2, f"{row['f1_macro_mean']:.3f}",
                va="center", ha="left", fontsize=7, color="#2E86AB")
        ax.text(1.01, i + h/2, f"{row['balanced_accuracy_mean']:.3f}",
                va="center", ha="left", fontsize=7, color="#E07A5F")

    ax.tick_params(axis="x", colors=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)
    ax.grid(axis="x", alpha=0.3)


def build_class_matrix(df, metric, stages):
    """Return a (n_experiments x n_stages) matrix of metric_<stage>_mean."""
    cols = [f"{metric}_{s}_mean" for s in stages]
    return df[cols].to_numpy()


def short_id(exp_name):
    """Extract experiment ID prefix, e.g. 'E26_CDRSUM_HVLTDR_LASSI' -> 'E26'."""
    return exp_name.split("_", 1)[0]


def plot_heatmap(ax, matrix, df, stages_short, title, cmap="viridis"):
    # vmax=1.15 keeps the darkest cells light enough for black annotations
    # to remain readable, while colour differences across [0,1] are preserved.
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1.15)
    ax.set_xticks(np.arange(len(stages_short)))
    ax.set_xticklabels(stages_short, fontsize=9, color=TEXT_COLOR)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels([short_id(e) for e in df["experiment"]],
                       fontsize=8, color=TEXT_COLOR)
    ax.set_title(title, loc="left", fontweight="bold", color=TEXT_COLOR)
    ax.tick_params(axis="x", colors=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)

    # Annotate cells — all black.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=TEXT_COLOR, fontsize=7)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors=TEXT_COLOR)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])  # cap displayed ticks at 1.0
    return im


def render_figure(df, cfg, out_path, model_name="XGBoost", subtitle=None):
    """Render and save one figure for a given (already-sorted) df slice."""
    stages = cfg["stages"]
    stages_short = cfg["stages_short"]

    n = len(df)
    fig_height = max(8, 0.32 * n + 4)
    fig = plt.figure(figsize=(16, fig_height))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[2.2, 1, 1], wspace=0.45)

    ax_overall = fig.add_subplot(gs[0, 0])
    ax_recall  = fig.add_subplot(gs[0, 1])
    ax_prec    = fig.add_subplot(gs[0, 2])

    plot_overall(ax_overall, df, n_classes=len(stages))
    plot_heatmap(ax_recall,
                 build_class_matrix(df, "recall", stages),
                 df, stages_short, "B. Recall per class", cmap="Oranges")
    plot_heatmap(ax_prec,
                 build_class_matrix(df, "precision", stages),
                 df, stages_short, "C. Precision per class", cmap="Blues")

    title = f"{model_name} 4-fold CV — feature-set comparison — {cfg['title_suffix']}"
    if subtitle:
        title = f"{title}\n{subtitle}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995, color=TEXT_COLOR)
    fig.tight_layout(rect=[0, 0, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


def split_path(path, suffix):
    """foo/bar.png + 'top' -> foo/bar_top.png"""
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def main():
    args = parse_args()
    df = load_and_sort(args.csv, top_n=args.top)

    scenario_name = (detect_scenario(df) if args.scenario == "auto"
                     else args.scenario)
    cfg = SCENARIOS[scenario_name]
    stages = cfg["stages"]

    # Verify the chosen scenario's columns actually exist (catches the case
    # where the user forces --scenario on a CSV that doesn't match).
    missing = [f"precision_{s}_mean" for s in stages
               if f"precision_{s}_mean" not in df.columns]
    if missing:
        raise ValueError(
            f"Scenario '{scenario_name}' expects columns {missing} "
            f"which are not in {args.csv}. Try --scenario auto or pick "
            f"the other scenario."
        )

    model_name = args.model or infer_model_name(args.csv)
    out_path = args.out or args.csv.with_name(f"{args.csv.stem}_figure.png")

    print(f"Loaded {len(df)} experiments from {args.csv}")
    print(f"Model        : {model_name}")
    print(f"Scenario     : {scenario_name}  (classes: {cfg['stages_short']})")
    print(f"Best macro-F1: {df.iloc[0]['experiment']} = {df.iloc[0]['f1_macro_mean']:.3f}")

    if args.split:
        # Split point: for odd N the top half gets the extra row.
        # Top is rank 1..mid, bottom is rank mid+1..N.
        mid = (len(df) + 1) // 2
        df_top = df.iloc[:mid].reset_index(drop=True)
        df_bot = df.iloc[mid:].reset_index(drop=True)

        render_figure(
            df_top, cfg, split_path(out_path, "top"),
            model_name=model_name,
            subtitle=f"Top {len(df_top)} experiments (rank 1–{len(df_top)})",
        )
        render_figure(
            df_bot, cfg, split_path(out_path, "bottom"),
            model_name=model_name,
            subtitle=f"Bottom {len(df_bot)} experiments "
                    f"(rank {len(df_top)+1}–{len(df)})",
        )
    else:
        render_figure(df, cfg, out_path, model_name=model_name)


if __name__ == "__main__":
    main()