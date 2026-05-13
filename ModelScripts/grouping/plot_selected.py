"""
Plot a hand-picked subset of feature-set experiments.

Experiments are selected by passing one or more identifiers after the CSV path.
Each identifier is matched against the 'experiment' column as a case-insensitive
substring, so you can use short IDs (E5), group names (MRI), or full names.

Usage:
    python plot_selected.py results.csv E5 E12 E26
    python plot_selected.py results.csv MRI PLASMA --scenario drop
    python plot_selected.py results.csv E1 E3 E7 --out selected.png
    python plot_selected.py results.csv --list          # print all experiment names
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from plot import (
    detect_scenario,
    infer_model_name,
    load_and_sort,
    render_figure,
    SCENARIOS,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csv", type=Path, help="Path to results CSV")
    p.add_argument("experiments", nargs="*",
                   help="Substrings to match against experiment names "
                        "(case-insensitive). Multiple values are OR-ed.")
    p.add_argument("--scenario", choices=["drop", "merge", "auto"], default="auto")
    p.add_argument("--model", default=None)
    p.add_argument("--out", type=Path, default=None,
                   help="Output figure path (default: <csv_stem>_selected.png)")
    p.add_argument("--list", action="store_true",
                   help="Print all experiment names in the CSV and exit.")
    return p.parse_args()


def select_experiments(df, identifiers):
    if not identifiers:
        return df
    mask = pd.Series(False, index=df.index)
    for ident in identifiers:
        pattern = rf'\b{re.escape(ident)}\b'
        mask |= df["experiment"].str.contains(pattern, case=False, regex=True)
    return df[mask].reset_index(drop=True)


def main():
    args = parse_args()
    df = load_and_sort(args.csv)

    if args.list:
        for name in df["experiment"]:
            print(name)
        return

    if not args.experiments:
        print("No experiments specified. Pass substrings to match, or --list to see all.")
        return

    df_sel = select_experiments(df, args.experiments)
    if df_sel.empty:
        print(f"No experiments matched: {args.experiments}")
        print("Run with --list to see available experiment names.")
        return

    scenario_name = detect_scenario(df_sel) if args.scenario == "auto" else args.scenario
    cfg = SCENARIOS[scenario_name]
    model_name = args.model or infer_model_name(args.csv)
    out_path = args.out or args.csv.with_name(f"{args.csv.stem}_selected.png")

    print(f"Selected {len(df_sel)} experiment(s): {list(df_sel['experiment'])}")
    render_figure(df_sel, cfg, out_path, model_name=model_name)


if __name__ == "__main__":
    main()
