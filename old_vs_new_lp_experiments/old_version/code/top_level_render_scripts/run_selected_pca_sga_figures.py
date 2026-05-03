#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render final figures with one case-selected PCA/SGA baseline.

For the requested cases, SGA is retained; for all other cases, PCA is retained.
The retained row is renamed to a single method label so the figures show one
uniformly styled PCA/SGA baseline instead of separate PCA and SGA curves.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

import final_projection_figure_suite as fps


BASELINE_METHOD = "data-driven proj"
KEEP_SGA_CASES = {
    "packing",
    "maxflow",
    "random_lp_A",
    "random_lp_B",
    "random_lp_C",
    "random_lp_D",
    "grow7",
    "scagr25",
    "stair",
}
BASELINE_STYLE = {"color": "#1f77b4", "marker": "s", "linestyle": (0, (1, 2))}


def choose_baseline(case: str) -> str:
    return "SGA" if str(case) in KEEP_SGA_CASES else "PCA"


def select_one_baseline(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        method = str(row["method"])
        if method in {"PCA", "SGA"}:
            if method != choose_baseline(str(row["case"])):
                continue
            row = row.copy()
            row["source_method"] = method
            row["method"] = BASELINE_METHOD
        else:
            row = row.copy()
            row["source_method"] = method
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def write_selection_manifest(out_dir: Path, source_dir: Path) -> None:
    payload = {
        "source_dir": str(source_dir.resolve()),
        "baseline_method": BASELINE_METHOD,
        "sga_cases": sorted(KEEP_SGA_CASES),
        "pca_cases": "all cases not listed in sga_cases",
    }
    (out_dir / "pca_sga_selection_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def copy_rank_artifacts(source_dir: Path, out_dir: Path) -> None:
    for name in ["ours_rank_growth_synthetic.csv", "ours_rank_growth_netlib.csv"]:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def configure_plot_styles() -> None:
    fps.METHOD_DISPLAY[BASELINE_METHOD] = BASELINE_METHOD
    fps.METHOD_STYLE[BASELINE_METHOD] = dict(BASELINE_STYLE)
    fps.SAMPLE_METHOD_STYLE[BASELINE_METHOD] = dict(BASELINE_STYLE)
    fps.METHOD_ORDER = ["Full", "OursExact", "Rand", BASELINE_METHOD, "CostOnly"]
    fps.SAMPLE_METHOD_ORDER = ["OursExact", BASELINE_METHOD, "CostOnly"]


def read_required_csv(source_dir: Path, name: str) -> pd.DataFrame:
    path = source_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required source CSV: {path}")
    return pd.read_csv(path)


def replace_case(base: pd.DataFrame, replacement: pd.DataFrame, case: str = "shortest_path") -> pd.DataFrame:
    return pd.concat([base[base["case"] != case].copy(), replacement.copy()], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_dir", default="results_projection_suite_natural_refined_v7_robust_allmethods")
    parser.add_argument("--out_dir", default="results_projection_suite_natural_refined_v7_selected_pca_sga")
    parser.add_argument(
        "--shortest_replacement_dir",
        default="",
        help="Optional directory containing large-C shortest_path replacement summary CSVs.",
    )
    parser.add_argument(
        "--shortest_rank_replacement_dir",
        default="",
        help="Optional directory containing only a shortest_path ours_rank_growth_synthetic.csv replacement.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    source_dir = (root / args.source_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_styles()

    synth = select_one_baseline(read_required_csv(source_dir, "k_sweep_synthetic_summary.csv"))
    net = select_one_baseline(read_required_csv(source_dir, "k_sweep_netlib_summary.csv"))
    sample = select_one_baseline(read_required_csv(source_dir, "sample_efficiency_summary.csv"))
    rank_syn = read_required_csv(source_dir, "ours_rank_growth_synthetic.csv")
    rank_net = read_required_csv(source_dir, "ours_rank_growth_netlib.csv")

    replacement_dir = (root / args.shortest_replacement_dir).resolve() if str(args.shortest_replacement_dir).strip() else None
    if replacement_dir is not None:
        repl_synth = select_one_baseline(read_required_csv(replacement_dir, "k_sweep_synthetic_summary.csv"))
        repl_rank = read_required_csv(replacement_dir, "ours_rank_growth_synthetic.csv")
        synth = replace_case(synth, repl_synth)
        rank_syn = replace_case(rank_syn, repl_rank)
        repl_sample_path = replacement_dir / "sample_efficiency_summary.csv"
        if repl_sample_path.exists():
            repl_sample = select_one_baseline(pd.read_csv(repl_sample_path))
            sample = replace_case(sample, repl_sample)

    rank_replacement_dir = (
        (root / args.shortest_rank_replacement_dir).resolve()
        if str(args.shortest_rank_replacement_dir).strip()
        else None
    )
    if rank_replacement_dir is not None:
        repl_rank = read_required_csv(rank_replacement_dir, "ours_rank_growth_synthetic.csv")
        rank_syn = replace_case(rank_syn, repl_rank)

    synth.to_csv(out_dir / "k_sweep_synthetic_summary.csv", index=False)
    net.to_csv(out_dir / "k_sweep_netlib_summary.csv", index=False)
    sample.to_csv(out_dir / "sample_efficiency_summary.csv", index=False)
    rank_syn.to_csv(out_dir / "ours_rank_growth_synthetic.csv", index=False)
    rank_net.to_csv(out_dir / "ours_rank_growth_netlib.csv", index=False)

    synth_cases = fps.synthetic_cases(root)
    net_cases = fps.netlib_cases(root)
    sample_cases = fps.sample_efficiency_cases(root)
    synth_order = [c.slug for c in synth_cases]
    net_order = [c.slug for c in net_cases]
    sample_order = [c.slug for c in sample_cases]
    title_map = {c.slug: c.title for c in [*synth_cases, *net_cases, *sample_cases]}
    sample_k_map = {c.slug: int(c.sample_k) for c in sample_cases}

    fps.plot_metric_grid(
        synth,
        synth_order,
        title_map,
        out_dir / "figure_k_sweep_synthetic.png",
        methods=fps.METHOD_ORDER,
        y_col="objective_ratio_mean",
        yerr_col="objective_ratio_se",
        ylabel="Average test objective ratio",
    )
    fps.plot_metric_grid(
        net,
        net_order,
        title_map,
        out_dir / "figure_k_sweep_netlib.png",
        methods=fps.METHOD_ORDER,
        y_col="objective_ratio_mean",
        yerr_col="objective_ratio_se",
        ylabel="Average test objective ratio",
    )
    fps.plot_sample_efficiency_grid(
        sample,
        sample_order,
        title_map,
        sample_k_map,
        out_dir / "figure_sample_efficiency_fixedK.png",
        y_col="objective_ratio_mean",
        yerr_col="objective_ratio_se",
        ylabel="Average test objective ratio",
    )

    fps.plot_rank_grid(
        rank_syn,
        synth_order,
        title_map,
        out_dir / "figure_ours_rank_growth_synthetic.png",
        per_case_scale=True,
    )
    fps.plot_rank_grid(
        rank_net,
        net_order,
        title_map,
        out_dir / "figure_ours_rank_growth_netlib.png",
        per_case_scale=True,
    )
    if replacement_dir is None and rank_replacement_dir is None:
        copy_rank_artifacts(source_dir, out_dir)
    write_selection_manifest(out_dir, source_dir)
    print(f"Saved selected PCA/SGA figures to: {out_dir}")


if __name__ == "__main__":
    main()
