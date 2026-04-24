#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate multi-seed all-method results and render final robustness figures.

Usage example:
python run_allmethods_robustness_figures.py \
  --runs "17=results_projection_suite_natural_refined_v7_full,23=results_projection_suite_natural_refined_v7_seed023_full" \
  --out_dir results_projection_suite_natural_refined_v7_robust_allmethods
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import final_projection_figure_suite as fps


def parse_runs(text: str) -> List[Tuple[int, str]]:
    pairs: List[Tuple[int, str]] = []
    for chunk in str(text).split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid run spec '{item}'. Expected format: seed=dir")
        seed_s, dir_s = item.split("=", 1)
        pairs.append((int(seed_s.strip()), dir_s.strip()))
    if not pairs:
        raise ValueError("No runs parsed from --runs.")
    return pairs


def load_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def aggregate_metric_table(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = (
        df.groupby(group_cols, as_index=False)
        .agg(
            objective_ratio_mean=(value_col, "mean"),
            objective_ratio_se=(value_col, "std"),
            seed_count=("seed", "count"),
        )
        .reset_index(drop=True)
    )
    out["objective_ratio_se"] = out["objective_ratio_se"].fillna(0.0)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--runs",
        type=str,
        default="17=results_projection_suite_natural_refined_v7_full,23=results_projection_suite_natural_refined_v7_seed023_full",
        help="Comma-separated run list in the format seed=dir",
    )
    p.add_argument("--out_dir", type=str, default="results_projection_suite_natural_refined_v7_robust_allmethods")
    p.add_argument(
        "--rank_source_seed",
        type=int,
        default=17,
        help="Seed id used as single-run source for rank-growth curves.",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_specs = parse_runs(args.runs)
    seed_to_dir: Dict[int, Path] = {seed: (root / d).resolve() for seed, d in run_specs}

    synth_frames: List[pd.DataFrame] = []
    net_frames: List[pd.DataFrame] = []
    sample_frames: List[pd.DataFrame] = []
    for seed, run_dir in run_specs:
        run_path = (root / run_dir).resolve()
        synth = load_csv_required(run_path / "k_sweep_synthetic_summary.csv")
        net = load_csv_required(run_path / "k_sweep_netlib_summary.csv")
        sample = load_csv_required(run_path / "sample_efficiency_summary.csv")
        synth["seed"] = int(seed)
        net["seed"] = int(seed)
        sample["seed"] = int(seed)
        synth_frames.append(synth)
        net_frames.append(net)
        sample_frames.append(sample)

    synth_all = pd.concat(synth_frames, ignore_index=True)
    net_all = pd.concat(net_frames, ignore_index=True)
    sample_all = pd.concat(sample_frames, ignore_index=True)

    synth_group = ["case", "title", "family", "method", "K"]
    net_group = ["case", "title", "family", "method", "K"]
    sample_group = ["case", "title", "family", "method", "K", "n_train", "sample_k"]

    synth_summary = aggregate_metric_table(synth_all, synth_group, "objective_ratio_mean")
    net_summary = aggregate_metric_table(net_all, net_group, "objective_ratio_mean")
    sample_summary = aggregate_metric_table(sample_all, sample_group, "objective_ratio_mean")

    synth_summary.to_csv(out_dir / "k_sweep_synthetic_summary.csv", index=False)
    net_summary.to_csv(out_dir / "k_sweep_netlib_summary.csv", index=False)
    sample_summary.to_csv(out_dir / "sample_efficiency_summary.csv", index=False)

    rank_seed = int(args.rank_source_seed)
    rank_src = seed_to_dir.get(rank_seed)
    if rank_src is None:
        rank_src = seed_to_dir[run_specs[0][0]]

    rank_syn = load_csv_required(rank_src / "ours_rank_growth_synthetic.csv")
    rank_net = load_csv_required(rank_src / "ours_rank_growth_netlib.csv")
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
        synth_summary,
        synth_order,
        title_map,
        out_dir / "figure_k_sweep_synthetic.png",
        methods=fps.METHOD_ORDER,
        y_col="objective_ratio_mean",
        yerr_col="objective_ratio_se",
        ylabel="Average test objective ratio",
    )
    fps.plot_metric_grid(
        net_summary,
        net_order,
        title_map,
        out_dir / "figure_k_sweep_netlib.png",
        methods=fps.METHOD_ORDER,
        y_col="objective_ratio_mean",
        yerr_col="objective_ratio_se",
        ylabel="Average test objective ratio",
    )
    fps.plot_sample_efficiency_grid(
        sample_summary,
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

    payload = {
        "runs": [{"seed": int(seed), "dir": str((root / d).resolve())} for seed, d in run_specs],
        "rank_source_seed": int(rank_seed),
    }
    (out_dir / "robustness_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved all-method robustness outputs to: {out_dir}")


if __name__ == "__main__":
    main()
