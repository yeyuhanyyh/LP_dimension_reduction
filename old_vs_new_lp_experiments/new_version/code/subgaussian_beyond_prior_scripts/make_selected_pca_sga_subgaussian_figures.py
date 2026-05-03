#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create selected PCA/SGA figures from a sub-Gaussian beyond-prior run.

The input directory must contain the CSVs produced by
run_subgaussian_beyond_prior_suite.py.  This script does not rerun any LP
solve; it only applies the case-wise PCA/SGA selection used for the paper
figures and redraws the grids.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parent
OLD_CODE = ROOT / "2026-04-24_snapshot_before_harder_prior" / "code"
sys.path.insert(0, str(OLD_CODE))

import final_projection_figure_suite as fps  # noqa: E402


SGA_CASES = {
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
PCA_CASES = {
    "mincostflow",
    "shortest_path",
    "sc205",
}
METHOD_ORDER = ["Full", "OursEstC", "Rand", "DataDriven", "CostOnly"]


def patch_styles() -> None:
    fps.METHOD_STYLE["OursEstC"] = {"color": "#d62728", "marker": "D", "linestyle": (0, (6, 2))}
    fps.METHOD_DISPLAY["OursEstC"] = "Ours (est. C)"
    fps.METHOD_STYLE["DataDriven"] = {"color": "#1f77b4", "marker": "s", "linestyle": (0, (1, 2))}
    fps.METHOD_DISPLAY["DataDriven"] = "data-driven proj"
    fps.METHOD_ORDER[:] = METHOD_ORDER


def selected_method(case: str) -> str:
    if case in SGA_CASES:
        return "SGA"
    if case in PCA_CASES:
        return "PCA"
    return "PCA"


def combine_projection_methods(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rows = []
    for case, subcase in df.groupby("case", sort=False):
        keep = selected_method(str(case))
        sub = subcase[subcase["method"].isin(["Full", "OursEstC", "Rand", "CostOnly"])].copy()
        proj = subcase[subcase["method"] == keep].copy()
        proj["method"] = "DataDriven"
        rows.append(pd.concat([sub, proj], ignore_index=True))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["case", "method", "K"]).reset_index(drop=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def case_order(df: pd.DataFrame, fallback: Iterable[str]) -> List[str]:
    cases = list(dict.fromkeys(str(x) for x in fallback))
    if cases:
        return cases
    if df.empty:
        return []
    return list(dict.fromkeys(df["case"].astype(str).tolist()))


def title_map_from(df: pd.DataFrame, rank_df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for frame in (df, rank_df):
        if frame.empty or "case" not in frame or "title" not in frame:
            continue
        for _, row in frame[["case", "title"]].drop_duplicates().iterrows():
            out[str(row["case"])] = str(row["title"])
    return out


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    patch_styles()
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = in_dir / "subgaussian_beyond_prior_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest.get("same_est_train_samples", False):
        manifest.setdefault("requested_pilot_n", manifest.get("pilot_n"))
        manifest["pilot_n"] = int(manifest.get("n_train", manifest.get("pilot_n", 0)))
        manifest["n0_for_estimated_prior"] = int(manifest.get("n_train", manifest.get("pilot_n", 0)))
    manifest["postprocess"] = "selected_pca_sga_merged"
    manifest["selection_rule"] = {
        "SGA": sorted(SGA_CASES),
        "PCA": sorted(PCA_CASES),
        "default": "PCA",
        "merged_label": "data-driven proj",
    }
    manifest["methods"] = METHOD_ORDER
    (out_dir / "subgaussian_beyond_prior_manifest_selected_pca_sga.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    synth = combine_projection_methods(read_csv(in_dir / "k_sweep_synthetic_summary.csv"))
    net = combine_projection_methods(read_csv(in_dir / "k_sweep_netlib_summary.csv"))
    synth_rank = read_csv(in_dir / "ours_rank_growth_synthetic.csv")
    net_rank = read_csv(in_dir / "ours_rank_growth_netlib.csv")

    if not synth.empty:
        synth.to_csv(out_dir / "k_sweep_synthetic_summary_selected_pca_sga.csv", index=False)
        order = case_order(synth, manifest.get("synthetic_cases", []))
        fps.plot_metric_grid(
            synth,
            order,
            title_map_from(synth, synth_rank),
            out_dir / "figure_k_sweep_synthetic.png",
            methods=METHOD_ORDER,
            y_col="objective_ratio_mean",
            yerr_col="objective_ratio_se",
            ylabel="Average test objective ratio",
        )
    if not net.empty:
        net.to_csv(out_dir / "k_sweep_netlib_summary_selected_pca_sga.csv", index=False)
        order = case_order(net, manifest.get("netlib_cases", []))
        fps.plot_metric_grid(
            net,
            order,
            title_map_from(net, net_rank),
            out_dir / "figure_k_sweep_netlib.png",
            methods=METHOD_ORDER,
            y_col="objective_ratio_mean",
            yerr_col="objective_ratio_se",
            ylabel="Average test objective ratio",
        )

    if not synth_rank.empty:
        synth_rank.to_csv(out_dir / "ours_rank_growth_synthetic.csv", index=False)
        fps.plot_rank_grid(
            synth_rank,
            case_order(synth_rank, manifest.get("synthetic_cases", [])),
            title_map_from(synth, synth_rank),
            out_dir / "figure_ours_rank_growth_synthetic.png",
            per_case_scale=True,
        )
    if not net_rank.empty:
        net_rank.to_csv(out_dir / "ours_rank_growth_netlib.csv", index=False)
        fps.plot_rank_grid(
            net_rank,
            case_order(net_rank, manifest.get("netlib_cases", [])),
            title_map_from(net, net_rank),
            out_dir / "figure_ours_rank_growth_netlib.png",
            per_case_scale=True,
        )

    copy_if_exists(in_dir / "k_sweep_synthetic_summary.csv", out_dir / "k_sweep_synthetic_summary_unmerged.csv")
    copy_if_exists(in_dir / "k_sweep_netlib_summary.csv", out_dir / "k_sweep_netlib_summary_unmerged.csv")


if __name__ == "__main__":
    main()
