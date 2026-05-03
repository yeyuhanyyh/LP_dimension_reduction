#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make one 4x2 k-sweep figure for the selected eight LP instances."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent
OLD_CODE = ROOT / "2026-04-24_snapshot_before_harder_prior" / "code"
sys.path.insert(0, str(OLD_CODE))

import final_projection_figure_suite as fps  # noqa: E402


CASE_ORDER = [
    "packing",
    "mincostflow",
    "random_lp_A",
    "random_lp_B",
    "random_lp_C",
    "random_lp_D",
    "grow7",
    "sc205",
]
KEEP_SGA_CASES = {"packing", "maxflow", "random_lp_A", "random_lp_B", "random_lp_C", "random_lp_D", "grow7"}
KEEP_PCA_CASES = {"mincostflow", "shortest_path", "sc205"}
METHOD_ORDER = ["Full", "OursEstC", "Rand", "DataDriven", "CostOnly"]


def configure_styles() -> None:
    fps.METHOD_STYLE["OursEstC"] = {"color": "#d62728", "marker": "D", "linestyle": (0, (6, 2))}
    fps.METHOD_DISPLAY["OursEstC"] = "Ours (est. C)"
    fps.METHOD_STYLE["DataDriven"] = {"color": "#1f77b4", "marker": "s", "linestyle": (0, (1, 2))}
    fps.METHOD_DISPLAY["DataDriven"] = "data-driven proj"
    fps.METHOD_ORDER[:] = METHOD_ORDER


def selected_projection(case: str) -> str:
    if case in KEEP_SGA_CASES:
        return "SGA"
    if case in KEEP_PCA_CASES:
        return "PCA"
    return "PCA"


def load_summary(in_dir: Path, shortest_replacement_dir: Path | None = None) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for name in ["k_sweep_synthetic_summary.csv", "k_sweep_netlib_summary.csv"]:
        path = in_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No k-sweep summary CSVs found in {in_dir}")
    out = pd.concat(frames, ignore_index=True)
    if shortest_replacement_dir is not None:
        repl_path = shortest_replacement_dir / "k_sweep_synthetic_summary.csv"
        if not repl_path.exists():
            raise FileNotFoundError(f"Missing shortest_path replacement CSV: {repl_path}")
        repl = pd.read_csv(repl_path)
        repl = repl[repl["case"] == "shortest_path"].copy()
        out = pd.concat([out[out["case"] != "shortest_path"].copy(), repl], ignore_index=True)
    return out


def select_and_merge(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = df[df["case"].isin(CASE_ORDER)].copy()
    for case, subcase in df.groupby("case", sort=False):
        base = subcase[subcase["method"].isin(["Full", "OursEstC", "Rand", "CostOnly"])].copy()
        proj = subcase[subcase["method"] == selected_projection(str(case))].copy()
        proj["method"] = "DataDriven"
        rows.append(pd.concat([base, proj], ignore_index=True))
    out = pd.concat(rows, ignore_index=True)
    out["case_order"] = out["case"].map({c: i for i, c in enumerate(CASE_ORDER)})
    out["method_order"] = out["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return out.sort_values(["case_order", "method_order", "K"]).drop(columns=["case_order", "method_order"])


def title_map_from(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for _, row in df[["case", "title"]].drop_duplicates().iterrows():
        out[str(row["case"])] = str(row["title"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shortest_replacement_dir", default="")
    args = parser.parse_args()

    configure_styles()
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shortest_replacement_dir = Path(args.shortest_replacement_dir).resolve() if str(args.shortest_replacement_dir).strip() else None
    selected = select_and_merge(load_summary(in_dir, shortest_replacement_dir))
    selected.to_csv(out_dir / "k_sweep_paper8_selected_pca_sga.csv", index=False)
    fps.plot_metric_grid(
        selected,
        CASE_ORDER,
        title_map_from(selected),
        out_dir / "figure_k_sweep_paper8.png",
        methods=METHOD_ORDER,
        y_col="objective_ratio_mean",
        yerr_col="objective_ratio_se",
        ylabel="Average test objective ratio",
    )
    manifest = {
        "source_dir": str(in_dir),
        "shortest_replacement_dir": str(shortest_replacement_dir) if shortest_replacement_dir is not None else "",
        "case_order": CASE_ORDER,
        "methods": METHOD_ORDER,
        "pca_sga_selection": {
            "SGA": sorted(KEEP_SGA_CASES),
            "PCA": sorted(KEEP_PCA_CASES),
            "merged_label": "data-driven proj",
        },
        "shadow_band": "mean +/- standard error, drawn by final_projection_figure_suite.plot_metric_grid",
    }
    (out_dir / "paper8_k_sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
