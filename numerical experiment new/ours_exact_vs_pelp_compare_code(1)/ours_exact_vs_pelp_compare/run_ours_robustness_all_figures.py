#!/usr/bin/env python3
"""OursExact-only multi-seed robustness figures matching the five main plots.

The main suite compares all baselines.  This script keeps those results intact
and adds a robustness companion for OursExact: multiple random seeds are run,
then the mean curve and seed-to-seed standard-deviation band are plotted for
the same figure families.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import final_projection_figure_suite as fps
from compare_fixedX_family_suite import append_ours_exact_rows
from compare_ours_exact_vs_pelp_fixedX import alg1_pointwise, alg2_cumulative, solve_standard_form_min
from iwata_sakaue_pelp_projection_compare import objective_ratio, summarize_ratios_and_times


ROBUST_STYLE = {
    "color": "#d62728",
    "marker": "D",
    "linestyle": (0, (6, 2)),
}


def parse_seeds(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_names(text: str) -> set[str]:
    return {x.strip() for x in str(text).split(",") if x.strip()}


def seed_tag(seed: int) -> str:
    return f"seed_{int(seed):03d}"


def mean_std_summary(df: pd.DataFrame, group_cols: Sequence[str], value_col: str) -> pd.DataFrame:
    out = df.groupby(list(group_cols), as_index=False).agg(
        mean=(value_col, "mean"),
        std=(value_col, "std"),
        n=("seed", "count"),
    )
    out["std"] = out["std"].fillna(0.0)
    return out


def run_ours_k_sweep_case(
    case: fps.CaseSpec,
    seed: int,
    base_args: argparse.Namespace,
    out_dir: Path,
    force: bool,
) -> Dict[str, pd.DataFrame]:
    case_out = out_dir / seed_tag(seed) / "k_sweep" / case.slug
    summary_path = case_out / "summary_results.csv"
    rank_path = case_out / "ours_rank_after_sample.csv"
    if not force and summary_path.exists() and rank_path.exists():
        return {"summary": pd.read_csv(summary_path), "rank": pd.read_csv(rank_path)}

    args = copy.deepcopy(base_args)
    args.seed = int(seed)
    args.n_train = int(getattr(base_args, "n_train", 24))
    args.k_list = str(getattr(base_args, "k_list", "5,10,20,30,40,50"))
    data = fps.prepare_case_data(case, args)
    k_list = fps.parse_k_list(data.args.k_list)
    stage = alg2_cumulative(data.bundle.stdlp, data.bundle.Ctrain, data.bundle.prior, verbose=False)

    rows: List[dict[str, object]] = []
    append_ours_exact_rows(rows, data.bundle.stdlp, stage.U, stage.x_anchor, data.bundle.test, comparison_ks=k_list)
    raw = pd.DataFrame(rows)
    raw["case"] = case.slug
    raw["title"] = case.title
    raw["family"] = case.family
    summary = summarize_ratios_and_times(raw.to_dict("records"))
    summary["case"] = case.slug
    summary["title"] = case.title
    summary["family"] = case.family
    rank = pd.DataFrame(
        {
            "sample": np.arange(0, len(stage.rank_after_sample) + 1),
            "rank_after_sample": np.concatenate([[0], np.asarray(stage.rank_after_sample, dtype=int)]),
            "case": case.slug,
            "title": case.title,
            "family": case.family,
        }
    )
    case_out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(case_out / "raw_results.csv", index=False)
    summary.to_csv(summary_path, index=False)
    rank.to_csv(rank_path, index=False)
    return {"summary": summary, "rank": rank}


def run_ours_sample_case(
    case: fps.CaseSpec,
    seed: int,
    base_args: argparse.Namespace,
    out_dir: Path,
    force: bool,
) -> pd.DataFrame:
    case_out = out_dir / seed_tag(seed) / "sample_efficiency" / case.slug
    summary_path = case_out / "summary_results.csv"
    if not force and summary_path.exists():
        return pd.read_csv(summary_path)

    args = copy.deepcopy(base_args)
    args.seed = int(seed)
    args.n_train = max(fps.sample_train_grid(case))
    args.k_list = str(int(case.sample_k))
    data = fps.prepare_case_data(case, args)
    rows: List[pd.DataFrame] = []
    grid = fps.sample_train_grid(case)
    prefix_u: Dict[int, np.ndarray] = {}
    prefix_anchor: Dict[int, np.ndarray] = {}
    positive_grid = sorted(int(n) for n in grid if int(n) > 0)
    if positive_grid:
        D = np.zeros((data.bundle.stdlp.dim, 0))
        x_anchor, _ = solve_standard_form_min(data.bundle.stdlp, data.bundle.Ctrain[0])
        wanted = set(positive_grid)
        max_train = max(positive_grid)
        for idx, c_train in enumerate(data.bundle.Ctrain[:max_train], start=1):
            D, _cert, _trace = alg1_pointwise(data.bundle.stdlp, c_train, data.bundle.prior, D)
            if idx in wanted:
                U = np.zeros((data.bundle.stdlp.dim, 0)) if D.size == 0 else np.linalg.qr(D, mode="reduced")[0]
                prefix_u[idx] = U
                prefix_anchor[idx] = x_anchor

    for n_train in grid:
        if int(n_train) == 0:
            raw_rows = []
            for proj_inst, full_inst in zip(data.proj_test, data.bundle.test):
                anchor_obj = float(getattr(proj_inst, "objective_constant", 0.0))
                raw_rows.append(
                    {
                        "method": "OursExact",
                        "K": int(case.sample_k),
                        "instance": full_inst.name,
                        "objective": anchor_obj,
                        "full_objective": float(full_inst.full_obj),
                        "objective_ratio": objective_ratio(anchor_obj, float(full_inst.full_obj)),
                        "time": 0.0,
                        "success": 1.0,
                    }
                )
            raw = pd.DataFrame(raw_rows)
        else:
            raw_rows = []
            append_ours_exact_rows(
                raw_rows,
                data.bundle.stdlp,
                prefix_u[int(n_train)],
                prefix_anchor[int(n_train)],
                data.bundle.test,
                comparison_ks=[int(case.sample_k)],
            )
            raw = pd.DataFrame(raw_rows)
        summary = summarize_ratios_and_times(raw.to_dict("records"))
        summary = summary[summary["method"] == "OursExact"].copy()
        summary["case"] = case.slug
        summary["title"] = case.title
        summary["family"] = case.family
        summary["n_train"] = int(n_train)
        summary["sample_k"] = int(case.sample_k)
        rows.append(summary)

    out = pd.concat(rows, ignore_index=True)
    case_out.mkdir(parents=True, exist_ok=True)
    out.to_csv(summary_path, index=False)
    return out


def plot_metric_robust_grid(
    summary: pd.DataFrame,
    case_order: Sequence[str],
    title_map: Dict[str, str],
    out_path: Path,
    x_col: str,
    y_col: str,
    std_col: str,
    xlabel: str,
    ylabel: str,
) -> None:
    cases = [c for c in case_order if c in set(summary["case"])]
    if not cases:
        return
    ncols = min(4, len(cases))
    nrows = int(math.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    ymin = float(np.nanmin(summary[y_col] - summary[std_col]))
    ymax = float(np.nanmax(summary[y_col] + summary[std_col]))
    pad = 0.05 * max(ymax - ymin, 1e-6)
    ymin, ymax = ymin - pad, ymax + pad

    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        sub = summary[summary["case"] == case].sort_values(x_col)
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        std = sub[std_col].fillna(0.0).to_numpy(dtype=float)
        ax.fill_between(x, y - std, y + std, color=ROBUST_STYLE["color"], alpha=0.16, linewidth=0)
        ax.plot(
            x,
            y,
            color=ROBUST_STYLE["color"],
            marker=ROBUST_STYLE["marker"],
            linestyle=ROBUST_STYLE["linestyle"],
            linewidth=2.2,
            markersize=5.0,
        )
        ax.set_title(f"({chr(97 + idx)}) {title_map[case]}")
        ax.set_xlabel(xlabel)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(sorted(set(int(round(v)) for v in x.tolist())))
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    for row in axes_arr:
        row[0].set_ylabel(ylabel)
    handle = axes_arr[0, 0].plot(
        [],
        [],
        color=ROBUST_STYLE["color"],
        marker=ROBUST_STYLE["marker"],
        linestyle=ROBUST_STYLE["linestyle"],
        label="OursExact mean",
    )[0]
    patch = axes_arr[0, 0].fill_between([], [], [], color=ROBUST_STYLE["color"], alpha=0.16, label="+/- 1 std. over seeds")
    fig.legend([handle, patch], ["OursExact mean", "+/- 1 std. over seeds"], loc="lower center", ncol=2, frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_rank_robust_grid(
    summary: pd.DataFrame,
    case_order: Sequence[str],
    title_map: Dict[str, str],
    out_path: Path,
) -> None:
    cases = [c for c in case_order if c in set(summary["case"])]
    if not cases:
        return
    ncols = min(4, len(cases))
    nrows = int(math.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.25 * nrows), sharey=False)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        sub = summary[summary["case"] == case].sort_values("sample")
        sub = sub[sub["sample"] <= fps.RANK_CURVE_MAX_SAMPLES]
        x = sub["sample"].to_numpy(dtype=float)
        y = sub["rank_mean"].to_numpy(dtype=float)
        std = sub["rank_std"].fillna(0.0).to_numpy(dtype=float)
        ax.fill_between(x, np.maximum(0.0, y - std), y + std, color=ROBUST_STYLE["color"], alpha=0.16, linewidth=0)
        ax.plot(
            x,
            y,
            color=ROBUST_STYLE["color"],
            marker="o",
            linestyle=ROBUST_STYLE["linestyle"],
            linewidth=2.0,
            markersize=4.8,
        )
        ax.set_title(f"({chr(97 + idx)}) {title_map[case]}")
        ax.set_xlabel("# training samples processed")
        ax.set_xlim(0, fps.RANK_CURVE_MAX_SAMPLES)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(0.0, max(1.0, float(np.nanmax(y + std)) + 0.5))
        if idx % ncols == 0:
            ax.set_ylabel("Learned dimension")
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def aggregate_and_plot(out_dir: Path, synth_cases: Sequence[fps.CaseSpec], net_cases: Sequence[fps.CaseSpec], sample_cases: Sequence[fps.CaseSpec]) -> None:
    title_map = {c.slug: c.title for c in [*synth_cases, *net_cases, *sample_cases]}
    synth_order = [c.slug for c in synth_cases]
    net_order = [c.slug for c in net_cases]
    sample_order = [c.slug for c in sample_cases]

    def load_many(pattern_parts: Iterable[str]) -> pd.DataFrame:
        frames = []
        for path in out_dir.glob(str(Path(*pattern_parts))):
            seed_name = path.parts[-4] if "sample_efficiency" in path.parts else path.parts[-4]
            try:
                seed = int(str(seed_name).split("_")[-1])
            except Exception:
                seed = -1
            df = pd.read_csv(path)
            df["seed"] = seed
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    k_frames = []
    rank_frames = []
    sample_frames = []
    for seed_dir in sorted(out_dir.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_")[-1])
        except Exception:
            continue
        for summary_path in seed_dir.glob("k_sweep/*/summary_results.csv"):
            df = pd.read_csv(summary_path)
            df["seed"] = seed
            k_frames.append(df)
        for rank_path in seed_dir.glob("k_sweep/*/ours_rank_after_sample.csv"):
            df = pd.read_csv(rank_path)
            df["seed"] = seed
            rank_frames.append(df)
        for sample_path in seed_dir.glob("sample_efficiency/*/summary_results.csv"):
            df = pd.read_csv(sample_path)
            df["seed"] = seed
            sample_frames.append(df)

    k_all = pd.concat(k_frames, ignore_index=True) if k_frames else pd.DataFrame()
    rank_all = pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame()
    sample_all = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    k_all.to_csv(out_dir / "robust_k_sweep_ours_raw.csv", index=False)
    rank_all.to_csv(out_dir / "robust_rank_growth_ours_raw.csv", index=False)
    sample_all.to_csv(out_dir / "robust_sample_efficiency_ours_raw.csv", index=False)

    if not k_all.empty:
        k_summary = mean_std_summary(k_all, ["case", "title", "family", "K"], "objective_ratio_mean")
        k_summary = k_summary.rename(columns={"mean": "objective_ratio_mean", "std": "objective_ratio_std"})
        k_summary.to_csv(out_dir / "robust_k_sweep_ours_summary.csv", index=False)
        plot_metric_robust_grid(
            k_summary[k_summary["case"].isin(synth_order)],
            synth_order,
            title_map,
            out_dir / "robust_figure_k_sweep_synthetic_ours.png",
            x_col="K",
            y_col="objective_ratio_mean",
            std_col="objective_ratio_std",
            xlabel="Reduced dimension K",
            ylabel="Average test objective ratio",
        )
        plot_metric_robust_grid(
            k_summary[k_summary["case"].isin(net_order)],
            net_order,
            title_map,
            out_dir / "robust_figure_k_sweep_netlib_ours.png",
            x_col="K",
            y_col="objective_ratio_mean",
            std_col="objective_ratio_std",
            xlabel="Reduced dimension K",
            ylabel="Average test objective ratio",
        )

    if not rank_all.empty:
        rank_summary = rank_all.groupby(["case", "title", "family", "sample"], as_index=False).agg(
            rank_mean=("rank_after_sample", "mean"),
            rank_std=("rank_after_sample", "std"),
            n=("seed", "count"),
        )
        rank_summary["rank_std"] = rank_summary["rank_std"].fillna(0.0)
        rank_summary.to_csv(out_dir / "robust_rank_growth_ours_summary.csv", index=False)
        plot_rank_robust_grid(
            rank_summary[rank_summary["case"].isin(synth_order)],
            synth_order,
            title_map,
            out_dir / "robust_figure_ours_rank_growth_synthetic.png",
        )
        plot_rank_robust_grid(
            rank_summary[rank_summary["case"].isin(net_order)],
            net_order,
            title_map,
            out_dir / "robust_figure_ours_rank_growth_netlib.png",
        )

    if not sample_all.empty:
        sample_summary = mean_std_summary(
            sample_all,
            ["case", "title", "family", "n_train", "K", "sample_k"],
            "objective_ratio_mean",
        )
        sample_summary = sample_summary.rename(columns={"mean": "objective_ratio_mean", "std": "objective_ratio_std"})
        sample_summary.to_csv(out_dir / "robust_sample_efficiency_ours_summary.csv", index=False)
        plot_metric_robust_grid(
            sample_summary,
            sample_order,
            title_map,
            out_dir / "robust_figure_sample_efficiency_fixedK_ours.png",
            x_col="n_train",
            y_col="objective_ratio_mean",
            std_col="objective_ratio_std",
            xlabel="# training LPs",
            ylabel="Average test objective ratio",
        )


def run(args: argparse.Namespace) -> None:
    base = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    base_args = fps.default_namespace()
    base_args.device = args.device
    base_args.verbose = bool(args.verbose)
    base_args.suite_profile = "natural"

    synth_cases = fps.synthetic_cases(base)
    net_cases = fps.netlib_cases(base)
    sample_cases = fps.sample_efficiency_cases(base)
    only_cases = parse_names(args.only_cases)
    skip_cases = parse_names(args.skip_cases)
    if only_cases:
        synth_cases = [c for c in synth_cases if c.slug in only_cases]
        net_cases = [c for c in net_cases if c.slug in only_cases]
        sample_cases = [c for c in sample_cases if c.slug in only_cases]
    if skip_cases:
        synth_cases = [c for c in synth_cases if c.slug not in skip_cases]
        net_cases = [c for c in net_cases if c.slug not in skip_cases]
        sample_cases = [c for c in sample_cases if c.slug not in skip_cases]

    for seed in seeds:
        print(f"[robust OursExact] seed={seed}", flush=True)
        for case in synth_cases:
            print(f"  K/rank synthetic: {case.slug}", flush=True)
            run_ours_k_sweep_case(case, seed, base_args, out_dir, force=bool(args.force))
        for case in net_cases:
            print(f"  K/rank netlib: {case.slug}", flush=True)
            run_ours_k_sweep_case(case, seed, base_args, out_dir, force=bool(args.force))
        for case in sample_cases:
            print(f"  sample-eff: {case.slug}", flush=True)
            run_ours_sample_case(case, seed, base_args, out_dir, force=bool(args.force))

    aggregate_and_plot(out_dir, synth_cases, net_cases, sample_cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--out_dir", default="results_projection_suite_natural_refined_v7_full/ours_robustness_all_figures")
    parser.add_argument("--seeds", default="17,23,31")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--only_cases", default="", help="Comma-separated case slugs to run (all if empty).")
    parser.add_argument("--skip_cases", default="", help="Comma-separated case slugs to skip in this run.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
