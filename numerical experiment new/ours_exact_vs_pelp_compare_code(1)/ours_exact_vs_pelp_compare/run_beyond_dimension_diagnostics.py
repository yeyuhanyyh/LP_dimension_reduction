#!/usr/bin/env python3
"""All-case Beyond-prior dimension diagnostics without running all baselines."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import final_projection_figure_suite as fps
import prior_stress_and_beyond_suite as ps
from compare_ours_exact_vs_pelp_fixedX import alg2_cumulative


def diag_base_args():
    ns = fps.default_namespace()
    ns.n_train = 20
    ns.n_val = 0
    ns.n_test = 0
    ns.k_list = "20"
    return ns


def collect_rank_curve(stdlp, costs, prior):
    stage = alg2_cumulative(stdlp, costs, prior, verbose=False)
    curve = [0, *stage.rank_after_sample]
    return stage, curve


def run_rho_sweep(root: Path, case_slugs, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = diag_base_args()
    rho_list = [0.40, 0.20, 0.10, 0.03, 0.01]
    rows = []
    curve_rows = []
    for slug in case_slugs:
        case = ps.find_case(root, slug)
        cache = ps.build_beyond_case_cache(
            case=case,
            base_args=base,
            max_pilot_n0=80,
            ambient_radius_scale=1.5,
            accept_pool_size=200,
            test_pool_size=40,
        )
        for rho in rho_list:
            prior = ps.estimate_ellipsoid_prior(
                cache["pilot_costs"][:80],
                nominal_k=int(cache["args"].prior_nominal_k),
                target_outside_mass=float(rho),
                radius_scale=1.0,
                sample_radius_frac=float(cache["args"].sample_radius_frac),
            )
            accepted_records, accepted_costs, stats = ps._collect_inside_records_with_extension(
                cache,
                prior,
                int(base.n_train),
                pilot_n0=80,
                rho_target=float(rho),
                radius_scale=1.0,
            )
            del accepted_records
            stage, curve = collect_rank_curve(cache["stdlp"], accepted_costs[: int(base.n_train)], prior)
            test_costs = cache["test_costs"]
            inside_rate = float(sum(ps.inside_ellipsoid(c, prior) for c in test_costs) / max(len(test_costs), 1))
            rows.append(
                {
                    "case": case.slug,
                    "title": case.title,
                    "rho_target": float(rho),
                    "pilot_n0": 80,
                    "radius_scale": 1.0,
                    "final_rank": int(stage.U.shape[1]),
                    "first_rank": int(stage.rank_after_sample[0]) if stage.rank_after_sample else 0,
                    "estimated_rho": float(prior.rho),
                    "inside_rate_test_pool": inside_rate,
                    "accept_extra_draws": int(stats.get("extra_draws", 0)),
                    "accept_pool_size_final": int(stats.get("pool_size_final", len(cache["accept_records"]))),
                }
            )
            for sample_idx, rank in enumerate(curve):
                curve_rows.append(
                    {
                        "case": case.slug,
                        "title": case.title,
                        "rho_target": float(rho),
                        "sample": int(sample_idx),
                        "rank_after_sample": int(rank),
                        "final_rank": int(stage.U.shape[1]),
                    }
                )
    pd.DataFrame(rows).to_csv(out_dir / "rho_sweep_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out_dir / "rho_sweep_rank_curves.csv", index=False)


def run_n0_sweep(root: Path, case_slugs, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = diag_base_args()
    n0_list = [20, 40, 80, 160]
    rows = []
    for slug in case_slugs:
        case = ps.find_case(root, slug)
        cache = ps.build_beyond_case_cache(
            case=case,
            base_args=base,
            max_pilot_n0=max(n0_list),
            ambient_radius_scale=1.5,
            accept_pool_size=200,
            test_pool_size=40,
        )
        for n0 in n0_list:
            prior = ps.estimate_ellipsoid_prior(
                cache["pilot_costs"][: int(n0)],
                nominal_k=int(cache["args"].prior_nominal_k),
                target_outside_mass=0.10,
                radius_scale=1.0,
                sample_radius_frac=float(cache["args"].sample_radius_frac),
            )
            _accepted_records, accepted_costs, stats = ps._collect_inside_records_with_extension(
                cache,
                prior,
                int(base.n_train),
                pilot_n0=int(n0),
                rho_target=0.10,
                radius_scale=1.0,
            )
            stage, _curve = collect_rank_curve(cache["stdlp"], accepted_costs[: int(base.n_train)], prior)
            test_costs = cache["test_costs"]
            inside_rate = float(sum(ps.inside_ellipsoid(c, prior) for c in test_costs) / max(len(test_costs), 1))
            rows.append(
                {
                    "case": case.slug,
                    "title": case.title,
                    "rho_target": 0.10,
                    "pilot_n0": int(n0),
                    "radius_scale": 1.0,
                    "final_rank": int(stage.U.shape[1]),
                    "first_rank": int(stage.rank_after_sample[0]) if stage.rank_after_sample else 0,
                    "estimated_rho": float(prior.rho),
                    "inside_rate_test_pool": inside_rate,
                    "accept_extra_draws": int(stats.get("extra_draws", 0)),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "n0_sweep_summary.csv", index=False)


def run_radius_scale_sweep(root: Path, case_slugs, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = diag_base_args()
    rs_list = [0.85, 1.00, 1.15, 1.30]
    rows = []
    for slug in case_slugs:
        case = ps.find_case(root, slug)
        cache = ps.build_beyond_case_cache(
            case=case,
            base_args=base,
            max_pilot_n0=80,
            ambient_radius_scale=1.5,
            accept_pool_size=200,
            test_pool_size=40,
        )
        for rs in rs_list:
            prior = ps.estimate_ellipsoid_prior(
                cache["pilot_costs"][:80],
                nominal_k=int(cache["args"].prior_nominal_k),
                target_outside_mass=0.10,
                radius_scale=float(rs),
                sample_radius_frac=float(cache["args"].sample_radius_frac),
            )
            _accepted_records, accepted_costs, stats = ps._collect_inside_records_with_extension(
                cache,
                prior,
                int(base.n_train),
                pilot_n0=80,
                rho_target=0.10,
                radius_scale=float(rs),
            )
            stage, _curve = collect_rank_curve(cache["stdlp"], accepted_costs[: int(base.n_train)], prior)
            test_costs = cache["test_costs"]
            inside_rate = float(sum(ps.inside_ellipsoid(c, prior) for c in test_costs) / max(len(test_costs), 1))
            rows.append(
                {
                    "case": case.slug,
                    "title": case.title,
                    "rho_target": 0.10,
                    "pilot_n0": 80,
                    "radius_scale": float(rs),
                    "final_rank": int(stage.U.shape[1]),
                    "first_rank": int(stage.rank_after_sample[0]) if stage.rank_after_sample else 0,
                    "estimated_rho": float(prior.rho),
                    "inside_rate_test_pool": inside_rate,
                    "accept_extra_draws": int(stats.get("extra_draws", 0)),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "radius_scale_sweep_summary.csv", index=False)


def plot_grid(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, xlabel: str, ylabel: str) -> None:
    titles = list(dict.fromkeys(df["title"].tolist()))
    n = len(titles)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, title in zip(axes.ravel(), titles):
        ax.axis("on")
        sub = df[df["title"] == title].sort_values(x_col)
        ax.plot(sub[x_col], sub[y_col], color="#d62728", marker="o", linestyle="--", linewidth=1.7, markersize=4.0)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path(".").resolve()
    out_root = root / "results_beyond_dimension_diagnostics_v1"
    case_slugs = [case.slug for case in ps.all_cases(root)]
    rho_dir = out_root / "rho_sweep"
    n0_dir = out_root / "n0_sweep"
    rs_dir = out_root / "radius_scale_sweep"
    run_rho_sweep(root, case_slugs, rho_dir)
    run_n0_sweep(root, case_slugs, n0_dir)
    run_radius_scale_sweep(root, case_slugs, rs_dir)
    plot_grid(pd.read_csv(rho_dir / "rho_sweep_summary.csv"), "rho_target", "final_rank", rho_dir / "figure_rho_vs_final_rank.png", "rho", "Learned dimension")
    plot_grid(pd.read_csv(n0_dir / "n0_sweep_summary.csv"), "pilot_n0", "final_rank", n0_dir / "figure_n0_vs_final_rank.png", "# pilot samples", "Learned dimension")
    plot_grid(pd.read_csv(rs_dir / "radius_scale_sweep_summary.csv"), "radius_scale", "final_rank", rs_dir / "figure_radius_scale_vs_final_rank.png", "radius_scale", "Learned dimension")
    plot_grid(pd.read_csv(rho_dir / "rho_sweep_summary.csv"), "rho_target", "inside_rate_test_pool", rho_dir / "figure_rho_vs_inside_rate.png", "rho", "Inside-rate")
    plot_grid(pd.read_csv(n0_dir / "n0_sweep_summary.csv"), "pilot_n0", "inside_rate_test_pool", n0_dir / "figure_n0_vs_inside_rate.png", "# pilot samples", "Inside-rate")
    plot_grid(pd.read_csv(rs_dir / "radius_scale_sweep_summary.csv"), "radius_scale", "inside_rate_test_pool", rs_dir / "figure_radius_scale_vs_inside_rate.png", "radius_scale", "Inside-rate")


if __name__ == "__main__":
    main()
