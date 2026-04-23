#!/usr/bin/env python3
"""Small OursExact-only robustness check over multiple random seeds.

This script intentionally does not replace the main experimental figures.  It
creates a lightweight diagnostic showing how quickly Algorithm 2 learns the
decision-sufficient dimension across several random seeds on a natural
random-standard-form family.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import final_projection_figure_suite as fps
from compare_ours_exact_vs_pelp_fixedX import alg2_cumulative


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def robustness_case() -> fps.CaseSpec:
    return fps.CaseSpec(
        slug="random_stdform_robust",
        title="RandomStdForm Robust",
        family="random_lp",
        args_updates={
            "randlp_n_vars": 90,
            "randlp_n_eq": 22,
            "randlp_n_ineq": 30,
            "prior_nominal_k": 18,
            "cost_rank": 18,
            "sample_radius_frac": 0.90,
            "sample_mode": "factor_gaussian",
            "factor_scale_frac": 0.70,
            "factor_decay": 0.86,
            "center_noise_frac": 0.0,
        },
        sample_k=20,
        seed_offset=9000,
    )


def run(args: argparse.Namespace) -> None:
    base = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    case = robustness_case()

    rows: list[dict[str, object]] = []
    for seed in seeds:
        ns = fps.default_namespace()
        ns.seed = int(seed)
        ns.suite_profile = "natural"
        ns.n_train = int(args.n_train)
        ns.n_val = int(args.n_val)
        ns.n_test = int(args.n_test)
        ns.k_list = "20"
        data = fps.prepare_case_data(case, ns)
        stage = alg2_cumulative(data.bundle.stdlp, data.bundle.Ctrain[: ns.n_train], data.bundle.prior, verbose=False)
        ranks = [0] + [int(x) for x in stage.rank_after_sample]
        final_rank = max(ranks)
        for n_train, rank in enumerate(ranks):
            rows.append(
                {
                    "case": case.slug,
                    "title": case.title,
                    "seed": int(seed),
                    "n_train": int(n_train),
                    "rank": int(rank),
                    "final_rank": int(final_rank),
                    "rank_fraction": float(rank / final_rank) if final_rank > 0 else 0.0,
                }
            )

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "ours_rank_multiseed_raw.csv", index=False)
    summary = raw.groupby(["case", "title", "n_train"], as_index=False).agg(
        rank_mean=("rank", "mean"),
        rank_std=("rank", "std"),
        rank_fraction_mean=("rank_fraction", "mean"),
        rank_fraction_std=("rank_fraction", "std"),
        n=("seed", "count"),
    )
    summary["rank_se"] = summary["rank_std"] / np.sqrt(summary["n"])
    summary["rank_fraction_se"] = summary["rank_fraction_std"] / np.sqrt(summary["n"])
    summary.to_csv(out_dir / "ours_rank_multiseed_summary.csv", index=False)

    plot_summary(summary, out_dir / "figure_ours_multiseed_rank_robustness.png")
    print(f"Wrote {out_dir / 'figure_ours_multiseed_rank_robustness.png'}")


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    summary = summary[summary["n_train"] <= 20].sort_values("n_train")
    x = summary["n_train"].to_numpy(dtype=float)
    y = summary["rank_fraction_mean"].to_numpy(dtype=float)
    std = summary["rank_fraction_std"].fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.45))
    ax.fill_between(
        x,
        np.maximum(0.0, y - std),
        np.minimum(1.05, y + std),
        color="#d62728",
        alpha=0.16,
        linewidth=0,
        label="+/- 1 std. over seeds",
    )
    ax.plot(
        x,
        y,
        color="#d62728",
        marker="D",
        linestyle=(0, (6, 2)),
        linewidth=2.2,
        markersize=5.0,
        label="OursExact mean",
    )
    ax.axhline(1.0, color="#555555", linestyle=(0, (4, 2)), linewidth=1.1, alpha=0.7)
    ax.set_title("OursExact Robustness: RandomStdForm")
    ax.set_xlabel("# training LPs")
    ax.set_ylabel("Learned dimension / final learned dimension")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([0, 1, 2, 3, 4, 8, 12, 16, 20])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", default=".")
    parser.add_argument(
        "--out_dir",
        default="results_projection_suite_natural_refined_v7_full/ours_robustness_multiseed",
    )
    parser.add_argument("--seeds", default="17,23,31,47,71")
    parser.add_argument("--n_train", type=int, default=20)
    parser.add_argument("--n_val", type=int, default=6)
    parser.add_argument("--n_test", type=int, default=12)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
