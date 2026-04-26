#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import final_projection_figure_suite as fps
import prior_stress_and_beyond_suite as ps
import run_anchor_quality_unknownC_suite as aq
from compare_fixedX_family_suite import FixedXBundle
from iwata_sakaue_pelp_projection_compare import ensure_full_solutions


CASES = ["maxflow", "sc205", "random_lp_C"]
BUDGETS = [0, 8, 12, 20, 40]
TEST_N = 6
RHO_TARGET = 0.10
RADIUS_SCALE = 1.15
BASELINE_METHODS = ["Full", "PCA", "SGA", "CostOnly"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stack_costs(records: Sequence[Dict[str, object]], dim: int) -> np.ndarray:
    rows = [np.asarray(rec["c_std"], dtype=float) for rec in records]
    if rows:
        return np.vstack(rows)
    return np.zeros((0, int(dim)), dtype=float)


def build_prepared_from_records(
    *,
    problem,
    stdlp,
    prior,
    records_train: Sequence[Dict[str, object]],
    records_val: Sequence[Dict[str, object]],
    records_test: Sequence[Dict[str, object]],
    case_mod,
    args,
    truth_mode: str,
) -> fps.PreparedCaseData:
    train = [ps.build_instance_from_record(problem, rec) for rec in records_train]
    val = [ps.build_instance_from_record(problem, rec) for rec in records_val]
    test = [ps.build_instance_from_record(problem, rec) for rec in records_test]
    ensure_full_solutions(train)
    ensure_full_solutions(val)
    ensure_full_solutions(test)
    for rec, inst in zip(records_train, train):
        ps._store_full_solution_to_record(inst, rec, prefix="orig")
    for rec, inst in zip(records_val, val):
        ps._store_full_solution_to_record(inst, rec, prefix="orig")
    for rec, inst in zip(records_test, test):
        ps._store_full_solution_to_record(inst, rec, prefix="orig")

    proj_train = fps.bridge_instances_for_projection(problem, train)
    proj_val = fps.bridge_instances_for_projection(problem, val)
    proj_test = fps.bridge_instances_for_projection(problem, test)
    for rec, inst in zip(records_train, proj_train):
        ps._restore_full_solution_from_record(inst, rec, prefix="proj")
    for rec, inst in zip(records_val, proj_val):
        ps._restore_full_solution_from_record(inst, rec, prefix="proj")
    for rec, inst in zip(records_test, proj_test):
        ps._restore_full_solution_from_record(inst, rec, prefix="proj")
    ensure_full_solutions(proj_train)
    ensure_full_solutions(proj_val)
    ensure_full_solutions(proj_test)

    bundle = FixedXBundle(
        train=train,
        val=val,
        test=test,
        Ctrain=stack_costs(records_train, stdlp.dim),
        Cval=stack_costs(records_val, stdlp.dim),
        Ctest=stack_costs(records_test, stdlp.dim),
        stdlp=stdlp,
        prior=prior,
        truth={"prior_mode": truth_mode},
        problem=problem,
    )
    return fps.PreparedCaseData(
        case=case_mod,
        args=args,
        problem=problem,
        bundle=bundle,
        proj_train=proj_train,
        proj_val=proj_val,
        proj_test=proj_test,
    )


def plot_budget_compare(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    case_order = [slug for slug in CASES if slug in set(df["case"].tolist())]
    title_map = {"maxflow": "MaxFlow", "sc205": "SC205", "random_lp_C": "RandomLP C"}
    styles = {
        "Full": dict(color="#555555", marker="o", ls="--"),
        "OursExact": dict(color="#d62728", marker="D", ls="--"),
        "PCA": dict(color="#1f77b4", marker="s", ls=":"),
        "SGA": dict(color="#ff7f0e", marker="P", ls="--"),
        "CostOnly": dict(color="#8c564b", marker="v", ls="-."),
    }
    fig, axes = plt.subplots(1, len(case_order), figsize=(4.8 * len(case_order), 3.8), sharey=False)
    if len(case_order) == 1:
        axes = [axes]
    for ax, slug in zip(axes, case_order):
        sub = df[df["case"] == slug].copy()
        for method in ["Full", "OursExact", "PCA", "SGA", "CostOnly"]:
            cur = sub[sub["method"] == method].sort_values("total_budget")
            if cur.empty:
                continue
            x = cur["total_budget"].to_numpy()
            y = cur["anchor_quality_mean"].to_numpy()
            se = cur["anchor_quality_se"].fillna(0.0).to_numpy()
            st = styles[method]
            ax.plot(x, y, color=st["color"], marker=st["marker"], linestyle=st["ls"], linewidth=2.0, label=method)
            ax.fill_between(x, np.clip(y - se, 0.0, 1.0), np.clip(y + se, 0.0, 1.0), color=st["color"], alpha=0.10)
        ax.set_title(title_map.get(slug, slug))
        ax.set_xlabel("Total sample budget B")
        ax.set_ylabel("Average normalized improvement")
        ax.set_xticks(BUDGETS)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_budget_acceptance(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    sub = df[df["method"] == "OursExact"][["case", "title", "total_budget", "pilot_n0", "accepted_train"]].drop_duplicates()
    case_order = [slug for slug in CASES if slug in set(sub["case"].tolist())]
    fig, axes = plt.subplots(1, len(case_order), figsize=(4.8 * len(case_order), 3.6), sharey=False)
    if len(case_order) == 1:
        axes = [axes]
    for ax, slug in zip(axes, case_order):
        cur = sub[sub["case"] == slug].sort_values("total_budget")
        x = cur["total_budget"].to_numpy()
        ax.plot(x, cur["accepted_train"].to_numpy(), color="#d62728", marker="D", linestyle="--", linewidth=2.0, label="accepted")
        ax.plot(x, cur["pilot_n0"].to_numpy(), color="#1f77b4", marker="s", linestyle=":", linewidth=2.0, label="pilot n0")
        ax.set_title(cur["title"].iloc[0])
        ax.set_xlabel("Total sample budget B")
        ax.set_ylabel("# samples")
        ax.set_xticks(BUDGETS)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path(".").resolve()
    out_dir = ensure_dir(root / "results_unknownC_budget_matched_v1")
    rows: List[pd.DataFrame] = []
    manifest: List[Dict[str, object]] = []

    base_args = aq.beyond_base_args()
    base_args.n_val = 0
    base_args.n_test = TEST_N

    for slug in CASES:
        case = aq.selected_case(root, slug)
        case_mod = ps.clone_case(case, aq.subgaussian_updates(case))
        cache = aq.build_custom_beyond_cache(
            case_mod=case_mod,
            base_args=base_args,
            max_pilot_n0=max(BUDGETS) // 2 if max(BUDGETS) > 0 else 1,
            ambient_radius_scale=1.6,
            accept_pool_size=max(BUDGETS),
            test_pool_size=30,
        )
        args = copy.deepcopy(cache["args"])
        args.n_val = 0
        args.n_test = TEST_N
        args.k_list = str(int(case.sample_k))
        stdlp = cache["stdlp"]
        problem = cache["problem"]

        for budget in BUDGETS:
            case_out = ensure_dir(out_dir / slug / f"budget_{budget:03d}")
            if budget == 0:
                base_data = aq.prepare_raw_ambient_data_from_cache(cache, n_train_max=0, n_val=0, test_n=TEST_N)
                zero = aq.zero_train_result(base_data, ["Full", "OursExact", "PCA", "SGA", "CostOnly"])
                cur = aq.merge_summary_quality(zero, case, 0)
                cur["total_budget"] = 0
                cur["pilot_n0"] = 0
                cur["raw_tail"] = 0
                cur["accepted_train"] = 0
                cur["accept_rate"] = 0.0
                rows.append(cur)
                continue

            pilot_n0 = budget // 2
            raw_tail = budget - pilot_n0
            prior = ps.estimate_ellipsoid_prior(
                np.asarray(cache["pilot_costs"], dtype=float)[:pilot_n0],
                nominal_k=int(args.prior_nominal_k),
                target_outside_mass=RHO_TARGET,
                radius_scale=RADIUS_SCALE,
                sample_radius_frac=float(getattr(args, "sample_radius_frac", 0.98)),
            )
            raw_records = list(cache["accept_records"])[:raw_tail]
            accepted_records = [
                rec
                for rec in raw_records
                if ps.inside_ellipsoid(np.asarray(rec["c_std"], dtype=float), prior)
            ]
            test_records = list(cache["test_records"])[:TEST_N]
            ours_data = build_prepared_from_records(
                problem=problem,
                stdlp=stdlp,
                prior=prior,
                records_train=accepted_records,
                records_val=[],
                records_test=test_records,
                case_mod=case_mod,
                args=args,
                truth_mode="budget_matched_estimated_prior",
            )
            if accepted_records:
                ours_result = fps.run_case_from_prepared(
                    ours_data,
                    ensure_dir(case_out / "ours"),
                    methods=["OursExact"],
                    train_limit=len(accepted_records),
                    force=True,
                    write_rank=True,
                )
            else:
                ours_result = aq.zero_train_result(ours_data, ["OursExact"])

            base_data = aq.prepare_raw_ambient_data_from_cache(cache, n_train_max=raw_tail, n_val=0, test_n=TEST_N)
            base_result = fps.run_case_from_prepared(
                base_data,
                ensure_dir(case_out / "baselines"),
                methods=BASELINE_METHODS,
                train_limit=raw_tail,
                force=True,
                write_rank=False,
            )
            cur = pd.concat(
                [
                    aq.merge_summary_quality(ours_result, case, raw_tail),
                    aq.merge_summary_quality(base_result, case, raw_tail),
                ],
                ignore_index=True,
            )
            cur["total_budget"] = int(budget)
            cur["pilot_n0"] = int(pilot_n0)
            cur["raw_tail"] = int(raw_tail)
            cur["accepted_train"] = int(len(accepted_records))
            cur["accept_rate"] = float(len(accepted_records) / max(raw_tail, 1))
            rows.append(cur)
            manifest.append(
                {
                    "case": slug,
                    "budget": int(budget),
                    "pilot_n0": int(pilot_n0),
                    "raw_tail": int(raw_tail),
                    "accepted_train": int(len(accepted_records)),
                }
            )

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(out_dir / "budget_matched_compare.csv", index=False)
    (out_dir / "budget_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_budget_compare(out, out_dir / "figure_budget_matched_anchor_quality.png")
    plot_budget_acceptance(out, out_dir / "figure_budget_matched_acceptance.png")


if __name__ == "__main__":
    main()
