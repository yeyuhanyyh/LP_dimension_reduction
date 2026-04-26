#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import final_projection_figure_suite as fps
import prior_stress_and_beyond_suite as ps
from compare_fixedX_family_suite import FixedXBundle, make_fixed_x_bundle
from compare_ours_exact_vs_pelp_fixedX import alg2_cumulative
from iwata_sakaue_pelp_projection_compare import ensure_full_solutions


OURS_ONLY_CASES = ["maxflow", "random_lp_C"]
BEYOND_COMPARE_CASES = ["maxflow", "random_lp_C"]
BEYOND_PARAM_CASES = ["maxflow", "mincostflow", "random_lp_C", "sc205"]
TRAIN_GRID = [0, 1, 2, 3, 4, 8, 12, 16, 20]
OURS_ONLY_TRAIN_GRID = [0, 1, 2, 3, 4, 8, 12]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def selected_case(root: Path, slug: str) -> fps.CaseSpec:
    for case in ps.all_cases(root):
        if case.slug == slug:
            return case
    raise KeyError(slug)


def known_prior_weird_updates(case: fps.CaseSpec) -> Dict[str, object]:
    updates = ps.harder_case_updates(case)
    updates.update(
        {
            "sample_mode": "sparse_rare_ball",
            "prior_rho_scale": 4.0,
            "sample_radius_frac": 0.99,
            "center_noise_frac": 0.01,
            "rare_prob": 0.95,
            "rare_amp_frac": 0.98,
            "sparse_rare_max_active": 1,
        }
    )
    if case.slug == "maxflow":
        updates.update(
            {
                "prior_rho_scale": 3.0,
                "rare_prob": 0.85,
                "center_noise_frac": 0.0,
                "n_nodes": 40,
                "n_edges": 160,
                "cost_rank": 10,
            }
        )
    elif case.slug == "random_lp_C":
        updates.update(
            {
                "prior_rho_scale": 4.0,
                "rare_prob": 0.95,
                "randlp_n_vars": 140,
                "randlp_n_eq": 30,
                "randlp_n_ineq": 40,
                "cost_rank": 20,
            }
        )
    return updates


def known_prior_eval_updates(case: fps.CaseSpec) -> Dict[str, object]:
    updates = known_prior_weird_updates(case)
    updates.update(
        {
            "sample_mode": "factor_regime_mixture",
            "regime_count": 8,
            "regime_shift_frac": 0.78,
            "regime_common_frac": 0.04,
            "regime_noise_frac": 0.10,
            "regime_decay": 0.96,
            "center_noise_frac": 0.0,
        }
    )
    return updates


def subgaussian_updates(case: fps.CaseSpec) -> Dict[str, object]:
    updates = ps.harder_case_updates(case)
    updates.update(
        {
            "sample_mode": "masked_factor_gaussian",
            "factor_scale_frac": 0.90,
            "factor_decay": 0.97,
            "mask_prob": 0.08,
            "mask_prob_decay": 0.995,
            "mask_min_active": 1,
            "mask_max_active": 1,
            "mask_boost_active_scale": True,
            "center_noise_frac": 0.0,
            "sample_radius_frac": 0.98,
        }
    )
    if case.slug == "random_lp_C":
        updates.update({"mask_prob": 0.10, "factor_scale_frac": 0.95, "factor_decay": 0.98})
    elif case.slug == "sc205":
        updates.update({"mask_prob": 0.06, "factor_scale_frac": 0.85})
    elif case.slug == "mincostflow":
        updates.update({"mask_prob": 0.06, "factor_scale_frac": 0.82, "factor_decay": 0.98})
    elif case.slug == "maxflow":
        updates.update({"mask_prob": 0.07, "factor_scale_frac": 0.88})
    return updates


def known_prior_base_args() -> object:
    ns = fps.default_namespace()
    ns.n_train = max(OURS_ONLY_TRAIN_GRID)
    ns.n_val = 2
    ns.n_test = 6
    ns.k_list = "20"
    return ns


def beyond_base_args() -> object:
    ns = fps.default_namespace()
    ns.n_train = max(TRAIN_GRID)
    ns.n_val = 4
    ns.n_test = 6
    ns.k_list = "20"
    ns.costonly_epochs = 3
    ns.costonly_patience = 1
    ns.costonly_hidden_dim = 8
    ns.batch_size = 2
    return ns


def build_custom_beyond_cache(
    case_mod: fps.CaseSpec,
    base_args: object,
    max_pilot_n0: int,
    ambient_radius_scale: float,
    accept_pool_size: int,
    test_pool_size: int,
) -> Dict[str, object]:
    args = fps.case_namespace(base_args, case_mod)
    root = Path(".").resolve()
    base_case = selected_case(root, case_mod.slug)
    rng_problem = np.random.default_rng(int(args.seed) + int(base_case.seed_offset))
    problem = fps.build_problem_for_case(case_mod, args, rng_problem)
    stdlp, c0_std, U_reward, rho0, ambient_radius = ps.ambient_sampling_context(problem, args, ambient_radius_scale)

    rng_pilot = np.random.default_rng(int(args.seed) + int(base_case.seed_offset) + 100000)
    pilot_costs = []
    for idx in range(int(max_pilot_n0)):
        rec = ps.sample_ambient_record(problem, args, rng_pilot, stdlp, c0_std, U_reward, ambient_radius, idx, "pilot")
        pilot_costs.append(np.asarray(rec["c_std"], dtype=float))

    rng_accept = np.random.default_rng(int(args.seed) + int(base_case.seed_offset) + 200000)
    accept_records = []
    accept_costs = []
    for idx in range(int(accept_pool_size)):
        rec = ps.sample_ambient_record(problem, args, rng_accept, stdlp, c0_std, U_reward, ambient_radius, idx, "accept")
        accept_records.append(rec)
        accept_costs.append(np.asarray(rec["c_std"], dtype=float))

    test_records = []
    test_costs = []
    for idx in range(int(test_pool_size)):
        rec = ps.sample_ambient_record(problem, args, rng_accept, stdlp, c0_std, U_reward, ambient_radius, idx, "test")
        test_records.append(rec)
        test_costs.append(np.asarray(rec["c_std"], dtype=float))

    return {
        "case": base_case,
        "case_mod": case_mod,
        "args": args,
        "problem": problem,
        "stdlp": stdlp,
        "rho0": float(rho0),
        "ambient_radius_scale": float(ambient_radius_scale),
        "pilot_costs": np.vstack(pilot_costs),
        "accept_records": list(accept_records),
        "accept_costs": np.vstack(accept_costs),
        "test_records": list(test_records),
        "test_costs": np.vstack(test_costs),
    }


def prepare_raw_ambient_data_from_cache(
    cache: Dict[str, object],
    n_train_max: int,
    n_val: int,
    test_n: int,
) -> fps.PreparedCaseData:
    args = copy.deepcopy(cache["args"])
    args.n_train = int(n_train_max)
    args.n_val = int(n_val)
    args.n_test = int(test_n)
    args.k_list = str(int(cache["case_mod"].sample_k))
    problem = cache["problem"]
    stdlp = cache["stdlp"]

    train_records = list(cache["accept_records"])[: int(n_train_max)]
    val_records = list(cache["accept_records"])[int(n_train_max) : int(n_train_max + n_val)]
    test_records = list(cache["test_records"])[: int(test_n)]
    train = [ps.build_instance_from_record(problem, rec) for rec in train_records]
    val = [ps.build_instance_from_record(problem, rec) for rec in val_records]
    test = [ps.build_instance_from_record(problem, rec) for rec in test_records]
    ensure_full_solutions(train)
    ensure_full_solutions(val)
    ensure_full_solutions(test)
    for rec, inst in zip(train_records, train):
        ps._store_full_solution_to_record(inst, rec, prefix="orig")
    for rec, inst in zip(val_records, val):
        ps._store_full_solution_to_record(inst, rec, prefix="orig")
    for rec, inst in zip(test_records, test):
        ps._store_full_solution_to_record(inst, rec, prefix="orig")

    proj_train = fps.bridge_instances_for_projection(problem, train)
    proj_val = fps.bridge_instances_for_projection(problem, val)
    proj_test = fps.bridge_instances_for_projection(problem, test)
    for rec, inst in zip(train_records, proj_train):
        ps._restore_full_solution_from_record(inst, rec, prefix="proj")
    for rec, inst in zip(val_records, proj_val):
        ps._restore_full_solution_from_record(inst, rec, prefix="proj")
    for rec, inst in zip(test_records, proj_test):
        ps._restore_full_solution_from_record(inst, rec, prefix="proj")
    ensure_full_solutions(proj_train)
    ensure_full_solutions(proj_val)
    ensure_full_solutions(proj_test)
    for rec, inst in zip(train_records, proj_train):
        ps._store_full_solution_to_record(inst, rec, prefix="proj")
    for rec, inst in zip(val_records, proj_val):
        ps._store_full_solution_to_record(inst, rec, prefix="proj")
    for rec, inst in zip(test_records, proj_test):
        ps._store_full_solution_to_record(inst, rec, prefix="proj")

    prior = ps.estimate_ellipsoid_prior(
        np.asarray(cache["pilot_costs"], dtype=float)[:40],
        nominal_k=int(args.prior_nominal_k),
        target_outside_mass=0.10,
        radius_scale=1.0,
        sample_radius_frac=float(getattr(args, "sample_radius_frac", 0.98)),
    )
    cost_dim = int(stdlp.dim)

    def _stack_costs(records: Sequence[Dict[str, object]]) -> np.ndarray:
        rows = [np.asarray(rec["c_std"], dtype=float) for rec in records]
        if rows:
            return np.vstack(rows)
        return np.zeros((0, cost_dim), dtype=float)

    bundle = FixedXBundle(
        train=train,
        val=val,
        test=test,
        Ctrain=_stack_costs(train_records),
        Cval=_stack_costs(val_records),
        Ctest=_stack_costs(test_records),
        stdlp=stdlp,
        prior=prior,
        truth={"prior_mode": "ambient_raw", "ambient_radius_scale": float(cache["ambient_radius_scale"])},
        problem=problem,
    )
    return fps.PreparedCaseData(
        case=cache["case_mod"],
        args=args,
        problem=problem,
        bundle=bundle,
        proj_train=proj_train,
        proj_val=proj_val,
        proj_test=proj_test,
    )


def zero_train_result(data: fps.PreparedCaseData, methods: Sequence[str]) -> Dict[str, pd.DataFrame]:
    raw_rows = []
    for proj_inst, full_inst in zip(data.proj_test, data.bundle.test):
        anchor_obj = float(getattr(proj_inst, "objective_constant", 0.0))
        ratio = fps.objective_ratio(anchor_obj, float(full_inst.full_obj))
        for method in methods:
            raw_rows.append(
                {
                    "method": method,
                    "K": int(data.case.sample_k),
                    "instance": full_inst.name,
                    "objective": anchor_obj,
                    "full_objective": float(full_inst.full_obj),
                    "objective_ratio": ratio,
                    "time": 0.0,
                    "success": 1.0,
                }
            )
    raw_df = pd.DataFrame(raw_rows)
    summary = fps.summarize_ratios_and_times(raw_df.to_dict("records"))
    base_obj_by_instance = {inst.name: float(getattr(inst, "objective_constant", 0.0)) for inst in data.proj_test}
    quality = fps.summarize_anchor_quality(raw_df, base_obj_by_instance)
    return {"summary": summary, "quality": quality}


def merge_summary_quality(result: Dict[str, pd.DataFrame], case: fps.CaseSpec, n_train: int) -> pd.DataFrame:
    summary = result["summary"].copy()
    quality = result["quality"].copy()
    summary = summary[summary["K"] == int(case.sample_k)].copy()
    if not quality.empty:
        quality = quality[quality["K"] == int(case.sample_k)].copy()
        summary = summary.merge(
            quality[["method", "K", "anchor_quality_mean", "anchor_quality_se"]],
            on=["method", "K"],
            how="left",
        )
    summary["case"] = case.slug
    summary["title"] = case.title
    summary["family"] = case.family
    summary["n_train"] = int(n_train)
    return summary


def plot_ours_only_anchor_quality(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    cases = list(dict.fromkeys(df["case"].tolist()))
    ncols = min(3, len(cases))
    nrows = int(math.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        sub = df[df["case"] == case].sort_values("n_train")
        ax.plot(sub["n_train"], sub["anchor_quality_mean"], color="#d62728", marker="o", linestyle="--", linewidth=2.0)
        if "rank_after_sample" in sub.columns:
            for x, y, r in zip(sub["n_train"], sub["anchor_quality_mean"], sub["rank_after_sample"]):
                ax.text(float(x), float(y) + 0.02, f"{int(r)}", fontsize=8, ha="center", color="#555555")
        ax.set_title(sub["title"].iloc[0])
        ax.set_xlabel("# training LPs")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    for row in axes_arr:
        row[0].set_ylabel("Average normalized improvement")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_rank_curve_grid(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    cases = list(dict.fromkeys(df["case"].tolist()))
    ncols = min(3, len(cases))
    nrows = int(math.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), sharey=False)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        sub = df[df["case"] == case].sort_values("sample")
        ax.plot(sub["sample"], sub["rank_after_sample"], color="#d62728", marker="o", linestyle="--", linewidth=2.0)
        ax.set_title(sub["title"].iloc[0])
        ax.set_xlabel("# training LPs")
        ax.set_ylabel("Learned dimension")
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_param_grid(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, xlabel: str, ylabel: str) -> None:
    if df.empty:
        return
    cases = list(dict.fromkeys(df["case"].tolist()))
    ncols = min(3, len(cases))
    nrows = int(math.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), sharey=False)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        sub = df[df["case"] == case].sort_values(x_col)
        ax.plot(sub[x_col], sub[y_col], color="#d62728", marker="o", linestyle="--", linewidth=2.0)
        ax.set_title(sub["title"].iloc[0])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def extract_learned_rank(result: Dict[str, pd.DataFrame]) -> float:
    rank_df = result.get("rank", pd.DataFrame())
    if rank_df is not None and not rank_df.empty and "rank_after_sample" in rank_df.columns:
        return float(np.nanmax(rank_df["rank_after_sample"].to_numpy(dtype=float)))
    fit_df = result.get("fit", pd.DataFrame())
    if fit_df is not None and not fit_df.empty and "method" in fit_df.columns:
        ours = fit_df[fit_df["method"] == "OursStageI"]
        if not ours.empty and "K" in ours.columns:
            return float(ours["K"].iloc[0])
    return float("nan")


def run_ours_only_gradual(root: Path, out_dir: Path) -> None:
    out_dir = ensure_dir(out_dir)
    base_args = known_prior_base_args()
    rows = []
    rank_rows = []
    for slug in OURS_ONLY_CASES:
        case = selected_case(root, slug)
        train_case = ps.clone_case(case, known_prior_weird_updates(case))
        test_case = ps.clone_case(case, known_prior_eval_updates(case))
        train_args = fps.case_namespace(base_args, train_case)
        train_args.n_train = int(max(OURS_ONLY_TRAIN_GRID))
        train_args.n_val = int(base_args.n_val)
        train_args.n_test = 1
        train_args.k_list = str(int(case.sample_k))
        test_args = fps.case_namespace(base_args, test_case)
        test_args.n_train = 1
        test_args.n_val = 1
        test_args.n_test = int(base_args.n_test)
        test_args.k_list = str(int(case.sample_k))
        rng_problem = np.random.default_rng(int(train_args.seed) + int(case.seed_offset))
        problem = fps.build_problem_for_case(train_case, train_args, rng_problem)
        train_bundle = make_fixed_x_bundle(problem, train_args, np.random.default_rng(int(train_args.seed) + int(case.seed_offset) + 11))
        test_bundle = make_fixed_x_bundle(problem, test_args, np.random.default_rng(int(train_args.seed) + int(case.seed_offset) + 29))
        bundle = FixedXBundle(
            train=train_bundle.train,
            val=train_bundle.val,
            test=test_bundle.test,
            Ctrain=train_bundle.Ctrain,
            Cval=train_bundle.Cval,
            Ctest=test_bundle.Ctest,
            stdlp=train_bundle.stdlp,
            prior=train_bundle.prior,
            truth={
                "prior_mode": "known_mixed_split",
                "train_sample_mode": np.asarray([train_args.sample_mode]),
                "test_sample_mode": np.asarray([test_args.sample_mode]),
                **train_bundle.truth,
            },
            problem=problem,
        )
        ensure_full_solutions(bundle.train)
        ensure_full_solutions(bundle.val)
        ensure_full_solutions(bundle.test)
        proj_train = fps.bridge_instances_for_projection(problem, bundle.train)
        proj_val = fps.bridge_instances_for_projection(problem, bundle.val)
        proj_test = fps.bridge_instances_for_projection(problem, bundle.test)
        ensure_full_solutions(proj_train)
        ensure_full_solutions(proj_val)
        ensure_full_solutions(proj_test)
        data = fps.PreparedCaseData(
            case=train_case,
            args=train_args,
            problem=problem,
            bundle=bundle,
            proj_train=proj_train,
            proj_val=proj_val,
            proj_test=proj_test,
        )
        stage = alg2_cumulative(data.bundle.stdlp, data.bundle.Ctrain, data.bundle.prior, verbose=False)
        rank_curve = [0, *stage.rank_after_sample]
        for sample, rank in enumerate(rank_curve):
            rank_rows.append(
                {
                    "case": case.slug,
                    "title": case.title,
                    "sample": int(sample),
                    "rank_after_sample": int(rank),
                    "final_rank": int(stage.U.shape[1]),
                }
            )
        for n_train in OURS_ONLY_TRAIN_GRID:
            case_out = ensure_dir(out_dir / case.slug / f"ntrain_{n_train:03d}")
            if int(n_train) == 0:
                result = zero_train_result(data, ["OursExact"])
            else:
                result = fps.run_case_from_prepared(
                    data,
                    case_out,
                    methods=["OursExact"],
                    train_limit=int(n_train),
                    force=False,
                    write_rank=(int(n_train) == int(max(OURS_ONLY_TRAIN_GRID))),
                )
            cur = merge_summary_quality(result, case, int(n_train))
            cur = cur[cur["method"] == "OursExact"].copy()
            cur["rank_after_sample"] = int(rank_curve[min(int(n_train), len(rank_curve) - 1)])
            rows.append(cur)
    summary_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    rank_df = pd.DataFrame(rank_rows)
    summary_df.to_csv(out_dir / "ours_only_gradual_summary.csv", index=False)
    rank_df.to_csv(out_dir / "ours_only_gradual_rank.csv", index=False)
    plot_ours_only_anchor_quality(summary_df, out_dir / "figure_ours_only_anchor_quality.png")
    plot_rank_curve_grid(rank_df, out_dir / "figure_ours_only_rank_growth.png")


def run_beyond_ours_parameter_sweep(root: Path, out_dir: Path) -> None:
    out_dir = ensure_dir(out_dir)
    base_args = beyond_base_args()
    rho_values = [0.40, 0.20, 0.10, 0.03, 0.01]
    n0_values = [20, 40, 80, 160]
    rs_values = [0.85, 1.00, 1.15, 1.30]
    rho_rows = []
    n0_rows = []
    rs_rows = []
    for slug in BEYOND_PARAM_CASES:
        case = selected_case(root, slug)
        case_mod = ps.clone_case(case, subgaussian_updates(case))
        cache = build_custom_beyond_cache(
            case_mod=case_mod,
            base_args=base_args,
            max_pilot_n0=max(n0_values),
            ambient_radius_scale=1.6,
            accept_pool_size=200,
            test_pool_size=30,
        )
        for rho in rho_values:
            data, diag = ps.make_beyond_prior_data_from_cache(cache, rho_target=float(rho), radius_scale=1.0, pilot_n0=80, test_n=int(base_args.n_test))
            result = fps.run_case_from_prepared(
                data,
                ensure_dir(out_dir / f"rho_{slug}_{str(rho).replace('.', 'p')}"),
                methods=["OursExact"],
                train_limit=int(base_args.n_train),
                force=False,
                write_rank=True,
            )
            cur = merge_summary_quality(result, case, int(base_args.n_train))
            cur = cur[cur["method"] == "OursExact"].copy()
            cur["rho_target"] = float(rho)
            cur["pilot_n0"] = 80
            cur["radius_scale"] = 1.0
            cur["learned_rank"] = extract_learned_rank(result)
            cur["accept_extra_draws"] = int(diag.get("accept_extra_draws", 0))
            rho_rows.append(cur)
        for n0 in n0_values:
            data, diag = ps.make_beyond_prior_data_from_cache(cache, rho_target=0.10, radius_scale=1.0, pilot_n0=int(n0), test_n=int(base_args.n_test))
            result = fps.run_case_from_prepared(
                data,
                ensure_dir(out_dir / f"n0_{slug}_{int(n0):03d}"),
                methods=["OursExact"],
                train_limit=int(base_args.n_train),
                force=False,
                write_rank=True,
            )
            cur = merge_summary_quality(result, case, int(base_args.n_train))
            cur = cur[cur["method"] == "OursExact"].copy()
            cur["rho_target"] = 0.10
            cur["pilot_n0"] = int(n0)
            cur["radius_scale"] = 1.0
            cur["learned_rank"] = extract_learned_rank(result)
            cur["accept_extra_draws"] = int(diag.get("accept_extra_draws", 0))
            n0_rows.append(cur)
        for rs in rs_values:
            data, diag = ps.make_beyond_prior_data_from_cache(cache, rho_target=0.10, radius_scale=float(rs), pilot_n0=80, test_n=int(base_args.n_test))
            result = fps.run_case_from_prepared(
                data,
                ensure_dir(out_dir / f"rs_{slug}_{str(rs).replace('.', 'p')}"),
                methods=["OursExact"],
                train_limit=int(base_args.n_train),
                force=False,
                write_rank=True,
            )
            cur = merge_summary_quality(result, case, int(base_args.n_train))
            cur = cur[cur["method"] == "OursExact"].copy()
            cur["rho_target"] = 0.10
            cur["pilot_n0"] = 80
            cur["radius_scale"] = float(rs)
            cur["learned_rank"] = extract_learned_rank(result)
            cur["accept_extra_draws"] = int(diag.get("accept_extra_draws", 0))
            rs_rows.append(cur)
    rho_df = pd.concat(rho_rows, ignore_index=True) if rho_rows else pd.DataFrame()
    n0_df = pd.concat(n0_rows, ignore_index=True) if n0_rows else pd.DataFrame()
    rs_df = pd.concat(rs_rows, ignore_index=True) if rs_rows else pd.DataFrame()
    rho_df.to_csv(out_dir / "beyond_ours_rho_sweep.csv", index=False)
    n0_df.to_csv(out_dir / "beyond_ours_n0_sweep.csv", index=False)
    rs_df.to_csv(out_dir / "beyond_ours_radius_scale_sweep.csv", index=False)
    plot_param_grid(rho_df, "rho_target", "anchor_quality_mean", out_dir / "figure_beyond_ours_rho_anchor_quality.png", "rho", "Average normalized improvement")
    plot_param_grid(rho_df, "rho_target", "learned_rank", out_dir / "figure_beyond_ours_rho_rank.png", "rho", "Learned dimension")
    plot_param_grid(n0_df, "pilot_n0", "anchor_quality_mean", out_dir / "figure_beyond_ours_n0_anchor_quality.png", "# pilot samples", "Average normalized improvement")
    plot_param_grid(n0_df, "pilot_n0", "learned_rank", out_dir / "figure_beyond_ours_n0_rank.png", "# pilot samples", "Learned dimension")
    plot_param_grid(rs_df, "radius_scale", "anchor_quality_mean", out_dir / "figure_beyond_ours_radius_scale_anchor_quality.png", "radius scale", "Average normalized improvement")
    plot_param_grid(rs_df, "radius_scale", "learned_rank", out_dir / "figure_beyond_ours_radius_scale_rank.png", "radius scale", "Learned dimension")


def run_beyond_anchor_compare(root: Path, out_dir: Path) -> None:
    out_dir = ensure_dir(out_dir)
    base_args = beyond_base_args()
    rows = []
    case_order = []
    title_map = {}
    k_map = {}
    for slug in BEYOND_COMPARE_CASES:
        case = selected_case(root, slug)
        case_mod = ps.clone_case(case, subgaussian_updates(case))
        cache = build_custom_beyond_cache(
            case_mod=case_mod,
            base_args=base_args,
            max_pilot_n0=80,
            ambient_radius_scale=1.6,
            accept_pool_size=200,
            test_pool_size=30,
        )
        ours_data = ps.make_beyond_prior_data_from_cache(
            cache,
            rho_target=0.10,
            radius_scale=1.15,
            pilot_n0=80,
            test_n=int(base_args.n_test),
        )[0]
        baseline_data = prepare_raw_ambient_data_from_cache(
            cache,
            n_train_max=int(base_args.n_train),
            n_val=int(base_args.n_val),
            test_n=int(base_args.n_test),
        )
        case_order.append(case.slug)
        title_map[case.slug] = case.title
        k_map[case.slug] = int(case.sample_k)
        for n_train in TRAIN_GRID:
            sub_out = ensure_dir(out_dir / case.slug / f"ntrain_{n_train:03d}")
            if int(n_train) == 0:
                zero = zero_train_result(ours_data, ["Full", "OursExact", "PCA", "SGA", "CostOnly"])
                cur = merge_summary_quality(zero, case, 0)
            else:
                ours_result = fps.run_case_from_prepared(
                    ours_data,
                    ensure_dir(sub_out / "ours"),
                    methods=["OursExact"],
                    train_limit=int(n_train),
                    force=False,
                    write_rank=(int(n_train) == int(max(TRAIN_GRID))),
                )
                base_result = fps.run_case_from_prepared(
                    baseline_data,
                    ensure_dir(sub_out / "baselines"),
                    methods=["Full", "PCA", "SGA", "CostOnly"],
                    train_limit=int(n_train),
                    force=False,
                    write_rank=False,
                )
                cur = pd.concat(
                    [
                        merge_summary_quality(ours_result, case, int(n_train)),
                        merge_summary_quality(base_result, case, int(n_train)),
                    ],
                    ignore_index=True,
                )
            rows.append(cur)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(out_dir / "beyond_anchor_compare_sample_efficiency.csv", index=False)
    fps.plot_sample_efficiency_grid(
        out,
        case_order,
        title_map,
        k_map,
        out_dir / "figure_beyond_anchor_compare_anchor_quality.png",
        y_col="anchor_quality_mean",
        yerr_col="anchor_quality_se",
        ylabel="Average normalized improvement",
    )


def main() -> None:
    root = Path(".").resolve()
    out_root = ensure_dir(root / "results_anchor_quality_unknownC_v1")
    run_ours_only_gradual(root, ensure_dir(out_root / "ours_only_gradual"))
    run_beyond_ours_parameter_sweep(root, ensure_dir(out_root / "beyond_ours_param_sweep"))
    run_beyond_anchor_compare(root, ensure_dir(out_root / "beyond_anchor_compare"))
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "ours_only_cases": list(OURS_ONLY_CASES),
                "beyond_compare_cases": list(BEYOND_COMPARE_CASES),
                "beyond_param_cases": list(BEYOND_PARAM_CASES),
                "train_grid": list(TRAIN_GRID),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
