#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Targeted experiments for harder priors and beyond-prior estimation.

This script keeps the main final suite untouched and adds two focused probes:

1. Harder-prior stress test:
   enlarge the working prior radius by a factor and measure whether the learned
   OursExact rank grows more gradually with sample size.

2. Beyond-a-given-prior-set probe:
   draw ambient costs from a natural factor / multiplicative distribution,
   estimate an ellipsoidal working prior from an independent pilot sample,
   then compare methods on fresh test costs while sweeping the target outside
   mass rho and/or the radius scale.
"""
from __future__ import annotations

import argparse
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

from compare_fixedX_family_suite import (
    FixedXBundle,
    OriginalFixedXProblem,
    build_lp_instance,
    build_standard_form_problem,
    choose_nominal_prior_k,
    choose_prior_for_problem,
    orthonormalize_columns,
    sample_factor_regime_theta,
    sample_masked_factor_gaussian_theta,
    reward_from_standard_cost,
    sample_ball_point,
    sample_factor_gaussian_theta,
    sample_sparse_rare_theta,
    standard_cost_from_reward,
)
from compare_ours_exact_vs_pelp_fixedX import EllipsoidCostPrior, alg2_cumulative
import final_projection_figure_suite as fps
from iwata_sakaue_pelp_projection_compare import ensure_full_solutions


TARGET_METHODS = ["Full", "OursExact", "PCA", "SGA", "CostOnly"]
HARDER_SAMPLE_GRID = [0, 1, 2, 3, 4, 8, 12, 16, 20]


def all_cases(root: Path) -> List[fps.CaseSpec]:
    return [*fps.synthetic_cases(root), *fps.netlib_cases(root)]


def find_case(root: Path, slug: str) -> fps.CaseSpec:
    slug = str(slug).strip().lower()
    for case in all_cases(root):
        if case.slug.lower() == slug:
            return case
    raise KeyError(f"Unknown case slug: {slug}")


def clone_case(case: fps.CaseSpec, updates: Dict[str, object]) -> fps.CaseSpec:
    merged = dict(case.args_updates)
    merged.update(updates)
    return fps.CaseSpec(
        slug=case.slug,
        title=case.title,
        family=case.family,
        args_updates=merged,
        sample_k=case.sample_k,
        seed_offset=case.seed_offset,
        mps_path=case.mps_path,
        reuse_dir="",
    )


def harder_case_updates(case: fps.CaseSpec) -> Dict[str, object]:
    family = str(case.family)
    updates: Dict[str, object] = {
        "origin_margin_frac": 0.75,
        "reward_margin_frac": 0.92,
        "sample_radius_frac": 0.98,
        "center_noise_frac": 0.02,
        "regime_shift_frac": 0.68,
        "regime_common_frac": 0.10,
        "regime_noise_frac": 0.16,
        "regime_decay": 0.90,
    }
    if family == "mincostflow":
        updates.update(
            {
                "sample_mode": "multiplicative_regime_mixture",
                "mult_log_scale": 0.45,
                "regime_count": 8,
                "regime_shift_frac": 0.72,
                "regime_noise_frac": 0.14,
            }
        )
    elif family == "netlib":
        updates.update(
            {
                "sample_mode": "factor_regime_mixture",
                "regime_count": 6,
                "regime_shift_frac": 0.62,
                "regime_noise_frac": 0.12,
                "center_noise_frac": 0.0,
            }
        )
    else:
        updates.update(
            {
                "sample_mode": "factor_regime_mixture",
                "regime_count": 8,
            }
        )
    if case.slug.startswith("random_lp_"):
        updates.update(
            {
                "regime_count": 10,
                "regime_shift_frac": 0.74,
                "regime_noise_frac": 0.12,
                "center_noise_frac": 0.0,
            }
        )
    if case.slug in {"packing", "maxflow", "shortest_path"}:
        updates.update(
            {
                "regime_shift_frac": 0.76,
                "regime_noise_frac": 0.10,
                "center_noise_frac": 0.0,
            }
        )
    if case.slug == "stair":
        updates.update(
            {
                "sample_mode": "factor_gaussian",
                "origin_margin_frac": 0.60,
                "reward_margin_frac": 0.85,
            }
        )
    return updates


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarize_fulltrain_case(data: fps.PreparedCaseData, out_dir: Path) -> pd.DataFrame:
    saved_k_list = str(data.args.k_list)
    data.args.k_list = str(int(data.case.sample_k))
    try:
        result = fps.run_case_from_prepared(
            data,
            out_dir,
            methods=TARGET_METHODS,
            train_limit=len(data.bundle.train),
            force=False,
            write_rank=True,
        )
    finally:
        data.args.k_list = saved_k_list
    summary = result["summary"].copy()
    summary = summary[summary["K"] == int(data.case.sample_k)].copy()
    quality = result["quality"].copy()
    if not quality.empty:
        quality = quality[quality["K"] == int(data.case.sample_k)].copy()
        summary = summary.merge(
            quality[["method", "K", "anchor_quality_mean", "anchor_quality_se"]],
            on=["method", "K"],
            how="left",
        )
    return summary


def run_harder_prior_probe(
    root: Path,
    case_slugs: Sequence[str],
    scales: Sequence[float],
    out_dir: Path,
    base_args: argparse.Namespace | None = None,
) -> None:
    if base_args is None:
        base_args = fps.default_namespace()
    out_dir = ensure_dir(out_dir)
    rows: List[Dict[str, object]] = []
    compare_rows: List[pd.DataFrame] = []
    title_map: Dict[str, str] = {}

    for slug in case_slugs:
        case = find_case(root, slug)
        title_map[case.slug] = case.title
        for scale in scales:
            updates = harder_case_updates(case)
            updates["prior_rho_scale"] = float(scale)
            case_mod = clone_case(case, updates)
            data = fps.prepare_case_data(case_mod, base_args)
            stage = alg2_cumulative(data.bundle.stdlp, data.bundle.Ctrain, data.bundle.prior, verbose=False)
            for sample_idx, rank in enumerate([0, *stage.rank_after_sample]):
                rows.append(
                    {
                        "case": case.slug,
                        "title": case.title,
                        "prior_rho_scale": float(scale),
                        "sample": int(sample_idx),
                        "rank_after_sample": int(rank),
                        "final_rank": int(stage.U.shape[1]),
                        "first_rank": int(stage.rank_after_sample[0]) if stage.rank_after_sample else 0,
                        "prior_rho": float(data.bundle.prior.rho),
                        "prior_sample_radius": float(data.bundle.prior.sample_radius),
                    }
                )
            case_out = ensure_dir(out_dir / f"{case.slug}_scale_{str(scale).replace('.', 'p')}")
            compare = summarize_fulltrain_case(data, case_out)
            compare["case"] = case.slug
            compare["title"] = case.title
            compare["prior_rho_scale"] = float(scale)
            compare["prior_rho"] = float(data.bundle.prior.rho)
            compare["prior_sample_radius"] = float(data.bundle.prior.sample_radius)
            compare_rows.append(compare)

    rank_df = pd.DataFrame(rows)
    compare_df = pd.concat(compare_rows, ignore_index=True) if compare_rows else pd.DataFrame()
    rank_df.to_csv(out_dir / "harder_prior_rank_growth.csv", index=False)
    compare_df.to_csv(out_dir / "harder_prior_fulltrain_summary.csv", index=False)

    plot_harder_prior_rank(rank_df, out_dir / "figure_harder_prior_rank_growth.png")
    plot_harder_prior_compare(compare_df, out_dir / "figure_harder_prior_fulltrain.png")
    plot_harder_prior_compare(
        compare_df,
        out_dir / "figure_harder_prior_fulltrain_quality.png",
        y_col="anchor_quality_mean",
        y_label="Average normalized improvement",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps({"cases": list(case_slugs), "scales": [float(x) for x in scales]}, indent=2),
        encoding="utf-8",
    )


def run_harder_prior_sample_efficiency(
    root: Path,
    case_slugs: Sequence[str],
    scales: Sequence[float],
    out_dir: Path,
    base_args: argparse.Namespace | None = None,
) -> None:
    if base_args is None:
        base_args = fps.default_namespace()
    out_dir = ensure_dir(out_dir)
    rows: List[pd.DataFrame] = []
    case_order: List[str] = []
    title_map: Dict[str, str] = {}
    k_map: Dict[str, int] = {}

    for slug in case_slugs:
        case = find_case(root, slug)
        for scale in scales:
            scale_tag = str(scale).replace(".", "p")
            panel_case = f"{case.slug}__rho_x_{scale_tag}"
            case_order.append(panel_case)
            title_map[panel_case] = f"{case.title} ($\\rho \\times {scale:g}$)"
            k_map[panel_case] = int(case.sample_k)

            max_train = max(HARDER_SAMPLE_GRID)
            updates = harder_case_updates(case)
            updates.update(
                {
                    "prior_rho_scale": float(scale),
                    "n_train": int(max_train),
                    "k_list": str(int(case.sample_k)),
                }
            )
            case_mod = clone_case(case, updates)
            data = fps.prepare_case_data(case_mod, base_args)
            data.args.k_list = str(int(case.sample_k))
            case_root = ensure_dir(out_dir / panel_case)

            for n_train in HARDER_SAMPLE_GRID:
                sub_out = ensure_dir(case_root / f"ntrain_{int(n_train):03d}")
                if int(n_train) == 0:
                    raw_rows = []
                    for proj_inst, full_inst in zip(data.proj_test, data.bundle.test):
                        anchor_obj = float(getattr(proj_inst, "objective_constant", 0.0))
                        ratio = fps.objective_ratio(anchor_obj, float(full_inst.full_obj))
                        for method in fps.SAMPLE_METHOD_ORDER:
                            raw_rows.append(
                                {
                                    "method": method,
                                    "K": int(case.sample_k),
                                    "instance": full_inst.name,
                                    "objective": anchor_obj,
                                    "full_objective": float(full_inst.full_obj),
                                    "objective_ratio": ratio,
                                    "time": 0.0,
                                    "success": 1.0,
                                }
                            )
                    raw_df = pd.DataFrame(raw_rows)
                    raw_df.to_csv(sub_out / "raw_results.csv", index=False)
                    result_summary = fps.summarize_ratios_and_times(raw_df.to_dict("records"))
                    base_obj_by_instance = {
                        inst.name: float(getattr(inst, "objective_constant", 0.0)) for inst in data.proj_test
                    }
                    result_quality = fps.summarize_anchor_quality(raw_df, base_obj_by_instance)
                    result_summary.to_csv(sub_out / "summary_results.csv", index=False)
                    result_quality.to_csv(sub_out / "quality_summary.csv", index=False)
                    result = {"summary": result_summary, "quality": result_quality}
                else:
                    result = fps.run_case_from_prepared(
                        data,
                        sub_out,
                        methods=fps.SAMPLE_METHOD_ORDER,
                        train_limit=int(n_train),
                        force=False,
                        write_rank=(int(n_train) == int(max_train)),
                    )

                summary = result["summary"].copy()
                quality = result["quality"].copy()
                summary = summary[summary["method"].isin(fps.SAMPLE_METHOD_ORDER)].copy()
                summary["n_train"] = int(n_train)
                summary["sample_k"] = int(case.sample_k)
                summary = summary[summary["K"] == int(case.sample_k)].copy()
                if not quality.empty:
                    quality = quality[quality["method"].isin(fps.SAMPLE_METHOD_ORDER)].copy()
                    quality["n_train"] = int(n_train)
                    quality["sample_k"] = int(case.sample_k)
                    quality = quality[quality["K"] == int(case.sample_k)].copy()
                    summary = summary.merge(
                        quality[["method", "K", "anchor_quality_mean", "anchor_quality_se"]],
                        on=["method", "K"],
                        how="left",
                    )
                summary["case"] = panel_case
                summary["title"] = title_map[panel_case]
                summary["family"] = case.family
                summary["prior_rho_scale"] = float(scale)
                rows.append(summary)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(out_dir / "harder_prior_sample_efficiency.csv", index=False)
    if not out.empty:
        fps.plot_sample_efficiency_grid(
            out,
            case_order,
            title_map,
            k_map,
            out_dir / "figure_harder_prior_sample_efficiency.png",
        )
        fps.plot_sample_efficiency_grid(
            out,
            case_order,
            title_map,
            k_map,
            out_dir / "figure_harder_prior_sample_efficiency_quality.png",
            y_col="anchor_quality_mean",
            yerr_col="anchor_quality_se",
            ylabel="Average normalized improvement",
        )


def estimate_ellipsoid_prior(
    pilot_costs: np.ndarray,
    nominal_k: int,
    target_outside_mass: float,
    radius_scale: float,
    sample_radius_frac: float,
) -> EllipsoidCostPrior:
    X = np.asarray(pilot_costs, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("Need at least two pilot samples to estimate an ellipsoidal prior.")
    c0 = X.mean(axis=0)
    cov = np.cov(X, rowvar=False, bias=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=float)
    d = int(cov.shape[0])
    ridge = max(1e-8, 1e-6 * float(np.trace(cov) / max(d, 1)))
    cov_reg = np.asarray(cov, dtype=float) + ridge * np.eye(d, dtype=float)
    evals, evecs = np.linalg.eigh(cov_reg)
    evals = np.maximum(evals, 1e-10)
    cov_sqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    cov_inv = evecs @ np.diag(1.0 / evals) @ evecs.T
    centered = X - c0.reshape(1, -1)
    maha = np.sqrt(np.maximum(np.sum((centered @ cov_inv) * centered, axis=1), 0.0))
    q = float(np.clip(1.0 - float(target_outside_mass), 0.50, 0.999))
    rho = float(max(1e-6, radius_scale * np.quantile(maha, q)))
    return EllipsoidCostPrior(
        c0=c0,
        cov=cov_reg,
        cov_sqrt=cov_sqrt,
        cov_inv=cov_inv,
        rho=rho,
        nominal_k=int(nominal_k),
        sample_radius=float(sample_radius_frac) * rho,
        origin_margin=float("nan"),
        tau_sorted=np.asarray([], dtype=float),
        pilot_size=int(X.shape[0]),
        target_outside_mass=float(target_outside_mass),
        radius_scale=float(radius_scale),
    )


def inside_ellipsoid(c_std: np.ndarray, prior: EllipsoidCostPrior, tol: float = 1e-9) -> bool:
    diff = np.asarray(c_std, dtype=float).reshape(-1) - np.asarray(prior.c0, dtype=float).reshape(-1)
    val = float(diff @ prior.cov_inv @ diff)
    return val <= float(prior.rho) ** 2 + tol


def ambient_sampling_context(
    problem: OriginalFixedXProblem,
    args: argparse.Namespace,
    ambient_radius_scale: float,
) -> tuple[object, np.ndarray, np.ndarray, float, float]:
    stdlp = build_standard_form_problem(problem)
    c0_std = standard_cost_from_reward(problem.reward_center, stdlp.n_slack)
    nominal_k = choose_nominal_prior_k(args)
    reward_margin_cap = (
        float(args.reward_margin_frac) * float(np.min(problem.reward_center))
        if np.all(problem.reward_center > 1e-8)
        else float("inf")
    )
    rho0, _tau, _basis = choose_prior_for_problem(
        stdlp,
        c0_std,
        nominal_k=nominal_k,
        origin_margin_frac=float(args.origin_margin_frac),
        reward_margin_cap=reward_margin_cap,
    )
    ambient_radius = float(max(ambient_radius_scale, 1e-8)) * float(rho0)
    U_reward = orthonormalize_columns(problem.U_reward)
    return stdlp, c0_std, U_reward, float(rho0), float(ambient_radius)


def sample_ambient_record(
    problem: OriginalFixedXProblem,
    args: argparse.Namespace,
    rng: np.random.Generator,
    stdlp,
    c0_std: np.ndarray,
    U_reward: np.ndarray,
    ambient_radius: float,
    idx: int,
    prefix: str,
) -> Dict[str, object]:
    if args.sample_mode == "factor_gaussian":
        theta, chosen = sample_factor_gaussian_theta(
            rng,
            U_reward.shape[1],
            ambient_radius,
            scale_frac=float(getattr(args, "factor_scale_frac", 0.70)),
            decay=float(getattr(args, "factor_decay", 0.82)),
        )
        reward = np.asarray(problem.reward_center, dtype=float) + U_reward @ theta
    elif args.sample_mode == "masked_factor_gaussian":
        theta, chosen = sample_masked_factor_gaussian_theta(
            rng,
            U_reward.shape[1],
            ambient_radius,
            scale_frac=float(getattr(args, "factor_scale_frac", 0.80)),
            decay=float(getattr(args, "factor_decay", 0.90)),
            mask_prob=float(getattr(args, "mask_prob", 0.18)),
            prob_decay=float(getattr(args, "mask_prob_decay", 0.98)),
            min_active=int(getattr(args, "mask_min_active", 1)),
            max_active=int(getattr(args, "mask_max_active", 2)),
            boost_active_scale=bool(getattr(args, "mask_boost_active_scale", True)),
        )
        reward = np.asarray(problem.reward_center, dtype=float) + U_reward @ theta
    elif args.sample_mode == "multiplicative_factor":
        theta, chosen = sample_factor_gaussian_theta(
            rng,
            U_reward.shape[1],
            ambient_radius,
            scale_frac=float(getattr(args, "factor_scale_frac", 0.50)),
            decay=float(getattr(args, "factor_decay", 0.88)),
        )
        latent = U_reward @ theta
        denom = float(max(np.max(np.abs(latent)), 1e-12))
        mult_log_scale = float(max(getattr(args, "mult_log_scale", 0.35), 1e-4))
        log_mult = np.clip((mult_log_scale / denom) * latent, -mult_log_scale, mult_log_scale)
        reward = np.asarray(problem.reward_center, dtype=float) * np.exp(log_mult)
        if float(getattr(args, "center_noise_frac", 0.0)) > 0.0:
            jitter = sample_ball_point(rng, U_reward.shape[1], float(args.center_noise_frac) * ambient_radius)
            reward = reward + 0.10 * (U_reward @ jitter)
    elif args.sample_mode == "uniform_ball":
        theta = sample_ball_point(rng, U_reward.shape[1], ambient_radius)
        chosen = 0
        reward = np.asarray(problem.reward_center, dtype=float) + U_reward @ theta
    elif args.sample_mode == "sparse_rare_ball":
        theta, chosen = sample_sparse_rare_theta(
            rng,
            U_reward.shape[1],
            ambient_radius,
            center_noise_frac=float(getattr(args, "center_noise_frac", 0.02)),
            rare_prob=float(getattr(args, "rare_prob", 0.25)),
            rare_amp_frac=float(getattr(args, "rare_amp_frac", 0.85)),
            max_active=int(getattr(args, "sparse_rare_max_active", 1)),
        )
        reward = np.asarray(problem.reward_center, dtype=float) + U_reward @ theta
    elif args.sample_mode == "factor_regime_mixture":
        theta, chosen = sample_factor_regime_theta(
            rng,
            U_reward.shape[1],
            ambient_radius,
            regime_count=int(getattr(args, "regime_count", max(4, min(U_reward.shape[1], 8)))),
            shift_frac=float(getattr(args, "regime_shift_frac", 0.60)),
            common_frac=float(getattr(args, "regime_common_frac", 0.12)),
            noise_frac=float(getattr(args, "regime_noise_frac", 0.18)),
            decay=float(getattr(args, "regime_decay", 0.90)),
        )
        reward = np.asarray(problem.reward_center, dtype=float) + U_reward @ theta
    elif args.sample_mode == "multiplicative_regime_mixture":
        theta, chosen = sample_factor_regime_theta(
            rng,
            U_reward.shape[1],
            ambient_radius,
            regime_count=int(getattr(args, "regime_count", max(4, min(U_reward.shape[1], 8)))),
            shift_frac=float(getattr(args, "regime_shift_frac", 0.60)),
            common_frac=float(getattr(args, "regime_common_frac", 0.10)),
            noise_frac=float(getattr(args, "regime_noise_frac", 0.16)),
            decay=float(getattr(args, "regime_decay", 0.92)),
        )
        latent = U_reward @ theta
        denom = float(max(np.max(np.abs(latent)), 1e-12))
        mult_log_scale = float(max(getattr(args, "mult_log_scale", 0.35), 1e-4))
        log_mult = np.clip((mult_log_scale / denom) * latent, -mult_log_scale, mult_log_scale)
        reward = np.asarray(problem.reward_center, dtype=float) * np.exp(log_mult)
        if float(getattr(args, "center_noise_frac", 0.0)) > 0.0:
            jitter = sample_ball_point(rng, U_reward.shape[1], float(args.center_noise_frac) * ambient_radius)
            reward = reward + 0.10 * (U_reward @ jitter)
    else:
        raise ValueError(f"Ambient beyond-prior sampler does not support sample_mode={args.sample_mode!r}")
    c_std = standard_cost_from_reward(reward, stdlp.n_slack)
    if np.linalg.norm(c_std) <= 1e-10:
        c_std = c0_std + 1e-4 * sample_ball_point(rng, stdlp.dim, max(1.0, ambient_radius))
        reward = reward_from_standard_cost(c_std, stdlp.n_slack)
    return {
        "reward": np.asarray(reward, dtype=float),
        "c_std": np.asarray(c_std, dtype=float),
        "theta": np.asarray(theta, dtype=float),
        "active_direction": int(chosen),
        "name": f"{problem.name}_{prefix}_{int(idx):05d}",
    }


def build_instance_from_record(problem: OriginalFixedXProblem, record: Dict[str, object]) -> object:
    inst = build_lp_instance(problem, np.asarray(record["reward"], dtype=float), name=str(record["name"]))
    inst.c_std = np.asarray(record["c_std"], dtype=float)
    inst.theta = np.asarray(record["theta"], dtype=float)
    inst.active_direction = int(record["active_direction"])
    _restore_full_solution_from_record(inst, record, prefix="orig")
    return inst


def _restore_full_solution_from_record(inst: object, record: Dict[str, object], prefix: str) -> None:
    x_key = f"{prefix}_full_x"
    obj_key = f"{prefix}_full_obj"
    time_key = f"{prefix}_full_time"
    if x_key in record and obj_key in record:
        inst.full_x = np.asarray(record[x_key], dtype=float).copy()
        inst.full_obj = float(record[obj_key])
        inst.full_time = float(record.get(time_key, 0.0))


def _store_full_solution_to_record(inst: object, record: Dict[str, object], prefix: str) -> None:
    full_obj = getattr(inst, "full_obj", None)
    full_x = getattr(inst, "full_x", None)
    if full_obj is None or full_x is None:
        return
    record[f"{prefix}_full_x"] = np.asarray(full_x, dtype=float).copy()
    record[f"{prefix}_full_obj"] = float(full_obj)
    record[f"{prefix}_full_time"] = float(getattr(inst, "full_time", 0.0))


def sample_ambient_instances(
    problem: OriginalFixedXProblem,
    args: argparse.Namespace,
    rng: np.random.Generator,
    total: int,
    ambient_radius_scale: float,
) -> tuple[list, np.ndarray, object, float]:
    stdlp, c0_std, U_reward, rho0, ambient_radius = ambient_sampling_context(problem, args, ambient_radius_scale)
    instances = []
    costs = []
    for idx in range(int(total)):
        rec = sample_ambient_record(problem, args, rng, stdlp, c0_std, U_reward, ambient_radius, idx, "ambient")
        inst = build_instance_from_record(problem, rec)
        instances.append(inst)
        costs.append(np.asarray(rec["c_std"], dtype=float))
    return instances, np.vstack(costs), stdlp, float(rho0)


def build_beyond_case_cache(
    case: fps.CaseSpec,
    base_args: argparse.Namespace,
    max_pilot_n0: int,
    ambient_radius_scale: float,
    accept_pool_size: int,
    test_pool_size: int,
) -> Dict[str, object]:
    case_mod = clone_case(case, harder_case_updates(case))
    args = fps.case_namespace(base_args, case_mod)
    rng_problem = np.random.default_rng(int(args.seed) + int(case.seed_offset))
    problem = fps.build_problem_for_case(case_mod, args, rng_problem)
    stdlp, c0_std, U_reward, rho0, ambient_radius = ambient_sampling_context(problem, args, ambient_radius_scale)

    rng_pilot = np.random.default_rng(int(args.seed) + int(case.seed_offset) + 100000)
    pilot_costs = []
    for idx in range(int(max_pilot_n0)):
        rec = sample_ambient_record(problem, args, rng_pilot, stdlp, c0_std, U_reward, ambient_radius, idx, "pilot")
        pilot_costs.append(np.asarray(rec["c_std"], dtype=float))

    rng_accept = np.random.default_rng(int(args.seed) + int(case.seed_offset) + 200000)
    accept_records = []
    accept_costs = []
    for idx in range(int(accept_pool_size)):
        rec = sample_ambient_record(problem, args, rng_accept, stdlp, c0_std, U_reward, ambient_radius, idx, "accept")
        accept_records.append(rec)
        accept_costs.append(np.asarray(rec["c_std"], dtype=float))

    test_records = []
    test_costs = []
    for idx in range(int(test_pool_size)):
        rec = sample_ambient_record(problem, args, rng_accept, stdlp, c0_std, U_reward, ambient_radius, idx, "test")
        test_records.append(rec)
        test_costs.append(np.asarray(rec["c_std"], dtype=float))
    return {
        "case": case,
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


def _take_inside_prefix(
    records: Sequence[Dict[str, object]],
    costs: np.ndarray,
    prior: EllipsoidCostPrior,
    need: int,
) -> tuple[list, np.ndarray]:
    chosen_records: List[Dict[str, object]] = []
    chosen_costs: List[np.ndarray] = []
    for rec, c_std in zip(records, costs):
        if inside_ellipsoid(c_std, prior):
            chosen_records.append(rec)
            chosen_costs.append(np.asarray(c_std, dtype=float))
            if len(chosen_records) >= int(need):
                break
    if len(chosen_records) < int(need):
        raise RuntimeError(f"Not enough accepted samples inside estimated ellipsoid; need={need}, got={len(chosen_records)}.")
    return chosen_records, np.vstack(chosen_costs)


def _collect_inside_records_with_extension(
    cache: Dict[str, object],
    prior: EllipsoidCostPrior,
    need: int,
    pilot_n0: int,
    rho_target: float,
    radius_scale: float,
    max_extra_draws: int = 50000,
) -> tuple[list, np.ndarray, Dict[str, int]]:
    records = list(cache["accept_records"])
    costs = np.asarray(cache["accept_costs"], dtype=float)
    chosen_records: List[Dict[str, object]] = []
    chosen_costs: List[np.ndarray] = []
    inside_existing = 0
    for rec, c_std in zip(records, costs):
        if inside_ellipsoid(c_std, prior):
            inside_existing += 1
            if len(chosen_records) < int(need):
                chosen_records.append(rec)
                chosen_costs.append(np.asarray(c_std, dtype=float))
    if len(chosen_records) >= int(need):
        return chosen_records, np.vstack(chosen_costs), {
            "inside_existing": int(inside_existing),
            "extra_draws": 0,
            "pool_size_final": int(len(records)),
        }

    case = cache["case"]
    args = cache["args"]
    problem = cache["problem"]
    ambient_radius_scale = float(cache["ambient_radius_scale"])
    stdlp, c0_std, U_reward, _rho0, ambient_radius = ambient_sampling_context(problem, args, ambient_radius_scale)
    del _rho0
    seed_token = (
        int(round(1000.0 * float(rho_target)))
        + 1009 * int(pilot_n0)
        + 917 * int(round(100.0 * float(radius_scale)))
    )
    rng_extra = np.random.default_rng(int(args.seed) + int(case.seed_offset) + 700000 + seed_token)
    start_idx = int(len(records))
    extra_records: List[Dict[str, object]] = []
    extra_costs: List[np.ndarray] = []
    extra_draws = 0
    while len(chosen_records) < int(need) and extra_draws < int(max_extra_draws):
        rec = sample_ambient_record(
            problem,
            args,
            rng_extra,
            stdlp,
            c0_std,
            U_reward,
            ambient_radius,
            start_idx + extra_draws,
            "accept_extra",
        )
        c_std = np.asarray(rec["c_std"], dtype=float)
        extra_records.append(rec)
        extra_costs.append(c_std)
        if inside_ellipsoid(c_std, prior):
            chosen_records.append(rec)
            chosen_costs.append(c_std)
        extra_draws += 1

    if extra_records:
        cache["accept_records"] = records + extra_records
        extra_costs_arr = np.vstack(extra_costs)
        cache["accept_costs"] = np.vstack([costs, extra_costs_arr])
        records = cache["accept_records"]
        costs = cache["accept_costs"]

    if len(chosen_records) < int(need):
        raise RuntimeError(
            "Not enough accepted samples inside estimated ellipsoid even after extension; "
            f"need={need}, got={len(chosen_records)}, extra_draws={extra_draws}."
        )

    inside_total = int(sum(int(inside_ellipsoid(c_std, prior)) for c_std in np.asarray(costs, dtype=float)))
    return chosen_records, np.vstack(chosen_costs), {
        "inside_existing": int(inside_existing),
        "extra_draws": int(extra_draws),
        "pool_size_final": int(len(records)),
        "inside_total": int(inside_total),
    }


def make_beyond_prior_data(
    root: Path,
    case: fps.CaseSpec,
    base_args: argparse.Namespace,
    rho_target: float,
    radius_scale: float,
    pilot_n0: int,
    ambient_radius_scale: float,
    test_n: int,
) -> tuple[fps.PreparedCaseData, Dict[str, object]]:
    case_mod = clone_case(case, harder_case_updates(case))
    args = fps.case_namespace(base_args, case_mod)
    args.k_list = str(int(case.sample_k))
    args.n_test = int(test_n)
    rng_problem = np.random.default_rng(int(args.seed) + int(case.seed_offset))
    problem = fps.build_problem_for_case(case_mod, args, rng_problem)

    rng_pilot = np.random.default_rng(int(args.seed) + int(case.seed_offset) + 100000)
    pilot_instances, pilot_costs, stdlp, rho0 = sample_ambient_instances(problem, args, rng_pilot, int(pilot_n0), float(ambient_radius_scale))
    del pilot_instances
    prior = estimate_ellipsoid_prior(
        pilot_costs,
        nominal_k=int(args.prior_nominal_k),
        target_outside_mass=float(rho_target),
        radius_scale=float(radius_scale),
        sample_radius_frac=float(args.sample_radius_frac),
    )

    n_accept = int(args.n_train + args.n_val)
    accepted_instances = []
    accepted_costs = []
    full_test_instances = []
    full_test_costs = []
    full_test_inside = []

    rng_accept = np.random.default_rng(int(args.seed) + int(case.seed_offset) + 200000)
    max_attempts = max(500, 80 * (n_accept + args.n_test))
    attempts = 0
    while len(accepted_instances) < n_accept and attempts < max_attempts:
        attempts += 1
        insts, costs, _stdlp, _ = sample_ambient_instances(problem, args, rng_accept, 1, float(ambient_radius_scale))
        inst = insts[0]
        c_std = costs[0]
        if inside_ellipsoid(c_std, prior):
            accepted_instances.append(inst)
            accepted_costs.append(c_std)
    if len(accepted_instances) < n_accept:
        raise RuntimeError(f"Could not collect enough accepted samples for case={case.slug}, rho={rho_target}.")

    while len(full_test_instances) < int(args.n_test) and attempts < max_attempts + 2000:
        attempts += 1
        insts, costs, _stdlp, _ = sample_ambient_instances(problem, args, rng_accept, 1, float(ambient_radius_scale))
        inst = insts[0]
        c_std = costs[0]
        full_test_instances.append(inst)
        full_test_costs.append(c_std)
        full_test_inside.append(int(inside_ellipsoid(c_std, prior)))
    if len(full_test_instances) < int(args.n_test):
        raise RuntimeError(f"Could not collect enough full test samples for case={case.slug}, rho={rho_target}.")

    train = accepted_instances[: int(args.n_train)]
    val = accepted_instances[int(args.n_train) :]
    test = full_test_instances
    bundle = FixedXBundle(
        train=train,
        val=val,
        test=test,
        Ctrain=np.vstack(accepted_costs[: int(args.n_train)]),
        Cval=np.vstack(accepted_costs[int(args.n_train) :]),
        Ctest=np.vstack(full_test_costs),
        stdlp=stdlp,
        prior=prior,
        truth={
            "prior_mode": "estimated_ellipsoid",
            "pilot_n0": int(pilot_n0),
            "rho_target": float(rho_target),
            "radius_scale": float(radius_scale),
            "ambient_radius_scale": float(ambient_radius_scale),
            "empirical_test_inside_rate": float(np.mean(full_test_inside)),
            "baseline_ball_rho": float(rho0),
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
        case=case,
        args=args,
        problem=problem,
        bundle=bundle,
        proj_train=proj_train,
        proj_val=proj_val,
        proj_test=proj_test,
    )
    diag = {
        "case": case.slug,
        "title": case.title,
        "pilot_n0": int(pilot_n0),
        "rho_target": float(rho_target),
        "radius_scale": float(radius_scale),
        "ambient_radius_scale": float(ambient_radius_scale),
        "accepted_train_val": int(len(accepted_instances)),
        "full_test": int(len(full_test_instances)),
        "empirical_test_inside_rate": float(np.mean(full_test_inside)),
        "estimated_rho": float(prior.rho),
        "baseline_ball_rho": float(rho0),
    }
    return data, diag


def make_beyond_prior_data_from_cache(
    cache: Dict[str, object],
    rho_target: float,
    radius_scale: float,
    pilot_n0: int,
    test_n: int,
) -> tuple[fps.PreparedCaseData, Dict[str, object]]:
    case = cache["case"]
    case_mod = cache["case_mod"]
    args = copy.deepcopy(cache["args"])
    args.k_list = str(int(case.sample_k))
    args.n_test = int(test_n)
    problem = cache["problem"]
    stdlp = cache["stdlp"]
    pilot_costs = np.asarray(cache["pilot_costs"], dtype=float)[: int(pilot_n0)]
    rho0 = float(cache["rho0"])
    prior = estimate_ellipsoid_prior(
        pilot_costs,
        nominal_k=int(args.prior_nominal_k),
        target_outside_mass=float(rho_target),
        radius_scale=float(radius_scale),
        sample_radius_frac=float(args.sample_radius_frac),
    )

    n_accept = int(args.n_train + args.n_val)
    accepted_records, accepted_costs, accept_stats = _collect_inside_records_with_extension(
        cache,
        prior,
        n_accept,
        pilot_n0=int(pilot_n0),
        rho_target=float(rho_target),
        radius_scale=float(radius_scale),
    )
    accepted_instances = [build_instance_from_record(problem, rec) for rec in accepted_records]
    full_test_records = list(cache["test_records"])[: int(test_n)]
    full_test_instances = [build_instance_from_record(problem, rec) for rec in full_test_records]
    full_test_costs = np.asarray(cache["test_costs"], dtype=float)[: int(test_n)]
    full_test_inside = [int(inside_ellipsoid(c_std, prior)) for c_std in full_test_costs]

    train = accepted_instances[: int(args.n_train)]
    val = accepted_instances[int(args.n_train) :]
    test = full_test_instances
    bundle = FixedXBundle(
        train=train,
        val=val,
        test=test,
        Ctrain=np.vstack(accepted_costs[: int(args.n_train)]),
        Cval=np.vstack(accepted_costs[int(args.n_train) :]),
        Ctest=np.vstack(full_test_costs),
        stdlp=stdlp,
        prior=prior,
        truth={
            "prior_mode": "estimated_ellipsoid",
            "pilot_n0": int(pilot_n0),
            "rho_target": float(rho_target),
            "radius_scale": float(radius_scale),
            "ambient_radius_scale": float(cache["ambient_radius_scale"]),
            "empirical_test_inside_rate": float(np.mean(full_test_inside)),
            "baseline_ball_rho": float(rho0),
        },
        problem=problem,
    )
    ensure_full_solutions(bundle.train)
    ensure_full_solutions(bundle.val)
    ensure_full_solutions(bundle.test)
    train_records = accepted_records[: int(args.n_train)]
    val_records = accepted_records[int(args.n_train) :]
    for rec, inst in zip(train_records, bundle.train):
        _store_full_solution_to_record(inst, rec, prefix="orig")
    for rec, inst in zip(val_records, bundle.val):
        _store_full_solution_to_record(inst, rec, prefix="orig")
    for rec, inst in zip(full_test_records, bundle.test):
        _store_full_solution_to_record(inst, rec, prefix="orig")
    proj_train = fps.bridge_instances_for_projection(problem, bundle.train)
    proj_val = fps.bridge_instances_for_projection(problem, bundle.val)
    proj_test = fps.bridge_instances_for_projection(problem, bundle.test)
    for rec, inst in zip(train_records, proj_train):
        _restore_full_solution_from_record(inst, rec, prefix="proj")
    for rec, inst in zip(val_records, proj_val):
        _restore_full_solution_from_record(inst, rec, prefix="proj")
    for rec, inst in zip(full_test_records, proj_test):
        _restore_full_solution_from_record(inst, rec, prefix="proj")
    ensure_full_solutions(proj_train)
    ensure_full_solutions(proj_val)
    ensure_full_solutions(proj_test)
    for rec, inst in zip(train_records, proj_train):
        _store_full_solution_to_record(inst, rec, prefix="proj")
    for rec, inst in zip(val_records, proj_val):
        _store_full_solution_to_record(inst, rec, prefix="proj")
    for rec, inst in zip(full_test_records, proj_test):
        _store_full_solution_to_record(inst, rec, prefix="proj")
    data = fps.PreparedCaseData(
        case=case_mod,
        args=args,
        problem=problem,
        bundle=bundle,
        proj_train=proj_train,
        proj_val=proj_val,
        proj_test=proj_test,
    )
    diag = {
        "case": case.slug,
        "title": case.title,
        "pilot_n0": int(pilot_n0),
        "rho_target": float(rho_target),
        "radius_scale": float(radius_scale),
        "accepted_train_val": int(len(accepted_instances)),
        "full_test": int(len(full_test_instances)),
        "empirical_test_inside_rate": float(np.mean(full_test_inside)),
        "estimated_rho": float(prior.rho),
        "baseline_ball_rho": float(rho0),
        "accept_inside_existing": int(accept_stats.get("inside_existing", 0)),
        "accept_extra_draws": int(accept_stats.get("extra_draws", 0)),
        "accept_pool_size_final": int(accept_stats.get("pool_size_final", len(cache["accept_records"]))),
        "accept_inside_total": int(accept_stats.get("inside_total", len(accepted_instances))),
    }
    return data, diag


def run_beyond_prior_probe(
    root: Path,
    case_slugs: Sequence[str],
    rho_list: Sequence[float],
    radius_scale_list: Sequence[float],
    out_dir: Path,
    pilot_n0: int,
    ambient_radius_scale: float,
    test_n: int,
    base_args: argparse.Namespace | None = None,
    ) -> None:
    if base_args is None:
        base_args = fps.default_namespace()
    out_dir = ensure_dir(out_dir)
    summary_frames: List[pd.DataFrame] = []
    diag_rows: List[Dict[str, object]] = []
    rank_rows: List[Dict[str, object]] = []
    quality_frames: List[pd.DataFrame] = []
    for slug in case_slugs:
        case = find_case(root, slug)
        max_pilot_n0 = int(max(int(pilot_n0), 1))
        base_args_case = base_args
        accept_pool = max(600, 80 * (int(base_args_case.n_train) + int(base_args_case.n_val)))
        test_pool = max(4 * int(test_n), 80)
        cache = build_beyond_case_cache(
            case=case,
            base_args=base_args_case,
            max_pilot_n0=max_pilot_n0,
            ambient_radius_scale=float(ambient_radius_scale),
            accept_pool_size=accept_pool,
            test_pool_size=test_pool,
        )
        for rho_target in rho_list:
            for radius_scale in radius_scale_list:
                data, diag = make_beyond_prior_data_from_cache(
                    cache=cache,
                    rho_target=float(rho_target),
                    radius_scale=float(radius_scale),
                    pilot_n0=int(pilot_n0),
                    test_n=int(test_n),
                )
                case_tag = f"{case.slug}_rho_{str(rho_target).replace('.', 'p')}_rs_{str(radius_scale).replace('.', 'p')}"
                result = fps.run_case_from_prepared(
                    data,
                    ensure_dir(out_dir / case_tag),
                    methods=TARGET_METHODS,
                    train_limit=len(data.bundle.train),
                    force=False,
                    write_rank=True,
                )
                summary = result["summary"].copy()
                summary = summary[summary["K"] == int(case.sample_k)].copy()
                summary["rho_target"] = float(rho_target)
                summary["radius_scale"] = float(radius_scale)
                summary["case"] = case.slug
                summary["title"] = case.title
                quality = result["quality"].copy()
                if not quality.empty:
                    quality = quality[quality["K"] == int(case.sample_k)].copy()
                    summary = summary.merge(
                        quality[["method", "K", "anchor_quality_mean", "anchor_quality_se"]],
                        on=["method", "K"],
                        how="left",
                    )
                    quality["rho_target"] = float(rho_target)
                    quality["radius_scale"] = float(radius_scale)
                    quality["case"] = case.slug
                    quality["title"] = case.title
                    quality_frames.append(quality)
                summary_frames.append(summary)
                diag_rows.append(diag)
                rank = result["rank"].copy()
                if not rank.empty:
                    rank["rho_target"] = float(rho_target)
                    rank["radius_scale"] = float(radius_scale)
                    rank_rows.extend(rank.to_dict("records"))

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    diag_df = pd.DataFrame(diag_rows)
    rank_df = pd.DataFrame(rank_rows)
    summary_df.to_csv(out_dir / "beyond_prior_summary.csv", index=False)
    diag_df.to_csv(out_dir / "beyond_prior_diagnostics.csv", index=False)
    rank_df.to_csv(out_dir / "beyond_prior_rank_growth.csv", index=False)
    quality_df = pd.concat(quality_frames, ignore_index=True) if quality_frames else pd.DataFrame()
    quality_df.to_csv(out_dir / "beyond_prior_quality.csv", index=False)
    plot_beyond_prior(summary_df, out_dir / "figure_beyond_prior_rho_sweep.png")
    plot_beyond_prior(
        summary_df,
        out_dir / "figure_beyond_prior_rho_sweep_quality.png",
        y_col="anchor_quality_mean",
        y_label="Average normalized improvement",
    )
    plot_beyond_prior_rank(rank_df, out_dir / "figure_beyond_prior_rank_growth.png")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cases": list(case_slugs),
                "rho_list": [float(x) for x in rho_list],
                "radius_scale_list": [float(x) for x in radius_scale_list],
                "pilot_n0": int(pilot_n0),
                "ambient_radius_scale": float(ambient_radius_scale),
                "test_n": int(test_n),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_beyond_prior_n0_probe(
    root: Path,
    case_slugs: Sequence[str],
    pilot_n0_list: Sequence[int],
    rho_target: float,
    out_dir: Path,
    ambient_radius_scale: float,
    test_n: int,
    radius_scale: float = 1.0,
    base_args: argparse.Namespace | None = None,
    ) -> None:
    if base_args is None:
        base_args = fps.default_namespace()
    out_dir = ensure_dir(out_dir)
    summary_frames: List[pd.DataFrame] = []
    rank_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []
    for slug in case_slugs:
        case = find_case(root, slug)
        pilot_n0_vals = [int(x) for x in pilot_n0_list]
        max_pilot_n0 = max(pilot_n0_vals) if pilot_n0_vals else 1
        accept_pool = max(600, 80 * (int(base_args.n_train) + int(base_args.n_val)))
        test_pool = max(4 * int(test_n), 80)
        cache = build_beyond_case_cache(
            case=case,
            base_args=base_args,
            max_pilot_n0=max_pilot_n0,
            ambient_radius_scale=float(ambient_radius_scale),
            accept_pool_size=accept_pool,
            test_pool_size=test_pool,
        )
        for pilot_n0 in pilot_n0_list:
            data, diag = make_beyond_prior_data_from_cache(
                cache=cache,
                rho_target=float(rho_target),
                radius_scale=float(radius_scale),
                pilot_n0=int(pilot_n0),
                test_n=int(test_n),
            )
            case_tag = f"{case.slug}_n0_{int(pilot_n0):04d}"
            result = fps.run_case_from_prepared(
                data,
                ensure_dir(out_dir / case_tag),
                methods=TARGET_METHODS,
                train_limit=len(data.bundle.train),
                force=False,
                write_rank=True,
            )
            summary = result["summary"].copy()
            summary = summary[summary["K"] == int(case.sample_k)].copy()
            quality = result["quality"].copy()
            if not quality.empty:
                quality = quality[quality["K"] == int(case.sample_k)].copy()
                summary = summary.merge(
                    quality[["method", "K", "anchor_quality_mean", "anchor_quality_se"]],
                    on=["method", "K"],
                    how="left",
                )
            summary["pilot_n0"] = int(pilot_n0)
            summary["rho_target"] = float(rho_target)
            summary["radius_scale"] = float(radius_scale)
            summary["case"] = case.slug
            summary["title"] = case.title
            summary_frames.append(summary)
            diag["pilot_n0"] = int(pilot_n0)
            diag_rows.append(diag)
            rank = result["rank"].copy()
            if not rank.empty:
                rank["pilot_n0"] = int(pilot_n0)
                rank["rho_target"] = float(rho_target)
                rank["radius_scale"] = float(radius_scale)
                rank_rows.extend(rank.to_dict("records"))

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    rank_df = pd.DataFrame(rank_rows)
    diag_df = pd.DataFrame(diag_rows)
    summary_df.to_csv(out_dir / "beyond_prior_n0_summary.csv", index=False)
    rank_df.to_csv(out_dir / "beyond_prior_n0_rank_growth.csv", index=False)
    diag_df.to_csv(out_dir / "beyond_prior_n0_diagnostics.csv", index=False)
    plot_beyond_prior_n0(
        summary_df,
        out_dir / "figure_beyond_prior_n0_sweep.png",
        y_col="objective_ratio_mean",
        y_label="Average test objective ratio",
    )
    plot_beyond_prior_n0(
        summary_df,
        out_dir / "figure_beyond_prior_n0_sweep_quality.png",
        y_col="anchor_quality_mean",
        y_label="Average normalized improvement",
    )
    plot_beyond_prior_n0_rank(rank_df, out_dir / "figure_beyond_prior_n0_rank_growth.png")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cases": list(case_slugs),
                "pilot_n0_list": [int(x) for x in pilot_n0_list],
                "rho_target": float(rho_target),
                "radius_scale": float(radius_scale),
                "ambient_radius_scale": float(ambient_radius_scale),
                "test_n": int(test_n),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def plot_harder_prior_rank(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    cases = list(dict.fromkeys(df["case"].tolist()))
    ncols = min(3, len(cases))
    nrows = int(math.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows), sharex=False, sharey=False)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#8c564b"]
    scale_list = sorted(df["prior_rho_scale"].unique())
    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        sub = df[df["case"] == case]
        for j, scale in enumerate(scale_list):
            cur = sub[sub["prior_rho_scale"] == scale].sort_values("sample")
            ax.plot(cur["sample"], cur["rank_after_sample"], marker="o", linewidth=2.0, color=colors[j % len(colors)], label=f"rho x {scale:g}")
        ax.set_title(sub["title"].iloc[0])
        ax.set_xlabel("# training samples")
        ax.set_ylabel("Learned dimension")
        ax.set_xlim(0, max(sub["sample"]))
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_harder_prior_compare(
    df: pd.DataFrame,
    out_path: Path,
    y_col: str = "objective_ratio_mean",
    y_label: str = "Average test objective ratio",
) -> None:
    if df.empty:
        return
    cases = list(dict.fromkeys(df["case"].tolist()))
    ncols = min(3, len(cases))
    nrows = int(math.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    lo, hi = -0.02, 1.05
    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        sub = df[df["case"] == case]
        methods = [m for m in TARGET_METHODS if m in set(sub["method"])]
        for method in methods:
            cur = sub[sub["method"] == method].sort_values("prior_rho_scale")
            style = fps.METHOD_STYLE.get(method, {"color": None, "marker": "o", "linestyle": "-"})
            ax.plot(cur["prior_rho_scale"], cur[y_col], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=2.0, markersize=5)
        ax.set_title(sub["title"].iloc[0])
        ax.set_xlabel("Prior radius scale")
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    for row in axes_arr:
        row[0].set_ylabel(y_label)
    handles = []
    labels = []
    for method in TARGET_METHODS:
        style = fps.METHOD_STYLE.get(method, {"color": None, "marker": "o", "linestyle": "-"})
        handles.append(axes_arr[0, 0].plot([], [], color=style["color"], marker=style["marker"], linestyle=style["linestyle"])[0])
        labels.append(fps.METHOD_DISPLAY.get(method, method))
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_beyond_prior(
    df: pd.DataFrame,
    out_path: Path,
    y_col: str = "objective_ratio_mean",
    y_label: str = "Average test objective ratio",
) -> None:
    if df.empty:
        return
    cases = list(dict.fromkeys(df["case"].tolist()))
    ncols = min(3, len(cases))
    nrows = int(math.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.8 * nrows), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    lo, hi = -0.02, 1.05
    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        sub = df[df["case"] == case]
        if len(set(sub["radius_scale"])) == 1:
            x_col = "rho_target"
            xlabel = "Target outside mass rho"
        else:
            x_col = "radius_scale"
            xlabel = "Ellipsoid radius scale"
        for method in [m for m in TARGET_METHODS if m in set(sub["method"])]:
            cur = sub[sub["method"] == method].sort_values(x_col)
            style = fps.METHOD_STYLE.get(method, {"color": None, "marker": "o", "linestyle": "-"})
            ax.plot(cur[x_col], cur[y_col], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=2.0, markersize=5)
        ax.set_title(sub["title"].iloc[0])
        ax.set_xlabel(xlabel)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    for row in axes_arr:
        row[0].set_ylabel(y_label)
    handles = []
    labels = []
    for method in TARGET_METHODS:
        style = fps.METHOD_STYLE.get(method, {"color": None, "marker": "o", "linestyle": "-"})
        handles.append(axes_arr[0, 0].plot([], [], color=style["color"], marker=style["marker"], linestyle=style["linestyle"])[0])
        labels.append(fps.METHOD_DISPLAY.get(method, method))
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_beyond_prior_rank(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    cases = list(dict.fromkeys(df["case"].tolist()))
    ncols = min(3, len(cases))
    nrows = int(math.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.8 * nrows), sharey=False)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        sub = df[df["case"] == case]
        combos = sub[["rho_target", "radius_scale"]].drop_duplicates().sort_values(["rho_target", "radius_scale"])
        for j, row in enumerate(combos.itertuples(index=False)):
            cur = sub[(sub["rho_target"] == float(row.rho_target)) & (sub["radius_scale"] == float(row.radius_scale))].sort_values("sample")
            label = f"rho={row.rho_target:g}, rs={row.radius_scale:g}"
            ax.plot(cur["sample"], cur["rank_after_sample"], color=colors[j % len(colors)], linewidth=2.0, marker="o", label=label)
        ax.set_title(sub["title"].iloc[0])
        ax.set_xlabel("# training samples")
        ax.set_ylabel("Learned dimension")
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_beyond_prior_n0(
    df: pd.DataFrame,
    out_path: Path,
    y_col: str,
    y_label: str,
) -> None:
    if df.empty:
        return
    cases = list(dict.fromkeys(df["case"].tolist()))
    ncols = min(3, len(cases))
    nrows = int(math.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.8 * nrows), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    lo, hi = -0.02, 1.05
    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        sub = df[df["case"] == case]
        for method in [m for m in TARGET_METHODS if m in set(sub["method"])]:
            cur = sub[sub["method"] == method].sort_values("pilot_n0")
            style = fps.METHOD_STYLE.get(method, {"color": None, "marker": "o", "linestyle": "-"})
            ax.plot(cur["pilot_n0"], cur[y_col], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=2.0, markersize=5)
        ax.set_title(sub["title"].iloc[0])
        ax.set_xlabel("Pilot samples $n_0$")
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    for row in axes_arr:
        row[0].set_ylabel(y_label)
    handles = []
    labels = []
    for method in TARGET_METHODS:
        style = fps.METHOD_STYLE.get(method, {"color": None, "marker": "o", "linestyle": "-"})
        handles.append(axes_arr[0, 0].plot([], [], color=style["color"], marker=style["marker"], linestyle=style["linestyle"])[0])
        labels.append(fps.METHOD_DISPLAY.get(method, method))
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_beyond_prior_n0_rank(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    cases = list(dict.fromkeys(df["case"].tolist()))
    ncols = min(3, len(cases))
    nrows = int(math.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.8 * nrows), sharey=False)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        sub = df[df["case"] == case]
        for pilot_n0 in sorted(sub["pilot_n0"].unique()):
            cur = sub[sub["pilot_n0"] == int(pilot_n0)].sort_values("sample")
            ax.plot(cur["sample"], cur["rank_after_sample"], linewidth=2.0, marker="o", label=f"$n_0$={int(pilot_n0)}")
        ax.set_title(sub["title"].iloc[0])
        ax.set_xlabel("# training samples")
        ax.set_ylabel("Learned dimension")
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_csv_list(text: str, cast=float) -> List:
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", type=str, default="harder", choices=["harder", "beyond", "beyond_n0", "both"])
    p.add_argument("--cases", type=str, default="packing,maxflow,random_lp_C")
    p.add_argument("--harder_scales", type=str, default="1.0,1.5,2.0,2.5")
    p.add_argument("--harder_sample_cases", type=str, default="")
    p.add_argument("--harder_sample_scales", type=str, default="1.0,1.5")
    p.add_argument("--rho_list", type=str, default="0.30,0.15,0.05")
    p.add_argument("--radius_scale_list", type=str, default="1.0")
    p.add_argument("--beyond_n0_list", type=str, default="40,120,240")
    p.add_argument("--beyond_n0_rho", type=float, default=0.15)
    p.add_argument("--beyond_n0_radius_scale", type=float, default=1.0)
    p.add_argument("--pilot_n0", type=int, default=120)
    p.add_argument("--ambient_radius_scale", type=float, default=1.35)
    p.add_argument("--beyond_n_test", type=int, default=80)
    p.add_argument("--out_dir", type=str, default="results_prior_stress_and_beyond")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(__file__).resolve().parent
    out_dir = ensure_dir((root / args.out_dir).resolve())
    case_slugs = [x.strip() for x in str(args.cases).split(",") if x.strip()]
    if args.mode in {"harder", "both"}:
        run_harder_prior_probe(
            root=root,
            case_slugs=case_slugs,
            scales=parse_csv_list(args.harder_scales, float),
            out_dir=ensure_dir(out_dir / "harder_prior"),
        )
        sample_case_slugs = [x.strip() for x in str(args.harder_sample_cases).split(",") if x.strip()]
        if sample_case_slugs:
            run_harder_prior_sample_efficiency(
                root=root,
                case_slugs=sample_case_slugs,
                scales=parse_csv_list(args.harder_sample_scales, float),
                out_dir=ensure_dir(out_dir / "harder_prior_sample_efficiency"),
            )
    if args.mode in {"beyond", "both"}:
        run_beyond_prior_probe(
            root=root,
            case_slugs=case_slugs,
            rho_list=parse_csv_list(args.rho_list, float),
            radius_scale_list=parse_csv_list(args.radius_scale_list, float),
            out_dir=ensure_dir(out_dir / "beyond_prior"),
            pilot_n0=int(args.pilot_n0),
            ambient_radius_scale=float(args.ambient_radius_scale),
            test_n=int(args.beyond_n_test),
        )
    if args.mode in {"beyond_n0", "both"}:
        run_beyond_prior_n0_probe(
            root=root,
            case_slugs=case_slugs,
            pilot_n0_list=parse_int_list(args.beyond_n0_list),
            rho_target=float(args.beyond_n0_rho),
            out_dir=ensure_dir(out_dir / "beyond_prior_n0"),
            ambient_radius_scale=float(args.ambient_radius_scale),
            test_n=int(args.beyond_n_test),
            radius_scale=float(args.beyond_n0_radius_scale),
        )
    print(f"Saved targeted prior experiments under: {out_dir}")


if __name__ == "__main__":
    main()
