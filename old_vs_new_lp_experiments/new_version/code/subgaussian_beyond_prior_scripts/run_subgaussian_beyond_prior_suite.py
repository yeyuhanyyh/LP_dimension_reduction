#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sub-Gaussian beyond-prior experiment using the 2026-04-24 old suite code.

This script intentionally imports the old snapshot implementation from
2026-04-24_snapshot_before_harder_prior/code.  It keeps the old LP geometry and
case sizes, replaces the old known-prior cost sampler by an additive Gaussian
cost distribution, estimates a high-mass ellipsoidal prior from an independent
pilot sample, then compares our estimated-prior reduction against the old
projection baselines trained directly on the original cost samples.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent
OLD_CODE = ROOT / "2026-04-24_snapshot_before_harder_prior" / "code"
sys.path.insert(0, str(OLD_CODE))

import final_projection_figure_suite as fps  # noqa: E402
import compare_fixedX_family_suite as cff  # noqa: E402
from compare_ours_exact_vs_pelp_fixedX import (  # noqa: E402
    OurStageIResult,
    append_direction,
    enumerate_edge_directions,
    recover_basis,
    solve_standard_form_min,
)
from iwata_sakaue_pelp_projection_compare import (  # noqa: E402
    ensure_full_solutions,
    LPInstance,
    objective_ratio,
    summarize_ratios_and_times,
)


METHOD_ORDER = ["Full", "OursEstC", "Rand", "PCA", "SGA", "CostOnly"]


@dataclasses.dataclass
class EllipsoidPrior:
    """Estimated prior C={mu + E y : ||y||_2 <= radius}."""

    mu: np.ndarray
    E: np.ndarray
    radius: float
    rho: float
    pilot_n: int
    effective_dim: int
    radius_scale: float
    retained_train: int
    pilot_maha_q: float


@dataclasses.dataclass
class SubgaussianPreparedCase:
    case: object
    args: argparse.Namespace
    problem: object
    bundle: object
    proj_train: List[object]
    proj_val: List[object]
    proj_test: List[object]
    pilot_costs: np.ndarray
    estimated_prior: EllipsoidPrior
    retained_train_indices: np.ndarray
    true_distribution: Dict[str, object]


def _orthonormalize(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=float)
    if arr.size == 0:
        return np.zeros((arr.shape[0], 0), dtype=float)
    q, _ = np.linalg.qr(arr, mode="reduced")
    return q


def _old_prior_scale(problem: object, args: argparse.Namespace, stdlp: object) -> Tuple[float, float, np.ndarray]:
    c0_std = cff.standard_cost_from_reward(problem.reward_center, stdlp.n_slack)
    nominal_k = cff.choose_nominal_prior_k(args)
    if np.all(np.asarray(problem.reward_center, dtype=float) > 1e-8):
        reward_margin_cap = float(args.reward_margin_frac) * float(np.min(problem.reward_center))
    else:
        reward_margin_cap = float("inf")
    rho, tau_sorted, _center_basis = cff.choose_prior_for_problem(
        stdlp,
        c0_std,
        nominal_k=nominal_k,
        origin_margin_frac=float(args.origin_margin_frac),
        reward_margin_cap=reward_margin_cap,
    )
    U_reward = _orthonormalize(problem.U_reward)
    prior_floor_frac = float(max(getattr(args, "prior_floor_frac", 0.0), 0.0))
    if U_reward.shape[1] > 0 and prior_floor_frac > 0.0:
        rho_floor = float(prior_floor_frac) * float(np.linalg.norm(c0_std))
        if np.isfinite(reward_margin_cap):
            rho_floor = min(rho_floor, float(reward_margin_cap))
        rho = max(float(rho), float(max(1e-6, rho_floor)))
    sample_radius = float(args.sample_radius_frac) * float(rho)
    return float(rho), float(sample_radius), np.asarray(tau_sorted, dtype=float)


def _sample_gaussian_costs(
    rng: np.random.Generator,
    c0_std: np.ndarray,
    U_reward: np.ndarray,
    n_slack: int,
    count: int,
    sample_radius: float,
    factor_scale_frac: float,
    factor_decay: float,
    gaussian_std_scale: float,
    cost_space: str = "reward_subspace",
) -> Tuple[np.ndarray, np.ndarray]:
    if str(cost_space) == "full_standard":
        U_cost = np.eye(int(np.asarray(c0_std).shape[0]), dtype=float)
        rank = int(U_cost.shape[1])
    else:
        U_reward = _orthonormalize(U_reward)
        rank = int(U_reward.shape[1])
        U_cost = np.vstack([np.zeros((int(n_slack), rank), dtype=float), -U_reward]) if rank > 0 else np.zeros((int(np.asarray(c0_std).shape[0]), 0), dtype=float)
    if rank <= 0:
        costs = np.repeat(np.asarray(c0_std, dtype=float).reshape(1, -1), int(count), axis=0)
        theta = np.zeros((int(count), 0), dtype=float)
        return costs, theta
    if str(cost_space) == "full_standard":
        scales = np.full(rank, float(gaussian_std_scale) * float(sample_radius) / math.sqrt(max(rank, 1)), dtype=float)
    else:
        scales = (
            float(gaussian_std_scale)
            * float(factor_scale_frac)
            * float(sample_radius)
            * np.power(float(factor_decay), np.arange(rank, dtype=float))
        )
    scales = np.maximum(scales, 1e-12)
    theta = rng.normal(loc=0.0, scale=scales.reshape(1, -1), size=(int(count), rank))
    costs = np.asarray(c0_std, dtype=float).reshape(1, -1) + theta @ U_cost.T
    return costs, theta


def _build_instances(problem: object, costs: np.ndarray, n_slack: int, prefix: str) -> List[object]:
    out: List[object] = []
    for idx, c_std in enumerate(np.asarray(costs, dtype=float)):
        reward = cff.reward_from_standard_cost(c_std, n_slack)
        inst = cff.build_lp_instance(problem, reward, name=f"{prefix}_{idx:05d}")
        inst.c_std = np.asarray(c_std, dtype=float).copy()
        out.append(inst)
    return out


def _build_standard_instances(stdlp: object, costs: np.ndarray, prefix: str) -> List[object]:
    out: List[object] = []
    dim = int(stdlp.dim)
    empty_A = np.zeros((0, dim), dtype=float)
    empty_b = np.zeros(0, dtype=float)
    for idx, c_std in enumerate(np.asarray(costs, dtype=float)):
        inst = LPInstance(
            c=-np.asarray(c_std, dtype=float).copy(),
            A=empty_A.copy(),
            b=empty_b.copy(),
            A_eq=np.asarray(stdlp.Aeq, dtype=float).copy(),
            b_eq=np.asarray(stdlp.b, dtype=float).copy(),
            name=f"{prefix}_{idx:05d}",
            var_bounds=[(0.0, None)] * dim,
        )
        inst.c_std = np.asarray(c_std, dtype=float).copy()
        out.append(inst)
    return out


def _estimate_ellipsoid_prior(
    pilot_costs: np.ndarray,
    rho: float,
    radius_scale: float,
    svd_tol: float,
) -> Tuple[EllipsoidPrior, np.ndarray]:
    pilot = np.asarray(pilot_costs, dtype=float)
    mu = pilot.mean(axis=0)
    centered = pilot - mu.reshape(1, -1)
    if centered.shape[0] <= 1 or np.linalg.norm(centered) <= 1e-14:
        E = np.zeros((pilot.shape[1], 0), dtype=float)
        radius = 0.0
        maha = np.zeros(pilot.shape[0], dtype=float)
        return EllipsoidPrior(mu, E, radius, float(rho), pilot.shape[0], 0, float(radius_scale), 0, 0.0), maha

    _, s, vt = np.linalg.svd(centered / math.sqrt(max(1, centered.shape[0] - 1)), full_matrices=False)
    keep = s > float(svd_tol) * max(float(s[0]), 1.0)
    if not np.any(keep):
        keep[0] = True
    E = vt[keep, :].T * s[keep].reshape(1, -1)
    z = centered @ vt[keep, :].T / s[keep].reshape(1, -1)
    maha = np.linalg.norm(z, axis=1)
    p_eff = int(E.shape[1])
    theo = math.sqrt(float(chi2.ppf(1.0 - float(rho), max(1, p_eff))))
    emp = float(np.quantile(maha, 1.0 - float(rho))) if maha.size else 0.0
    radius = float(radius_scale) * max(theo, emp)
    prior = EllipsoidPrior(
        mu=mu,
        E=E,
        radius=radius,
        rho=float(rho),
        pilot_n=int(pilot.shape[0]),
        effective_dim=p_eff,
        radius_scale=float(radius_scale),
        retained_train=0,
        pilot_maha_q=emp,
    )
    return prior, maha


def _mahalanobis_in_estimated_prior(costs: np.ndarray, prior: EllipsoidPrior) -> np.ndarray:
    if prior.E.shape[1] == 0:
        return np.zeros(np.asarray(costs).shape[0], dtype=float)
    # E = V diag(sigma), with orthonormal V inherited from the SVD.
    sig = np.linalg.norm(prior.E, axis=0)
    V = prior.E / np.maximum(sig.reshape(1, -1), 1e-12)
    z = (np.asarray(costs, dtype=float) - prior.mu.reshape(1, -1)) @ V / np.maximum(sig.reshape(1, -1), 1e-12)
    return np.linalg.norm(z, axis=1)


def prepare_subgaussian_case(
    case: object,
    base_args: argparse.Namespace,
    pilot_n: int,
    rho: float,
    radius_scale: float,
    gaussian_std_scale: float,
    svd_tol: float,
    same_est_train_samples: bool = False,
    enclose_train_in_estimated_prior: bool = True,
    cost_space: str = "reward_subspace",
) -> SubgaussianPreparedCase:
    args = fps.case_namespace(base_args, case)
    rng = np.random.default_rng(int(args.seed) + int(case.seed_offset))
    problem = fps.build_problem_for_case(case, args, rng)
    stdlp = cff.build_standard_form_problem(problem)
    c0_std = cff.standard_cost_from_reward(problem.reward_center, stdlp.n_slack)
    _old_rho, sample_radius, tau_sorted = _old_prior_scale(problem, args, stdlp)

    pilot_count = 0 if bool(same_est_train_samples) else int(pilot_n)
    total = int(pilot_count + args.n_train + args.n_val + args.n_test)
    factor_scale_frac = float(getattr(args, "factor_scale_frac", 0.70))
    factor_decay = float(getattr(args, "factor_decay", 0.82))
    costs, theta = _sample_gaussian_costs(
        rng,
        c0_std,
        problem.U_reward,
        stdlp.n_slack,
        total,
        sample_radius=sample_radius,
        factor_scale_frac=factor_scale_frac,
        factor_decay=factor_decay,
        gaussian_std_scale=gaussian_std_scale,
        cost_space=str(cost_space),
    )
    train_start = pilot_count
    train_end = train_start + int(args.n_train)
    val_end = train_end + int(args.n_val)
    train_costs = costs[train_start:train_end]
    val_costs = costs[train_end:val_end]
    test_costs = costs[val_end:]
    pilot_costs = train_costs.copy() if bool(same_est_train_samples) else costs[:pilot_count]

    prior, _pilot_maha = _estimate_ellipsoid_prior(pilot_costs, rho=rho, radius_scale=radius_scale, svd_tol=svd_tol)
    train_maha = _mahalanobis_in_estimated_prior(train_costs, prior)
    if bool(same_est_train_samples) and bool(enclose_train_in_estimated_prior) and train_maha.size > 0:
        prior.radius = max(float(prior.radius), float(np.max(train_maha)) * (1.0 + 1e-8))
        train_maha = _mahalanobis_in_estimated_prior(train_costs, prior)
    retained = np.flatnonzero(train_maha <= prior.radius + 1e-10)
    if retained.size == 0:
        retained = np.asarray([int(np.argmin(train_maha))], dtype=int)
    prior.retained_train = int(retained.size)

    if str(cost_space) == "full_standard":
        train = _build_standard_instances(stdlp, train_costs, f"{case.slug}_train")
        val = _build_standard_instances(stdlp, val_costs, f"{case.slug}_val")
        test = _build_standard_instances(stdlp, test_costs, f"{case.slug}_test")
    else:
        train = _build_instances(problem, train_costs, stdlp.n_slack, f"{case.slug}_train")
        val = _build_instances(problem, val_costs, stdlp.n_slack, f"{case.slug}_val")
        test = _build_instances(problem, test_costs, stdlp.n_slack, f"{case.slug}_test")
    ensure_full_solutions(train)
    ensure_full_solutions(val)
    ensure_full_solutions(test)

    if str(cost_space) == "full_standard":
        proj_train = train
        proj_val = val
        proj_test = test
    else:
        proj_train = fps.bridge_instances_for_projection(problem, train)
        proj_val = fps.bridge_instances_for_projection(problem, val)
        proj_test = fps.bridge_instances_for_projection(problem, test)
    ensure_full_solutions(proj_train)
    ensure_full_solutions(proj_val)
    ensure_full_solutions(proj_test)

    dummy_prior = cff.BallCostPrior(
        c0=prior.mu,
        rho=float(sample_radius),
        nominal_k=int(cff.choose_nominal_prior_k(args)),
        sample_radius=float(sample_radius),
        origin_margin=float(np.linalg.norm(prior.mu) - sample_radius),
        tau_sorted=tau_sorted,
    )
    truth = {
        "subgaussian_center": c0_std,
        "subgaussian_theta": theta,
        "subgaussian_sample_radius_reference": np.asarray([sample_radius], dtype=float),
        "subgaussian_factor_scale_frac": np.asarray([factor_scale_frac], dtype=float),
        "subgaussian_factor_decay": np.asarray([factor_decay], dtype=float),
        "subgaussian_gaussian_std_scale": np.asarray([gaussian_std_scale], dtype=float),
        "subgaussian_cost_space": np.asarray([str(cost_space)], dtype=object),
        "same_est_train_samples": np.asarray([bool(same_est_train_samples)]),
        "enclose_train_in_estimated_prior": np.asarray([bool(enclose_train_in_estimated_prior)]),
        "estimated_prior_mu": prior.mu,
        "estimated_prior_E": prior.E,
        "estimated_prior_radius": np.asarray([prior.radius], dtype=float),
        "estimated_prior_rho": np.asarray([prior.rho], dtype=float),
        "estimated_prior_effective_dim": np.asarray([prior.effective_dim], dtype=int),
        "retained_train_indices": retained,
    }
    bundle = cff.FixedXBundle(
        train=train,
        val=val,
        test=test,
        Ctrain=train_costs,
        Cval=val_costs,
        Ctest=test_costs,
        stdlp=stdlp,
        prior=dummy_prior,
        truth=truth,
        problem=problem,
    )
    return SubgaussianPreparedCase(
        case=case,
        args=args,
        problem=problem,
        bundle=bundle,
        proj_train=proj_train,
        proj_val=proj_val,
        proj_test=proj_test,
        pilot_costs=pilot_costs,
        estimated_prior=prior,
        retained_train_indices=retained,
        true_distribution={
            "sample_radius_reference": sample_radius,
            "factor_scale_frac": factor_scale_frac,
            "factor_decay": factor_decay,
            "gaussian_std_scale": gaussian_std_scale,
            "cost_space": str(cost_space),
        },
    )


def fi_min_ellipsoid(q: np.ndarray, D: np.ndarray, c_anchor: np.ndarray, prior: EllipsoidPrior, tol: float = 1e-9):
    q = np.asarray(q, dtype=float).reshape(-1)
    mu = np.asarray(prior.mu, dtype=float).reshape(-1)
    E = np.asarray(prior.E, dtype=float)
    p = int(E.shape[1])
    if p == 0 or float(prior.radius) <= tol:
        return float(q @ mu), mu.copy(), None

    if D.size == 0:
        A = np.zeros((0, p), dtype=float)
        b = np.zeros(0, dtype=float)
    else:
        Q, _ = np.linalg.qr(np.asarray(D, dtype=float), mode="reduced")
        A = Q.T @ E
        b = Q.T @ (np.asarray(c_anchor, dtype=float).reshape(-1) - mu)

    if A.shape[0] == 0:
        y0 = np.zeros(p, dtype=float)
        N = np.eye(p, dtype=float)
    else:
        y0 = np.linalg.pinv(A, rcond=1e-10) @ b
        _u, s, vt = np.linalg.svd(A, full_matrices=True)
        rank = int(np.sum(s > 1e-10 * max(float(s[0]) if s.size else 1.0, 1.0)))
        N = vt[rank:, :].T.copy()

    y0_norm = float(np.linalg.norm(y0))
    if y0_norm > float(prior.radius) + 1e-7:
        y0 = y0 * (float(prior.radius) / max(y0_norm, 1e-12))
        y0_norm = float(np.linalg.norm(y0))

    a = E.T @ q
    if N.size == 0 or N.shape[1] == 0:
        y = y0
    else:
        aN = N.T @ a
        aN_norm = float(np.linalg.norm(aN))
        rem = math.sqrt(max(float(prior.radius) ** 2 - y0_norm**2, 0.0))
        if aN_norm <= tol or rem <= tol:
            y = y0
        else:
            y = y0 - rem * (N @ (aN / aN_norm))
    c_out = mu + E @ y
    return float(q @ c_out), c_out, None


def alg1_pointwise_ellipsoid(
    lp: object,
    c_anchor: np.ndarray,
    prior: EllipsoidPrior,
    Dinit: Optional[np.ndarray] = None,
    fi_tol: float = 1e-8,
    indep_tol: float = 1e-8,
    max_iters: int = 300,
):
    d = lp.dim
    D = np.zeros((d, 0)) if Dinit is None else np.asarray(Dinit, dtype=float).copy()
    trace = {"rank_after_query": [], "mmin": [], "basis_fallback": [], "added": []}
    for _it in range(max_iters):
        x_opt, _ = solve_standard_form_min(lp, c_anchor)
        B, used_fallback = recover_basis(lp.Aeq, x_opt)
        Delta, _N = enumerate_edge_directions(lp.Aeq, B)
        mvals = np.zeros(Delta.shape[1])
        couts = np.zeros((d, Delta.shape[1]))
        for k in range(Delta.shape[1]):
            mvals[k], couts[:, k], _ = fi_min_ellipsoid(Delta[:, k], D, c_anchor, prior)
        j0 = int(np.argmin(mvals))
        mmin = float(mvals[j0])
        trace["mmin"].append(mmin)
        trace["basis_fallback"].append(bool(used_fallback))
        if mmin >= -fi_tol:
            return D, {"success": True, "basis": B, "x": x_opt, "message": "certified in estimated ellipsoid"}, trace

        c_out = couts[:, j0]
        cin_vals = Delta.T @ c_anchor
        cout_vals = Delta.T @ c_out
        viol = np.flatnonzero(cout_vals < -fi_tol)
        if len(viol) == 0:
            viol = np.asarray([j0], dtype=int)
        alpha = []
        for k in viol:
            denom = cin_vals[k] - cout_vals[k]
            alpha.append(max(0.0, cin_vals[k]) / max(denom, 1e-15))
        jstar = int(viol[int(np.argmin(alpha))])
        q_new = Delta[:, jstar]
        Dnew, added, _ = append_direction(D, q_new, indep_tol)
        if not added:
            return D, {"success": False, "basis": B, "x": x_opt, "message": "stopped: dependent ellipsoid direction"}, trace
        D = Dnew
        trace["rank_after_query"].append(D.shape[1])
        trace["added"].append(q_new)
    return D, {"success": False, "basis": None, "x": None, "message": "stopped: max_iters reached"}, trace


def alg2_cumulative_ellipsoid(
    lp: object,
    Ctrain: np.ndarray,
    prior: EllipsoidPrior,
    verbose: bool = False,
    fi_tol: float = 1e-8,
    indep_tol: float = 1e-8,
) -> OurStageIResult:
    D = np.zeros((lp.dim, 0), dtype=float)
    hard: List[int] = []
    rank_after_sample: List[int] = []
    rank_after_query: List[int] = []
    messages: List[str] = []
    for i, c in enumerate(np.asarray(Ctrain, dtype=float)):
        old_rank = D.shape[1]
        D, cert, tr = alg1_pointwise_ellipsoid(lp, c, prior, D, fi_tol=fi_tol, indep_tol=indep_tol)
        if D.shape[1] > old_rank:
            hard.append(i)
        rank_after_sample.append(D.shape[1])
        rank_after_query.extend(tr["rank_after_query"])
        messages.append(str(cert["message"]))
        if verbose:
            print(
                f"Ours EstC sample {i+1:3d}/{len(Ctrain):3d}: "
                f"rank {D.shape[1]:3d}, added {D.shape[1]-old_rank:2d}, {cert['message']}",
                flush=True,
            )
    U = np.zeros((lp.dim, 0)) if D.size == 0 else np.linalg.qr(D, mode="reduced")[0]
    x_anchor, _ = solve_standard_form_min(lp, np.asarray(Ctrain[0], dtype=float))
    return OurStageIResult(
        D=D,
        U=U,
        x_anchor=x_anchor,
        hard_indices=hard,
        rank_after_sample=rank_after_sample,
        rank_after_query=rank_after_query,
        messages=messages,
    )


def append_ours_estc_rows(rows: List[Dict[str, object]], data: SubgaussianPreparedCase, stage1: OurStageIResult, k_list: Sequence[int]) -> None:
    intrinsic_k = int(stage1.U.shape[1])
    for inst in data.bundle.test:
        c_std = np.asarray(inst.c_std, dtype=float)
        x_std, _obj_min, runtime, success, msg = cff.solve_ours_exact_on_instance(
            data.bundle.stdlp,
            c_std,
            stage1.U,
            stage1.x_anchor,
        )
        if success and x_std is not None:
            if np.asarray(inst.c).shape[0] == np.asarray(c_std).shape[0]:
                obj = float(-c_std @ x_std)
            else:
                obj = float(inst.c @ x_std[data.bundle.stdlp.n_slack :])
        else:
            obj = float("nan")
        for k in k_list:
            rows.append(
                {
                    "method": "OursEstC",
                    "K": int(k),
                    "intrinsic_K": intrinsic_k,
                    "instance": inst.name,
                    "objective": obj,
                    "full_objective": inst.full_obj,
                    "objective_ratio": objective_ratio(obj, inst.full_obj) if success else np.nan,
                    "time": runtime,
                    "success": float(success),
                    "message": msg,
                }
            )


def save_rank_curve(stage1: OurStageIResult, data: SubgaussianPreparedCase, out_dir: Path) -> pd.DataFrame:
    ranks = np.concatenate([[0], np.asarray(stage1.rank_after_sample, dtype=int)])
    df = pd.DataFrame({"sample": np.arange(len(ranks)), "rank_after_sample": ranks})
    df.to_csv(out_dir / "ours_rank_after_sample.csv", index=False)
    df["case"] = data.case.slug
    df["title"] = data.case.title
    df["family"] = data.case.family
    return df


def save_case_metadata(data: SubgaussianPreparedCase, out_dir: Path, train_limit: int, intrinsic_rank: int) -> None:
    meta = {
        "case": data.case.slug,
        "title": data.case.title,
        "family": data.case.family,
        "old_code_path": str(OLD_CODE),
        "n_pilot": int(data.estimated_prior.pilot_n),
        "same_est_train_samples": bool(np.asarray(data.bundle.truth.get("same_est_train_samples", [False])).reshape(-1)[0]),
        "enclose_train_in_estimated_prior": bool(np.asarray(data.bundle.truth.get("enclose_train_in_estimated_prior", [False])).reshape(-1)[0]),
        "n_train": int(train_limit),
        "n_train_retained_for_ours": int(data.estimated_prior.retained_train),
        "n_val": int(len(data.bundle.val)),
        "n_test": int(len(data.bundle.test)),
        "k_list": cff.parse_k_list(data.args.k_list),
        "rho_outside_mass": float(data.estimated_prior.rho),
        "estimated_prior_radius": float(data.estimated_prior.radius),
        "estimated_prior_effective_dim": int(data.estimated_prior.effective_dim),
        "estimated_prior_radius_scale": float(data.estimated_prior.radius_scale),
        "estimated_prior_pilot_maha_q": float(data.estimated_prior.pilot_maha_q),
        "intrinsic_rank_ours_estC": int(intrinsic_rank),
        "projection_family_dim": int(data.proj_train[0].n_vars),
        "original_family_dim": int(data.problem.n_vars),
        **data.true_distribution,
    }
    (out_dir / "case_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez(
        out_dir / "estimated_prior_and_distribution.npz",
        pilot_costs=data.pilot_costs,
        estimated_mu=data.estimated_prior.mu,
        estimated_E=data.estimated_prior.E,
        retained_train_indices=data.retained_train_indices,
        Ctrain=data.bundle.Ctrain,
        Cval=data.bundle.Cval,
        Ctest=data.bundle.Ctest,
    )


def run_case(
    case: object,
    root_out: Path,
    base_args: argparse.Namespace,
    methods: Sequence[str],
    pilot_n: int,
    rho: float,
    radius_scale: float,
    gaussian_std_scale: float,
    svd_tol: float,
    same_est_train_samples: bool,
    enclose_train_in_estimated_prior: bool,
    cost_space: str,
    force: bool,
) -> Dict[str, pd.DataFrame]:
    case_out = root_out / "k_sweep" / case.slug
    if not force and (case_out / "summary_results.csv").exists():
        return fps.load_case_result(case, case_out)
    case_out.mkdir(parents=True, exist_ok=True)

    data = prepare_subgaussian_case(
        case,
        base_args,
        pilot_n=pilot_n,
        rho=rho,
        radius_scale=radius_scale,
        gaussian_std_scale=gaussian_std_scale,
        svd_tol=svd_tol,
        same_est_train_samples=same_est_train_samples,
        enclose_train_in_estimated_prior=enclose_train_in_estimated_prior,
        cost_space=cost_space,
    )
    fps_data = fps.PreparedCaseData(
        case=data.case,
        args=data.args,
        problem=data.problem,
        bundle=data.bundle,
        proj_train=data.proj_train,
        proj_val=data.proj_val,
        proj_test=data.proj_test,
    )
    baseline_methods = [m for m in methods if m not in {"OursEstC"}]
    result = fps.run_case_from_prepared(
        fps_data,
        case_out,
        methods=baseline_methods,
        train_limit=len(data.bundle.train),
        force=True,
        write_rank=False,
    )
    raw = pd.read_csv(case_out / "raw_results.csv") if (case_out / "raw_results.csv").exists() else pd.DataFrame()
    fit_df = result.get("fit", pd.DataFrame())

    rank_df = pd.DataFrame()
    if "OursEstC" in set(methods):
        t0 = time.perf_counter()
        retained_costs = data.bundle.Ctrain[data.retained_train_indices]
        stage1 = alg2_cumulative_ellipsoid(
            data.bundle.stdlp,
            retained_costs,
            data.estimated_prior,
            verbose=bool(data.args.verbose),
        )
        fit_time = time.perf_counter() - t0
        rows = raw.to_dict("records") if not raw.empty else []
        append_ours_estc_rows(rows, data, stage1, cff.parse_k_list(data.args.k_list))
        raw = pd.DataFrame(rows)
        rank_df = save_rank_curve(stage1, data, case_out)
        np.savez(
            case_out / "ours_estC_stageI_learned_basis.npz",
            U=stage1.U,
            D=stage1.D,
            x_anchor=stage1.x_anchor,
            hard_indices=np.asarray(stage1.hard_indices, dtype=int),
            rank_after_sample=np.asarray(stage1.rank_after_sample, dtype=int),
            rank_after_query=np.asarray(stage1.rank_after_query, dtype=int),
        )
        fit_extra = pd.DataFrame(
            [
                {
                    "method": "OursEstC",
                    "K": int(stage1.U.shape[1]),
                    "fit_time": fit_time,
                    "train_samples": int(len(retained_costs)),
                    "val_samples": int(len(data.bundle.val)),
                    "test_samples": int(len(data.bundle.test)),
                    "epochs": 0,
                    "batch_size": 0,
                    "notes": f"Algorithm 2 with estimated ellipsoidal prior, rho={rho}",
                    "case": data.case.slug,
                    "title": data.case.title,
                    "family": data.case.family,
                }
            ]
        )
        fit_df = pd.concat([fit_df, fit_extra], ignore_index=True) if not fit_df.empty else fit_extra

    raw["method"] = raw["method"].str.replace(r"Rand#\d+", "Rand", regex=True)
    raw["case"] = data.case.slug
    raw["title"] = data.case.title
    raw["family"] = data.case.family
    raw.to_csv(case_out / "raw_results.csv", index=False)
    summary = summarize_ratios_and_times(raw.to_dict("records"))
    summary["case"] = data.case.slug
    summary["title"] = data.case.title
    summary["family"] = data.case.family
    summary.to_csv(case_out / "summary_results.csv", index=False)
    if not fit_df.empty:
        fit_df.to_csv(case_out / "fit_times.csv", index=False)
    save_case_metadata(data, case_out, train_limit=len(data.bundle.train), intrinsic_rank=int(raw.loc[raw["method"] == "OursEstC", "intrinsic_K"].max()) if "intrinsic_K" in raw and (raw["method"] == "OursEstC").any() else -1)
    return {"summary": summary, "quality": pd.DataFrame(), "rank": rank_df, "fit": fit_df}


def patch_plot_styles() -> None:
    fps.METHOD_STYLE["OursEstC"] = {"color": "#d62728", "marker": "D", "linestyle": (0, (6, 2))}
    fps.METHOD_DISPLAY["OursEstC"] = "Ours (est. C)"
    fps.METHOD_ORDER[:] = METHOD_ORDER


def split_cases(all_cases: Sequence[object], case_names: str) -> List[object]:
    if not case_names.strip() or case_names.strip().lower() in {"none", "skip", "empty"}:
        return []
    if case_names.strip().lower() == "all":
        return list(all_cases)
    wanted = {s.strip() for s in case_names.split(",") if s.strip()}
    case_map = {c.slug: c for c in all_cases}
    missing = sorted(wanted - set(case_map))
    if missing:
        raise ValueError(f"Unknown cases: {missing}")
    return [case_map[name] for name in [s.strip() for s in case_names.split(",") if s.strip()]]


def write_suite_manifest(
    out_dir: Path,
    synth_cases: Sequence[object],
    net_cases: Sequence[object],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "experiment": "subgaussian_beyond_prior_estimated_C",
        "old_code_path": str(OLD_CODE),
        "rho_outside_mass": float(args.rho),
        "pilot_n": int(args.n_train if args.same_est_train_samples else args.pilot_n),
        "requested_pilot_n": int(args.pilot_n),
        "n0_for_estimated_prior": int(args.n_train if args.same_est_train_samples else args.pilot_n),
        "same_est_train_samples": bool(args.same_est_train_samples),
        "enclose_train_in_estimated_prior": bool(args.enclose_train_in_estimated_prior),
        "radius_scale": float(args.radius_scale),
        "gaussian_std_scale": float(args.gaussian_std_scale),
        "cost_space": str(args.cost_space),
        "case_gaussian_std_scales": str(getattr(args, "case_gaussian_std_scales", "")),
        "methods": METHOD_ORDER,
        "synthetic_cases": [c.slug for c in synth_cases],
        "netlib_cases": [c.slug for c in net_cases],
        "k_list": str(args.k_list),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
        "seed": int(args.seed),
    }
    (out_dir / "subgaussian_beyond_prior_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_case_float_map(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected case=value in --case_gaussian_std_scales, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value.strip())
    return out


def run_suite(args: argparse.Namespace) -> None:
    patch_plot_styles()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_args = fps.default_namespace()
    base_args.seed = int(args.seed)
    base_args.device = str(args.device)
    base_args.verbose = bool(args.verbose)
    base_args.n_train = int(args.n_train)
    base_args.n_val = int(args.n_val)
    base_args.n_test = int(args.n_test)
    base_args.k_list = str(args.k_list)
    base_args.rand_trials = int(args.rand_trials)
    base_args.make_quality_figures = False

    synth_all = fps.synthetic_cases(ROOT)
    net_all = fps.netlib_cases(ROOT)
    synth_cases = split_cases(synth_all, args.synthetic_cases)
    net_cases = split_cases(net_all, args.netlib_cases)
    for c in [*synth_cases, *net_cases]:
        c.args_updates["k_list"] = str(args.k_list)
        if c.slug == "shortest_path" and str(args.shortest_path_design).strip():
            c.args_updates["shortest_design"] = str(args.shortest_path_design).strip()
            c.args_updates["grid_size"] = int(args.shortest_path_grid_size)
            c.args_updates["shortest_gadgets"] = int(args.shortest_path_gadgets)
            c.args_updates["sp_cost_low"] = float(args.shortest_path_cost_low)
            c.args_updates["sp_cost_high"] = float(args.shortest_path_cost_high)

    write_suite_manifest(out_dir, synth_cases, net_cases, args)
    title_map = {c.slug: c.title for c in [*synth_cases, *net_cases]}
    case_scale_map = parse_case_float_map(getattr(args, "case_gaussian_std_scales", ""))

    synth_results = []
    for case in synth_cases:
        print(f"[subgaussian beyond-prior | synthetic] {case.slug}", flush=True)
        synth_results.append(
            run_case(
                case,
                out_dir,
                base_args,
                methods=METHOD_ORDER,
                pilot_n=int(args.pilot_n),
                rho=float(args.rho),
                radius_scale=float(args.radius_scale),
                gaussian_std_scale=float(case_scale_map.get(case.slug, float(args.gaussian_std_scale))),
                svd_tol=float(args.svd_tol),
                same_est_train_samples=bool(args.same_est_train_samples),
                enclose_train_in_estimated_prior=bool(args.enclose_train_in_estimated_prior),
                cost_space=str(args.cost_space),
                force=bool(args.force),
            )
        )
    synth_summary = pd.concat([r["summary"] for r in synth_results if not r["summary"].empty], ignore_index=True) if synth_results else pd.DataFrame()
    synth_ranks = pd.concat([r["rank"] for r in synth_results if not r["rank"].empty], ignore_index=True) if synth_results and any(not r["rank"].empty for r in synth_results) else pd.DataFrame()
    if not synth_summary.empty:
        synth_summary.to_csv(out_dir / "k_sweep_synthetic_summary.csv", index=False)
        fps.plot_metric_grid(
            synth_summary,
            [c.slug for c in synth_cases],
            title_map,
            out_dir / "figure_k_sweep_synthetic.png",
            methods=METHOD_ORDER,
            y_col="objective_ratio_mean",
            yerr_col="objective_ratio_se",
            ylabel="Average test objective ratio",
        )
    if not synth_ranks.empty:
        synth_ranks.to_csv(out_dir / "ours_rank_growth_synthetic.csv", index=False)
        fps.plot_rank_grid(
            synth_ranks,
            [c.slug for c in synth_cases],
            title_map,
            out_dir / "figure_ours_rank_growth_synthetic.png",
            per_case_scale=True,
        )

    net_results = []
    for case in net_cases:
        print(f"[subgaussian beyond-prior | netlib] {case.slug}", flush=True)
        net_results.append(
            run_case(
                case,
                out_dir,
                base_args,
                methods=METHOD_ORDER,
                pilot_n=int(args.pilot_n),
                rho=float(args.rho),
                radius_scale=float(args.radius_scale),
                gaussian_std_scale=float(case_scale_map.get(case.slug, float(args.gaussian_std_scale))),
                svd_tol=float(args.svd_tol),
                same_est_train_samples=bool(args.same_est_train_samples),
                enclose_train_in_estimated_prior=bool(args.enclose_train_in_estimated_prior),
                cost_space=str(args.cost_space),
                force=bool(args.force),
            )
        )
    net_summary = pd.concat([r["summary"] for r in net_results if not r["summary"].empty], ignore_index=True) if net_results else pd.DataFrame()
    net_ranks = pd.concat([r["rank"] for r in net_results if not r["rank"].empty], ignore_index=True) if net_results and any(not r["rank"].empty for r in net_results) else pd.DataFrame()
    if not net_summary.empty:
        net_summary.to_csv(out_dir / "k_sweep_netlib_summary.csv", index=False)
        fps.plot_metric_grid(
            net_summary,
            [c.slug for c in net_cases],
            title_map,
            out_dir / "figure_k_sweep_netlib.png",
            methods=METHOD_ORDER,
            y_col="objective_ratio_mean",
            yerr_col="objective_ratio_se",
            ylabel="Average test objective ratio",
        )
    if not net_ranks.empty:
        net_ranks.to_csv(out_dir / "ours_rank_growth_netlib.csv", index=False)
        fps.plot_rank_grid(
            net_ranks,
            [c.slug for c in net_cases],
            title_map,
            out_dir / "figure_ours_rank_growth_netlib.png",
            per_case_scale=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out_dir", default="results_subgaussian_beyond_prior_rho010_oldcode_v1")
    p.add_argument("--synthetic_cases", default="all")
    p.add_argument("--netlib_cases", default="all")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--n_train", type=int, default=24)
    p.add_argument("--n_val", type=int, default=6)
    p.add_argument("--n_test", type=int, default=12)
    p.add_argument("--pilot_n", type=int, default=96)
    p.add_argument(
        "--same_est_train_samples",
        action="store_true",
        help="Use the same n_train costs to estimate C_hat_rho and train Algorithm 2; baselines also train on these n_train costs.",
    )
    p.add_argument(
        "--no_enclose_train_in_estimated_prior",
        dest="enclose_train_in_estimated_prior",
        action="store_false",
        help="Do not enlarge C_hat_rho to contain all same-sample training costs.",
    )
    p.set_defaults(enclose_train_in_estimated_prior=True)
    p.add_argument("--rho", type=float, default=0.10, help="Target outside mass for estimated prior.")
    p.add_argument("--radius_scale", type=float, default=1.10, help="Finite-sample inflation for the estimated ellipsoid.")
    p.add_argument("--gaussian_std_scale", type=float, default=0.35, help="Scale of Gaussian costs relative to old prior sample radius.")
    p.add_argument(
        "--cost_space",
        choices=["reward_subspace", "full_standard"],
        default="reward_subspace",
        help="Where Gaussian cost variation lives: old reward/original-cost subspace, or full standard-form cost space including slack coordinates.",
    )
    p.add_argument(
        "--case_gaussian_std_scales",
        default="",
        help="Optional comma list like random_lp_A=20,random_lp_B=20 overriding --gaussian_std_scale per case.",
    )
    p.add_argument("--shortest_path_design", default="", help="Optional shortest_path design override, e.g. corridor.")
    p.add_argument("--shortest_path_grid_size", type=int, default=16)
    p.add_argument("--shortest_path_gadgets", type=int, default=7)
    p.add_argument("--shortest_path_cost_low", type=float, default=10.0)
    p.add_argument("--shortest_path_cost_high", type=float, default=100.0)
    p.add_argument("--svd_tol", type=float, default=1e-10)
    p.add_argument("--k_list", default="5,10,20,30,40,50")
    p.add_argument("--rand_trials", type=int, default=3)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
