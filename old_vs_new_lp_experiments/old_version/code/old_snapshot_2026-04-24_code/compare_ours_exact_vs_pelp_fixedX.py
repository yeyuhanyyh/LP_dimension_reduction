#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare our exact decision-sufficient reduction with Iwata--Sakaue PELP_NN.

This script builds a fixed-feasible-region packing LP family

    max_y w^T y     s.t. T y <= b, y >= 0,

where only the objective vector w changes.  The same problem is lifted to
standard form

    min_x c^T x     s.t. [I,T] x = b, x >= 0,   x=(slack,y),

so Algorithm 1/2 from the decision-sufficient manuscript can learn a basis U of
W* in the standard-form variables.  Once U is learned, we solve the exact affine
reduced LP

    min_z (U^T c)^T z   s.t. x_anchor + U z >= 0,

and recover y from x=(slack,y).  This is compared against Full, Rand, PCA,
SharedP, FCNN, and PELP_NN projection methods on the original packing LP.

Quick run:
    python compare_ours_exact_vs_pelp_fixedX.py --quick --make_plots --verbose
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Install with: pip install torch\n" + str(exc))
from iwata_sakaue_pelp_projection_compare import (
    LPInstance,
    PELPProjectionNet,
    SharedProjection,
    FCNNProjectionNet,
    append_fixed_projection_rows,
    append_learned_projector_rows,
    ensure_full_solutions,
    pca_projection_from_training,
    plot_results,
    objective_ratio,
    random_column_projection,
    save_training_history,
    summarize_ratios_and_times,
    train_implicit_projection_model,
)


@dataclasses.dataclass
class StandardFormLP:
    """min c^T x s.t. Aeq x = b, x >= 0, with x=(slack,y)."""

    Aeq: np.ndarray
    b: np.ndarray
    n_slack: int
    n_y: int

    @property
    def dim(self) -> int:
        return int(self.Aeq.shape[1])

    @property
    def m(self) -> int:
        return int(self.Aeq.shape[0])


@dataclasses.dataclass
class BallCostPrior:
    """Full-ball prior C = {c : ||c-c0||_2 <= rho}, with 0 excluded by design."""

    c0: np.ndarray
    rho: float
    nominal_k: int
    sample_radius: float
    origin_margin: float
    tau_sorted: np.ndarray


@dataclasses.dataclass
class OurStageIResult:
    D: np.ndarray
    U: np.ndarray
    x_anchor: np.ndarray
    hard_indices: List[int]
    rank_after_sample: List[int]
    rank_after_query: List[int]
    messages: List[str]


def rand_unit_vector(rng: np.random.Generator, d: int) -> np.ndarray:
    u = rng.normal(size=int(d))
    nu = np.linalg.norm(u)
    if nu <= 1e-12:
        u = np.zeros(int(d), dtype=float)
        u[0] = 1.0
        return u
    return u / nu


def sample_ball_point(rng: np.random.Generator, d: int, radius: float) -> np.ndarray:
    d = int(d)
    radius = float(max(radius, 0.0))
    if d <= 0 or radius <= 0.0:
        return np.zeros(d, dtype=float)
    return rand_unit_vector(rng, d) * radius * (rng.random() ** (1.0 / d))


def choose_nominal_prior_k(args: argparse.Namespace) -> int:
    if int(args.prior_nominal_k) > 0:
        return int(args.prior_nominal_k)
    k_list = parse_k_list(args.k_list)
    if k_list:
        return int(max(k_list))
    return int(max(4, round(np.sqrt(max(1, args.n_train)))))


def choose_matlab_like_ball_radius(
    lp: StandardFormLP,
    c_center: np.ndarray,
    nominal_k: int,
    reward_margin_cap: float,
    origin_frac_cap: float = 0.45,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Choose rho by the same order-statistics idea used in the MATLAB demo.

    We solve the center LP, recover one optimal basis, compute all adjacent edge
    directions Delta, and define tau_j = reduced_cost_j / ||delta_j||.  The
    radius is then chosen between the k-th and (k+1)-th smallest tau values,
    clipped so that the origin stays outside the ball.
    """
    x_center, _ = solve_standard_form_min(lp, c_center)
    B0, _used_fallback = recover_basis(lp.Aeq, x_center)
    Delta, _N = enumerate_edge_directions(lp.Aeq, B0)
    if Delta.shape[1] == 0:
        rho_cap = min(float(origin_frac_cap) * float(np.linalg.norm(c_center)), float(reward_margin_cap))
        return max(1e-6, 0.25 * rho_cap), np.zeros(0, dtype=float), B0

    reduced_costs = np.maximum(Delta.T @ np.asarray(c_center, dtype=float), 0.0)
    delta_norms = np.maximum(np.linalg.norm(Delta, axis=0), 1e-12)
    tau = reduced_costs / delta_norms
    tau_sorted = np.sort(tau)

    rho_cap = min(float(origin_frac_cap) * float(np.linalg.norm(c_center)), float(reward_margin_cap))
    if rho_cap <= 1e-8:
        raise ValueError("Prior radius cap is nonpositive; the center cost is too close to the origin.")

    if len(tau_sorted) == 1:
        rho_candidate = 0.5 * float(tau_sorted[0])
    else:
        k_eff = min(max(int(nominal_k), 1), len(tau_sorted) - 1)
        lo = float(tau_sorted[k_eff - 1])
        hi = float(tau_sorted[k_eff])
        if hi > lo + 1e-12:
            rho_candidate = 0.5 * (lo + hi)
        elif lo > 1e-12:
            rho_candidate = 1.05 * lo
        else:
            positive = tau_sorted[tau_sorted > 1e-10]
            rho_candidate = 0.25 * float(positive[0]) if positive.size > 0 else 0.25 * rho_cap

    rho = min(rho_candidate, rho_cap)
    if rho <= 1e-10:
        positive = tau_sorted[tau_sorted > 1e-10]
        if positive.size > 0:
            rho = min(0.5 * float(positive[0]), rho_cap)
        else:
            rho = 0.25 * rho_cap
    return float(rho), tau_sorted, B0


def make_planted_fixed_packing_data(args: argparse.Namespace, rng: np.random.Generator, nominal_k: int):
    """Build fixed packing LPs and a MATLAB-style ball prior for stage I.

    Hidden truth:
        only a planted subset of variables changes across instances.

    Learned/test prior:
        a Euclidean ball C={c: ||c-c0||<=rho} centered at the standard-form cost
        center c0, with rho selected from reduced-cost order statistics as in
        the MATLAB benchmark.  The origin is kept strictly outside C.
    """
    m = int(args.n_cons)
    n = int(args.n_vars)
    dstar = min(int(args.dstar), n)
    total = int(args.n_train + args.n_val + args.n_test)

    T = rng.uniform(0.4, 1.2, size=(m, n))
    b = float(args.b_scale) * rng.uniform(0.8, 1.2, size=m)
    selected = np.sort(rng.choice(n, size=dstar, replace=False))
    col_scale = np.full(n, float(args.nonselected_column_scale))
    col_scale[selected] = 1.0
    T = T * col_scale.reshape(1, -1)

    Aeq = np.hstack([np.eye(m), T])
    stdlp = StandardFormLP(Aeq=Aeq, b=b, n_slack=m, n_y=n)

    c0 = np.zeros(m + n)
    reward0 = np.full(n, float(args.nonselected_profit))
    reward0[selected] = float(args.selected_profit_base)
    c0[m:] = -reward0

    Uc = np.zeros((m + n, dstar))
    for k, j in enumerate(selected):
        Uc[m + j, k] = -1.0

    reward_margin_cap = float(args.reward_margin_frac) * float(np.min(reward0[selected])) if dstar > 0 else float("inf")
    rho, tau_sorted, center_basis = choose_matlab_like_ball_radius(
        stdlp,
        c0,
        nominal_k=nominal_k,
        reward_margin_cap=reward_margin_cap,
        origin_frac_cap=float(args.origin_margin_frac),
    )
    sample_radius = float(args.sample_radius_frac) * rho
    prior = BallCostPrior(
        c0=c0,
        rho=float(rho),
        nominal_k=int(nominal_k),
        sample_radius=float(sample_radius),
        origin_margin=float(np.linalg.norm(c0) - rho),
        tau_sorted=tau_sorted,
    )

    instances: List[LPInstance] = []
    Cstd: List[np.ndarray] = []
    Theta: List[np.ndarray] = []
    for i in range(total):
        theta = sample_ball_point(rng, dstar, sample_radius)
        c_std = c0 + Uc @ theta
        if np.linalg.norm(c_std - c0) > rho + 1e-8:
            raise RuntimeError("sampled cost fell outside the declared prior ball")
        if np.linalg.norm(c_std) <= 1e-10:
            raise RuntimeError("sampled cost hit the origin, which is disallowed")
        w = -c_std[m:]
        inst = LPInstance(c=w, A=T, b=b, name=f"fixedX_{i:05d}")
        inst.c_std = c_std
        inst.theta = theta
        instances.append(inst)
        Cstd.append(c_std)
        Theta.append(theta)

    train = instances[: args.n_train]
    val = instances[args.n_train : args.n_train + args.n_val]
    test = instances[args.n_train + args.n_val :]
    Ctrain = np.vstack(Cstd[: args.n_train])
    Cval = np.vstack(Cstd[args.n_train : args.n_train + args.n_val])
    Ctest = np.vstack(Cstd[args.n_train + args.n_val :])
    truth = {
        "selected_variables": selected,
        "T": T,
        "b": b,
        "theta": np.vstack(Theta),
        "prior_U_cost": Uc,
        "prior_c0": c0,
        "prior_rho": np.asarray([rho], dtype=float),
        "prior_sample_radius": np.asarray([sample_radius], dtype=float),
        "prior_origin_margin": np.asarray([prior.origin_margin], dtype=float),
        "prior_nominal_k": np.asarray([nominal_k], dtype=int),
        "prior_tau_sorted": tau_sorted,
        "prior_center_basis": center_basis,
    }
    return train, val, test, Ctrain, Cval, Ctest, stdlp, prior, truth


def _linprog_min_options() -> Dict[str, object]:
    return {"presolve": True}


def solve_standard_form_min(lp: StandardFormLP, c: np.ndarray):
    res = linprog(
        c=np.asarray(c, dtype=float),
        A_eq=lp.Aeq,
        b_eq=lp.b,
        bounds=[(0.0, None)] * lp.dim,
        method="highs",
        options=_linprog_min_options(),
    )
    if not res.success:
        raise RuntimeError(f"standard-form LP solve failed: {res.message}")
    return np.asarray(res.x, dtype=float), float(res.fun)


def recover_basis(A: np.ndarray, x: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, bool]:
    """Recover an invertible basis from a BFS-like solver solution."""
    m, d = A.shape
    pos = np.flatnonzero(x > tol)
    if len(pos) == m and np.linalg.matrix_rank(A[:, pos], tol=1e-10) == m:
        return np.sort(pos), False

    order = np.argsort(-x)
    B: List[int] = []
    rank = 0
    for j in order:
        if int(j) in B:
            continue
        trial = B + [int(j)]
        new_rank = np.linalg.matrix_rank(A[:, trial], tol=1e-10)
        if new_rank > rank:
            B.append(int(j))
            rank = new_rank
            if rank == m:
                break
    if len(B) != m or np.linalg.matrix_rank(A[:, B], tol=1e-10) < m:
        raise RuntimeError("Could not recover an invertible basis; LP may be degenerate/ill-conditioned.")
    return np.array(sorted(B), dtype=int), True


def enumerate_edge_directions(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return standard-form edge directions delta(B,j) for all nonbasic j."""
    m, d = A.shape
    B = np.array(sorted(B), dtype=int)
    Bset = set(B.tolist())
    N = np.array([j for j in range(d) if j not in Bset], dtype=int)
    AB = A[:, B]
    AN = A[:, N]
    coeff = np.linalg.solve(AB, AN)
    Delta = np.zeros((d, len(N)))
    for k, j in enumerate(N):
        delta = np.zeros(d)
        delta[B] = -coeff[:, k]
        delta[j] = 1.0
        Delta[:, k] = delta
    return Delta, N


def append_direction(D: np.ndarray, q: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, bool, float]:
    q = np.asarray(q, dtype=float).reshape(-1)
    nq = np.linalg.norm(q)
    if nq <= 1e-14:
        return D, False, 0.0
    q = q / nq
    if D.size == 0:
        return q.reshape(-1, 1), True, 1.0
    Q, _ = np.linalg.qr(D, mode="reduced")
    r = q - Q @ (Q.T @ q)
    nr = np.linalg.norm(r)
    if nr <= tol:
        return D, False, nr
    Dnew = np.column_stack([Q, r / nr])
    return Dnew, True, nr


def fi_min_ball(q: np.ndarray, D: np.ndarray, c_anchor: np.ndarray, prior: BallCostPrior, tol: float = 1e-9):
    """FI(q; fiber) for the ball prior C={c: ||c-c0||<=rho}."""
    q = np.asarray(q, dtype=float).reshape(-1)
    if D.size == 0:
        Q = np.zeros((q.shape[0], 0), dtype=float)
    else:
        Q, _ = np.linalg.qr(np.asarray(D, dtype=float), mode="reduced")
    offset = np.asarray(c_anchor, dtype=float) - np.asarray(prior.c0, dtype=float)
    proj = Q @ (Q.T @ offset)
    center_fiber = np.asarray(prior.c0, dtype=float) + proj
    rad_sq = float(prior.rho) ** 2 - float(np.dot(proj, proj))
    rad_eff = np.sqrt(max(rad_sq, 0.0))
    q_perp = q - Q @ (Q.T @ q)
    nq = float(np.linalg.norm(q_perp))
    if nq <= tol:
        c_out = center_fiber
    else:
        c_out = center_fiber - (rad_eff / nq) * q_perp
    return float(q @ c_out), c_out, None


def alg1_pointwise(
    lp: StandardFormLP,
    c_anchor: np.ndarray,
    prior: BallCostPrior,
    Dinit: Optional[np.ndarray] = None,
    fi_tol: float = 1e-8,
    indep_tol: float = 1e-8,
    max_iters: int = 200,
):
    """Algorithm 1 with the common-witness facet-hit rule."""
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
            mvals[k], couts[:, k], _ = fi_min_ball(Delta[:, k], D, c_anchor, prior)
        j0 = int(np.argmin(mvals))
        mmin = float(mvals[j0])
        trace["mmin"].append(mmin)
        trace["basis_fallback"].append(bool(used_fallback))
        if mmin >= -fi_tol:
            return D, {"success": True, "basis": B, "x": x_opt, "message": "certified: fiber contained in an optimality cone"}, trace

        # Correct facet-hit rule: choose ONE witness c_out, then find first hit.
        c_out = couts[:, j0]
        cin_vals = Delta.T @ c_anchor
        cout_vals = Delta.T @ c_out
        viol = np.flatnonzero(cout_vals < -fi_tol)
        if len(viol) == 0:
            viol = np.array([j0], dtype=int)
        alpha = []
        for k in viol:
            denom = cin_vals[k] - cout_vals[k]
            alpha.append(max(0.0, cin_vals[k]) / max(denom, 1e-15))
        jstar = int(viol[int(np.argmin(alpha))])
        q_new = Delta[:, jstar]
        Dnew, added, _ = append_direction(D, q_new, indep_tol)
        if not added:
            return D, {"success": False, "basis": B, "x": x_opt, "message": "stopped: facet-hit direction numerically dependent"}, trace
        D = Dnew
        trace["rank_after_query"].append(D.shape[1])
        trace["added"].append(q_new)
    return D, {"success": False, "basis": None, "x": None, "message": "stopped: max_iters reached"}, trace


def alg2_cumulative(
    lp: StandardFormLP,
    Ctrain: np.ndarray,
    prior: BallCostPrior,
    verbose: bool = False,
    fi_tol: float = 1e-8,
    indep_tol: float = 1e-8,
) -> OurStageIResult:
    D = np.zeros((lp.dim, 0))
    hard: List[int] = []
    rank_after_sample: List[int] = []
    rank_after_query: List[int] = []
    messages: List[str] = []
    for i, c in enumerate(Ctrain):
        old_rank = D.shape[1]
        D, cert, tr = alg1_pointwise(lp, c, prior, D, fi_tol=fi_tol, indep_tol=indep_tol)
        if D.shape[1] > old_rank:
            hard.append(i)
        rank_after_sample.append(D.shape[1])
        rank_after_query.extend(tr["rank_after_query"])
        messages.append(str(cert["message"]))
        if verbose:
            print(f"Ours Alg2 sample {i+1:3d}/{len(Ctrain):3d}: rank {D.shape[1]:3d}, added {D.shape[1]-old_rank:2d}, {cert['message']}")

    U = np.zeros((lp.dim, 0)) if D.size == 0 else np.linalg.qr(D, mode="reduced")[0]
    x_anchor, _ = solve_standard_form_min(lp, Ctrain[0])
    return OurStageIResult(D=D, U=U, x_anchor=x_anchor, hard_indices=hard,
                           rank_after_sample=rank_after_sample,
                           rank_after_query=rank_after_query, messages=messages)


def solve_ours_exact_on_instance(lp: StandardFormLP, c_std: np.ndarray, U: np.ndarray, x_anchor: np.ndarray):
    """Solve our exact affine reduced LP in standard-form variables."""
    t0 = time.perf_counter()
    r = U.shape[1]
    if r == 0:
        x = x_anchor.copy()
        return x, float(c_std @ x), time.perf_counter() - t0, True, ""
    A_ub = -U
    b_ub = x_anchor
    A_eq = lp.Aeq @ U
    b_eq = lp.b - lp.Aeq @ x_anchor
    if np.linalg.norm(A_eq, ord="fro") <= 1e-8 * max(1.0, np.linalg.norm(U, ord="fro")):
        A_eq_solve = None
        b_eq_solve = None
    else:
        A_eq_solve = A_eq
        b_eq_solve = b_eq
    res = linprog(c=U.T @ c_std,
                  A_ub=A_ub,
                  b_ub=b_ub,
                  A_eq=A_eq_solve,
                  b_eq=b_eq_solve,
                  bounds=[(None, None)] * r,
                  method="highs",
                  options=_linprog_min_options())
    runtime = time.perf_counter() - t0
    if not res.success:
        return None, float("nan"), runtime, False, res.message
    x = x_anchor + U @ np.asarray(res.x, dtype=float)
    return x, float(c_std @ x), runtime, True, res.message


def append_ours_exact_rows(
    rows: List[Dict[str, object]],
    stage1: OurStageIResult,
    stdlp: StandardFormLP,
    instances: Sequence[LPInstance],
    comparison_ks: Optional[Sequence[int]] = None,
):
    intrinsic_k = int(stage1.U.shape[1])
    display_ks = [intrinsic_k] if comparison_ks is None else [int(k) for k in comparison_ks]
    for inst in instances:
        c_std = np.asarray(inst.c_std, dtype=float)
        x_std, _min_val, runtime, success, msg = solve_ours_exact_on_instance(stdlp, c_std, stage1.U, stage1.x_anchor)
        if success and x_std is not None:
            y = x_std[stdlp.n_slack:]
            obj = float(inst.c @ y)
        else:
            obj = float("nan")
        for k in display_ks:
            rows.append({
                "method": "OursExact",
                "K": k,
                "intrinsic_K": intrinsic_k,
                "instance": inst.name,
                "objective": obj,
                "full_objective": inst.full_obj,
                "objective_ratio": objective_ratio(obj, inst.full_obj) if success else np.nan,
                "time": runtime,
                "success": float(success),
                "message": msg,
            })


def plot_ours_rank(stage1: OurStageIResult, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, len(stage1.rank_after_sample) + 1), stage1.rank_after_sample, marker="o")
    ax.set_xlabel("# training samples processed")
    ax.set_ylabel("learned rank dim(span D)")
    ax.set_title("Our Algorithm 2: learned dimension vs samples")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ours_learned_dimension_vs_samples.png", dpi=200)
    plt.close(fig)

    if stage1.rank_after_query:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(np.arange(1, len(stage1.rank_after_query) + 1), stage1.rank_after_query, marker="o")
        ax.set_xlabel("# facet-hit queries")
        ax.set_ylabel("learned rank dim(span D)")
        ax.set_title("Our Algorithm 1/2: learned dimension vs facet-hit queries")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "ours_learned_dimension_vs_queries.png", dpi=200)
        plt.close(fig)


def parse_k_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out_dir", type=str, default="ours_exact_vs_pelp_results")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--quick", action="store_true", help="small setting for a fast smoke test")

    p.add_argument("--n_vars", type=int, default=60, help="number of original packing variables y")
    p.add_argument("--n_cons", type=int, default=16, help="number of packing constraints")
    p.add_argument("--dstar", type=int, default=8,
                   help="hidden truth rank used only for synthetic data generation")
    p.add_argument("--n_train", type=int, default=50)
    p.add_argument("--n_val", type=int, default=8)
    p.add_argument("--n_test", type=int, default=15)
    p.add_argument("--k_list", type=str, default="4,8,12")
    p.add_argument("--prior_nominal_k", type=int, default=0,
                   help="nominal local dimension for MATLAB-style prior-radius selection; 0 means auto=max(k_list)")
    p.add_argument("--sample_radius_frac", type=float, default=0.98,
                   help="sample costs inside sample_radius_frac * rho so train/val/test stay strictly inside the prior ball")
    p.add_argument("--origin_margin_frac", type=float, default=0.45,
                   help="enforce rho <= origin_margin_frac * ||c0|| so the prior ball excludes the origin")
    p.add_argument("--reward_margin_frac", type=float, default=0.8,
                   help="enforce rho <= reward_margin_frac * min(base selected profit) to keep rewards positive")
    p.add_argument("--selected_profit_base", type=float, default=0.9,
                   help="base profit shared by selected variables at the center cost c0")
    p.add_argument("--nonselected_profit", type=float, default=0.08,
                   help="small fixed profit on nonselected variables")
    p.add_argument("--nonselected_column_scale", type=float, default=3.0,
                   help="resource inflation factor on nonselected variables")
    p.add_argument("--b_scale", type=float, default=3.0)

    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--n_pelp_layers", type=int, default=4)
    p.add_argument("--generator_hidden_dim", type=int, default=32)

    p.add_argument("--run_sharedp", action="store_true", help="train SharedP baseline")
    p.add_argument("--run_fcnn", action="store_true", help="train FCNN baseline")
    p.add_argument("--skip_pelp", action="store_true", help="skip PELP_NN training")
    p.add_argument("--rand_trials", type=int, default=3)
    p.add_argument("--make_plots", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def apply_quick_settings(args: argparse.Namespace) -> None:
    args.n_vars = 40
    args.n_cons = 12
    args.dstar = 6
    args.n_train = 30
    args.n_val = 6
    args.n_test = 10
    args.k_list = "3,6,10"
    args.epochs = 8
    args.batch_size = 4
    args.patience = 8
    args.rand_trials = 3
    args.b_scale = 2.5


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.quick:
        apply_quick_settings(args)

    k_list = parse_k_list(args.k_list)
    nominal_k = choose_nominal_prior_k(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train, val, test, Ctrain, _Cval, _Ctest, stdlp, prior, truth = make_planted_fixed_packing_data(args, rng, nominal_k)
    np.savez(out_dir / "dataset_truth_and_prior.npz", **truth)
    print(
        f"Fixed-X planted packing: train={len(train)}, val={len(val)}, test={len(test)}, "
        f"M={args.n_cons}, N={args.n_vars}, hidden truth rank={args.dstar}, nominal prior K={nominal_k}"
    )
    print(f"Selected profitable variables: {truth['selected_variables'].tolist()}")
    print(
        f"Prior ball: rho={prior.rho:.4f}, sample_radius={prior.sample_radius:.4f}, "
        f"origin_margin=||c0||-rho={prior.origin_margin:.4f}"
    )
    print("Split meaning: train fits ours/PCA/neural baselines, val is only for early stopping, test is held out for final metrics.")

    print("Solving full packing LPs for train/val/test optima...")
    ensure_full_solutions(train)
    ensure_full_solutions(val)
    ensure_full_solutions(test)

    rows: List[Dict[str, object]] = []

    for k in k_list:
        for inst in test:
            rows.append({
                "method": "Full",
                "K": k,
                "instance": inst.name,
                "objective": inst.full_obj,
                "full_objective": inst.full_obj,
                "objective_ratio": 1.0,
                "time": inst.full_time,
                "success": 1.0,
            })

    print("\nRunning our Algorithm 1/2 stage-I learner...")
    t0 = time.perf_counter()
    stage1 = alg2_cumulative(stdlp, Ctrain, prior, verbose=args.verbose)
    stage1_time = time.perf_counter() - t0
    print(
        f"Ours stage-I finished: learned rank={stage1.U.shape[1]}, "
        f"hard samples={len(stage1.hard_indices)}, time={stage1_time:.3f}s"
    )
    append_ours_exact_rows(rows, stage1, stdlp, test, comparison_ks=k_list)
    np.savez(out_dir / "ours_stageI_learned_basis.npz", U=stage1.U, D=stage1.D,
             x_anchor=stage1.x_anchor, rank_after_sample=np.asarray(stage1.rank_after_sample),
             hard_indices=np.asarray(stage1.hard_indices, dtype=int))
    pd.DataFrame({"sample": np.arange(1, len(stage1.rank_after_sample) + 1),
                  "rank_after_sample": stage1.rank_after_sample}).to_csv(
        out_dir / "ours_rank_after_sample.csv", index=False)
    pd.DataFrame({"query": np.arange(1, len(stage1.rank_after_query) + 1),
                  "rank_after_query": stage1.rank_after_query}).to_csv(
        out_dir / "ours_rank_after_query.csv", index=False)
    if args.make_plots:
        plot_ours_rank(stage1, out_dir)

    for k in k_list:
        print(f"\n===== Projection baselines, K={k} =====")
        for tr in range(args.rand_trials):
            P_rand = random_column_projection(args.n_vars, k, rng)
            append_fixed_projection_rows(rows, f"Rand#{tr+1}", P_rand, test, k)

        print("Building PCA projection from the SAME training instances...")
        P_pca = pca_projection_from_training(train, k)
        append_fixed_projection_rows(rows, "PCA", P_pca, test, k)

        if args.run_sharedp:
            print("Training SharedP...")
            shared = SharedProjection(args.n_vars, k)
            shared, hist = train_implicit_projection_model(
                shared, train, val, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, device=device, seed=args.seed + 1000 + k,
                patience=args.patience, verbose=args.verbose)
            save_training_history(hist, out_dir / f"history_SharedP_K{k}.csv")
            append_learned_projector_rows(rows, "SharedP", shared, test, k, device)
            torch.save(shared.state_dict(), out_dir / f"SharedP_K{k}.pt")

        if args.run_fcnn:
            print("Training FCNN...")
            fcnn = FCNNProjectionNet(args.n_vars, args.n_cons, k, args.hidden_dim)
            fcnn, hist = train_implicit_projection_model(
                fcnn, train, val, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, device=device, seed=args.seed + 2000 + k,
                patience=args.patience, verbose=args.verbose)
            save_training_history(hist, out_dir / f"history_FCNN_K{k}.csv")
            append_learned_projector_rows(rows, "FCNN", fcnn, test, k, device)
            torch.save(fcnn.state_dict(), out_dir / f"FCNN_K{k}.pt")

        if not args.skip_pelp:
            print("Training PELP_NN...")
            pelp = PELPProjectionNet(k=k, hidden_dim=args.hidden_dim,
                                     n_layers=args.n_pelp_layers,
                                     generator_hidden_dim=args.generator_hidden_dim)
            pelp, hist = train_implicit_projection_model(
                pelp, train, val, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, device=device, seed=args.seed + 3000 + k,
                patience=args.patience, verbose=args.verbose)
            save_training_history(hist, out_dir / f"history_PELP_NN_K{k}.csv")
            append_learned_projector_rows(rows, "PELP_NN", pelp, test, k, device)
            torch.save(pelp.state_dict(), out_dir / f"PELP_NN_K{k}.pt")

    raw = pd.DataFrame(rows)
    raw_for_summary = raw.copy()
    raw_for_summary["method"] = raw_for_summary["method"].str.replace(r"Rand#\d+", "Rand", regex=True)
    summary = summarize_ratios_and_times(raw_for_summary)
    raw.to_csv(out_dir / "raw_results.csv", index=False)
    summary.to_csv(out_dir / "summary_results.csv", index=False)
    if args.make_plots:
        plot_results(summary, out_dir)

    print("\nSummary:")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 20, "display.width", 140):
        print(summary.sort_values(["K", "method"]))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
