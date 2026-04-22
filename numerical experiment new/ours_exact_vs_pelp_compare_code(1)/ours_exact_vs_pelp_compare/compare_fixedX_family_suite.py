#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified fixed-X, varying-c comparison across several LP families.

Every dataset in this script is constructed in the repeated-LP regime:

    maximize_z reward^T z
    subject to A_ineq z <= b_ineq, A_eq z = b_eq, 0 <= z <= ub,

where the feasible set X is fixed across train / validation / test instances,
and only the objective vector changes.  For our exact method we lift the same
family to standard form

    minimize_x c_std^T x
    subject to Aeq_std x = b_std, x >= 0,

with x = (slack, z).  The uncertainty set is an explicit Euclidean ball
C = { c_std : ||c_std - c0||_2 <= rho } with the origin excluded.

Compared methods:

- Full
- OursExact
- Rand, PCA, SharedP, FCNN, PELP_NN   (Iwata--Sakaue 2025 style)
- PCA2024, SGA_FinalP                 (Sakaue--Oki 2024 style)

The 2025-style baselines are evaluated on the original fixed-X LP family using
the same LPInstance representation throughout.  The 2024-style baselines are
implemented on an inequality-only reformulation with free reduced variables and
shared projection matrix P, including the final column projection heuristic.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, linprog, minimize

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Install with: pip install torch\n" + str(exc))

from compare_ours_exact_vs_pelp_fixedX import (
    BallCostPrior,
    StandardFormLP,
    alg2_cumulative,
    choose_matlab_like_ball_radius,
    enumerate_edge_directions,
    recover_basis,
    solve_standard_form_min,
    solve_ours_exact_on_instance,
)
from iwata_sakaue_pelp_projection_compare import (
    FCNNProjectionNet,
    LPInstance,
    PELPProjectionNet,
    SharedProjection,
    append_fixed_projection_rows,
    append_learned_projector_rows,
    ensure_full_solutions,
    make_monotone_grid_edges,
    make_node_arc_incidence,
    objective_ratio,
    pca_projection_from_training,
    plot_results,
    random_column_projection,
    reward_from_costs_via_potential,
    save_training_history,
    summarize_ratios_and_times,
    train_implicit_projection_model,
)


@dataclasses.dataclass
class OriginalFixedXProblem:
    name: str
    reward_center: np.ndarray
    A_ineq: np.ndarray
    b_ineq: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    ub: Optional[np.ndarray]
    U_reward: np.ndarray
    metadata: Dict[str, object]

    @property
    def n_vars(self) -> int:
        return int(self.reward_center.shape[0])


@dataclasses.dataclass
class FixedXBundle:
    train: List[LPInstance]
    val: List[LPInstance]
    test: List[LPInstance]
    Ctrain: np.ndarray
    Cval: np.ndarray
    Ctest: np.ndarray
    stdlp: StandardFormLP
    prior: BallCostPrior
    truth: Dict[str, object]
    problem: OriginalFixedXProblem


@dataclasses.dataclass
class GeneralProjectionInstance:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    name: str
    objective_constant: float = 0.0
    full_obj: Optional[float] = None

    @property
    def n_vars(self) -> int:
        return int(self.c.shape[0])


@dataclasses.dataclass
class GeneralProjectionSolve:
    x: Optional[np.ndarray]
    y: Optional[np.ndarray]
    obj: float
    success: bool
    runtime: float
    dual_for_max: Optional[np.ndarray]
    message: str = ""


def parse_k_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def rand_unit_vector(rng: np.random.Generator, d: int) -> np.ndarray:
    u = rng.normal(size=int(d))
    nu = np.linalg.norm(u)
    if nu <= 1e-12:
        out = np.zeros(int(d), dtype=float)
        if d > 0:
            out[0] = 1.0
        return out
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


def orthonormalize_columns(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=float)
    if arr.size == 0:
        return np.zeros((arr.shape[0], 0), dtype=float)
    q, _ = np.linalg.qr(arr, mode="reduced")
    return q


def make_random_cost_basis(n_vars: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    if rank <= 0 or rank >= n_vars:
        return np.eye(n_vars, dtype=float)
    raw = rng.normal(size=(n_vars, rank))
    return orthonormalize_columns(raw)


def evenly_spaced_indices(total: int, count: int) -> np.ndarray:
    total = int(total)
    count = int(min(max(count, 0), total))
    if count <= 0:
        return np.zeros(0, dtype=int)
    if count >= total:
        return np.arange(total, dtype=int)
    grid = np.linspace(0, total - 1, count)
    idx = np.unique(np.round(grid).astype(int))
    if len(idx) < count:
        extras = [j for j in range(total) if j not in set(idx.tolist())]
        idx = np.concatenate([idx, np.asarray(extras[: count - len(idx)], dtype=int)])
    return np.sort(idx[:count])


def add_internal_forward_edges(
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
    target_n_edges: int,
    rng: np.random.Generator,
    source: int,
    sink: int,
) -> List[Tuple[int, int]]:
    edge_set = set((int(u), int(v)) for u, v in edges)
    if len(edge_set) >= int(target_n_edges):
        return sorted(edge_set)
    candidates: List[Tuple[int, int]] = []
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            e = (u, v)
            if e in edge_set:
                continue
            if u == sink or v == source:
                continue
            if u == source or v == sink:
                continue
            candidates.append(e)
    need = int(target_n_edges) - len(edge_set)
    if need > len(candidates):
        raise ValueError("Not enough internal forward edges to reach the requested target.")
    if need > 0:
        chosen = rng.choice(len(candidates), size=need, replace=False)
        for idx in chosen:
            edge_set.add(candidates[int(idx)])
    return sorted(edge_set)


def block_partition(n_items: int, n_blocks: int) -> List[np.ndarray]:
    base = n_items // n_blocks
    rem = n_items % n_blocks
    out: List[np.ndarray] = []
    start = 0
    for b in range(n_blocks):
        size = base + (1 if b < rem else 0)
        out.append(np.arange(start, start + size, dtype=int))
        start += size
    return out


def sample_sparse_rare_theta(
    rng: np.random.Generator,
    rank: int,
    radius: float,
    center_noise_frac: float,
    rare_prob: float,
    rare_amp_frac: float,
    max_active: int,
) -> Tuple[np.ndarray, int]:
    rank = int(rank)
    radius = float(max(radius, 0.0))
    theta = np.zeros(rank, dtype=float)
    if rank <= 0 or radius <= 0.0:
        return theta, 0

    center_scale = float(max(center_noise_frac, 0.0)) * radius
    if center_scale > 0.0:
        theta = sample_ball_point(rng, rank, center_scale)

    active_tag = 0
    if rng.random() < float(max(0.0, rare_prob)):
        max_active = max(1, min(int(max_active), rank))
        n_active = int(rng.integers(1, max_active + 1))
        chosen = rng.choice(rank, size=n_active, replace=False)
        for pos, idx in enumerate(np.asarray(chosen, dtype=int)):
            sign = -1.0 if rng.random() < 0.5 else 1.0
            amp = float(rng.uniform(max(0.25, rare_amp_frac) * radius, radius))
            theta[idx] += sign * amp
            if pos == 0:
                active_tag = int(np.sign(sign)) * (int(idx) + 1)

    ntheta = float(np.linalg.norm(theta))
    if ntheta > radius + 1e-12:
        theta *= radius / max(ntheta, 1e-12)
    return theta, active_tag


def sample_factor_gaussian_theta(
    rng: np.random.Generator,
    rank: int,
    radius: float,
    scale_frac: float,
    decay: float,
) -> Tuple[np.ndarray, int]:
    rank = int(rank)
    radius = float(max(radius, 0.0))
    theta = np.zeros(rank, dtype=float)
    if rank <= 0 or radius <= 0.0:
        return theta, 0
    scale_frac = float(max(scale_frac, 0.0))
    decay = float(min(max(decay, 0.05), 1.0))
    scales = scale_frac * radius * np.power(decay, np.arange(rank, dtype=float))
    theta = rng.normal(loc=0.0, scale=np.maximum(scales, 1e-12), size=rank)
    ntheta = float(np.linalg.norm(theta))
    if ntheta > radius + 1e-12:
        theta *= radius / max(ntheta, 1e-12)
    if theta.size == 0:
        return theta, 0
    lead = int(np.argmax(np.abs(theta)))
    sign = 1 if theta[lead] >= 0.0 else -1
    return theta, sign * (lead + 1)


def clip_cost_to_local_ball(c_std: np.ndarray, c0_std: np.ndarray, sample_radius: float, rho: float) -> np.ndarray:
    c_std = np.asarray(c_std, dtype=float).reshape(-1)
    c0_std = np.asarray(c0_std, dtype=float).reshape(-1)
    diff = c_std - c0_std
    ndiff = float(np.linalg.norm(diff))
    if ndiff > sample_radius + 1e-12:
        c_std = c0_std + (float(sample_radius) / max(ndiff, 1e-12)) * diff
    if np.linalg.norm(c_std - c0_std) > rho + 1e-8:
        raise RuntimeError("Sampled cost fell outside the declared prior ball.")
    if np.linalg.norm(c_std) <= 1e-10:
        raise RuntimeError("Sampled cost hit the origin, which is not allowed.")
    return c_std


def reward_from_standard_cost(c_std: np.ndarray, n_slack: int) -> np.ndarray:
    c_std = np.asarray(c_std, dtype=float).reshape(-1)
    n_slack = int(n_slack)
    return -c_std[n_slack:]


def choose_independent_columns(mat: np.ndarray, target: int, tol: float = 1e-10) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    target = int(target)
    chosen: List[int] = []
    rank = 0
    for j in range(mat.shape[1]):
        trial = chosen + [j]
        new_rank = np.linalg.matrix_rank(mat[:, trial], tol=tol)
        if new_rank > rank:
            chosen.append(j)
            rank = new_rank
            if rank == target:
                break
    if len(chosen) != target:
        raise RuntimeError("Could not find the requested number of independent columns.")
    return np.asarray(chosen, dtype=int)


def stack_inequalities_with_upper_bounds(problem: OriginalFixedXProblem) -> Tuple[np.ndarray, np.ndarray]:
    n = problem.n_vars
    blocks: List[np.ndarray] = []
    rhs: List[np.ndarray] = []
    if problem.A_ineq.size > 0:
        blocks.append(np.asarray(problem.A_ineq, dtype=float))
        rhs.append(np.asarray(problem.b_ineq, dtype=float).reshape(-1))
    if problem.ub is not None:
        ub = np.asarray(problem.ub, dtype=float).reshape(-1)
        blocks.append(np.eye(n, dtype=float))
        rhs.append(ub)
    if not blocks:
        return np.zeros((0, n), dtype=float), np.zeros(0, dtype=float)
    return np.vstack(blocks), np.concatenate(rhs)


def reduce_to_independent_rows(A: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.size == 0 or A.shape[0] == 0:
        return np.zeros((0, A.shape[1] if A.ndim == 2 else 0), dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=int)
    keep: List[int] = []
    rank = 0
    for i in range(A.shape[0]):
        trial = keep + [i]
        new_rank = np.linalg.matrix_rank(A[trial, :], tol=tol)
        if new_rank > rank:
            keep.append(i)
            rank = new_rank
    idx = np.asarray(keep, dtype=int)
    return A[idx, :], b[idx], idx


def enumerate_center_directions(stdlp: StandardFormLP, c0_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_center, _ = solve_standard_form_min(stdlp, c0_std)
    B0, _used_fallback = recover_basis(stdlp.Aeq, x_center)
    Delta, N0 = enumerate_edge_directions(stdlp.Aeq, B0)
    return x_center, B0, Delta


def build_standard_form_problem(problem: OriginalFixedXProblem) -> StandardFormLP:
    A_ub, b_ub = stack_inequalities_with_upper_bounds(problem)
    q = int(A_ub.shape[0])
    n = problem.n_vars
    blocks: List[np.ndarray] = []
    rhs: List[np.ndarray] = []
    if q > 0:
        blocks.append(np.hstack([np.eye(q, dtype=float), A_ub]))
        rhs.append(np.asarray(b_ub, dtype=float).reshape(-1))
    if problem.A_eq.size > 0:
        Aeq_red, beq_red, _keep_eq = reduce_to_independent_rows(problem.A_eq, problem.b_eq)
        blocks.append(np.hstack([np.zeros((Aeq_red.shape[0], q), dtype=float), Aeq_red]))
        rhs.append(beq_red)
    if not blocks:
        raise ValueError("The feasible set is unbounded without inequalities or equalities; cannot build a bounded standard-form LP.")
    Aeq = np.vstack(blocks)
    b = np.concatenate(rhs)
    return StandardFormLP(Aeq=Aeq, b=b, n_slack=q, n_y=n)


def standard_cost_from_reward(reward: np.ndarray, n_slack: int) -> np.ndarray:
    reward = np.asarray(reward, dtype=float).reshape(-1)
    return np.concatenate([np.zeros(int(n_slack), dtype=float), -reward])


def build_lp_instance(problem: OriginalFixedXProblem, reward: np.ndarray, name: str) -> LPInstance:
    A_ub, b_ub = stack_inequalities_with_upper_bounds(problem)
    A_eq = None if problem.A_eq.size == 0 else np.asarray(problem.A_eq, dtype=float)
    b_eq = None if problem.A_eq.size == 0 else np.asarray(problem.b_eq, dtype=float).reshape(-1)
    return LPInstance(
        c=np.asarray(reward, dtype=float).reshape(-1),
        A=A_ub,
        b=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        name=name,
    )


def choose_prior_for_problem(
    stdlp: StandardFormLP,
    c0_std: np.ndarray,
    nominal_k: int,
    origin_margin_frac: float,
    reward_margin_cap: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    try:
        return choose_matlab_like_ball_radius(
            stdlp,
            c0_std,
            nominal_k=nominal_k,
            reward_margin_cap=reward_margin_cap,
            origin_frac_cap=origin_margin_frac,
        )
    except Exception:
        rho_cap = min(float(origin_margin_frac) * float(np.linalg.norm(c0_std)), float(reward_margin_cap))
        rho = max(1e-6, 0.25 * rho_cap)
        return float(rho), np.zeros(0, dtype=float), np.zeros(0, dtype=int)


def make_fixed_x_bundle(problem: OriginalFixedXProblem, args: argparse.Namespace, rng: np.random.Generator) -> FixedXBundle:
    stdlp = build_standard_form_problem(problem)
    c0_std = standard_cost_from_reward(problem.reward_center, stdlp.n_slack)
    nominal_k = choose_nominal_prior_k(args)
    if np.all(problem.reward_center > 1e-8):
        reward_margin_cap = float(args.reward_margin_frac) * float(np.min(problem.reward_center))
    else:
        reward_margin_cap = float("inf")
    rho, tau_sorted, center_basis = choose_prior_for_problem(
        stdlp,
        c0_std,
        nominal_k=nominal_k,
        origin_margin_frac=float(args.origin_margin_frac),
        reward_margin_cap=reward_margin_cap,
    )
    U_reward = orthonormalize_columns(problem.U_reward)
    if args.sample_mode == "sparse_rare_ball" and U_reward.shape[1] > 0:
        rho_floor = 0.10 * float(np.linalg.norm(c0_std))
        if np.isfinite(reward_margin_cap):
            rho_floor = min(rho_floor, float(reward_margin_cap))
        rho = max(float(rho), float(max(1e-6, rho_floor)))
    sample_radius = float(args.sample_radius_frac) * float(rho)
    prior = BallCostPrior(
        c0=c0_std,
        rho=float(rho),
        nominal_k=int(nominal_k),
        sample_radius=float(sample_radius),
        origin_margin=float(np.linalg.norm(c0_std) - rho),
        tau_sorted=np.asarray(tau_sorted, dtype=float),
    )

    if U_reward.shape[0] not in {0, problem.n_vars}:
        raise ValueError("U_reward has the wrong ambient dimension.")
    U_cost = (
        np.zeros((stdlp.dim, 0), dtype=float)
        if U_reward.shape[1] == 0
        else np.vstack([np.zeros((stdlp.n_slack, U_reward.shape[1]), dtype=float), -U_reward])
    )

    total = int(args.n_train + args.n_val + args.n_test)
    instances: List[LPInstance] = []
    Cstd: List[np.ndarray] = []
    theta_all: List[np.ndarray] = []
    active_idx = np.zeros(total, dtype=int)
    if args.sample_mode == "uniform_ball":
        if U_reward.shape[0] != problem.n_vars:
            raise ValueError("uniform_ball sampling requires an explicit reward subspace U_reward.")
        for idx in range(total):
            theta = sample_ball_point(rng, U_reward.shape[1], sample_radius)
            reward = np.asarray(problem.reward_center, dtype=float) + U_reward @ theta
            c_std = standard_cost_from_reward(reward, stdlp.n_slack)
            if np.linalg.norm(c_std - c0_std) > rho + 1e-8:
                raise RuntimeError("Sampled cost fell outside the declared prior ball.")
            if np.linalg.norm(c_std) <= 1e-10:
                raise RuntimeError("Sampled cost hit the origin, which is not allowed.")
            inst = build_lp_instance(problem, reward, name=f"{problem.name}_{idx:05d}")
            inst.c_std = c_std
            inst.theta = theta
            instances.append(inst)
            Cstd.append(c_std)
            theta_all.append(theta)
    elif args.sample_mode == "factor_gaussian":
        if U_reward.shape[0] != problem.n_vars or U_reward.shape[1] == 0:
            raise ValueError("factor_gaussian sampling requires a nontrivial explicit reward subspace U_reward.")
        for idx in range(total):
            theta, chosen = sample_factor_gaussian_theta(
                rng,
                U_reward.shape[1],
                sample_radius,
                scale_frac=float(getattr(args, "factor_scale_frac", 0.55)),
                decay=float(getattr(args, "factor_decay", 0.82)),
            )
            reward = np.asarray(problem.reward_center, dtype=float) + U_reward @ theta
            c_std = clip_cost_to_local_ball(standard_cost_from_reward(reward, stdlp.n_slack), c0_std, sample_radius, rho)
            reward = reward_from_standard_cost(c_std, stdlp.n_slack)
            inst = build_lp_instance(problem, reward, name=f"{problem.name}_{idx:05d}")
            inst.c_std = c_std
            inst.theta = theta
            inst.active_direction = chosen
            instances.append(inst)
            Cstd.append(c_std)
            theta_all.append(theta)
            active_idx[idx] = chosen
    elif args.sample_mode == "sparse_rare_ball":
        if U_reward.shape[0] != problem.n_vars or U_reward.shape[1] == 0:
            raise ValueError("sparse_rare_ball sampling requires a nontrivial explicit reward subspace U_reward.")
        for idx in range(total):
            theta, chosen = sample_sparse_rare_theta(
                rng,
                U_reward.shape[1],
                sample_radius,
                center_noise_frac=float(args.center_noise_frac),
                rare_prob=float(args.rare_prob),
                rare_amp_frac=float(args.rare_amp_frac),
                max_active=int(args.sparse_rare_max_active),
            )
            reward = np.asarray(problem.reward_center, dtype=float) + U_reward @ theta
            c_std = standard_cost_from_reward(reward, stdlp.n_slack)
            if np.linalg.norm(c_std - c0_std) > rho + 1e-8:
                raise RuntimeError("Sampled cost fell outside the declared prior ball.")
            if np.linalg.norm(c_std) <= 1e-10:
                raise RuntimeError("Sampled cost hit the origin, which is not allowed.")
            inst = build_lp_instance(problem, reward, name=f"{problem.name}_{idx:05d}")
            inst.c_std = c_std
            inst.theta = theta
            inst.active_direction = chosen
            instances.append(inst)
            Cstd.append(c_std)
            theta_all.append(theta)
            active_idx[idx] = chosen
    elif args.sample_mode == "multiplicative_factor":
        if U_reward.shape[0] != problem.n_vars or U_reward.shape[1] == 0:
            raise ValueError("multiplicative_factor sampling requires a nontrivial explicit reward subspace U_reward.")
        mult_log_scale = float(max(getattr(args, "mult_log_scale", 0.35), 1e-4))
        for idx in range(total):
            theta, chosen = sample_factor_gaussian_theta(
                rng,
                U_reward.shape[1],
                sample_radius,
                scale_frac=float(getattr(args, "factor_scale_frac", 0.50)),
                decay=float(getattr(args, "factor_decay", 0.88)),
            )
            latent = U_reward @ theta
            denom = float(max(np.max(np.abs(latent)), 1e-12))
            log_mult = np.clip((mult_log_scale / denom) * latent, -mult_log_scale, mult_log_scale)
            reward = np.asarray(problem.reward_center, dtype=float) * np.exp(log_mult)
            if float(args.center_noise_frac) > 0.0:
                jitter = sample_ball_point(rng, U_reward.shape[1], float(args.center_noise_frac) * sample_radius)
                reward = reward + 0.10 * (U_reward @ jitter)
            c_std = clip_cost_to_local_ball(standard_cost_from_reward(reward, stdlp.n_slack), c0_std, sample_radius, rho)
            reward = reward_from_standard_cost(c_std, stdlp.n_slack)
            inst = build_lp_instance(problem, reward, name=f"{problem.name}_{idx:05d}")
            inst.c_std = c_std
            inst.theta = theta
            inst.active_direction = chosen
            instances.append(inst)
            Cstd.append(c_std)
            theta_all.append(theta)
            active_idx[idx] = chosen
    elif args.sample_mode == "iid_local_rare":
        if stdlp.n_slack != 0:
            raise ValueError("iid_local_rare sampling is currently implemented only for native standard-form families.")
        _x_center, _B0, Delta = enumerate_center_directions(stdlp, c0_std)
        reduced_costs = np.maximum(Delta.T @ np.asarray(c0_std, dtype=float), 0.0)
        delta_norms = np.maximum(np.linalg.norm(Delta, axis=0), 1e-12)
        tau = reduced_costs / delta_norms
        Slocal = np.flatnonzero(tau <= rho + 1e-12)
        if len(Slocal) == 0:
            raise RuntimeError("iid_local_rare sampling needs at least one locally relevant direction inside the prior ball.")
        for idx in range(total):
            c_std = np.asarray(c0_std, dtype=float) + float(args.center_noise_frac) * sample_radius * rand_unit_vector(rng, stdlp.dim)
            chosen = 0
            if rng.random() < float(args.rare_prob):
                s = int(rng.choice(Slocal))
                q = Delta[:, s]
                qn = q / max(np.linalg.norm(q), 1e-12)
                lo = min(sample_radius, float(tau[s]) + 1e-4 * rho)
                hi = sample_radius
                amount = hi if hi <= lo else float(rng.uniform(lo, hi))
                c_std = np.asarray(c0_std, dtype=float) - amount * qn
                chosen = s + 1
            diff = c_std - c0_std
            ndiff = float(np.linalg.norm(diff))
            if ndiff > sample_radius + 1e-12:
                c_std = np.asarray(c0_std, dtype=float) + (sample_radius / ndiff) * diff
            if np.linalg.norm(c_std - c0_std) > rho + 1e-8:
                raise RuntimeError("Sampled cost fell outside the declared prior ball.")
            if np.linalg.norm(c_std) <= 1e-10:
                raise RuntimeError("Sampled cost hit the origin, which is not allowed.")
            reward = -np.asarray(c_std, dtype=float)
            inst = build_lp_instance(problem, reward, name=f"{problem.name}_{idx:05d}")
            inst.c_std = c_std
            inst.theta = np.zeros(0, dtype=float)
            inst.active_direction = chosen
            instances.append(inst)
            Cstd.append(c_std)
            theta_all.append(np.zeros(0, dtype=float))
            active_idx[idx] = chosen
        truth_slocal = Slocal
        truth_tau = tau
    else:
        raise ValueError(f"Unknown sample_mode={args.sample_mode!r}")

    train = instances[: args.n_train]
    val = instances[args.n_train : args.n_train + args.n_val]
    test = instances[args.n_train + args.n_val :]
    truth = {
        "dataset": problem.name,
        "reward_center": np.asarray(problem.reward_center, dtype=float),
        "U_reward": U_reward,
        "U_cost": U_cost,
        "prior_c0": c0_std,
        "prior_rho": np.asarray([rho], dtype=float),
        "prior_sample_radius": np.asarray([sample_radius], dtype=float),
        "prior_origin_margin": np.asarray([prior.origin_margin], dtype=float),
        "prior_nominal_k": np.asarray([nominal_k], dtype=int),
        "prior_tau_sorted": np.asarray(tau_sorted, dtype=float),
        "prior_center_basis": np.asarray(center_basis),
        "theta": np.vstack(theta_all) if theta_all else np.zeros((0, U_reward.shape[1] if U_reward.ndim == 2 else 0), dtype=float),
        "sample_mode": np.asarray([args.sample_mode]),
        "active_direction": active_idx,
        "A_ineq": np.asarray(problem.A_ineq, dtype=float),
        "b_ineq": np.asarray(problem.b_ineq, dtype=float),
        "A_eq": np.asarray(problem.A_eq, dtype=float),
        "b_eq": np.asarray(problem.b_eq, dtype=float),
        **problem.metadata,
    }
    if args.sample_mode == "iid_local_rare":
        truth["local_tau"] = np.asarray(truth_tau, dtype=float)
        truth["local_relevant_indices"] = np.asarray(truth_slocal, dtype=int)
    return FixedXBundle(
        train=train,
        val=val,
        test=test,
        Ctrain=np.vstack(Cstd[: args.n_train]),
        Cval=np.vstack(Cstd[args.n_train : args.n_train + args.n_val]),
        Ctest=np.vstack(Cstd[args.n_train + args.n_val :]),
        stdlp=stdlp,
        prior=prior,
        truth=truth,
        problem=problem,
    )


def make_packing_problem(args: argparse.Namespace, rng: np.random.Generator) -> OriginalFixedXProblem:
    if str(getattr(args, "packing_design", "random")).lower() == "block_gadget":
        n = int(args.n_vars)
        m = int(args.n_cons)
        if m < 4 or n < 4 * m:
            raise ValueError("block_gadget packing needs at least four variables per constraint row.")
        A_ineq = np.zeros((m, n), dtype=float)
        b_ineq = np.ones(m, dtype=float)
        reward_center = np.zeros(n, dtype=float)
        dirs: List[np.ndarray] = []
        gadget_meta: List[List[int]] = []
        cursor = 0
        requested_rank = int(args.cost_rank)
        n_gadgets = max(6, requested_rank // 2) if requested_rank > 0 else max(6, m // 6)
        n_gadgets = min(n_gadgets, m // 2, n // 10)
        if n_gadgets <= 0:
            raise ValueError("block_gadget packing needs enough rows and variables to plant paired-choice gadgets.")

        for g in range(n_gadgets):
            row_a = 2 * g
            row_b = 2 * g + 1
            base_idx = np.arange(cursor, min(cursor + 10, n), dtype=int)
            if len(base_idx) < 10:
                break
            cursor += 10

            upper = [int(base_idx[0]), int(base_idx[1])]
            lower = [int(base_idx[2]), int(base_idx[3])]
            decoy_a = [int(v) for v in base_idx[4:7]]
            decoy_b = [int(v) for v in base_idx[7:10]]

            cols_a = upper[:1] + lower[:1] + decoy_a
            cols_b = upper[1:] + lower[1:] + decoy_b
            A_ineq[row_a, cols_a] = np.clip(1.0 + 0.015 * rng.normal(size=len(cols_a)), 0.96, 1.04)
            A_ineq[row_b, cols_b] = np.clip(1.0 + 0.015 * rng.normal(size=len(cols_b)), 0.96, 1.04)

            pair_level = float(max(3.50, 4.00 + 0.03 * rng.normal()))
            reward_center[upper] = pair_level
            reward_center[lower] = pair_level
            reward_center[decoy_a] = 0.0
            reward_center[decoy_b] = 0.0

            v = np.zeros(n, dtype=float)
            v[upper] = 1.0
            v[lower] = -1.0
            dirs.append(v)
            gadget_meta.append([upper[0], upper[1], lower[0], lower[1]])

        filler_rows = list(range(2 * len(gadget_meta), m))
        for row in filler_rows:
            width = min(6, n - cursor)
            if width <= 0:
                break
            idx = np.arange(cursor, cursor + width, dtype=int)
            cursor += width
            A_ineq[row, idx] = np.clip(1.0 + 0.02 * rng.normal(size=len(idx)), 0.95, 1.05)
            if len(idx) > 0:
                reward_center[idx[0]] = float(max(0.06, 0.10 + 0.01 * rng.normal()))
            if len(idx) > 1:
                reward_center[idx[1:]] = 0.0

        row_cycle = filler_rows if filler_rows else list(range(2 * len(gadget_meta)))
        if not row_cycle:
            row_cycle = list(range(m))
        ptr = 0
        while cursor < n:
            row = int(row_cycle[ptr % len(row_cycle)])
            A_ineq[row, cursor] = float(np.clip(1.0 + 0.02 * rng.normal(), 0.95, 1.05))
            reward_center[cursor] = 0.0
            cursor += 1
            ptr += 1

        rank = len(dirs) if int(args.cost_rank) <= 0 else min(int(args.cost_rank), len(dirs))
        keep = evenly_spaced_indices(len(dirs), rank)
        U_reward = orthonormalize_columns(np.column_stack([dirs[i] for i in keep])) if len(keep) > 0 else np.zeros((n, 0), dtype=float)
        anchor_x0 = np.zeros(n, dtype=float)
        return OriginalFixedXProblem(
            name="packing_fixedX",
            reward_center=reward_center,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            A_eq=np.zeros((0, n), dtype=float),
            b_eq=np.zeros(0, dtype=float),
            ub=None,
            U_reward=U_reward,
            metadata={
                "packing_A": A_ineq,
                "packing_b": b_ineq,
                "packing_gadgets": np.asarray(gadget_meta, dtype=int),
                "anchor_x0": anchor_x0,
                "packing_design": np.asarray(["block_gadget"]),
            },
        )

    n = int(args.n_vars)
    m = int(args.n_cons)
    A_ineq = rng.uniform(0.3, 1.1, size=(m, n))
    b_ineq = float(max(1, n)) * rng.uniform(0.6, 1.0, size=m)
    reward_center = rng.uniform(0.2, 1.2, size=n)
    U_reward = make_random_cost_basis(n, int(args.cost_rank), rng)
    anchor_x0 = np.zeros(n, dtype=float)
    return OriginalFixedXProblem(
        name="packing_fixedX",
        reward_center=reward_center,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        A_eq=np.zeros((0, n), dtype=float),
        b_eq=np.zeros(0, dtype=float),
        ub=None,
        U_reward=U_reward,
        metadata={"packing_A": A_ineq, "packing_b": b_ineq, "anchor_x0": anchor_x0},
    )


def random_dag_edges_fixed(
    n_nodes: int,
    n_edges: int,
    rng: np.random.Generator,
    source: int = 0,
    sink: Optional[int] = None,
) -> List[Tuple[int, int]]:
    sink = n_nodes - 1 if sink is None else int(sink)
    mandatory = {(i, i + 1) for i in range(n_nodes - 1)}
    mandatory.add((source, sink))
    candidates = [(u, v) for u in range(n_nodes) for v in range(u + 1, n_nodes)]
    if n_edges < len(mandatory) or n_edges > len(candidates):
        raise ValueError("Invalid number of DAG edges.")
    remain = [e for e in candidates if e not in mandatory]
    chosen = rng.choice(len(remain), size=n_edges - len(mandatory), replace=False)
    edges = list(mandatory) + [remain[int(i)] for i in chosen]
    edges.sort()
    return edges


def build_parallel_channel_gadget_graph(
    n_nodes: int,
    target_n_edges: int,
    rng: np.random.Generator,
    n_channels: int = 3,
    gadgets_per_channel: int = 2,
) -> Tuple[List[Tuple[int, int]], List[Tuple[List[int], List[int]]], int, int]:
    source = 0
    sink = n_nodes - 1
    edges: List[Tuple[int, int]] = []
    gadget_edges: List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]] = []
    next_node = 1

    for _ in range(n_channels):
        start = next_node
        next_node += 1
        edges.append((source, start))
        current = start
        for g in range(gadgets_per_channel):
            up = next_node
            low = next_node + 1
            next_node += 2
            if g == gadgets_per_channel - 1:
                nxt = next_node
                next_node += 1
                upper_path = [(current, up), (up, nxt)]
                edges.extend(upper_path)
                lower_path = [(current, low), (low, nxt)]
                edges.extend(lower_path)
                gadget_edges.append((upper_path, lower_path))
                edges.append((nxt, sink))
            else:
                nxt = next_node
                next_node += 1
                upper_path = [(current, up), (up, nxt)]
                edges.extend(upper_path)
                lower_path = [(current, low), (low, nxt)]
                edges.extend(lower_path)
                gadget_edges.append((upper_path, lower_path))
                current = nxt

    if next_node > sink:
        raise ValueError("Not enough nodes for the requested maxflow gadget graph.")
    edges = add_internal_forward_edges(n_nodes, edges, target_n_edges, rng, source=source, sink=sink)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    gadgets = [([edge_to_idx[e] for e in up], [edge_to_idx[e] for e in low]) for up, low in gadget_edges]
    return edges, gadgets, source, sink


def build_serial_gadget_path_graph(
    n_nodes: int,
    target_n_edges: int,
    rng: np.random.Generator,
    n_gadgets: int = 7,
    include_direct_edge: bool = True,
) -> Tuple[List[Tuple[int, int]], List[Tuple[List[int], List[int]]], int, int, int]:
    source = 0
    sink = n_nodes - 1
    edges: List[Tuple[int, int]] = []
    gadget_edges: List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]] = []
    current = source
    next_node = 1

    for g in range(n_gadgets):
        up = next_node
        low = next_node + 1
        next_node += 2
        if g == n_gadgets - 1:
            nxt = sink
        else:
            nxt = next_node
            next_node += 1
        upper_path = [(current, up), (up, nxt)]
        edges.extend(upper_path)
        lower_path = [(current, low), (low, nxt)]
        edges.extend(lower_path)
        gadget_edges.append((upper_path, lower_path))
        current = nxt

    direct_edge = (-1, -1)
    if include_direct_edge:
        direct_edge = (source, sink)
        edges.append(direct_edge)

    if next_node > sink:
        raise ValueError("Not enough nodes for the requested serial gadget graph.")
    edges = add_internal_forward_edges(n_nodes, edges, target_n_edges, rng, source=source, sink=sink)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    gadgets = [([edge_to_idx[e] for e in up], [edge_to_idx[e] for e in low]) for up, low in gadget_edges]
    direct_idx = edge_to_idx[direct_edge] if include_direct_edge else -1
    return edges, gadgets, source, sink, direct_idx


def make_maxflow_problem(args: argparse.Namespace, rng: np.random.Generator) -> OriginalFixedXProblem:
    if str(getattr(args, "flow_design", "random")).lower() == "path_gadget":
        n_nodes = int(args.n_nodes)
        n_edges = int(args.n_edges)
        requested_rank = int(args.cost_rank)
        n_gadgets = max(6, requested_rank // 2) if requested_rank > 0 else max(6, (n_nodes - 2) // 6)
        n_gadgets = min(n_gadgets, max(2, (n_nodes - 2) // 3))
        edges, gadgets, source, sink, _direct_idx = build_serial_gadget_path_graph(
            n_nodes,
            n_edges,
            rng,
            n_gadgets=n_gadgets,
            include_direct_edge=False,
        )
        B = make_node_arc_incidence(n_nodes, edges)
        caps = 0.22 * np.ones(len(edges), dtype=float)
        reward_center = np.zeros(len(edges), dtype=float)
        U_cols: List[np.ndarray] = []
        for g, (up_idx, low_idx) in enumerate(gadgets):
            caps[up_idx] = 1.0
            caps[low_idx] = 1.0
            if g == 0:
                caps[up_idx] = 0.5
                caps[low_idx] = 0.5
            pair_level = float(max(2.80, 3.20 + 0.03 * rng.normal()))
            reward_center[up_idx] = pair_level
            reward_center[low_idx] = pair_level
            v = np.zeros(len(edges), dtype=float)
            v[up_idx] = 1.0
            v[low_idx] = -1.0
            U_cols.append(v)
        rank = len(U_cols) if int(args.cost_rank) <= 0 else min(int(args.cost_rank), len(U_cols))
        keep = evenly_spaced_indices(len(U_cols), rank)
        U_reward = orthonormalize_columns(np.column_stack([U_cols[i] for i in keep])) if len(keep) > 0 else np.zeros((len(edges), 0), dtype=float)
        anchor_x0 = np.zeros(len(edges), dtype=float)
        internal = [vtx for vtx in range(n_nodes) if vtx not in (source, sink)]
        A_eq = B[np.asarray(internal, dtype=int), :] if internal else np.zeros((0, len(edges)), dtype=float)
        A_ineq = np.eye(len(edges), dtype=float)
        return OriginalFixedXProblem(
            name="maxflow_fixedX",
            reward_center=reward_center,
            A_ineq=A_ineq,
            b_ineq=caps,
            A_eq=A_eq,
            b_eq=np.zeros(A_eq.shape[0], dtype=float),
            ub=None,
            U_reward=U_reward,
            metadata={
                "edges": np.asarray(edges, dtype=int),
                "caps": caps,
                "incidence": B,
                "maxflow_gadgets": np.asarray([[up, low] for up, low in gadgets], dtype=int),
                "anchor_x0": anchor_x0,
                "flow_design": np.asarray(["path_gadget"]),
                "projection_coordinates": "original_nonnegative",
            },
        )

    n_nodes = int(args.n_nodes)
    n_edges = int(args.n_edges)
    source = 0
    sink = n_nodes - 1
    edges = random_dag_edges_fixed(n_nodes, n_edges, rng, source=source, sink=sink)
    B = make_node_arc_incidence(n_nodes, edges)
    caps = rng.uniform(0.5, 1.5, size=n_edges)
    reward_center = 0.05 * rng.normal(size=n_edges)
    for j, (u, vtx) in enumerate(edges):
        if u == source or vtx == sink:
            reward_center[j] += 1.5
    internal = [vtx for vtx in range(n_nodes) if vtx not in (source, sink)]
    A_eq = B[np.asarray(internal, dtype=int), :] if internal else np.zeros((0, n_edges), dtype=float)
    A_ineq = np.eye(n_edges, dtype=float)
    U_reward = make_random_cost_basis(n_edges, int(args.cost_rank), rng)
    anchor_x0 = np.zeros(n_edges, dtype=float)
    return OriginalFixedXProblem(
        name="maxflow_fixedX",
        reward_center=reward_center,
        A_ineq=A_ineq,
        b_ineq=caps,
        A_eq=A_eq,
        b_eq=np.zeros(A_eq.shape[0], dtype=float),
        ub=None,
        U_reward=U_reward,
        metadata={
            "edges": np.asarray(edges, dtype=int),
            "caps": caps,
            "incidence": B,
            "anchor_x0": anchor_x0,
            "projection_coordinates": "original_nonnegative",
        },
    )


def make_mincostflow_problem(args: argparse.Namespace, rng: np.random.Generator) -> OriginalFixedXProblem:
    if str(getattr(args, "flow_design", "random")).lower() == "path_gadget":
        n_nodes = int(args.n_nodes)
        n_edges = int(args.n_edges)
        n_gadgets = max(4, (n_nodes - 1) // 3)
        edges, gadgets, source, sink, direct_idx = build_serial_gadget_path_graph(
            n_nodes,
            n_edges,
            rng,
            n_gadgets=n_gadgets,
            include_direct_edge=True,
        )
        B = make_node_arc_incidence(n_nodes, edges)
        demand = np.zeros(n_nodes, dtype=float)
        demand[source] = 1.0
        demand[sink] = -1.0
        costs = np.full(len(edges), 3.2, dtype=float)
        if direct_idx >= 0:
            costs[direct_idx] = float(max(12.0, 0.5 * len(edges)))
        U_cols: List[np.ndarray] = []
        for up_idx, low_idx in gadgets:
            costs[up_idx] = 1.00
            costs[low_idx] = 1.04
            v = np.zeros(len(edges), dtype=float)
            v[up_idx] = 1.0
            v[low_idx] = -1.0
            U_cols.append(v)
        reward_center = reward_from_costs_via_potential(edges, costs, n_nodes, scale=6.0)
        rank = len(U_cols) if int(args.cost_rank) <= 0 else min(int(args.cost_rank), len(U_cols))
        keep = evenly_spaced_indices(len(U_cols), rank)
        U_reward = orthonormalize_columns(np.column_stack([U_cols[i] for i in keep])) if len(keep) > 0 else np.zeros((len(edges), 0), dtype=float)
        anchor_x0 = np.zeros(len(edges), dtype=float)
        if direct_idx >= 0:
            anchor_x0[direct_idx] = 1.0
        return OriginalFixedXProblem(
            name="mincostflow_fixedX",
            reward_center=reward_center,
            A_ineq=np.zeros((0, len(edges)), dtype=float),
            b_ineq=np.zeros(0, dtype=float),
            A_eq=B,
            b_eq=demand,
            ub=np.ones(len(edges), dtype=float),
            U_reward=U_reward,
            metadata={
                "edges": np.asarray(edges, dtype=int),
                "incidence": B,
                "demand": demand,
                "base_costs": costs,
                "direct_edge_index": np.asarray([direct_idx], dtype=int),
                "mincost_gadgets": np.asarray([[up, low] for up, low in gadgets], dtype=int),
                "anchor_x0": anchor_x0,
                "flow_design": np.asarray(["path_gadget"]),
            },
        )

    n_nodes = int(args.n_nodes)
    n_edges = int(args.n_edges)
    source = 0
    sink = n_nodes - 1
    edges = random_dag_edges_fixed(n_nodes, n_edges, rng, source=source, sink=sink)
    B = make_node_arc_incidence(n_nodes, edges)
    demand = np.zeros(n_nodes, dtype=float)
    demand[source] = 1.0
    demand[sink] = -1.0
    costs = rng.uniform(0.0, 1.0, size=n_edges)
    direct_idx = edges.index((source, sink))
    costs[direct_idx] = float(max(10.0, n_edges))
    reward_center = reward_from_costs_via_potential(edges, costs, n_nodes, scale=4.0)
    U_reward = make_random_cost_basis(n_edges, int(args.cost_rank), rng)
    anchor_x0 = np.zeros(n_edges, dtype=float)
    anchor_x0[direct_idx] = 1.0
    return OriginalFixedXProblem(
        name="mincostflow_fixedX",
        reward_center=reward_center,
        A_ineq=np.zeros((0, n_edges), dtype=float),
        b_ineq=np.zeros(0, dtype=float),
        A_eq=B,
        b_eq=demand,
        ub=np.ones(n_edges, dtype=float),
        U_reward=U_reward,
        metadata={
            "edges": np.asarray(edges, dtype=int),
            "incidence": B,
            "demand": demand,
            "base_costs": costs,
            "direct_edge_index": np.asarray([direct_idx], dtype=int),
            "anchor_x0": anchor_x0,
        },
    )


def make_shortest_path_problem(args: argparse.Namespace, rng: np.random.Generator) -> OriginalFixedXProblem:
    grid_size = int(args.grid_size)
    edges, edge_to_idx = make_monotone_grid_edges(grid_size)
    n_edges = len(edges)
    n_nodes = grid_size * grid_size
    B = make_node_arc_incidence(n_nodes, edges)
    demand = np.zeros(n_nodes, dtype=float)
    demand[0] = 1.0
    demand[-1] = -1.0

    cost_low = float(args.sp_cost_low)
    cost_high = float(args.sp_cost_high)
    costs0 = np.full(n_edges, cost_high, dtype=float)
    band_width = int(args.sp_band_width)
    for idx, (u, v) in enumerate(edges):
        ui, uj = divmod(u, grid_size)
        vi, vj = divmod(v, grid_size)
        if abs(ui - uj) <= band_width and abs(vi - vj) <= band_width:
            costs0[idx] = cost_low

    candidate_cells = [(r, c) for r in range(grid_size - 1) for c in range(grid_size - 1) if (r + c) % 2 == 0]
    max_gadgets = len(candidate_cells)
    auto_gadgets = min(max_gadgets, max(4, min(int(args.cost_rank) if int(args.cost_rank) > 0 else 8, grid_size)))
    n_gadgets = int(args.shortest_gadgets) if int(args.shortest_gadgets) > 0 else auto_gadgets
    chosen_cells = [candidate_cells[i] for i in evenly_spaced_indices(len(candidate_cells), min(n_gadgets, len(candidate_cells)))]
    gadgets: List[Tuple[int, int, int, int]] = []
    for r, c in chosen_cells:
        e_top = edge_to_idx[(r * grid_size + c, r * grid_size + c + 1)]
        e_right = edge_to_idx[(r * grid_size + c + 1, (r + 1) * grid_size + c + 1)]
        e_left = edge_to_idx[(r * grid_size + c, (r + 1) * grid_size + c)]
        e_bottom = edge_to_idx[((r + 1) * grid_size + c, (r + 1) * grid_size + c + 1)]
        gadgets.append((e_top, e_right, e_left, e_bottom))
        costs0[[e_top, e_right, e_left, e_bottom]] = cost_low

    reward_center = reward_from_costs_via_potential(edges, costs0, n_nodes, scale=float(args.sp_potential_scale))
    if gadgets:
        U_reward = np.zeros((n_edges, len(gadgets)), dtype=float)
        for k, (e_top, e_right, e_left, e_bottom) in enumerate(gadgets):
            U_reward[e_top, k] = 0.5
            U_reward[e_right, k] = 0.5
            U_reward[e_left, k] = -0.5
            U_reward[e_bottom, k] = -0.5
        U_reward = orthonormalize_columns(U_reward)
    else:
        U_reward = make_random_cost_basis(n_edges, int(args.cost_rank), rng)
    anchor_x0 = np.zeros(n_edges, dtype=float)
    cur = 0
    for _ in range(grid_size - 1):
        nxt = cur + 1
        anchor_x0[edge_to_idx[(cur, nxt)]] = 1.0
        cur = nxt
    for _ in range(grid_size - 1):
        nxt = cur + grid_size
        anchor_x0[edge_to_idx[(cur, nxt)]] = 1.0
        cur = nxt
    return OriginalFixedXProblem(
        name="shortest_path_fixedX",
        reward_center=reward_center,
        A_ineq=np.zeros((0, n_edges), dtype=float),
        b_ineq=np.zeros(0, dtype=float),
        A_eq=B,
        b_eq=demand,
        ub=np.ones(n_edges, dtype=float),
        U_reward=U_reward,
        metadata={
            "edges": np.asarray(edges, dtype=int),
            "incidence": B,
            "demand": demand,
            "gadgets": np.asarray(gadgets, dtype=int),
            "base_costs": costs0,
            "grid_size": np.asarray([grid_size], dtype=int),
            "anchor_x0": anchor_x0,
        },
    )


def make_random_lp_problem(args: argparse.Namespace, rng: np.random.Generator) -> OriginalFixedXProblem:
    n = int(args.randlp_n_vars)
    m_eq = int(args.randlp_n_eq)
    m_ineq = int(args.randlp_n_ineq)
    ub = rng.uniform(1.0, 2.0, size=n)
    z0 = ub * rng.uniform(0.2, 0.8, size=n)
    A_eq = rng.normal(size=(m_eq, n)) if m_eq > 0 else np.zeros((0, n), dtype=float)
    b_eq = A_eq @ z0 if m_eq > 0 else np.zeros(0, dtype=float)
    A_ineq = rng.normal(size=(m_ineq, n)) if m_ineq > 0 else np.zeros((0, n), dtype=float)
    if m_ineq > 0:
        margin = rng.uniform(0.2, 0.8, size=m_ineq)
        b_ineq = A_ineq @ z0 + margin
    else:
        b_ineq = np.zeros(0, dtype=float)
    reward_center = rng.normal(size=n)
    if np.linalg.norm(reward_center) <= 1e-8:
        reward_center[0] = 1.0
    U_reward = make_random_cost_basis(n, int(args.cost_rank), rng)
    return OriginalFixedXProblem(
        name="random_lp_fixedX",
        reward_center=reward_center,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        A_eq=A_eq,
        b_eq=b_eq,
        ub=ub,
        U_reward=U_reward,
        metadata={"random_anchor_z0": z0, "random_ub": ub, "anchor_x0": z0},
    )


def make_random_standardform_problem(args: argparse.Namespace, rng: np.random.Generator) -> OriginalFixedXProblem:
    m = int(args.std_m)
    d = int(args.std_d)
    if d <= m + 5:
        raise ValueError("Need std_d substantially larger than std_m for the random standard-form benchmark.")
    B = np.sort(rng.choice(d, size=m, replace=False))
    N = np.array([j for j in range(d) if j not in set(B.tolist())], dtype=int)
    AB = np.abs(rng.random((m, m)) + float(args.std_Anoise) * rng.normal(size=(m, m))) + 0.05 * np.eye(m, dtype=float)
    while np.linalg.cond(AB) > 1e6:
        AB = np.abs(rng.random((m, m)) + float(args.std_Anoise) * rng.normal(size=(m, m))) + 0.05 * np.eye(m, dtype=float)
    AN = np.abs(rng.random((m, len(N))) + float(args.std_Anoise) * rng.normal(size=(m, len(N)))) + 0.02 * rng.random((m, len(N)))
    A = np.zeros((m, d), dtype=float)
    A[:, B] = AB
    A[:, N] = AN

    xB0 = 0.5 + rng.random(m)
    b = AB @ xB0
    x0 = np.zeros(d, dtype=float)
    x0[B] = xB0
    ub = np.min(b.reshape(-1, 1) / np.maximum(A, 1e-12), axis=0)

    cB = 5.0 + rng.normal(size=m)
    lam = np.linalg.solve(AB.T, cB)
    r_raw = np.exp(float(args.std_reduced_cost_spread) * rng.normal(size=len(N)))
    r_raw = r_raw / max(float(np.median(r_raw)), 1e-12)
    cN = AN.T @ lam + r_raw
    c0 = np.zeros(d, dtype=float)
    c0[B] = cB
    c0[N] = cN

    T = np.linalg.solve(AB, AN)
    DeltaN = np.zeros((d, len(N)), dtype=float)
    tau = np.zeros(len(N), dtype=float)
    for k in range(len(N)):
        delta = np.zeros(d, dtype=float)
        delta[B] = -T[:, k]
        delta[N[k]] = 1.0
        DeltaN[:, k] = delta
        tau[k] = max(0.0, float(r_raw[k])) / max(float(np.linalg.norm(delta)), 1e-12)
    rank = len(N) if int(args.cost_rank) <= 0 else min(int(args.cost_rank), len(N))
    keep = np.argsort(tau)[:rank]
    delta_scaled = DeltaN / np.maximum(np.linalg.norm(DeltaN, axis=0, keepdims=True), 1e-12)
    U_reward = orthonormalize_columns(delta_scaled[:, keep]) if len(keep) > 0 else np.zeros((d, 0), dtype=float)

    reward_center = -c0
    return OriginalFixedXProblem(
        name="random_stdform_fixedX",
        reward_center=reward_center,
        A_ineq=np.zeros((0, d), dtype=float),
        b_ineq=np.zeros(0, dtype=float),
        A_eq=A,
        b_eq=b,
        ub=None,
        U_reward=U_reward,
        metadata={
            "center_cost_std": c0,
            "basis_B0": B,
            "nonbasis_N0": N,
            "anchor_x0": x0,
            "projection_ub": ub,
            "anchor_basis_values": xB0,
            "anchor_cB": cB,
            "anchor_reduced_costs": r_raw,
            "pivot_matrix_T": T,
            "pivot_directions": DeltaN,
            "pivot_tau": tau,
            "natural_basis_keep": keep,
            "natural_basis_mode": np.asarray(["low_tau_pivot"], dtype=object),
        },
    )


def build_problem(args: argparse.Namespace, rng: np.random.Generator) -> OriginalFixedXProblem:
    if args.dataset == "packing":
        return make_packing_problem(args, rng)
    if args.dataset == "maxflow":
        return make_maxflow_problem(args, rng)
    if args.dataset == "mincostflow":
        return make_mincostflow_problem(args, rng)
    if args.dataset == "shortest_path":
        return make_shortest_path_problem(args, rng)
    if args.dataset == "random_lp":
        return make_random_lp_problem(args, rng)
    if args.dataset == "random_stdform":
        return make_random_standardform_problem(args, rng)
    raise ValueError(args.dataset)


def append_ours_exact_rows(
    rows: List[Dict[str, object]],
    stdlp: StandardFormLP,
    U: np.ndarray,
    x_anchor: np.ndarray,
    instances: Sequence[LPInstance],
    comparison_ks: Sequence[int],
) -> None:
    intrinsic_k = int(U.shape[1])
    for inst in instances:
        c_std = np.asarray(inst.c_std, dtype=float)
        x_std, _obj_min, runtime, success, msg = solve_ours_exact_on_instance(stdlp, c_std, U, x_anchor)
        obj = float(inst.c @ x_std[stdlp.n_slack:]) if success and x_std is not None else float("nan")
        for k in comparison_ks:
            rows.append(
                {
                    "method": "OursExact",
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


def append_full_rows(rows: List[Dict[str, object]], instances: Sequence[LPInstance], k_list: Sequence[int]) -> None:
    for k in k_list:
        for inst in instances:
            rows.append(
                {
                    "method": "Full",
                    "K": int(k),
                    "instance": inst.name,
                    "objective": inst.full_obj,
                    "full_objective": inst.full_obj,
                    "objective_ratio": 1.0,
                    "time": inst.full_time,
                    "success": 1.0,
                }
            )


def detect_anchor_for_capture(truth: Dict[str, object]) -> Optional[np.ndarray]:
    for key in ("anchor_x0", "random_anchor_z0"):
        if key in truth:
            return np.asarray(truth[key], dtype=float).reshape(-1)
    return None


def summarize_improvement_capture(
    raw: pd.DataFrame,
    instances: Sequence[LPInstance],
    anchor_x: Optional[np.ndarray],
) -> pd.DataFrame:
    if anchor_x is None or raw.empty:
        return pd.DataFrame()
    base_obj = {inst.name: float(np.asarray(inst.c, dtype=float) @ anchor_x) for inst in instances}
    out_rows: List[Dict[str, object]] = []
    for (method, k), grp in raw.groupby(["method", "K"], dropna=False):
        imp_full = 0.0
        imp_method = 0.0
        n = 0
        for row in grp.itertuples(index=False):
            name = getattr(row, "instance")
            if name not in base_obj:
                continue
            base = base_obj[name]
            full_obj = float(getattr(row, "full_objective"))
            meth_obj = float(getattr(row, "objective")) if pd.notna(getattr(row, "objective")) else float("nan")
            imp_full += max(0.0, full_obj - base)
            if np.isfinite(meth_obj):
                imp_method += max(0.0, meth_obj - base)
            n += 1
        cap = np.nan if imp_full <= 1e-12 else imp_method / imp_full
        out_rows.append(
            {
                "method": method,
                "K": k,
                "improvement_capture": cap,
                "method_improvement_total": imp_method,
                "full_improvement_total": imp_full,
                "n": n,
            }
        )
    return pd.DataFrame(out_rows)


def convert_to_general_projection_instance(inst: LPInstance) -> GeneralProjectionInstance:
    blocks = []
    rhs = []
    if inst.A.shape[0] > 0:
        blocks.append(np.asarray(inst.A, dtype=float))
        rhs.append(np.asarray(inst.b, dtype=float).reshape(-1))
    if inst.A_eq is not None and inst.A_eq.shape[0] > 0:
        Aeq = np.asarray(inst.A_eq, dtype=float)
        beq = np.asarray(inst.b_eq, dtype=float).reshape(-1)
        blocks.extend([Aeq, -Aeq])
        rhs.extend([beq, -beq])
    blocks.append(-np.eye(inst.n_vars, dtype=float))
    rhs.append(np.zeros(inst.n_vars, dtype=float))
    return GeneralProjectionInstance(
        c=np.asarray(inst.c, dtype=float).reshape(-1),
        A=np.vstack(blocks),
        b=np.concatenate(rhs),
        name=inst.name,
        objective_constant=float(getattr(inst, "objective_constant", 0.0)),
        full_obj=inst.full_obj,
    )


def solve_projected_general_free_y(inst: GeneralProjectionInstance, P: np.ndarray) -> GeneralProjectionSolve:
    t0 = time.perf_counter()
    P = np.asarray(P, dtype=float)
    k = int(P.shape[1])
    q = P.T @ np.asarray(inst.c, dtype=float)
    G = np.asarray(inst.A, dtype=float) @ P
    G_split = np.hstack([G, -G])
    q_split = np.concatenate([q, -q])
    res = linprog(
        c=-q_split,
        A_ub=G_split,
        b_ub=np.asarray(inst.b, dtype=float),
        bounds=[(0.0, None)] * (2 * k),
        method="highs",
        options={"presolve": True},
    )
    runtime = time.perf_counter() - t0
    if not res.success:
        return GeneralProjectionSolve(None, None, float("nan"), False, runtime, None, res.message)
    u = np.asarray(res.x, dtype=float)
    y = u[:k] - u[k:]
    x = P @ y
    try:
        lam = -np.asarray(res.ineqlin.marginals, dtype=float)
        lam = np.maximum(lam, 0.0)
    except Exception:
        lam = np.zeros(inst.A.shape[0], dtype=float)
    return GeneralProjectionSolve(x, y, float(inst.objective_constant + inst.c @ x), True, runtime, lam, res.message)


def raw_pca_projection(train: Sequence[LPInstance], k: int) -> np.ndarray:
    if any(inst.full_x is None for inst in train):
        ensure_full_solutions(train)
    X = np.vstack([inst.full_x for inst in train])
    xbar = X.mean(axis=0)
    if k <= 1:
        return xbar.reshape(-1, 1)
    Xc = X - xbar[None, :]
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T[:, : max(0, k - 1)]
    P = np.column_stack([xbar, V])
    if P.shape[1] < k:
        pad = np.zeros((X.shape[1], k - P.shape[1]), dtype=float)
        P = np.column_stack([P, pad])
    return P[:, :k]


def project_point_to_common_region(
    p: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    feasible_hint: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(-1)
    if np.all(A @ p <= b + tol):
        return p

    def fun(x: np.ndarray) -> float:
        diff = x - p
        return 0.5 * float(diff @ diff)

    def jac(x: np.ndarray) -> np.ndarray:
        return x - p

    cons = LinearConstraint(A, -np.inf * np.ones(A.shape[0], dtype=float), b)
    x0 = np.asarray(feasible_hint, dtype=float).reshape(-1)
    res = minimize(
        fun=fun,
        x0=x0,
        jac=jac,
        method="SLSQP",
        constraints=[cons],
        options={"ftol": 1e-9, "maxiter": 400, "disp": False},
    )
    if not res.success:
        return x0.copy()
    return np.asarray(res.x, dtype=float)


def project_columns_to_common_region(
    P: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    feasible_hint: np.ndarray,
) -> np.ndarray:
    cols = []
    for j in range(P.shape[1]):
        cols.append(project_point_to_common_region(P[:, j], A, b, feasible_hint))
    return np.column_stack(cols)


def general_projection_grad(
    inst: GeneralProjectionInstance,
    y: np.ndarray,
    lam: np.ndarray,
) -> np.ndarray:
    grad_vec = np.asarray(inst.c, dtype=float) - np.asarray(inst.A, dtype=float).T @ np.asarray(lam, dtype=float).reshape(-1)
    grad = np.outer(grad_vec, np.asarray(y, dtype=float).reshape(-1))
    denom = max(abs(float(inst.full_obj or 0.0)), 1e-8)
    return grad / denom


def evaluate_general_projection_matrix(
    P: np.ndarray,
    instances: Sequence[GeneralProjectionInstance],
) -> Dict[str, float]:
    ratios = []
    qualities = []
    times = []
    for inst in instances:
        sol = solve_projected_general_free_y(inst, P)
        if sol.success:
            ratios.append(objective_ratio(sol.obj, inst.full_obj))
            if inst.full_obj is not None and np.isfinite(inst.full_obj):
                denom = max(abs(float(inst.full_obj)), 1e-8)
                qualities.append(1.0 - (float(inst.full_obj) - float(sol.obj)) / denom)
            times.append(sol.runtime)
    return {
        "objective_ratio_mean": float(np.mean(ratios)) if ratios else float("nan"),
        "quality_mean": float(np.mean(qualities)) if qualities else float("nan"),
        "time_mean": float(np.mean(times)) if times else float("nan"),
    }


def append_general_projection_rows(
    rows: List[Dict[str, object]],
    method: str,
    P: np.ndarray,
    instances: Sequence[GeneralProjectionInstance],
    k: int,
) -> None:
    for inst in instances:
        sol = solve_projected_general_free_y(inst, P)
        rows.append(
            {
                "method": method,
                "K": int(k),
                "instance": inst.name,
                "objective": sol.obj,
                "full_objective": inst.full_obj,
                "objective_ratio": objective_ratio(sol.obj, inst.full_obj) if sol.success else np.nan,
                "time": sol.runtime,
                "success": float(sol.success),
            }
        )


def train_sga_final_projection(
    train_lp: Sequence[LPInstance],
    val_lp: Sequence[LPInstance],
    k: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    train = [convert_to_general_projection_instance(inst) for inst in train_lp]
    val = [convert_to_general_projection_instance(inst) for inst in val_lp]
    common_A = np.asarray(train[0].A, dtype=float)
    common_b = np.asarray(train[0].b, dtype=float)
    feasible_hint = np.asarray(train_lp[0].full_x, dtype=float).reshape(-1)

    P = raw_pca_projection(train_lp, k)
    P = project_columns_to_common_region(P, common_A, common_b, feasible_hint)
    history: Dict[str, List[float]] = {"train_quality": [], "val_quality": [], "train_ratio": [], "val_ratio": []}
    best_P = P.copy()
    init_eval = evaluate_general_projection_matrix(P, val) if val else {"quality_mean": -float("inf"), "objective_ratio_mean": float("nan")}
    best_val = float(init_eval["quality_mean"]) if val else -float("inf")
    rng = np.random.default_rng(seed)

    for ep in range(1, epochs + 1):
        order = rng.permutation(len(train))
        for start in range(0, len(order), batch_size):
            batch_ids = order[start : start + batch_size]
            P = project_columns_to_common_region(P, common_A, common_b, feasible_hint)
            grad = np.zeros_like(P)
            n_used = 0
            for idx in batch_ids:
                inst = train[int(idx)]
                sol = solve_projected_general_free_y(inst, P)
                if not sol.success or sol.y is None or sol.dual_for_max is None:
                    continue
                grad += general_projection_grad(inst, sol.y, sol.dual_for_max)
                n_used += 1
            if n_used > 0:
                P = P + (float(lr) / float(n_used)) * grad

        P_eval = project_columns_to_common_region(P, common_A, common_b, feasible_hint)
        train_eval = evaluate_general_projection_matrix(P_eval, train)
        val_eval = evaluate_general_projection_matrix(P_eval, val) if val else {"quality_mean": float("nan"), "objective_ratio_mean": float("nan")}
        train_quality = float(train_eval["quality_mean"])
        val_quality = float(val_eval["quality_mean"])
        train_ratio = float(train_eval["objective_ratio_mean"])
        val_ratio = float(val_eval["objective_ratio_mean"])
        history["train_quality"].append(train_quality)
        history["val_quality"].append(val_quality)
        history["train_ratio"].append(train_ratio)
        history["val_ratio"].append(val_ratio)
        if np.isfinite(val_quality) and val_quality > best_val + 1e-6:
            best_val = val_quality
            best_P = P_eval.copy()
        if verbose and (ep == 1 or ep == epochs or ep % max(1, epochs // 10) == 0):
            print(
                f"SGA epoch {ep:04d}/{epochs} | "
                f"train quality {train_quality:.4f} | val quality {val_quality:.4f} | "
                f"train ratio {train_ratio:.4f} | val ratio {val_ratio:.4f}"
            )

    best_P = project_columns_to_common_region(best_P, common_A, common_b, feasible_hint)
    return best_P, history


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", type=str, default="packing", choices=["packing", "maxflow", "mincostflow", "shortest_path", "random_lp", "random_stdform"])
    p.add_argument("--out_dir", type=str, default="fixedX_family_results")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--quick", action="store_true", help="small setting for quick verification")

    p.add_argument("--n_train", type=int, default=50)
    p.add_argument("--n_val", type=int, default=8)
    p.add_argument("--n_test", type=int, default=15)
    p.add_argument("--k_list", type=str, default="4,8,12")
    p.add_argument("--prior_nominal_k", type=int, default=0)
    p.add_argument("--sample_radius_frac", type=float, default=0.98)
    p.add_argument("--origin_margin_frac", type=float, default=0.45)
    p.add_argument("--reward_margin_frac", type=float, default=0.8)
    p.add_argument("--cost_rank", type=int, default=0, help="hidden reward-variation subspace rank; 0 means full original dimension except shortest_path gadgets")
    p.add_argument(
        "--sample_mode",
        type=str,
        default="uniform_ball",
        choices=["uniform_ball", "factor_gaussian", "multiplicative_factor", "sparse_rare_ball", "iid_local_rare"],
    )
    p.add_argument("--rare_prob", type=float, default=0.25, help="iidLocalRare: per-sample probability of activating one rare local direction")
    p.add_argument("--center_noise_frac", type=float, default=0.02, help="iidLocalRare: center noise radius as a fraction of sample_radius")
    p.add_argument("--rare_amp_frac", type=float, default=0.85, help="sparseRareBall: rare-activation amplitude as a fraction of sample_radius")
    p.add_argument("--sparse_rare_max_active", type=int, default=1, help="sparseRareBall: maximum number of active basis directions per sample")
    p.add_argument("--factor_scale_frac", type=float, default=0.55, help="factorGaussian: base latent std as a fraction of sample_radius")
    p.add_argument("--factor_decay", type=float, default=0.82, help="factorGaussian: geometric decay across latent-factor scales")
    p.add_argument("--mult_log_scale", type=float, default=0.35, help="multiplicativeFactor: maximum absolute log-multiplier")
    p.add_argument("--packing_design", type=str, default="random", choices=["random", "block_gadget"])
    p.add_argument("--flow_design", type=str, default="random", choices=["random", "path_gadget"])

    p.add_argument("--n_vars", type=int, default=60, help="packing variables")
    p.add_argument("--n_cons", type=int, default=16, help="packing constraints")
    p.add_argument("--n_nodes", type=int, default=18, help="network nodes")
    p.add_argument("--n_edges", type=int, default=60, help="network edges")
    p.add_argument("--grid_size", type=int, default=8, help="shortest-path grid size")
    p.add_argument("--shortest_gadgets", type=int, default=0, help="number of planted shortest-path gadget directions; 0 means auto")
    p.add_argument("--sp_band_width", type=int, default=1)
    p.add_argument("--sp_cost_low", type=float, default=1.0)
    p.add_argument("--sp_cost_high", type=float, default=3.0)
    p.add_argument("--sp_potential_scale", type=float, default=6.0)
    p.add_argument("--randlp_n_vars", type=int, default=28)
    p.add_argument("--randlp_n_eq", type=int, default=8)
    p.add_argument("--randlp_n_ineq", type=int, default=10)
    p.add_argument("--std_m", type=int, default=120, help="random standard-form: number of equalities")
    p.add_argument("--std_d", type=int, default=2200, help="random standard-form: number of variables")
    p.add_argument("--std_Anoise", type=float, default=0.20)
    p.add_argument("--std_reduced_cost_spread", type=float, default=3.0)

    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--n_pelp_layers", type=int, default=4)
    p.add_argument("--generator_hidden_dim", type=int, default=32)
    p.add_argument("--sga_lr", type=float, default=1e-2)
    p.add_argument("--sga_epochs", type=int, default=8)

    p.add_argument("--rand_trials", type=int, default=3)
    p.add_argument("--run_sharedp", action="store_true")
    p.add_argument("--run_fcnn", action="store_true")
    p.add_argument("--skip_pelp", action="store_true")
    p.add_argument("--skip_sga", action="store_true")
    p.add_argument("--skip_pca2024", action="store_true")
    p.add_argument("--make_plots", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def apply_quick_settings(args: argparse.Namespace) -> None:
    args.n_train = 30
    args.n_val = 6
    args.n_test = 10
    args.k_list = "3,6"
    args.epochs = 6
    args.batch_size = 4
    args.patience = 6
    args.rand_trials = 2
    args.sga_epochs = 5
    if args.dataset == "packing":
        args.n_vars = 32
        args.n_cons = 10
    elif args.dataset in {"maxflow", "mincostflow"}:
        args.n_nodes = 12
        args.n_edges = 30
    elif args.dataset == "shortest_path":
        args.grid_size = 7
        args.shortest_gadgets = 3
    elif args.dataset == "random_lp":
        args.randlp_n_vars = 20
        args.randlp_n_eq = 5
        args.randlp_n_ineq = 6
    elif args.dataset == "random_stdform":
        args.std_m = 40
        args.std_d = 220
        args.sample_mode = "iid_local_rare"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.quick:
        apply_quick_settings(args)
    if args.dataset == "random_stdform" and args.sample_mode == "uniform_ball":
        args.sample_mode = "iid_local_rare"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    problem = build_problem(args, rng)
    bundle = make_fixed_x_bundle(problem, args, rng)
    np.savez(out_dir / "dataset_truth_and_prior.npz", **bundle.truth)
    print(
        f"Dataset={args.dataset}: train={len(bundle.train)}, val={len(bundle.val)}, test={len(bundle.test)}, "
        f"n_orig={problem.n_vars}, n_slack={bundle.stdlp.n_slack}, prior_rho={bundle.prior.rho:.4f}"
    )
    print(f"Prior origin margin ||c0||-rho = {bundle.prior.origin_margin:.4f}")

    print("Solving full LPs for train / val / test...")
    t_full_fit = time.perf_counter()
    ensure_full_solutions(bundle.train)
    ensure_full_solutions(bundle.val)
    ensure_full_solutions(bundle.test)
    full_fit_time = time.perf_counter() - t_full_fit

    k_list = parse_k_list(args.k_list)
    rows: List[Dict[str, object]] = []
    fit_rows: List[Dict[str, object]] = [
        {
            "method": "FullPrecompute",
            "K": np.nan,
            "fit_time": full_fit_time,
            "train_samples": len(bundle.train),
            "val_samples": len(bundle.val),
            "test_samples": len(bundle.test),
            "epochs": 0,
            "batch_size": 0,
            "notes": "full LP solves for train/val/test objective references",
        }
    ]
    append_full_rows(rows, bundle.test, k_list)

    print("\nRunning our Algorithm 1/2 stage-I learner...")
    t0 = time.perf_counter()
    stage1 = alg2_cumulative(bundle.stdlp, bundle.Ctrain, bundle.prior, verbose=args.verbose)
    stage1_time = time.perf_counter() - t0
    print(
        f"Ours stage-I finished: learned rank={stage1.U.shape[1]}, "
        f"hard samples={len(stage1.hard_indices)}, time={stage1_time:.3f}s"
    )
    fit_rows.append(
        {
            "method": "OursStageI",
            "K": int(stage1.U.shape[1]),
            "fit_time": stage1_time,
            "train_samples": len(bundle.train),
            "val_samples": 0,
            "test_samples": len(bundle.test),
            "epochs": 0,
            "batch_size": 0,
            "notes": "Algorithm 2 cumulative learner over train costs only",
        }
    )
    append_ours_exact_rows(rows, bundle.stdlp, stage1.U, stage1.x_anchor, bundle.test, comparison_ks=k_list)
    np.savez(
        out_dir / "ours_stageI_learned_basis.npz",
        U=stage1.U,
        D=stage1.D,
        x_anchor=stage1.x_anchor,
        rank_after_sample=np.asarray(stage1.rank_after_sample),
        hard_indices=np.asarray(stage1.hard_indices, dtype=int),
    )
    pd.DataFrame(
        {"sample": np.arange(1, len(stage1.rank_after_sample) + 1), "rank_after_sample": stage1.rank_after_sample}
    ).to_csv(out_dir / "ours_rank_after_sample.csv", index=False)
    pd.DataFrame(
        {"query": np.arange(1, len(stage1.rank_after_query) + 1), "rank_after_query": stage1.rank_after_query}
    ).to_csv(out_dir / "ours_rank_after_query.csv", index=False)

    general_test = [convert_to_general_projection_instance(inst) for inst in bundle.test]
    general_train = [convert_to_general_projection_instance(inst) for inst in bundle.train]
    general_common_A = np.asarray(general_train[0].A, dtype=float)
    general_common_b = np.asarray(general_train[0].b, dtype=float)
    general_feasible_hint = np.asarray(bundle.train[0].full_x, dtype=float).reshape(-1)

    for k in k_list:
        print(f"\n===== K={k} =====")
        for tr in range(args.rand_trials):
            P_rand = random_column_projection(problem.n_vars, k, rng)
            append_fixed_projection_rows(rows, f"Rand#{tr+1}", P_rand, bundle.test, k)
        fit_rows.append(
            {
                "method": "Rand",
                "K": int(k),
                "fit_time": 0.0,
                "train_samples": 0,
                "val_samples": 0,
                "test_samples": len(bundle.test),
                "epochs": 0,
                "batch_size": 0,
                "notes": f"{args.rand_trials} random projector draws; no learning",
            }
        )

        print("Building PCA projection (2025 style)...")
        t_fit = time.perf_counter()
        P_pca = pca_projection_from_training(bundle.train, k)
        pca_fit_time = time.perf_counter() - t_fit
        append_fixed_projection_rows(rows, "PCA", P_pca, bundle.test, k)
        fit_rows.append(
            {
                "method": "PCA",
                "K": int(k),
                "fit_time": pca_fit_time,
                "train_samples": len(bundle.train),
                "val_samples": 0,
                "test_samples": len(bundle.test),
                "epochs": 1,
                "batch_size": len(bundle.train),
                "notes": "SVD on training optimal solutions",
            }
        )

        if not args.skip_pca2024:
            print("Building PCA2024 projection...")
            t_fit = time.perf_counter()
            P_pca_general = raw_pca_projection(bundle.train, k)
            P_pca_general = project_columns_to_common_region(
                P_pca_general,
                general_common_A,
                general_common_b,
                general_feasible_hint,
            )
            pca2024_fit_time = time.perf_counter() - t_fit
            append_general_projection_rows(rows, "PCA2024", P_pca_general, general_test, k)
            fit_rows.append(
                {
                    "method": "PCA2024",
                    "K": int(k),
                    "fit_time": pca2024_fit_time,
                    "train_samples": len(bundle.train),
                    "val_samples": 0,
                    "test_samples": len(bundle.test),
                    "epochs": 1,
                    "batch_size": len(bundle.train),
                    "notes": "Affine PCA on training optimal solutions before general-form projection",
                }
            )

        if not args.skip_sga:
            print("Training SGA_FinalP baseline...")
            t_fit = time.perf_counter()
            P_sga, hist = train_sga_final_projection(
                bundle.train,
                bundle.val,
                k=k,
                epochs=args.sga_epochs,
                batch_size=args.batch_size,
                lr=args.sga_lr,
                seed=args.seed + 5000 + k,
                verbose=args.verbose,
            )
            sga_fit_time = time.perf_counter() - t_fit
            save_training_history(hist, out_dir / f"history_SGA_FinalP_K{k}.csv")
            append_general_projection_rows(rows, "SGA_FinalP", P_sga, general_test, k)
            np.save(out_dir / f"SGA_FinalP_K{k}.npy", P_sga)
            fit_rows.append(
                {
                    "method": "SGA_FinalP",
                    "K": int(k),
                    "fit_time": sga_fit_time,
                    "train_samples": len(bundle.train),
                    "val_samples": len(bundle.val),
                    "test_samples": len(bundle.test),
                    "epochs": int(args.sga_epochs),
                    "batch_size": int(args.batch_size),
                    "notes": "Shared projection matrix with SGD-style updates plus final column projection",
                }
            )

        if args.run_sharedp:
            print("Training SharedP...")
            shared = SharedProjection(problem.n_vars, k)
            t_fit = time.perf_counter()
            shared, hist = train_implicit_projection_model(
                shared,
                bundle.train,
                bundle.val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=args.seed + 1000 + k,
                patience=args.patience,
                verbose=args.verbose,
            )
            shared_fit_time = time.perf_counter() - t_fit
            save_training_history(hist, out_dir / f"history_SharedP_K{k}.csv")
            append_learned_projector_rows(rows, "SharedP", shared, bundle.test, k, device)
            torch.save(shared.state_dict(), out_dir / f"SharedP_K{k}.pt")
            fit_rows.append(
                {
                    "method": "SharedP",
                    "K": int(k),
                    "fit_time": shared_fit_time,
                    "train_samples": len(bundle.train),
                    "val_samples": len(bundle.val),
                    "test_samples": len(bundle.test),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "notes": "2025 shared softmax projector trained with implicit gradients",
                }
            )

        if args.run_fcnn:
            print("Training FCNN...")
            n_cons_eff = bundle.train[0].n_feature_cons
            fcnn = FCNNProjectionNet(problem.n_vars, n_cons_eff, k, args.hidden_dim)
            t_fit = time.perf_counter()
            fcnn, hist = train_implicit_projection_model(
                fcnn,
                bundle.train,
                bundle.val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=args.seed + 2000 + k,
                patience=args.patience,
                verbose=args.verbose,
            )
            fcnn_fit_time = time.perf_counter() - t_fit
            save_training_history(hist, out_dir / f"history_FCNN_K{k}.csv")
            append_learned_projector_rows(rows, "FCNN", fcnn, bundle.test, k, device)
            torch.save(fcnn.state_dict(), out_dir / f"FCNN_K{k}.pt")
            fit_rows.append(
                {
                    "method": "FCNN",
                    "K": int(k),
                    "fit_time": fcnn_fit_time,
                    "train_samples": len(bundle.train),
                    "val_samples": len(bundle.val),
                    "test_samples": len(bundle.test),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "notes": "2025 non-equivariant neural projector trained with implicit gradients",
                }
            )

        if not args.skip_pelp:
            print("Training PELP_NN...")
            pelp = PELPProjectionNet(
                k=k,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_pelp_layers,
                generator_hidden_dim=args.generator_hidden_dim,
            )
            t_fit = time.perf_counter()
            pelp, hist = train_implicit_projection_model(
                pelp,
                bundle.train,
                bundle.val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=args.seed + 3000 + k,
                patience=args.patience,
                verbose=args.verbose,
            )
            pelp_fit_time = time.perf_counter() - t_fit
            save_training_history(hist, out_dir / f"history_PELP_NN_K{k}.csv")
            append_learned_projector_rows(rows, "PELP_NN", pelp, bundle.test, k, device)
            torch.save(pelp.state_dict(), out_dir / f"PELP_NN_K{k}.pt")
            fit_rows.append(
                {
                    "method": "PELP_NN",
                    "K": int(k),
                    "fit_time": pelp_fit_time,
                    "train_samples": len(bundle.train),
                    "val_samples": len(bundle.val),
                    "test_samples": len(bundle.test),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "notes": "2025 permutation-equivariant neural projector trained with implicit gradients",
                }
            )

    raw = pd.DataFrame(rows)
    raw_for_summary = raw.copy()
    raw_for_summary["method"] = raw_for_summary["method"].str.replace(r"Rand#\d+", "Rand", regex=True)
    summary = summarize_ratios_and_times(raw_for_summary)
    anchor_x = detect_anchor_for_capture(bundle.truth)
    capture_summary = summarize_improvement_capture(raw_for_summary, bundle.test, anchor_x)
    raw.to_csv(out_dir / "raw_results.csv", index=False)
    summary.to_csv(out_dir / "summary_results.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(out_dir / "fit_times.csv", index=False)
    if not capture_summary.empty:
        capture_summary.to_csv(out_dir / "improvement_capture_summary.csv", index=False)
        quality_summary = summary[["method", "K", "objective_ratio_mean", "success_rate", "n"]].merge(
            capture_summary[["method", "K", "improvement_capture"]],
            on=["method", "K"],
            how="left",
        )
    else:
        quality_summary = summary[["method", "K", "objective_ratio_mean", "success_rate", "n"]].copy()
    quality_summary.to_csv(out_dir / "quality_summary.csv", index=False)
    if args.make_plots:
        plot_results(summary, out_dir)

    print("\nSummary:")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 20, "display.width", 160):
        print(summary.sort_values(["K", "method"]))
    if not capture_summary.empty:
        print("\nImprovement capture:")
        with pd.option_context("display.max_rows", 200, "display.max_columns", 20, "display.width", 160):
            print(capture_summary.sort_values(["K", "method"]))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
