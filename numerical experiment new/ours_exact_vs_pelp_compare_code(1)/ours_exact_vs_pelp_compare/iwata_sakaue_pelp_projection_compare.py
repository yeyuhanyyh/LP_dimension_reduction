#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iwata--Sakaue ICML 2025 PELP neural projection baseline for LPs.

This script implements the permutation-equivariant / invariant neural network
from:
    Tomoharu Iwata and Shinsaku Sakaue,
    "Learning to Generate Projections for Reducing Dimensionality of
    Heterogeneous Linear Programming Problems", ICML 2025.

The model takes an inequality-form LP instance

    max_x c^T x    s.t. A x <= b, x >= 0,

and generates an instance-dependent nonnegative projection matrix P in R^{N x K}.
The reduced LP is

    max_y c^T P y  s.t. A P y <= b, y >= 0,

and the recovered solution is x = P y.  The nonnegativity of P is obtained by a
column-wise softmax, matching Appendix D.1 of Iwata--Sakaue.

Implemented comparisons:
    Full        : solve the original LP.
    Rand        : column-randomized / variable-selection projection.
    PCA         : shared PCA projection from full training optima, with negative
                  entries clipped to zero and columns normalized.
    SharedP     : shared trainable projection matrix learned by the same implicit
                  gradient rule, similar to Sakaue--Oki 2024 / Iwata--Sakaue.
    PELP_NN     : the instance-dependent neural projection model.
    OursExact   : optional hook for a decision-sufficient exact affine reduction
                  x = x_anchor + U z.  Provide --ours_npz containing U and x_anchor.

Default settings are deliberately small for a quick smoke test.  Use
--paper_settings to switch to the Packing dimensions/splits in the ICML paper:
N=500 variables, M=50 constraints, 425 train / 25 validation / 50 test,
K in {10,20,30,40,50}, 500 epochs, batch size 8.

Dependencies:
    numpy, scipy, pandas, matplotlib, torch

Example quick run:
    python iwata_sakaue_pelp_projection_compare.py --quick --out_dir pelp_quick

Paper-style packing run:
    python iwata_sakaue_pelp_projection_compare.py --paper_settings --out_dir pelp_packing

Comparison with our exact reduction, when an external Stage-I routine has learned U:
    python iwata_sakaue_pelp_projection_compare.py --quick --ours_npz ours_stageI.npz

The .npz file for --ours_npz must contain:
    U          : N x r matrix
    x_anchor   : N-vector feasible anchor
For fixed-feasible-region LPs, use --packing_mode fixed_feasible.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gzip
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires PyTorch. Install with e.g. `pip install torch`. "
        f"Import failed: {exc}"
    )

plt = None  # matplotlib is imported lazily only when --make_plots is used.


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LPInstance:
    """LP instance: max c^T x s.t. A x <= b, Aeq x = beq, x >= 0.

    The original Iwata--Sakaue papers focus on inequality-form LPs with x >= 0.
    For comparison with flow and path problems, we also allow optional equality
    constraints while keeping the same nonnegative-variable projection model.
    """

    c: np.ndarray  # (N,)
    A: np.ndarray  # (M_ub,N), may be empty with shape (0,N)
    b: np.ndarray  # (M_ub,)
    A_eq: Optional[np.ndarray] = None  # (M_eq,N)
    b_eq: Optional[np.ndarray] = None  # (M_eq,)
    objective_constant: float = 0.0
    name: str = ""
    var_bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None
    full_x: Optional[np.ndarray] = None
    full_obj: Optional[float] = None
    full_time: Optional[float] = None

    @property
    def n_vars(self) -> int:
        return int(self.c.shape[0])

    @property
    def n_cons(self) -> int:
        return int(self.b.shape[0])

    @property
    def n_cons_eq(self) -> int:
        return 0 if self.A_eq is None else int(self.A_eq.shape[0])

    @property
    def n_feature_cons(self) -> int:
        return int(self.n_cons + 2 * self.n_cons_eq)


@dataclasses.dataclass
class LPSolveResult:
    x: Optional[np.ndarray]
    obj: float
    success: bool
    runtime: float
    dual_ub_for_max: Optional[np.ndarray] = None
    dual_eq_for_max: Optional[np.ndarray] = None
    message: str = ""


# ---------------------------------------------------------------------------
# LP solvers: SciPy HiGHS backend
# ---------------------------------------------------------------------------


def _linprog_options() -> Dict[str, object]:
    # HiGHS is the SciPy default and returns dual sensitivity information.
    return {"presolve": True}


def _as_optional_array(mat: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mat is None:
        return None
    arr = np.asarray(mat, dtype=float)
    if arr.size == 0:
        return None
    return arr


def _objective_scale(vec: np.ndarray) -> float:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    return max(1.0, float(np.linalg.norm(arr, ord=np.inf)))


def _linprog_with_fallback(
    *,
    c: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
    bounds,
):
    last = None
    for method in ("highs", "highs-ipm", "highs-ds"):
        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=method,
            options=_linprog_options(),
        )
        last = res
        if res.success:
            return res
    return last


def _extract_duals_for_max(res, n_ub: int, n_eq: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert SciPy minimization marginals to max-LP multipliers.

    We solve maximization LPs as minimization of the negated objective.  HiGHS
    returns marginals for the minimization convention, so the corresponding
    maximization multipliers are their negatives.
    """
    lam = None
    nu = None
    if n_ub > 0:
        try:
            lam = -np.asarray(res.ineqlin.marginals, dtype=float)
            lam = np.maximum(lam, 0.0)
        except Exception:
            lam = np.zeros(n_ub, dtype=float)
    if n_eq > 0:
        try:
            nu = -np.asarray(res.eqlin.marginals, dtype=float)
        except Exception:
            nu = np.zeros(n_eq, dtype=float)
    return lam, nu


def _projected_lp_eq_tolerance(beq: Optional[np.ndarray]) -> float:
    if beq is None:
        return 0.0
    arr = np.asarray(beq, dtype=float).reshape(-1)
    scale = max(1.0, float(np.linalg.norm(arr, ord=np.inf)))
    return 1e-7 * scale


def solve_full_lp_max(inst: LPInstance) -> LPSolveResult:
    """Solve max c^T x subject to A x <= b, Aeq x = beq, x >= 0."""
    t0 = time.perf_counter()
    n = inst.n_vars
    Aub = _as_optional_array(inst.A)
    bub = None if Aub is None else np.asarray(inst.b, dtype=float)
    Aeq = _as_optional_array(inst.A_eq)
    beq = None if Aeq is None else np.asarray(inst.b_eq, dtype=float)
    c_vec = np.asarray(inst.c, dtype=float)
    obj_scale = _objective_scale(c_vec)
    bounds = inst.var_bounds
    if bounds is None:
        bounds = [(0.0, None)] * n
    res = _linprog_with_fallback(
        c=-(c_vec / obj_scale),
        A_ub=Aub,
        b_ub=bub,
        A_eq=Aeq,
        b_eq=beq,
        bounds=bounds,
    )
    if not res.success and Aeq is not None:
        eq_tol = _projected_lp_eq_tolerance(beq)
        if eq_tol > 0.0:
            eq_aug = np.vstack([Aeq, -Aeq])
            rhs_aug = np.concatenate([beq + eq_tol, -beq + eq_tol])
            Aub_relaxed = eq_aug if Aub is None else np.vstack([Aub, eq_aug])
            bub_relaxed = rhs_aug if bub is None else np.concatenate([bub, rhs_aug])
            res = _linprog_with_fallback(
                c=-(c_vec / obj_scale),
                A_ub=Aub_relaxed,
                b_ub=bub_relaxed,
                A_eq=None,
                b_eq=None,
                bounds=bounds,
            )
    if not res.success:
        try:
            res = linprog(
                c=-(c_vec / obj_scale),
                A_ub=Aub,
                b_ub=bub,
                A_eq=Aeq,
                b_eq=beq,
                bounds=bounds,
                method="interior-point",
                options={"presolve": True},
            )
        except Exception:
            pass
    runtime = time.perf_counter() - t0
    if not res.success:
        return LPSolveResult(None, float("nan"), False, runtime, None, None, res.message)
    x = np.asarray(res.x, dtype=float)
    lam, nu = _extract_duals_for_max(res, inst.n_cons, inst.n_cons_eq)
    return LPSolveResult(x, float(inst.objective_constant + inst.c @ x), True, runtime, lam, nu, res.message)


def solve_projected_lp_max(
    inst: LPInstance,
    P: np.ndarray,
    nonnegative_y: bool = True,
) -> LPSolveResult:
    """Solve max_y c^T P y s.t. A P y <= b, Aeq P y = beq.

    Returns the recovered x = P y.  The dual_ub_for_max is the nonnegative
    multiplier lambda for A P y <= b in the maximization KKT convention.
    SciPy solves minimization of -q^T y; its inequality marginals have the
    opposite sign of the max-LP multipliers, so lambda = -marginals.
    """
    t0 = time.perf_counter()
    P = np.asarray(P, dtype=float)
    k = P.shape[1]
    Aub = _as_optional_array(inst.A)
    Aeq_inst = _as_optional_array(inst.A_eq)
    G = None if Aub is None else Aub @ P
    Heq = None if Aeq_inst is None else Aeq_inst @ P
    q = P.T @ np.asarray(inst.c, dtype=float)
    obj_scale = _objective_scale(q)
    bounds = [(0.0, None)] * k if nonnegative_y else [(None, None)] * k
    eq_tol = _projected_lp_eq_tolerance(None if Heq is None else inst.b_eq)
    G_aug = G
    b_aug = None if G is None else np.asarray(inst.b, dtype=float)
    use_eq_as_ineq = Heq is not None and eq_tol > 0.0
    if use_eq_as_ineq:
        beq = np.asarray(inst.b_eq, dtype=float)
        H_blocks = [Heq, -Heq]
        h_rhs = [beq + eq_tol, -beq + eq_tol]
        G_aug = np.vstack([blk for blk in ([G] if G is not None else []) + H_blocks])
        b_aug = np.concatenate([rhs for rhs in ([np.asarray(inst.b, dtype=float)] if G is not None else []) + h_rhs])
    res = _linprog_with_fallback(
        c=-(q / obj_scale),
        A_ub=G_aug,
        b_ub=b_aug,
        A_eq=None if use_eq_as_ineq else Heq,
        b_eq=None if Heq is None or use_eq_as_ineq else np.asarray(inst.b_eq, dtype=float),
        bounds=bounds,
    )
    runtime = time.perf_counter() - t0
    if not res.success:
        return LPSolveResult(None, float("nan"), False, runtime, None, None, res.message)
    y = np.asarray(res.x, dtype=float)
    x = P @ y
    obj = float(inst.objective_constant + inst.c @ x)
    if use_eq_as_ineq:
        lam_aug, _ = _extract_duals_for_max(res, inst.n_cons + 2 * inst.n_cons_eq, 0)
        if lam_aug is None:
            lam = np.zeros(inst.n_cons, dtype=float) if inst.n_cons > 0 else None
            nu = np.zeros(inst.n_cons_eq, dtype=float) if inst.n_cons_eq > 0 else None
        else:
            lam = lam_aug[: inst.n_cons] if inst.n_cons > 0 else None
            if inst.n_cons_eq > 0:
                lam_pos = lam_aug[inst.n_cons : inst.n_cons + inst.n_cons_eq]
                lam_neg = lam_aug[inst.n_cons + inst.n_cons_eq :]
                nu = lam_pos - lam_neg
            else:
                nu = None
    else:
        lam, nu = _extract_duals_for_max(res, inst.n_cons, inst.n_cons_eq)
    return LPSolveResult(x, obj, True, runtime, lam, nu, res.message)


def solve_affine_exact_reduction_max(
    inst: LPInstance, U: np.ndarray, x_anchor: np.ndarray
) -> LPSolveResult:
    """Evaluate our exact affine reduction x = x_anchor + U z.

    Solves
        max_z c^T (x_anchor + U z)
        s.t. A(x_anchor + U z) <= b, x_anchor + U z >= 0.

    This is exact for the user's method when span(U)=W* and x_anchor is a
    reachable optimal solution; otherwise it is just a reduced affine baseline.
    """
    t0 = time.perf_counter()
    U = np.asarray(U, dtype=float)
    x0 = np.asarray(x_anchor, dtype=float).reshape(-1)
    r = U.shape[1]
    if r == 0:
        Aub = _as_optional_array(inst.A)
        Aeq = _as_optional_array(inst.A_eq)
        feas_ub = True if Aub is None else np.all(Aub @ x0 <= inst.b + 1e-7)
        feas_eq = True if Aeq is None else np.allclose(Aeq @ x0, inst.b_eq, atol=1e-7)
        feas = feas_ub and feas_eq and np.all(x0 >= -1e-7)
        obj = float(inst.objective_constant + inst.c @ x0) if feas else float("nan")
        return LPSolveResult(x0 if feas else None, obj, feas, time.perf_counter() - t0)

    Aub = _as_optional_array(inst.A)
    Aeq_inst = _as_optional_array(inst.A_eq)
    ub_blocks = []
    ub_rhs = []
    if Aub is not None:
        ub_blocks.append(Aub @ U)
        ub_rhs.append(np.asarray(inst.b, dtype=float) - Aub @ x0)
    ub_blocks.append(-U)
    ub_rhs.append(x0)
    A_ub = np.vstack(ub_blocks)
    b_ub = np.concatenate(ub_rhs)
    A_eq = None if Aeq_inst is None else Aeq_inst @ U
    b_eq = None if Aeq_inst is None else np.asarray(inst.b_eq, dtype=float) - Aeq_inst @ x0
    f = -(U.T @ inst.c)
    res = linprog(
        c=f,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(None, None)] * r,
        method="highs",
        options=_linprog_options(),
    )
    runtime = time.perf_counter() - t0
    if not res.success:
        return LPSolveResult(None, float("nan"), False, runtime, None, None, res.message)
    z = np.asarray(res.x, dtype=float)
    x = x0 + U @ z
    lam, nu = _extract_duals_for_max(res, A_ub.shape[0], 0 if A_eq is None else A_eq.shape[0])
    return LPSolveResult(x, float(inst.objective_constant + inst.c @ x), True, runtime, lam, nu, res.message)


def ensure_full_solutions(instances: Sequence[LPInstance], force: bool = False) -> None:
    """Fill full_x, full_obj, and full_time in-place."""
    for i, inst in enumerate(instances):
        if inst.full_obj is not None and not force:
            continue
        sol = solve_full_lp_max(inst)
        if not sol.success:
            raise RuntimeError(f"Full LP solve failed for {inst.name or i}: {sol.message}")
        inst.full_x = sol.x
        inst.full_obj = sol.obj
        inst.full_time = sol.runtime


# ---------------------------------------------------------------------------
# Synthetic data: packing LPs, matching the paper's first dataset family
# ---------------------------------------------------------------------------


def generate_packing_instances(
    n_instances: int,
    n_vars: int,
    n_cons: int,
    rng: np.random.Generator,
    mode: str = "heterogeneous",
) -> List[LPInstance]:
    """Generate Packing LPs used by Iwata--Sakaue.

    Heterogeneous mode: every instance has fresh c,A,b, as in ICML 2025.
    Fixed-feasible mode: A,b are shared and only c varies; this is the regime in
    which the user's exact decision-sufficient method is directly comparable.
    """
    if mode not in {"heterogeneous", "fixed_feasible"}:
        raise ValueError("mode must be 'heterogeneous' or 'fixed_feasible'")
    instances: List[LPInstance] = []
    if mode == "fixed_feasible":
        A0 = rng.uniform(0.0, 1.0, size=(n_cons, n_vars))
        b0 = n_vars * rng.uniform(0.0, 1.0, size=n_cons)
    else:
        A0 = None
        b0 = None

    for t in range(n_instances):
        c = rng.uniform(0.0, 1.0, size=n_vars)
        if mode == "fixed_feasible":
            A = A0.copy()
            b = b0.copy()
        else:
            A = rng.uniform(0.0, 1.0, size=(n_cons, n_vars))
            b = n_vars * rng.uniform(0.0, 1.0, size=n_cons)
        instances.append(LPInstance(c=c, A=A, b=b, name=f"packing_{t:05d}"))
    return instances


def make_node_arc_incidence(n_nodes: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    B = np.zeros((n_nodes, len(edges)), dtype=float)
    for j, (u, v) in enumerate(edges):
        B[u, j] = 1.0
        B[v, j] = -1.0
    return B


def random_dag_edges(
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
        raise ValueError("invalid number of DAG edges")
    remain = [e for e in candidates if e not in mandatory]
    extra = rng.choice(len(remain), size=n_edges - len(mandatory), replace=False)
    edges = list(mandatory) + [remain[int(i)] for i in extra]
    edges.sort()
    return edges


def reward_from_costs_via_potential(
    edges: Sequence[Tuple[int, int]],
    costs: np.ndarray,
    n_nodes: int,
    scale: float = 4.0,
) -> np.ndarray:
    """Convert a min-cost unit-flow objective into an equivalent max-reward one.

    For any feasible unit flow Bx=d, the term (B^T p)^T x = p^T d is constant,
    so maximizing (B^T p - cost)^T x is equivalent to minimizing cost^T x.
    """
    potential = scale * np.arange(n_nodes - 1, -1, -1, dtype=float)
    rewards = np.zeros(len(edges), dtype=float)
    for j, (u, v) in enumerate(edges):
        rewards[j] = potential[u] - potential[v] - float(costs[j])
    return rewards


def nullspace_basis(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    u, s, vh = np.linalg.svd(np.asarray(A, dtype=float), full_matrices=True)
    rank = int(np.sum(s > tol * max(1.0, s[0] if len(s) else 1.0)))
    return vh[rank:, :].T.copy()


def transform_fixed_feasible_problem(
    reward: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    x0: np.ndarray,
    name: str,
    A_ineq: Optional[np.ndarray] = None,
    b_ineq: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
) -> LPInstance:
    """Bridge fixed-feasible flow/path LPs to the zero-feasible projection form.

    Original problem:
        max reward^T z  s.t. A_ineq z <= b_ineq, A_eq z = b_eq, 0 <= z <= ub.

    Following Appendix C of Sakaue--Oki (2024), we remove the equalities via
    z = x0 + L x, where L = I - A_eq^\\dagger A_eq projects onto ker(A_eq).
    This yields an equivalent inequality-form LP with x = 0 feasible while
    preserving the ambient dimension, which is substantially more numerically
    stable than an explicit nullspace basis / sign-split parametrization.
    """
    reward = np.asarray(reward, dtype=float).reshape(-1)
    A_eq = np.asarray(A_eq, dtype=float)
    b_eq = np.asarray(b_eq, dtype=float).reshape(-1)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    A_ineq = _as_optional_array(A_ineq)
    b_ineq = None if A_ineq is None else np.asarray(b_ineq, dtype=float).reshape(-1)
    if ub is not None:
        ub = np.asarray(ub, dtype=float).reshape(-1)
    if np.linalg.norm(A_eq @ x0 - b_eq) > 1e-7:
        raise ValueError("x0 is not feasible for the equality constraints")
    if np.any(x0 < -1e-9):
        raise ValueError("x0 violates the nonnegativity bounds")
    if ub is not None and np.any(x0 > ub + 1e-9):
        raise ValueError("x0 violates the upper bounds")
    if A_ineq is not None and np.any(A_ineq @ x0 > b_ineq + 1e-7):
        raise ValueError("x0 violates the inequality constraints")

    n = reward.shape[0]
    if A_eq.size == 0:
        L = np.eye(n, dtype=float)
    else:
        L = np.eye(n, dtype=float) - np.linalg.pinv(A_eq) @ A_eq
    ub_blocks = []
    ub_rhs = []
    if A_ineq is not None:
        ub_blocks.append(A_ineq @ L)
        ub_rhs.append(b_ineq - A_ineq @ x0)
    if ub is not None:
        ub_blocks.append(L)
        ub_rhs.append(ub - x0)
    ub_blocks.append(-L)
    ub_rhs.append(x0)
    Aub = np.vstack(ub_blocks)
    bub = np.concatenate(ub_rhs)
    c_new = L.T @ reward
    inst = LPInstance(
        c=c_new,
        A=Aub,
        b=bub,
        objective_constant=float(reward @ x0),
        name=name,
        var_bounds=[(None, None)] * n,
    )
    inst.transform_L = L
    inst.transform_x0 = x0
    inst.original_reward = reward
    return inst


def generate_maxflow_instances(
    n_instances: int,
    n_nodes: int,
    n_edges: int,
    rng: np.random.Generator,
    mode: str = "heterogeneous",
) -> List[LPInstance]:
    """Generate max-flow LPs and bridge them to the projection-friendly form."""
    if mode not in {"heterogeneous", "fixed_feasible"}:
        raise ValueError("mode must be 'heterogeneous' or 'fixed_feasible'")
    source = 0
    sink = n_nodes - 1
    demand = np.zeros(n_nodes, dtype=float)
    demand[source] = 1.0
    demand[sink] = -1.0
    base_edges = random_dag_edges(n_nodes, n_edges, rng, source, sink)
    base_B = make_node_arc_incidence(n_nodes, base_edges)
    base_caps = rng.uniform(0.5, 1.5, size=n_edges)

    instances: List[LPInstance] = []
    for t in range(n_instances):
        if mode == "heterogeneous":
            edges = random_dag_edges(n_nodes, n_edges, rng, source, sink)
            B = make_node_arc_incidence(n_nodes, edges)
            caps = rng.uniform(0.5, 1.5, size=n_edges)
        else:
            edges = base_edges
            B = base_B
            caps = base_caps
        Aeq = np.hstack([B, -demand.reshape(-1, 1)])
        Aub = np.hstack([np.eye(n_edges), np.zeros((n_edges, 1))])
        reward = np.zeros(n_edges + 1, dtype=float)
        reward[-1] = 1.0
        instances.append(
            transform_fixed_feasible_problem(
                reward,
                Aeq,
                np.zeros(n_nodes, dtype=float),
                np.zeros(n_edges + 1, dtype=float),
                name=f"maxflow_{t:05d}",
                A_ineq=Aub,
                b_ineq=caps,
            )
        )
    return instances


def generate_mincostflow_instances(
    n_instances: int,
    n_nodes: int,
    n_edges: int,
    rng: np.random.Generator,
    mode: str = "heterogeneous",
) -> List[LPInstance]:
    """Generate unit min-cost-flow LPs, evaluated through an equivalent max reward."""
    if mode not in {"heterogeneous", "fixed_feasible"}:
        raise ValueError("mode must be 'heterogeneous' or 'fixed_feasible'")
    source = 0
    sink = n_nodes - 1
    demand = np.zeros(n_nodes, dtype=float)
    demand[source] = 1.0
    demand[sink] = -1.0
    base_edges = random_dag_edges(n_nodes, n_edges, rng, source, sink)
    base_B = make_node_arc_incidence(n_nodes, base_edges)

    instances: List[LPInstance] = []
    for t in range(n_instances):
        if mode == "heterogeneous":
            edges = random_dag_edges(n_nodes, n_edges, rng, source, sink)
            B = make_node_arc_incidence(n_nodes, edges)
        else:
            edges = base_edges
            B = base_B
        costs = rng.uniform(0.0, 1.0, size=n_edges)
        direct_idx = edges.index((source, sink))
        # Match the paper's "trivially feasible but costly" direct arc.
        costs[direct_idx] = float(max(10.0, n_edges))
        rewards = reward_from_costs_via_potential(edges, costs, n_nodes, scale=4.0)
        x0 = np.zeros(n_edges, dtype=float)
        x0[direct_idx] = 1.0
        instances.append(
            transform_fixed_feasible_problem(
                rewards,
                B,
                demand,
                x0,
                name=f"mincost_{t:05d}",
                ub=np.ones(n_edges, dtype=float),
            )
        )
    return instances


def make_monotone_grid_edges(grid_size: int) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    edges: List[Tuple[int, int]] = []
    for i in range(grid_size):
        for j in range(grid_size):
            u = i * grid_size + j
            if j + 1 < grid_size:
                edges.append((u, i * grid_size + (j + 1)))
            if i + 1 < grid_size:
                edges.append((u, (i + 1) * grid_size + j))
    edge_to_idx = {e: idx for idx, e in enumerate(edges)}
    return edges, edge_to_idx


def generate_shortest_path_instances(
    n_instances: int,
    grid_size: int,
    dstar: int,
    rng: np.random.Generator,
    band_width: int = 1,
    cost_low: float = 1.0,
    cost_high: float = 3.0,
    theta_abs_max: float = 0.45,
) -> List[LPInstance]:
    """Structured repeated-shortest-path family used by the exact-reduction note."""
    if grid_size < 2 * dstar + 2:
        raise ValueError("grid_size is too small for the requested number of gadgets")

    edges, edge_to_idx = make_monotone_grid_edges(grid_size)
    n_nodes = grid_size * grid_size
    B = make_node_arc_incidence(n_nodes, edges)
    demand = np.zeros(n_nodes, dtype=float)
    demand[0] = 1.0
    demand[-1] = -1.0

    costs0 = np.full(len(edges), cost_high, dtype=float)
    for idx, (u, v) in enumerate(edges):
        ui, uj = divmod(u, grid_size)
        vi, vj = divmod(v, grid_size)
        if abs(ui - uj) <= band_width and abs(vi - vj) <= band_width:
            costs0[idx] = cost_low

    gadget_rows = list(range(1, 1 + 2 * dstar, 2))
    gadgets: List[Tuple[int, int, int, int]] = []
    for r in gadget_rows:
        c = r
        e_top = edge_to_idx[(r * grid_size + c, r * grid_size + c + 1)]
        e_right = edge_to_idx[(r * grid_size + c + 1, (r + 1) * grid_size + c + 1)]
        e_left = edge_to_idx[(r * grid_size + c, (r + 1) * grid_size + c)]
        e_bottom = edge_to_idx[((r + 1) * grid_size + c, (r + 1) * grid_size + c + 1)]
        gadgets.append((e_top, e_right, e_left, e_bottom))
        costs0[[e_top, e_right, e_left, e_bottom]] = cost_low

    instances: List[LPInstance] = []
    x0 = np.zeros(len(edges), dtype=float)
    # Canonical monotone path: move right first, then move down.
    cur = 0
    for j in range(grid_size - 1):
        nxt = cur + 1
        x0[edge_to_idx[(cur, nxt)]] = 1.0
        cur = nxt
    for _ in range(grid_size - 1):
        nxt = cur + grid_size
        x0[edge_to_idx[(cur, nxt)]] = 1.0
        cur = nxt
    for t in range(n_instances):
        theta = rng.uniform(-theta_abs_max, theta_abs_max, size=dstar)
        costs = costs0.copy()
        for k, (e_top, e_right, e_left, e_bottom) in enumerate(gadgets):
            costs[e_top] -= 0.5 * theta[k]
            costs[e_right] -= 0.5 * theta[k]
            costs[e_left] += 0.5 * theta[k]
            costs[e_bottom] += 0.5 * theta[k]
        rewards = reward_from_costs_via_potential(edges, costs, n_nodes, scale=6.0)
        inst = transform_fixed_feasible_problem(
            rewards,
            B,
            demand,
            x0,
            name=f"spath_{t:05d}",
            ub=np.ones(len(edges), dtype=float),
        )
        inst.theta = theta
        inst.gadgets = np.asarray(gadgets, dtype=int)
        instances.append(inst)
    return instances


def parse_plain_mps(path: str) -> Tuple[str, str, List[str], List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a plain-text MPS file into objective, rows, matrix, rhs, and bounds.

    This light-weight parser is sufficient for the Netlib LP cases used in the
    Sakaue/Oki paper and is more robust than depending on an external parser on
    Windows text encodings.
    """
    name = Path(path).stem
    section = None
    objective_row = None
    row_names: List[str] = []
    row_types: List[str] = []
    row_index: Dict[str, int] = {}
    col_names: List[str] = []
    col_index: Dict[str, int] = {}
    column_entries: List[Tuple[int, int, float]] = []
    obj_entries: Dict[int, float] = {}
    rhs_map: Dict[str, float] = {}
    lo_map: Dict[str, float] = {}
    up_map: Dict[str, float] = {}

    with open(path, "r", encoding="latin-1", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line or line.startswith("*"):
                continue
            tokens = line.split()
            if not tokens:
                continue
            head = tokens[0].upper()
            if head == "NAME":
                section = head
                if len(tokens) > 1:
                    name = tokens[1]
                continue
            if len(tokens) == 1 and head in {"ROWS", "COLUMNS", "RHS", "BOUNDS", "RANGES", "ENDATA"}:
                section = head
                if head == "ENDATA":
                    break
                continue

            if section == "ROWS":
                rtype = tokens[0].upper()
                rname = tokens[1]
                if rtype == "N":
                    objective_row = rname
                else:
                    row_index[rname] = len(row_names)
                    row_names.append(rname)
                    row_types.append(rtype)
            elif section == "COLUMNS":
                if len(tokens) > 1 and tokens[1] == "'MARKER'":
                    continue
                cname = tokens[0]
                if cname not in col_index:
                    col_index[cname] = len(col_names)
                    col_names.append(cname)
                j = col_index[cname]
                for idx in range(1, len(tokens), 2):
                    if idx + 1 >= len(tokens):
                        break
                    rname = tokens[idx]
                    val = float(tokens[idx + 1])
                    if objective_row is not None and rname == objective_row:
                        obj_entries[j] = val
                    elif rname in row_index:
                        column_entries.append((row_index[rname], j, val))
            elif section == "RHS":
                start = 0
                if len(tokens) >= 3 and tokens[0] not in row_index:
                    start = 1
                for idx in range(start, len(tokens), 2):
                    if idx + 1 >= len(tokens):
                        break
                    rname = tokens[idx]
                    val = float(tokens[idx + 1])
                    if rname in row_index:
                        rhs_map[rname] = val
            elif section == "BOUNDS":
                if len(tokens) < 3:
                    continue
                btype = tokens[0].upper()
                cname = tokens[2]
                val = float(tokens[3]) if len(tokens) > 3 else 0.0
                if btype == "LO":
                    lo_map[cname] = val
                elif btype == "UP":
                    up_map[cname] = val
                elif btype == "FX":
                    lo_map[cname] = val
                    up_map[cname] = val
                elif btype == "FR":
                    lo_map[cname] = -np.inf
                    up_map[cname] = np.inf
                elif btype == "MI":
                    lo_map[cname] = -np.inf
                elif btype == "PL":
                    up_map[cname] = np.inf
                elif btype == "BV":
                    lo_map[cname] = 0.0
                    up_map[cname] = 1.0
                elif btype == "LI":
                    lo_map[cname] = val
                elif btype == "UI":
                    up_map[cname] = val

    m = len(row_names)
    n = len(col_names)
    A = np.zeros((m, n), dtype=float)
    for i, j, val in column_entries:
        A[i, j] = val
    c = np.zeros(n, dtype=float)
    for j, val in obj_entries.items():
        c[j] = val
    rhs = np.zeros(m, dtype=float)
    for rname, val in rhs_map.items():
        rhs[row_index[rname]] = val
    lo = np.zeros(n, dtype=float)
    up = np.full(n, np.inf, dtype=float)
    for cname, val in lo_map.items():
        if cname in col_index:
            lo[col_index[cname]] = val
    for cname, val in up_map.items():
        if cname in col_index:
            up[col_index[cname]] = val

    return name, objective_row or "OBJ", row_names, row_types, c, A, rhs, np.vstack([lo, up])


def load_mps_fixed_feasible_data(mps_path: str) -> Dict[str, np.ndarray]:
    """Load a Netlib-style MPS file and prepare a fixed-feasible origin-form bridge."""
    src = Path(mps_path)
    tmp_path: Optional[str] = None
    load_path = str(src)
    if src.suffix.lower() == ".gz":
        with gzip.open(src, "rt", encoding="utf-8", errors="ignore") as fin, tempfile.NamedTemporaryFile(
            "w", suffix=".mps", delete=False, encoding="utf-8"
        ) as fout:
            fout.write(fin.read())
            tmp_path = fout.name
            load_path = tmp_path

    try:
        name, _obj_name, row_names, row_types, c, A, rhs, bounds = parse_plain_mps(load_path)
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    A = np.asarray(A, dtype=float)
    c = np.asarray(c, dtype=float).reshape(-1)
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    lo = np.asarray(bounds[0], dtype=float).reshape(-1)
    up = np.asarray(bounds[1], dtype=float).reshape(-1)
    if lo.shape[0] != c.shape[0] or up.shape[0] != c.shape[0]:
        raise ValueError("MPS bounds shape mismatch.")

    # Convert general bounds to nonnegative variables:
    # finite lower bounds are shifted away, upper-only variables are flipped, and
    # free variables are split into positive/negative parts.
    rhs_trans = rhs.copy()
    transformed_cols: List[np.ndarray] = []
    transformed_c: List[float] = []
    transformed_up: List[float] = []
    for j in range(c.shape[0]):
        col = A[:, j]
        lo_j = float(lo[j])
        up_j = float(up[j])
        if np.isfinite(lo_j):
            rhs_trans = rhs_trans - col * lo_j
            transformed_cols.append(col.copy())
            transformed_c.append(float(c[j]))
            if np.isfinite(up_j):
                width = up_j - lo_j
                if width < -1e-8:
                    raise ValueError(f"Inconsistent bounds on variable {j}: [{lo_j}, {up_j}]")
                transformed_up.append(float(max(width, 0.0)))
            else:
                transformed_up.append(np.inf)
        elif np.isfinite(up_j):
            rhs_trans = rhs_trans - col * up_j
            transformed_cols.append((-col).copy())
            transformed_c.append(float(-c[j]))
            transformed_up.append(np.inf)
        else:
            transformed_cols.append(col.copy())
            transformed_c.append(float(c[j]))
            transformed_up.append(np.inf)
            transformed_cols.append((-col).copy())
            transformed_c.append(float(-c[j]))
            transformed_up.append(np.inf)

    A = np.column_stack(transformed_cols) if transformed_cols else np.zeros((A.shape[0], 0), dtype=float)
    c = np.asarray(transformed_c, dtype=float)
    rhs = rhs_trans
    up = np.asarray(transformed_up, dtype=float)
    n = c.shape[0]
    finite_up = np.isfinite(up)

    A_ineq_rows: List[np.ndarray] = []
    b_ineq_rows: List[float] = []
    A_eq_rows: List[np.ndarray] = []
    b_eq_rows: List[float] = []
    for i, rtype in enumerate(row_types):
        row = A[i].reshape(-1)
        if rtype == "L":
            A_ineq_rows.append(row)
            b_ineq_rows.append(float(rhs[i]))
        elif rtype == "G":
            A_ineq_rows.append(-row)
            b_ineq_rows.append(float(-rhs[i]))
        elif rtype == "E":
            A_eq_rows.append(row)
            b_eq_rows.append(float(rhs[i]))
    for j in np.flatnonzero(finite_up):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        A_ineq_rows.append(e)
        b_ineq_rows.append(float(up[j]))

    A_ineq = np.vstack(A_ineq_rows) if A_ineq_rows else np.zeros((0, n), dtype=float)
    b_ineq = np.asarray(b_ineq_rows, dtype=float)
    A_eq = np.vstack(A_eq_rows) if A_eq_rows else np.zeros((0, n), dtype=float)
    b_eq = np.asarray(b_eq_rows, dtype=float)

    # Feasible anchor in the shifted original variables.
    feas = linprog(
        c=np.zeros(n, dtype=float),
        A_ub=A_ineq if A_ineq.shape[0] > 0 else None,
        b_ub=b_ineq if A_ineq.shape[0] > 0 else None,
        A_eq=A_eq if A_eq.shape[0] > 0 else None,
        b_eq=b_eq if A_eq.shape[0] > 0 else None,
        bounds=[(0.0, None)] * n,
        method="highs",
        options=_linprog_options(),
    )
    if not feas.success:
        raise RuntimeError(f"failed to find a feasible anchor for {mps_path}: {feas.message}")

    return {
        "name": str(name or src.stem),
        "reward_base": -np.asarray(c, dtype=float).reshape(-1),
        "A_ineq": A_ineq,
        "b_ineq": b_ineq,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "x0": np.asarray(feas.x, dtype=float).reshape(-1),
        "shift": np.zeros(n, dtype=float),
    }


def generate_netlib_perturbed_instances(
    n_instances: int,
    mps_path: str,
    rng: np.random.Generator,
    normal_scale: float = 0.1,
    outlier_scale: float = 1.0,
    outlier_fraction: float = 0.02,
) -> List[LPInstance]:
    """Generate perturbed LP instances from a fixed Netlib MPS base problem."""
    data = load_mps_fixed_feasible_data(mps_path)
    base_reward = data["reward_base"]
    instances: List[LPInstance] = []
    for t in range(n_instances):
        sigma = outlier_scale if rng.random() < outlier_fraction else normal_scale
        noise = 1.0 + sigma * rng.normal(size=base_reward.shape[0])
        reward = base_reward * noise
        inst = transform_fixed_feasible_problem(
            reward,
            data["A_eq"],
            data["b_eq"],
            data["x0"],
            name=f"{data['name']}_{t:05d}",
            A_ineq=data["A_ineq"],
            b_ineq=data["b_ineq"],
        )
        inst.mps_path = str(mps_path)
        instances.append(inst)
    return instances


# ---------------------------------------------------------------------------
# Neural architecture: PELP layer + projection generator
# ---------------------------------------------------------------------------


class PELPLayer(nn.Module):
    """One PELP layer implementing Eqs. (3), (4), and (5).

    Shapes:
        zc: (N, Hin)
        zA: (M, N, Hin)
        zb: (M, Hin)
    Returns:
        zc_next: (N, Hout)
        zA_next: (M, N, Hout)
        zb_next: (M, Hout)
    """

    def __init__(self, in_dim: int, out_dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

        # A embedding update: c_n, A_mn, mean_m A_mn, mean_n A_mn, b_m, bias.
        self.A_from_c = nn.Linear(in_dim, out_dim, bias=False)
        self.A_self = nn.Linear(in_dim, out_dim, bias=False)
        self.A_from_colmean = nn.Linear(in_dim, out_dim, bias=False)
        self.A_from_rowmean = nn.Linear(in_dim, out_dim, bias=False)
        self.A_from_b = nn.Linear(in_dim, out_dim, bias=False)
        self.A_bias = nn.Parameter(torch.zeros(out_dim))

        # b embedding update: mean_n A_mn, b_m, mean_m b_m, bias.
        self.b_from_rowmean_A = nn.Linear(in_dim, out_dim, bias=False)
        self.b_self = nn.Linear(in_dim, out_dim, bias=False)
        self.b_from_global_b = nn.Linear(in_dim, out_dim, bias=False)
        self.b_bias = nn.Parameter(torch.zeros(out_dim))

        # c embedding update: c_n, mean_n c_n, mean_m A_mn, bias.
        self.c_self = nn.Linear(in_dim, out_dim, bias=False)
        self.c_from_global_c = nn.Linear(in_dim, out_dim, bias=False)
        self.c_from_colmean_A = nn.Linear(in_dim, out_dim, bias=False)
        self.c_bias = nn.Parameter(torch.zeros(out_dim))

        self.norm_A = nn.LayerNorm(out_dim)
        self.norm_b = nn.LayerNorm(out_dim)
        self.norm_c = nn.LayerNorm(out_dim)
        self.use_residual = in_dim == out_dim

    def forward(
        self, zc: torch.Tensor, zA: torch.Tensor, zb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mean pooling is the average-pooling operation described in the paper.
        A_col_mean = zA.mean(dim=0)  # (N,Hin), aggregate over constraints m.
        A_row_mean = zA.mean(dim=1)  # (M,Hin), aggregate over variables n.
        c_global = zc.mean(dim=0, keepdim=True)  # (1,Hin)
        b_global = zb.mean(dim=0, keepdim=True)  # (1,Hin)

        # Eq. (3): A_{mn} update.
        A_update = (
            self.A_from_c(zc).unsqueeze(0)
            + self.A_self(zA)
            + self.A_from_colmean(A_col_mean).unsqueeze(0)
            + self.A_from_rowmean(A_row_mean).unsqueeze(1)
            + self.A_from_b(zb).unsqueeze(1)
            + self.A_bias.view(1, 1, -1)
        )
        zA_next = self.act(A_update)
        if self.use_residual:
            zA_next = zA_next + zA
        zA_next = self.norm_A(zA_next)

        # Eq. (4): b_m update.
        b_update = (
            self.b_from_rowmean_A(A_row_mean)
            + self.b_self(zb)
            + self.b_from_global_b(b_global).expand_as(self.b_self(zb))
            + self.b_bias.view(1, -1)
        )
        zb_next = self.act(b_update)
        if self.use_residual:
            zb_next = zb_next + zb
        zb_next = self.norm_b(zb_next)

        # Eq. (5): c_n update.
        c_update = (
            self.c_self(zc)
            + self.c_from_global_c(c_global).expand_as(self.c_self(zc))
            + self.c_from_colmean_A(A_col_mean)
            + self.c_bias.view(1, -1)
        )
        zc_next = self.act(c_update)
        if self.use_residual:
            zc_next = zc_next + zc
        zc_next = self.norm_c(zc_next)

        return zc_next, zA_next, zb_next


class ProjectionGenerator(nn.Module):
    """Shared row-wise generator g([z_c_n, max_m z_A_mn]) -> R^K."""

    def __init__(self, embed_dim: int, k: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, k),
        )

    def forward(self, zc: torch.Tensor, zA: torch.Tensor) -> torch.Tensor:
        # max over constraints for each variable n: shape (N,H)
        A_var_max = zA.max(dim=0).values
        h = torch.cat([zc, A_var_max], dim=-1)
        logits = self.net(h)  # (N,K)
        # Appendix D.1: softmax gives nonnegative projection matrices normalized
        # for each reduced variable, i.e. each column sums to one.
        return torch.softmax(logits, dim=0)


class PELPProjectionNet(nn.Module):
    """Iwata--Sakaue permutation-equivariant neural projection model."""

    def __init__(
        self,
        k: int,
        hidden_dim: int = 32,
        n_layers: int = 4,
        generator_hidden_dim: int = 32,
    ):
        super().__init__()
        dims = [1] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList(
            [PELPLayer(dims[i], dims[i + 1]) for i in range(n_layers)]
        )
        self.generator = ProjectionGenerator(hidden_dim, k, generator_hidden_dim)

    def forward(self, c: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Scalar LP parameters initialize the zeroth-layer embeddings.
        zc = c.reshape(-1, 1)
        zA = A.reshape(A.shape[0], A.shape[1], 1)
        zb = b.reshape(-1, 1)
        for layer in self.layers:
            zc, zA, zb = layer(zc, zA, zb)
        return self.generator(zc, zA)


class SharedProjection(nn.Module):
    """Shared projection matrix baseline trained by the same implicit gradient."""

    def __init__(self, n_vars: int, k: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_vars, k))
        nn.init.normal_(self.logits, mean=0.0, std=0.01)

    def forward(self, c: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return torch.softmax(self.logits, dim=0)


class FCNNProjectionNet(nn.Module):
    """Non-equivariant FCNN baseline for fixed-size LPs.

    This is included for completeness, corresponding to the FCNN comparison in
    the ICML paper.  It cannot handle varying M,N without padding.
    """

    def __init__(self, n_vars: int, n_cons: int, k: int, hidden_dim: int = 32):
        super().__init__()
        self.n_vars = n_vars
        self.n_cons = n_cons
        input_dim = n_vars + n_cons * n_vars + n_cons
        output_dim = n_vars * k
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
        self.k = k

    def forward(self, c: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([c.reshape(-1), A.reshape(-1), b.reshape(-1)], dim=0)
        logits = self.net(flat).reshape(self.n_vars, self.k)
        return torch.softmax(logits, dim=0)


class CostOnlyProjectionNet(nn.Module):
    """Objective-only projector baseline for fixed-polytope LP families."""

    def __init__(self, n_vars: int, k: int, hidden_dim: int = 32):
        super().__init__()
        self.n_vars = int(n_vars)
        self.k = int(k)
        self.net = nn.Sequential(
            nn.Linear(self.n_vars, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.n_vars * self.k),
        )

    def forward(self, c: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        logits = self.net(c.reshape(-1)).reshape(self.n_vars, self.k)
        return torch.softmax(logits, dim=0)


# ---------------------------------------------------------------------------
# Projection construction and implicit-gradient training
# ---------------------------------------------------------------------------


def lp_instance_feature_matrices(inst: LPInstance) -> Tuple[np.ndarray, np.ndarray]:
    """Encode equalities as paired inequalities for the neural input."""
    blocks = []
    rhs = []
    Aub = _as_optional_array(inst.A)
    Aeq = _as_optional_array(inst.A_eq)
    if Aub is not None:
        blocks.append(Aub)
        rhs.append(np.asarray(inst.b, dtype=float))
    if Aeq is not None:
        beq = np.asarray(inst.b_eq, dtype=float)
        blocks.extend([Aeq, -Aeq])
        rhs.extend([beq, -beq])
    if not blocks:
        return np.zeros((0, inst.n_vars), dtype=float), np.zeros(0, dtype=float)
    return np.vstack(blocks), np.concatenate(rhs)


def to_torch_instance(inst: LPInstance, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = torch.as_tensor(inst.c, dtype=torch.float32, device=device)
    A_np, b_np = lp_instance_feature_matrices(inst)
    A = torch.as_tensor(A_np, dtype=torch.float32, device=device)
    b = torch.as_tensor(b_np, dtype=torch.float32, device=device)
    return c, A, b


def implicit_grad_P(
    inst: LPInstance,
    y_x_result: LPSolveResult,
    P: np.ndarray,
    normalize_by_full_obj: bool = True,
) -> np.ndarray:
    """Gradient d u(P,pi) / dP = (c - A^T lambda - Aeq^T nu) y^T.

    We reconstruct y by solving least-squares P y = x when it is not stored.
    In normal training calls, x = P y and P has full column rank in practice.
    """
    if y_x_result.x is None:
        return np.zeros_like(P, dtype=float)
    # Re-solve y from x for a lightweight interface; for exactness, the caller
    # can use solve_projected_y_max below, but this is adequate and robust.
    y, *_ = np.linalg.lstsq(P, y_x_result.x, rcond=None)
    grad_vec = np.asarray(inst.c, dtype=float).copy()
    if y_x_result.dual_ub_for_max is not None and inst.n_cons > 0:
        grad_vec = grad_vec - np.asarray(inst.A, dtype=float).T @ y_x_result.dual_ub_for_max.reshape(-1)
    if y_x_result.dual_eq_for_max is not None and inst.n_cons_eq > 0:
        grad_vec = grad_vec - np.asarray(inst.A_eq, dtype=float).T @ y_x_result.dual_eq_for_max.reshape(-1)
    grad = np.outer(grad_vec, y)
    if normalize_by_full_obj:
        denom = max(abs(float(inst.full_obj or 0.0)), 1e-8)
        grad = grad / denom
    return grad


def solve_projected_y_max(
    inst: LPInstance,
    P: np.ndarray,
    nonnegative_y: bool = True,
) -> Tuple[LPSolveResult, Optional[np.ndarray]]:
    """Projected LP solver that also returns the reduced variable y."""
    t0 = time.perf_counter()
    P = np.asarray(P, dtype=float)
    k = P.shape[1]
    Aub = _as_optional_array(inst.A)
    Aeq_inst = _as_optional_array(inst.A_eq)
    G = None if Aub is None else Aub @ P
    Heq = None if Aeq_inst is None else Aeq_inst @ P
    q = P.T @ inst.c
    obj_scale = _objective_scale(q)
    bounds = [(0.0, None)] * k if nonnegative_y else [(None, None)] * k
    eq_tol = _projected_lp_eq_tolerance(None if Heq is None else inst.b_eq)
    G_aug = G
    b_aug = None if G is None else inst.b
    use_eq_as_ineq = Heq is not None and eq_tol > 0.0
    if use_eq_as_ineq:
        beq = np.asarray(inst.b_eq, dtype=float)
        H_blocks = [Heq, -Heq]
        h_rhs = [beq + eq_tol, -beq + eq_tol]
        G_aug = np.vstack([blk for blk in ([G] if G is not None else []) + H_blocks])
        b_aug = np.concatenate([rhs for rhs in ([np.asarray(inst.b, dtype=float)] if G is not None else []) + h_rhs])
    res = _linprog_with_fallback(
        c=-(q / obj_scale),
        A_ub=G_aug,
        b_ub=b_aug,
        A_eq=None if use_eq_as_ineq else Heq,
        b_eq=None if Heq is None or use_eq_as_ineq else inst.b_eq,
        bounds=bounds,
    )
    runtime = time.perf_counter() - t0
    if not res.success:
        return LPSolveResult(None, float("nan"), False, runtime, None, None, res.message), None
    y = np.asarray(res.x, dtype=float)
    x = P @ y
    if use_eq_as_ineq:
        lam_aug, _ = _extract_duals_for_max(res, inst.n_cons + 2 * inst.n_cons_eq, 0)
        if lam_aug is None:
            lam = np.zeros(inst.n_cons, dtype=float) if inst.n_cons > 0 else None
            nu = np.zeros(inst.n_cons_eq, dtype=float) if inst.n_cons_eq > 0 else None
        else:
            lam = lam_aug[: inst.n_cons] if inst.n_cons > 0 else None
            if inst.n_cons_eq > 0:
                lam_pos = lam_aug[inst.n_cons : inst.n_cons + inst.n_cons_eq]
                lam_neg = lam_aug[inst.n_cons + inst.n_cons_eq :]
                nu = lam_pos - lam_neg
            else:
                nu = None
    else:
        lam, nu = _extract_duals_for_max(res, inst.n_cons, inst.n_cons_eq)
    return LPSolveResult(x, float(inst.objective_constant + inst.c @ x), True, runtime, lam, nu, res.message), y


def implicit_grad_P_with_y(
    inst: LPInstance,
    y: np.ndarray,
    lam: np.ndarray,
    nu: Optional[np.ndarray] = None,
    normalize_by_full_obj: bool = True,
) -> np.ndarray:
    grad_vec = np.asarray(inst.c, dtype=float).copy()
    if inst.n_cons > 0:
        grad_vec = grad_vec - np.asarray(inst.A, dtype=float).T @ lam.reshape(-1)
    if nu is not None and inst.n_cons_eq > 0:
        grad_vec = grad_vec - np.asarray(inst.A_eq, dtype=float).T @ np.asarray(nu, dtype=float).reshape(-1)
    grad = np.outer(grad_vec, y.reshape(-1))
    if normalize_by_full_obj:
        denom = max(abs(float(inst.full_obj or 0.0)), 1e-8)
        grad = grad / denom
    return grad


def train_implicit_projection_model(
    model: nn.Module,
    train: Sequence[LPInstance],
    val: Sequence[LPInstance],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    patience: int = 30,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train PELP_NN / SharedP / FCNN using implicit projected-LP gradients."""
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, List[float]] = {"train_ratio": [], "val_ratio": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val = evaluate_learned_projector(model, val, device)["objective_ratio_mean"] if len(val) > 0 else -float("inf")
    bad_epochs = 0
    if verbose and np.isfinite(best_val):
        print(f"initial val ratio before implicit training: {best_val:.4f}")

    for ep in range(1, epochs + 1):
        order = rng.permutation(len(train))
        model.train()
        batch_terms: List[torch.Tensor] = []
        train_ratios: List[float] = []
        skipped = 0

        opt.zero_grad(set_to_none=True)
        for pos, idx in enumerate(order, start=1):
            inst = train[int(idx)]
            c_t, A_t, b_t = to_torch_instance(inst, device)
            P_t = model(c_t, A_t, b_t)
            P_np = P_t.detach().cpu().numpy().astype(float)
            sol, y = solve_projected_y_max(inst, P_np)
            if not sol.success or y is None:
                skipped += 1
                continue
            lam = sol.dual_ub_for_max
            if lam is None:
                lam = np.zeros(inst.n_cons, dtype=float)
            ratio = sol.obj / max(abs(float(inst.full_obj or 0.0)), 1e-8)
            train_ratios.append(float(ratio))
            grad_np = implicit_grad_P_with_y(inst, y, lam, sol.dual_eq_for_max, True)
            grad_t = torch.as_tensor(grad_np, dtype=P_t.dtype, device=device)
            # This scalar has gradient vec(grad_P)^T d vec(P) / d theta.
            batch_terms.append((P_t * grad_t).sum())

            if len(batch_terms) == batch_size or pos == len(order):
                if batch_terms:
                    loss = -torch.stack(batch_terms).mean()  # Adam minimizes; we maximize.
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    batch_terms = []

        val_ratio = evaluate_learned_projector(model, val, device)["objective_ratio_mean"]
        train_ratio = float(np.mean(train_ratios)) if train_ratios else float("nan")
        history["train_ratio"].append(train_ratio)
        history["val_ratio"].append(val_ratio)

        if val_ratio > best_val + 1e-5:
            best_val = val_ratio
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose and (ep == 1 or ep % max(1, epochs // 20) == 0 or ep == epochs):
            print(
                f"epoch {ep:04d}/{epochs} | train ratio {train_ratio:.4f} | "
                f"val ratio {val_ratio:.4f} | skipped {skipped}"
            )
        if patience > 0 and bad_epochs >= patience:
            if verbose:
                print(f"early stopping at epoch {ep}; best val ratio={best_val:.4f}")
            break

    model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def normalize_columns_nonnegative(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.asarray(P, dtype=float).copy()
    P[P < 0] = 0.0
    sums = P.sum(axis=0, keepdims=True)
    zero_cols = sums.reshape(-1) <= eps
    if np.any(zero_cols):
        P[:, zero_cols] = 1.0 / P.shape[0]
        sums = P.sum(axis=0, keepdims=True)
    return P / np.maximum(sums, eps)


def pca_projection_from_training(train: Sequence[LPInstance], k: int) -> np.ndarray:
    """PCA baseline P=[xbar,V_{k-1}], with negative entries clipped to zero."""
    if any(inst.full_x is None for inst in train):
        ensure_full_solutions(train)
    X = np.vstack([inst.full_x for inst in train])  # D x N
    xbar = X.mean(axis=0)
    Xc = X - xbar[None, :]
    if k <= 1:
        P = xbar.reshape(-1, 1)
    else:
        # Right singular vectors of centered optima.
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt.T[:, : max(0, k - 1)]
        P = np.column_stack([xbar, V])
        if P.shape[1] < k:
            # Pad with random positive columns if rank is too small.
            pad = np.ones((X.shape[1], k - P.shape[1])) / X.shape[1]
            P = np.column_stack([P, pad])
    return normalize_columns_nonnegative(P[:, :k])


def random_column_projection(n_vars: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """ColRand/Rand: choose k variables and set P to selected identity columns."""
    k_eff = min(k, n_vars)
    idx = rng.choice(n_vars, size=k_eff, replace=False)
    P = np.zeros((n_vars, k_eff), dtype=float)
    P[idx, np.arange(k_eff)] = 1.0
    if k_eff < k:
        pad = np.ones((n_vars, k - k_eff)) / n_vars
        P = np.column_stack([P, pad])
    return P


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def objective_ratio(obj: float, full_obj: Optional[float]) -> float:
    if full_obj is None or not np.isfinite(full_obj) or abs(float(full_obj)) <= 1e-8:
        return float("nan")
    return float(obj) / float(full_obj)


def summarize_ratios_and_times(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summary = (
        df.groupby(["method", "K"], dropna=False)
        .agg(
            objective_ratio_mean=("objective_ratio", "mean"),
            objective_ratio_se=("objective_ratio", lambda x: float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            time_mean=("time", "mean"),
            time_se=("time", lambda x: float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            success_rate=("success", "mean"),
            n=("success", "size"),
        )
        .reset_index()
    )
    return summary


@torch.no_grad()
def evaluate_learned_projector(
    model: nn.Module, instances: Sequence[LPInstance], device: torch.device
) -> Dict[str, float]:
    model.eval()
    ratios = []
    times = []
    for inst in instances:
        c_t, A_t, b_t = to_torch_instance(inst, device)
        t0 = time.perf_counter()
        P = model(c_t, A_t, b_t).detach().cpu().numpy().astype(float)
        forward_time = time.perf_counter() - t0
        sol = solve_projected_lp_max(inst, P)
        if sol.success:
            ratios.append(objective_ratio(sol.obj, inst.full_obj))
            times.append(forward_time + sol.runtime)
    return {
        "objective_ratio_mean": float(np.mean(ratios)) if ratios else float("nan"),
        "time_mean": float(np.mean(times)) if times else float("nan"),
    }


@torch.no_grad()
def append_learned_projector_rows(
    rows: List[Dict[str, object]],
    method: str,
    model: nn.Module,
    instances: Sequence[LPInstance],
    k: int,
    device: torch.device,
) -> None:
    model.eval()
    for inst in instances:
        c_t, A_t, b_t = to_torch_instance(inst, device)
        t0 = time.perf_counter()
        P = model(c_t, A_t, b_t).detach().cpu().numpy().astype(float)
        forward_time = time.perf_counter() - t0
        sol = solve_projected_lp_max(inst, P)
        rows.append(
            {
                "method": method,
                "K": k,
                "instance": inst.name,
                "objective": sol.obj,
                "full_objective": inst.full_obj,
                "objective_ratio": objective_ratio(sol.obj, inst.full_obj) if sol.success else np.nan,
                "time": forward_time + sol.runtime,
                "success": float(sol.success),
            }
        )


def append_fixed_projection_rows(
    rows: List[Dict[str, object]],
    method: str,
    P: np.ndarray,
    instances: Sequence[LPInstance],
    k: int,
) -> None:
    for inst in instances:
        sol = solve_projected_lp_max(inst, P)
        rows.append(
            {
                "method": method,
                "K": k,
                "instance": inst.name,
                "objective": sol.obj,
                "full_objective": inst.full_obj,
                "objective_ratio": objective_ratio(sol.obj, inst.full_obj) if sol.success else np.nan,
                "time": sol.runtime,
                "success": float(sol.success),
            }
        )


def append_ours_exact_rows(
    rows: List[Dict[str, object]],
    U: np.ndarray,
    x_anchor: np.ndarray,
    instances: Sequence[LPInstance],
    comparison_ks: Optional[Sequence[int]] = None,
) -> None:
    intrinsic_k = int(U.shape[1])
    display_ks = [intrinsic_k] if comparison_ks is None else [int(k) for k in comparison_ks]
    for inst in instances:
        sol = solve_affine_exact_reduction_max(inst, U, x_anchor)
        for k in display_ks:
            rows.append(
                {
                    "method": "OursExact",
                    "K": k,
                    "intrinsic_K": intrinsic_k,
                    "instance": inst.name,
                    "objective": sol.obj,
                    "full_objective": inst.full_obj,
                    "objective_ratio": objective_ratio(sol.obj, inst.full_obj) if sol.success else np.nan,
                    "time": sol.runtime,
                    "success": float(sol.success),
                }
            )


def append_anchor_rows(
    rows: List[Dict[str, object]],
    instances: Sequence[LPInstance],
    k: int,
) -> None:
    """Reference baseline corresponding to the bridge anchor / y=0 solution."""
    for inst in instances:
        anchor_obj = float(inst.objective_constant)
        rows.append(
            {
                "method": "Anchor",
                "K": int(k),
                "instance": inst.name,
                "objective": anchor_obj,
                "full_objective": inst.full_obj,
                "objective_ratio": objective_ratio(anchor_obj, inst.full_obj),
                "time": 0.0,
                "success": 1.0,
            }
        )


def plot_results(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = list(summary["method"].drop_duplicates())
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        sub = summary[summary["method"] == method].sort_values("K")
        ax.errorbar(sub["K"], sub["objective_ratio_mean"], yerr=sub["objective_ratio_se"], marker="o", label=method)
    ax.set_xlabel("reduced dimension K")
    ax.set_ylabel("test objective ratio")
    ax.set_title("Projection quality")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "objective_ratio_vs_K.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        sub = summary[summary["method"] == method].sort_values("K")
        ax.errorbar(sub["K"], sub["time_mean"], yerr=sub["time_se"], marker="o", label=method)
    ax.set_xlabel("reduced dimension K")
    ax.set_ylabel("average solve time (s)")
    ax.set_yscale("log")
    ax.set_title("Runtime")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_vs_K.png", dpi=200)
    plt.close(fig)


def save_training_history(history: Dict[str, List[float]], out_path: Path) -> None:
    pd.DataFrame(history).to_csv(out_path, index_label="epoch")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def parse_k_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out_dir", type=str, default="pelp_results")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    preset = p.add_mutually_exclusive_group()
    preset.add_argument("--quick", action="store_true", help="small smoke-test setting")
    preset.add_argument("--paper_settings", action="store_true", help="Packing setting from ICML 2025")

    p.add_argument("--dataset", type=str, default="packing",
                   choices=["packing", "maxflow", "mincostflow", "shortest_path", "netlib"])
    p.add_argument("--packing_mode", type=str, default="heterogeneous", choices=["heterogeneous", "fixed_feasible"])
    p.add_argument("--n_vars", type=int, default=80)
    p.add_argument("--n_cons", type=int, default=20)
    p.add_argument("--n_nodes", type=int, default=50, help="number of nodes for network-flow datasets")
    p.add_argument("--n_edges", type=int, default=500, help="number of arcs for network-flow datasets")
    p.add_argument("--grid_size", type=int, default=12, help="grid size for shortest-path dataset")
    p.add_argument("--dstar", type=int, default=4, help="number of planted shortest-path gadgets")
    p.add_argument("--mps_path", type=str, default="", help="path to a Netlib-style .mps or .mps.gz file")
    p.add_argument("--netlib_normal_scale", type=float, default=0.1,
                   help="objective perturbation scale for normal Netlib samples")
    p.add_argument("--netlib_outlier_scale", type=float, default=1.0,
                   help="objective perturbation scale for Netlib outliers")
    p.add_argument("--netlib_outlier_fraction", type=float, default=0.02,
                   help="fraction of Netlib outlier samples")
    p.add_argument("--n_train", type=int, default=40)
    p.add_argument("--n_val", type=int, default=8)
    p.add_argument("--n_test", type=int, default=12)
    p.add_argument("--k_list", type=str, default="5,10")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--n_pelp_layers", type=int, default=4)
    p.add_argument("--generator_hidden_dim", type=int, default=32)

    p.add_argument("--run_sharedp", action="store_true", help="train SharedP baseline")
    p.add_argument("--run_fcnn", action="store_true", help="train non-equivariant FCNN baseline")
    p.add_argument("--skip_pelp", action="store_true", help="do not train the PELP NN")
    p.add_argument("--rand_trials", type=int, default=3)
    p.add_argument("--ours_npz", type=str, default="", help="optional NPZ with U and x_anchor for OursExact")
    p.add_argument("--make_plots", action="store_true", help="save matplotlib plots; CSVs are always saved")
    p.add_argument("--verbose", action="store_true")
    return p


def apply_paper_settings(args: argparse.Namespace) -> None:
    if args.dataset == "packing":
        args.n_vars = 500
        args.n_cons = 50
    else:
        args.n_nodes = 50
        args.n_edges = 500
    args.n_train = 425
    args.n_val = 25
    args.n_test = 50
    args.k_list = "10,20,30,40,50"
    args.epochs = 500
    args.batch_size = 8
    args.lr = 1e-3
    args.patience = 60
    args.hidden_dim = 32
    args.n_pelp_layers = 4
    args.generator_hidden_dim = 32
    args.rand_trials = 10
    args.run_sharedp = True


def apply_quick_settings(args: argparse.Namespace) -> None:
    if args.dataset == "packing":
        args.n_vars = 80
        args.n_cons = 20
    elif args.dataset in {"maxflow", "mincostflow"}:
        args.n_nodes = 18
        args.n_edges = 60
    else:
        args.grid_size = 8
        args.dstar = 3
    args.n_train = 40
    args.n_val = 8
    args.n_test = 12
    args.k_list = "5,10"
    args.epochs = 8
    args.batch_size = 4
    args.patience = 8
    args.rand_trials = 3


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.paper_settings:
        apply_paper_settings(args)
    elif args.quick:
        apply_quick_settings(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_total = args.n_train + args.n_val + args.n_test
    if args.dataset == "packing":
        all_instances = generate_packing_instances(
            n_total, args.n_vars, args.n_cons, rng, mode=args.packing_mode
        )
    elif args.dataset == "maxflow":
        all_instances = generate_maxflow_instances(
            n_total, args.n_nodes, args.n_edges, rng, mode=args.packing_mode
        )
    elif args.dataset == "mincostflow":
        all_instances = generate_mincostflow_instances(
            n_total, args.n_nodes, args.n_edges, rng, mode=args.packing_mode
        )
    elif args.dataset == "shortest_path":
        if args.packing_mode != "fixed_feasible":
            raise ValueError("shortest_path is a fixed-feasible repeated-LP family; use --packing_mode fixed_feasible")
        all_instances = generate_shortest_path_instances(
            n_total, args.grid_size, args.dstar, rng
        )
    elif args.dataset == "netlib":
        if not args.mps_path:
            raise ValueError("--mps_path is required when --dataset netlib")
        all_instances = generate_netlib_perturbed_instances(
            n_total,
            args.mps_path,
            rng,
            normal_scale=args.netlib_normal_scale,
            outlier_scale=args.netlib_outlier_scale,
            outlier_fraction=args.netlib_outlier_fraction,
        )
    else:  # pragma: no cover
        raise ValueError(args.dataset)

    train = all_instances[: args.n_train]
    val = all_instances[args.n_train : args.n_train + args.n_val]
    test = all_instances[args.n_train + args.n_val :]
    n_vars_eff = train[0].n_vars
    n_cons_eff = train[0].n_feature_cons
    print(
        f"Generated {args.dataset} data: train={len(train)}, val={len(val)}, test={len(test)}, "
        f"M(feature)={n_cons_eff}, N={n_vars_eff}, mode={args.packing_mode}"
    )

    print("Solving full LPs for train/val/test optima...")
    ensure_full_solutions(train)
    ensure_full_solutions(val)
    ensure_full_solutions(test)

    k_list = parse_k_list(args.k_list)
    bridge_mode = bool(hasattr(train[0], "transform_x0"))
    rows: List[Dict[str, object]] = []

    # Full LP reference rows.  Repeat for all K so the plots display a flat reference line.
    for k in k_list:
        for inst in test:
            rows.append(
                {
                    "method": "Full",
                    "K": k,
                    "instance": inst.name,
                    "objective": inst.full_obj,
                    "full_objective": inst.full_obj,
                    "objective_ratio": 1.0,
                    "time": inst.full_time,
                    "success": 1.0,
                }
            )
        if bridge_mode:
            append_anchor_rows(rows, test, k)

    # Optional OursExact affine reduction hook.
    if args.ours_npz:
        npz = np.load(args.ours_npz)
        U = np.asarray(npz["U"], dtype=float)
        x_anchor = np.asarray(npz["x_anchor"], dtype=float).reshape(-1)
        append_ours_exact_rows(rows, U, x_anchor, test, comparison_ks=k_list)
        print(f"Added OursExact from {args.ours_npz}: dim={U.shape[1]}")

    for k in k_list:
        print(f"\n===== K={k} =====")

        # Rand / column-randomized baseline.
        for tr in range(args.rand_trials):
            P_rand = random_column_projection(n_vars_eff, k, rng)
            append_fixed_projection_rows(rows, f"Rand#{tr+1}", P_rand, test, k)

        # PCA baseline, trained on exactly the same training instances.
        print("Building PCA projection...")
        P_pca = pca_projection_from_training(train, k)
        append_fixed_projection_rows(rows, "PCA", P_pca, test, k)
        if bridge_mode:
            print(
                "Bridge/fixed-feasible mode detected: the 2025 softmax projection family is strictly positive, "
                "so on nullspace-bridge instances it can collapse to the Anchor solution when exact sparse zeros are needed."
            )

        # SharedP baseline.
        if args.run_sharedp:
            print("Training SharedP...")
            shared = SharedProjection(n_vars_eff, k)
            shared, hist = train_implicit_projection_model(
                shared,
                train,
                val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=args.seed + 1000 + k,
                patience=args.patience,
                verbose=args.verbose,
            )
            save_training_history(hist, out_dir / f"history_SharedP_K{k}.csv")
            append_learned_projector_rows(rows, "SharedP", shared, test, k, device)
            torch.save(shared.state_dict(), out_dir / f"SharedP_K{k}.pt")

        # Non-equivariant FCNN baseline.
        if args.run_fcnn:
            print("Training FCNN baseline...")
            fcnn = FCNNProjectionNet(n_vars_eff, n_cons_eff, k, args.hidden_dim)
            fcnn, hist = train_implicit_projection_model(
                fcnn,
                train,
                val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=args.seed + 2000 + k,
                patience=args.patience,
                verbose=args.verbose,
            )
            save_training_history(hist, out_dir / f"history_FCNN_K{k}.csv")
            append_learned_projector_rows(rows, "FCNN", fcnn, test, k, device)
            torch.save(fcnn.state_dict(), out_dir / f"FCNN_K{k}.pt")

        # Iwata--Sakaue PELP NN.
        if not args.skip_pelp:
            print("Training PELP_NN...")
            pelp = PELPProjectionNet(
                k=k,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_pelp_layers,
                generator_hidden_dim=args.generator_hidden_dim,
            )
            pelp, hist = train_implicit_projection_model(
                pelp,
                train,
                val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=args.seed + 3000 + k,
                patience=args.patience,
                verbose=args.verbose,
            )
            save_training_history(hist, out_dir / f"history_PELP_NN_K{k}.csv")
            append_learned_projector_rows(rows, "PELP_NN", pelp, test, k, device)
            torch.save(pelp.state_dict(), out_dir / f"PELP_NN_K{k}.pt")

    raw = pd.DataFrame(rows)
    # Merge Rand trials under a single method for aggregate plots.
    raw_for_summary = raw.copy()
    raw_for_summary["method"] = raw_for_summary["method"].str.replace(r"Rand#\d+", "Rand", regex=True)
    summary = summarize_ratios_and_times(raw_for_summary)
    raw.to_csv(out_dir / "raw_results.csv", index=False)
    summary.to_csv(out_dir / "summary_results.csv", index=False)
    if args.make_plots:
        plot_results(summary, out_dir)

    if bridge_mode and not summary.empty and "Anchor" in set(summary["method"]):
        anchor_map = {
            int(row.K): float(row.objective_ratio_mean)
            for row in summary[summary["method"] == "Anchor"].itertuples()
        }
        collapsed = []
        for row in summary.itertuples():
            if row.method in {"Anchor", "Full", "OursExact"}:
                continue
            anchor_ratio = anchor_map.get(int(row.K))
            if anchor_ratio is not None and np.isfinite(anchor_ratio) and abs(float(row.objective_ratio_mean) - anchor_ratio) <= 1e-6:
                collapsed.append(f"{row.method}@K={int(row.K)}")
        if collapsed:
            print(
                "Bridge diagnostic: these methods matched the Anchor baseline exactly, "
                "which usually means the softmax-positive projection family could not move away from the bridge anchor: "
                + ", ".join(collapsed)
            )

    print("\nSummary:")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 20, "display.width", 120):
        print(summary.sort_values(["K", "method"]))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
