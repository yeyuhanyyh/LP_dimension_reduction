#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final paper-style experiment suite for fixed-X, varying-c LP families.

This is the high-level entry point for the final figures requested in this
workspace.  It produces:

1. A paper-style K-sweep (fixed n_train, varying reduced dimension K) with
   the requested fixed-(A,b) baseline family:
       Full / OursExact / Rand / PCA / SGA / FCNN-c
   on
       packing, maxflow, mincostflow, shortest_path,
       multiple random standard-form cases,
       selected Netlib instances.

2. A learned-dimension curve for our exact method:
       learned rank dim(span D) vs number of training samples processed.

3. A separate sample-efficiency comparison:
       OursExact / PCA / SGA
   at fixed K while sweeping n_train.

The suite reuses cached results when possible and only runs missing cases.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Install with: pip install torch\n" + str(exc))

from compare_fixedX_family_suite import (
    OriginalFixedXProblem,
    append_general_projection_rows,
    append_ours_exact_rows,
    build_lp_instance,
    build_problem,
    build_standard_form_problem,
    choose_independent_columns,
    convert_to_general_projection_instance,
    enumerate_center_directions,
    make_fixed_x_bundle,
    orthonormalize_columns,
    parse_k_list,
    project_columns_to_common_region,
    raw_pca_projection,
    stack_inequalities_with_upper_bounds,
    standard_cost_from_reward,
    train_sga_final_projection,
)
from compare_ours_exact_vs_pelp_fixedX import alg2_cumulative
from iwata_sakaue_pelp_projection_compare import (
    CostOnlyProjectionNet,
    FCNNProjectionNet,
    PELPProjectionNet,
    append_fixed_projection_rows,
    append_learned_projector_rows,
    ensure_full_solutions,
    load_mps_fixed_feasible_data,
    objective_ratio,
    random_column_projection,
    save_training_history,
    summarize_ratios_and_times,
    train_implicit_projection_model,
    transform_fixed_feasible_problem,
)

METHOD_ORDER = ["Full", "OursExact", "Rand", "PCA", "SGA", "CostOnly"]
METHOD_DISPLAY = {
    "Full": "Full",
    "OursExact": "OursExact",
    "Rand": "Rand",
    "PCA": "PCA",
    "SGA": "SGA",
    "CostOnly": "FCNN-c",
    "FCNN": "FCNN",
    "PELP": "PELP",
}
METHOD_STYLE = {
    "Full": {"color": "#555555", "marker": "o", "linestyle": (0, (5, 2))},
    "OursExact": {"color": "#d62728", "marker": "D", "linestyle": (0, (6, 2))},
    "Rand": {"color": "#6f7d55", "marker": "x", "linestyle": (0, (2, 2))},
    "PCA": {"color": "#1f77b4", "marker": "s", "linestyle": (0, (1, 2))},
    "SGA": {"color": "#ff7f0e", "marker": "+", "linestyle": (0, (3, 2))},
    "CostOnly": {"color": "#8c564b", "marker": "v", "linestyle": (0, (4, 1, 1, 1))},
    "FCNN": {"color": "#9467bd", "marker": "^", "linestyle": (0, (3, 1, 1, 1))},
    "PELP": {"color": "#2ca02c", "marker": "P", "linestyle": (0, (7, 2, 2, 2))},
}
SAMPLE_METHOD_ORDER = ["OursExact", "PCA", "SGA", "CostOnly"]
SAMPLE_METHOD_STYLE = {k: v for k, v in METHOD_STYLE.items() if k in SAMPLE_METHOD_ORDER}
RANK_CURVE_MAX_SAMPLES = 20


@dataclass
class CaseSpec:
    slug: str
    title: str
    family: str
    args_updates: Dict[str, object]
    sample_k: int
    seed_offset: int
    mps_path: str = ""
    reuse_dir: str = ""


@dataclass
class PreparedCaseData:
    case: CaseSpec
    args: argparse.Namespace
    problem: OriginalFixedXProblem
    bundle: object
    proj_train: List[object]
    proj_val: List[object]
    proj_test: List[object]


@dataclass
class SignedNullspaceInstance:
    name: str
    c_std: np.ndarray
    g_reward: np.ndarray
    objective_constant: float
    full_objective: float
    full_time: float


@dataclass
class SignedNullspaceData:
    N: np.ndarray
    x_ref: np.ndarray
    train: List[SignedNullspaceInstance]
    val: List[SignedNullspaceInstance]
    test: List[SignedNullspaceInstance]


class SignedCostOnlyProjectionNet(torch.nn.Module):
    """Signed nullspace-coordinate projector for fixed-(A,b) families."""

    def __init__(self, input_dim: int, reduced_dim: int, k: int, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = int(input_dim)
        self.reduced_dim = int(reduced_dim)
        self.k = int(k)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden_dim),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_dim, self.reduced_dim * self.k),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        if self.reduced_dim == 0 or self.k == 0:
            return torch.zeros((self.reduced_dim, self.k), dtype=g.dtype, device=g.device)
        raw = self.net(g.reshape(-1)).reshape(self.reduced_dim, self.k)
        P = torch.tanh(raw)
        norms = torch.linalg.norm(P, dim=0, keepdim=True).clamp_min(1e-6)
        return P / norms


class SignedDirectCostOnlyProjectionNet(torch.nn.Module):
    """Signed c-only projector in the direct projection coordinates."""

    def __init__(self, n_vars: int, k: int, hidden_dim: int = 32):
        super().__init__()
        self.n_vars = int(n_vars)
        self.k = int(k)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.n_vars, hidden_dim),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_dim, self.n_vars * self.k),
        )

    def forward(self, c: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        if self.n_vars == 0 or self.k == 0:
            return torch.zeros((self.n_vars, self.k), dtype=c.dtype, device=c.device)
        raw = self.net(c.reshape(-1)).reshape(self.n_vars, self.k)
        P = torch.tanh(raw)
        norms = torch.linalg.norm(P, dim=0, keepdim=True).clamp_min(1e-6)
        return P / norms


def mean_se(x: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(arr.size))


def common_anchor(problem: OriginalFixedXProblem) -> np.ndarray:
    anchor = np.asarray(problem.metadata.get("anchor_x0", np.zeros(problem.n_vars, dtype=float)), dtype=float).reshape(-1)
    if anchor.shape[0] != problem.n_vars:
        raise ValueError("anchor_x0 dimension mismatch.")
    return anchor


def nullspace_basis(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.size == 0 or A.shape[0] == 0:
        return np.eye(A.shape[1], dtype=float)
    _u, s, vh = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > tol * max(1.0, s[0] if len(s) else 1.0)))
    return vh[rank:, :].T.copy()


def standard_form_reference(problem: OriginalFixedXProblem, stdlp) -> np.ndarray:
    z_ref = common_anchor(problem)
    A_ub, b_ub = stack_inequalities_with_upper_bounds(problem)
    slack = np.zeros(0, dtype=float) if A_ub.shape[0] == 0 else np.asarray(b_ub, dtype=float) - A_ub @ z_ref
    x_ref = np.concatenate([slack, z_ref]).astype(float)
    x_ref[np.abs(x_ref) <= 1e-10] = 0.0
    if np.any(x_ref < -1e-7):
        raise ValueError("Computed standard-form reference point is not nonnegative.")
    if np.linalg.norm(stdlp.Aeq @ x_ref - stdlp.b) > 1e-7:
        raise ValueError("Computed standard-form reference point is infeasible.")
    return x_ref


def prepare_signed_nullspace_data(data: PreparedCaseData) -> SignedNullspaceData:
    x_ref = standard_form_reference(data.problem, data.bundle.stdlp)
    N = orthonormalize_columns(nullspace_basis(data.bundle.stdlp.Aeq))

    def _convert(instances: Sequence[object]) -> List[SignedNullspaceInstance]:
        out: List[SignedNullspaceInstance] = []
        for inst in instances:
            c_std = np.asarray(inst.c_std, dtype=float).reshape(-1)
            out.append(
                SignedNullspaceInstance(
                    name=inst.name,
                    c_std=c_std,
                    g_reward=-(N.T @ c_std),
                    objective_constant=float(-c_std @ x_ref),
                    full_objective=float(inst.full_obj),
                    full_time=float(inst.full_time),
                )
            )
        return out

    return SignedNullspaceData(
        N=N,
        x_ref=x_ref,
        train=_convert(data.bundle.train),
        val=_convert(data.bundle.val),
        test=_convert(data.bundle.test),
    )


def signed_costonly_warmstart_matrix(data: PreparedCaseData, signed: SignedNullspaceData, k: int) -> np.ndarray:
    A_ub, b_ub = stack_inequalities_with_upper_bounds(data.problem)
    cols: List[np.ndarray] = []
    for inst in data.bundle.train:
        z = np.asarray(inst.full_x, dtype=float).reshape(-1)
        slack = np.zeros(0, dtype=float) if A_ub.shape[0] == 0 else np.asarray(b_ub, dtype=float) - A_ub @ z
        x_std = np.concatenate([slack, z])
        u = signed.N.T @ (x_std - signed.x_ref)
        if np.linalg.norm(u) > 1e-10:
            cols.append(u)
    if not cols:
        return np.zeros((signed.N.shape[1], int(k)), dtype=float)
    U = np.column_stack(cols)
    if U.shape[1] >= int(k):
        idx = np.linspace(0, U.shape[1] - 1, int(k)).round().astype(int)
        P0 = U[:, idx]
    else:
        repeats = [U[:, j % U.shape[1]] for j in range(int(k))]
        P0 = np.column_stack(repeats)
    norms = np.maximum(np.linalg.norm(P0, axis=0, keepdims=True), 1e-8)
    return P0 / norms


def direct_costonly_costspace_matrix(instances: Sequence[object], n_vars: int, k: int) -> np.ndarray:
    n_vars = int(n_vars)
    k = int(k)
    if n_vars <= 0 or k <= 0:
        return np.zeros((n_vars, k), dtype=float)
    cols = [np.asarray(inst.c, dtype=float).reshape(-1) for inst in instances if np.linalg.norm(inst.c) > 1e-10]
    if not cols:
        return np.zeros((n_vars, k), dtype=float)
    G = np.column_stack(cols)
    pieces: List[np.ndarray] = []
    g_bar = np.mean(G, axis=1)
    if np.linalg.norm(g_bar) > 1e-10:
        pieces.append(g_bar / max(np.linalg.norm(g_bar), 1e-12))
    G_center = G - g_bar.reshape(-1, 1)
    if G_center.size > 0:
        try:
            U, s, _vh = np.linalg.svd(G_center, full_matrices=False)
            for j in range(U.shape[1]):
                if s[j] <= 1e-10:
                    continue
                pieces.append(U[:, j])
                if len(pieces) >= k:
                    break
        except np.linalg.LinAlgError:
            pass
    if not pieces:
        pieces.append(G[:, 0] / max(np.linalg.norm(G[:, 0]), 1e-12))
    if len(pieces) < k:
        pieces.extend(pieces[j % len(pieces)].copy() for j in range(k - len(pieces)))
    P0 = np.column_stack(pieces[:k]).astype(float)
    norms = np.maximum(np.linalg.norm(P0, axis=0, keepdims=True), 1e-8)
    return P0 / norms


def signed_costonly_costspace_matrix(instances: Sequence[SignedNullspaceInstance], reduced_dim: int, k: int) -> np.ndarray:
    reduced_dim = int(reduced_dim)
    k = int(k)
    if reduced_dim <= 0 or k <= 0:
        return np.zeros((reduced_dim, k), dtype=float)
    cols = [np.asarray(inst.g_reward, dtype=float).reshape(-1) for inst in instances if np.linalg.norm(inst.g_reward) > 1e-10]
    if not cols:
        return np.zeros((reduced_dim, k), dtype=float)
    G = np.column_stack(cols)
    pieces: List[np.ndarray] = []
    g_bar = np.mean(G, axis=1)
    if np.linalg.norm(g_bar) > 1e-10:
        pieces.append(g_bar / max(np.linalg.norm(g_bar), 1e-12))
    G_center = G - g_bar.reshape(-1, 1)
    if G_center.size > 0:
        try:
            U, s, _vh = np.linalg.svd(G_center, full_matrices=False)
            for j in range(U.shape[1]):
                if s[j] <= 1e-10:
                    continue
                pieces.append(U[:, j])
                if len(pieces) >= k:
                    break
        except np.linalg.LinAlgError:
            pass
    if not pieces:
        pieces.append(G[:, 0] / max(np.linalg.norm(G[:, 0]), 1e-12))
    if len(pieces) < k:
        pieces.extend(pieces[j % len(pieces)].copy() for j in range(k - len(pieces)))
    P0 = np.column_stack(pieces[:k]).astype(float)
    norms = np.maximum(np.linalg.norm(P0, axis=0, keepdims=True), 1e-8)
    return P0 / norms


def signed_costonly_fewopt_matrix(data: PreparedCaseData, signed: SignedNullspaceData, k: int, n_unique: int) -> np.ndarray:
    P0 = signed_costonly_warmstart_matrix(data, signed, int(k))
    if P0.size == 0:
        return P0
    n_unique = int(max(1, min(int(n_unique), P0.shape[1])))
    cols = [P0[:, j % n_unique].copy() for j in range(int(k))]
    out = np.column_stack(cols).astype(float)
    norms = np.maximum(np.linalg.norm(out, axis=0, keepdims=True), 1e-8)
    return out / norms


def signed_costonly_random_feasible_matrix(signed: SignedNullspaceData, k: int, seed: int) -> np.ndarray:
    r = int(signed.N.shape[1])
    k = int(k)
    if r <= 0 or k <= 0:
        return np.zeros((r, k), dtype=float)
    rng = np.random.default_rng(seed)
    cols: List[np.ndarray] = []
    tries = 0
    max_tries = max(20, 40 * k)
    while len(cols) < k and tries < max_tries:
        tries += 1
        d = rng.normal(size=r)
        nd = float(np.linalg.norm(d))
        if nd <= 1e-10:
            continue
        d = d / nd
        x_dir = signed.N @ d
        if np.linalg.norm(x_dir) <= 1e-10:
            continue
        neg = x_dir < -1e-10
        if np.any(neg):
            step_max = float(np.min(signed.x_ref[neg] / (-x_dir[neg])))
            if not np.isfinite(step_max) or step_max <= 1e-10:
                continue
        cols.append(d)
    if not cols:
        return np.zeros((r, k), dtype=float)
    if len(cols) < k:
        cols.extend(cols[j % len(cols)].copy() for j in range(k - len(cols)))
    P0 = np.column_stack(cols[:k]).astype(float)
    norms = np.maximum(np.linalg.norm(P0, axis=0, keepdims=True), 1e-8)
    return P0 / norms


def initialize_signed_costonly_model(model: SignedCostOnlyProjectionNet, P0: np.ndarray) -> None:
    if P0.size == 0:
        return
    last = model.net[-1]
    if not isinstance(last, torch.nn.Linear):
        return
    target = np.asarray(P0, dtype=np.float32)
    target = np.clip(target, -0.95, 0.95)
    raw = np.arctanh(target).reshape(-1)
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(torch.from_numpy(raw))


def add_explicit_nonnegativity(instances: Sequence[object]) -> List[object]:
    out: List[object] = []
    for inst in instances:
        new_inst = copy.deepcopy(inst)
        n = int(new_inst.n_vars)
        A_old = np.zeros((0, n), dtype=float) if getattr(new_inst, "A", None) is None else np.asarray(new_inst.A, dtype=float)
        b_old = np.zeros(0, dtype=float) if getattr(new_inst, "b", None) is None else np.asarray(new_inst.b, dtype=float).reshape(-1)
        new_inst.A = np.vstack([A_old, -np.eye(n, dtype=float)])
        new_inst.b = np.concatenate([b_old, np.zeros(n, dtype=float)])
        out.append(new_inst)
    return out


def should_bridge_for_projection(problem: OriginalFixedXProblem) -> bool:
    if str(problem.metadata.get("projection_coordinates", "")).lower() == "original_nonnegative":
        return False
    anchor = common_anchor(problem)
    if problem.A_eq.size > 0:
        return True
    if np.linalg.norm(anchor) > 1e-10:
        return True
    if problem.A_ineq.size > 0 and np.any(np.asarray(problem.b_ineq, dtype=float).reshape(-1) < -1e-10):
        return True
    return False


def maybe_none_matrix(arr: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(arr, dtype=float)
    return None if arr.size == 0 or arr.shape[0] == 0 else arr


def transform_instances_to_bridge(problem: OriginalFixedXProblem, instances: Sequence[object]) -> List[object]:
    x0 = common_anchor(problem)
    A_eq = np.asarray(problem.A_eq, dtype=float)
    b_eq = np.asarray(problem.b_eq, dtype=float).reshape(-1)
    A_ineq = maybe_none_matrix(problem.A_ineq)
    b_ineq = None if A_ineq is None else np.asarray(problem.b_ineq, dtype=float).reshape(-1)
    ub_meta = problem.metadata.get("projection_ub", None)
    if ub_meta is not None:
        ub = np.asarray(ub_meta, dtype=float).reshape(-1)
    else:
        ub = None if problem.ub is None else np.asarray(problem.ub, dtype=float).reshape(-1)

    out = []
    for inst in instances:
        bridged = transform_fixed_feasible_problem(
            reward=np.asarray(inst.c, dtype=float),
            A_eq=A_eq,
            b_eq=b_eq,
            x0=x0,
            name=inst.name,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            ub=ub,
        )
        if hasattr(inst, "c_std"):
            bridged.c_std = np.asarray(inst.c_std, dtype=float)
        if hasattr(inst, "theta"):
            bridged.theta = np.asarray(inst.theta, dtype=float)
        if getattr(inst, "full_x", None) is not None:
            bridged.full_x = np.asarray(inst.full_x, dtype=float).reshape(-1) - x0
            bridged.full_obj = float(inst.full_obj)
            bridged.full_time = float(getattr(inst, "full_time", 0.0) or 0.0)
        out.append(bridged)
    return out


def bridge_instances_for_projection(problem: OriginalFixedXProblem, instances: Sequence[object]) -> List[object]:
    if not should_bridge_for_projection(problem):
        return list(instances)
    return transform_instances_to_bridge(problem, instances)


def solve_signed_nullspace_lp_max(
    inst: SignedNullspaceInstance,
    P: np.ndarray,
    N: np.ndarray,
    x_ref: np.ndarray,
) -> Tuple[bool, float, float, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    t0 = time.perf_counter()
    P = np.asarray(P, dtype=float)
    N = np.asarray(N, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
    k = int(P.shape[1])
    if k == 0:
        return True, float(inst.objective_constant), time.perf_counter() - t0, np.zeros(0, dtype=float), np.zeros(x_ref.shape[0], dtype=float), x_ref.copy()

    q = P.T @ np.asarray(inst.g_reward, dtype=float)
    G = (-N) @ P
    scale = max(1.0, float(np.linalg.norm(q, ord=np.inf)))
    feas_tol = 1e-6 * max(1.0, float(np.linalg.norm(x_ref, ord=np.inf)))
    res = linprog(
        c=-(q / scale),
        A_ub=G,
        b_ub=x_ref,
        bounds=[(None, None)] * k,
        method="highs",
        options={"presolve": True},
    )
    runtime = time.perf_counter() - t0
    if not res.success:
        return False, float("nan"), runtime, None, None, None
    y = np.asarray(res.x, dtype=float)
    lam = np.zeros(x_ref.shape[0], dtype=float)
    try:
        lam = -np.asarray(res.ineqlin.marginals, dtype=float)
        lam = np.maximum(lam, 0.0)
    except Exception:
        pass
    x_std = x_ref + N @ (P @ y)
    if float(np.max(-x_std)) > feas_tol:
        return False, float("nan"), runtime, y, lam, x_std
    obj = float(inst.objective_constant + inst.g_reward @ (P @ y))
    return True, obj, runtime, y, lam, x_std


def evaluate_signed_costonly(
    model: SignedCostOnlyProjectionNet,
    instances: Sequence[SignedNullspaceInstance],
    N: np.ndarray,
    x_ref: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ratios: List[float] = []
    times: List[float] = []
    with torch.no_grad():
        for inst in instances:
            g_t = torch.as_tensor(inst.g_reward, dtype=torch.float32, device=device)
            t0 = time.perf_counter()
            P = model(g_t).detach().cpu().numpy().astype(float)
            forward_time = time.perf_counter() - t0
            success, obj, runtime, _y, _lam, _x_std = solve_signed_nullspace_lp_max(inst, P, N, x_ref)
            if success:
                ratios.append(objective_ratio(obj, inst.full_objective))
                times.append(forward_time + runtime)
    return {
        "objective_ratio_mean": float(np.mean(ratios)) if ratios else float("nan"),
        "time_mean": float(np.mean(times)) if times else float("nan"),
    }


def train_signed_costonly_model(
    model: SignedCostOnlyProjectionNet,
    train: Sequence[SignedNullspaceInstance],
    val: Sequence[SignedNullspaceInstance],
    N: np.ndarray,
    x_ref: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    patience: int = 30,
    verbose: bool = True,
) -> Tuple[SignedCostOnlyProjectionNet, Dict[str, List[float]]]:
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, List[float]] = {"train_ratio": [], "val_ratio": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val = evaluate_signed_costonly(model, val, N, x_ref, device)["objective_ratio_mean"] if len(val) > 0 else -float("inf")
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        order = rng.permutation(len(train))
        model.train()
        batch_terms: List[torch.Tensor] = []
        train_ratios: List[float] = []
        opt.zero_grad(set_to_none=True)
        skipped = 0

        for pos, idx in enumerate(order, start=1):
            inst = train[int(idx)]
            g_t = torch.as_tensor(inst.g_reward, dtype=torch.float32, device=device)
            P_t = model(g_t)
            P_np = P_t.detach().cpu().numpy().astype(float)
            success, obj, _runtime, y, lam, _x_std = solve_signed_nullspace_lp_max(inst, P_np, N, x_ref)
            if not success or y is None or lam is None:
                skipped += 1
                continue
            train_ratios.append(objective_ratio(obj, inst.full_objective))
            grad_vec = np.asarray(inst.g_reward, dtype=float) + np.asarray(N, dtype=float).T @ np.asarray(lam, dtype=float)
            grad_np = np.outer(grad_vec, np.asarray(y, dtype=float))
            denom = max(abs(float(inst.full_objective)), 1e-8)
            grad_np = grad_np / denom
            grad_t = torch.as_tensor(grad_np, dtype=P_t.dtype, device=device)
            batch_terms.append((P_t * grad_t).sum())

            if len(batch_terms) == batch_size or pos == len(order):
                if batch_terms:
                    loss = -torch.stack(batch_terms).mean()
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    batch_terms = []

        val_ratio = evaluate_signed_costonly(model, val, N, x_ref, device)["objective_ratio_mean"]
        train_ratio = float(np.nanmean(train_ratios)) if train_ratios else float("nan")
        history["train_ratio"].append(train_ratio)
        history["val_ratio"].append(val_ratio)

        if val_ratio > best_val + 1e-5:
            best_val = val_ratio
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose and (ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs):
            print(
                f"signed cost-only epoch {ep:04d}/{epochs} | "
                f"train ratio {train_ratio:.4f} | val ratio {val_ratio:.4f} | skipped {skipped}"
            )
        if patience > 0 and bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return model, history


def append_signed_costonly_rows(
    rows: List[Dict[str, object]],
    model: SignedCostOnlyProjectionNet,
    instances: Sequence[SignedNullspaceInstance],
    N: np.ndarray,
    x_ref: np.ndarray,
    k: int,
    device: torch.device,
    n_slack: int,
) -> None:
    model.eval()
    with torch.no_grad():
        for inst in instances:
            g_t = torch.as_tensor(inst.g_reward, dtype=torch.float32, device=device)
            t0 = time.perf_counter()
            P = model(g_t).detach().cpu().numpy().astype(float)
            forward_time = time.perf_counter() - t0
            success, obj, runtime, _y, _lam, x_std = solve_signed_nullspace_lp_max(inst, P, N, x_ref)
            rows.append(
                {
                    "method": "CostOnly",
                    "K": int(k),
                    "instance": inst.name,
                    "objective": obj,
                    "full_objective": inst.full_objective,
                    "objective_ratio": objective_ratio(obj, inst.full_objective) if success else np.nan,
                    "time": forward_time + runtime,
                    "success": float(success),
                    "signed_nullspace_obj": obj,
                    "x_std_norm": float(np.linalg.norm(x_std)) if success and x_std is not None else float("nan"),
                    "y_support_dim": int(max(0, np.count_nonzero(x_std[n_slack:] > 1e-8))) if success and x_std is not None else 0,
                }
            )


def append_full_rows(rows: List[Dict[str, object]], instances: Sequence[object], k_list: Sequence[int]) -> None:
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


def summarize_anchor_quality(
    raw: pd.DataFrame,
    base_obj_by_instance: Dict[str, float],
    clip_hi: float = 1.05,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["anchor_objective"] = df["instance"].map(base_obj_by_instance)

    def _instance_quality(row) -> float:
        base = float(row["anchor_objective"])
        full = float(row["full_objective"])
        obj = float(row["objective"]) if pd.notna(row["objective"]) else float("nan")
        if not np.isfinite(base) or not np.isfinite(full) or not np.isfinite(obj):
            return float("nan")
        denom = full - base
        if denom <= 1e-12:
            return 1.0 if abs(obj - full) <= 1e-8 else float("nan")
        score = max(0.0, obj - base) / denom
        return float(max(0.0, min(float(clip_hi), score)))

    df["anchor_quality"] = df.apply(_instance_quality, axis=1)
    rows: List[Dict[str, object]] = []
    for (method, k), grp in df.groupby(["method", "K"], dropna=False):
        q_mean, q_se = mean_se(grp["anchor_quality"].tolist())
        rows.append(
            {
                "method": method,
                "K": int(k),
                "anchor_quality_mean": q_mean,
                "anchor_quality_se": q_se,
                "success_rate": float(np.nanmean(grp["success"])) if len(grp) > 0 else float("nan"),
                "n": int(len(grp)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["K", "method"]).reset_index(drop=True)
    return out


def family_sample_mode(family: str, profile: str) -> str:
    _profile = str(profile).lower()
    if family in {"packing", "maxflow", "netlib"}:
        return "multiplicative_factor"
    if family == "affine_edge":
        return "masked_factor_gaussian"
    return "factor_gaussian"


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_namespace() -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.seed = 17
    ns.device = "cpu"
    ns.verbose = False
    ns.suite_profile = "natural"

    ns.n_train = 24
    ns.n_val = 6
    ns.n_test = 12
    ns.k_list = "5,10,20,30,40,50"

    ns.epochs = 8
    ns.batch_size = 4
    ns.lr = 1e-3
    ns.patience = 4
    ns.hidden_dim = 16
    ns.n_pelp_layers = 4
    ns.generator_hidden_dim = 16
    ns.costonly_epochs = 6
    ns.costonly_batch_size = 4
    ns.costonly_lr = 1e-3
    ns.costonly_patience = 3
    ns.costonly_hidden_dim = 12
    ns.costonly_use_warmstart = False
    ns.costonly_init = "cost_pca"
    ns.costonly_solver_space = "auto"
    ns.costonly_warm_cols = 2

    ns.sga_epochs = 1
    ns.sga_lr = 1e-2
    ns.rand_trials = 3

    ns.prior_nominal_k = 24
    ns.prior_rho_scale = 1.0
    ns.origin_margin_frac = 0.45
    ns.reward_margin_frac = 0.8
    ns.sample_radius_frac = 0.95
    ns.sample_mode = "factor_gaussian"
    ns.rare_prob = 0.70
    ns.center_noise_frac = 0.01
    ns.rare_amp_frac = 0.90
    ns.sparse_rare_max_active = 2
    ns.factor_scale_frac = 0.70
    ns.factor_decay = 0.82
    ns.mult_log_scale = 0.45
    ns.make_quality_figures = False

    ns.n_vars = 360
    ns.n_cons = 60
    ns.cost_rank = 24
    ns.packing_design = "block_gadget"

    ns.n_nodes = 64
    ns.n_edges = 300
    ns.flow_design = "path_gadget"

    ns.grid_size = 16
    ns.shortest_gadgets = 0
    ns.sp_cost_low = 1.0
    ns.sp_cost_high = 3.0
    ns.sp_band_width = 1
    ns.sp_potential_scale = 6.0

    ns.randlp_n_vars = 160
    ns.randlp_n_eq = 36
    ns.randlp_n_ineq = 48
    ns.randlp_basis_mode = "random"
    ns.randlp_pool_factor = 5
    ns.affine_blocks = 20
    ns.affine_anchor_reward = 1.0
    ns.affine_switch_margin = 0.03

    ns.netlib_cost_rank = 24
    ns.netlib_basis_mode = "topabs"
    return ns


def case_namespace(base: argparse.Namespace, case: CaseSpec) -> argparse.Namespace:
    ns = copy.deepcopy(base)
    for key, value in case.args_updates.items():
        setattr(ns, key, value)
    if "sample_mode" not in case.args_updates:
        ns.sample_mode = family_sample_mode(case.family, getattr(ns, "suite_profile", "natural"))
    return ns


def topabs_coordinate_basis(vec: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    vec = np.asarray(vec, dtype=float).reshape(-1)
    n = vec.shape[0]
    if rank <= 0 or rank >= n:
        idx = np.arange(n, dtype=int)
        return np.eye(n, dtype=float), idx
    order = np.argsort(-np.abs(vec))
    idx = np.asarray(order[:rank], dtype=int)
    U = np.zeros((n, len(idx)), dtype=float)
    U[idx, np.arange(len(idx))] = 1.0
    return U, idx


def local_edge_reward_basis(problem: OriginalFixedXProblem, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    rank = int(rank)
    if rank <= 0:
        return np.zeros((problem.n_vars, 0), dtype=float), np.zeros(0, dtype=int)
    stdlp = build_standard_form_problem(problem)
    c0_std = standard_cost_from_reward(problem.reward_center, stdlp.n_slack)
    _x_center, _B0, Delta = enumerate_center_directions(stdlp, c0_std)
    if Delta.size == 0:
        return np.zeros((problem.n_vars, 0), dtype=float), np.zeros(0, dtype=int)
    tau = np.maximum(Delta.T @ np.asarray(c0_std, dtype=float), 0.0) / np.maximum(np.linalg.norm(Delta, axis=0), 1e-12)
    orig = np.asarray(Delta[stdlp.n_slack :, :], dtype=float)
    norms = np.linalg.norm(orig, axis=0)
    good = np.flatnonzero(norms > 1e-10)
    if good.size == 0:
        return np.zeros((problem.n_vars, 0), dtype=float), np.zeros(0, dtype=int)
    order = good[np.argsort(tau[good], kind="stable")]
    pool = order[: max(rank, min(len(order), 4 * rank))]
    mat = orig[:, pool]
    target = min(rank, int(np.linalg.matrix_rank(mat)))
    if target <= 0:
        return np.zeros((problem.n_vars, 0), dtype=float), np.zeros(0, dtype=int)
    try:
        chosen_local = choose_independent_columns(mat, target=target)
        chosen = pool[chosen_local]
    except Exception:
        chosen = pool[:target]
    U = orig[:, chosen]
    return orthonormalize_columns(U), np.asarray(chosen, dtype=int)


def make_netlib_problem(case: CaseSpec, args: argparse.Namespace, rng: np.random.Generator) -> OriginalFixedXProblem:
    del rng  # the current Netlib fixed-X construction is deterministic from the MPS file
    data = load_mps_fixed_feasible_data(case.mps_path)
    reward_center = np.asarray(data["reward_base"], dtype=float).reshape(-1)
    if np.linalg.norm(reward_center) <= 1e-10:
        raise ValueError(f"Degenerate Netlib reward center for {case.mps_path}")

    rank = int(getattr(args, "netlib_cost_rank", 12))
    basis_mode = str(getattr(args, "netlib_basis_mode", "topabs")).lower()
    if basis_mode == "topabs":
        U_reward, top_idx = topabs_coordinate_basis(reward_center, rank)
        basis_meta = {"netlib_topabs_indices": top_idx}
    elif basis_mode == "local_edge":
        temp_problem = OriginalFixedXProblem(
            name=f"{Path(case.mps_path).stem.lower()}_basis_temp",
            reward_center=reward_center,
            A_ineq=np.asarray(data["A_ineq"], dtype=float),
            b_ineq=np.asarray(data["b_ineq"], dtype=float).reshape(-1),
            A_eq=np.asarray(data["A_eq"], dtype=float),
            b_eq=np.asarray(data["b_eq"], dtype=float).reshape(-1),
            ub=None,
            U_reward=np.zeros((reward_center.shape[0], 0), dtype=float),
            metadata={"anchor_x0": np.asarray(data["x0"], dtype=float).reshape(-1)},
        )
        U_reward, chosen_idx = local_edge_reward_basis(temp_problem, rank)
        basis_meta = {"netlib_local_edge_indices": chosen_idx}
    elif basis_mode == "ray":
        u = reward_center / max(np.linalg.norm(reward_center), 1e-12)
        U_reward = u.reshape(-1, 1)
        basis_meta = {"netlib_topabs_indices": np.asarray([], dtype=int)}
    else:
        raise ValueError(f"Unsupported netlib_basis_mode={basis_mode!r}")
    U_reward = orthonormalize_columns(U_reward)

    stem = Path(case.mps_path).stem.lower()
    anchor_x0 = np.asarray(data["x0"], dtype=float).reshape(-1)
    problem = OriginalFixedXProblem(
        name=f"{stem}_fixedX",
        reward_center=reward_center,
        A_ineq=np.asarray(data["A_ineq"], dtype=float),
        b_ineq=np.asarray(data["b_ineq"], dtype=float).reshape(-1),
        A_eq=np.asarray(data["A_eq"], dtype=float),
        b_eq=np.asarray(data["b_eq"], dtype=float).reshape(-1),
        ub=None,
        U_reward=U_reward,
        metadata={
            "anchor_x0": anchor_x0,
            "mps_path": str(case.mps_path),
            "netlib_basis_mode": basis_mode,
            "netlib_basis_rank": int(U_reward.shape[1]),
            **basis_meta,
        },
    )
    if str(getattr(args, "netlib_anchor_mode", "feasible")).lower() == "center_optimal":
        inst_center = build_lp_instance(problem, reward_center, name=f"{stem}_center_anchor")
        ensure_full_solutions([inst_center], force=True)
        problem.metadata["anchor_x0"] = np.asarray(inst_center.full_x, dtype=float).reshape(-1)
    return problem


def build_problem_for_case(case: CaseSpec, args: argparse.Namespace, rng: np.random.Generator) -> OriginalFixedXProblem:
    if case.family == "netlib":
        return make_netlib_problem(case, args, rng)
    args.dataset = case.family
    return build_problem(args, rng)


def prepare_case_data(case: CaseSpec, base_args: argparse.Namespace) -> PreparedCaseData:
    args = case_namespace(base_args, case)
    rng = np.random.default_rng(int(args.seed) + int(case.seed_offset))
    problem = build_problem_for_case(case, args, rng)
    bundle = make_fixed_x_bundle(problem, args, rng)

    ensure_full_solutions(bundle.train)
    ensure_full_solutions(bundle.val)
    ensure_full_solutions(bundle.test)

    proj_train = bridge_instances_for_projection(problem, bundle.train)
    proj_val = bridge_instances_for_projection(problem, bundle.val)
    proj_test = bridge_instances_for_projection(problem, bundle.test)
    ensure_full_solutions(proj_train)
    ensure_full_solutions(proj_val)
    ensure_full_solutions(proj_test)

    return PreparedCaseData(
        case=case,
        args=args,
        problem=problem,
        bundle=bundle,
        proj_train=proj_train,
        proj_val=proj_val,
        proj_test=proj_test,
    )


def save_case_metadata(data: PreparedCaseData, out_dir: Path, train_limit: int, intrinsic_rank: int) -> None:
    meta = {
        "case": data.case.slug,
        "title": data.case.title,
        "family": data.case.family,
        "mps_path": data.case.mps_path,
        "n_train": int(train_limit),
        "n_val": int(len(data.bundle.val)),
        "n_test": int(len(data.bundle.test)),
        "k_list": parse_k_list(data.args.k_list),
        "prior_nominal_k": int(data.args.prior_nominal_k),
        "prior_origin_margin": float(data.bundle.prior.origin_margin),
        "prior_sample_radius": float(data.bundle.prior.sample_radius),
        "intrinsic_rank_ours": int(intrinsic_rank),
        "projection_family_dim": int(data.proj_train[0].n_vars),
        "original_family_dim": int(data.problem.n_vars),
    }
    (out_dir / "case_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_case_result(case: CaseSpec, case_dir: Path) -> Dict[str, pd.DataFrame]:
    summary = pd.read_csv(case_dir / "summary_results.csv")
    summary["case"] = case.slug
    summary["title"] = case.title
    summary["family"] = case.family

    quality_path = case_dir / "quality_summary.csv"
    quality_df = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()
    if not quality_df.empty:
        quality_df["case"] = case.slug
        quality_df["title"] = case.title
        quality_df["family"] = case.family

    rank_path = case_dir / "ours_rank_after_sample.csv"
    rank_df = pd.read_csv(rank_path) if rank_path.exists() else pd.DataFrame(columns=["sample", "rank_after_sample"])
    if not rank_df.empty:
        rank_df["case"] = case.slug
        rank_df["title"] = case.title
        rank_df["family"] = case.family

    fit_path = case_dir / "fit_times.csv"
    fit_df = pd.read_csv(fit_path) if fit_path.exists() else pd.DataFrame()
    if not fit_df.empty:
        fit_df["case"] = case.slug
        fit_df["title"] = case.title
        fit_df["family"] = case.family
    return {"summary": summary, "quality": quality_df, "rank": rank_df, "fit": fit_df}


def run_case_from_prepared(
    data: PreparedCaseData,
    out_dir: Path,
    methods: Sequence[str],
    train_limit: Optional[int] = None,
    force: bool = False,
    write_rank: bool = True,
) -> Dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not force and (out_dir / "summary_results.csv").exists():
        return load_case_result(data.case, out_dir)

    train_limit = len(data.bundle.train) if train_limit is None else int(train_limit)
    if train_limit <= 0 or train_limit > len(data.bundle.train):
        raise ValueError(f"Invalid train_limit={train_limit} for case {data.case.slug}")

    train = data.bundle.train[:train_limit]
    proj_train = data.proj_train[:train_limit]
    val = data.bundle.val
    proj_val = data.proj_val
    test = data.bundle.test
    proj_test = data.proj_test
    Ctrain = data.bundle.Ctrain[:train_limit]
    k_list = parse_k_list(data.args.k_list)

    rows: List[Dict[str, object]] = []
    fit_rows: List[Dict[str, object]] = []
    methods_set = set(methods)

    if "Full" in methods_set:
        append_full_rows(rows, proj_test, k_list)

    stage1 = None
    intrinsic_rank = -1
    if "OursExact" in methods_set or write_rank:
        t0 = time.perf_counter()
        stage1 = alg2_cumulative(data.bundle.stdlp, Ctrain, data.bundle.prior, verbose=bool(data.args.verbose))
        stage1_fit = time.perf_counter() - t0
        intrinsic_rank = int(stage1.U.shape[1])
        fit_rows.append(
            {
                "method": "OursStageI",
                "K": intrinsic_rank,
                "fit_time": stage1_fit,
                "train_samples": train_limit,
                "val_samples": len(val),
                "test_samples": len(test),
                "epochs": 0,
                "batch_size": 0,
                "notes": "Algorithm 2 cumulative learner on fixed-X cost samples",
            }
        )
        if "OursExact" in methods_set:
            append_ours_exact_rows(rows, data.bundle.stdlp, stage1.U, stage1.x_anchor, test, comparison_ks=k_list)
        if write_rank:
            np.savez(
                out_dir / "ours_stageI_learned_basis.npz",
                U=stage1.U,
                D=stage1.D,
                x_anchor=stage1.x_anchor,
                rank_after_sample=np.asarray(stage1.rank_after_sample, dtype=int),
                rank_after_query=np.asarray(stage1.rank_after_query, dtype=int),
                hard_indices=np.asarray(stage1.hard_indices, dtype=int),
            )
            pd.DataFrame(
                {
                    "sample": np.arange(0, len(stage1.rank_after_sample) + 1),
                    "rank_after_sample": np.concatenate([[0], np.asarray(stage1.rank_after_sample, dtype=int)]),
                }
            ).to_csv(out_dir / "ours_rank_after_sample.csv", index=False)
            pd.DataFrame(
                {"query": np.arange(1, len(stage1.rank_after_query) + 1), "rank_after_query": stage1.rank_after_query}
            ).to_csv(out_dir / "ours_rank_after_query.csv", index=False)

    need_general = any(m in methods_set for m in ("PCA", "SGA"))
    if need_general:
        general_train = [convert_to_general_projection_instance(inst) for inst in proj_train]
        general_test = [convert_to_general_projection_instance(inst) for inst in proj_test]
        general_common_A = np.asarray(general_train[0].A, dtype=float)
        general_common_b = np.asarray(general_train[0].b, dtype=float)
        feasible_hint = np.asarray(proj_train[0].full_x, dtype=float).reshape(-1)
    else:
        general_train = []
        general_test = []
        general_common_A = np.zeros((0, 0), dtype=float)
        general_common_b = np.zeros(0, dtype=float)
        feasible_hint = np.zeros(0, dtype=float)

    if "CostOnly" in methods_set:
        signed_cost_data = prepare_signed_nullspace_data(data)
        signed_train = signed_cost_data.train[:train_limit]
        signed_val = signed_cost_data.val
        signed_test = signed_cost_data.test
    else:
        signed_cost_data = None
        signed_train = []
        signed_val = []
        signed_test = []

    need_neural = any(m in methods_set for m in ("CostOnly", "FCNN", "PELP"))
    if need_neural:
        device = torch.device(str(data.args.device))
        n_cons_eff = proj_train[0].n_feature_cons
    else:
        device = torch.device("cpu")
        n_cons_eff = 0

    for k in k_list:
        if "Rand" in methods_set:
            for tr in range(int(data.args.rand_trials)):
                seed = int(data.args.seed) + int(data.case.seed_offset) + 10000 + 97 * tr + int(k) + 31 * train_limit
                P_rand = random_column_projection(proj_train[0].n_vars, int(k), np.random.default_rng(seed))
                append_fixed_projection_rows(rows, f"Rand#{tr+1}", P_rand, proj_test, int(k))
            fit_rows.append(
                {
                    "method": "Rand",
                    "K": int(k),
                    "fit_time": 0.0,
                    "train_samples": train_limit,
                    "val_samples": len(val),
                    "test_samples": len(test),
                    "epochs": 0,
                    "batch_size": 0,
                    "notes": f"{data.args.rand_trials} random coordinate projectors",
                }
            )

        if "PCA" in methods_set:
            t_fit = time.perf_counter()
            P_pca = raw_pca_projection(proj_train, int(k))
            pca_fit = time.perf_counter() - t_fit
            append_general_projection_rows(rows, "PCA", P_pca, general_test, int(k))
            np.save(out_dir / f"PCA_K{k}.npy", P_pca)
            fit_rows.append(
                {
                    "method": "PCA",
                    "K": int(k),
                    "fit_time": pca_fit,
                    "train_samples": train_limit,
                    "val_samples": len(val),
                    "test_samples": len(test),
                    "epochs": 1,
                    "batch_size": train_limit,
                    "notes": "Raw signed PCA projection without nonnegative truncation",
                }
            )

        if "SGA" in methods_set:
            t_fit = time.perf_counter()
            P_sga, hist_sga = train_sga_final_projection(
                proj_train,
                proj_val,
                k=int(k),
                epochs=int(data.args.sga_epochs),
                batch_size=int(data.args.batch_size),
                lr=float(data.args.sga_lr),
                seed=int(data.args.seed) + int(data.case.seed_offset) + 2000 + int(k) + 53 * train_limit,
                verbose=bool(data.args.verbose),
            )
            sga_fit = time.perf_counter() - t_fit
            append_general_projection_rows(rows, "SGA", P_sga, general_test, int(k))
            np.save(out_dir / f"SGA_K{k}.npy", P_sga)
            save_training_history(hist_sga, out_dir / f"history_SGA_K{k}.csv")
            fit_rows.append(
                {
                    "method": "SGA",
                    "K": int(k),
                    "fit_time": sga_fit,
                    "train_samples": train_limit,
                    "val_samples": len(val),
                    "test_samples": len(test),
                    "epochs": int(data.args.sga_epochs),
                    "batch_size": int(data.args.batch_size),
                    "notes": "PCA-init SGA with final projection (Sakaue--Oki 2024 style)",
                }
            )

        if "FCNN" in methods_set:
            t_fit = time.perf_counter()
            fcnn = FCNNProjectionNet(proj_train[0].n_vars, n_cons_eff, int(k), int(data.args.hidden_dim))
            fcnn, hist_fcnn = train_implicit_projection_model(
                fcnn,
                proj_train,
                proj_val,
                epochs=int(data.args.epochs),
                batch_size=int(data.args.batch_size),
                lr=float(data.args.lr),
                device=device,
                seed=int(data.args.seed) + int(data.case.seed_offset) + 3000 + int(k) + 59 * train_limit,
                patience=int(data.args.patience),
                verbose=bool(data.args.verbose),
            )
            fcnn_fit = time.perf_counter() - t_fit
            append_learned_projector_rows(rows, "FCNN", fcnn, proj_test, int(k), device)
            save_training_history(hist_fcnn, out_dir / f"history_FCNN_K{k}.csv")
            torch.save(fcnn.state_dict(), out_dir / f"FCNN_K{k}.pt")
            fit_rows.append(
                {
                    "method": "FCNN",
                    "K": int(k),
                    "fit_time": fcnn_fit,
                    "train_samples": train_limit,
                    "val_samples": len(val),
                    "test_samples": len(test),
                    "epochs": int(data.args.epochs),
                    "batch_size": int(data.args.batch_size),
                    "notes": "Iwata--Sakaue 2025 FCNN baseline on preprocessed LP family",
                }
            )

        if "CostOnly" in methods_set and signed_cost_data is not None:
            t_fit = time.perf_counter()
            init_mode = str(getattr(data.args, "costonly_init", "cost_pca")).lower()
            solver_space = str(getattr(data.args, "costonly_solver_space", "auto")).lower()
            direct_mode = (solver_space != "nullspace") and (not should_bridge_for_projection(data.problem))
            if direct_mode:
                if data.problem.A_eq.size > 0:
                    cost_train = transform_instances_to_bridge(data.problem, data.bundle.train[:train_limit])
                    cost_val = transform_instances_to_bridge(data.problem, data.bundle.val)
                    cost_test = transform_instances_to_bridge(data.problem, data.bundle.test)
                    ensure_full_solutions(cost_train)
                    ensure_full_solutions(cost_val)
                    ensure_full_solutions(cost_test)
                    init_note = "signed c-only projector on equality-removed flow bridge"
                else:
                    cost_train = add_explicit_nonnegativity(proj_train[:train_limit])
                    cost_val = add_explicit_nonnegativity(proj_val)
                    cost_test = add_explicit_nonnegativity(proj_test)
                    init_note = "signed direct c-only projector with explicit x>=0 inequalities"
                cost_only = SignedDirectCostOnlyProjectionNet(
                    cost_train[0].n_vars,
                    int(k),
                    int(getattr(data.args, "costonly_hidden_dim", 8)),
                )
                P0 = direct_costonly_costspace_matrix(cost_train, cost_train[0].n_vars, int(k))
                initialize_signed_costonly_model(cost_only, P0)
                cost_only, hist_cost = train_implicit_projection_model(
                    cost_only,
                    cost_train,
                    cost_val,
                    epochs=int(getattr(data.args, "costonly_epochs", data.args.epochs)),
                    batch_size=int(getattr(data.args, "costonly_batch_size", data.args.batch_size)),
                    lr=float(getattr(data.args, "costonly_lr", data.args.lr)),
                    device=device,
                    seed=int(data.args.seed) + int(data.case.seed_offset) + 3500 + int(k) + 61 * train_limit,
                    patience=int(getattr(data.args, "costonly_patience", data.args.patience)),
                    verbose=bool(data.args.verbose),
                )
                append_learned_projector_rows(rows, "CostOnly", cost_only, cost_test, int(k), device)
            else:
                cost_only = SignedCostOnlyProjectionNet(
                    signed_cost_data.N.shape[1],
                    signed_cost_data.N.shape[1],
                    int(k),
                    int(getattr(data.args, "costonly_hidden_dim", 8)),
                )
                if bool(getattr(data.args, "costonly_use_warmstart", False)) or init_mode == "warmstart":
                    P0 = signed_costonly_warmstart_matrix(data, signed_cost_data, int(k))
                elif init_mode == "fewopt":
                    P0 = signed_costonly_fewopt_matrix(
                        data,
                        signed_cost_data,
                        int(k),
                        int(getattr(data.args, "costonly_warm_cols", 2)),
                    )
                elif init_mode == "cost_pca":
                    P0 = signed_costonly_costspace_matrix(signed_train, signed_cost_data.N.shape[1], int(k))
                else:
                    P0 = signed_costonly_random_feasible_matrix(
                        signed_cost_data,
                        int(k),
                        seed=int(data.args.seed) + int(data.case.seed_offset) + 3300 + int(k) + 41 * train_limit,
                    )
                initialize_signed_costonly_model(cost_only, P0)
                cost_only, hist_cost = train_signed_costonly_model(
                    cost_only,
                    signed_train,
                    signed_val,
                    signed_cost_data.N,
                    signed_cost_data.x_ref,
                    epochs=int(getattr(data.args, "costonly_epochs", data.args.epochs)),
                    batch_size=int(getattr(data.args, "costonly_batch_size", data.args.batch_size)),
                    lr=float(getattr(data.args, "costonly_lr", data.args.lr)),
                    device=device,
                    seed=int(data.args.seed) + int(data.case.seed_offset) + 3500 + int(k) + 61 * train_limit,
                    patience=int(getattr(data.args, "costonly_patience", data.args.patience)),
                    verbose=bool(data.args.verbose),
                )
                append_signed_costonly_rows(
                    rows,
                    cost_only,
                    signed_test,
                    signed_cost_data.N,
                    signed_cost_data.x_ref,
                    int(k),
                    device,
                    data.bundle.stdlp.n_slack,
                )
                init_note = f"signed nullspace c-only projector (init={init_mode})"
            cost_fit = time.perf_counter() - t_fit
            save_training_history(hist_cost, out_dir / f"history_CostOnly_K{k}.csv")
            torch.save(cost_only.state_dict(), out_dir / f"CostOnly_K{k}.pt")
            fit_rows.append(
                {
                    "method": "CostOnly",
                    "K": int(k),
                    "fit_time": cost_fit,
                    "train_samples": train_limit,
                    "val_samples": len(val),
                    "test_samples": len(test),
                    "epochs": int(getattr(data.args, "costonly_epochs", data.args.epochs)),
                    "batch_size": int(getattr(data.args, "costonly_batch_size", data.args.batch_size)),
                    "notes": f"{init_note} (light budget)",
                }
            )

        if "PELP" in methods_set:
            t_fit = time.perf_counter()
            pelp = PELPProjectionNet(
                k=int(k),
                hidden_dim=int(data.args.hidden_dim),
                n_layers=int(data.args.n_pelp_layers),
                generator_hidden_dim=int(data.args.generator_hidden_dim),
            )
            pelp, hist_pelp = train_implicit_projection_model(
                pelp,
                proj_train,
                proj_val,
                epochs=int(data.args.epochs),
                batch_size=int(data.args.batch_size),
                lr=float(data.args.lr),
                device=device,
                seed=int(data.args.seed) + int(data.case.seed_offset) + 4000 + int(k) + 67 * train_limit,
                patience=int(data.args.patience),
                verbose=bool(data.args.verbose),
            )
            pelp_fit = time.perf_counter() - t_fit
            append_learned_projector_rows(rows, "PELP", pelp, proj_test, int(k), device)
            save_training_history(hist_pelp, out_dir / f"history_PELP_K{k}.csv")
            torch.save(pelp.state_dict(), out_dir / f"PELP_K{k}.pt")
            fit_rows.append(
                {
                    "method": "PELP",
                    "K": int(k),
                    "fit_time": pelp_fit,
                    "train_samples": train_limit,
                    "val_samples": len(val),
                    "test_samples": len(test),
                    "epochs": int(data.args.epochs),
                    "batch_size": int(data.args.batch_size),
                    "notes": "Iwata--Sakaue 2025 PELP baseline on preprocessed LP family",
                }
            )

    raw = pd.DataFrame(rows)
    raw["method"] = raw["method"].str.replace(r"Rand#\d+", "Rand", regex=True)
    raw["case"] = data.case.slug
    raw["title"] = data.case.title
    raw["family"] = data.case.family
    raw.to_csv(out_dir / "raw_results.csv", index=False)

    summary = summarize_ratios_and_times(raw.to_dict("records"))
    summary["case"] = data.case.slug
    summary["title"] = data.case.title
    summary["family"] = data.case.family
    summary.to_csv(out_dir / "summary_results.csv", index=False)

    base_obj_by_instance = {inst.name: float(getattr(inst, "objective_constant", 0.0)) for inst in proj_test}
    quality_df = summarize_anchor_quality(raw, base_obj_by_instance)
    if not quality_df.empty:
        quality_df["case"] = data.case.slug
        quality_df["title"] = data.case.title
        quality_df["family"] = data.case.family
    quality_df.to_csv(out_dir / "quality_summary.csv", index=False)

    fit_df = pd.DataFrame(fit_rows)
    if not fit_df.empty:
        fit_df["case"] = data.case.slug
        fit_df["title"] = data.case.title
        fit_df["family"] = data.case.family
    fit_df.to_csv(out_dir / "fit_times.csv", index=False)

    save_case_metadata(data, out_dir, train_limit=train_limit, intrinsic_rank=max(intrinsic_rank, -1))
    rank_df = pd.read_csv(out_dir / "ours_rank_after_sample.csv") if (out_dir / "ours_rank_after_sample.csv").exists() else pd.DataFrame()
    if not rank_df.empty:
        rank_df["case"] = data.case.slug
        rank_df["title"] = data.case.title
        rank_df["family"] = data.case.family
    return {"summary": summary, "quality": quality_df, "rank": rank_df, "fit": fit_df}


def synthetic_cases(base_root: Path) -> List[CaseSpec]:
    del base_root
    return [
        CaseSpec(
            "packing",
            "Packing",
            "packing",
            {
                "n_vars": 360,
                "n_cons": 60,
                "cost_rank": 28,
                "sample_mode": "factor_gaussian",
                "factor_scale_frac": 0.92,
                "factor_decay": 0.94,
                "center_noise_frac": 0.0,
                "sample_radius_frac": 0.90,
                "prior_floor_frac": 0.04,
            },
            sample_k=20,
            seed_offset=0,
        ),
        CaseSpec(
            "maxflow",
            "MaxFlow",
            "maxflow",
            {
                "n_nodes": 64,
                "n_edges": 300,
                "cost_rank": 12,
                "sample_mode": "factor_gaussian",
                "factor_scale_frac": 0.95,
                "factor_decay": 1.0,
                "center_noise_frac": 0.0,
                "sample_radius_frac": 0.90,
                "costonly_solver_space": "nullspace",
                "costonly_init": "fewopt",
                "costonly_warm_cols": 2,
            },
            sample_k=20,
            seed_offset=100,
        ),
        CaseSpec(
            "mincostflow",
            "MinCostFlow",
            "mincostflow",
            {
                "n_nodes": 86,
                "n_edges": 360,
                "cost_rank": 28,
                "sample_mode": "factor_gaussian",
                "factor_scale_frac": 0.72,
                "factor_decay": 0.94,
                "sample_radius_frac": 0.88,
                "prior_floor_frac": 0.04,
            },
            sample_k=20,
            seed_offset=200,
        ),
        CaseSpec(
            "shortest_path",
            "ShortestPath",
            "shortest_path",
            {
                "grid_size": 16,
                "cost_rank": 24,
            },
            sample_k=20,
            seed_offset=300,
        ),
        CaseSpec(
            "random_lp_A",
            "RandomLP A",
            "random_lp",
            {
                "randlp_n_vars": 140,
                "randlp_n_eq": 30,
                "randlp_n_ineq": 40,
                "prior_nominal_k": 24,
                "cost_rank": 16,
                "randlp_basis_mode": "local_edge",
                "randlp_pool_factor": 5,
                "sample_mode": "factor_regime_mixture",
                "sample_radius_frac": 0.98,
                "prior_floor_frac": 0.08,
                "regime_shift_frac": 0.90,
                "regime_common_frac": 0.08,
                "regime_noise_frac": 0.10,
                "regime_decay": 0.96,
            },
            sample_k=20,
            seed_offset=350,
        ),
        CaseSpec(
            "random_lp_B",
            "RandomLP B",
            "random_lp",
            {
                "randlp_n_vars": 180,
                "randlp_n_eq": 42,
                "randlp_n_ineq": 56,
                "prior_nominal_k": 24,
                "cost_rank": 16,
                "randlp_basis_mode": "local_edge",
                "randlp_pool_factor": 5,
                "sample_mode": "factor_regime_mixture",
                "sample_radius_frac": 0.98,
                "prior_floor_frac": 0.08,
                "regime_shift_frac": 0.90,
                "regime_common_frac": 0.08,
                "regime_noise_frac": 0.10,
                "regime_decay": 0.96,
            },
            sample_k=20,
            seed_offset=450,
        ),
        CaseSpec(
            "random_lp_C",
            "RandomLP C",
            "random_lp",
            {
                "randlp_n_vars": 220,
                "randlp_n_eq": 54,
                "randlp_n_ineq": 72,
                "prior_nominal_k": 24,
                "cost_rank": 16,
                "randlp_basis_mode": "local_edge",
                "randlp_pool_factor": 5,
                "sample_mode": "factor_regime_mixture",
                "sample_radius_frac": 0.98,
                "prior_floor_frac": 0.08,
                "regime_shift_frac": 0.90,
                "regime_common_frac": 0.08,
                "regime_noise_frac": 0.10,
                "regime_decay": 0.96,
            },
            sample_k=20,
            seed_offset=550,
        ),
        CaseSpec(
            "random_lp_D",
            "RandomLP D",
            "random_lp",
            {
                "randlp_n_vars": 260,
                "randlp_n_eq": 66,
                "randlp_n_ineq": 88,
                "prior_nominal_k": 24,
                "cost_rank": 16,
                "randlp_basis_mode": "local_edge",
                "randlp_pool_factor": 5,
                "sample_mode": "factor_regime_mixture",
                "sample_radius_frac": 0.98,
                "prior_floor_frac": 0.08,
                "regime_shift_frac": 0.90,
                "regime_common_frac": 0.08,
                "regime_noise_frac": 0.10,
                "regime_decay": 0.96,
            },
            sample_k=20,
            seed_offset=650,
        ),
    ]


def netlib_cases(base_root: Path) -> List[CaseSpec]:
    data_root = base_root / "data" / "netlib" / "selected"
    common = {
        "n_train": 24,
        "n_val": 6,
        "n_test": 12,
        "k_list": "5,10,20,30,40,50",
        "hidden_dim": 16,
        "generator_hidden_dim": 16,
        "prior_nominal_k": 24,
        "sample_radius_frac": 0.92,
        "center_noise_frac": 0.006,
        "netlib_cost_rank": 24,
    }
    names = [
        (
            "grow7",
            "GROW7",
            1000,
            {
                "sample_mode": "factor_regime_mixture",
                "sample_radius_frac": 0.95,
                "prior_floor_frac": 0.06,
                "regime_shift_frac": 0.80,
                "regime_common_frac": 0.05,
                "regime_noise_frac": 0.06,
                "regime_decay": 0.98,
                "center_noise_frac": 0.0,
                "netlib_cost_rank": 12,
                "netlib_basis_mode": "local_edge",
                "netlib_anchor_mode": "feasible",
                "costonly_epochs": 8,
                "costonly_hidden_dim": 16,
            },
        ),
        (
            "sc205",
            "SC205",
            1200,
            {
                "sample_mode": "factor_gaussian",
                "sample_radius_frac": 0.92,
                "center_noise_frac": 0.0,
                "netlib_cost_rank": 20,
                "netlib_basis_mode": "local_edge",
                "netlib_anchor_mode": "feasible",
                "costonly_epochs": 8,
                "costonly_hidden_dim": 16,
            },
        ),
        ("scagr25", "SCAGR25", 1300, {}),
        ("stair", "STAIR", 1400, {"sample_radius_frac": 0.02, "netlib_cost_rank": 1, "netlib_basis_mode": "ray"}),
    ]
    out = []
    for stem, title, seed_offset, extra in names:
        cfg = dict(common)
        cfg.update(extra)
        out.append(
            CaseSpec(
                slug=stem,
                title=title,
                family="netlib",
                args_updates=cfg,
                sample_k=20,
                seed_offset=seed_offset,
                mps_path=str(data_root / f"{stem}.mps"),
            )
        )
    return out


def run_or_load_k_sweep_case(
    case: CaseSpec,
    root_out: Path,
    base_args: argparse.Namespace,
    force: bool,
) -> Dict[str, pd.DataFrame]:
    case_out = root_out / "k_sweep" / case.slug
    if not force and (case_out / "summary_results.csv").exists():
        return load_case_result(case, case_out)
    if case.reuse_dir and not force and (Path(case.reuse_dir) / "summary_results.csv").exists():
        return load_case_result(case, Path(case.reuse_dir))
    data = prepare_case_data(case, base_args)
    return run_case_from_prepared(data, case_out, methods=METHOD_ORDER, train_limit=len(data.bundle.train), force=force, write_rank=True)


def figure_bounds(df: pd.DataFrame, ycol: str, yerrcol: Optional[str] = None) -> Tuple[float, float]:
    if df.empty:
        return 0.0, 1.0
    if ycol in {"objective_ratio_mean", "anchor_quality_mean"}:
        # Keep ratio/quality figures on a stable visual scale so a few noisy
        # ribbons do not dominate the whole panel layout.
        return -0.02, 1.05
    y = df[ycol].astype(float).to_numpy()
    if yerrcol is not None and yerrcol in df.columns:
        err = df[yerrcol].astype(float).to_numpy()
    else:
        err = np.zeros_like(y)
    lo = np.nanmin(y - err)
    hi = np.nanmax(y + err)
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 1.0
    if lo >= hi:
        return lo - 0.05, hi + 0.05
    pad = 0.05 * max(hi - lo, 1e-6)
    return float(lo - pad), float(hi + pad)


def clipped_band(y: np.ndarray, yerr: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    band = np.asarray(yerr, dtype=float).copy()
    if hi <= 1.1 and lo >= -0.05:
        # Robustness plots can have only 2 seeds; cap the displayed ribbon width
        # so a single unstable point does not visually swamp the whole panel.
        band = np.minimum(band, 0.25)
    lower = np.clip(y - band, lo, hi)
    upper = np.clip(y + band, lo, hi)
    return lower, upper


def plot_metric_grid(
    df: pd.DataFrame,
    case_order: Sequence[str],
    title_map: Dict[str, str],
    out_path: Path,
    methods: Sequence[str] = METHOD_ORDER,
    x_col: str = "K",
    y_col: str = "objective_ratio_mean",
    yerr_col: str = "objective_ratio_se",
    xlabel: str = "Reduced dimension K",
    ylabel: str = "Average test objective ratio",
) -> None:
    if df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [c for c in case_order if c in set(df["case"])]
    if not cases:
        return
    n = len(cases)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    lo, hi = figure_bounds(df, y_col, yerr_col)

    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        subcase = df[df["case"] == case]
        xticks: List[int] = []
        for method in methods:
            sub = subcase[subcase["method"] == method].sort_values(x_col)
            if sub.empty:
                continue
            style = METHOD_STYLE.get(method, {"color": None, "marker": "o", "linestyle": "-"})
            x = sub[x_col].astype(float).to_numpy()
            xticks.extend([int(round(v)) for v in x.tolist()])
            y = sub[y_col].astype(float).to_numpy()
            yerr = sub[yerr_col].astype(float).to_numpy()
            ylo, yhi = clipped_band(y, yerr, lo, hi)
            ax.fill_between(x, ylo, yhi, color=style["color"], alpha=0.08, linewidth=0.0)
            ax.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.9 if method != "OursExact" else 2.2,
                markersize=5.0,
                markeredgewidth=0.9,
            )
        ax.set_title(f"({chr(97 + idx)}) {title_map[case]}")
        ax.set_xlabel(xlabel)
        ax.set_ylim(lo, hi)
        if xticks:
            xt = sorted(set(xticks))
            ax.set_xticks(xt)
            ax.set_xlim(min(xt), max(xt))
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    for row in axes_arr:
        row[0].set_ylabel(ylabel)
    handles = []
    labels = []
    if cases:
        first_ax = axes_arr[0, 0]
        for method in methods:
            style = METHOD_STYLE.get(method, {"color": None, "marker": "o", "linestyle": "-"})
            label = METHOD_DISPLAY.get(method, method)
            handle = first_ax.plot([], [], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], label=label)[0]
            handles.append(handle)
            labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 8), frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_rank_grid(
    df: pd.DataFrame,
    case_order: Sequence[str],
    title_map: Dict[str, str],
    out_path: Path,
    isolate_case: Optional[str] = None,
    per_case_scale: bool = False,
) -> None:
    if df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [c for c in case_order if c in set(df["case"])]
    if not cases:
        return
    def _draw_rank_axis(ax, case: str, title_idx: int, ymax: float) -> None:
        sub = df[(df["case"] == case) & (df["sample"] <= RANK_CURVE_MAX_SAMPLES)].sort_values("sample")
        ax.plot(
            sub["sample"],
            sub["rank_after_sample"],
            color=METHOD_STYLE["OursExact"]["color"],
            marker="o",
            linestyle=METHOD_STYLE["OursExact"]["linestyle"],
            linewidth=2.0,
            markersize=4.8,
        )
        ax.set_title(f"({chr(97 + title_idx)}) {title_map[case]}")
        ax.set_xlabel("# training samples processed")
        ax.set_xlim(0, RANK_CURVE_MAX_SAMPLES)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(0.0, ymax)
        ax.grid(True, alpha=0.25)

    if per_case_scale:
        n = len(cases)
        ncols = min(4, n)
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.25 * nrows), sharey=False)
        axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

        for idx, case in enumerate(cases):
            ax = axes_arr[idx // ncols, idx % ncols]
            case_sub = df[(df["case"] == case) & (df["sample"] <= RANK_CURVE_MAX_SAMPLES)]
            ymax_case = max(1.0, float(case_sub["rank_after_sample"].max()) + 0.5)
            _draw_rank_axis(ax, case, idx, ymax_case)
            if idx % ncols == 0:
                ax.set_ylabel("Learned dimension")
        for idx in range(len(cases), nrows * ncols):
            axes_arr[idx // ncols, idx % ncols].axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    if isolate_case is not None and isolate_case in cases and len(cases) > 1:
        lead_case = str(isolate_case)
        rest_cases = [c for c in cases if c != lead_case]
        n_other = len(rest_cases)
        ncols = min(4, max(1, n_other))
        nrows_other = int(math.ceil(n_other / ncols))
        fig = plt.figure(figsize=(4.2 * ncols, 3.4 * (nrows_other + 1)))
        gs = fig.add_gridspec(nrows_other + 1, ncols)

        ymax_lead = max(
            1.0,
            float(
                df.loc[
                    (df["case"] == lead_case) & (df["sample"] <= RANK_CURVE_MAX_SAMPLES),
                    "rank_after_sample",
                ].max()
            )
            + 0.5,
        )
        ymax_rest = max(
            1.0,
            float(
                df.loc[
                    (df["case"].isin(rest_cases)) & (df["sample"] <= RANK_CURVE_MAX_SAMPLES),
                    "rank_after_sample",
                ].max()
            )
            + 0.5,
        )

        ax0 = fig.add_subplot(gs[0, :])
        _draw_rank_axis(ax0, lead_case, 0, ymax_lead)
        ax0.set_ylabel("Learned dimension")

        shared_ax = None
        for idx, case in enumerate(rest_cases):
            row = 1 + idx // ncols
            col = idx % ncols
            ax = fig.add_subplot(gs[row, col], sharey=shared_ax)
            if shared_ax is None:
                shared_ax = ax
            _draw_rank_axis(ax, case, idx + 1, ymax_rest)
            if col == 0:
                ax.set_ylabel("Learned dimension")
        for idx in range(len(rest_cases), nrows_other * ncols):
            row = 1 + idx // ncols
            col = idx % ncols
            fig.add_subplot(gs[row, col]).axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return

    n = len(cases)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    ymax = max(1.0, float(df["rank_after_sample"].max()) + 0.5)

    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        _draw_rank_axis(ax, case, idx, ymax)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    for row in axes_arr:
        row[0].set_ylabel("Learned dimension")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_sample_efficiency_grid(
    df: pd.DataFrame,
    case_order: Sequence[str],
    title_map: Dict[str, str],
    k_map: Dict[str, int],
    out_path: Path,
    y_col: str = "objective_ratio_mean",
    yerr_col: str = "objective_ratio_se",
    ylabel: str = "Average test objective ratio",
) -> None:
    if df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [c for c in case_order if c in set(df["case"])]
    if not cases:
        return
    n = len(cases)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    lo, hi = figure_bounds(df, y_col, yerr_col)

    for idx, case in enumerate(cases):
        ax = axes_arr[idx // ncols, idx % ncols]
        subcase = df[df["case"] == case]
        xticks: List[int] = []
        for method in SAMPLE_METHOD_ORDER:
            sub = subcase[subcase["method"] == method].sort_values("n_train")
            if sub.empty:
                continue
            style = SAMPLE_METHOD_STYLE[method]
            x = sub["n_train"].astype(float).to_numpy()
            xticks.extend([int(round(v)) for v in x.tolist()])
            y = sub[y_col].astype(float).to_numpy()
            yerr = sub[yerr_col].astype(float).to_numpy()
            ylo, yhi = clipped_band(y, yerr, lo, hi)
            ax.fill_between(x, ylo, yhi, color=style["color"], alpha=0.08, linewidth=0.0)
            ax.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0 if method == "OursExact" else 1.9,
                markersize=5.0,
                markeredgewidth=0.9,
            )
        ax.set_title(f"({chr(97 + idx)}) {title_map[case]} (K={k_map[case]})")
        ax.set_xlabel("# training LPs")
        if xticks:
            xt = sorted(set(xticks))
            ax.set_xticks(xt)
            ax.set_xlim(min(xt), max(xt))
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25)
    for idx in range(len(cases), nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")
    for row in axes_arr:
        row[0].set_ylabel(ylabel)
    handles = []
    labels = []
    if cases:
        first_ax = axes_arr[0, 0]
        for method in SAMPLE_METHOD_ORDER:
            style = SAMPLE_METHOD_STYLE[method]
            label = METHOD_DISPLAY.get(method, method)
            handle = first_ax.plot([], [], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], label=label)[0]
            handles.append(handle)
            labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def sample_efficiency_cases(base_root: Path) -> List[CaseSpec]:
    synth = synthetic_cases(base_root)
    selected = list(synth)
    selected.extend(netlib_cases(base_root))
    return selected


def sample_train_grid(case: CaseSpec) -> List[int]:
    del case
    return [0, 1, 2, 3, 4, 8, 12, 16, 20]


def run_sample_efficiency_case(
    case: CaseSpec,
    root_out: Path,
    base_args: argparse.Namespace,
    force: bool,
) -> pd.DataFrame:
    case_out_root = root_out / "sample_efficiency" / case.slug
    summary_out = case_out_root / "summary_results.csv"
    if not force and summary_out.exists():
        return pd.read_csv(summary_out)
    grid = sample_train_grid(case)
    if not force:
        cached_rows = []
        all_cached = True
        for n_train in grid:
            sub_out = case_out_root / f"ntrain_{n_train:03d}"
            if not (sub_out / "summary_results.csv").exists():
                all_cached = False
                break
            result = load_case_result(case, sub_out)
            summary = result["summary"].copy()
            quality = result["quality"].copy()
            summary = summary[summary["method"].isin(SAMPLE_METHOD_ORDER)].copy()
            summary["n_train"] = int(n_train)
            summary["sample_k"] = int(case.sample_k)
            summary = summary[summary["K"] == int(case.sample_k)].copy()
            if not quality.empty:
                quality = quality[quality["method"].isin(SAMPLE_METHOD_ORDER)].copy()
                quality["n_train"] = int(n_train)
                quality["sample_k"] = int(case.sample_k)
                quality = quality[quality["K"] == int(case.sample_k)].copy()
                summary = summary.merge(
                    quality[["method", "K", "anchor_quality_mean", "anchor_quality_se"]],
                    on=["method", "K"],
                    how="left",
                )
            cached_rows.append(summary)
        if all_cached and cached_rows:
            out = pd.concat(cached_rows, ignore_index=True)
            case_out_root.mkdir(parents=True, exist_ok=True)
            out.to_csv(summary_out, index=False)
            return out

    max_train = max(grid)
    temp_case = copy.deepcopy(case)
    temp_case.args_updates = dict(case.args_updates)
    temp_case.args_updates["n_train"] = max_train
    # Sample-efficiency figures fix K and vary only the number of training
    # instances.  Running the full K sweep here is unnecessary and made this
    # part of the suite several times slower.
    temp_case.args_updates["k_list"] = str(int(case.sample_k))
    data = prepare_case_data(temp_case, base_args)

    rows = []
    for n_train in grid:
        sub_out = case_out_root / f"ntrain_{n_train:03d}"
        if int(n_train) == 0:
            sub_out.mkdir(parents=True, exist_ok=True)
            raw_rows = []
            for proj_inst, full_inst in zip(data.proj_test, data.bundle.test):
                anchor_obj = float(getattr(proj_inst, "objective_constant", 0.0))
                ratio = objective_ratio(anchor_obj, float(full_inst.full_obj))
                for method in SAMPLE_METHOD_ORDER:
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
                            "case": case.slug,
                            "title": case.title,
                            "family": case.family,
                        }
                    )
            raw_df = pd.DataFrame(raw_rows)
            raw_df.to_csv(sub_out / "raw_results.csv", index=False)
            result_summary = summarize_ratios_and_times(raw_df.to_dict("records"))
            result_summary["case"] = case.slug
            result_summary["title"] = case.title
            result_summary["family"] = case.family
            result_summary.to_csv(sub_out / "summary_results.csv", index=False)
            base_obj_by_instance = {inst.name: float(getattr(inst, "objective_constant", 0.0)) for inst in data.proj_test}
            result_quality = summarize_anchor_quality(raw_df, base_obj_by_instance)
            if not result_quality.empty:
                result_quality["case"] = case.slug
                result_quality["title"] = case.title
                result_quality["family"] = case.family
            result_quality.to_csv(sub_out / "quality_summary.csv", index=False)
            result = {"summary": result_summary, "quality": result_quality}
        else:
            result = run_case_from_prepared(
                data,
                sub_out,
                methods=SAMPLE_METHOD_ORDER,
                train_limit=n_train,
                force=force,
                write_rank=(n_train == max_train),
            )
        summary = result["summary"].copy()
        quality = result["quality"].copy()
        summary = summary[summary["method"].isin(SAMPLE_METHOD_ORDER)].copy()
        summary["n_train"] = int(n_train)
        summary["sample_k"] = int(case.sample_k)
        summary = summary[summary["K"] == int(case.sample_k)].copy()
        if not quality.empty:
            quality = quality[quality["method"].isin(SAMPLE_METHOD_ORDER)].copy()
            quality["n_train"] = int(n_train)
            quality["sample_k"] = int(case.sample_k)
            quality = quality[quality["K"] == int(case.sample_k)].copy()
            summary = summary.merge(
                quality[["method", "K", "anchor_quality_mean", "anchor_quality_se"]],
                on=["method", "K"],
                how="left",
            )
        rows.append(summary)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        case_out_root.mkdir(parents=True, exist_ok=True)
        out.to_csv(summary_out, index=False)
    return out


def write_suite_manifest(out_dir: Path, synthetic: Sequence[CaseSpec], netlib: Sequence[CaseSpec], sample_cases: Sequence[CaseSpec]) -> None:
    payload = {
        "synthetic_k_sweep_cases": [c.__dict__ for c in synthetic],
        "netlib_k_sweep_cases": [c.__dict__ for c in netlib],
        "sample_efficiency_cases": [c.__dict__ for c in sample_cases],
    }
    (out_dir / "suite_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_profile_suite(
    profile: str,
    out_dir: Path,
    base_args: argparse.Namespace,
    synth_cases: Sequence[CaseSpec],
    net_cases: Sequence[CaseSpec],
    sample_cases: Sequence[CaseSpec],
    force: bool,
) -> None:
    profile = str(profile).lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_suite_manifest(out_dir, synth_cases, net_cases, sample_cases)

    base_args = copy.deepcopy(base_args)
    base_args.suite_profile = profile

    synth_results = []
    for case in synth_cases:
        print(f"[{profile} | K-sweep synthetic] {case.slug}")
        synth_results.append(run_or_load_k_sweep_case(case, out_dir, base_args, force=bool(force)))
    synth_summary = pd.concat([r["summary"] for r in synth_results if not r["summary"].empty], ignore_index=True)
    synth_quality = pd.concat([r["quality"] for r in synth_results if not r["quality"].empty], ignore_index=True) if any(not r["quality"].empty for r in synth_results) else pd.DataFrame()
    synth_ranks = pd.concat([r["rank"] for r in synth_results if not r["rank"].empty], ignore_index=True) if any(not r["rank"].empty for r in synth_results) else pd.DataFrame()
    if not synth_summary.empty:
        synth_summary.to_csv(out_dir / "k_sweep_synthetic_summary.csv", index=False)
    if not synth_quality.empty:
        synth_quality.to_csv(out_dir / "k_sweep_synthetic_quality.csv", index=False)
    if not synth_ranks.empty:
        synth_ranks.to_csv(out_dir / "ours_rank_growth_synthetic.csv", index=False)

    net_results = []
    for case in net_cases:
        print(f"[{profile} | K-sweep netlib] {case.slug}")
        net_results.append(run_or_load_k_sweep_case(case, out_dir, base_args, force=bool(force)))
    net_summary = pd.concat([r["summary"] for r in net_results if not r["summary"].empty], ignore_index=True) if net_results else pd.DataFrame()
    net_quality = pd.concat([r["quality"] for r in net_results if not r["quality"].empty], ignore_index=True) if net_results and any(not r["quality"].empty for r in net_results) else pd.DataFrame()
    net_ranks = pd.concat([r["rank"] for r in net_results if not r["rank"].empty], ignore_index=True) if net_results and any(not r["rank"].empty for r in net_results) else pd.DataFrame()
    if not net_summary.empty:
        net_summary.to_csv(out_dir / "k_sweep_netlib_summary.csv", index=False)
    if not net_quality.empty:
        net_quality.to_csv(out_dir / "k_sweep_netlib_quality.csv", index=False)
    if not net_ranks.empty:
        net_ranks.to_csv(out_dir / "ours_rank_growth_netlib.csv", index=False)

    sample_summary_frames = []
    for case in sample_cases:
        print(f"[{profile} | Sample-efficiency] {case.slug}")
        sample_summary_frames.append(run_sample_efficiency_case(case, out_dir, base_args, force=bool(force)))
    sample_summary = pd.concat(sample_summary_frames, ignore_index=True) if sample_summary_frames else pd.DataFrame()
    if not sample_summary.empty:
        sample_summary.to_csv(out_dir / "sample_efficiency_summary.csv", index=False)

    synth_order = [c.slug for c in synth_cases]
    net_order = [c.slug for c in net_cases]
    sample_order = [c.slug for c in sample_cases]
    title_map = {c.slug: c.title for c in [*synth_cases, *net_cases, *sample_cases]}
    sample_k_map = {c.slug: int(c.sample_k) for c in sample_cases}

    if getattr(base_args, "make_quality_figures", False) and not synth_quality.empty:
        plot_metric_grid(
            synth_quality,
            synth_order,
            title_map,
            out_dir / "figure_k_sweep_synthetic_quality.png",
            methods=METHOD_ORDER,
            y_col="anchor_quality_mean",
            yerr_col="anchor_quality_se",
            ylabel="Average test quality",
        )
    if not synth_summary.empty:
        plot_metric_grid(
            synth_summary,
            synth_order,
            title_map,
            out_dir / "figure_k_sweep_synthetic.png",
            methods=METHOD_ORDER,
            y_col="objective_ratio_mean",
            yerr_col="objective_ratio_se",
            ylabel="Average test objective ratio",
        )
    if getattr(base_args, "make_quality_figures", False) and not net_quality.empty:
        plot_metric_grid(
            net_quality,
            net_order,
            title_map,
            out_dir / "figure_k_sweep_netlib_quality.png",
            methods=METHOD_ORDER,
            y_col="anchor_quality_mean",
            yerr_col="anchor_quality_se",
            ylabel="Average test quality",
        )
    if not net_summary.empty:
        plot_metric_grid(
            net_summary,
            net_order,
            title_map,
            out_dir / "figure_k_sweep_netlib.png",
            methods=METHOD_ORDER,
            y_col="objective_ratio_mean",
            yerr_col="objective_ratio_se",
            ylabel="Average test objective ratio",
        )
    if not synth_ranks.empty:
        plot_rank_grid(
            synth_ranks,
            synth_order,
            title_map,
            out_dir / "figure_ours_rank_growth_synthetic.png",
            per_case_scale=True,
        )
    if not net_ranks.empty:
        plot_rank_grid(
            net_ranks,
            net_order,
            title_map,
            out_dir / "figure_ours_rank_growth_netlib.png",
            per_case_scale=True,
        )
    if not sample_summary.empty:
        plot_sample_efficiency_grid(
            sample_summary,
            sample_order,
            title_map,
            sample_k_map,
            out_dir / "figure_sample_efficiency_fixedK.png",
        )
    if getattr(base_args, "make_quality_figures", False) and "anchor_quality_mean" in sample_summary.columns:
        plot_sample_efficiency_grid(
            sample_summary,
            sample_order,
            title_map,
            sample_k_map,
                out_dir / "figure_sample_efficiency_fixedK_quality.png",
                y_col="anchor_quality_mean",
                yerr_col="anchor_quality_se",
                ylabel="Average test quality",
            )

    print(f"\nFinished final projection-figure suite for profile={profile}.")
    if not synth_summary.empty:
        print(f"\nSynthetic K-sweep summary ({profile}):")
        with pd.option_context("display.max_rows", 300, "display.max_columns", 30, "display.width", 220):
            print(synth_summary[["case", "method", "K", "objective_ratio_mean", "success_rate"]].sort_values(["case", "K", "method"]))
    if not net_summary.empty:
        print(f"\nNetlib K-sweep summary ({profile}):")
        with pd.option_context("display.max_rows", 300, "display.max_columns", 30, "display.width", 220):
            print(net_summary[["case", "method", "K", "objective_ratio_mean", "success_rate"]].sort_values(["case", "K", "method"]))
    if not sample_summary.empty:
        print(f"\nSample-efficiency summary ({profile}):")
        with pd.option_context("display.max_rows", 300, "display.max_columns", 30, "display.width", 220):
            print(sample_summary[["case", "method", "n_train", "K", "anchor_quality_mean"]].sort_values(["case", "n_train", "method"]))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out_dir", type=str, default="results_projection_suite_natural_refined")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--force", action="store_true", help="rerun even when cached case outputs already exist")
    p.add_argument("--skip_netlib", action="store_true")
    p.add_argument("--skip_sample_eff", action="store_true")
    p.add_argument("--suite_profile", type=str, default="natural", choices=["natural"])
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    root = repo_root()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_args = default_namespace()
    base_args.device = args.device
    base_args.seed = int(args.seed)
    base_args.verbose = bool(args.verbose)

    synth_cases = synthetic_cases(root)
    net_cases = [] if args.skip_netlib else netlib_cases(root)
    sample_cases = [] if args.skip_sample_eff else sample_efficiency_cases(root)
    profiles = [str(args.suite_profile)]
    for profile in profiles:
        profile_out = out_dir / profile if len(profiles) > 1 else out_dir
        run_profile_suite(profile, profile_out, base_args, synth_cases, net_cases, sample_cases, force=bool(args.force))


if __name__ == "__main__":
    main()
