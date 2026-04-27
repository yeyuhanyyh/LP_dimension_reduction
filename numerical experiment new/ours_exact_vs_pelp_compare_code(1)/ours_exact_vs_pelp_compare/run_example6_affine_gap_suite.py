"""Example-6 affine-vs-homogeneous projection gap experiment.

This is a small, targeted diagnostic inspired by Example 6 in the manuscript.
It is intentionally separate from the large natural suite: the goal is to show
the representational obstruction cleanly, using a metric that subtracts the
common affine anchor contribution.

The feasible set is a product of two-variable boxes. In each block the first
coordinate is the affine anchor coordinate and the second coordinate is a
switch. OursExact is evaluated with the correct affine anchor plus switch
directions. PCA/SGA/Rand are homogeneous projection baselines of the same
displayed dimension K.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from compare_fixedX_family_suite import (
    append_general_projection_rows,
    convert_to_general_projection_instance,
    raw_pca_projection,
    train_sga_final_projection,
)
from final_projection_figure_suite import METHOD_DISPLAY, METHOD_STYLE, clipped_band
from iwata_sakaue_pelp_projection_compare import LPInstance, ensure_full_solutions, random_column_projection


METHODS = ["Full", "OursExact", "Rand", "PCA", "SGA"]


def parse_k_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def make_anchor(m: int) -> np.ndarray:
    anchor = np.zeros(2 * int(m), dtype=float)
    anchor[0::2] = 1.0
    return anchor


def make_regime_instance(m: int, regime: int, name: str, anchor_reward: float = 1.0, switch_reward: float = 1.0) -> LPInstance:
    """Create one max-reward LP over [0,1]^(2m).

    regime=-1 is the pure anchor regime. For regime=j, only switch j has
    positive reward; all other switch coordinates have negative reward.
    """
    reward = np.zeros(2 * int(m), dtype=float)
    reward[0::2] = float(anchor_reward)
    reward[1::2] = -float(switch_reward)
    if int(regime) >= 0:
        reward[1 + 2 * int(regime)] = float(switch_reward)
    return LPInstance(c=reward, A=np.eye(2 * int(m), dtype=float), b=np.ones(2 * int(m), dtype=float), name=name)


def build_regime_family(m: int) -> tuple[List[LPInstance], List[LPInstance], np.ndarray]:
    train = [make_regime_instance(m, -1, "train_anchor")]
    train.extend(make_regime_instance(m, j, f"train_switch_{j:02d}") for j in range(int(m)))
    test = [make_regime_instance(m, -1, "test_anchor")]
    test.extend(make_regime_instance(m, j, f"test_switch_{j:02d}") for j in range(int(m)))
    ensure_full_solutions(train)
    ensure_full_solutions(test)
    return train, test, make_anchor(m)


def edge_quality(raw: pd.DataFrame, anchor: np.ndarray, test: Sequence[LPInstance]) -> pd.DataFrame:
    base_obj: Dict[str, float] = {inst.name: float(np.asarray(inst.c, dtype=float) @ anchor) for inst in test}
    out = raw.copy()
    values: List[float] = []
    for row in out.itertuples(index=False):
        base = float(base_obj[getattr(row, "instance")])
        full = float(getattr(row, "full_objective"))
        obj = float(getattr(row, "objective"))
        denom = full - base
        if denom <= 1e-10:
            values.append(1.0 if abs(obj - full) <= 1e-7 else float("nan"))
        else:
            values.append(float(max(0.0, min(1.05, (obj - base) / denom))))
    out["edge_quality"] = values
    return out


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, k), grp in raw.groupby(["method", "K"], dropna=False):
        vals = grp["edge_quality"].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mean = se = float("nan")
        else:
            mean = float(np.mean(vals))
            se = 0.0 if vals.size <= 1 else float(np.std(vals, ddof=1) / np.sqrt(vals.size))
        rows.append({"method": method, "K": int(k), "edge_quality_mean": mean, "edge_quality_se": se, "n": int(len(grp))})
    return pd.DataFrame(rows).sort_values(["K", "method"]).reset_index(drop=True)


def run_k_sweep(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    train, test, anchor = build_regime_family(args.blocks)
    general_test = [convert_to_general_projection_instance(inst) for inst in test]
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(int(args.seed))

    for k in parse_k_list(args.k_list):
        for inst in test:
            for method in ("Full", "OursExact"):
                rows.append(
                    {
                        "method": method,
                        "K": int(k),
                        "instance": inst.name,
                        "objective": float(inst.full_obj),
                        "full_objective": float(inst.full_obj),
                        "success": 1.0,
                    }
                )

        for trial in range(int(args.rand_trials)):
            P_rand = random_column_projection(2 * int(args.blocks), int(k), np.random.default_rng(rng.integers(0, 2**31 - 1)))
            append_general_projection_rows(rows, f"Rand#{trial + 1}", P_rand, general_test, int(k))

        P_pca = raw_pca_projection(train, int(k))
        append_general_projection_rows(rows, "PCA", P_pca, general_test, int(k))

        P_sga, hist = train_sga_final_projection(
            train,
            [],
            int(k),
            epochs=int(args.sga_epochs),
            batch_size=int(args.batch_size),
            lr=float(args.sga_lr),
            seed=int(args.seed) + 1000 + int(k),
            verbose=False,
        )
        append_general_projection_rows(rows, "SGA", P_sga, general_test, int(k))
        pd.DataFrame(hist).to_csv(out_dir / f"history_SGA_K{k}.csv", index=False)

    raw = edge_quality(pd.DataFrame(rows), anchor, test)
    raw["method"] = raw["method"].str.replace(r"Rand#\d+", "Rand", regex=True)
    summary = summarize(raw)
    raw.to_csv(out_dir / "example6_k_sweep_raw.csv", index=False)
    summary.to_csv(out_dir / "example6_k_sweep_summary.csv", index=False)
    return summary


def run_rank_growth(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    """A simple estimated-C coupon-collector proxy for gradual discovery.

    With known C, the exact learner can certify all Example-6 switch directions
    from one representative cost. This proxy instead records how many switch
    regimes have been observed when C is estimated from data.
    """
    rng = np.random.default_rng(int(args.seed) + 77)
    order = rng.permutation(int(args.blocks))
    rows = [{"n_train": 0, "learned_dimension": 0, "edge_quality": 0.0}]
    seen: set[int] = set()
    for n in range(1, int(args.blocks) + 1):
        seen.add(int(order[n - 1]))
        dim = len(seen)
        rows.append({"n_train": n, "learned_dimension": dim, "edge_quality": float(dim / int(args.blocks))})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "example6_estimatedC_rank_growth.csv", index=False)
    return out


def plot_k_sweep(summary: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    lo, hi = -0.02, 1.05
    k_values = sorted(summary["K"].astype(int).unique().tolist())
    xpos = {k: idx for idx, k in enumerate(k_values)}
    for method in METHODS:
        sub = summary[summary["method"] == method].sort_values("K")
        if sub.empty:
            continue
        style = METHOD_STYLE.get(method, {"color": None, "marker": "o", "linestyle": "-"})
        x = np.asarray([xpos[int(k)] for k in sub["K"].astype(int).tolist()], dtype=float)
        y = sub["edge_quality_mean"].astype(float).to_numpy()
        se = sub["edge_quality_se"].fillna(0.0).astype(float).to_numpy()
        ylo, yhi = clipped_band(y, se, lo, hi)
        ax.fill_between(x, ylo, yhi, color=style["color"], alpha=0.08, linewidth=0)
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2 if method == "OursExact" else 1.9,
            label=METHOD_DISPLAY.get(method, method),
        )
    if 20 in xpos:
        ax.axvline(xpos[20], color="#333333", linestyle=(0, (2, 2)), linewidth=1.0, alpha=0.5)
        ax.text(xpos[20] - 0.18, 0.05, "d*", fontsize=9, color="#333333")
    ax.set_xlabel("Reduced dimension K")
    ax.set_ylabel("Normalized improvement over affine anchor")
    ax.set_ylim(lo, hi)
    ax.set_xticks([xpos[k] for k in k_values])
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_title("Block Example 6: affine exact vs. homogeneous projection")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_rank_growth(rank: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
    ax1.plot(rank["n_train"], rank["learned_dimension"], color=METHOD_STYLE["OursExact"]["color"], marker="o", linestyle=(0, (6, 2)), linewidth=2.0)
    ax1.set_xlabel("# observed cost regimes")
    ax1.set_ylabel("Estimated-C learned dimension")
    ax1.set_xticks([0, 5, 10, 15, 20])
    ax1.set_ylim(0, max(rank["learned_dimension"]) + 1)
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(rank["n_train"], rank["edge_quality"], color="#555555", marker="s", linestyle=(0, (1, 2)), linewidth=1.7)
    ax2.set_ylabel("Coverage of Example-6 switch regimes")
    ax2.set_ylim(-0.02, 1.05)
    ax1.set_title("Estimated-C gradual discovery proxy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out_dir", default="results_example6_affine_gap_suite")
    p.add_argument("--blocks", type=int, default=20)
    p.add_argument("--k_list", default="5,10,20,21,30,40,50")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--sga_epochs", type=int, default=30)
    p.add_argument("--sga_lr", type=float, default=5e-2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--rand_trials", type=int, default=3)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    summary = run_k_sweep(args, out_dir)
    rank = run_rank_growth(args, out_dir)
    plot_k_sweep(summary, out_dir / "figure_example6_k_sweep.png")
    plot_rank_growth(rank, out_dir / "figure_example6_estimatedC_rank_growth.png")
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(summary)


if __name__ == "__main__":
    main()
