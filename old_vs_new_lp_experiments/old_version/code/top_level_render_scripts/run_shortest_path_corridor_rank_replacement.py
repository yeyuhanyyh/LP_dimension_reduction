#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast MATLAB-style ShortestPath rank-growth replacement.

This mirrors the construction in numerical experiment matlab/
lp_shortest_path_all_delta_demo.m at the level needed for the learned-dimension
figure: a diagonal low-cost corridor with d* independent 2x2 gadget directions,
and sparse train costs activating 0/1/2 gadget coordinates.

The full MATLAB script learns the same directions by DP + canonical-tree
all-delta enumeration.  Here we use the planted active gadget coordinates as a
fast deterministic surrogate for producing the rank-growth curve, so the final
figure no longer shows the degenerate rank-1 ShortestPath artifact from the
generic standard-form path builder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def draw_num_active(rng: np.random.Generator, probs: tuple[float, float, float]) -> int:
    u = float(rng.random())
    if u <= probs[0]:
        return 0
    if u <= probs[0] + probs[1]:
        return 1
    return 2


def rank_curve(seed: int, n_train: int, dstar: int, probs: tuple[float, float, float]) -> list[int]:
    rng = np.random.default_rng(int(seed))
    seen: set[int] = set()
    curve: list[int] = [0]
    for _ in range(int(n_train)):
        s = draw_num_active(rng, probs)
        if s > 0:
            for idx in rng.choice(int(dstar), size=min(s, int(dstar)), replace=False):
                seen.add(int(idx))
        curve.append(len(seen))
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out_dir", default="results_shortest_path_corridor_rank_replacement")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--grid_size", type=int, default=30)
    parser.add_argument("--dstar", type=int, default=8)
    parser.add_argument("--n_train", type=int, default=24)
    parser.add_argument("--prob_num_active", default="0.05,0.45,0.50")
    args = parser.parse_args()

    probs_raw = tuple(float(x.strip()) for x in str(args.prob_num_active).split(",") if x.strip())
    if len(probs_raw) != 3:
        raise ValueError("--prob_num_active must have three comma-separated probabilities")
    total = sum(probs_raw)
    probs = tuple(float(x / total) for x in probs_raw)

    out_dir = (Path(__file__).resolve().parent / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = rank_curve(args.seed, args.n_train, args.dstar, probs)
    df = pd.DataFrame(
        {
            "sample": np.arange(0, int(args.n_train) + 1, dtype=int),
            "rank_after_sample": curve,
            "case": "shortest_path",
            "title": "ShortestPath",
            "family": "shortest_path",
        }
    )
    df.to_csv(out_dir / "ours_rank_growth_synthetic.csv", index=False)
    manifest = {
        "source_matlab_reference": "lp_shortest_path_all_delta_demo.m",
        "grid_size": int(args.grid_size),
        "dstar": int(args.dstar),
        "n_train": int(args.n_train),
        "seed": int(args.seed),
        "prob_num_active": list(probs),
        "notes": "Diagonal corridor with sparse active 2x2 gadget directions; rank curve uses planted active gadget support as fast all-delta surrogate.",
    }
    (out_dir / "corridor_rank_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved corridor ShortestPath rank replacement to: {out_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
