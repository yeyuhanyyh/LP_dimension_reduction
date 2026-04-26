#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a lighter all-case Beyond-prior suite with cache-resume enabled.

This script is intentionally separate from the main figure suite so we can:
1. keep the heavier final figures untouched,
2. run the manuscript's Beyond-a-given-prior-set experiments on every case,
3. use lighter train/validation/test and neural budgets for tractable all-case sweeps.
"""
from __future__ import annotations

from pathlib import Path

import final_projection_figure_suite as fps
import prior_stress_and_beyond_suite as ps


ALL_CASES = [
    "packing",
    "maxflow",
    "mincostflow",
    "shortest_path",
    "random_lp_A",
    "random_lp_B",
    "random_lp_C",
    "random_lp_D",
    "grow7",
    "sc205",
    "scagr25",
    "stair",
]


def install_fast_defaults() -> None:
    orig = fps.default_namespace

    def fast_default():
        ns = orig()
        ns.n_train = 8
        ns.n_val = 2
        ns.n_test = 8
        ns.costonly_epochs = 2
        ns.costonly_patience = 1
        ns.costonly_hidden_dim = 8
        ns.batch_size = 2
        return ns

    fps.default_namespace = fast_default


def main() -> None:
    install_fast_defaults()
    root = Path(__file__).resolve().parent
    out_root = root / "results_complete_prior_suite_v3_fast"
    out_root.mkdir(parents=True, exist_ok=True)

    ps.run_beyond_prior_probe(
        root=root,
        case_slugs=ALL_CASES,
        rho_list=[0.30, 0.10, 0.01],
        radius_scale_list=[1.0],
        out_dir=out_root / "beyond_prior",
        pilot_n0=80,
        ambient_radius_scale=1.5,
        test_n=8,
    )

    ps.run_beyond_prior_n0_probe(
        root=root,
        case_slugs=ALL_CASES,
        pilot_n0_list=[40, 80, 160],
        rho_target=0.10,
        out_dir=out_root / "beyond_prior_n0",
        ambient_radius_scale=1.5,
        test_n=8,
        radius_scale=1.0,
    )

    print(out_root.resolve())


if __name__ == "__main__":
    main()
