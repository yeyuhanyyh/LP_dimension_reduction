#!/usr/bin/env python3
"""Run the broader harder-prior and beyond-prior parameter sweeps."""
from __future__ import annotations

from pathlib import Path

import final_projection_figure_suite as fps
import prior_stress_and_beyond_suite as ps


def beyond_base_args() -> object:
    ns = fps.default_namespace()
    ns.n_train = 8
    ns.n_val = 2
    ns.n_test = 8
    ns.k_list = "20"
    ns.costonly_epochs = 3
    ns.costonly_patience = 2
    ns.costonly_hidden_dim = 10
    ns.batch_size = 2
    return ns


def main() -> None:
    root = Path(".").resolve()
    out_root = root / "results_complete_prior_suite_v5_parameters"
    case_slugs = [case.slug for case in ps.all_cases(root)]

    # 1. Larger-C / harder-distribution sweep in the original known-prior setting.
    ps.run_harder_prior_probe(
        root=root,
        case_slugs=case_slugs,
        scales=[1.0, 2.0, 3.0, 4.0],
        out_dir=out_root / "harder_prior_broad",
    )

    # 2. Sample-efficiency on the larger-C setting, using the most expanded prior.
    ps.run_harder_prior_sample_efficiency(
        root=root,
        case_slugs=case_slugs,
        scales=[4.0],
        out_dir=out_root / "harder_prior_sample_efficiency",
    )

    # 3. Beyond-a-given-prior-set: primary sweep over target outside mass rho.
    bargs = beyond_base_args()
    ps.run_beyond_prior_probe(
        root=root,
        case_slugs=case_slugs,
        rho_list=[0.40, 0.20, 0.10, 0.03, 0.01],
        radius_scale_list=[1.0],
        out_dir=out_root / "beyond_prior_rho",
        pilot_n0=80,
        ambient_radius_scale=1.5,
        test_n=int(bargs.n_test),
        base_args=bargs,
    )

    # 4. Beyond: secondary sweep over pilot sample size n0 at a fixed rho.
    ps.run_beyond_prior_n0_probe(
        root=root,
        case_slugs=case_slugs,
        pilot_n0_list=[20, 40, 80, 160],
        rho_target=0.10,
        out_dir=out_root / "beyond_prior_n0",
        ambient_radius_scale=1.5,
        test_n=int(bargs.n_test),
        radius_scale=1.0,
        base_args=bargs,
    )

    # 5. Beyond: implementation calibration sweep over ellipsoid inflation.
    ps.run_beyond_prior_probe(
        root=root,
        case_slugs=case_slugs,
        rho_list=[0.10],
        radius_scale_list=[0.85, 1.00, 1.15, 1.30],
        out_dir=out_root / "beyond_prior_radius_scale",
        pilot_n0=80,
        ambient_radius_scale=1.5,
        test_n=int(bargs.n_test),
        base_args=bargs,
    )


if __name__ == "__main__":
    main()
