#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one cached sample-efficiency case from final_projection_figure_suite.

This tiny helper lets the paper-figure suite fill independent sample-size
curves in parallel.  It intentionally delegates all experiment definitions to
``final_projection_figure_suite.py`` so the generated CSVs remain identical to
the main suite.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from final_projection_figure_suite import (
    default_namespace,
    repo_root,
    run_sample_efficiency_case,
    sample_efficiency_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case", required=True, help="Case slug, e.g. packing or random_lp_A")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    out_dir = (root / args.out_dir).resolve()
    base_args = default_namespace()
    base_args.seed = int(args.seed)
    base_args.device = "cpu"
    base_args.verbose = False

    cases = {case.slug: case for case in sample_efficiency_cases(root)}
    if args.case not in cases:
        known = ", ".join(sorted(cases))
        raise SystemExit(f"Unknown case slug {args.case!r}. Known cases: {known}")

    result = run_sample_efficiency_case(cases[args.case], Path(out_dir), base_args, force=bool(args.force))
    print(f"finished {args.case} seed={args.seed} rows={len(result)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
