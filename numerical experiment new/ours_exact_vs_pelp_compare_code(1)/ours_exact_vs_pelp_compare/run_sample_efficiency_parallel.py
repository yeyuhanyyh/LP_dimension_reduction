#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel runner for sample-efficiency cases.

The main figure suite remains the canonical entry point.  This helper only
fills the independent ``sample_efficiency/<case>`` caches faster; afterwards
rerunning ``final_projection_figure_suite.py`` without ``--force`` aggregates
the cached CSVs and redraws the figure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from final_projection_figure_suite import (
    default_namespace,
    repo_root,
    run_sample_efficiency_case,
    sample_efficiency_cases,
)


def _run_one(seed: int, out_dir_str: str, slug: str, force: bool) -> str:
    root = repo_root()
    out_dir = (root / out_dir_str).resolve()
    base_args = default_namespace()
    base_args.seed = int(seed)
    base_args.device = "cpu"
    base_args.verbose = False
    cases = {case.slug: case for case in sample_efficiency_cases(root)}
    result = run_sample_efficiency_case(cases[slug], Path(out_dir), base_args, force=bool(force))
    return f"finished {slug} seed={seed} rows={len(result)}"


def parse_cases(value: str, known: Sequence[str]) -> list[str]:
    if not value or value.lower() == "all":
        return list(known)
    wanted = [part.strip() for part in value.split(",") if part.strip()]
    missing = sorted(set(wanted).difference(known))
    if missing:
        raise SystemExit(f"Unknown case slug(s): {', '.join(missing)}. Known: {', '.join(known)}")
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    known_cases = [case.slug for case in sample_efficiency_cases(repo_root())]
    cases = parse_cases(args.cases, known_cases)
    workers = max(1, min(int(args.workers), len(cases)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, int(args.seed), str(args.out_dir), slug, bool(args.force)) for slug in cases]
        for fut in as_completed(futures):
            print(fut.result(), flush=True)


if __name__ == "__main__":
    main()
