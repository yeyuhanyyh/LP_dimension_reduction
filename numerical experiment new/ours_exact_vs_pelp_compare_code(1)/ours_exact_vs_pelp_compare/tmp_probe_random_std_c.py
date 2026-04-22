from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np

import final_projection_figure_suite as fps


LOG = Path("tmp_probe_random_std_c.log")


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {msg}\n")


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    base_root = Path.cwd()
    base_args = fps.default_namespace()
    case = next(c for c in fps.synthetic_cases(base_root) if c.slug == "random_std_C")
    args = fps.case_namespace(base_args, case)
    rng = np.random.default_rng(int(args.seed) + int(case.seed_offset))

    log("start")
    t0 = time.time()
    problem = fps.build_problem_for_case(case, args, rng)
    log(f"build_problem dt={time.time() - t0:.3f}")

    t0 = time.time()
    bundle = fps.make_fixed_x_bundle(problem, args, rng)
    log(f"make_bundle dt={time.time() - t0:.3f} train={len(bundle.train)} val={len(bundle.val)} test={len(bundle.test)}")

    indices = [0, 1, 5, 10, 20, 35, 39]
    for idx in indices:
        inst = bundle.train[idx]
        log(f"train[{idx}] name={inst.name} full_solve:start")
        t1 = time.time()
        fps.ensure_full_solutions([inst])
        log(f"train[{idx}] full_solve:ok dt={time.time() - t1:.3f} full_value={getattr(inst, 'full_value', None)}")

        log(f"train[{idx}] bridge:start")
        proj = fps.bridge_instances_for_projection(problem, [inst])[0]
        log(f"train[{idx}] bridge:ok n_vars={proj.n_vars} n_cons={proj.A.shape[0]}")

        log(f"train[{idx}] proj_full:start")
        t2 = time.time()
        fps.ensure_full_solutions([proj])
        log(f"train[{idx}] proj_full:ok dt={time.time() - t2:.3f} full_value={getattr(proj, 'full_value', None)}")

    log("all_ok")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # pragma: no cover - debugging helper
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        raise
