# Our Exact Decision-Sufficient Reduction vs Iwata--Sakaue / Sakaue--Oki Baselines

## Current final entry point

- `final_projection_figure_suite.py` is the current final experiment driver.
- It produces the requested paper-style fixed-`n_train`, sweep-`K` figures for
  `Full`, `OursExact`, `Rand`, `PCA`, `SGA`, `FCNN`, and `PELP`.
- It covers
  `packing`, `maxflow`, `mincostflow`, `shortest_path`,
  multiple `random_stdform` cases, and selected Netlib LPs.
- It also produces:
  - learned-dimension growth curves for `OursExact`
  - fixed-`K`, sweep-`n_train` sample-efficiency comparisons for `OursExact / PCA / SGA`
- The current outputs are written under `results_projection_suite/`.

## What Each Python File Does

- `final_projection_figure_suite.py`
  Top-level paper-figure driver. It builds the synthetic and Netlib case lists,
  runs or reuses per-case experiments, aggregates averages / standard errors,
  and draws the final dashed-line + shaded-band figures:
  - fixed `n_train`, sweep `K`
  - `OursExact` learned-dimension growth
  - fixed `K`, sweep `n_train`
- `compare_fixedX_family_suite.py`
  Core experiment engine for fixed-`X`, varying-`c` LP families. It:
  - constructs the LP families (`packing`, `maxflow`, `mincostflow`,
    `shortest_path`, `random_lp`, `random_stdform`)
  - builds the standard-form lift for our exact method
  - implements the signed `PCA` and `SGA + final projection` baselines
  - computes auxiliary metrics such as `improvement_capture`
- `compare_ours_exact_vs_pelp_fixedX.py`
  Core implementation of our Algorithm 1 / 2 pipeline on standard-form LPs:
  prior construction, edge-direction enumeration, cumulative learning
  (`alg2_cumulative`), and the exact affine reduced solve
  `x = x_anchor + U z`.
- `iwata_sakaue_pelp_projection_compare.py`
  2025-style projection baselines and LP preprocessing utilities. It contains:
  - `FCNNProjectionNet`
  - `PELPProjectionNet`
  - random / PCA / neural projection evaluation helpers
  - fixed-feasible to zero-feasible bridging
  - Netlib MPS loading and preprocessing

## Reproduce The Current Figures And Tables

Run the final suite from the repository root:

```bash
python final_projection_figure_suite.py --out_dir results_projection_suite
```

If you want to recompute everything from scratch instead of reusing cached case
results, add `--force`:

```bash
python final_projection_figure_suite.py --out_dir results_projection_suite --force
```

Main output files:

- `results_projection_suite/figure_k_sweep_synthetic.png`
- `results_projection_suite/figure_k_sweep_netlib.png`
- `results_projection_suite/figure_ours_rank_growth_synthetic.png`
- `results_projection_suite/figure_ours_rank_growth_netlib.png`
- `results_projection_suite/figure_sample_efficiency_fixedK.png`
- `results_projection_suite/k_sweep_synthetic_summary.csv`
- `results_projection_suite/k_sweep_synthetic_quality.csv`
- `results_projection_suite/k_sweep_netlib_summary.csv`
- `results_projection_suite/k_sweep_netlib_quality.csv`
- `results_projection_suite/ours_rank_growth_synthetic.csv`
- `results_projection_suite/ours_rank_growth_netlib.csv`
- `results_projection_suite/sample_efficiency_summary.csv`
- `results_projection_suite/suite_manifest.json`

## Which Metric The Main Figures Use

- The raw summary tables still keep `objective_ratio_mean`.
- The main paper-style figures currently plot `anchor_quality_mean`, not the raw
  objective ratio.

Reason:

- for fixed-`X`, varying-`c` families, especially `random_stdform` and several
  real LPs, raw objective ratios can look artificially close to `1` because all
  methods inherit a large common objective offset from the anchor solution;
- `anchor_quality` normalizes only the improvement above the common anchor:

  `anchor_quality = (obj - obj_anchor) / (obj_full - obj_anchor)`

  clipped to `[0, 1.05]` for plotting robustness.

So in the current figures:

- `1.0` means the method captures essentially all improvement over the anchor;
- `0.0` means it stays at the anchor level;
- values in between show how much of the full solver's improvement is retained.

## Current Default Figure Sizes And K Range

The current final-suite defaults in `final_projection_figure_suite.py` are:

- fixed-`n_train`, sweep-`K` now uses `K in {10, 20, 30, 40, 50}`
- packing: `n_vars = 240`, `n_cons = 60`, with `block_gadget` structure
- maxflow / mincostflow: `n_nodes = 50`, `n_edges = 220`, with `path_gadget` structure
- shortest path: `grid_size = 14`, with many planted square gadgets
- random standard form A: `std_m = 60`, `std_d = 480`
- random standard form B: `std_m = 72`, `std_d = 560`
- random standard form C: `std_m = 84`, `std_d = 640`

For Netlib, the suite currently uses:

- `GROW7`
- `ISRAEL`
- `SC205`
- `SCAGR25`
- `STAIR`

The fixed-`K` sample-efficiency figures currently use `sample_k = 20`.

## Keep These Python Files

- `final_projection_figure_suite.py`
  Final top-level suite and figure generator.
- `compare_fixedX_family_suite.py`
  Core fixed-`X` synthetic-family builder plus signed `PCA/SGA` helpers.
- `compare_ours_exact_vs_pelp_fixedX.py`
  Core implementation of our Algorithm 1/2 learner, prior construction, and exact affine reduction solver.
- `iwata_sakaue_pelp_projection_compare.py`
  2025 neural baselines, LP/MPS preprocessing, and Netlib loader.

This folder contains a complete Python comparison script:

- `compare_ours_exact_vs_pelp_fixedX.py`: driver that integrates our Algorithm 1/2 stage-I learner and exact affine reduced LP on a fixed-feasible packing family with a planted low-dimensional profit prior.
- `iwata_sakaue_pelp_projection_compare.py`: companion implementation of PELP_NN / SharedP / FCNN / PCA / Rand baselines, now with support for `packing`, `maxflow`, `mincostflow`, and `shortest_path` synthetic families.
- `requirements.txt`: Python packages.

Important reproduction note:

- `PELP_NN`, `SharedP`, and `FCNN` in `iwata_sakaue_pelp_projection_compare.py` follow the ICML 2025 softmax-nonnegative projection setup of Iwata--Sakaue.
- They are not the same as the NeurIPS 2024 Sakaue--Oki `SGA + final projection` procedure; that older paper's shared-projection method uses a different training/feasibility heuristic.
- When these 2025-style softmax baselines are evaluated on the nullspace-bridge fixed-feasible comparisons (`shortest_path`, `netlib`, or other bridge settings), the script now adds an `Anchor` baseline and prints a bridge diagnostic if the learned models collapse back to the anchor solution.

## Unified fixed-X suite

- `compare_fixedX_family_suite.py` builds one fixed feasible region `X` per dataset family and varies only the objective vector `c` inside an explicit Euclidean uncertainty ball `C`, with `0 notin C`.
- Supported families are `packing`, `maxflow`, `mincostflow`, `shortest_path`, and `random_lp`.
- The same feasible set is compared in two equivalent views:
  - our method uses a standard-form lift with slack variables;
  - the Japanese baselines are evaluated on the original LP family in general form.
- The script also adds the NeurIPS 2024-style `SGA_FinalP` baseline, implemented as shared-gradient learning plus final column projection on an inequality reformulation with free reduced variables.
- Network-flow equality systems are row-reduced to full row rank before running our Algorithm 1/2 stage-I learner. This does not change `X`; it only removes redundant conservation equations that otherwise break basis recovery.

Quick fixed-X benchmark over all five families:

```bash
python compare_fixedX_family_suite.py --dataset packing --quick --run_sharedp --run_fcnn --out_dir tmp_fixedX_packing_quick
python compare_fixedX_family_suite.py --dataset maxflow --quick --run_sharedp --run_fcnn --out_dir tmp_fixedX_maxflow_quick
python compare_fixedX_family_suite.py --dataset mincostflow --quick --run_sharedp --run_fcnn --out_dir tmp_fixedX_mincostflow_quick
python compare_fixedX_family_suite.py --dataset shortest_path --quick --run_sharedp --run_fcnn --out_dir tmp_fixedX_shortest_path_quick
python compare_fixedX_family_suite.py --dataset random_lp --quick --run_sharedp --run_fcnn --out_dir tmp_fixedX_randomlp_quick
```

Helper runner:

```bash
powershell -ExecutionPolicy Bypass -File .\run_fixedX_family_benchmark.ps1 -Quick -BaseOutDir results_fixedX_family_quick
```

The combined quick summary can then be collected at `results_fixedX_family_quick/all_summary_results.csv`.

## Quick command

```bash
python compare_ours_exact_vs_pelp_fixedX.py --quick --run_sharedp --run_fcnn --make_plots --out_dir results_quick
```

The quick mode is a smoke test. It is deliberately tiny so it finishes quickly on CPU.

## More meaningful CPU command

```bash
python compare_ours_exact_vs_pelp_fixedX.py ^
  --n_vars 60 --n_cons 16 --dstar 8 ^
  --n_train 50 --n_val 8 --n_test 15 ^
  --k_list 4,8,12 ^
  --epochs 12 --batch_size 4 ^
  --run_sharedp --run_fcnn ^
  --make_plots --out_dir results_medium
```

On Windows PowerShell, use backtick ` instead of `^` for line continuation, or put the whole command on one line.

## Iwata/Sakaue-style baseline runs

Packing paper-style setting:

```bash
python iwata_sakaue_pelp_projection_compare.py --dataset packing --paper_settings --run_sharedp --run_fcnn --out_dir pelp_packing
```

MaxFlow synthetic family:

```bash
python iwata_sakaue_pelp_projection_compare.py --dataset maxflow --quick --out_dir pelp_maxflow
```

MinCostFlow synthetic family:

```bash
python iwata_sakaue_pelp_projection_compare.py --dataset mincostflow --quick --out_dir pelp_mincost
```

Structured shortest-path family:

```bash
python iwata_sakaue_pelp_projection_compare.py --dataset shortest_path --packing_mode fixed_feasible --quick --out_dir pelp_spath
```

## Notes on LP forms

- Our exact method is implemented on standard-form LPs `min c^T x s.t. Aeq x = b, x >= 0`.
- The 2025 projection baselines are evaluated on `x = P y, y >= 0` in the original variables.
- The 2024 `SGA_FinalP` baseline is evaluated on an inequality-only reformulation with free reduced variables `y`.
- Equality-heavy families can therefore expose a real modeling conflict for the 2025 nonnegative projector family: if a shared nonnegative basis cannot represent the common equality structure, projected LPs may become infeasible or collapse to trivial feasible solutions such as zero flow. That behavior is informative and should not be confused with a bug in our standard-form lift.

## Real LP data

- The workspace now includes decompressed Netlib-style MPS files for `GROW7`, `ISRAEL`, `SC205`, `SCAGR25`, and `STAIR` under `data/netlib/selected/`.
- See `EXPERIMENT_CASES.md` for paper-like commands covering `packing`, `maxflow`, `mincostflow`, `shortest_path`, and `netlib`.

## Output files

The output folder contains:

- `summary_results.csv`: aggregate objective ratios and solve times.
- `raw_results.csv`: per-test-instance data.
- `objective_ratio_vs_K.png`: objective-ratio comparison.
- `runtime_vs_K.png`: runtime comparison.
- `ours_learned_dimension_vs_samples.png`: our learned dimension as sample count grows.
- `ours_stageI_learned_basis.npz`: learned basis U and anchor x.
- `history_*.csv`: neural training logs.
