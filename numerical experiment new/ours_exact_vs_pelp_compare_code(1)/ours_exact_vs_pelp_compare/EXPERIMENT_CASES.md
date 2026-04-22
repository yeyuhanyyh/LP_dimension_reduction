# Experiment Cases

This workspace now includes the main experiment entry points needed to compare
our exact reformulation method with the Iwata/Sakaue projection baselines.

## Available entry points

- `compare_ours_exact_vs_pelp_fixedX.py`
  - Fixed-feasible planted packing comparison.
  - Best for checking why the learned projection baselines can or cannot train.
- `iwata_sakaue_pelp_projection_compare.py`
  - Unified runner for `packing`, `maxflow`, `mincostflow`, `shortest_path`,
    and `netlib`.
  - Includes `PELP_NN`, `SharedP`, `FCNN`, `PCA`, `Rand`, and `OursExact`.

## Prepared datasets

- Synthetic families matching the Japanese papers:
  - `packing`
  - `maxflow`
  - `mincostflow`
- Structured repeated-LP family matching our exact reformulation note:
  - `shortest_path`
- Real LP instances from the Sakaue/Oki paper:
  - `data/netlib/selected/grow7.mps`
  - `data/netlib/selected/israel.mps`
  - `data/netlib/selected/sc205.mps`
  - `data/netlib/selected/scagr25.mps`
  - `data/netlib/selected/stair.mps`

## Paper-like commands

Synthetic packing:

```powershell
python iwata_sakaue_pelp_projection_compare.py `
  --dataset packing `
  --paper_settings `
  --run_sharedp `
  --run_fcnn `
  --out_dir results_medium/packing_paper_like
```

Synthetic maxflow:

```powershell
python iwata_sakaue_pelp_projection_compare.py `
  --dataset maxflow `
  --n_nodes 20 `
  --n_edges 80 `
  --n_train 60 `
  --n_val 12 `
  --n_test 18 `
  --k_list 6,12 `
  --epochs 8 `
  --batch_size 4 `
  --run_sharedp `
  --run_fcnn `
  --out_dir results_medium/maxflow_medium
```

Synthetic min-cost flow:

```powershell
python iwata_sakaue_pelp_projection_compare.py `
  --dataset mincostflow `
  --n_nodes 20 `
  --n_edges 80 `
  --n_train 60 `
  --n_val 12 `
  --n_test 18 `
  --k_list 6,12 `
  --epochs 8 `
  --batch_size 4 `
  --run_sharedp `
  --run_fcnn `
  --out_dir results_medium/mincost_medium
```

Our repeated-shortest-path setting:

```powershell
python iwata_sakaue_pelp_projection_compare.py `
  --dataset shortest_path `
  --packing_mode fixed_feasible `
  --grid_size 12 `
  --dstar 4 `
  --n_train 60 `
  --n_val 12 `
  --n_test 18 `
  --k_list 4,8 `
  --epochs 8 `
  --batch_size 4 `
  --run_sharedp `
  --run_fcnn `
  --out_dir results_medium/shortest_path_medium
```

Real Netlib LPs:

```powershell
python iwata_sakaue_pelp_projection_compare.py `
  --dataset netlib `
  --mps_path data/netlib/selected/grow7.mps `
  --n_train 60 `
  --n_val 12 `
  --n_test 18 `
  --k_list 8,16 `
  --epochs 8 `
  --batch_size 4 `
  --run_sharedp `
  --run_fcnn `
  --out_dir results_medium/netlib_grow7_medium
```

Swap the `--mps_path` to `israel.mps`, `sc205.mps`, `scagr25.mps`, or
`stair.mps` to reproduce the other real-data cases.

## Formulation note

Our method is exact on the standard-form LP viewed through the
decision-sufficient subspace. The Iwata/Sakaue baselines assume a projection
model in a nonnegative-variable inequality form. For fixed-feasible flow and
real-data LPs, this code bridges the two through a feasible anchor plus a
nullspace-based nonnegative reformulation, so the comparison can be run inside
one unified pipeline.
