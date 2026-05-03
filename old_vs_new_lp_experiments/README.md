# Old vs New LP Projection Experiments

This bundle keeps the old paper figures/data/code and the new sub-Gaussian
beyond-prior experiment figures/data/code in separate folders.

## Layout

- `old_version/figures_data/robust_allmethods`
  - Original old complete robust/all-methods figures and CSV summaries.
- `old_version/figures_data/selected_pca_sga_corridor_rank_d8`
  - Old selected PCA/SGA figure set used for comparison.
- `old_version/code`
  - Old snapshot code plus top-level rendering scripts used around the old figures.
- `new_version/figures_data/paper8_k_sweep_randomlp_cd_v1`
  - Final new 8-panel k-sweep figure and selected CSV.
- `new_version/data/raw_selected_case_runs`
  - Raw per-case artifacts for the final selected 8 LP cases:
    `packing`, `mincostflow`, `random_lp_A`, `random_lp_B`,
    `random_lp_C`, `random_lp_D`, `grow7`, `sc205`.
- `new_version/code`
  - New sub-Gaussian beyond-prior scripts and the old snapshot code they import.

## New Experiment Notes

- Same samples are used to estimate `C_hat_rho` and train our Algorithm 2.
- Baselines use the same `n_train=24` training costs.
- `rho=0.1`, `radius_scale=1.10`.
- PCA/SGA are merged as `data-driven proj`.
- The final figure shadow bands are mean plus/minus standard error.

The top-level `bundle_manifest.json` records the exact folders included.
