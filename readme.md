# R code for "Stabilized Estimation in Semiparametric Logistic Regression Using Iterative Ridge-Type Estimation"

## Files

- `simulation_IWSRTE.R`: Monte Carlo study (Section 5, Appendix A7). Base R only.
- `real_data_IWSRTE.R`: Pima Indians Diabetes application (Section 6). Needs the `mlbench` package for the data.

## Run

```
Rscript simulation_IWSRTE.R
Rscript real_data_IWSRTE.R
```

Outputs are written to `tables/`, `figures/` (simulation) and `tables_real/`, `figures_real/` (real data). The simulation also saves `all_res.rds`.

## Settings used in the paper

- Simulation: 18 settings (p = 3, 6; n = 100, 250, 400; rho = 0.90, 0.99, 0.999), R = 500 replications, seed 2025.
- Grids: h in {0.08, 0.13, 0.18}; k in {0.05, 0.15, ..., 1.95, 2.5, 3, 4, 5, 7.5, 10, 15, 20, 30, 50}. k = 0 is used only for the IWSLSE.
- Real data: five-fold CV, seed 2025; h in {0.08, 0.11, ..., 0.38}; k in {0.05, 0.10, ..., 5, 7.5, 10, 15, 20, 30, 50, 100}.
- (h, k) selected by BIC, Eq. (4.10); GCV and AICc are also reported for the simulation (Table 7). Only converged IRLS fits are eligible.
- Working weights are bounded below by w_min = 1e-3.

## Outputs and the paper

| File | Paper |
|---|---|
| `tables/T1_smse.csv` | Table 1 |
| `tables/T2_bias_variance.csv` | Table 2 |
| `tables/T3_coverage.csv` | Table 3 |
| `tables_real/T4_cv.csv`, `T4_cv_folds.csv` | Table 4 |
| `tables_real/T5_coef.csv` | Table 5 |
| `tables/T6_mcr.csv` | Table 6 |
| `tables/T7_criteria.csv` | Table 7 |
| `tables/T8_mcse.csv` | Table 8 |
| `tables/T9a_selected_hk.csv`, `T9b_replications.csv` | Table 9 |
| `tables/T10_fhat_error.csv` | Table 10 |
| `tables/text_*.csv` | values quoted in Section 5.3 |
| `figures/kpath_*`, `fhat_*`, `decision_*`, `qq_*`, `fconv_*` | Figures 1-5 |
| `figures_real/fig_diagnostics.png`, `fig_fhat_real.png` | Figures 6-7 |

## Notes

- Failed replications (no converged fit on the grid) are discarded and not replaced. Requested, valid and failed counts are in `T9b_replications.csv`.
- The full simulation takes several hours on a standard PC.
