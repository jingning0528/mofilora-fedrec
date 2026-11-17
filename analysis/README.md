# Analysis and plotting tools

Run these commands from the project root. Dataset/experiment settings remain
in `config/`; current training models remain in `zoo/`.

| Task | Command | Default output |
| --- | --- | --- |
| Extract A/B update metrics and diagnostic plots | `python analysis/extract_ab_update_metrics.py` | `csv/instrument_AB/`, `figures/instrument_AB/` |
| Plot A-versus-B magnitude and cosine | `python analysis/plot_a_vs_b_updates.py` | `figures/instrument_AB/comparison/` |
| Compare A across LoRA, FixedB, MoFiLoRA | `python analysis/plot_a_updates_across_methods.py` | `figures/ML1M_AB/comparison/` |
| Analyze full-model embedding updates | `python analysis/analyze_full_embedding_updates.py` | `csv/full_convergence/`, `figures/full_convergence/` |
| Compare training curves | `python analysis/plot_training_method_comparison.py` | `figures/` |
| Plot one training run | `python analysis/plot_single_run_metrics.py` | `figures/` |
| Extract run summaries/timing | `python analysis/extract_run_performance_metrics.py` | `csv/baseline_1000_summary.csv` |
| Build dataset comparison tables | `python analysis/build_multiseed_result_tables.py` | `csv/` |
| Plot efficiency figures | `python analysis/plot_model_size_and_training_time.py` | `figures/Efficiency/` |

The plotting and summary scripts retain their original selected input files.
Review input selections before executing them. Run CLI-based tools with
`--help` for available input/output options. Historical scripts have not been
silently retargeted to the final multi-seed dataset.

## Overlap and shared code

- `extract_ab_update_metrics.py` reads training logs and produces metric CSVs
  plus diagnostic plots. `plot_a_vs_b_updates.py` reads one of those CSVs to
  create the two compact paper PDFs. They cover the same measurements at
  different workflow stages; the second does not recalculate updates.
- `plot_a_updates_across_methods.py` compares A between methods using the
  existing wide comparison table. Its input format and comparison differ from
  the A-versus-B plot within one run.
- `plot_training_method_comparison.py` overlays multiple methods;
  `plot_single_run_metrics.py` combines HR and NDCG from one run. Their duplicated
  parser, smoothing, sampling, and drawing functions now live in
  `_training_plot_utils.py`. This private helper is not a runnable command.
- `extract_run_performance_metrics.py` creates per-run performance/timing
  summaries; `build_multiseed_result_tables.py` aggregates summary CSVs into
  mean/std and relative-to-LoRA tables. These are consecutive stages.
