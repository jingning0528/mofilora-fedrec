# ML1M A/B analysis

The 1000-round analysis is organized by method:

```text
AB/
  LoRA/
    logs/FedNCF-LoRA.txt
    tables/ab_convergence_metrics.csv
    figures/AB_cosine_similarity.pdf
    figures/AB_norm_magnitude.pdf
  FixedB/
    logs/FedNCF-FixedB.txt
    tables/ab_convergence_metrics.csv
    figures/AB_cosine_similarity.pdf
    figures/AB_norm_magnitude.pdf
  MoFiLoRA/
    logs/MofiLoRA.txt
    tables/ab_convergence_metrics.csv
    figures/AB_cosine_similarity.pdf
    figures/AB_norm_magnitude.pdf
  comparison/
    tables/
    figures/
```

Each method also includes the existing evaluation tables, summaries, and
individual diagnostic PNGs. Each metric CSV has 100 observations, sampled
every 10 rounds. Source paths and checksums are in `results/index.csv`.

The compact PDFs inside each method's figures folder were generated from that
method's own metric CSV. `comparison/` retains the earlier combined artifacts
unchanged; it is separate from these explicitly sourced method figures.

Normalized magnitude is the Frobenius norm of the update divided by the
previous parameter norm. Cosine similarity compares consecutive updates.
For FixedB and MoFiLoRA, B is fixed: its update magnitude is zero and its update
cosine is undefined (NaN), so there is no B cosine curve. The first update also
has no preceding update for computing cosine similarity.

These A/B analyses are separate from the final multi-seed performance results.
The original `csv/ML1M_AB/`, `figures/ML1M_AB/`, and `log/analyze_1000/` files
remain unchanged.

To regenerate the compact PDFs for any method (replace LoRA as needed):

```bash
python analysis/plot_a_vs_b_updates.py --input-csv results/ML1M/AB/LoRA/tables/ab_convergence_metrics.csv --out-dir results/ML1M/AB/LoRA/figures
```

Instrument has no A/B analysis outputs locally yet. Its three methods can be
organized the same way after the analysis runs are imported; regular training
logs without A/B tracking cannot supply these measurements.
