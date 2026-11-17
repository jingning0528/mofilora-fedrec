# Experiment results

The pre-organization code snapshot is tagged `MoFiLoRA_V1_Paper` on GitHub.
This directory organizes copies of existing results. Source files in `csv/`,
`figures/`, and `log/` are retained so existing analysis scripts continue to work.
No numerical values or figure contents have been changed.

## Layout

```text
results/
  paper/
    tables/paper_table.csv           Existing cross-dataset paper table
    efficiency/figures/              Efficiency and model-size figures
  ML1M/
    multi_seed/logs/                 Final paper runs (2024–2028)
    comparison/tables/              Dataset summary tables
    AB/LoRA/                        logs/, tables/, figures/
    AB/FixedB/                      logs/, tables/, figures/
    AB/MoFiLoRA/                    logs/, tables/, figures/
    AB/comparison/                  Cross-method tables and figures
    full_model/logs/                Recorded full-model analysis run
  Instrument/
    multi_seed/logs/                 Final paper runs
    comparison/tables/
    AB/                             Populated when Instrument A/B outputs exist
  Industrial/                       Final multi_seed/logs/ and comparison/tables/
  Software/                         Final multi_seed/logs/ and comparison/tables/
  archive/
    csv/                            Alternate tables and older analyses
    figures/                        Historical figures, including timestamp-only plots
    log/                            Historical logs and alternate multi-seed copies
  index.csv                         File titles, original paths, status, size, SHA-256
```

## Final paper result source

The user confirmed `log/multi_seed/` as the final result source. The four
standard dataset folders (ML1M, Instrument, Industrial, Software) are organized
under `results/<dataset>/multi_seed/logs/`. These are distinct from the A/B
analysis experiments. `paper/multi_seed_runs.csv` lists each run's model, seed,
configured rounds, last logged round, and whether a final-test marker exists.

The user confirmed `log/multi_seed/Industrial/` as the final Industrial paper
source. `Industrial-less`, loose Industrial logs, and ZIP exports remain in
the archive as alternate copies and are excluded from the final run index. Existing summary tables are preserved, but
have not been recomputed or validated against the final logs.

## Reading the tables

- `*_method_comparison.csv`: existing per-method dataset summary.
- `*_method_comparison_mean_std.csv`: existing mean/std presentation table.
- `*_relative_to_lora.csv`: existing comparison relative to LoRA.
- `*_all_method_results.csv`: existing combined results table.

These are candidates for the paper, not a claim that all files come from the
same run or configuration. No metric headings or units have been rewritten.
Industrial's numbered table alternatives remain under `archive/csv/`; final
Industrial summaries must be validated against `log/multi_seed/Industrial/`. Timestamp-only figures retain their original names;
their dataset cannot safely be inferred from their filenames.

The archive also includes original `ab_convergence/` copies: these deliberately
remain available even where equivalent files exist in `ML1M/AB/`.

## Provenance and updates

`index.csv` provides a human-readable title, source path, organized path,
classification status, byte size, and SHA-256 checksum for every copied file.
MacOS metadata and Python source files are excluded.

The existing training and plotting commands still write to their documented
original locations, including `csv/instrument_AB/` and `figures/instrument_AB/`.
After producing new results, preview or refresh the organized copies with:

```bash
python scripts/organize_results.py
python scripts/organize_results.py --apply
```

The refresh refuses to overwrite an organized file with different content.
Version a changed source result before importing it again. These commands do
not delete or move original files.

See [ML1M A/B analysis](ML1M/AB/README.md) for the LoRA, FixedB, and MoFiLoRA
measurements and compact PDFs. Instrument A/B outputs have not been imported yet.
