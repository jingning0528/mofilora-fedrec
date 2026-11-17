# MoFiLoRA: Momentum-Enhanced LoRA for Federated Recommendation

MoFiLoRA is a research implementation of communication-efficient federated recommendation using low-rank adaptation (LoRA) and momentum-based aggregation.

Train with `main.py`; find analysis and plotting tools in `analysis/`.
Run commands from the project root.

```text
main.py              Training entry point
config/              Dataset paths and experiment settings
zoo/                 Current model implementations, including Analyze_* tracking
framework/           Shared training and model components
dataloaders/         Dataset loading
data_pre_process/    Data preparation
analysis/            A/B analysis, plots, result tables, timing summaries
scripts/             Reusable project maintenance tools
results/             Organized paper results and historical result archive
data/                Dataset inputs
checkpoints/         Training logs/checkpoints
log/, csv/, figures/ Original inputs/outputs used by the analysis tools
archive/             Older model implementations and scripts
```

Start here:

- Train Instrument with A/B tracking: `python main.py --expid Instrument --model Analyze_LoRA --gpu 0`
- Extract A/B metrics: `python analysis/extract_ab_update_metrics.py`
- Draw A/B figures: `python analysis/plot_a_vs_b_updates.py`
- Final paper runs: `results/<dataset>/multi_seed/logs/`
- ML1M A/B results: `results/ML1M/AB/{LoRA,FixedB,MoFiLoRA}/`
- Refresh organized results: `python scripts/organize_results.py --apply`

## Datasets

The experiments use public recommendation datasets:

- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- [Amazon Review Data](https://nijianmo.github.io/amazon/index.html)

Dataset definitions and local paths are configured in [`config/dataset_config.yaml`](config/dataset_config.yaml). Update its paths before running an experiment on a new machine.

Preprocessing scripts are available in `data_pre_process/`.

## Installation

Create a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Experiment settings are defined in [`config/model_config.yaml`](config/model_config.yaml). Each top-level key is an experiment ID containing its dataset, model, training, optimizer, and evaluation settings.

Important options include:

- `model` — model implementation exported by `zoo/`
- `dataset_id` — dataset entry from `config/dataset_config.yaml`
- `latent_dim` — LoRA rank
- `train_turn` — number of federated communication rounds
- `local_epoch` — local epochs per round
- `server_optimizer` — server update method, such as heavy-ball momentum
- `beta` and `eta_s` — momentum and server step-size parameters

## Running an Experiment

Run an experiment by passing its configuration ID:

```bash
python main.py --expid <experiment_id> --gpu <device>
```

Use `-1` for CPU or a CUDA device index such as `0` for GPU execution.

Example:

```bash
python main.py --expid ML1M --gpu 0
```

Override the configured random seed or model when needed:

```bash
python main.py \
  --expid Industrial \
  --gpu 0 \
  --seed 2025 \
  --model LoRA_MomA_FixedB
```

Add `--save_csv` to write a result summary. Use `--result_file` to choose its destination.

## Project Structure

```text
config/                # Dataset and experiment configuration
data_pre_process/      # MovieLens and Amazon preprocessing
dataloaders/           # Federated dataset loaders
framework/             # Training utilities and shared modules
zoo/                   # Current model implementations
zoo_v1/, zoo_v2/       # Earlier experimental variants
main.py                 # Experiment entry point
analyze_*.py            # Convergence and ablation analysis
figure*.py              # Plot generation
```

## Acknowledgements

The training framework is based on ideas and components from [FuxiCTR](https://github.com/reczoo/FuxiCTR).

## A/B update analysis for Musical Instruments

The existing `log/analyze_1000/` A/B measurements use ML1M. The ordinary
`log/instrument/` logs do not contain these measurements; evaluation metrics
alone cannot reconstruct them. Run the analysis models with the existing
`Instrument` experiment (1000 rounds, seed 2024, rank 4):

```bash
mkdir -p log/instrument_AB
python main.py --expid Instrument --model Analyze_LoRA --gpu 0 2>&1 | tee log/instrument_AB/LoRA.txt
python main.py --expid Instrument --model Analyze_LoRA_FixedB --gpu 0 2>&1 | tee log/instrument_AB/LoRA-FixedB.txt
python main.py --expid Instrument --model Analyze_LoRA_MomA_FixedB --gpu 0 2>&1 | tee log/instrument_AB/MoFiLoRA.txt
```

First set the `Amazon_instrument_t5` paths in `config/dataset_config.yaml` for
your machine. The current paths target Google Colab/Drive. These are full
training runs, and require the prepared Instrument data and model inputs.

Extract measurements and generate figures for all three methods:

```bash
python analysis/extract_ab_update_metrics.py --logs instrument_AB/LoRA.txt instrument_AB/LoRA-FixedB.txt instrument_AB/MoFiLoRA.txt --output-name instrument_AB
```

This writes CSVs under `csv/instrument_AB/` and figures under
`figures/instrument_AB/`, including per-run A/B plots and method
comparisons. The existing ML1M outputs are preserved byte-for-byte in `csv/ML1M_AB/`
and `figures/ML1M_AB/`; the original `ab_convergence/` folders are also retained.
For the compact A-versus-B PDF figures from the LoRA run:

```bash
python analysis/plot_a_vs_b_updates.py --input-csv csv/instrument_AB/instrument_AB__LoRA/ab_convergence_metrics.csv --out-dir figures/instrument_AB/comparison
```

For each matrix X (A or B), normalized magnitude is
`||X_t - X_(t-1)||_F / (||X_(t-1)||_F + 1e-12)`; cosine similarity compares
successive update matrices, not the parameter matrices themselves. The first
cosine has no previous update. For FixedB models B has zero update magnitude
and undefined cosine (NaN), so its cosine curve is absent.

The default workflow for the two A/B figures is now:

```bash
# After training Analyze_LoRA into log/instrument_AB/LoRA.txt:
python analysis/extract_ab_update_metrics.py
python analysis/plot_a_vs_b_updates.py
```

Both scripts default to `instrument_AB`, so these commands leave the saved
ML1M results untouched. `plot_a_updates_across_methods.py` continues to use the archived
ML1M method-comparison table in `csv/ML1M_AB/`.


## Organized results

See [results/README.md](results/README.md) for dataset folders, paper tables,
the historical archive, and a checksum-based provenance index. Original result
paths remain available for existing scripts. The pre-organization paper snapshot
is tagged `MoFiLoRA_V1_Paper` on GitHub.
