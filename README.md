# Diffusion Factor Model

<p align="center">
  <img src="assets/demo.png" alt="Diffusion Factor Model demo" width="700"/>
</p>

A sequential diffusion framework for modeling and generating high-dimensional financial return series with latent factor structure.

## Overview

This repository trains a **conditional transformer + Gaussian diffusion** model on `.npy` return data and generates synthetic samples for downstream analysis (distribution matching, factor recovery, and portfolio evaluation).

Compared with earlier versions in the commit history, the current training workflow adds:

- Sequential transformer-based diffusion training
- Sampling window controls for partial-sequence generation
- Optional prefix conditioning during sampling
- Optional checkpoint loading and sampling-only runs
- Reproducibility metadata (commit hash, dirty status, CLI/config snapshot)
- Optional architectural controls such as BOS token and ALiBi-style positional bias (via config)

## Repository structure

```text
.
├── config/                         # Hyperparameters and runtime paths
├── diffusion_factor_model/         # Core model, diffusion process, trainer
├── eval/                           # Evaluation modules and notebooks
├── simulation_experiment_data/     # Example simulation training data
├── empirical_analysis_data/        # Example empirical training data
├── train.py                        # Main entry point for training/sampling
├── model_results/                  # Created automatically (checkpoints + run metadata)
└── samples/                        # Created automatically (generated .npy batches)
```

## Installation

```bash
git clone https://github.com/xymmmm00/diffusion_factor_model.git
cd diffusion_factor_model
pip install -r requirements.txt
```

## Data format

`train.py` expects a NumPy array saved as `.npy` with shape:

- `(num_samples, sequence_length)` (recommended)
- `(sequence_length,)` (auto-expanded to one sample)
- Higher-dimensional arrays are flattened to `(num_samples, -1)`

Examples are included at:

- `simulation_experiment_data/training_data_example.npy`
- `empirical_analysis_data/training_data_example.npy`

## Training and sampling

Minimal run:

```bash
python train.py \
  --data_path simulation_experiment_data/training_data_example.npy \
  --seed 42 \
  --gpu 0
```

Run with explicit controls:

```bash
python train.py \
  --data_path empirical_analysis_data/training_data_example.npy \
  --seed 42 \
  --gpu 0 \
  --epochs 500 \
  --num_samples 1024 \
  --sample_window_start 0 \
  --sample_window_length 256 \
  --save_timesteps 20 50 100
```

Conditioned sampling (prefix known, remainder generated):

```bash
python train.py \
  --data_path empirical_analysis_data/training_data_example.npy \
  --conditioning_path empirical_analysis_data/training_data_example.npy \
  --conditioning_length 64 \
  --gpu 0
```

Sampling-only from a saved checkpoint:

```bash
python train.py \
  --data_path empirical_analysis_data/training_data_example.npy \
  --checkpoint_path model_results/<experiment_id>/model-*.pt \
  --skip_training \
  --gpu 0
```

## CLI arguments (`train.py`)

- `--data_path` (required): training data `.npy` path
- `--seed`: random seed (default from `config`)
- `--num_samples`: truncate training set to first N samples
- `--gpu`: CUDA device id (sets `CUDA_VISIBLE_DEVICES`)
- `--epochs`: override config epoch count
- `--save_timesteps`: save selected denoising timesteps during sampling
- `--sample_window_start`: start index (inclusive) for training/sampling window
- `--sample_window_length`: number of indices in the selected window
- `--conditioning_path`: optional conditioning sequence `.npy`
- `--conditioning_length`: conditioned prefix length
- `--checkpoint_path`: checkpoint to load before training/sampling
- `--skip_training`: skip optimization and run sampling only (requires checkpoint)

## Outputs and reproducibility

Each run creates an experiment directory under `model_results/dfm_<data>_ts<timestamp>_seed<seed>/` and stores:

- model checkpoints
- `commit_hash.txt`
- `run_config.json` (CLI args + config snapshot)
- `git_status.txt` and `git_diff.patch` when running on a dirty working tree

Generated samples are saved under `samples/<experiment_id>/sample_batch*.npy`.

## Evaluation

Evaluation utilities live in `eval/`, including:

- `simulation_eval.py` for simulation distribution/subspace checks
- `mean_cov.py` for mean-covariance estimation helpers
- `mv_portfolio_eval.py` for mean-variance portfolio metrics
- `ft_portfolio_eval.py` for factor-timing portfolio evaluation
- notebooks (`ARMA.ipynb`, `GP.ipynb`, `QQplot.ipynb`) for exploratory analyses

<p align="center">
  <img src="assets/distribution_example.png" alt="distribution example" width="700"/>
</p>

<p align="center">
  <img src="assets/portfolio_example.png" alt="portfolio example" width="700"/>
</p>

## Citation

```bibtex
@article{chen2025diffusion,
  title={Diffusion Factor Models: Generating High-Dimensional Returns with Factor Structure},
  author={Chen, Minshuo and Xu, Renyuan and Xu, Yumin and Zhang, Ruixun},
  journal={arXiv preprint arXiv:2504.06566},
  year={2025}
}
```
