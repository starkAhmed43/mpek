# MPEK EMULaToR Bench Plan

This repository now uses `emulator_bench/` as a thin retraining bench around the original MPEK code and architecture. The bench targets EMULaToR split trees under `/home/adhil/github/EMULaToR/data/processed/baselines/MPEK`.

## Baseline Facts

- MPEK predicts enzymatic kinetic parameters from enzyme sequence and substrate SMILES.
- The paper uses ProtT5 for enzyme encoding and Mole-BERT for substrate encoding.
- The local model projects ProtT5's 1024-dimensional enzyme vector to the 300-dimensional ligand space before concatenation.
- The available EMULaToR split files contain `smiles`, `sequence`, `value`, `smiles_hash`, `uniprot_date`, and `log10_value`.
- Organism, pH, and temperature columns are not present, so the bench disables auxiliary features by default.

## Bench Mapping

- `kcat` and `km` are trained as separate single-task jobs because their split trees are separate.
- ProtT5 embeddings are cached once as 1024-dimensional mean-pooled vectors.
- Mole-BERT embeddings are cached once as 300-dimensional mean-pooled graph vectors.
- Cached features are loaded by lightweight train/val/test datasets.
- Training uses a MPEK-style PLE model with no auxiliary branch: ligand 300 + projected protein 300 = 600 input features.

## Execution Sequence

1. Validate assets with `emulator_bench/prepare_assets.py`.
2. Cache protein and ligand embeddings with `emulator_bench/cache_embeddings.py`.
3. Train one split with `emulator_bench/train_single_split.py`, or all splits with `run_split_benchmarks.py` / `launch_parallel_retrain.py`.
4. Optionally tune optimization-only hyperparameters with Optuna.
5. Aggregate summaries with `emulator_bench/aggregate_mpek_results.py`.

## Testing

- Validate imports in conda env `mldb`.
- Validate Mole-BERT checkpoint loading from the `Mole-BERT` submodule.
- Discover both `kcat` and `km` split jobs.
- Smoke-cache and smoke-train tiny subsets on `CUDA_VISIBLE_DEVICES=1`.
- Verify checkpoint resume from `checkpoint_latest.pt`.
