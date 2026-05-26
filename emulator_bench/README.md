# MPEK emulator_bench

This bench retrains MPEK on EMULaToR split files without depending on MPEK's original shell wrappers.

Default data root:

```text
/home/adhil/github/EMULaToR/data/processed/baselines/MPEK
```

The bench treats `kcat` and `km` as separate single-task runs because their split trees are separate.

## Inputs

Each split file is expected to contain:

- `sequence`: enzyme amino-acid sequence
- `smiles`: substrate SMILES
- `log10_value`: regression target

The current EMULaToR MPEK splits do not contain organism, pH, or temperature, so auxiliary inputs are disabled. No dummy auxiliary values are generated.

## Embeddings

Protein embeddings:

- Encoder: ProtT5 from `Rostlab/prot_t5_xl_uniref50`
- Cache value: original 1024-dimensional mean-pooled embedding
- Cache path: `embeddings/proteins/<hash-prefix>/<hash>.npz`

Substrate embeddings:

- Encoder: upstream `Mole-BERT` submodule
- Checkpoint: `Mole-BERT/model_gin/Mole-BERT.pth`
- Cache value: 300-dimensional mean-pooled graph embedding
- Cache path: `embeddings/ligands/molebert/<hash-prefix>/<hash>.npz`

`prepare_assets.py --auto` validates the checkpoint and can recover it from the submodule history if it is missing.

## Model

The cached-input model follows the local MPEK implementation:

- ProtT5 1024-d protein vector
- learned `Linear(1024, 300)` projection
- Mole-BERT 300-d ligand vector
- concatenated no-aux input size: `600`
- original `MTLKcatKM.layers.PLE` and `MLP` blocks

Default training settings follow the paper-style configuration:

- `lr=1e-4`
- `num_experts=4`
- `expert_layers=1`
- `expert_dim=768`
- `dropout=0.2`
- `batch_size=1`
- `epochs=50`
- `tower_layers=3`
- `tower_hidden=128`

CUDA runs enable TF32 and automatic mixed precision. AMP uses bf16 on Ampere-or-newer GPUs and fp16 otherwise.

## Commands

Validate assets:

```bash
conda run -n mldb python emulator_bench/prepare_assets.py --auto
```

Cache all embeddings:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n mldb python emulator_bench/cache_embeddings.py \
  --base_dir /home/adhil/github/EMULaToR/data/processed/baselines/MPEK \
  --tasks kcat km --device cuda:0
```

Run all splits sequentially:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n mldb python emulator_bench/run_split_benchmarks.py \
  --tasks kcat km --device cuda:0 --cache_device cuda:0
```

Run all splits through the parallel launcher:

```bash
conda run -n mldb python emulator_bench/launch_parallel_retrain.py \
  --gpus 1 --runs_per_gpu 1 --tasks kcat km --default_settings
```

Run multiple concurrent split jobs per GPU:

```bash
conda run -n mldb python emulator_bench/launch_parallel_retrain.py \
  --gpus 0 1 --runs_per_gpu 2 --tasks kcat km --default_settings
```

The default seeds are `666 777 888`. Override them with `--seeds`, for example:

```bash
conda run -n mldb python emulator_bench/launch_parallel_retrain.py \
  --gpus 0 1 --runs_per_gpu 2 --tasks kcat km --seeds 111 222 333
```

Training shows tqdm bars with live batch loss and epoch RMSE by default. With many concurrent runs, terminal output can interleave; add `--disable_tqdm` for cleaner log files.

Optional Optuna:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n mldb python emulator_bench/launch_parallel_optuna.py \
  --gpus 1 --tasks kcat km --n_trials 20
```

Aggregate existing results:

```bash
conda run -n mldb python emulator_bench/aggregate_mpek_results.py --tasks kcat km
```

## Resumability

- Embedding cache files are skipped when present.
- Cache writes use temporary files followed by atomic rename.
- Training writes `checkpoint_latest.pt` every epoch.
- Restarted runs resume from `checkpoint_latest.pt`.
- Completed runs with `final_results_test.csv` are skipped unless `--overwrite` is passed.
- Optuna studies use SQLite and `load_if_exists=True`.
