# Unified experiment runner

This folder contains a single parameterized runner for the paper artifact:

```text
run_experiment.py      # unified runner
env_single_gpu.sh      # single-GPU torchrun environment variables
README.md              # this file
```

The original dataset-specific scripts can remain in this folder for transparency. For reviewers, the recommended entry point is `run_experiment.py`.

## 1. Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_torch.txt
pip install -r requirements.txt
cd dicee/run_scripts
```

If using Conda instead:

```bash
conda create -n dicee-byte python=3.11 -y
conda activate dicee-byte
pip install -r requirements_torch.txt
pip install -r requirements.txt
cd dicee/run_scripts
```

## 2. Configure single-GPU torchrun environment

```bash
source env_single_gpu.sh
```

This sets:

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL
NCCL_DEBUG=INFO
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=lo
GLOO_SOCKET_IFNAME=lo
CUDA_VISIBLE_DEVICES=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
```

## 3. List available experiments

```bash
python run_experiment.py --list-experiments
```

This prints all configured dataset/variant combinations and all supported settings.

## 4. Validate one configuration without training

Use `--dry-run` first. This resolves paths and prints the exact configuration.

```bash
python run_experiment.py \
  --dataset umls \
  --variant normal \
  --setting custom-attention \
  --dry-run
```

## 5. Run an experiment

Use `torchrun` from inside `dicee/run_scripts`:

```bash
torchrun --standalone \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
  run_experiment.py \
  --dataset umls \
  --variant normal \
  --setting custom-attention
```

A full run uses the default grid:

```text
models: DistMult, ComplEx, QMult, Keci
embedding dimensions: 32, 64
learning rates: 0.1, 0.01, 0.001
batch size: 512
epochs: 500
```

That means 24 runs for non-transformer settings: `4 models x 2 dimensions x 3 learning rates x 1 batch size`.

## 6. Smoke test

For a quick reviewer sanity check, run only the first grid combination and use fewer epochs:

```bash
torchrun --standalone \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
  run_experiment.py \
  --dataset umls \
  --variant normal \
  --setting gpt2 \
  --epochs 1 \
  --max-runs 1
```

## 7. Run in the background

```bash
setsid torchrun --standalone \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
  run_experiment.py \
  --dataset umls \
  --variant normal \
  --setting custom-attention \
  > umls_custom_attention.log 2>&1 &
```

Monitor progress with:

```bash
tail -f umls_custom_attention.log
```

## 8. Dataset variants

The runner currently supports these dataset names:

```text
umls
fb15k237
wn18rr
countries-s1
countries-s2
countries-s3
nell-h25
nell-h75
nell-h100
```

Common variants are:

```text
normal
augmented-5
augmented-10
```

Not every dataset has every variant. Use `python run_experiment.py --list-experiments` to see the valid combinations.

Examples:

```bash
# Normal UMLS
torchrun --standalone --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
  run_experiment.py --dataset umls --variant normal --setting custom-attention

# UMLS augmented with 5 text augmentations
torchrun --standalone --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
  run_experiment.py --dataset umls --variant augmented-5 --setting custom-attention

# UMLS augmented with 10 text augmentations
torchrun --standalone --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
  run_experiment.py --dataset umls --variant augmented-10 --setting custom-attention
```

## 9. Experiment settings

The five main settings are encoded as command-line options:

| Setting | Byte-pair encoding | Tokenizer | Layer mode |
|---|---:|---|---|
| `no-byte` | `False` | none | `none` |
| `gpt2` | `True` | GPT2/default | `none` |
| `custom` | `True` | dataset-specific custom tokenizer | `none` |
| `gpt2-attention` | `True` | GPT2/default | `attention` |
| `custom-attention` | `True` | dataset-specific custom tokenizer | `attention` |

Examples:

```bash
# No-byte baseline
run_experiment.py --dataset umls --variant normal --setting no-byte

# GPT2 tokenizer without attention
run_experiment.py --dataset umls --variant normal --setting gpt2

# Custom tokenizer without attention
run_experiment.py --dataset umls --variant normal --setting custom

# GPT2 tokenizer with attention
run_experiment.py --dataset umls --variant normal --setting gpt2-attention

# Custom tokenizer with attention
run_experiment.py --dataset umls --variant normal --setting custom-attention
```

When running augmented datasets with a custom tokenizer, the runner automatically selects the corresponding augmented tokenizer path. For example, `--dataset umls --variant augmented-5 --setting custom-attention` uses the UMLS augmented-5 tokenizer, while `--variant augmented-10` uses the UMLS augmented-10 tokenizer.

## 10. Customizing the grid

You can override the default grid:

```bash
torchrun --standalone \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
  run_experiment.py \
  --dataset fb15k237 \
  --variant normal \
  --setting custom-attention \
  --models DistMult,Keci \
  --embedding-dims 32 \
  --learning-rates 0.01,0.001 \
  --batch-sizes 512
```

## 11. Outputs

By default, outputs are written to:

```text
results/<dataset>/<variant>/<setting>/
```

Each output directory contains:

```text
resolved_config.json       # exact resolved paths and options
grid_search_results.csv    # one row per run
raw_reports.txt            # full reports from Dicee
run*.txt                   # per-run summary files
```

You can change the root output directory:

```bash
run_experiment.py --dataset umls --variant normal --setting gpt2 --output-root my_results
```

## 12. Troubleshooting

### Dataset directory not found

Check that the `KGs` directory is present in the expected location relative to `dicee/run_scripts`.

Use:

```bash
python run_experiment.py --dataset umls --variant normal --setting gpt2 --dry-run
```

### Custom tokenizer not found

For `custom` or `custom-attention`, check that the corresponding tokenizer exists under `../Tokenizer/.../tokenizer.json` relative to `dicee/run_scripts`.

For GPT2/default tokenizer runs, use `gpt2` or `gpt2-attention`.

### NCCL/GLOO issues on a single machine

Re-source the environment file:

```bash
source env_single_gpu.sh
```

Then rerun with `--nproc_per_node=1`.

## 13. Notes for anonymous review

The runner avoids requiring reviewers to edit source-code line numbers. All reviewer-facing choices are made through command-line arguments. The original run files may remain in the repository as reference scripts, but the recommended reproducible entry point is this unified runner.
